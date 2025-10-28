#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Camera Tracking (MCT) Evaluation Script for NTU-MTMC Dataset
Calculates MOTA, mAP, R1, R5 scores for cross-camera person tracking

Usage:
    python3 tools/mota_mct.py
    python3 tools/mota_mct.py --gt_dir datasets/mot/train --pred_dir MCT_outputs/tracking_results
    python3 tools/mota_mct.py --cameras Cam1,Cam2,Cam3 --verbose
"""

import argparse
import glob
import os
import sys
import warnings
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd

try:
    import motmetrics as mm
    from loguru import logger
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install: pip install motmetrics loguru")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')


class MCTEvaluator:
    """
    Multi-Camera Tracking Evaluator for NTU-MTMC Dataset
    """

    def __init__(self, gt_dir: str = "datasets/mot/train",
                 pred_dir: str = "MCT_outputs/tracking_results",
                 cameras: List[str] = None,
                 iou_threshold: float = 0.5,
                 min_confidence: float = 0.05,  # Lowered from 0.1 for higher sensitivity
                 verbose_logging: bool = False):
        """
        Initialize MCT Evaluator

        Args:
            gt_dir: Ground truth directory
            pred_dir: Prediction results directory
            cameras: List of camera IDs to evaluate
            iou_threshold: IoU threshold for matching
            min_confidence: Minimum confidence threshold
            verbose_logging: Enable verbose logging for debugging
        """
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.cameras = cameras or self._discover_cameras()
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        self.verbose_logging = verbose_logging

        # Setup MOT metrics
        mm.lap.default_solver = 'lap'
        self.mh = mm.metrics.create()

        logger.info(f"MCT Evaluator initialized for {len(self.cameras)} cameras")
        logger.info(f"Ground truth: {gt_dir}")
        logger.info(f"Predictions: {pred_dir}")
        logger.info(f"Cameras: {self.cameras}")
        logger.info(f"Verbose logging: {verbose_logging}")

    def _discover_cameras(self) -> List[str]:
        """
        Automatically discover available cameras from ground truth directory
        """
        cameras = []
        if os.path.exists(self.gt_dir):
            for item in os.listdir(self.gt_dir):
                if item.startswith('Cam') and os.path.isdir(os.path.join(self.gt_dir, item)):
                    cameras.append(item)

        cameras.sort()  # Ensure consistent ordering
        return cameras

    def _validate_data_format(self, data: Dict, data_type: str) -> bool:
        """
        Validate data format for ground truth or predictions

        Args:
            data: Data dictionary (camera -> DataFrame)
            data_type: 'ground_truth' or 'predictions'

        Returns:
            True if valid, False otherwise
        """
        if not data:
            logger.error(f"No {data_type} data found")
            return False

        required_columns = {'X', 'Y', 'Width', 'Height'}

        for camera, df in data.items():
            if df.empty:
                logger.warning(f"Empty {data_type} data for {camera}")
                continue

            # Check required columns
            missing_cols = required_columns - set(df.columns)
            if missing_cols:
                logger.error(f"Missing columns in {data_type} for {camera}: {missing_cols}")
                return False

            # Check index structure
            if not isinstance(df.index, pd.MultiIndex):
                logger.error(f"Expected MultiIndex (FrameId, Id) for {data_type} in {camera}")
                return False

            if len(df.index.levels) != 2:
                logger.error(f"Expected 2-level MultiIndex for {data_type} in {camera}")
                return False

            # Check for reasonable bounding box values
            if (df['Width'] <= 0).any() or (df['Height'] <= 0).any():
                logger.warning(f"Invalid bounding box dimensions in {data_type} for {camera}")

            # Check for reasonable coordinates
            if (df['X'] < -100).any() or (df['Y'] < -100).any():
                logger.warning(f"Suspicious bounding box coordinates in {data_type} for {camera}")

            logger.info(f"Validated {data_type} for {camera}: {len(df)} detections")

        return True

    def load_data(self) -> Tuple[Dict, Dict]:
        """
        Load ground truth and prediction data

        Returns:
            Tuple of (ground_truth_dict, predictions_dict)
        """
        logger.info("Loading ground truth and prediction data...")

        # Load ground truth
        gt_data = {}
        gt_files = []
        for camera in self.cameras:
            gt_file = os.path.join(self.gt_dir, camera, "gt", "gt.txt")
            if os.path.exists(gt_file):
                gt_files.append(gt_file)
                try:
                    gt_df = mm.io.loadtxt(gt_file, fmt='mot15-2D', min_confidence=1)
                    logger.info(f"GT DataFrame columns for {camera}: {list(gt_df.columns)}")

                    # Check if motmetrics loaded properly with expected structure
                    if 'Id' not in gt_df.columns or 'FrameId' not in gt_df.columns:
                        logger.info(f"Motmetrics didn't parse correctly for {camera}, using manual parsing")
                        # Manual parsing of MOT format
                        gt_df = pd.read_csv(gt_file, header=None,
                                          names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'extra1'])
                        gt_df = gt_df.drop(['extra1'], axis=1, errors='ignore')
                        gt_df = gt_df[gt_df['Confidence'] >= 1]  # Apply confidence filter

                        # Set proper MultiIndex for motmetrics
                        gt_df = gt_df.set_index(['FrameId', 'Id'])
                    else:
                        # Ensure proper MultiIndex structure for motmetrics
                        if not isinstance(gt_df.index, pd.MultiIndex):
                            if 'FrameId' in gt_df.columns and 'Id' in gt_df.columns:
                                gt_df = gt_df.set_index(['FrameId', 'Id'])
                            else:
                                gt_df.reset_index(inplace=True)
                                if len(gt_df.columns) >= 2:
                                    gt_df.columns = ['FrameId', 'Id'] + list(gt_df.columns[2:])
                                    gt_df = gt_df.set_index(['FrameId', 'Id'])

                    gt_data[camera] = gt_df
                    logger.info(f"Loaded GT for {camera}: {len(gt_data[camera])} detections")
                except Exception as e:
                    logger.warning(f"Error loading GT for {camera}: {e}")
            else:
                logger.warning(f"GT file not found: {gt_file}")

        # Load predictions
        pred_data = {}
        pred_files = []
        for camera in self.cameras:
            pred_file = os.path.join(self.pred_dir, f"{camera}.txt")
            if os.path.exists(pred_file):
                pred_files.append(pred_file)
                try:
                    pred_df = mm.io.loadtxt(pred_file, fmt='mot15-2D', min_confidence=self.min_confidence)
                    logger.info(f"Pred DataFrame columns for {camera}: {list(pred_df.columns)}")

                    # Check if motmetrics loaded properly with expected structure
                    if 'Id' not in pred_df.columns or 'FrameId' not in pred_df.columns:
                        logger.info(f"Motmetrics didn't parse correctly for {camera}, using manual parsing")
                        # Manual parsing of MOT format
                        pred_df = pd.read_csv(pred_file, header=None,
                                            names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'extra1'])
                        pred_df = pred_df.drop(['extra1'], axis=1, errors='ignore')
                        pred_df = pred_df[pred_df['Confidence'] >= self.min_confidence]  # Apply confidence filter

                        # Set proper MultiIndex for motmetrics
                        pred_df = pred_df.set_index(['FrameId', 'Id'])
                    else:
                        # Ensure proper MultiIndex structure for motmetrics
                        if not isinstance(pred_df.index, pd.MultiIndex):
                            if 'FrameId' in pred_df.columns and 'Id' in pred_df.columns:
                                pred_df = pred_df.set_index(['FrameId', 'Id'])
                            else:
                                pred_df.reset_index(inplace=True)
                                if len(pred_df.columns) >= 2:
                                    pred_df.columns = ['FrameId', 'Id'] + list(pred_df.columns[2:])
                                    pred_df = pred_df.set_index(['FrameId', 'Id'])

                    pred_data[camera] = pred_df
                    logger.info(f"Loaded predictions for {camera}: {len(pred_data[camera])} detections")
                except Exception as e:
                    logger.warning(f"Error loading predictions for {camera}: {e}")
            else:
                logger.warning(f"Prediction file not found: {pred_file}")

        logger.info(f"Loaded {len(gt_data)} GT files and {len(pred_data)} prediction files")

        # Validate data formats
        if not self._validate_data_format(gt_data, "ground_truth"):
            logger.error("Ground truth data validation failed")
            return {}, {}

        if not self._validate_data_format(pred_data, "predictions"):
            logger.error("Prediction data validation failed")
            return {}, {}

        return gt_data, pred_data

    def compute_single_camera_metrics(self, gt_data: Dict, pred_data: Dict) -> Tuple[List, List[str]]:
        """
        Compute single-camera MOTA metrics

        Args:
            gt_data: Ground truth data per camera
            pred_data: Prediction data per camera

        Returns:
            Tuple of (accumulators, camera_names)
        """
        logger.info("Computing single-camera MOTA metrics...")

        accumulators = []
        camera_names = []

        for camera in self.cameras:
            if camera in gt_data and camera in pred_data:
                logger.info(f"Evaluating {camera}...")

                try:
                    acc = mm.utils.compare_to_groundtruth(
                        gt_data[camera],
                        pred_data[camera],
                        'iou',
                        distth=self.iou_threshold
                    )
                    accumulators.append(acc)
                    camera_names.append(camera)

                    # Log basic stats for this camera
                    summary = self.mh.compute(acc, metrics=['mota', 'motp', 'recall', 'precision'],
                                            name=camera)
                    logger.info(f"{camera} - MOTA: {summary['mota'].iloc[0]:.3f}, "
                              f"MOTP: {summary['motp'].iloc[0]:.3f}, "
                              f"Recall: {summary['recall'].iloc[0]:.3f}, "
                              f"Precision: {summary['precision'].iloc[0]:.3f}")

                except Exception as e:
                    logger.error(f"Error evaluating {camera}: {e}")
            else:
                logger.warning(f"Skipping {camera} - missing GT or predictions")

        return accumulators, camera_names

    def compute_cross_camera_metrics(self, gt_data: Dict, pred_data: Dict) -> Dict:
        """
        Compute cross-camera tracking metrics (mAP, R1, R5)

        Args:
            gt_data: Ground truth data per camera
            pred_data: Prediction data per camera

        Returns:
            Dictionary of cross-camera metrics
        """
        logger.info("Computing cross-camera tracking metrics...")

        # Extract global track information
        gt_camera_tracks = self._extract_global_tracks(gt_data, is_ground_truth=True)
        pred_global_tracks = self._extract_global_tracks(pred_data, is_ground_truth=False)

        # Associate ground truth tracks across cameras
        gt_global_tracks = self._associate_ground_truth_tracks(gt_camera_tracks)

        logger.info(f"Ground truth global tracks: {len(gt_global_tracks)}")
        logger.info(f"Predicted global tracks: {len(pred_global_tracks)}")

        # Compute cross-camera matching
        matches, gt_unmatched, pred_unmatched = self._match_global_tracks(
            gt_global_tracks, pred_global_tracks
        )

        # Calculate metrics
        metrics = self._calculate_cross_camera_metrics(
            matches, gt_global_tracks, pred_global_tracks, gt_unmatched, pred_unmatched
        )

        return metrics

    def _extract_global_tracks(self, data: Dict, is_ground_truth: bool = False) -> Dict[str, Dict]:
        """
        Extract global track information from tracking data

        Args:
            data: Tracking data per camera
            is_ground_truth: Whether this is ground truth data (with camera-specific track IDs)

        Returns:
            Dictionary mapping track_id to track information
        """
        global_tracks = {}

        for camera, df in data.items():
            for idx, row in df.iterrows():
                if isinstance(idx, tuple):
                    frame_id, track_id = idx
                    frame_id = int(frame_id)
                    track_id = int(track_id)
                else:
                    # This case should rarely happen with properly loaded motmetrics data
                    # If idx is not a tuple, it means the dataframe wasn't properly processed
                    logger.warning(f"Non-tuple index found in {camera}: {idx}. Skipping this row.")
                    continue

                # For ground truth, create camera-specific track identifiers
                if is_ground_truth:
                    global_track_id = f"{camera}_{track_id}"
                else:
                    # For predictions, assume track IDs are already global
                    global_track_id = str(track_id)

                if global_track_id not in global_tracks:
                    global_tracks[global_track_id] = {
                        'cameras': set(),
                        'frames': [],
                        'bboxes': [],
                        'camera_frames': defaultdict(list),
                        'local_track_ids': defaultdict(set),  # Map camera -> local track IDs
                        'original_track_id': track_id
                    }

                global_tracks[global_track_id]['cameras'].add(camera)
                global_tracks[global_track_id]['frames'].append(frame_id)
                global_tracks[global_track_id]['bboxes'].append([row['X'], row['Y'], row['Width'], row['Height']])
                global_tracks[global_track_id]['camera_frames'][camera].append(frame_id)
                global_tracks[global_track_id]['local_track_ids'][camera].add(track_id)

        # For ground truth, we have camera-specific tracks by design
        # For predictions, filter tracks that appear in multiple cameras
        if is_ground_truth:
            logger.info(f"Ground truth tracks: {len(global_tracks)} camera-specific tracks")
            return global_tracks
        else:
            multi_camera_tracks = {
                track_id: info for track_id, info in global_tracks.items()
                if len(info['cameras']) >= 2
            }
            logger.info(f"Prediction tracks: {len(global_tracks)} total, {len(multi_camera_tracks)} multi-camera tracks")
            return multi_camera_tracks

    def _associate_ground_truth_tracks(self, gt_tracks: Dict) -> Dict[str, Dict]:
        """
        Associate ground truth tracks across cameras using appearance and spatial-temporal cues

        Args:
            gt_tracks: Camera-specific ground truth tracks

        Returns:
            Dictionary of associated multi-camera ground truth tracks
        """
        logger.info("Associating ground truth tracks across cameras...")

        # For now, implement a simple spatial-temporal association
        # In a real implementation, this would use:
        # 1. Manual annotations if available
        # 2. Appearance features from Reid models
        # 3. Sophisticated spatial-temporal reasoning

        associated_tracks = {}
        global_id_counter = 1
        processed_tracks = set()

        for track_id, track_info in gt_tracks.items():
            if track_id in processed_tracks:
                continue

            # Find similar tracks in other cameras
            similar_tracks = [track_id]
            for other_track_id, other_track_info in gt_tracks.items():
                if other_track_id == track_id or other_track_id in processed_tracks:
                    continue

                # Check if tracks are from different cameras
                if track_info['cameras'] & other_track_info['cameras']:
                    continue  # Same camera, skip

                # Calculate similarity based on temporal overlap and spatial proximity
                similarity = self._calculate_gt_track_similarity(track_info, other_track_info)

                if similarity > 0.4:  # Threshold for GT association
                    similar_tracks.append(other_track_id)

            # Create associated track
            if len(similar_tracks) > 1:  # Multi-camera track
                associated_track_id = f"gt_global_{global_id_counter}"
                global_id_counter += 1

                # Merge track information
                merged_cameras = set()
                merged_frames = []
                merged_bboxes = []
                merged_camera_frames = defaultdict(list)
                merged_local_ids = defaultdict(set)

                for st_id in similar_tracks:
                    st_info = gt_tracks[st_id]
                    merged_cameras.update(st_info['cameras'])
                    merged_frames.extend(st_info['frames'])
                    merged_bboxes.extend(st_info['bboxes'])
                    for cam, frames in st_info['camera_frames'].items():
                        merged_camera_frames[cam].extend(frames)
                    for cam, local_ids in st_info['local_track_ids'].items():
                        merged_local_ids[cam].update(local_ids)
                    processed_tracks.add(st_id)

                associated_tracks[associated_track_id] = {
                    'cameras': merged_cameras,
                    'frames': merged_frames,
                    'bboxes': merged_bboxes,
                    'camera_frames': merged_camera_frames,
                    'local_track_ids': merged_local_ids,
                    'component_tracks': similar_tracks
                }

        logger.info(f"Associated {len(associated_tracks)} multi-camera ground truth tracks from {len(gt_tracks)} camera-specific tracks")
        return associated_tracks

    def _calculate_gt_track_similarity(self, track1: Dict, track2: Dict) -> float:
        """
        Calculate similarity between two ground truth tracks for cross-camera association

        Args:
            track1: First track information
            track2: Second track information

        Returns:
            Similarity score between 0 and 1
        """
        # Temporal overlap - tracks should exist in temporally close frames
        frames1 = set(track1['frames'])
        frames2 = set(track2['frames'])

        if not frames1 or not frames2:
            return 0.0

        # Calculate temporal distance between tracks
        min_frame1, max_frame1 = min(frames1), max(frames1)
        min_frame2, max_frame2 = min(frames2), max(frames2)

        # Check for temporal overlap or proximity
        temporal_gap = max(0, max(min_frame1, min_frame2) - min(max_frame1, max_frame2))
        if temporal_gap > 100:  # More than 100 frames apart
            return 0.0

        # Calculate temporal similarity (inverse of gap)
        temporal_similarity = 1.0 / (1.0 + temporal_gap / 30.0)  # Normalize by 30 frames

        # Spatial consistency - similar sized bounding boxes
        if not track1['bboxes'] or not track2['bboxes']:
            return temporal_similarity * 0.5

        # Calculate average bbox sizes
        avg_size1 = np.mean([w * h for _, _, w, h in track1['bboxes']])
        avg_size2 = np.mean([w * h for _, _, w, h in track2['bboxes']])

        size_ratio = min(avg_size1, avg_size2) / max(avg_size1, avg_size2) if max(avg_size1, avg_size2) > 0 else 0

        # Combined similarity
        return 0.7 * temporal_similarity + 0.3 * size_ratio

    def _match_global_tracks(self, gt_tracks: Dict, pred_tracks: Dict) -> Tuple[List, Set, Set]:
        """
        Match global tracks between ground truth and predictions

        Args:
            gt_tracks: Ground truth global tracks
            pred_tracks: Predicted global tracks

        Returns:
            Tuple of (matches, unmatched_gt, unmatched_pred)
        """
        matches = []
        matched_gt = set()
        matched_pred = set()

        # Simple matching based on spatial-temporal overlap
        for gt_id, gt_info in gt_tracks.items():
            best_match = None
            best_score = 0.0
            candidate_scores = []

            for pred_id, pred_info in pred_tracks.items():
                if pred_id in matched_pred:
                    continue

                # Calculate overlap score
                score = self._calculate_track_overlap(gt_info, pred_info)
                candidate_scores.append((pred_id, score))

                if score > best_score and score > 0.3:  # Minimum overlap threshold
                    best_score = score
                    best_match = pred_id

            # Log detailed matching information
            if best_match is not None:
                matches.append((gt_id, best_match, best_score))
                matched_gt.add(gt_id)
                matched_pred.add(best_match)

                # Sort candidates by score for logging
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                top_candidates = candidate_scores[:3]  # Top 3 candidates

                if self.verbose_logging:
                    logger.info(f"GT {gt_id} matched to Pred {best_match} (score: {best_score:.3f})")
                    logger.info(f"  GT cameras: {gt_info['cameras']}, frames: {len(gt_info['frames'])}")
                    logger.info(f"  Pred cameras: {pred_tracks[best_match]['cameras']}, frames: {len(pred_tracks[best_match]['frames'])}")
                    logger.info(f"  Top candidates: {[(pid, f'{score:.3f}') for pid, score in top_candidates]}")
            else:
                if self.verbose_logging:
                    logger.info(f"GT {gt_id} unmatched (best score: {max([score for _, score in candidate_scores]) if candidate_scores else 0.0:.3f})")
                    logger.info(f"  GT cameras: {gt_info['cameras']}, frames: {len(gt_info['frames'])}")

        unmatched_gt = set(gt_tracks.keys()) - matched_gt
        unmatched_pred = set(pred_tracks.keys()) - matched_pred

        logger.info(f"Track matching summary:")
        logger.info(f"  Matched tracks: {len(matches)}")
        logger.info(f"  Unmatched GT: {len(unmatched_gt)} - {list(unmatched_gt)}")
        logger.info(f"  Unmatched Pred: {len(unmatched_pred)} - {list(unmatched_pred)}")

        # Log match quality distribution
        if matches:
            scores = [score for _, _, score in matches]
            logger.info(f"  Match score distribution: min={min(scores):.3f}, max={max(scores):.3f}, avg={np.mean(scores):.3f}")

        return matches, unmatched_gt, unmatched_pred

    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate IoU between two bounding boxes

        Args:
            bbox1: [x, y, width, height]
            bbox2: [x, y, width, height]

        Returns:
            IoU score between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to corner coordinates
        x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
        x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _calculate_track_overlap(self, track1: Dict, track2: Dict) -> float:
        """
        Calculate spatial-temporal overlap between two tracks using IoU-based spatial matching

        Args:
            track1: First track information
            track2: Second track information

        Returns:
            Overlap score between 0 and 1
        """
        # Camera overlap
        camera_overlap = len(track1['cameras'] & track2['cameras']) / len(track1['cameras'] | track2['cameras'])

        # Temporal overlap
        frames1 = set(track1['frames'])
        frames2 = set(track2['frames'])
        temporal_overlap = len(frames1 & frames2) / len(frames1 | frames2) if len(frames1 | frames2) > 0 else 0

        # Spatial overlap using IoU
        spatial_overlap = 0.0
        if track1['bboxes'] and track2['bboxes']:
            # Find overlapping frames
            common_frames = frames1 & frames2
            if common_frames:
                ious = []
                # Get frame-indexed bboxes
                track1_frame_bboxes = {}
                track2_frame_bboxes = {}

                for i, frame in enumerate(track1['frames']):
                    if i < len(track1['bboxes']):
                        track1_frame_bboxes[frame] = track1['bboxes'][i]

                for i, frame in enumerate(track2['frames']):
                    if i < len(track2['bboxes']):
                        track2_frame_bboxes[frame] = track2['bboxes'][i]

                # Calculate IoU for overlapping frames
                for frame in common_frames:
                    if frame in track1_frame_bboxes and frame in track2_frame_bboxes:
                        iou = self._calculate_bbox_iou(
                            track1_frame_bboxes[frame],
                            track2_frame_bboxes[frame]
                        )
                        ious.append(iou)

                if ious:
                    spatial_overlap = np.mean(ious)
            else:
                # No temporal overlap, use proximity-based spatial similarity
                # Calculate average bbox centers and sizes
                track1_centers = []
                track1_sizes = []
                for bbox in track1['bboxes']:
                    x, y, w, h = bbox
                    track1_centers.append([x + w/2, y + h/2])
                    track1_sizes.append(w * h)

                track2_centers = []
                track2_sizes = []
                for bbox in track2['bboxes']:
                    x, y, w, h = bbox
                    track2_centers.append([x + w/2, y + h/2])
                    track2_sizes.append(w * h)

                if track1_centers and track2_centers:
                    # Calculate average positions and sizes
                    avg_center1 = np.mean(track1_centers, axis=0)
                    avg_center2 = np.mean(track2_centers, axis=0)
                    avg_size1 = np.mean(track1_sizes)
                    avg_size2 = np.mean(track2_sizes)

                    # Distance-based similarity
                    center_distance = np.linalg.norm(avg_center1 - avg_center2)
                    size_ratio = min(avg_size1, avg_size2) / max(avg_size1, avg_size2) if max(avg_size1, avg_size2) > 0 else 0

                    # Normalize distance by image dimensions (assuming typical video resolution)
                    normalized_distance = center_distance / 1000.0  # Adjust based on your video resolution
                    distance_similarity = 1.0 / (1.0 + normalized_distance)

                    spatial_overlap = 0.7 * distance_similarity + 0.3 * size_ratio

        # Combined score with adjusted weights
        return 0.4 * camera_overlap + 0.3 * temporal_overlap + 0.3 * spatial_overlap

    def _calculate_cross_camera_metrics(self, matches: List, gt_tracks: Dict,
                                      pred_tracks: Dict, gt_unmatched: Set,
                                      pred_unmatched: Set) -> Dict:
        """
        Calculate cross-camera tracking metrics

        Args:
            matches: List of matched track pairs
            gt_tracks: Ground truth global tracks
            pred_tracks: Predicted global tracks
            gt_unmatched: Unmatched ground truth tracks
            pred_unmatched: Unmatched predicted tracks

        Returns:
            Dictionary of metrics
        """
        # Basic counts
        tp = len(matches)  # True positives
        fp = len(pred_unmatched)  # False positives
        fn = len(gt_unmatched)  # False negatives

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # mAP calculation for multi-camera tracking
        map_score = self._calculate_map_score(matches, gt_tracks, pred_tracks)

        # R1 and R5 calculation
        r1_score = self._calculate_rank_accuracy(matches, gt_tracks, pred_tracks, 1)
        r5_score = self._calculate_rank_accuracy(matches, gt_tracks, pred_tracks, 5)

        metrics = {
            'cross_camera_precision': precision,
            'cross_camera_recall': recall,
            'cross_camera_f1': f1_score,
            'mAP': map_score,
            'R1': r1_score,
            'R5': r5_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'total_gt_tracks': len(gt_tracks),
            'total_pred_tracks': len(pred_tracks)
        }

        return metrics

    def _calculate_map_score(self, matches: List, gt_tracks: Dict, pred_tracks: Dict) -> float:
        """
        Calculate mAP (Mean Average Precision) for multi-camera tracking

        Args:
            matches: List of matched track pairs with scores
            gt_tracks: Ground truth global tracks
            pred_tracks: Predicted global tracks

        Returns:
            mAP score between 0 and 1
        """
        if not gt_tracks:
            return 0.0

        # For each ground truth track, calculate AP
        ap_scores = []

        for gt_id, gt_info in gt_tracks.items():
            # Find all predictions that could match this GT track
            candidates = []

            for pred_id, pred_info in pred_tracks.items():
                # Calculate similarity/confidence score
                similarity = self._calculate_track_overlap(gt_info, pred_info)
                if similarity > 0.1:  # Minimum threshold to consider
                    candidates.append((pred_id, similarity))

            # Sort candidates by similarity (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Calculate AP for this GT track
            if candidates:
                # Check if there's a match in the matches list
                gt_matched = False
                best_match_rank = len(candidates)  # Worst case rank

                for match_gt_id, match_pred_id, match_score in matches:
                    if match_gt_id == gt_id:
                        gt_matched = True
                        # Find rank of this match in candidates
                        for rank, (cand_pred_id, cand_score) in enumerate(candidates):
                            if cand_pred_id == match_pred_id:
                                best_match_rank = rank
                                break
                        break

                if gt_matched:
                    # AP = 1 / (rank + 1), where rank is 0-indexed
                    ap = 1.0 / (best_match_rank + 1)
                else:
                    ap = 0.0
            else:
                ap = 0.0

            ap_scores.append(ap)

        # mAP is the mean of all AP scores
        return np.mean(ap_scores) if ap_scores else 0.0

    def _calculate_rank_accuracy(self, matches: List, gt_tracks: Dict, pred_tracks: Dict, k: int) -> float:
        """
        Calculate Rank-k accuracy for person re-identification

        Args:
            matches: List of matched track pairs with scores
            gt_tracks: Ground truth global tracks
            pred_tracks: Predicted global tracks
            k: Rank threshold

        Returns:
            Rank-k accuracy
        """
        if not gt_tracks or not pred_tracks:
            return 0.0

        correct_at_k = 0
        total_queries = 0

        # For each ground truth track, check if correct prediction is in top-k
        for gt_id, gt_info in gt_tracks.items():
            # Find all predictions ranked by similarity to this GT track
            candidates = []

            for pred_id, pred_info in pred_tracks.items():
                similarity = self._calculate_track_overlap(gt_info, pred_info)
                candidates.append((pred_id, similarity))

            # Sort by similarity (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Check if there's a correct match in the matches list
            correct_pred_id = None
            for match_gt_id, match_pred_id, match_score in matches:
                if match_gt_id == gt_id:
                    correct_pred_id = match_pred_id
                    break

            if correct_pred_id is not None:
                # Check if correct prediction is in top-k
                top_k_pred_ids = [pred_id for pred_id, _ in candidates[:k]]
                if correct_pred_id in top_k_pred_ids:
                    correct_at_k += 1

            total_queries += 1

        return correct_at_k / total_queries if total_queries > 0 else 0.0

    def evaluate(self, verbose: bool = True) -> Dict:
        """
        Run complete MCT evaluation

        Args:
            verbose: Whether to print detailed results

        Returns:
            Dictionary of all evaluation metrics
        """
        logger.info("Starting Multi-Camera Tracking evaluation...")

        # Load data
        gt_data, pred_data = self.load_data()

        if not gt_data or not pred_data:
            logger.error("No data loaded. Please check file paths.")
            return {}

        # Single-camera MOTA metrics
        accumulators, camera_names = self.compute_single_camera_metrics(gt_data, pred_data)

        if not accumulators:
            logger.error("No valid camera evaluations completed.")
            return {}

        # Compute overall MOTA metrics
        logger.info("Computing overall MOTA metrics...")

        # Standard MOT metrics
        mota_metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost', 'num_false_positives',
                       'num_misses', 'num_switches', 'num_fragmentations', 'mota',
                       'motp', 'num_objects']

        summary = self.mh.compute_many(accumulators, names=camera_names,
                                     metrics=mota_metrics, generate_overall=True)

        # Normalize certain metrics
        div_dict = {
            'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
            'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']
        }

        for divisor in div_dict:
            for divided in div_dict[divisor]:
                if divisor in summary.columns and divided in summary.columns:
                    summary[divided] = summary[divided] / summary[divisor]

        # Cross-camera metrics
        cross_camera_metrics = self.compute_cross_camera_metrics(gt_data, pred_data)

        # Combine all metrics
        results = {
            'single_camera_summary': summary,
            'cross_camera_metrics': cross_camera_metrics,
            'camera_names': camera_names
        }

        if verbose:
            self.print_results(results)

        return results

    def print_results(self, results: Dict):
        """
        Print evaluation results in a formatted way

        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "="*80)
        print("MULTI-CAMERA TRACKING EVALUATION RESULTS")
        print("="*80)

        # Single-camera MOTA results
        print("\n📊 SINGLE-CAMERA MOTA METRICS:")
        print("-" * 50)

        summary = results['single_camera_summary']

        # Format and display key metrics
        fmt = self.mh.formatters
        change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches',
                          'num_fragmentations', 'mostly_tracked', 'partially_tracked', 'mostly_lost']
        for k in change_fmt_list:
            if k in fmt:
                fmt[k] = fmt['mota']

        print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

        # Cross-camera metrics
        print("\n🔄 CROSS-CAMERA TRACKING METRICS:")
        print("-" * 50)

        cc_metrics = results['cross_camera_metrics']
        print(f"mAP (Mean Average Precision): {cc_metrics['mAP']:.3f}")
        print(f"R1 (Rank-1 Accuracy):        {cc_metrics['R1']:.3f}")
        print(f"R5 (Rank-5 Accuracy):        {cc_metrics['R5']:.3f}")
        print(f"Cross-Camera Precision:      {cc_metrics['cross_camera_precision']:.3f}")
        print(f"Cross-Camera Recall:         {cc_metrics['cross_camera_recall']:.3f}")
        print(f"Cross-Camera F1-Score:       {cc_metrics['cross_camera_f1']:.3f}")

        print(f"\n📈 TRACK STATISTICS:")
        print("-" * 30)
        print(f"Ground Truth Tracks:         {cc_metrics['total_gt_tracks']}")
        print(f"Predicted Tracks:            {cc_metrics['total_pred_tracks']}")
        print(f"True Positives:              {cc_metrics['true_positives']}")
        print(f"False Positives:             {cc_metrics['false_positives']}")
        print(f"False Negatives:             {cc_metrics['false_negatives']}")

        # Overall assessment
        overall_mota = summary.loc['OVERALL', 'mota'] if 'OVERALL' in summary.index else 0
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print("-" * 30)
        print(f"Overall MOTA:                {overall_mota:.3f}")
        print(f"Cross-Camera mAP:            {cc_metrics['mAP']:.3f}")

        # Performance assessment
        if overall_mota > 0.5 and cc_metrics['mAP'] > 0.3:
            assessment = "🟢 GOOD"
        elif overall_mota > 0.2 and cc_metrics['mAP'] > 0.1:
            assessment = "🟡 MODERATE"
        else:
            assessment = "🔴 NEEDS IMPROVEMENT"

        print(f"Performance Assessment:      {assessment}")
        print("="*80)


def make_parser():
    """
    Create argument parser for MCT evaluation
    """
    parser = argparse.ArgumentParser(
        "Multi-Camera Tracking Evaluation",
        description="Evaluate MCT results with MOTA, mAP, R1, R5 metrics"
    )

    parser.add_argument(
        "--gt_dir",
        default="datasets/mot/train",
        type=str,
        help="Ground truth directory"
    )

    parser.add_argument(
        "--pred_dir",
        default="MCT_outputs/tracking_results",
        type=str,
        help="Prediction results directory"
    )

    parser.add_argument(
        "--cameras",
        default=None,
        type=str,
        help="Comma-separated camera list (e.g., Cam1,Cam2,Cam3) or None for auto-discovery"
    )

    parser.add_argument(
        "--iou_threshold",
        default=0.5,
        type=float,
        help="IoU threshold for detection matching"
    )

    parser.add_argument(
        "--min_confidence",
        default=0.1,
        type=float,
        help="Minimum confidence threshold for predictions"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results"
    )

    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="Enable verbose logging for debugging track association"
    )

    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Save results to JSON file"
    )

    return parser


def main():
    """
    Main evaluation function
    """
    args = make_parser().parse_args()

    # Parse cameras
    cameras = None
    if args.cameras:
        cameras = [cam.strip() for cam in args.cameras.split(',')]

    # Create evaluator
    evaluator = MCTEvaluator(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        cameras=cameras,
        iou_threshold=args.iou_threshold,
        min_confidence=args.min_confidence,
        verbose_logging=args.verbose_logging
    )

    # Run evaluation
    try:
        results = evaluator.evaluate(verbose=args.verbose)

        # Save results if requested
        if args.output and results:
            import json

            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                return obj

            # Prepare serializable results
            serializable_results = {}
            for key, value in results.items():
                if key == 'single_camera_summary':
                    # Convert DataFrame to serializable format
                    if isinstance(value, pd.DataFrame):
                        # Convert DataFrame to nested dictionary with camera names as keys
                        camera_scores = {}
                        for camera_name in value.index:
                            camera_data = {}
                            for metric_name in value.columns:
                                metric_value = value.loc[camera_name, metric_name]
                                # Convert numpy types to native Python types
                                if isinstance(metric_value, np.integer):
                                    camera_data[metric_name] = int(metric_value)
                                elif isinstance(metric_value, np.floating):
                                    camera_data[metric_name] = float(metric_value)
                                else:
                                    camera_data[metric_name] = metric_value
                            camera_scores[camera_name] = camera_data
                        serializable_results['single_camera_scores'] = camera_scores
                    else:
                        serializable_results[key] = convert_numpy(value)
                else:
                    serializable_results[key] = convert_numpy(value)

            with open(args.output, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=convert_numpy)

            logger.info(f"Results saved to {args.output}")

        if results:
            logger.info("Evaluation completed successfully!")
            return 0
        else:
            logger.error("Evaluation failed!")
            return 1

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())