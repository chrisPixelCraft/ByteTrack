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
                 min_confidence: float = 0.1):
        """
        Initialize MCT Evaluator

        Args:
            gt_dir: Ground truth directory
            pred_dir: Prediction results directory
            cameras: List of camera IDs to evaluate
            iou_threshold: IoU threshold for matching
            min_confidence: Minimum confidence threshold
        """
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.cameras = cameras or self._discover_cameras()
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence

        # Setup MOT metrics
        mm.lap.default_solver = 'lap'
        self.mh = mm.metrics.create()

        logger.info(f"MCT Evaluator initialized for {len(self.cameras)} cameras")
        logger.info(f"Ground truth: {gt_dir}")
        logger.info(f"Predictions: {pred_dir}")
        logger.info(f"Cameras: {self.cameras}")

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

                    # Rename columns to match expected names
                    column_mapping = {
                        'frame': 'FrameId',
                        'id': 'Id',
                        'x': 'X',
                        'y': 'Y',
                        'w': 'Width',
                        'h': 'Height'
                    }
                    gt_df = gt_df.rename(columns=column_mapping)
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

                    # Rename columns to match expected names
                    column_mapping = {
                        'frame': 'FrameId',
                        'id': 'Id',
                        'x': 'X',
                        'y': 'Y',
                        'w': 'Width',
                        'h': 'Height'
                    }
                    pred_df = pred_df.rename(columns=column_mapping)
                    pred_data[camera] = pred_df
                    logger.info(f"Loaded predictions for {camera}: {len(pred_data[camera])} detections")
                except Exception as e:
                    logger.warning(f"Error loading predictions for {camera}: {e}")
            else:
                logger.warning(f"Prediction file not found: {pred_file}")

        logger.info(f"Loaded {len(gt_data)} GT files and {len(pred_data)} prediction files")
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
        gt_global_tracks = self._extract_global_tracks(gt_data)
        pred_global_tracks = self._extract_global_tracks(pred_data)

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

    def _extract_global_tracks(self, data: Dict) -> Dict[int, Dict]:
        """
        Extract global track information from tracking data

        Args:
            data: Tracking data per camera

        Returns:
            Dictionary mapping track_id to track information
        """
        global_tracks = {}

        for camera, df in data.items():
            for _, row in df.iterrows():
                track_id = int(row['Id'])
                frame_id = int(row['FrameId'])

                if track_id not in global_tracks:
                    global_tracks[track_id] = {
                        'cameras': set(),
                        'frames': [],
                        'bboxes': [],
                        'camera_frames': defaultdict(list)
                    }

                global_tracks[track_id]['cameras'].add(camera)
                global_tracks[track_id]['frames'].append(frame_id)
                global_tracks[track_id]['bboxes'].append([row['X'], row['Y'], row['Width'], row['Height']])
                global_tracks[track_id]['camera_frames'][camera].append(frame_id)

        # Filter tracks that appear in multiple cameras (as per NTU-MTMC requirement)
        multi_camera_tracks = {
            track_id: info for track_id, info in global_tracks.items()
            if len(info['cameras']) >= 2
        }

        logger.info(f"Total tracks: {len(global_tracks)}, Multi-camera tracks: {len(multi_camera_tracks)}")
        return multi_camera_tracks

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

            for pred_id, pred_info in pred_tracks.items():
                if pred_id in matched_pred:
                    continue

                # Calculate overlap score
                score = self._calculate_track_overlap(gt_info, pred_info)

                if score > best_score and score > 0.3:  # Minimum overlap threshold
                    best_score = score
                    best_match = pred_id

            if best_match is not None:
                matches.append((gt_id, best_match, best_score))
                matched_gt.add(gt_id)
                matched_pred.add(best_match)

        unmatched_gt = set(gt_tracks.keys()) - matched_gt
        unmatched_pred = set(pred_tracks.keys()) - matched_pred

        logger.info(f"Matched tracks: {len(matches)}, "
                   f"Unmatched GT: {len(unmatched_gt)}, "
                   f"Unmatched Pred: {len(unmatched_pred)}")

        return matches, unmatched_gt, unmatched_pred

    def _calculate_track_overlap(self, track1: Dict, track2: Dict) -> float:
        """
        Calculate spatial-temporal overlap between two tracks

        Args:
            track1: First track information
            track2: Second track information

        Returns:
            Overlap score between 0 and 1
        """
        # Camera overlap
        camera_overlap = len(track1['cameras'] & track2['cameras']) / len(track1['cameras'] | track2['cameras'])

        # Temporal overlap (simplified)
        frames1 = set(track1['frames'])
        frames2 = set(track2['frames'])
        temporal_overlap = len(frames1 & frames2) / len(frames1 | frames2) if len(frames1 | frames2) > 0 else 0

        # Combined score
        return 0.6 * camera_overlap + 0.4 * temporal_overlap

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

        # mAP calculation (simplified)
        if matches:
            ap_scores = [score for _, _, score in matches]
            map_score = np.mean(ap_scores)
        else:
            map_score = 0.0

        # R1 and R5 calculation
        r1_score = self._calculate_rank_accuracy(matches, 1)
        r5_score = self._calculate_rank_accuracy(matches, 5)

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

    def _calculate_rank_accuracy(self, matches: List, k: int) -> float:
        """
        Calculate Rank-k accuracy

        Args:
            matches: List of matched track pairs with scores
            k: Rank threshold

        Returns:
            Rank-k accuracy
        """
        if not matches:
            return 0.0

        # Sort matches by score (descending)
        sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)

        # Calculate rank-k accuracy (simplified - assumes perfect ranking)
        rank_k_correct = len([m for m in sorted_matches[:k] if m[2] > 0.5])
        total_queries = min(len(sorted_matches), k)

        return rank_k_correct / total_queries if total_queries > 0 else 0.0

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
        min_confidence=args.min_confidence
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
                if key != 'single_camera_summary':  # Skip DataFrame for now
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