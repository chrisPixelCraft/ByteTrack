"""
Multi-Camera Tracker for ByteTrack
Coordinates tracking across multiple cameras with cross-camera person association
"""

import numpy as np
import cv2
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Union
import threading
import time

from .byte_tracker import BYTETracker
from .matching import linear_assignment, iou_distance
from .global_track_manager import GlobalTrackManager


class MultiCameraTracker:
    """
    Multi-Camera Tracker that coordinates ByteTrack instances across multiple cameras
    and performs cross-camera association using simple heuristics.
    """

    def __init__(self,
                 args,
                 camera_ids: List[Union[int, str]],
                 reid_config: Optional[str] = None,
                 reid_model: Optional[str] = None,
                 cross_camera_thresh: float = 0.4,
                 cross_camera_interval: int = 30,
                 max_time_gap: int = 300):
        """
        Initialize Multi-Camera Tracker

        Args:
            args: Tracking arguments
            camera_ids: List of camera IDs
            reid_config: Path to Fast-Reid config (not used in simplified version)
            reid_model: Path to Fast-Reid model (not used in simplified version)
            cross_camera_thresh: Threshold for cross-camera association
            cross_camera_interval: Interval (frames) for cross-camera association
            max_time_gap: Maximum time gap for cross-camera association
        """
        self.args = args
        self.camera_ids = camera_ids
        self.cross_camera_thresh = cross_camera_thresh
        self.cross_camera_interval = cross_camera_interval
        self.max_time_gap = max_time_gap

        # Initialize single-camera trackers
        self.trackers = {}
        for camera_id in camera_ids:
            self.trackers[camera_id] = BYTETracker(args, frame_rate=30)

        # Global track manager
        self.global_track_manager = GlobalTrackManager()

        # Cross-camera association state
        self.last_cross_camera_frame = 0
        self.cross_camera_history = deque(maxlen=100)

        # Track buffers for cross-camera association
        self.track_buffers = {camera_id: deque(maxlen=50) for camera_id in camera_ids}

        # Frame counters
        self.frame_counters = {camera_id: 0 for camera_id in camera_ids}

        # Statistics
        self.stats = {
            'total_tracks': 0,
            'cross_camera_associations': 0,
            'active_global_tracks': 0
        }

        print(f"Multi-camera tracker initialized successfully")

    def update(self,
               detections_dict: Dict[Union[int, str], np.ndarray],
               img_info_dict: Dict[Union[int, str], Tuple],
               img_size_dict: Dict[Union[int, str], Tuple],
               img_dict: Optional[Dict[Union[int, str], np.ndarray]] = None) -> Dict[Union[int, str], List]:
        """
        Update tracking for all cameras

        Args:
            detections_dict: Dictionary of detections per camera
            img_info_dict: Dictionary of image info per camera
            img_size_dict: Dictionary of image sizes per camera
            img_dict: Dictionary of images per camera (optional)

        Returns:
            Dictionary of tracked objects per camera
        """
        # Update single-camera tracking
        camera_results = {}
        for camera_id in self.camera_ids:
            if camera_id in detections_dict and detections_dict[camera_id] is not None:
                # Increment frame counter
                self.frame_counters[camera_id] += 1

                tracks = self.trackers[camera_id].update(
                    output_results=detections_dict[camera_id],
                    img_info=img_info_dict[camera_id],
                    img_size=img_size_dict[camera_id]
                )

                camera_results[camera_id] = tracks

                # Add camera ID to tracks for cross-camera association
                for track in tracks:
                    track.camera_id = camera_id

                # Store tracks in buffer for cross-camera association
                self.track_buffers[camera_id].append({
                    'frame_id': self.frame_counters[camera_id],
                    'tracks': [track for track in tracks if hasattr(track, 'is_activated') and track.is_activated]
                })
            else:
                camera_results[camera_id] = []

        # Perform cross-camera association
        if self._should_perform_cross_camera_association():
            self._perform_cross_camera_association()

        # Update global track IDs
        self._update_global_track_ids(camera_results)

        # Update statistics
        self._update_statistics(camera_results)

        return camera_results

    def _should_perform_cross_camera_association(self) -> bool:
        """
        Check if cross-camera association should be performed
        """
        # Check if enough frames have passed
        if len(self.frame_counters) == 0:
            return False

        current_frame = max(self.frame_counters.values())

        if current_frame - self.last_cross_camera_frame >= self.cross_camera_interval:
            self.last_cross_camera_frame = current_frame
            return True

        return False

    def _perform_cross_camera_association(self):
        """
        Perform cross-camera association using simple heuristics
        """
        # Get recent tracks from all cameras
        all_tracks = []
        track_to_camera = {}

        for camera_id, buffer in self.track_buffers.items():
            if len(buffer) > 0:
                recent_tracks = buffer[-1]['tracks']
                for track in recent_tracks:
                    all_tracks.append(track)
                    track_to_camera[id(track)] = camera_id

        if len(all_tracks) < 2:
            return

        # Group tracks by camera
        camera_tracks = defaultdict(list)
        for track in all_tracks:
            camera_id = getattr(track, 'camera_id', track_to_camera.get(id(track)))
            if camera_id:
                camera_tracks[camera_id].append(track)

        # Perform pairwise cross-camera association
        camera_list = list(camera_tracks.keys())

        for i in range(len(camera_list)):
            for j in range(i + 1, len(camera_list)):
                camera_a = camera_list[i]
                camera_b = camera_list[j]

                tracks_a = camera_tracks[camera_a]
                tracks_b = camera_tracks[camera_b]

                if len(tracks_a) > 0 and len(tracks_b) > 0:
                    self._associate_camera_pair(tracks_a, tracks_b, camera_a, camera_b)

    def _associate_camera_pair(self, tracks_a: List, tracks_b: List, camera_a: str, camera_b: str):
        """
        Associate tracks between two cameras using simple size and position heuristics

        Args:
            tracks_a: Tracks from camera A
            tracks_b: Tracks from camera B
            camera_a: Camera A ID
            camera_b: Camera B ID
        """
        # Simple association based on track properties
        for track_a in tracks_a:
            best_match = None
            best_score = float('inf')

            for track_b in tracks_b:
                # Skip if already associated
                if hasattr(track_a, 'global_track_id') and hasattr(track_b, 'global_track_id'):
                    if track_a.global_track_id == track_b.global_track_id:
                        continue

                # Calculate similarity score
                score = self._calculate_track_similarity(track_a, track_b)

                if score < best_score and score < self.cross_camera_thresh:
                    best_score = score
                    best_match = track_b

            # Associate if good match found
            if best_match is not None:
                try:
                    self.global_track_manager.associate_tracks(track_a, best_match)
                    self.stats['cross_camera_associations'] += 1

                    # Store association history
                    self.cross_camera_history.append({
                        'frame_id': max(self.frame_counters[camera_a], self.frame_counters[camera_b]),
                        'track_a': track_a.track_id,
                        'track_b': best_match.track_id,
                        'camera_a': camera_a,
                        'camera_b': camera_b,
                        'cost': best_score
                    })
                except Exception as e:
                    print(f"Warning: Could not associate tracks: {e}")

    def _calculate_track_similarity(self, track_a, track_b) -> float:
        """
        Calculate similarity score between two tracks using simple heuristics
        """
        try:
            # Size similarity
            if hasattr(track_a, 'tlwh') and hasattr(track_b, 'tlwh'):
                size_a = track_a.tlwh[2] * track_a.tlwh[3]  # width * height
                size_b = track_b.tlwh[2] * track_b.tlwh[3]

                if size_a > 0 and size_b > 0:
                    size_ratio = min(size_a, size_b) / max(size_a, size_b)
                    size_score = 1.0 - size_ratio  # Lower is better
                else:
                    size_score = 1.0
            else:
                size_score = 1.0

            # Aspect ratio similarity
            if hasattr(track_a, 'tlwh') and hasattr(track_b, 'tlwh'):
                if track_a.tlwh[3] > 0 and track_b.tlwh[3] > 0:
                    aspect_a = track_a.tlwh[2] / track_a.tlwh[3]
                    aspect_b = track_b.tlwh[2] / track_b.tlwh[3]
                    aspect_ratio = min(aspect_a, aspect_b) / max(aspect_a, aspect_b)
                    aspect_score = 1.0 - aspect_ratio
                else:
                    aspect_score = 1.0
            else:
                aspect_score = 1.0

            # Combine scores
            final_score = 0.6 * size_score + 0.4 * aspect_score

            return final_score

        except Exception as e:
            print(f"Warning: Error calculating track similarity: {e}")
            return 1.0  # Return high cost on error

    def _update_global_track_ids(self, camera_results: Dict):
        """
        Update global track IDs for all tracks

        Args:
            camera_results: Dictionary of tracks per camera
        """
        for camera_id, tracks in camera_results.items():
            for track in tracks:
                # Set camera ID if not present
                if not hasattr(track, 'camera_id'):
                    track.camera_id = camera_id

                try:
                    global_id = self.global_track_manager.get_global_id(track)
                    # Set global ID on track
                    track.global_track_id = global_id
                except Exception as e:
                    # If global track manager fails, use local ID
                    track.global_track_id = track.track_id

    def _update_statistics(self, camera_results: Dict):
        """
        Update tracking statistics

        Args:
            camera_results: Dictionary of tracks per camera
        """
        total_tracks = sum(len(tracks) for tracks in camera_results.values())
        self.stats['total_tracks'] = total_tracks
        self.stats['active_global_tracks'] = self.global_track_manager.get_active_global_tracks_count()

    def get_statistics(self) -> Dict:
        """
        Get tracking statistics

        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()

    def get_cross_camera_associations(self) -> List[Dict]:
        """
        Get recent cross-camera associations

        Returns:
            List of association records
        """
        return list(self.cross_camera_history)

    def reset_camera(self, camera_id: Union[int, str]):
        """
        Reset tracking for a specific camera

        Args:
            camera_id: Camera ID to reset
        """
        if camera_id in self.trackers:
            self.trackers[camera_id] = BYTETracker(self.args, frame_rate=30)
            self.track_buffers[camera_id].clear()
            self.frame_counters[camera_id] = 0

    def reset_all(self):
        """
        Reset all tracking state
        """
        for camera_id in self.camera_ids:
            self.reset_camera(camera_id)

        self.global_track_manager.reset()
        self.cross_camera_history.clear()
        self.last_cross_camera_frame = 0
        self.stats = {
            'total_tracks': 0,
            'cross_camera_associations': 0,
            'active_global_tracks': 0
        }

    def get_global_tracks(self) -> Dict[int, List]:
        """
        Get all tracks grouped by global track ID

        Returns:
            Dictionary mapping global track ID to list of tracks
        """
        return self.global_track_manager.get_global_tracks()

    def _get_color_for_global_id(self, global_id: int) -> Tuple[int, int, int]:
        """
        Get consistent color for global track ID

        Args:
            global_id: Global track ID

        Returns:
            BGR color tuple
        """
        # Simple color mapping based on global ID
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]

        return colors[global_id % len(colors)]