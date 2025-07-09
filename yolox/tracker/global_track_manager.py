"""
Global Track Manager for Multi-Camera Tracking
Manages global track IDs and cross-camera associations
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple
import time


class GlobalTrackManager:
    """
    Manages global track IDs and cross-camera associations for multi-camera tracking
    """

    def __init__(self):
        """Initialize Global Track Manager"""
        self.next_global_id = 1

        # Mapping from local track to global track ID
        self.local_to_global = {}  # {(camera_id, track_id): global_id}

        # Mapping from global track ID to local tracks
        self.global_to_local = defaultdict(set)  # {global_id: {(camera_id, track_id)}}

        # Track association history
        self.association_history = deque(maxlen=1000)

        # Track lifecycle management
        self.active_global_tracks = set()
        self.inactive_global_tracks = set()

        # Statistics
        self.stats = {
            'total_global_tracks': 0,
            'active_global_tracks': 0,
            'cross_camera_associations': 0
        }

    def get_global_id(self, track) -> int:
        """
        Get or create global ID for a track

        Args:
            track: STrack object

        Returns:
            Global track ID
        """
        key = (track.camera_id, track.track_id)

        # Check if track already has global ID
        if key in self.local_to_global:
            global_id = self.local_to_global[key]
            self.active_global_tracks.add(global_id)
            return global_id

        # Create new global ID
        global_id = self.next_global_id
        self.next_global_id += 1

        # Register mapping
        self.local_to_global[key] = global_id
        self.global_to_local[global_id].add(key)
        self.active_global_tracks.add(global_id)

        # Update statistics
        self.stats['total_global_tracks'] += 1
        self.stats['active_global_tracks'] = len(self.active_global_tracks)

        return global_id

    def associate_tracks(self, track_a, track_b) -> bool:
        """
        Associate two tracks from different cameras

        Args:
            track_a: First track
            track_b: Second track

        Returns:
            True if association successful
        """
        # Ensure tracks are from different cameras
        if track_a.camera_id == track_b.camera_id:
            return False

        key_a = (track_a.camera_id, track_a.track_id)
        key_b = (track_b.camera_id, track_b.track_id)

        # Get or create global IDs
        global_id_a = self.get_global_id(track_a)
        global_id_b = self.get_global_id(track_b)

        # If they already have the same global ID, no action needed
        if global_id_a == global_id_b:
            return True

        # Merge global tracks
        target_global_id = min(global_id_a, global_id_b)
        source_global_id = max(global_id_a, global_id_b)

        # Move all tracks from source to target
        source_tracks = self.global_to_local[source_global_id].copy()
        for track_key in source_tracks:
            self.local_to_global[track_key] = target_global_id
            self.global_to_local[target_global_id].add(track_key)

        # Remove source global track
        del self.global_to_local[source_global_id]
        self.active_global_tracks.discard(source_global_id)
        self.inactive_global_tracks.add(source_global_id)

        # Record association
        self.association_history.append({
            'timestamp': time.time(),
            'track_a': key_a,
            'track_b': key_b,
            'global_id': target_global_id,
            'merged_from': source_global_id
        })

        # Update statistics
        self.stats['cross_camera_associations'] += 1
        self.stats['active_global_tracks'] = len(self.active_global_tracks)

        return True

    def remove_track(self, track):
        """
        Remove a track from global tracking

        Args:
            track: STrack object to remove
        """
        key = (track.camera_id, track.track_id)

        if key in self.local_to_global:
            global_id = self.local_to_global[key]

            # Remove from mappings
            del self.local_to_global[key]
            self.global_to_local[global_id].discard(key)

            # If no more tracks for this global ID, deactivate it
            if len(self.global_to_local[global_id]) == 0:
                del self.global_to_local[global_id]
                self.active_global_tracks.discard(global_id)
                self.inactive_global_tracks.add(global_id)

        # Update statistics
        self.stats['active_global_tracks'] = len(self.active_global_tracks)

    def get_global_tracks(self) -> Dict[int, List[Tuple]]:
        """
        Get all tracks grouped by global track ID

        Returns:
            Dictionary mapping global track ID to list of (camera_id, track_id) tuples
        """
        result = {}
        for global_id, tracks in self.global_to_local.items():
            result[global_id] = list(tracks)
        return result

    def get_tracks_for_global_id(self, global_id: int) -> List[Tuple]:
        """
        Get all tracks for a specific global ID

        Args:
            global_id: Global track ID

        Returns:
            List of (camera_id, track_id) tuples
        """
        return list(self.global_to_local.get(global_id, set()))

    def get_active_global_tracks_count(self) -> int:
        """
        Get count of active global tracks

        Returns:
            Number of active global tracks
        """
        return len(self.active_global_tracks)

    def is_cross_camera_track(self, global_id: int) -> bool:
        """
        Check if a global track spans multiple cameras

        Args:
            global_id: Global track ID

        Returns:
            True if track spans multiple cameras
        """
        tracks = self.global_to_local.get(global_id, set())
        cameras = set(camera_id for camera_id, _ in tracks)
        return len(cameras) > 1

    def get_cross_camera_tracks(self) -> List[int]:
        """
        Get list of global track IDs that span multiple cameras

        Returns:
            List of global track IDs
        """
        cross_camera_tracks = []
        for global_id in self.active_global_tracks:
            if self.is_cross_camera_track(global_id):
                cross_camera_tracks.append(global_id)
        return cross_camera_tracks

    def get_camera_tracks(self, camera_id) -> List[Tuple[int, int]]:
        """
        Get all tracks for a specific camera

        Args:
            camera_id: Camera ID

        Returns:
            List of (track_id, global_id) tuples
        """
        camera_tracks = []
        for (cam_id, track_id), global_id in self.local_to_global.items():
            if cam_id == camera_id:
                camera_tracks.append((track_id, global_id))
        return camera_tracks

    def get_association_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get association history

        Args:
            limit: Maximum number of records to return

        Returns:
            List of association records
        """
        history = list(self.association_history)
        if limit is not None:
            history = history[-limit:]
        return history

    def get_statistics(self) -> Dict:
        """
        Get tracking statistics

        Returns:
            Dictionary of statistics
        """
        # Update active count
        self.stats['active_global_tracks'] = len(self.active_global_tracks)
        return self.stats.copy()

    def cleanup_inactive_tracks(self, max_inactive_time: float = 300.0):
        """
        Clean up inactive tracks based on time

        Args:
            max_inactive_time: Maximum time to keep inactive tracks (seconds)
        """
        current_time = time.time()

        # Remove old associations from history
        while (self.association_history and
               current_time - self.association_history[0]['timestamp'] > max_inactive_time):
            self.association_history.popleft()

        # Note: Track cleanup should be handled by the caller based on track states
        # We don't automatically remove tracks here as they might become active again

    def reset(self):
        """
        Reset all tracking state
        """
        self.next_global_id = 1
        self.local_to_global.clear()
        self.global_to_local.clear()
        self.association_history.clear()
        self.active_global_tracks.clear()
        self.inactive_global_tracks.clear()

        self.stats = {
            'total_global_tracks': 0,
            'active_global_tracks': 0,
            'cross_camera_associations': 0
        }

    def export_associations(self) -> Dict:
        """
        Export all association data for analysis

        Returns:
            Dictionary containing all association data
        """
        return {
            'local_to_global': dict(self.local_to_global),
            'global_to_local': {k: list(v) for k, v in self.global_to_local.items()},
            'association_history': list(self.association_history),
            'active_global_tracks': list(self.active_global_tracks),
            'inactive_global_tracks': list(self.inactive_global_tracks),
            'statistics': self.get_statistics()
        }

    def import_associations(self, data: Dict):
        """
        Import association data

        Args:
            data: Association data dictionary
        """
        self.local_to_global = data.get('local_to_global', {})
        self.global_to_local = defaultdict(set)
        for k, v in data.get('global_to_local', {}).items():
            self.global_to_local[int(k)] = set(v)

        self.association_history = deque(data.get('association_history', []), maxlen=1000)
        self.active_global_tracks = set(data.get('active_global_tracks', []))
        self.inactive_global_tracks = set(data.get('inactive_global_tracks', []))

        # Update next global ID
        if self.active_global_tracks:
            self.next_global_id = max(self.active_global_tracks) + 1
        else:
            self.next_global_id = 1

        # Update statistics
        self.stats.update(data.get('statistics', {}))