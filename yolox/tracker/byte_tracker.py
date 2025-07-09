import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
from .reid_extractor import ReidExtractor

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, camera_id=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # Multi-camera tracking attributes
        self.camera_id = camera_id
        self.curr_feat = None  # Current appearance feature
        self.smooth_feat = None  # Smoothed appearance feature for matching
        self.alpha = 0.9  # Smoothing factor for appearance features
        self.feat_history = deque(maxlen=100)  # Feature history for cross-camera association
        self.cross_camera_matches = []  # Cross-camera match history
        self.global_track_id = None  # Global track ID across cameras

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        # Update appearance features if available
        if hasattr(new_track, 'curr_feat') and new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def update_features(self, feat):
        """
        Update appearance features with temporal smoothing
        :param feat: New appearance feature vector
        """
        if feat is None:
            return

        # Normalize feature
        feat = feat / (np.linalg.norm(feat) + 1e-8)

        # Store current feature
        self.curr_feat = feat

        # Update smooth feature with EMA
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            self.smooth_feat = self.smooth_feat / (np.linalg.norm(self.smooth_feat) + 1e-8)

        # Store in history
        self.feat_history.append(feat.copy())

    def get_feature_for_matching(self, use_smooth=True):
        """
        Get appearance feature for matching
        :param use_smooth: Whether to use smoothed feature
        :return: Feature vector for matching
        """
        if use_smooth and self.smooth_feat is not None:
            return self.smooth_feat
        elif self.curr_feat is not None:
            return self.curr_feat
        else:
            return None

    def compute_feature_distance(self, other_track, metric='cosine'):
        """
        Compute appearance feature distance with another track
        :param other_track: Another STrack
        :param metric: Distance metric
        :return: Distance value
        """
        feat1 = self.get_feature_for_matching()
        feat2 = other_track.get_feature_for_matching()

        if feat1 is None or feat2 is None:
            return 1.0  # Maximum distance

        if metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            similarity = np.dot(feat1, feat2)
            return 1.0 - similarity
        elif metric == 'euclidean':
            return np.linalg.norm(feat1 - feat2)
        else:
            return 1.0

    def is_same_camera(self, other_track):
        """
        Check if two tracks are from the same camera
        :param other_track: Another STrack
        :return: True if same camera
        """
        return self.camera_id == other_track.camera_id

    def can_associate_cross_camera(self, other_track, max_time_gap=300):
        """
        Check if this track can be associated with another track from different camera
        :param other_track: Another STrack
        :param max_time_gap: Maximum time gap for cross-camera association
        :return: True if can associate
        """
        # Same camera tracks cannot be cross-camera associated
        if self.is_same_camera(other_track):
            return False

        # Check time gap
        time_gap = abs(self.frame_id - other_track.frame_id)
        if time_gap > max_time_gap:
            return False

        # Check if both tracks have features
        if self.get_feature_for_matching() is None or other_track.get_feature_for_matching() is None:
            return False

        return True

    def set_global_track_id(self, global_id):
        """
        Set global track ID for cross-camera tracking
        :param global_id: Global track ID
        """
        self.global_track_id = global_id

    def get_display_id(self):
        """
        Get display ID for visualization (prefers global ID)
        :return: Display ID
        """
        if self.global_track_id is not None:
            return self.global_track_id
        else:
            return self.track_id

    def __repr__(self):
        cam_info = f"C{self.camera_id}" if self.camera_id is not None else "C?"
        global_info = f"G{self.global_track_id}" if self.global_track_id is not None else ""
        return 'OT_{}_{}_{}({}-{})'.format(self.track_id, cam_info, global_info, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30, camera_id=None, reid_config=None, reid_model=None):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.camera_id = camera_id  # Camera ID for multi-camera tracking
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # Initialize ReID extractor for appearance features
        self.reid_extractor = ReidExtractor(
            config_path=reid_config,
            model_path=reid_model,
            device=getattr(args, 'device', 'cuda')
        )

        # Enable appearance features if ReID is available
        self.use_reid = getattr(args, 'use_reid', True) and self.reid_extractor.is_available()

        if self.use_reid:
            print(f"Camera {camera_id}: ReID feature extraction enabled")
        else:
            print(f"Camera {camera_id}: ReID feature extraction disabled")

    def update(self, output_results, img_info, img_size, img=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, camera_id=self.camera_id) for
                          (tlbr, s) in zip(dets, scores_keep)]

            # Extract ReID features if available
            if self.use_reid and img is not None:
                self._extract_reid_features(img, detections, dets)
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, camera_id=self.camera_id) for
                          (tlbr, s) in zip(dets_second, scores_second)]

            # Extract ReID features for second detections if available
            if self.use_reid and img is not None:
                self._extract_reid_features(img, detections_second, dets_second)
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

    def _extract_reid_features(self, img, detections, bboxes):
        """
        Extract ReID features for detections
        :param img: Input image
        :param detections: List of STrack detections
        :param bboxes: Bounding boxes in x1y1x2y2 format
        """
        if len(detections) == 0:
            return

        try:
            # Extract features using ReID extractor
            features = self.reid_extractor.extract_features(img, bboxes)

            # Update detections with features
            for detection, feature in zip(detections, features):
                detection.curr_feat = feature
                detection.update_features(feature)

        except Exception as e:
            print(f"Error extracting ReID features: {e}")
            # Continue without features
            pass


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
