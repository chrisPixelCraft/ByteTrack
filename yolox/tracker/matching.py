import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from yolox.tracker import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine', use_smooth=True):
    """
    Enhanced appearance-based distance for multi-camera tracking
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric: Distance metric ('cosine', 'euclidean')
    :param use_smooth: Whether to use smoothed features
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    
    # Extract detection features
    det_features = []
    for track in detections:
        if hasattr(track, 'curr_feat') and track.curr_feat is not None:
            det_features.append(track.curr_feat)
        else:
            # Use dummy feature if no appearance feature available
            det_features.append(np.zeros(512, dtype=float))
    
    # Extract track features
    track_features = []
    for track in tracks:
        if use_smooth:
            feat = track.get_feature_for_matching(use_smooth=True)
        else:
            feat = track.get_feature_for_matching(use_smooth=False)
        
        if feat is not None:
            track_features.append(feat)
        else:
            # Use dummy feature if no appearance feature available
            track_features.append(np.zeros(512, dtype=float))
    
    # Convert to numpy arrays
    det_features = np.asarray(det_features, dtype=float)
    track_features = np.asarray(track_features, dtype=float)
    
    # Handle empty feature cases
    if det_features.size == 0 or track_features.size == 0:
        return cost_matrix
    
    # Compute distance matrix
    if metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        # Features should already be normalized
        similarity_matrix = np.dot(track_features, det_features.T)
        cost_matrix = 1.0 - similarity_matrix
    elif metric == 'euclidean':
        cost_matrix = cdist(track_features, det_features, metric='euclidean')
    else:
        # Fallback to scipy's cdist
        cost_matrix = cdist(track_features, det_features, metric=metric)
    
    # Ensure non-negative costs
    cost_matrix = np.maximum(0.0, cost_matrix)
    
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    """Fuse ReID and IoU similarities for robust matching"""
    if cost_matrix.size == 0:
        return cost_matrix
    
    # Convert appearance distance to similarity
    reid_sim = 1 - cost_matrix
    
    # Compute IoU distance and convert to similarity
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    
    # Fuse ReID and IoU similarities
    # Give more weight to ReID for cross-camera tracking
    reid_weight = 0.7
    iou_weight = 0.3
    fuse_sim = reid_weight * reid_sim + iou_weight * iou_sim
    
    # Optional: incorporate detection scores
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    
    # Boost similarity with detection confidence
    # fuse_sim = fuse_sim * (0.5 + 0.5 * det_scores)
    
    # Convert back to cost
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    """Fuse cost matrix with detection scores"""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def cross_camera_distance(tracks_a, tracks_b, metric='cosine', max_time_gap=300):
    """
    Compute cross-camera appearance distance between tracks from different cameras
    :param tracks_a: list[STrack] from camera A
    :param tracks_b: list[STrack] from camera B
    :param metric: Distance metric
    :param max_time_gap: Maximum time gap for valid cross-camera matching
    :return: cost_matrix np.ndarray
    """
    cost_matrix = np.full((len(tracks_a), len(tracks_b)), np.inf, dtype=float)
    
    if len(tracks_a) == 0 or len(tracks_b) == 0:
        return cost_matrix
    
    # Extract features from both camera tracks
    features_a = []
    features_b = []
    
    for track in tracks_a:
        feat = track.get_feature_for_matching(use_smooth=True)
        if feat is not None:
            features_a.append(feat)
        else:
            features_a.append(np.zeros(512, dtype=float))
    
    for track in tracks_b:
        feat = track.get_feature_for_matching(use_smooth=True)
        if feat is not None:
            features_b.append(feat)
        else:
            features_b.append(np.zeros(512, dtype=float))
    
    features_a = np.asarray(features_a, dtype=float)
    features_b = np.asarray(features_b, dtype=float)
    
    # Compute appearance distance
    if metric == 'cosine':
        similarity_matrix = np.dot(features_a, features_b.T)
        appearance_cost = 1.0 - similarity_matrix
    else:
        appearance_cost = cdist(features_a, features_b, metric=metric)
    
    # Apply temporal constraints
    for i, track_a in enumerate(tracks_a):
        for j, track_b in enumerate(tracks_b):
            # Skip if same camera
            if track_a.is_same_camera(track_b):
                cost_matrix[i, j] = np.inf
                continue
            
            # Apply time gap constraint
            time_gap = abs(track_a.frame_id - track_b.frame_id)
            if time_gap > max_time_gap:
                cost_matrix[i, j] = np.inf
                continue
            
            # Use appearance cost
            cost_matrix[i, j] = appearance_cost[i, j]
    
    return cost_matrix


def filter_same_camera_matches(matches, tracks_a, tracks_b):
    """
    Filter out matches between tracks from the same camera
    :param matches: List of match tuples
    :param tracks_a: List of tracks from first set
    :param tracks_b: List of tracks from second set
    :return: Filtered matches
    """
    filtered_matches = []
    
    for i, j in matches:
        if i < len(tracks_a) and j < len(tracks_b):
            if not tracks_a[i].is_same_camera(tracks_b[j]):
                filtered_matches.append((i, j))
    
    return filtered_matches