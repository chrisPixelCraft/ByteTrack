from loguru import logger

import argparse
import os
import random
import warnings
import glob
import json
import numpy as np
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Set


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


def eval_market1501_style(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric for cross-camera tracking
    Key: for each query identity, its gallery samples from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        matches = (g_pids[order] == q_pid).astype(np.int32)
        raw_cmc = matches[keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if num_valid_q == 0:
        logger.warning('Error: all query identities do not appear in gallery')
        return np.zeros(max_rank), [], []

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP


def extract_track_features(gt_data, pred_data):
    """
    Extract track features for cross-camera evaluation
    Creates a distance matrix between query and gallery tracks
    """
    # Extract all unique track IDs and their camera information
    query_tracks = []
    gallery_tracks = []

    # Debug: Check column names and data structure
    if gt_data:
        sample_gt = next(iter(gt_data.values()))
        if not sample_gt.empty:
            logger.info(f"Ground truth columns: {list(sample_gt.columns)}")
            logger.info(f"Ground truth index names: {sample_gt.index.names}")
            logger.info(f"Ground truth sample data:\n{sample_gt.head()}")
            logger.info(f"Ground truth data types: {sample_gt.dtypes}")

    if pred_data:
        sample_pred = next(iter(pred_data.values()))
        if not sample_pred.empty:
            logger.info(f"Prediction columns: {list(sample_pred.columns)}")
            logger.info(f"Prediction index names: {sample_pred.index.names}")
            logger.info(f"Prediction sample data:\n{sample_pred.head()}")
            logger.info(f"Prediction data types: {sample_pred.dtypes}")

    # Process ground truth tracks as queries
    for cam_name, gt_df in gt_data.items():
        if gt_df.empty:
            continue

        # Get unique track IDs from the index
        unique_track_ids = gt_df.index.get_level_values('Id').unique()
        logger.info(f"Processing ground truth for {cam_name}, total tracks: {len(unique_track_ids)}")

        # Group by track ID to get track-level information
        for track_id in unique_track_ids:
            # Get all rows for this track ID
            track_group = gt_df.xs(track_id, level='Id')

            if len(track_group) < 5:  # Skip very short tracks
                logger.debug(f"Skipping short track {track_id} in {cam_name} (length: {len(track_group)})")
                continue

            # Calculate track center and temporal info
            # NTU-MTMC format: [frame, ID, left, top, width, height, ...]
            # Convert left,top to center coordinates
            track_left = track_group['X'].mean()  # This is actually 'left' in NTU-MTMC
            track_top = track_group['Y'].mean()   # This is actually 'top' in NTU-MTMC
            track_width = track_group['Width'].mean()
            track_height = track_group['Height'].mean()

            # Calculate center coordinates
            track_center_x = track_left + track_width / 2
            track_center_y = track_top + track_height / 2

            # Get frame information from index
            track_start = track_group.index.min()
            track_end = track_group.index.max()
            track_duration = track_end - track_start + 1

            query_tracks.append({
                'track_id': f"{cam_name}_{track_id}",
                'cam_id': cam_name,
                'person_id': track_id,  # Use track ID as person ID for cross-camera matching
                'center_x': track_center_x,
                'center_y': track_center_y,
                'width': track_width,
                'height': track_height,
                'start_frame': track_start,
                'end_frame': track_end,
                'duration': track_duration,
                'num_detections': len(track_group)
            })

    # Process prediction tracks as gallery
    for cam_name, pred_df in pred_data.items():
        if pred_df.empty:
            continue

        # Get unique track IDs from the index
        unique_track_ids = pred_df.index.get_level_values('Id').unique()
        logger.info(f"Processing predictions for {cam_name}, total tracks: {len(unique_track_ids)}")

        for track_id in unique_track_ids:
            # Get all rows for this track ID
            track_group = pred_df.xs(track_id, level='Id')

            if len(track_group) < 5:  # Skip very short tracks
                logger.debug(f"Skipping short track {track_id} in {cam_name} (length: {len(track_group)})")
                continue

            # Calculate track center and temporal info
            # Prediction format might be different, check if it's center or corner coordinates
            if 'X' in track_group.columns and 'Y' in track_group.columns:
                # If X,Y are center coordinates
                track_center_x = track_group['X'].mean()
                track_center_y = track_group['Y'].mean()
            else:
                # If X,Y are corner coordinates (like NTU-MTMC format)
                track_left = track_group['X'].mean()
                track_top = track_group['Y'].mean()
                track_width = track_group['Width'].mean()
                track_height = track_group['Height'].mean()
                track_center_x = track_left + track_width / 2
                track_center_y = track_top + track_height / 2

            track_width = track_group['Width'].mean()
            track_height = track_group['Height'].mean()

            # Get frame information from index
            track_start = track_group.index.min()
            track_end = track_group.index.max()
            track_duration = track_end - track_start + 1

            gallery_tracks.append({
                'track_id': f"{cam_name}_{track_id}",
                'cam_id': cam_name,
                'person_id': track_id,
                'center_x': track_center_x,
                'center_y': track_center_y,
                'width': track_width,
                'height': track_height,
                'start_frame': track_start,
                'end_frame': track_end,
                'duration': track_duration,
                'num_detections': len(track_group)
            })

    logger.info(f"Extracted {len(query_tracks)} query tracks and {len(gallery_tracks)} gallery tracks")

    # Debug: Show some sample tracks
    if query_tracks:
        logger.info(f"Sample query track: {query_tracks[0]}")
    if gallery_tracks:
        logger.info(f"Sample gallery track: {gallery_tracks[0]}")

    return query_tracks, gallery_tracks


def compute_track_distance_matrix(query_tracks, gallery_tracks):
    """
    Compute distance matrix between query and gallery tracks
    """
    num_query = len(query_tracks)
    num_gallery = len(gallery_tracks)

    if num_query == 0 or num_gallery == 0:
        return np.array([]), [], [], [], []

    distmat = np.zeros((num_query, num_gallery))

    for i, query in enumerate(query_tracks):
        for j, gallery in enumerate(gallery_tracks):
            # Skip same camera matches (cross-camera evaluation)
            if query['cam_id'] == gallery['cam_id']:
                distmat[i, j] = float('inf')
                continue

            # Compute distance based on track characteristics
            # Spatial distance (normalized by image size)
            spatial_dist = np.sqrt(
                ((query['center_x'] - gallery['center_x']) / 1920) ** 2 +
                ((query['center_y'] - gallery['center_y']) / 1080) ** 2
            )

            # Size similarity
            size_similarity = 1 - abs(query['width'] * query['height'] - gallery['width'] * gallery['height']) / max(query['width'] * query['height'], gallery['width'] * gallery['height'])

            # Duration similarity
            duration_similarity = 1 - abs(query['duration'] - gallery['duration']) / max(query['duration'], gallery['duration'])

            # Combined distance (lower is better)
            distmat[i, j] = spatial_dist + (1 - size_similarity) + (1 - duration_similarity)

    # Extract arrays for evaluation
    q_pids = np.array([track['person_id'] for track in query_tracks])
    g_pids = np.array([track['person_id'] for track in gallery_tracks])
    q_camids = np.array([track['cam_id'] for track in query_tracks])
    g_camids = np.array([track['cam_id'] for track in gallery_tracks])

    return distmat, q_pids, g_pids, q_camids, g_camids


def print_formatted_table(summary, dataset_name="NTU-MTMC", save_to_file=True, output_file="ntu_mtmc_results.txt"):
    """Print evaluation results in a formatted table similar to Market1501 style"""
    output_lines = []

    output_lines.append(f"\n[Evaluation results for {dataset_name} in csv format:]")
    output_lines.append("[Evaluation results in csv format:]")

    # Define the metrics we want to display
    display_metrics = ['mota', 'motp', 'recall', 'precision', 'num_objects']
    metric_names = ['MOTA', 'MOTP', 'Recall', 'Precision', 'Objects']

    # Print header
    header = "Dataset"
    for metric in metric_names:
        header += f" | {metric}"
    output_lines.append(header)

    # Print separator
    separator = "---"
    for _ in metric_names:
        separator += " | :---:"
    output_lines.append(separator)

    # Print data rows for each camera
    for idx, camera in enumerate(summary.index):
        if camera == 'OVERALL':
            continue  # Skip overall for individual camera table

        row = f"{camera}"
        for metric in display_metrics:
            if metric in summary.columns:
                value = summary.loc[camera, metric]
                if metric in ['mota', 'motp', 'recall', 'precision']:
                    # Format as percentage
                    row += f" | {value:.2f}"
                else:
                    # Format as integer
                    row += f" | {int(value)}"
            else:
                row += " | N/A"
        output_lines.append(row)

    # Print overall results
    if 'OVERALL' in summary.index:
        output_lines.append("\n**Overall Results:**")
        overall_header = "Dataset"
        for metric in metric_names:
            overall_header += f" | {metric}"
        output_lines.append(overall_header)

        overall_separator = "---"
        for _ in metric_names:
            overall_separator += " | :---:"
        output_lines.append(overall_separator)

        row = f"{dataset_name}"
        for metric in display_metrics:
            if metric in summary.columns:
                value = summary.loc['OVERALL', metric]
                if metric in ['mota', 'motp', 'recall', 'precision']:
                    # Format as percentage
                    row += f" | {value:.2f}"
                else:
                    # Format as integer
                    row += f" | {int(value)}"
            else:
                row += " | N/A"
        output_lines.append(row)

    # Print to console
    for line in output_lines:
        print(line)

    # Save to file if requested
    if save_to_file:
        with open(output_file, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate MOTA and Cross-Camera metrics for NTU-MTMC dataset')
    parser.add_argument('--results_folder', type=str,
                       default='/root/ByteTrack/MCT_outputs/tracking_results',
                       help='Path to tracking results folder')
    parser.add_argument('--gt_folder', type=str,
                       default='/root/ByteTrack/NTU-MTMC/test',
                       help='Path to ground truth folder')
    parser.add_argument('--gt_type', type=str, default='',
                       help='Ground truth file suffix (empty for NTU-MTMC)')
    parser.add_argument('--output_file', type=str, default='ntu_mtmc_results.json',
                       help='Output JSON file to save results')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to file, only print to console')

    args = parser.parse_args()

    # evaluate MOTA for NTU-MTMC dataset
    results_folder = args.results_folder
    gt_folder = args.gt_folder
    mm.lap.default_solver = 'lap'

    # For NTU-MTMC, we don't use the _val_half suffix
    gt_type = args.gt_type
    print('gt_type (empty for NTU-MTMC)', gt_type)

    # Find ground truth files for NTU-MTMC cameras
    gtfiles = glob.glob(
        os.path.join(gt_folder, 'Cam*/gt/gt{}.txt'.format(gt_type)))
    print('gt_files:', gtfiles)

    # Find tracking result files
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, 'Cam*.txt')) if not os.path.basename(f).startswith('eval')]
    print('tracking_files:', tsfiles)

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')

    # Load ground truth files (extract camera name from path)
    gt = OrderedDict()
    for f in gtfiles:
        # Extract camera name (e.g., 'Cam1' from 'datasets/mot/train/Cam1/gt/gt.txt')
        cam_name = Path(f).parts[-3]
        gt[cam_name] = mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)

    # Load tracking results (extract camera name from filename)
    ts = OrderedDict()
    for f in tsfiles:
        # Extract camera name (e.g., 'Cam1' from 'Cam1.txt')
        cam_name = os.path.splitext(Path(f).parts[-1])[0]
        ts[cam_name] = mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1.0)

    print(f"Ground truth cameras: {list(gt.keys())}")
    print(f"Tracking result cameras: {list(ts.keys())}")

    # 1. Single Camera MOTA Evaluation
    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    logger.info('Running single-camera MOTA metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
                'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
                'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)

    # Normalize certain metrics
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])

    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                        'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']

    # Print the formatted table
    print_formatted_table(summary, "NTU-MTMC", save_to_file=False)

    # 2. Cross-Camera Evaluation (fast-reid style)
    logger.info('Running cross-camera evaluation')

    # Extract track features
    query_tracks, gallery_tracks = extract_track_features(gt, ts)

    if len(query_tracks) > 0 and len(gallery_tracks) > 0:
        # Compute distance matrix
        distmat, q_pids, g_pids, q_camids, g_camids = compute_track_distance_matrix(query_tracks, gallery_tracks)

        if distmat.size > 0:
            # Evaluate using Market1501 style metrics
            cmc, all_AP, all_INP = eval_market1501_style(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=10)

            # Calculate metrics
            mAP = np.mean(all_AP) if all_AP else 0.0
            mINP = np.mean(all_INP) if all_INP else 0.0
            rank1 = cmc[0] if len(cmc) > 0 else 0.0
            rank5 = cmc[4] if len(cmc) > 4 else 0.0
            rank10 = cmc[9] if len(cmc) > 9 else 0.0

            # Calculate metric score (same as fast-reid)
            metric_score = (mAP + rank1) / 2

            cross_camera_metrics = {
                'Rank-1': rank1 * 100,
                'Rank-5': rank5 * 100,
                'Rank-10': rank10 * 100,
                'mAP': mAP * 100,
                'mINP': mINP * 100,
                'metric': metric_score * 100,
                'num_query_tracks': len(query_tracks),
                'num_gallery_tracks': len(gallery_tracks)
            }
        else:
            cross_camera_metrics = {
                'Rank-1': 0.0,
                'Rank-5': 0.0,
                'Rank-10': 0.0,
                'mAP': 0.0,
                'mINP': 0.0,
                'metric': 0.0,
                'num_query_tracks': len(query_tracks),
                'num_gallery_tracks': len(gallery_tracks)
            }
    else:
        cross_camera_metrics = {
            'Rank-1': 0.0,
            'Rank-5': 0.0,
            'Rank-10': 0.0,
            'mAP': 0.0,
            'mINP': 0.0,
            'metric': 0.0,
            'num_query_tracks': len(query_tracks),
            'num_gallery_tracks': len(gallery_tracks)
        }

    # Print cross-camera results
    print("\n" + "="*80)
    print("CROSS-CAMERA EVALUATION RESULTS (fast-reid style)")
    print("="*80)
    print(f"Rank-1:  {cross_camera_metrics['Rank-1']:.2f}%")
    print(f"Rank-5:  {cross_camera_metrics['Rank-5']:.2f}%")
    print(f"Rank-10: {cross_camera_metrics['Rank-10']:.2f}%")
    print(f"mAP:     {cross_camera_metrics['mAP']:.2f}%")
    print(f"mINP:    {cross_camera_metrics['mINP']:.2f}%")
    print(f"metric:  {cross_camera_metrics['metric']:.2f}%")
    print(f"Query tracks: {cross_camera_metrics['num_query_tracks']}")
    print(f"Gallery tracks: {cross_camera_metrics['num_gallery_tracks']}")

    # Prepare results for JSON output
    results = {
        'dataset': 'NTU-MTMC',
        'single_camera_mota': {
            'cameras': list(summary.index),
            'metrics': summary.to_dict('index')
        },
        'cross_camera_metrics': cross_camera_metrics,
        'evaluation_info': {
            'gt_folder': gt_folder,
            'results_folder': results_folder,
            'num_gt_files': len(gtfiles),
            'num_pred_files': len(tsfiles),
            'gt_cameras': list(gt.keys()),
            'pred_cameras': list(ts.keys())
        }
    }

    # Save to JSON file
    if not args.no_save:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nComplete results saved to: {args.output_file}")

    print("\n" + "="*80)
    print("Detailed MOT Challenge Metrics:")
    print("="*80)
    detailed_results = mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names)
    print(detailed_results)

    # Also compute standard MOT metrics
    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    standard_results = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(standard_results)

    logger.info('Completed')


if __name__ == "__main__":
    main()