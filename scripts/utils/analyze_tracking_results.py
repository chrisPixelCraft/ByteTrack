import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_tracking_results(result_file):
    """
    Analyze tracking results without ground truth
    MOT format: frame_id, track_id, x, y, w, h, score, -1, -1, -1
    """
    if not os.path.exists(result_file):
        print(f"Result file {result_file} not found!")
        return

    # Read tracking results
    data = np.loadtxt(result_file, delimiter=',')

    if len(data) == 0:
        print("No tracking data found!")
        return

    frames = data[:, 0].astype(int)
    track_ids = data[:, 1].astype(int)
    scores = data[:, 6]

    # Basic statistics
    print("=== TRACKING ANALYSIS (Without Ground Truth) ===")
    print(f"Total frames processed: {frames.max()}")
    print(f"Total detections: {len(data)}")
    print(f"Unique tracks: {len(np.unique(track_ids))}")
    print(f"Average detections per frame: {len(data) / frames.max():.2f}")
    print(f"Average detection score: {scores.mean():.3f}")
    print(f"Min/Max detection scores: {scores.min():.3f} / {scores.max():.3f}")

    # Track duration analysis
    track_durations = {}
    track_frames = defaultdict(list)

    for i in range(len(data)):
        tid = track_ids[i]
        frame = frames[i]
        track_frames[tid].append(frame)

    for tid, frame_list in track_frames.items():
        duration = max(frame_list) - min(frame_list) + 1
        track_durations[tid] = duration

    durations = list(track_durations.values())
    print(f"\n=== TRACK DURATION ANALYSIS ===")
    print(f"Average track duration: {np.mean(durations):.2f} frames")
    print(f"Median track duration: {np.median(durations):.2f} frames")
    print(f"Longest track: {max(durations)} frames")
    print(f"Shortest track: {min(durations)} frames")

    # Tracks by duration categories
    very_short = sum(1 for d in durations if d <= 5)
    short = sum(1 for d in durations if 5 < d <= 20)
    medium = sum(1 for d in durations if 20 < d <= 50)
    long_tracks = sum(1 for d in durations if d > 50)

    print(f"\n=== TRACK CATEGORIES ===")
    print(f"Very short tracks (≤5 frames): {very_short}")
    print(f"Short tracks (6-20 frames): {short}")
    print(f"Medium tracks (21-50 frames): {medium}")
    print(f"Long tracks (>50 frames): {long_tracks}")

    # Objects per frame analysis
    objects_per_frame = defaultdict(int)
    for frame in frames:
        objects_per_frame[frame] += 1

    frame_counts = list(objects_per_frame.values())
    print(f"\n=== OBJECTS PER FRAME ===")
    print(f"Average objects per frame: {np.mean(frame_counts):.2f}")
    print(f"Max objects in single frame: {max(frame_counts)}")
    print(f"Min objects in single frame: {min(frame_counts)}")

    # Potential tracking issues (heuristics)
    print(f"\n=== POTENTIAL TRACKING ISSUES ===")
    print(f"Very short tracks (potential false positives): {very_short}")

    # Track ID reuse detection
    track_id_gaps = []
    for tid, frame_list in track_frames.items():
        frame_list.sort()
        gaps = []
        for i in range(1, len(frame_list)):
            gap = frame_list[i] - frame_list[i-1] - 1
            if gap > 0:
                gaps.append(gap)
        if gaps:
            track_id_gaps.extend(gaps)

    if track_id_gaps:
        print(f"Track gaps detected: {len(track_id_gaps)} gaps")
        print(f"Average gap size: {np.mean(track_id_gaps):.2f} frames")

    return {
        'total_frames': frames.max(),
        'total_detections': len(data),
        'unique_tracks': len(np.unique(track_ids)),
        'avg_detections_per_frame': len(data) / frames.max(),
        'avg_score': scores.mean(),
        'track_durations': durations,
        'objects_per_frame': frame_counts
    }

if __name__ == "__main__":
    # Example usage
    result_file = "YOLOX_outputs/bytetrack_x_mot17/track_results/Cam4.txt"

    # Check common output locations
    possible_files = [
        "YOLOX_outputs/bytetrack_x_mot17/track_results/Cam4.txt",
        "YOLOX_outputs/yolox_x_mix_det/track_results/Cam4.txt",
        "YOLOX_outputs/track_results/Cam4.txt",
        "videos/Cam4.txt"
    ]

    found = False
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"Found tracking results: {file_path}")
            analyze_tracking_results(file_path)
            found = True
            break

    if not found:
        print("No tracking result files found!")
        print("Expected locations:")
        for path in possible_files:
            print(f"  - {path}")
        print("\nRun tracking first with:")
        print("python3 tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result --path Cam4.mp4")