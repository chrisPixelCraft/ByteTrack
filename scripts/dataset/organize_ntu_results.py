#!/usr/bin/env python3

import os
import shutil
import glob
from pathlib import Path

def organize_tracking_results():
    """
    Organize NTU-MTMC tracking results for MOTA evaluation
    """

    # Source directory where tracking results are stored
    track_vis_dir = "YOLOX_outputs/yolox_m_mix_det/track_vis"

    # Destination directory for organized results
    track_results_dir = "YOLOX_outputs/yolox_m_mix_det/track_results"

    # Create destination directory if it doesn't exist
    os.makedirs(track_results_dir, exist_ok=True)

    print(f"Organizing tracking results from {track_vis_dir} to {track_results_dir}")

    # Find all timestamp directories
    timestamp_dirs = glob.glob(os.path.join(track_vis_dir, "2025_*"))

    if not timestamp_dirs:
        print("No tracking results found!")
        return

    print(f"Found {len(timestamp_dirs)} tracking result directories")

    for timestamp_dir in sorted(timestamp_dirs):
        timestamp = os.path.basename(timestamp_dir)

        # Look for video files to determine camera name
        video_files = glob.glob(os.path.join(timestamp_dir, "*.MP4"))

        if not video_files:
            print(f"No video files found in {timestamp_dir}")
            continue

        for video_file in video_files:
            video_name = os.path.splitext(os.path.basename(video_file))[0]

            # Look for corresponding text file
            txt_file = os.path.join(track_vis_dir, f"{timestamp}.txt")

            if os.path.exists(txt_file):
                dest_file = os.path.join(track_results_dir, f"{video_name}.txt")

                # Copy the tracking results
                shutil.copy2(txt_file, dest_file)
                print(f"Copied {txt_file} -> {dest_file}")
            else:
                print(f"Warning: No text file found for {video_name} in {timestamp_dir}")

    print("\nOrganization complete!")
    print(f"Tracking results are now available at: {track_results_dir}/")
    print("You can now run MOTA evaluation with: python3 tools/ntu_mota.py")

def check_tracking_status():
    """
    Check the status of ongoing tracking processes
    """
    track_vis_dir = "YOLOX_outputs/yolox_m_mix_det/track_vis"

    if not os.path.exists(track_vis_dir):
        print("No tracking results directory found.")
        return

    # Find all timestamp directories
    timestamp_dirs = glob.glob(os.path.join(track_vis_dir, "2025_*"))

    print(f"Found {len(timestamp_dirs)} tracking result directories:")

    for timestamp_dir in sorted(timestamp_dirs):
        timestamp = os.path.basename(timestamp_dir)

        # Check for video files
        video_files = glob.glob(os.path.join(timestamp_dir, "*.MP4"))

        # Check for corresponding text file
        txt_file = os.path.join(track_vis_dir, f"{timestamp}.txt")

        status = "Complete" if os.path.exists(txt_file) else "In Progress"
        video_count = len(video_files)

        print(f"  {timestamp}: {video_count} video(s), Status: {status}")

        if video_files:
            for video_file in video_files:
                video_name = os.path.basename(video_file)
                video_size = os.path.getsize(video_file) / (1024*1024)  # MB
                print(f"    - {video_name} ({video_size:.1f} MB)")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        check_tracking_status()
    else:
        organize_tracking_results()