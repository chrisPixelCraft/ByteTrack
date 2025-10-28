#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

def organize_ntu_dataset():
    """
    Organize NTU-MTMC dataset structure to match MOTA evaluation expectations
    """

    # Source directory
    ntu_test_dir = "NTU-MTMC/test"

    # Destination directory
    mot_train_dir = "datasets/mot/train"

    # Create destination directory if it doesn't exist
    os.makedirs(mot_train_dir, exist_ok=True)

    # Process each camera
    for cam_dir in sorted(os.listdir(ntu_test_dir)):
        if cam_dir.startswith("Cam") and os.path.isdir(os.path.join(ntu_test_dir, cam_dir)):
            source_gt_file = os.path.join(ntu_test_dir, cam_dir, "gt", "gt.txt")

            if os.path.exists(source_gt_file):
                # Create sequence directory
                seq_dir = os.path.join(mot_train_dir, cam_dir)
                gt_dir = os.path.join(seq_dir, "gt")
                os.makedirs(gt_dir, exist_ok=True)

                # Copy or link the ground truth file
                dest_gt_file = os.path.join(gt_dir, "gt.txt")

                # Create symbolic link (recommended) or copy
                if os.path.exists(dest_gt_file):
                    os.remove(dest_gt_file)

                # Create symbolic link
                os.symlink(os.path.abspath(source_gt_file), dest_gt_file)

                print(f"Linked {source_gt_file} -> {dest_gt_file}")
            else:
                print(f"Warning: No ground truth file found for {cam_dir}")

    print("\nDataset organization complete!")
    print(f"Ground truth files are now available at: {mot_train_dir}/[CamX]/gt/gt.txt")
    print(f"Make sure your tracking results are saved to: YOLOX_outputs/yolox_x_ablation/track_results/[CamX].txt")

if __name__ == "__main__":
    organize_ntu_dataset()