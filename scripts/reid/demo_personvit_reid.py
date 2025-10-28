#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PersonViT ReID Demonstration Script
Shows how to use PersonViT checkpoint for person re-identification tasks
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="PersonViT ReID Demonstration")
    parser.add_argument("--mode", choices=["test", "quick", "full"], default="test",
                       help="Demo mode: test (basic test), quick (2 cameras, 100 frames), full (all cameras)")
    parser.add_argument("--cameras", type=str, default="Cam1,Cam2",
                       help="Comma-separated list of camera IDs")
    parser.add_argument("--max_frames", type=int, default=100,
                       help="Maximum number of frames to process")
    parser.add_argument("--save_video", action="store_true",
                       help="Save visualization videos")
    parser.add_argument("--save_results", action="store_true",
                       help="Save tracking results")

    args = parser.parse_args()

    # Configuration paths
    reid_config = "reid_weight/personvit_config.yml"
    reid_model = "reid_weight/personvit_fastreid.pth"
    camera_dir = "NTU-MTMC/test"

    # Check if required files exist
    if not os.path.exists(reid_config):
        print(f"❌ Error: ReID config not found: {reid_config}")
        return 1

    if not os.path.exists(reid_model):
        print(f"❌ Error: ReID model not found: {reid_model}")
        return 1

    if not os.path.exists(camera_dir):
        print(f"❌ Error: Camera directory not found: {camera_dir}")
        return 1

    print("=" * 70)
    print("PersonViT ReID Multi-Camera Tracking Demonstration")
    print("=" * 70)
    print(f"ReID Config: {reid_config}")
    print(f"ReID Model: {reid_model}")
    print(f"Camera Directory: {camera_dir}")
    print(f"Mode: {args.mode}")
    print("=" * 70)

    # Build command based on mode
    if args.mode == "test":
        # Basic test with minimal processing
        cmd = f"""python tools/demo_multi_camera_track.py \\
    --camera_dir {camera_dir} \\
    --cameras {args.cameras} \\
    --reid_config {reid_config} \\
    --reid_model {reid_model} \\
    --max_frames 10 \\
    --progress_bar \\
    --save_results"""

    elif args.mode == "quick":
        # Quick demo with 2 cameras and 100 frames
        cmd = f"""python tools/demo_multi_camera_track.py \\
    --camera_dir {camera_dir} \\
    --cameras {args.cameras} \\
    --reid_config {reid_config} \\
    --reid_model {reid_model} \\
    --max_frames {args.max_frames} \\
    --progress_bar \\
    --save_results"""

    else:  # full mode
        # Full processing with all cameras
        cmd = f"""python tools/demo_multi_camera_track.py \\
    --camera_dir {camera_dir} \\
    --cameras all \\
    --reid_config {reid_config} \\
    --reid_model {reid_model} \\
    --save_video \\
    --save_results \\
    --progress_bar"""

    print("🚀 Running PersonViT ReID Multi-Camera Tracking...")
    print("\nCommand:")
    print(cmd)
    print("\n" + "=" * 70)

    # Execute the command
    exit_code = os.system(cmd)

    if exit_code == 0:
        print("\n✅ PersonViT ReID demonstration completed successfully!")
        print("\n📊 Results saved to:")
        print("   - Tracking results: MCT_outputs/tracking_results/")
        print("   - Visualization videos: MCT_outputs/ (if --save_video was used)")
        print("   - Cross-camera associations: MCT_outputs/tracking_results/cross_camera_associations.txt")
    else:
        print(f"\n❌ Demonstration failed with exit code: {exit_code}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())