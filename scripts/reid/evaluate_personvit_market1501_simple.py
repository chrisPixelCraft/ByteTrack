#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple PersonViT Market-1501 Evaluation Script
Uses Fast-Reid framework to evaluate PersonViT checkpoint on Market-1501 dataset
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Evaluate PersonViT on Market-1501 using Fast-Reid")
    parser.add_argument("--checkpoint", type=str,
                       default="/root/ByteTrack/pretrained/personvit_checkpoint0240.pth",
                       help="Path to PersonViT checkpoint")
    parser.add_argument("--dataset", type=str,
                       default="/root/ByteTrack/Market-1501-v15.09.15",
                       help="Path to Market-1501 dataset")
    parser.add_argument("--convert-checkpoint", action="store_true",
                       help="Convert PersonViT checkpoint to Fast-Reid format first")
    parser.add_argument("--device", type=str, default="0",
                       help="GPU device ID")

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found: {args.dataset}")
        return 1

    # Check if bounding_box_test exists
    test_dir = os.path.join(args.dataset, "bounding_box_test")
    if not os.path.exists(test_dir):
        print(f"Error: bounding_box_test directory not found: {test_dir}")
        return 1

    # Convert checkpoint if requested
    if args.convert_checkpoint:
        print("Converting PersonViT checkpoint to Fast-Reid format...")
        convert_cmd = f"python convert_personvit_to_fastreid.py --input {args.checkpoint}"
        result = subprocess.run(convert_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error converting checkpoint: {result.stderr}")
            return 1
        print("Checkpoint converted successfully!")
        args.checkpoint = "reid_weight/personvit_fastreid.pth"

    # Check if Fast-Reid config exists
    config_path = "fast-reid-mct/configs/Market1501/bagtricks_vit.yml"
    if not os.path.exists(config_path):
        print(f"Error: Fast-Reid config not found: {config_path}")
        return 1

    print("="*80)
    print("PersonViT Market-1501 Evaluation")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {config_path}")
    print(f"Device: {args.device}")
    print("="*80)

    # Set environment variables
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.device

    # Change to Fast-Reid directory
    os.chdir('fast-reid-mct')

    # Run evaluation using Fast-Reid
    eval_cmd = [
        "python", "tools/train_net.py",
        "--config-file", "../configs/Market1501/bagtricks_vit.yml",
        "--eval-only",
        "MODEL.WEIGHTS", f"../{args.checkpoint}",
        "DATASETS.ROOT_DIR", f"../{args.dataset}"
    ]

    print("Running evaluation command:")
    print(" ".join(eval_cmd))
    print()

    try:
        result = subprocess.run(eval_cmd, env=env, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Evaluation completed successfully!")
            print("\nResults:")
            print(result.stdout)

            # Save results to file
            output_file = "../personvit_market1501_fastreid_results.txt"
            with open(output_file, 'w') as f:
                f.write("PersonViT Market-1501 Evaluation Results (Fast-Reid)\n")
                f.write("="*60 + "\n")
                f.write(f"Checkpoint: {args.checkpoint}\n")
                f.write(f"Dataset: {args.dataset}\n")
                f.write(f"Config: {config_path}\n")
                f.write("="*60 + "\n")
                f.write(result.stdout)

            print(f"\nResults saved to: {output_file}")

        else:
            print(f"❌ Evaluation failed with return code: {result.returncode}")
            print("Error output:")
            print(result.stderr)
            return 1

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())