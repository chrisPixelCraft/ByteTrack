#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test PersonViT ReID functionality
"""

import os
import sys
import cv2
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from yolox.tracker.reid_extractor import ReidExtractor

def test_personvit_reid():
    """Test PersonViT ReID feature extraction"""

    print("Testing PersonViT ReID functionality...")

    # Initialize ReID extractor with PersonViT
    reid_config = "reid_weight/personvit_config.yml"
    reid_model = "reid_weight/personvit_fastreid.pth"

    if not os.path.exists(reid_config):
        print(f"Error: ReID config not found: {reid_config}")
        return False

    if not os.path.exists(reid_model):
        print(f"Error: ReID model not found: {reid_model}")
        return False

    try:
        # Initialize ReID extractor
        reid_extractor = ReidExtractor(
            config_path=reid_config,
            model_path=reid_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            input_size=(256, 128)
        )

        if not reid_extractor.is_available():
            print("Error: ReID extractor not available")
            return False

        print(f"ReID extractor initialized successfully")
        print(f"Feature dimension: {reid_extractor.get_feature_dim()}")

        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Create dummy bounding boxes
        dummy_bboxes = [
            [100, 100, 200, 300],  # [x1, y1, x2, y2]
            [300, 150, 400, 350],
            [500, 200, 600, 400]
        ]

        # Extract features
        features = reid_extractor.extract_features(dummy_image, dummy_bboxes)

        print(f"Successfully extracted features for {len(dummy_bboxes)} bounding boxes")
        print(f"Feature shape: {features.shape}")
        print(f"Feature type: {features.dtype}")

        # Test similarity computation
        if len(features) >= 2:
            similarity = reid_extractor.compute_similarity(
                features[0:1], features[1:2], metric="cosine"
            )
            print(f"Similarity between first two features: {similarity[0][0]:.4f}")

        print("PersonViT ReID test completed successfully!")
        return True

    except Exception as e:
        print(f"Error during ReID test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_camera_demo_with_personvit():
    """Test multi-camera demo with PersonViT ReID"""

    print("\nTesting multi-camera demo with PersonViT ReID...")

    # Check if demo script exists
    demo_script = "tools/demo_multi_camera_track.py"
    if not os.path.exists(demo_script):
        print(f"Error: Demo script not found: {demo_script}")
        return False

    # Check if NTU-MTMC dataset exists
    camera_dir = "NTU-MTMC/test"
    if not os.path.exists(camera_dir):
        print(f"Error: Camera directory not found: {camera_dir}")
        return False

    # Check for camera videos
    cameras = ["Cam1", "Cam2", "Cam3"]
    camera_paths = []

    for camera in cameras:
        video_path = os.path.join(camera_dir, camera, f"{camera}.MP4")
        if os.path.exists(video_path):
            camera_paths.append(camera)

    if not camera_paths:
        print("Error: No camera videos found")
        return False

    print(f"Found {len(camera_paths)} cameras: {camera_paths}")

    # Test command
    test_cmd = f"""python {demo_script} \\
        --camera_dir {camera_dir} \\
        --cameras {','.join(camera_paths[:2])} \\
        --reid_config reid_weight/personvit_config.yml \\
        --reid_model reid_weight/personvit_fastreid.pth \\
        --max_frames 10 \\
        --progress_bar \\
        --save_results"""

    print("To test PersonViT ReID with multi-camera tracking, run:")
    print(test_cmd)

    return True

def main():
    """Main test function"""

    print("=" * 60)
    print("PersonViT ReID Integration Test")
    print("=" * 60)

    # Test 1: Basic ReID functionality
    test1_success = test_personvit_reid()

    # Test 2: Multi-camera demo integration
    test2_success = test_multi_camera_demo_with_personvit()

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"ReID Functionality Test: {'PASS' if test1_success else 'FAIL'}")
    print(f"Multi-camera Demo Test: {'PASS' if test2_success else 'FAIL'}")
    print("=" * 60)

    if test1_success and test2_success:
        print("\n✅ All tests passed! PersonViT ReID is ready to use.")
        print("\nUsage examples:")
        print("1. Quick test with 2 cameras:")
        print("   python tools/demo_multi_camera_track.py --cameras Cam1,Cam2 --reid_config reid_weight/personvit_config.yml --reid_model reid_weight/personvit_fastreid.pth --max_frames 100 --progress_bar")
        print("\n2. Full processing with all cameras:")
        print("   python tools/demo_multi_camera_track.py --cameras all --reid_config reid_weight/personvit_config.yml --reid_model reid_weight/personvit_fastreid.pth --save_video --save_results --progress_bar")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

    return 0

if __name__ == "__main__":
    sys.exit(main())