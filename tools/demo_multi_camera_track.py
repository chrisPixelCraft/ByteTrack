#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Camera Tracking Demo for NTU-MTMC Dataset
Demonstrates cross-camera person tracking using ByteTrack + Fast-Reid

Usage Examples:
    # Quick test on 3 cameras
    python tools/demo_multi_camera_track.py --quick-test --cameras Cam1,Cam2,Cam3

    # Full processing with all cameras
    python tools/demo_multi_camera_track.py \
        --cameras all --save_video --save_results --progress_bar

    # Custom configuration
    python tools/demo_multi_camera_track.py \
        -f exps/example/mot/yolox_m_mix_det.py \
        -c pretrained/bytetrack_m_mot17.pth.tar \
        --camera_dir NTU-MTMC/test \
        --cameras Cam1,Cam2,Cam3,Cam4,Cam5 \
        --save_video --save_results --fp16 --fuse
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install required packages: pip install opencv-python torch numpy")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from yolox.exp import get_exp
    from yolox.utils import fuse_model, get_model_info, postprocess
    from yolox.tracker.multi_camera_tracker import MultiCameraTracker
    from yolox.utils.visualize import plot_tracking
except ImportError as e:
    print(f"Error importing YOLOX modules: {e}")
    print("Please ensure YOLOX is properly installed: python setup.py develop")
    sys.exit(1)

# Check for tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Progress bar will be disabled.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiCameraDemo:
    """
    Multi-Camera Tracking Demo Application
    """

    def __init__(self,
                 exp_file: str,
                 model_path: str,
                 camera_paths: Dict[str, str],
                 output_dir: str = "MCT_outputs",
                 reid_config: Optional[str] = None,
                 reid_model: Optional[str] = None,
                 save_video: bool = True,
                 save_results: bool = True,
                 fp16: bool = False,
                 fuse: bool = False):
        """
        Initialize Multi-Camera Demo

        Args:
            exp_file: Path to experiment file
            model_path: Path to model weights
            camera_paths: Dictionary of camera_id -> video_path
            output_dir: Output directory for results
            reid_config: Path to Fast-Reid config
            reid_model: Path to Fast-Reid model
            save_video: Whether to save visualization videos
            save_results: Whether to save tracking results
            fp16: Use FP16 precision
            fuse: Fuse model for faster inference
        """
        self.exp_file = exp_file
        self.model_path = model_path
        self.camera_paths = camera_paths
        self.output_dir = output_dir
        self.reid_config = reid_config
        self.reid_model = reid_model
        self.save_video = save_video
        self.save_results = save_results
        self.fp16 = fp16
        self.fuse = fuse

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize model and tracker
        self._setup_model()
        self._setup_tracker()
        self._setup_video_writers()

        # Results storage
        self.results = {camera_id: [] for camera_id in camera_paths.keys()}

        logger.info(f"Multi-Camera Demo initialized for {len(camera_paths)} cameras")
        logger.info(f"Output directory: {output_dir}")

    def _setup_model(self):
        """
        Setup detection model
        """
        # Validate model path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        # Load experiment
        exp = get_exp(self.exp_file, None)
        exp.test_conf = 0.25
        exp.nmsthre = 0.45
        exp.test_size = (800, 1440)

        # Setup model
        model = exp.get_model()

        # Check CUDA availability
        if torch.cuda.is_available():
            model.cuda()
            logger.info("Using CUDA for inference")
        else:
            logger.warning("CUDA not available, using CPU")

        model.eval()

        # Load weights
        try:
            ckpt = torch.load(self.model_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            logger.info(f"Model weights loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise RuntimeError(f"Failed to load model weights from {self.model_path}")

        # Fuse model if requested
        if self.fuse:
            model = fuse_model(model)
            logger.info("Model fused for faster inference")

        # Setup FP16 if requested
        if self.fp16:
            if torch.cuda.is_available():
                model.half()
                logger.info("FP16 precision enabled")
            else:
                logger.warning("FP16 requested but CUDA not available, using FP32")

        self.model = model
        self.exp = exp

        logger.info(f"Model info: {get_model_info(model, exp.test_size)}")

    def _setup_tracker(self):
        """
        Setup multi-camera tracker
        """
        # Create args object
        class Args:
            def __init__(self):
                self.track_thresh = 0.5
                self.track_buffer = 30
                self.match_thresh = 0.8
                self.mot20 = False
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.use_reid = True

        args = Args()

        # Initialize multi-camera tracker
        try:
            self.tracker = MultiCameraTracker(
                args=args,
                camera_ids=list(self.camera_paths.keys()),
                reid_config=self.reid_config,
                reid_model=self.reid_model,
                cross_camera_thresh=0.4,
                cross_camera_interval=30,
                max_time_gap=300
            )
            logger.info("Multi-camera tracker initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing tracker: {e}")
            raise RuntimeError(f"Failed to initialize multi-camera tracker: {e}")

    def _setup_video_writers(self):
        """
        Setup video writers for output
        """
        self.video_writers = {}
        self.video_caps = {}

        # Open video captures
        for camera_id, video_path in self.camera_paths.items():
            if not os.path.exists(video_path):
                logger.warning(f"Video file not found: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video {video_path}")
                continue

            # Verify video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Video {camera_id}: {width}x{height} @ {fps:.1f}fps, {frame_count} frames")

            self.video_caps[camera_id] = cap

            # Setup video writer if saving
            if self.save_video:
                output_path = os.path.join(self.output_dir, f"{camera_id}_mct_output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                self.video_writers[camera_id] = writer

        logger.info(f"Video captures opened for {len(self.video_caps)} cameras")

        if not self.video_caps:
            raise RuntimeError("No video captures could be opened!")

    def run(self, max_frames: Optional[int] = None, progress_bar: bool = True):
        """
        Run multi-camera tracking demo

        Args:
            max_frames: Maximum number of frames to process (for testing)
            progress_bar: Whether to show progress bar
        """
        logger.info("Starting multi-camera tracking...")

        # Calculate total frames for progress bar
        total_frames = None
        if self.video_caps:
            total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in self.video_caps.values())
            if max_frames is not None:
                total_frames = min(total_frames, max_frames)

        # Setup progress bar
        if progress_bar and TQDM_AVAILABLE and total_frames:
            pbar = tqdm(total=total_frames, desc="Processing frames", unit="frames",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            pbar = None

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                # Check max frames limit
                if max_frames is not None and frame_count >= max_frames:
                    logger.info(f"Reached max frames limit: {max_frames}")
                    break

                # Read frames from all cameras
                frames = {}
                frame_available = False

                for camera_id, cap in self.video_caps.items():
                    ret, frame = cap.read()
                    if ret:
                        frames[camera_id] = frame
                        frame_available = True

                # Break if no frames available
                if not frame_available:
                    logger.info("No more frames available")
                    break

                # Process frames
                try:
                    results = self._process_frames(frames)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
                    continue

                # Save results
                if self.save_results:
                    self._save_frame_results(results, frame_count)

                # Visualize and save videos
                if self.save_video:
                    self._save_visualization(frames, results)

                # Display statistics every 10 frames
                if frame_count % 10 == 0:
                    self._display_statistics(frame_count, start_time)

                # Update progress bar with real-time FPS
                if pbar:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    if elapsed > 0:
                        current_fps = frame_count / elapsed
                        pbar.set_description(f"Processing frames (FPS: {current_fps:.2f})")
                    pbar.update(1)

                frame_count += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(traceback.format_exc())
        finally:
            # Close progress bar
            if pbar:
                pbar.close()

            # Cleanup
            self._cleanup()

        # Print final statistics
        elapsed_time = time.time() - start_time
        logger.info(f"\\nMulti-camera tracking completed!")
        logger.info(f"Total frames processed: {frame_count}")
        logger.info(f"Total time: {elapsed_time:.2f}s")
        if frame_count > 0:
            logger.info(f"Average FPS: {frame_count / elapsed_time:.2f}")

        return frame_count, elapsed_time

    def _process_frames(self, frames: Dict[str, np.ndarray]) -> Dict[str, List]:
        """
        Process frames through detection and tracking

        Args:
            frames: Dictionary of camera_id -> frame

        Returns:
            Dictionary of camera_id -> tracked objects
        """
        detections_dict = {}
        img_info_dict = {}
        img_size_dict = {}

        # Run detection on all frames
        for camera_id, frame in frames.items():
            # Preprocess frame
            img, ratio = self._preprocess_frame(frame)

            # Run detection
            with torch.no_grad():
                outputs = self.model(img)
                outputs = postprocess(
                    outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre
                )

            # Store results
            if outputs[0] is not None:
                detections_dict[camera_id] = outputs[0]
                img_info_dict[camera_id] = (frame.shape[0], frame.shape[1])
                img_size_dict[camera_id] = img.shape[2:]
            else:
                detections_dict[camera_id] = np.empty((0, 5))
                img_info_dict[camera_id] = (frame.shape[0], frame.shape[1])
                img_size_dict[camera_id] = img.shape[2:]

        # Update tracker
        results = self.tracker.update(
            detections_dict=detections_dict,
            img_info_dict=img_info_dict,
            img_size_dict=img_size_dict,
            img_dict=frames
        )

        return results

    def _preprocess_frame(self, frame: np.ndarray) -> tuple:
        """
        Preprocess frame for detection

        Args:
            frame: Input frame

        Returns:
            Preprocessed tensor and ratio
        """
        img_size = self.exp.test_size
        img, ratio = self._resize_image(frame, img_size)

        # Convert to tensor
        img = torch.from_numpy(img).unsqueeze(0).float()

        if torch.cuda.is_available():
            img = img.cuda()

        if self.fp16:
            img = img.half()

        return img, ratio

    def _resize_image(self, image: np.ndarray, target_size: tuple) -> tuple:
        """
        Resize image while maintaining aspect ratio

        Args:
            image: Input image
            target_size: Target size (height, width)

        Returns:
            Resized image and ratio
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size

        # Calculate ratio
        ratio = min(target_h / h, target_w / w)

        # Resize image
        new_h, new_w = int(h * ratio), int(w * ratio)
        resized = cv2.resize(image, (new_w, new_h))

        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Convert to CHW format
        padded = padded.transpose(2, 0, 1)

        return padded, ratio

    def _save_frame_results(self, results: Dict[str, List], frame_id: int):
        """
        Save tracking results for current frame

        Args:
            results: Dictionary of camera_id -> tracked objects
            frame_id: Current frame ID
        """
        for camera_id, tracks in results.items():
            for track in tracks:
                if track.is_activated:
                    # MOT format: frame_id, track_id, x, y, w, h, conf, class, visibility
                    x, y, w, h = track.tlwh
                    global_id = track.get_display_id()

                    # Use actual frame number (frame_id + 1) and format coordinates as integers to match ground truth
                    # MOT format: [frame, ID, left, top, width, height, 1, -1, -1, -1] where 7th field is 1 for valid detection
                    actual_frame_num = frame_id + 1
                    result_line = f"{actual_frame_num},{global_id},{int(round(x))},{int(round(y))},{int(round(w))},{int(round(h))},1,-1,-1,-1\n"
                    self.results[camera_id].append(result_line)

    def _save_visualization(self, frames: Dict[str, np.ndarray], results: Dict[str, List]):
        """
        Save visualization videos

        Args:
            frames: Dictionary of camera_id -> frame
            results: Dictionary of camera_id -> tracked objects
        """
        # Create visualization with cross-camera associations
        try:
            vis_frames = self.tracker.visualize_cross_camera_associations(results, frames)

            if vis_frames:
                for camera_id, vis_frame in vis_frames.items():
                    if camera_id in self.video_writers:
                        self.video_writers[camera_id].write(vis_frame)
        except Exception as e:
            logger.warning(f"Error creating visualization: {e}")

    def _display_statistics(self, frame_count: int, start_time: float):
        """
        Display tracking statistics

        Args:
            frame_count: Current frame count
            start_time: Start time
        """
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        stats = self.tracker.get_statistics()

        logger.info(f"Frame {frame_count}: FPS={fps:.2f}, Total tracks={stats['total_tracks']}, "
                   f"Global tracks={stats['active_global_tracks']}, "
                   f"Cross-camera associations={stats['cross_camera_associations']}")

    def _cleanup(self):
        """
        Clean up resources
        """
        # Close video captures
        for cap in self.video_caps.values():
            cap.release()

        # Close video writers
        for writer in self.video_writers.values():
            writer.release()

        # Save tracking results
        if self.save_results:
            self._save_tracking_results()

        cv2.destroyAllWindows()

    def _save_tracking_results(self):
        """
        Save final tracking results
        """
        results_dir = os.path.join(self.output_dir, "tracking_results")
        os.makedirs(results_dir, exist_ok=True)

        for camera_id, results in self.results.items():
            output_path = os.path.join(results_dir, f"{camera_id}.txt")
            with open(output_path, 'w') as f:
                f.writelines(results)
            logger.info(f"Saved tracking results for {camera_id}: {output_path}")

        # Save cross-camera associations
        try:
            associations = self.tracker.get_cross_camera_associations()
            assoc_path = os.path.join(results_dir, "cross_camera_associations.txt")
            with open(assoc_path, 'w') as f:
                for assoc in associations:
                    f.write(f"{assoc}\n")
            logger.info(f"Saved cross-camera associations: {assoc_path}")
        except Exception as e:
            logger.warning(f"Error saving cross-camera associations: {e}")


def make_parser():
    """
    Create argument parser for multi-camera tracking demo
    """
    parser = argparse.ArgumentParser(
        "Multi-Camera Tracking Demo",
        description="Demonstrate cross-camera person tracking using ByteTrack + Fast-Reid"
    )

    # Model arguments
    parser.add_argument(
        "-f", "--exp_file",
        default="exps/example/mot/yolox_m_mix_det.py",
        type=str,
        help="Path to experiment file"
    )
    parser.add_argument(
        "-c", "--ckpt",
        default="pretrained/bytetrack_m_mot17.pth.tar",
        type=str,
        help="Path to checkpoint file"
    )

    # Camera arguments
    parser.add_argument(
        "--camera_dir",
        default="NTU-MTMC/test",
        type=str,
        help="Directory containing camera videos"
    )
    parser.add_argument(
        "--cameras",
        default="Cam1,Cam2,Cam3",
        type=str,
        help="Comma-separated list of camera IDs or 'all' for all cameras"
    )

    # ReID arguments
    parser.add_argument(
        "--reid_config",
        default=None,
        type=str,
        help="Path to Fast-Reid config file"
    )
    parser.add_argument(
        "--reid_model",
        default=None,
        type=str,
        help="Path to Fast-Reid model file"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        default="MCT_outputs",
        type=str,
        help="Output directory for results"
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save visualization videos"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save tracking results"
    )

    # Processing options
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (for testing)"
    )
    parser.add_argument(
        "--progress_bar",
        action="store_true",
        help="Show progress bar"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Quick test with first 100 frames"
    )

    # Model options
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision"
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Fuse model for faster inference"
    )

    return parser


def validate_arguments(args):
    """
    Validate command line arguments
    """
    # Check experiment file
    if not os.path.exists(args.exp_file):
        raise FileNotFoundError(f"Experiment file not found: {args.exp_file}")

    # Check checkpoint file
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")

    # Check camera directory
    if not os.path.exists(args.camera_dir):
        raise FileNotFoundError(f"Camera directory not found: {args.camera_dir}")

    # Validate cameras
    if args.cameras == "all":
        args.cameras = "Cam1,Cam2,Cam3,Cam4,Cam5,Cam6,Cam7,Cam8,Cam9,Cam10,Cam11"

    # Quick test settings
    if args.quick_test:
        args.max_frames = 100
        args.progress_bar = True
        logger.info("Quick test mode: processing first 100 frames")


def main():
    """
    Main function
    """
    # Parse arguments
    args = make_parser().parse_args()

    try:
        # Validate arguments
        validate_arguments(args)

        # Setup camera paths
        camera_ids = args.cameras.split(',')
        camera_paths = {}

        for camera_id in camera_ids:
            camera_id = camera_id.strip()
            video_path = os.path.join(args.camera_dir, camera_id, f"{camera_id}.MP4")
            if os.path.exists(video_path):
                camera_paths[camera_id] = video_path
            else:
                logger.warning(f"Video not found: {video_path}")

        if not camera_paths:
            logger.error("Error: No camera videos found!")
            return 1

        logger.info(f"Found {len(camera_paths)} cameras: {list(camera_paths.keys())}")

        # Create and run demo
        demo = MultiCameraDemo(
            exp_file=args.exp_file,
            model_path=args.ckpt,
            camera_paths=camera_paths,
            output_dir=args.output_dir,
            reid_config=args.reid_config,
            reid_model=args.reid_model,
            save_video=args.save_video,
            save_results=args.save_results,
            fp16=args.fp16,
            fuse=args.fuse
        )

        # Run tracking
        frame_count, elapsed_time = demo.run(
            max_frames=args.max_frames,
            progress_bar=args.progress_bar
        )

        logger.info("Demo completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())