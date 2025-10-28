#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PersonViT Market-1501 Evaluation Script
Evaluates PersonViT checkpoint on Market-1501 dataset using Fast-Reid framework
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
import glob
import re
from pathlib import Path
from typing import List, Tuple, Dict
import time

# Add Fast-Reid to path
sys.path.append('fast-reid-mct')

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from fastreid.evaluation.rank import eval_market1501
import torchvision.transforms as T


class PersonViTMarket1501Evaluator:
    """Evaluator for PersonViT on Market-1501 dataset"""

    def __init__(self, checkpoint_path: str, dataset_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.device = device

        # Setup Fast-Reid configuration
        self.cfg = self._setup_config()
        self.model = self._load_model()

        # Setup data loaders
        self.test_loader, self.num_query = self._setup_data_loader()

    def _setup_config(self):
        """Setup Fast-Reid configuration for PersonViT"""
        cfg = get_cfg()

        # Load base configuration for ViT
        cfg.merge_from_file("fast-reid-mct/configs/Market1501/personvit_market1501.yml")

        # Update dataset path
        cfg.DATASETS.ROOT_DIR = self.dataset_path

        # Set model weights path
        cfg.MODEL.WEIGHTS = self.checkpoint_path

        # Freeze configuration
        cfg.freeze()

        return cfg

    def _load_model(self):
        """Load PersonViT model"""
        print(f"Loading PersonViT model from: {self.checkpoint_path}")

        # Build model
        model = build_model(self.cfg)

        # Load checkpoint
        Checkpointer(model).load(self.checkpoint_path)

        # Move to device
        model.to(self.device)
        model.eval()

        print("PersonViT model loaded successfully!")
        return model

    def _setup_data_loader(self):
        """Setup data loader for Market-1501"""
        print(f"Setting up data loader for Market-1501 at: {self.dataset_path}")

        # Build test loader using config
        test_loader, num_query = build_reid_test_loader(self.cfg, dataset_name="Market1501")

        print(f"Data loader setup complete. Query images: {num_query}")
        return test_loader, num_query

    def extract_features(self) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Extract features from all images in the dataset"""
        print("Extracting features from Market-1501 dataset...")

        feats = []
        pids = []
        camids = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if batch_idx % 100 == 0:
                    print(f"Processing batch {batch_idx}/{len(self.test_loader)}")

                # Move batch to device
                images = batch["images"].to(self.device)

                # Extract features
                features = self.model(images)

                # Collect features and metadata
                feats.append(features.cpu())
                pids.extend(batch["targets"].numpy())
                camids.extend(batch["camids"].numpy())

        # Concatenate all features
        feats = torch.cat(feats, dim=0)

        print(f"Feature extraction complete. Total features: {len(feats)}")
        return feats, np.array(pids), np.array(camids)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate PersonViT on Market-1501"""
        print("Starting PersonViT evaluation on Market-1501...")

        # Extract features
        feats, pids, camids = self.extract_features()

        # Split into query and gallery
        q_feat = feats[:self.num_query]
        g_feat = feats[self.num_query:]
        q_pids = pids[:self.num_query]
        g_pids = pids[self.num_query:]
        q_camids = camids[:self.num_query]
        g_camids = camids[self.num_query:]

        print(f"Query features: {q_feat.shape}")
        print(f"Gallery features: {g_feat.shape}")

        # Compute distance matrix
        print("Computing distance matrix...")
        distmat = 1 - torch.mm(q_feat, g_feat.t())
        distmat = distmat.numpy()

        # Evaluate using Market-1501 metrics
        print("Computing evaluation metrics...")
        cmc, all_AP, all_INP = eval_market1501(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50
        )

        # Calculate metrics
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        rank1 = cmc[0]
        rank5 = cmc[4]
        rank10 = cmc[9]

        # Calculate metric score (same as fast-reid)
        metric_score = (mAP + rank1) / 2

        results = {
            'Rank-1': rank1 * 100,
            'Rank-5': rank5 * 100,
            'Rank-10': rank10 * 100,
            'mAP': mAP * 100,
            'mINP': mINP * 100,
            'metric': metric_score * 100,
            'num_query': self.num_query,
            'num_gallery': len(g_pids)
        }

        return results

    def print_results(self, results: Dict[str, float]):
        """Print evaluation results in a formatted table"""
        print("\n" + "="*80)
        print("PersonViT Market-1501 Evaluation Results")
        print("="*80)
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Device: {self.device}")
        print("="*80)

        # Print metrics
        print(f"Rank-1:  {results['Rank-1']:.2f}%")
        print(f"Rank-5:  {results['Rank-5']:.2f}%")
        print(f"Rank-10: {results['Rank-10']:.2f}%")
        print(f"mAP:     {results['mAP']:.2f}%")
        print(f"mINP:    {results['mINP']:.2f}%")
        print(f"metric:  {results['metric']:.2f}%")
        print(f"Query images: {results['num_query']}")
        print(f"Gallery images: {results['num_gallery']}")
        print("="*80)

        # Save results to file
        output_file = "personvit_market1501_results.txt"
        with open(output_file, 'w') as f:
            f.write("PersonViT Market-1501 Evaluation Results\n")
            f.write("="*50 + "\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Device: {self.device}\n")
            f.write("="*50 + "\n")
            f.write(f"Rank-1:  {results['Rank-1']:.2f}%\n")
            f.write(f"Rank-5:  {results['Rank-5']:.2f}%\n")
            f.write(f"Rank-10: {results['Rank-10']:.2f}%\n")
            f.write(f"mAP:     {results['mAP']:.2f}%\n")
            f.write(f"mINP:    {results['mINP']:.2f}%\n")
            f.write(f"metric:  {results['metric']:.2f}%\n")
            f.write(f"Query images: {results['num_query']}\n")
            f.write(f"Gallery images: {results['num_gallery']}\n")

        print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PersonViT on Market-1501")
    parser.add_argument("--checkpoint", type=str,
                       default="/root/ByteTrack/pretrained/personvit_checkpoint0240.pth",
                       help="Path to PersonViT checkpoint")
    parser.add_argument("--dataset", type=str,
                       default="/root/ByteTrack/Market-1501-v15.09.15",
                       help="Path to Market-1501 dataset")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--convert-checkpoint", action="store_true",
                       help="Convert PersonViT checkpoint to Fast-Reid format first")

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
        os.system(f"python convert_personvit_to_fastreid.py --input {args.checkpoint}")
        args.checkpoint = "reid_weight/personvit_fastreid.pth"

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    try:
        # Initialize evaluator
        evaluator = PersonViTMarket1501Evaluator(
            checkpoint_path=args.checkpoint,
            dataset_path=args.dataset,
            device=args.device
        )

        # Run evaluation
        results = evaluator.evaluate()

        # Print results
        evaluator.print_results(results)

        print("\n✅ Evaluation completed successfully!")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())