#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert PersonViT checkpoint to Fast-Reid compatible format
"""

import os
import sys
import torch
import argparse
from pathlib import Path

def convert_personvit_checkpoint(personvit_path, output_path):
    """
    Convert PersonViT checkpoint to Fast-Reid compatible format

    Args:
        personvit_path: Path to PersonViT checkpoint
        output_path: Path to save converted checkpoint
    """
    print(f"Loading PersonViT checkpoint from: {personvit_path}")

    # Load PersonViT checkpoint
    ckpt = torch.load(personvit_path, map_location='cpu')

    # Extract student model weights (PersonViT uses teacher-student architecture)
    student_weights = ckpt['student']

    # Create Fast-Reid compatible state dict
    fastreid_state_dict = {}

    # Map PersonViT keys to Fast-Reid keys
    key_mapping = {
        # Backbone mapping
        'module.backbone.cls_token': 'backbone.cls_token',
        'module.backbone.pos_embed': 'backbone.pos_embed',
        'module.backbone.masked_embed': 'backbone.masked_embed',
        'module.backbone.patch_embed.proj.weight': 'backbone.patch_embed.proj.weight',
        'module.backbone.patch_embed.proj.bias': 'backbone.patch_embed.proj.bias',
        'module.backbone.norm.weight': 'backbone.norm.weight',
        'module.backbone.norm.bias': 'backbone.norm.bias',
    }

    # Add transformer blocks
    for i in range(12):  # ViT-Base has 12 blocks
        block_prefix = f'module.backbone.blocks.{i}'
        new_prefix = f'backbone.blocks.{i}'

        key_mapping.update({
            f'{block_prefix}.norm1.weight': f'{new_prefix}.norm1.weight',
            f'{block_prefix}.norm1.bias': f'{new_prefix}.norm1.bias',
            f'{block_prefix}.attn.qkv.weight': f'{new_prefix}.attn.qkv.weight',
            f'{block_prefix}.attn.qkv.bias': f'{new_prefix}.attn.qkv.bias',
            f'{block_prefix}.attn.proj.weight': f'{new_prefix}.attn.proj.weight',
            f'{block_prefix}.attn.proj.bias': f'{new_prefix}.attn.proj.bias',
            f'{block_prefix}.norm2.weight': f'{new_prefix}.norm2.weight',
            f'{block_prefix}.norm2.bias': f'{new_prefix}.norm2.bias',
            f'{block_prefix}.mlp.fc1.weight': f'{new_prefix}.mlp.fc1.weight',
            f'{block_prefix}.mlp.fc1.bias': f'{new_prefix}.mlp.fc1.bias',
            f'{block_prefix}.mlp.fc2.weight': f'{new_prefix}.mlp.fc2.weight',
            f'{block_prefix}.mlp.fc2.bias': f'{new_prefix}.mlp.fc2.bias',
        })

    # Map the weights
    mapped_count = 0
    for old_key, new_key in key_mapping.items():
        if old_key in student_weights:
            fastreid_state_dict[new_key] = student_weights[old_key]
            mapped_count += 1
            print(f"Mapped: {old_key} -> {new_key}")

    print(f"Successfully mapped {mapped_count} parameters")

    # Create Fast-Reid checkpoint structure
    fastreid_ckpt = {
        'model': fastreid_state_dict,
        'optimizer': {},
        'scheduler': {},
        'iteration': 0,
        'epoch': 0
    }

    # Save converted checkpoint
    torch.save(fastreid_ckpt, output_path)
    print(f"Converted checkpoint saved to: {output_path}")

    # Print checkpoint info
    print(f"\nCheckpoint info:")
    print(f"Total parameters: {len(fastreid_state_dict)}")
    print(f"Model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

    return output_path

def create_personvit_reid_config():
    """
    Create a ReID configuration file for PersonViT
    """
    config_content = """CUDNN_BENCHMARK: true
DATALOADER:
  NUM_INSTANCE: 16
  NUM_WORKERS: 8
  SAMPLER_TRAIN: NaiveIdentitySampler
  SET_WEIGHT: []
DATASETS:
  COMBINEALL: false
  NAMES:
  - NTUMTMC
  TESTS:
  - NTUMTMC
INPUT:
  AFFINE:
    ENABLED: false
  AUGMIX:
    ENABLED: false
    PROB: 0.0
  AUTOAUG:
    ENABLED: false
    PROB: 0.0
  CJ:
    BRIGHTNESS: 0.15
    CONTRAST: 0.15
    ENABLED: false
    HUE: 0.1
    PROB: 0.5
    SATURATION: 0.1
  CROP:
    ENABLED: false
    RATIO:
    - 0.75
    - 1.3333333333333333
    SCALE:
    - 0.16
    - 1
    SIZE:
    - 224
    - 224
  FLIP:
    ENABLED: true
    PROB: 0.5
  PADDING:
    ENABLED: true
    MODE: constant
    SIZE: 10
  REA:
    ENABLED: true
    PROB: 0.5
    VALUE:
    - 123.675
    - 116.28
    - 103.53
  RPT:
    ENABLED: false
    PROB: 0.5
  SIZE_TEST:
  - 256
  - 128
  SIZE_TRAIN:
  - 256
  - 128
KD:
  EMA:
    ENABLED: false
    MOMENTUM: 0.999
  MODEL_CONFIG: []
  MODEL_WEIGHTS: []
MODEL:
  BACKBONE:
    ATT_DROP_RATE: 0.0
    DEPTH: base
    DROP_PATH_RATIO: 0.1
    DROP_RATIO: 0.0
    FEAT_DIM: 768
    LAST_STRIDE: 1
    NAME: build_vit_backbone
    NORM: LN
    PRETRAIN: true
    PRETRAIN_PATH: ''
    SIE_COE: 3.0
    STRIDE_SIZE:
    - 16
    - 16
    WITH_IBN: false
    WITH_NL: false
    WITH_SE: false
  DEVICE: cuda
  FREEZE_LAYERS: []
  HEADS:
    CLS_LAYER: Linear
    EMBEDDING_DIM: 0
    MARGIN: 0.0
    NAME: EmbeddingHead
    NECK_FEAT: before
    NORM: BN
    NUM_CLASSES: 1185
    POOL_LAYER: Identity
    SCALE: 1
    WITH_BNNECK: true
  LOSSES:
    CE:
      EPSILON: 0.0
      SCALE: 1.0
    CIRCLE:
      MARGIN: 0.25
      SCALE: 64
    COT:
      SCALE: 1.0
    FOCAL:
      ALPHA: 1
      GAMMA: 2
      SCALE: 1.0
    NAME: ("CrossEntropyLoss", "TripletLoss",)
    TRI:
      HARD_MINING: true
      MARGIN: 0.3
      NORM_FEAT: false
      SCALE: 1.0
  META_ARCHITECTURE: Baseline
  PIXEL_MEAN: [123.675, 116.28, 103.53]
  PIXEL_STD: [58.395, 57.12, 57.375]
  WEIGHTS: ''
SOLVER:
  AMP:
    ENABLED: false
  BASE_LR: 0.00035
  BIAS_LR_FACTOR: 1
  CHECKPOINT_PERIOD: 30
  CLIP_GRADIENTS:
    CLIP_TYPE: norm
    CLIP_VALUE: 1.0
    ENABLED: true
    NORM_TYPE: 2.0
  DELAY_EPOCHS: -1
  ETA_MIN_LR: 3.5e-07
  EVAL_PERIOD: 5
  FP16_ENABLED: false
  FREEZE_ITERS: 0
  GAMMA: 0.1
  HEADS_LR_FACTOR: 1
  IMS_PER_BATCH: 64
  ITERS_PER_EPOCH: 200
  MAX_EPOCH: 120
  MOMENTUM: 0.9
  NESTEROV: false
  OPT: AdamW
  SCHED: CosineAnnealingLR
  STEPS: [40, 70]
  WARMUP_FACTOR: 0.01
  WARMUP_ITERS: 1000
  WARMUP_METHOD: linear
  WEIGHT_DECAY: 0.0005
  WEIGHT_DECAY_BIAS: 0.0005
TEST:
  AQE:
    ALPHA: 3.0
    ENABLED: false
    QE_TIME: 1
    QE_K: 5
  EVAL_PERIOD: 5
  IMS_PER_BATCH: 128
  PRECISE_BN:
    DATASET: Market1501
    ENABLED: false
    NUM_ITER: 300
  RERANK:
    ENABLED: false
    K1: 20
    K2: 6
    LAMBDA_VALUE: 0.3
OUTPUT_DIR: logs/ntumtmc/personvit
"""

    config_path = "reid_weight/personvit_config.yml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"Created PersonViT ReID config: {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description="Convert PersonViT checkpoint to Fast-Reid format")
    parser.add_argument("--input", default="pretrained/personvit_checkpoint0240.pth",
                       help="Path to PersonViT checkpoint")
    parser.add_argument("--output", default="reid_weight/personvit_fastreid.pth",
                       help="Path to save converted checkpoint")
    parser.add_argument("--create-config", action="store_true",
                       help="Create ReID configuration file")

    args = parser.parse_args()

    # Convert checkpoint
    if os.path.exists(args.input):
        convert_personvit_checkpoint(args.input, args.output)
    else:
        print(f"Error: Input checkpoint not found: {args.input}")
        return 1

    # Create config if requested
    if args.create_config:
        create_personvit_reid_config()

    print("\nConversion completed successfully!")
    print(f"To use PersonViT for ReID, run:")
    print(f"python tools/demo_multi_camera_track.py --reid_config reid_weight/personvit_config.yml --reid_model {args.output}")

    return 0

if __name__ == "__main__":
    sys.exit(main())