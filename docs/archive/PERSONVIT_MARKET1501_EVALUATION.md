# PersonViT Market-1501 Evaluation Guide

This guide explains how to evaluate the PersonViT checkpoint on the Market-1501 dataset using the ByteTrack framework.

## Overview

PersonViT is a Vision Transformer-based model specifically designed for person re-identification. This guide provides two evaluation approaches:

1. **Custom Evaluation Script**: A comprehensive Python script that directly integrates with Fast-Reid
2. **Fast-Reid Integration**: Using the existing Fast-Reid framework with minimal modifications

## Prerequisites

- PersonViT checkpoint: `/root/ByteTrack/pretrained/personvit_checkpoint0240.pth`
- Market-1501 dataset: `/root/ByteTrack/Market-1501-v15.09.15/`
- Fast-Reid framework: `fast-reid-mct/`
- Python dependencies: torch, torchvision, opencv-python, numpy

## Dataset Structure

The Market-1501 dataset should have the following structure:
```
Market-1501-v15.09.15/
├── bounding_box_test/     # Gallery images (19732 images)
├── bounding_box_train/    # Training images (12936 images)
├── query/                 # Query images (3368 images)
├── gt_bbox/              # Ground truth bounding boxes
└── gt_query/             # Ground truth query annotations
```

## Evaluation Methods

### Method 1: Custom Evaluation Script (Recommended)

This method provides the most control and detailed output:

```bash
# Basic evaluation with automatic checkpoint conversion
python evaluate_personvit_market1501.py --convert-checkpoint

# Evaluation with custom paths
python evaluate_personvit_market1501.py \
    --checkpoint /root/ByteTrack/pretrained/personvit_checkpoint0240.pth \
    --dataset /root/ByteTrack/Market-1501-v15.09.15 \
    --convert-checkpoint \
    --device cuda

# CPU evaluation (if CUDA not available)
python evaluate_personvit_market1501.py --device cpu
```

**Features:**
- Automatic checkpoint conversion to Fast-Reid format
- Detailed progress reporting
- Comprehensive metrics calculation
- Results saved to `personvit_market1501_results.txt`

### Method 2: Fast-Reid Integration

This method uses the existing Fast-Reid framework:

```bash
# Simple evaluation using Fast-Reid
python evaluate_personvit_market1501_simple.py --convert-checkpoint

# With custom GPU device
python evaluate_personvit_market1501_simple.py --convert-checkpoint --device 0
```

**Features:**
- Uses standard Fast-Reid evaluation pipeline
- Compatible with existing Fast-Reid workflows
- Results saved to `personvit_market1501_fastreid_results.txt`

## Manual Steps (Alternative)

If you prefer to run the evaluation manually:

### Step 1: Convert PersonViT Checkpoint

```bash
python convert_personvit_to_fastreid.py \
    --input /root/ByteTrack/pretrained/personvit_checkpoint0240.pth \
    --output reid_weight/personvit_fastreid.pth \
    --create-config
```

### Step 2: Run Fast-Reid Evaluation

```bash
cd fast-reid-mct

python tools/train_net.py \
    --config-file ../configs/Market1501/bagtricks_vit.yml \
    --eval-only \
    MODEL.WEIGHTS ../reid_weight/personvit_fastreid.pth \
    DATASETS.ROOT_DIR ../Market-1501-v15.09.15
```

## Expected Results

The evaluation will output standard Market-1501 metrics:

- **Rank-1**: Percentage of queries where the correct match is found at rank 1
- **Rank-5**: Percentage of queries where the correct match is found within top 5
- **Rank-10**: Percentage of queries where the correct match is found within top 10
- **mAP**: Mean Average Precision
- **mINP**: Mean Inverse Negative Penalty
- **metric**: Combined metric score (mAP + Rank-1) / 2

## Configuration Details

The evaluation uses the following configuration:

- **Model**: ViT-Base with 768 embedding dimensions
- **Input Size**: 256×128 pixels
- **Feature Dimension**: 768
- **Evaluation Protocol**: Standard Market-1501 protocol
- **Distance Metric**: Cosine similarity

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   python evaluate_personvit_market1501.py --device cpu

   # Or reduce batch size in config
   # Edit fast-reid-mct/configs/Market1501/bagtricks_vit.yml
   # Set TEST.IMS_PER_BATCH to a smaller value (e.g., 32)
   ```

2. **Checkpoint Conversion Errors**
   ```bash
   # Check if checkpoint exists
   ls -la /root/ByteTrack/pretrained/personvit_checkpoint0240.pth

   # Manually convert checkpoint
   python convert_personvit_to_fastreid.py --input /path/to/checkpoint
   ```

3. **Dataset Path Issues**
   ```bash
   # Verify dataset structure
   ls -la /root/ByteTrack/Market-1501-v15.09.15/
   ls -la /root/ByteTrack/Market-1501-v15.09.15/bounding_box_test/ | head -10
   ```

4. **Fast-Reid Import Errors**
   ```bash
   # Install Fast-Reid dependencies
   cd fast-reid-mct
   pip install -r requirements.txt

   # Or install manually
   pip install torch torchvision opencv-python numpy
   ```

### Verification Commands

```bash
# Check file existence
ls -la pretrained/personvit_checkpoint0240.pth
ls -la Market-1501-v15.09.15/bounding_box_test/ | wc -l
ls -la Market-1501-v15.09.15/query/ | wc -l

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test Fast-Reid installation
cd fast-reid-mct
python -c "from fastreid.config import get_cfg; print('Fast-Reid OK')"
```

## Performance Optimization

### GPU Memory Optimization

```bash
# Reduce batch size for memory-constrained GPUs
# Edit fast-reid-mct/configs/Market1501/bagtricks_vit.yml:
TEST:
  IMS_PER_BATCH: 32  # Default is 128
```

### Speed Optimization

```bash
# Use mixed precision (if supported)
# Edit config to enable AMP:
SOLVER:
  AMP:
    ENABLED: true
```

## Output Files

After successful evaluation, you'll find:

1. **Console Output**: Real-time evaluation progress and final results
2. **Results File**: `personvit_market1501_results.txt` or `personvit_market1501_fastreid_results.txt`
3. **Log Files**: Fast-Reid logs in `fast-reid-mct/logs/` directory

## Comparison with Other Models

| Model | Feature Dim | Rank-1 | mAP | Model Size |
|-------|-------------|--------|-----|------------|
| PersonViT | 768 | TBD | TBD | 327MB |
| ResNet50 | 2048 | 94.4% | 86.1% | 251MB |
| ResNet50-IBN | 2048 | 94.9% | 87.6% | 287MB |
| ViT-Base | 768 | TBD | TBD | ~300MB |

*Note: PersonViT results will be populated after running the evaluation*

## References

- PersonViT Paper: [LakeAGI/PersonViT](https://huggingface.co/lakeAGI/PersonViT)
- Market-1501 Dataset: [Zheng et al. ICCV 2015](http://www.liangzheng.org/Project/project_reid.html)
- Fast-Reid Framework: [JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid)

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all file paths and dependencies
3. Check Fast-Reid documentation for additional help
4. Review the console output for specific error messages

## Quick Start Summary

```bash
# 1. Verify files exist
ls -la pretrained/personvit_checkpoint0240.pth
ls -la Market-1501-v15.09.15/bounding_box_test/ | head -5

# 2. Run evaluation (recommended method)
python evaluate_personvit_market1501.py --convert-checkpoint

# 3. Check results
cat personvit_market1501_results.txt
```

This will give you a complete evaluation of PersonViT on the Market-1501 dataset with all standard metrics.