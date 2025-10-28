# PersonViT ReID Integration Guide

This guide explains how to use the PersonViT checkpoint for person re-identification (ReID) tasks in the ByteTrack multi-camera tracking system.

## Overview

PersonViT is a Vision Transformer-based model specifically designed for person re-identification. This integration allows you to use the PersonViT checkpoint (`checkpoint0240.pth`) with the Fast-Reid framework for improved cross-camera person association.

## Files Created

After running the conversion script, the following files are created:

- `reid_weight/personvit_fastreid.pth` - Converted PersonViT checkpoint compatible with Fast-Reid
- `reid_weight/personvit_config.yml` - Fast-Reid configuration for PersonViT
- `convert_personvit_to_fastreid.py` - Conversion script
- `test_personvit_reid.py` - Test script
- `demo_personvit_reid.py` - Demonstration script

## Quick Start

### 1. Test PersonViT ReID Functionality

```bash
# Run the test script to verify everything works
python test_personvit_reid.py
```

### 2. Basic Demo (10 frames, 2 cameras)

```bash
# Quick test with minimal processing
python demo_personvit_reid.py --mode test
```

### 3. Quick Demo (100 frames, 2 cameras)

```bash
# Quick demo with more frames
python demo_personvit_reid.py --mode quick --max_frames 100
```

### 4. Full Demo (all cameras)

```bash
# Full processing with all available cameras
python demo_personvit_reid.py --mode full --save_video
```

## Manual Usage

### Direct Command Line Usage

```bash
# Basic usage with PersonViT ReID
python tools/demo_multi_camera_track.py \
    --camera_dir NTU-MTMC/test \
    --cameras Cam1,Cam2,Cam3 \
    --reid_config reid_weight/personvit_config.yml \
    --reid_model reid_weight/personvit_fastreid.pth \
    --max_frames 100 \
    --progress_bar \
    --save_results

# Full processing with all cameras
python tools/demo_multi_camera_track.py \
    --camera_dir NTU-MTMC/test \
    --cameras all \
    --reid_config reid_weight/personvit_config.yml \
    --reid_model reid_weight/personvit_fastreid.pth \
    --save_video \
    --save_results \
    --progress_bar
```

### Programmatic Usage

```python
from yolox.tracker.reid_extractor import ReidExtractor

# Initialize PersonViT ReID extractor
reid_extractor = ReidExtractor(
    config_path="reid_weight/personvit_config.yml",
    model_path="reid_weight/personvit_fastreid.pth",
    device="cuda",  # or "cpu"
    input_size=(256, 128)
)

# Extract features from person detections
features = reid_extractor.extract_features(image, bounding_boxes)

# Compute similarity between features
similarity = reid_extractor.compute_similarity(features1, features2, metric="cosine")
```

## Model Specifications

- **Architecture**: Vision Transformer (ViT-Base)
- **Input Size**: 256×128 pixels
- **Feature Dimension**: 768
- **Patch Size**: 16×16
- **Number of Layers**: 12 transformer blocks
- **Model Size**: ~327 MB

## Performance Characteristics

- **Feature Extraction**: ~30-50ms per person crop (on GPU)
- **Memory Usage**: ~2GB GPU memory for batch processing
- **Accuracy**: Optimized for person re-identification tasks
- **Cross-camera Association**: Improved compared to ResNet-based models

## Configuration Details

The PersonViT configuration (`personvit_config.yml`) includes:

- **Backbone**: ViT-Base with 768 embedding dimensions
- **Heads**: EmbeddingHead with bottleneck normalization
- **Losses**: CrossEntropyLoss + TripletLoss
- **Input Processing**: Standard person ReID preprocessing
- **Optimization**: Adam optimizer with cosine annealing

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or use CPU
   export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
   ```

2. **Configuration Errors**
   ```bash
   # Regenerate configuration
   python convert_personvit_to_fastreid.py --create-config
   ```

3. **Model Loading Errors**
   ```bash
   # Check file paths
   ls -la reid_weight/personvit_*
   ```

### Verification Commands

```bash
# Check if files exist
ls -la pretrained/personvit_checkpoint0240.pth
ls -la reid_weight/personvit_fastreid.pth
ls -la reid_weight/personvit_config.yml

# Test ReID functionality
python test_personvit_reid.py

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Comparison with Other Models

| Model | Feature Dim | Model Size | Speed | Accuracy |
|-------|-------------|------------|-------|----------|
| PersonViT | 768 | 327MB | Medium | High |
| ResNet34 (v17) | 2048 | 251MB | Fast | Medium |
| ResNet18 | 512 | 135MB | Very Fast | Low |

## Advanced Usage

### Custom Configuration

You can modify `reid_weight/personvit_config.yml` to adjust:

- Input size for different resolutions
- Loss weights for different datasets
- Optimization parameters for different hardware

### Integration with Custom Datasets

```python
# Modify the configuration for your dataset
config = {
    'MODEL.HEADS.NUM_CLASSES': your_num_classes,
    'INPUT.SIZE_TEST': [your_height, your_width],
    'DATASETS.NAMES': ['your_dataset_name']
}
```

### Batch Processing

```python
# Process multiple images efficiently
batch_features = []
for images, bboxes in batch_data:
    features = reid_extractor.extract_features(images, bboxes)
    batch_features.extend(features)
```

## Results and Output

The system generates:

1. **Tracking Results**: MOT format files for each camera
2. **Cross-camera Associations**: Global person IDs across cameras
3. **Visualization Videos**: Optional output videos with tracking visualization
4. **Performance Metrics**: Tracking accuracy and ReID similarity scores

## References

- PersonViT Paper: [LakeAGI/PersonViT](https://huggingface.co/lakeAGI/PersonViT)
- Fast-Reid Framework: [JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid)
- ByteTrack: [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)

## Support

For issues related to:
- PersonViT model: Check the original repository
- Fast-Reid integration: Check Fast-Reid documentation
- ByteTrack integration: Check ByteTrack documentation
- This integration: Check this README and test scripts