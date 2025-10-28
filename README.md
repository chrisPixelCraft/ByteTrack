# ByteTrack: Multi-Object Tracking by Associating Every Detection Box

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bytetrack-multi-object-tracking-by-1/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=bytetrack-multi-object-tracking-by-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bytetrack-multi-object-tracking-by-1/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=bytetrack-multi-object-tracking-by-1)

> **ByteTrack** is a simple, fast and strong multi-object tracker that associates every detection box instead of only high-score ones.
>
> [**ByteTrack: Multi-Object Tracking by Associating Every Detection Box**](https://arxiv.org/abs/2110.06864)
> Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang
> *ECCV 2022*

## 🎯 Features

- **Simple & Effective**: Clean algorithm design with strong performance
- **Fast**: 30 FPS on V100 GPU with 80.3 MOTA on MOT17
- **Multi-Camera Tracking (MCT)**: Cross-camera person tracking with global ID management
- **PersonViT ReID**: Vision Transformer-based person re-identification for improved cross-camera association
- **Comprehensive Evaluation**: MOTA, mAP, Rank-1/5 metrics for single and multi-camera scenarios

---

## 📚 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Directory Structure](#-directory-structure)
- [Single-Camera Tracking](#-single-camera-tracking)
- [Multi-Camera Tracking (MCT)](#-multi-camera-tracking-mct)
- [PersonViT ReID Integration](#-personvit-reid-integration)
- [Evaluation](#-evaluation)
- [Datasets](#-datasets)
- [Model Zoo](#-model-zoo)
- [Training](#-training)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)

---

## 🚀 Quick Start

### Basic Single-Camera Tracking

```bash
# Download pretrained model
bash scripts/utils/get_ckpt.sh

# Run tracking on video
python3 tools/demo_track.py video \
  -f exps/example/mot/yolox_x_mix_det.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --path /path/to/video.mp4 \
  --fp16 --fuse --save_result
```

### Multi-Camera Tracking (Quick Test)

```bash
# Quick test with 3 cameras (100 frames)
python3 tools/demo_multi_camera_track.py \
  --quick_test \
  --cameras Cam1,Cam2,Cam3
```

---

## 🔧 Installation

### 1. Clone Repository

```bash
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
```

### 2. Install Dependencies

```bash
pip3 install -r requirements.txt
python3 setup.py develop
```

### 3. Install Additional Packages

```bash
pip3 install cython
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
```

### 4. Download Pretrained Models

```bash
# ByteTrack detection models
bash scripts/utils/get_ckpt.sh

# PersonViT ReID model (for MCT)
bash scripts/utils/get_reid_ckpt.sh
```

---

## 📁 Directory Structure

```
ByteTrack/
├── yolox/                          # Core tracking package
│   ├── tracker/                    # Tracking algorithms
│   │   ├── byte_tracker.py         # Single-camera ByteTrack
│   │   ├── multi_camera_tracker.py # Multi-camera tracker
│   │   └── reid_extractor.py       # ReID feature extraction
│   ├── models/                     # YOLOX detection models
│   ├── data/                       # Dataset loaders
│   └── evaluators/                 # Evaluation metrics
│
├── tools/                          # Training & evaluation tools
│   ├── demo_track.py               # Single-camera demo
│   ├── demo_multi_camera_track.py  # Multi-camera demo
│   ├── train.py                    # Model training
│   ├── mota.py                     # MOT17/20 evaluation
│   ├── ntu_mota.py                 # NTU-MTMC evaluation
│   └── mota_mct.py                 # MCT evaluation (MOTA + mAP + R1/R5)
│
├── scripts/                        # Utility scripts
│   ├── dataset/                    # Dataset management
│   │   ├── organize_ntu_dataset.py
│   │   ├── download_market_1501.py
│   │   └── download_datasets.sh
│   ├── reid/                       # ReID utilities
│   │   ├── convert_personvit_to_fastreid.py
│   │   ├── test_personvit_reid.py
│   │   └── evaluate_personvit_market1501.py
│   └── utils/                      # General utilities
│       ├── analyze_tracking_results.py
│       └── cleanup_tracking.py
│
├── exps/                           # Experiment configurations
│   └── example/mot/                # MOT training configs
│
├── fast-reid-mct/                  # Fast-Reid ReID framework
│   ├── configs/Market1501/         # ReID configs
│   └── fastreid/                   # ReID core package
│
├── pretrained/                     # Model checkpoints (not in git)
├── datasets/                       # Datasets (not in git)
├── YOLOX_outputs/                  # Training outputs (not in git)
└── MCT_outputs/                    # MCT results (not in git)
```

---

## 🎬 Single-Camera Tracking

### Standard Demo

```bash
python3 tools/demo_track.py video \
  -f exps/example/mot/yolox_m_mix_det.py \
  -c pretrained/bytetrack_m_mot17.pth.tar \
  --path /path/to/video.mp4 \
  --fp16 --fuse --save_result
```

### NTU-MTMC Single Camera

```bash
# Organize dataset first (run once)
python3 scripts/dataset/organize_ntu_dataset.py

# Run tracking for Cam1
PYTHONPATH=/root/ByteTrack:$PYTHONPATH python3 tools/demo_track.py video \
  -f exps/example/mot/yolox_m_mix_det.py \
  -c pretrained/bytetrack_m_mot17.pth.tar \
  --path NTU-MTMC/test/Cam1/Cam1.MP4 \
  --fp16 --fuse --save_result
```

### Evaluation

```bash
# MOT17/MOT20 evaluation
python3 tools/mota.py

# NTU-MTMC evaluation
python3 tools/ntu_mota.py
```

---

## 🎥 Multi-Camera Tracking (MCT)

ByteTrack supports advanced multi-camera tracking for cross-camera person re-identification and global track association.

### Quick Start

```bash
# Quick test (3 cameras, 100 frames)
python3 tools/demo_multi_camera_track.py \
  --quick_test \
  --cameras Cam1,Cam2,Cam3

# Recommended test (1000 frames, optimized)
python3 tools/demo_multi_camera_track.py \
  --cameras Cam1,Cam2,Cam3 \
  --max_frames 1000 \
  --fp16 --fuse --fast_mode \
  --save_results --progress_bar
```

### Full MCT Processing

```bash
# All cameras with PersonViT ReID
python3 tools/demo_multi_camera_track.py \
  --cameras all \
  --reid_config reid_weight/personvit_config.yml \
  --reid_model reid_weight/personvit_fastreid.pth \
  --fp16 --fuse \
  --save_video --save_results --progress_bar
```

### MCT Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--cameras` | Camera selection (comma-separated or 'all') | - |
| `--quick_test` | Process first 100 frames only | False |
| `--max_frames` | Maximum frames to process | All |
| `--fast_mode` | Use smaller resolution (3-5x faster) | False |
| `--fp16` | Half-precision inference | False |
| `--fuse` | Fuse model layers | False |
| `--save_video` | Save visualization videos | False |
| `--save_results` | Save MOT format results | False |
| `--progress_bar` | Show progress bar | False |
| `--reid_config` | Path to Fast-Reid config | - |
| `--reid_model` | Path to Fast-Reid model | - |

### MCT Output Structure

```
MCT_outputs/
├── tracking_results/
│   ├── Cam1.txt                        # MOT format results per camera
│   ├── Cam2.txt
│   └── ...
├── cross_camera_associations.txt       # Cross-camera associations
├── Cam1_mct_output.mp4                 # Visualization videos
└── ...
```

### MCT Architecture

- **MultiCameraTracker**: Coordinates multiple BYTETracker instances
- **GlobalTrackManager**: Manages global track IDs
- **CrossCameraAssociator**: Cross-camera association logic
- **ReidExtractor**: Fast-Reid integration for appearance features

### MCT Workflow

1. **Setup**: Initialize multi-camera tracker with camera configuration
2. **Detection**: Run YOLOX detection on each camera frame
3. **Single-Camera Tracking**: Apply ByteTrack independently per camera
4. **Cross-Camera Association**: Associate tracks across cameras every 30 frames
5. **Global ID Assignment**: Assign consistent global IDs
6. **Visualization**: Generate videos with cross-camera associations

---

## 🧠 PersonViT ReID Integration

PersonViT is a Vision Transformer-based model for person re-identification, improving cross-camera tracking accuracy.

### Quick Start

```bash
# Test PersonViT ReID
python3 scripts/reid/test_personvit_reid.py

# Demo with PersonViT (100 frames, 2 cameras)
python3 scripts/reid/demo_personvit_reid.py --mode quick --max_frames 100

# MCT with PersonViT
python3 tools/demo_multi_camera_track.py \
  --cameras Cam1,Cam2,Cam3 \
  --reid_config reid_weight/personvit_config.yml \
  --reid_model reid_weight/personvit_fastreid.pth \
  --save_results
```

### PersonViT Specifications

- **Architecture**: ViT-Base
- **Input Size**: 256×128 pixels
- **Feature Dimension**: 768
- **Model Size**: ~327 MB
- **Performance**: ~30-50ms per crop on GPU

### Evaluation on Market-1501

```bash
python3 scripts/reid/evaluate_personvit_market1501.py
```

**Results**:
- Rank-1: 41.89%
- mAP: 15.50%
- Metric: 28.70%

---

## 📊 Evaluation

### 1. Single-Camera MOTA Evaluation

```bash
# MOT17/MOT20
python3 tools/mota.py

# NTU-MTMC
python3 tools/ntu_mota.py \
  --results_folder MCT_outputs/tracking_results \
  --gt_folder NTU-MTMC/test \
  --output_file results.json
```

### 2. Multi-Camera Tracking (MCT) Evaluation

Comprehensive evaluation with MOTA, mAP, R1, R5 metrics:

```bash
# Basic usage
python3 tools/mota_mct.py

# Specific cameras with verbose output
python3 tools/mota_mct.py \
  --cameras Cam1,Cam2,Cam3 \
  --verbose \
  --verbose_logging \
  --output mct_results.json

# Custom paths
python3 tools/mota_mct.py \
  --gt_dir datasets/mot/train \
  --pred_dir MCT_outputs/tracking_results \
  --cameras all
```

### Evaluation Metrics

**Single-Camera Metrics**:
- **MOTA**: Multi-Object Tracking Accuracy
- **MOTP**: Multi-Object Tracking Precision
- **IDF1**: ID F1 Score
- **Recall/Precision**: Detection quality

**Cross-Camera Metrics**:
- **mAP**: Mean Average Precision for cross-camera matching
- **R1/R5**: Rank-1 and Rank-5 accuracy (ReID)
- **F1**: Harmonic mean of precision and recall

**Performance Thresholds**:
- MOTA > 50% and mAP > 30% = 🟢 GOOD
- MOTA > 20% and mAP > 10% = 🟡 MODERATE
- Below these = 🔴 NEEDS IMPROVEMENT

---

## 💾 Datasets

### Supported Datasets

- **MOT17/MOT20**: Standard MOT Challenge datasets
- **CrowdHuman**: Dense crowd detection
- **Cityperson**: Person detection in city scenes
- **ETHZ**: European multi-object tracking
- **NTU-MTMC**: Multi-target multi-camera tracking
- **Market-1501**: Person re-identification

### Data Preparation

#### MOT17/MOT20

Download from [MOTChallenge](https://motchallenge.net/) and convert to COCO format:

```bash
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_mot20_to_coco.py
```

#### NTU-MTMC

1. Download the NTU-MTMC dataset
2. Organize for evaluation:

```bash
python3 scripts/dataset/organize_ntu_dataset.py
```

Expected structure:
```
datasets/mot/train/
├── Cam1/gt/gt.txt
├── Cam2/gt/gt.txt
└── ...
```

#### Market-1501 (for ReID)

```bash
python3 scripts/dataset/download_market_1501.py
```

#### Other ReID Datasets

For MSMT17, DukeMTMC-reID, VeRi, VehicleID, and VERIWild, manual download is required due to Google Drive restrictions. See dataset documentation for links and instructions.

---

## 🏆 Model Zoo

### ByteTrack Detection Models

| Model | MOTA | IDF1 | FPS | Download |
|-------|------|------|-----|----------|
| bytetrack_x_mot17 | 90.0 | 83.3 | 29.6 | [google](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view) |
| bytetrack_l_mot17 | 88.7 | 80.7 | 43.7 | [google](https://drive.google.com/file/d/1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz/view) |
| bytetrack_m_mot17 | 87.0 | 80.1 | 54.1 | [google](https://drive.google.com/file/d/11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun/view) |
| bytetrack_s_mot17 | 79.2 | 74.3 | 64.5 | [google](https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view) |
| bytetrack_x_mot20 | 93.4 | 89.3 | 17.5 | [google](https://drive.google.com/file/d/1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U/view) |

### PersonViT ReID Model

- **PersonViT checkpoint**: Download with `bash scripts/utils/get_reid_ckpt.sh`
- **Converted Fast-Reid format**: Generated by `scripts/reid/convert_personvit_to_fastreid.py`

---

## 🎓 Training

### Train Ablation Model (MOT17 half train + CrowdHuman)

```bash
python3 tools/train.py \
  -f exps/example/mot/yolox_x_ablation.py \
  -d 8 -b 48 --fp16 -o \
  -c pretrained/yolox_x.pth
```

### Train MOT17 Test Model

```bash
python3 tools/train.py \
  -f exps/example/mot/yolox_x_mix_det.py \
  -d 8 -b 48 --fp16 -o \
  -c pretrained/yolox_x.pth
```

### Train MOT20 Test Model

For MOT20, clip bounding boxes inside the image (see code comments in data augmentation files).

```bash
python3 tools/train.py \
  -f exps/example/mot/yolox_x_mix_mot20_ch.py \
  -d 8 -b 48 --fp16 -o \
  -c pretrained/yolox_x.pth
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. MCT Slow Performance (< 1 FPS)

**Solutions**:
```bash
# Use fast mode (3-5x faster)
python3 tools/demo_multi_camera_track.py \
  --cameras Cam1,Cam2,Cam3 \
  --fast_mode --fp16 --fuse

# Limit frames for testing
--max_frames 1000

# Use fewer cameras
--cameras Cam1,Cam2,Cam3  # instead of --cameras all
```

#### 2. Fast-Reid "Training from scratch" Warning

**Issue**: Missing ReID model weights at `reid_weight/R18.pth`

**Solutions**:
- Download with: `bash scripts/utils/get_reid_ckpt.sh`
- Or disable ReID (faster): omit `--reid_config` and `--reid_model`
- Or use default: `--enable_reid` flag

#### 3. No Ground Truth Files Found

**Solution**:
```bash
# Re-organize dataset
python3 scripts/dataset/organize_ntu_dataset.py

# Verify structure
ls -la datasets/mot/train/Cam*/gt/gt.txt
```

#### 4. Cross-Camera Metrics are 0

**Possible Causes**:
- Multi-camera tracking didn't generate global IDs
- Tracks don't span multiple cameras
- Cross-camera association didn't run

**Solution**:
- Check MCT output: `MCT_outputs/tracking_results/cross_camera_associations.txt`
- Verify global IDs in camera tracking results
- Enable verbose logging: `--verbose_logging`

#### 5. CUDA Out of Memory

**Solutions**:
```bash
# Use smaller model
-f exps/example/mot/yolox_m_mix_det.py  # instead of yolox_x

# Enable fast mode
--fast_mode

# Process fewer cameras
--cameras Cam1,Cam2,Cam3

# Limit frames
--max_frames 1000
```

### Dataset-Specific Issues

#### NTU-MTMC

**Symlink Issues**:
```bash
# Verify symlinks
ls -la datasets/mot/train/Cam*/gt/

# Re-create if broken
python3 scripts/dataset/organize_ntu_dataset.py
```

#### Market-1501

**Download Failures**:
- Google Drive requires manual download due to permission restrictions
- Use `scripts/dataset/download_market_1501.py` or download manually
- See dataset documentation for alternative sources

---

## 📖 Citation

```bibtex
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

---

## 🙏 Acknowledgement

A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [FairMOT](https://github.com/ifzhang/FairMOT), [TransTrack](https://github.com/PeizeSun/TransTrack) and [JDE-Cpp](https://github.com/samylee/Towards-Realtime-MOT-Cpp). Many thanks for their wonderful works.

---

## 📝 Additional Documentation

- **CLAUDE.md**: Instructions for Claude Code AI assistant
- See `/scripts` directories for detailed utility documentation
- Check `tools/` for training and evaluation tool details

---

## 🔗 Links

- [Paper (arXiv)](https://arxiv.org/abs/2110.06864)
- [Original Repository](https://github.com/ifzhang/ByteTrack)
- [Google Colab Demo](https://colab.research.google.com/drive/1bDilg4cmXFa8HCKHbsZ_p16p0vrhLyu0)
- [Hugging Face Spaces](https://huggingface.co/spaces/akhaliq/bytetrack)
- [YouTube Tutorial](https://youtu.be/QCG8QMhga9k)
