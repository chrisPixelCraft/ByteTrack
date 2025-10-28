# NTU-MTMC Tracking Workflow Guide

## Overview
This guide shows how to run ByteTrack inference on NTU-MTMC dataset and evaluate using MOTA metrics.

## Step 1: Dataset Organization (Already Complete)
Your dataset is now organized as:
```
datasets/mot/train/
├── Cam1/gt/gt.txt
├── Cam2/gt/gt.txt
├── ...
└── Cam11/gt/gt.txt
```

## Step 2: Run Tracking Inference
For each camera, you need to run ByteTrack inference. Here's the **working command** for Cam1:

### ✅ **Recommended: Using video files with demo_track.py**
```bash
# Activate your conda environment
conda activate side_proj

# Run tracking with the working command
PYTHONPATH=/root/ByteTrack:$PYTHONPATH python3 tools/demo_track.py video \
  -f exps/example/mot/yolox_m_mix_det.py \
  -c pretrained/bytetrack_m_mot17.pth.tar \
  --path NTU-MTMC/test/Cam1/Cam1.MP4 \
  --fp16 --fuse --save_result
```

### For other cameras:
```bash
# Replace Cam1 with Cam2, Cam3, etc.
PYTHONPATH=/root/ByteTrack:$PYTHONPATH python3 tools/demo_track.py video \
  -f exps/example/mot/yolox_m_mix_det.py \
  -c pretrained/bytetrack_m_mot17.pth.tar \
  --path NTU-MTMC/test/Cam2/Cam2.MP4 \
  --fp16 --fuse --save_result

# Available cameras: Cam1, Cam2, Cam3, Cam4, Cam5, Cam6, Cam7, Cam8, Cam9, Cam10, Cam11
```

### Alternative: Using image sequence (if available)
```bash
python3 tools/track.py \
  -f exps/example/mot/yolox_x_ablation.py \
  -c pretrained/bytetrack_ablation.pth.tar \
  -b 1 -d 1 --fp16 --fuse \
  --path NTU-MTMC/test/Cam1/img1 \
  --output_dir YOLOX_outputs/yolox_x_ablation/track_results/Cam1.txt
```

## Step 3: Organize Tracking Results
After running inference, use the helper script to organize your tracking results:

### ✅ **Automated Organization (Recommended)**
```bash
# Check tracking status
python3 organize_ntu_results.py status

# Organize all tracking results for MOTA evaluation
python3 organize_ntu_results.py
```

### Manual Organization (if needed)
```bash
# Find the latest tracking results
ls -la YOLOX_outputs/yolox_m_mix_det/track_vis/

# Copy results to the expected location
mkdir -p YOLOX_outputs/yolox_m_mix_det/track_results/
cp YOLOX_outputs/yolox_m_mix_det/track_vis/2025_07_09_08_42_03/2025_07_09_08_42_03.txt YOLOX_outputs/yolox_m_mix_det/track_results/Cam1.txt
```

Your tracking results should be organized as:
```
YOLOX_outputs/yolox_m_mix_det/track_results/
├── Cam1.txt
├── Cam2.txt
├── ...
└── Cam11.txt
```

## Step 4: Run MOTA Evaluation
```bash
python3 tools/ntu_mota.py
```

## Expected Output Format
### Ground Truth Format (MOT15-2D):
```
frame_id,track_id,x,y,w,h,conf,class,visibility
153,568,649,804,59,144,1,-1,-1,-1
```

### Tracking Results Format (MOT15-2D):
```
frame_id,track_id,x,y,w,h,conf,class,visibility
1,1,100,200,50,100,0.9,-1,-1,-1
```

## Troubleshooting

### If tracking results are not in the right location:
Move or copy your tracking results to the expected location:
```bash
mkdir -p YOLOX_outputs/yolox_x_ablation/track_results/
cp your_tracking_results/*.txt YOLOX_outputs/yolox_x_ablation/track_results/
```

### If ground truth files are missing:
Re-run the dataset organization script:
```bash
python3 organize_ntu_dataset.py
```

### If you need to change the results folder:
Edit the `results_folder` variable in `tools/ntu_mota.py`:
```python
results_folder = 'path/to/your/tracking/results'
```

## Directory Structure Summary
```
ByteTrack/
├── datasets/mot/train/
│   ├── Cam1/gt/gt.txt -> /root/ByteTrack/NTU-MTMC/test/Cam1/gt/gt.txt
│   ├── Cam2/gt/gt.txt -> /root/ByteTrack/NTU-MTMC/test/Cam2/gt/gt.txt
│   └── ...
├── YOLOX_outputs/yolox_x_ablation/track_results/
│   ├── Cam1.txt  # Your tracking results
│   ├── Cam2.txt  # Your tracking results
│   └── ...
└── NTU-MTMC/test/
    ├── Cam1/
    │   ├── gt/gt.txt  # Original ground truth
    │   └── Cam1.MP4   # Video file
    └── ...
```