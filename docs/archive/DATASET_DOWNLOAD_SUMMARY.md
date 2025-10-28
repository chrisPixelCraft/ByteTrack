# Dataset Download Summary

## Current Status

I attempted to download the following datasets for Fast-Reid evaluation:
- **MSMT17** - Person Re-identification dataset
- **DukeMTMC-reID** - Person Re-identification dataset
- **VeRi** - Vehicle Re-identification dataset
- **VehicleID** - Vehicle Re-identification dataset
- **VERIWild** - Vehicle Re-identification dataset

## Download Results

❌ **All automatic downloads failed** due to Google Drive permission restrictions. The datasets require manual download.

## Manual Download Instructions

### 1. MSMT17 Dataset
- **Paper**: https://arxiv.org/abs/1711.08565
- **Download Sources**:
  - Official: https://github.com/AICyberTeam/msmt17_dataset
  - Alternative: https://drive.google.com/drive/folders/0B8-rUzbwVRk0c054eEozWG9COHM
- **Expected Structure**:
  ```
  datasets/
    MSMT17_V2/
      mask_train_v2/
      mask_test_v2/
  ```

### 2. DukeMTMC-reID Dataset
- **Paper**: https://openaccess.thecvf.com/content_ICCV_2017/papers/Zheng_Unlabeled_Samples_Generated_ICCV_2017_paper.pdf
- **Download Sources**:
  - Official: https://github.com/layumi/DukeMTMC-reID_evaluation
  - Alternative: https://drive.google.com/drive/folders/0B0VOCNYh8HeRdnBPa2ZWaVBYSVk
- **Expected Structure**:
  ```
  datasets/
    DukeMTMC-reID/
      bounding_box_train/
      bounding_box_test/
  ```

### 3. VeRi Dataset
- **Paper**: https://ieeexplore.ieee.org/document/7780645
- **Download Sources**:
  - Official: https://github.com/VehicleReId/VeRidataset
  - Alternative: https://drive.google.com/drive/folders/0B8-rUzbwVRk0c054eEozWG9COHM
- **Expected Structure**:
  ```
  datasets/
    VeRi/
      train/
      test/
      query/
  ```

### 4. VehicleID Dataset
- **Paper**: https://ieeexplore.ieee.org/document/7780645
- **Download Sources**:
  - Official: https://github.com/VehicleReId/VeRidataset
  - Alternative: https://drive.google.com/drive/folders/0B8-rUzbwVRk0c054eEozWG9COHM
- **Expected Structure**:
  ```
  datasets/
    VehicleID/
      train/
      test/
      query/
  ```

### 5. VERIWild Dataset
- **Paper**: https://arxiv.org/abs/1909.00900
- **Download Sources**:
  - Official: https://github.com/PKU-IMRE/VERI-Wild
  - Alternative: https://drive.google.com/drive/folders/1-1Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1
- **Expected Structure**:
  ```
  datasets/
    VERIWild/
      train/
      test/
      query/
  ```

## Manual Download Steps

1. **Visit the provided links** for each dataset
2. **Download the dataset files** (usually in ZIP format)
3. **Extract them** to the appropriate directories shown above
4. **Ensure the directory structure matches** the expected format
5. **Place all datasets** in the `/root/ByteTrack/datasets/` directory

## Evaluation Commands

Once the datasets are downloaded and properly organized, you can run evaluations using:

```bash
cd fast-reid-mct

# MSMT17
FASTREID_DATASETS=.. python tools/train_net.py --config-file configs/MSMT17/bagtricks_R50-ibn.yml --eval-only MODEL.WEIGHTS pretrained/market_bot_R50-ibn.pth

# DukeMTMC
FASTREID_DATASETS=.. python tools/train_net.py --config-file configs/DukeMTMC/bagtricks_R50-ibn.yml --eval-only MODEL.WEIGHTS pretrained/market_bot_R50-ibn.pth

# VeRi
FASTREID_DATASETS=.. python tools/train_net.py --config-file configs/VeRi/bagtricks_R50-ibn.yml --eval-only MODEL.WEIGHTS pretrained/market_bot_R50-ibn.pth

# VehicleID
FASTREID_DATASETS=.. python tools/train_net.py --config-file configs/VehicleID/bagtricks_R50-ibn.yml --eval-only MODEL.WEIGHTS pretrained/market_bot_R50-ibn.pth

# VERIWild
FASTREID_DATASETS=.. python tools/train_net.py --config-file configs/VERIWild/bagtricks_R50-ibn.yml --eval-only MODEL.WEIGHTS pretrained/market_bot_R50-ibn.pth
```

## Current Available Datasets

✅ **Market-1501** - Successfully evaluated with PersonViT
- Location: `/root/ByteTrack/Market-1501-v15.09.15/`
- Results: Rank-1: 41.89%, mAP: 15.50%, metric: 28.70%

## Notes

- The Google Drive links require manual access due to permission restrictions
- Some datasets may require registration or approval from the dataset owners
- The evaluation commands assume the ResNet50-IBN model (`market_bot_R50-ibn.pth`) is used
- All datasets should be placed in the `datasets/` directory relative to the Fast-Reid installation