#!/bin/bash

# Alternative Dataset Download Script for Fast-Reid
# Provides download instructions and alternative sources for datasets

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create datasets directory
DATASETS_DIR="datasets"
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

print_info "=== Dataset Download Instructions ==="
print_info "Datasets will be downloaded to: $(pwd)"
print_info ""

# MSMT17
print_info "=== MSMT17 Dataset ==="
print_info "Official Paper: https://arxiv.org/abs/1711.08565"
print_info "Download from: https://github.com/AICyberTeam/msmt17_dataset"
print_info "Alternative: https://drive.google.com/drive/folders/0B8-rUzbwVRk0c054eEozWG9COHM"
print_info "Expected structure:"
print_info "  MSMT17_V2/"
print_info "    mask_train_v2/"
print_info "    mask_test_v2/"
print_info ""

# DukeMTMC-reID
print_info "=== DukeMTMC-reID Dataset ==="
print_info "Official Paper: https://openaccess.thecvf.com/content_ICCV_2017/papers/Zheng_Unlabeled_Samples_Generated_ICCV_2017_paper.pdf"
print_info "Download from: https://github.com/layumi/DukeMTMC-reID_evaluation"
print_info "Alternative: https://drive.google.com/drive/folders/0B0VOCNYh8HeRdnBPa2ZWaVBYSVk"
print_info "Expected structure:"
print_info "  DukeMTMC-reID/"
print_info "    bounding_box_train/"
print_info "    bounding_box_test/"
print_info ""

# VeRi
print_info "=== VeRi Dataset ==="
print_info "Official Paper: https://ieeexplore.ieee.org/document/7780645"
print_info "Download from: https://github.com/VehicleReId/VeRidataset"
print_info "Alternative: https://drive.google.com/drive/folders/0B8-rUzbwVRk0c054eEozWG9COHM"
print_info "Expected structure:"
print_info "  VeRi/"
print_info "    train/"
print_info "    test/"
print_info "    query/"
print_info ""

# VehicleID
print_info "=== VehicleID Dataset ==="
print_info "Official Paper: https://ieeexplore.ieee.org/document/7780645"
print_info "Download from: https://github.com/VehicleReId/VeRidataset"
print_info "Alternative: https://drive.google.com/drive/folders/0B8-rUzbwVRk0c054eEozWG9COHM"
print_info "Expected structure:"
print_info "  VehicleID/"
print_info "    train/"
print_info "    test/"
print_info "    query/"
print_info ""

# VERIWild
print_info "=== VERIWild Dataset ==="
print_info "Official Paper: https://arxiv.org/abs/1909.00900"
print_info "Download from: https://github.com/PKU-IMRE/VERI-Wild"
print_info "Alternative: https://drive.google.com/drive/folders/1-1Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1"
print_info "Expected structure:"
print_info "  VERIWild/"
print_info "    train/"
print_info "    test/"
print_info "    query/"
print_info ""

print_info "=== Manual Download Instructions ==="
print_info "1. Visit the provided links above"
print_info "2. Download the dataset files"
print_info "3. Extract them to the appropriate directories shown above"
print_info "4. Ensure the directory structure matches the expected format"
print_info ""

print_info "=== Quick Download Commands (if available) ==="

# Try to download MSMT17 using alternative method
print_info "Attempting to download MSMT17..."
if command -v wget &> /dev/null; then
    # Try alternative MSMT17 download
    MSMT17_ALT_URL="https://github.com/AICyberTeam/msmt17_dataset/archive/refs/heads/master.zip"
    if wget --progress=bar:force:noscroll -O "msmt17_github.zip" "$MSMT17_ALT_URL"; then
        print_success "Downloaded MSMT17 from GitHub"
        unzip -q "msmt17_github.zip"
        print_info "Extracted MSMT17 dataset"
    else
        print_warning "Failed to download MSMT17 automatically"
    fi
fi

print_info ""
print_info "=== After Downloading ==="
print_info "You can run evaluations using:"
print_info "cd fast-reid-mct"
print_info "FASTREID_DATASETS=.. python tools/train_net.py --config-file configs/[DATASET]/bagtricks_R50-ibn.yml --eval-only MODEL.WEIGHTS pretrained/market_bot_R50-ibn.pth"
print_info ""
print_info "Example commands:"
print_info "  # MSMT17"
print_info "  FASTREID_DATASETS=.. python tools/train_net.py --config-file configs/MSMT17/bagtricks_R50-ibn.yml --eval-only MODEL.WEIGHTS pretrained/market_bot_R50-ibn.pth"
print_info ""
print_info "  # DukeMTMC"
print_info "  FASTREID_DATASETS=.. python tools/train_net.py --config-file configs/DukeMTMC/bagtricks_R50-ibn.yml --eval-only MODEL.WEIGHTS pretrained/market_bot_R50-ibn.pth"
print_info ""
print_info "  # VeRi"
print_info "  FASTREID_DATASETS=.. python tools/train_net.py --config-file configs/VeRi/bagtricks_R50-ibn.yml --eval-only MODEL.WEIGHTS pretrained/market_bot_R50-ibn.pth"
print_info ""
print_info "  # VehicleID"
print_info "  FASTREID_DATASETS=.. python tools/train_net.py --config-file configs/VehicleID/bagtricks_R50-ibn.yml --eval-only MODEL.WEIGHTS pretrained/market_bot_R50-ibn.pth"
print_info ""
print_info "  # VERIWild"
print_info "  FASTREID_DATASETS=.. python tools/train_net.py --config-file configs/VERIWild/bagtricks_R50-ibn.yml --eval-only MODEL.WEIGHTS pretrained/market_bot_R50-ibn.pth"