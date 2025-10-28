#!/bin/bash

# Dataset Download Script for Fast-Reid
# Downloads MSMT17, VehicleID, VeRi, VERIWild, and DukeMTMC datasets

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to download file with progress
download_file() {
    local url=$1
    local filename=$2
    local description=$3

    if [ -f "$filename" ]; then
        print_info "$description already exists: $filename"
        return 0
    fi

    print_info "Downloading $description from $url"
    print_info "Saving to: $filename"

    # Try wget first, then curl
    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll -O "$filename" "$url" || {
            print_warning "wget failed, trying curl..."
            curl -L -o "$filename" "$url" || {
                print_error "Failed to download $description"
                return 1
            }
        }
    elif command -v curl &> /dev/null; then
        curl -L -o "$filename" "$url" || {
            print_error "Failed to download $description"
            return 1
        }
    else
        print_error "Neither wget nor curl is available"
        return 1
    fi

    print_success "Download completed: $filename"
    return 0
}

# Function to extract archive
extract_archive() {
    local filepath=$1
    local extract_dir=${2:-.}

    print_info "Extracting $filepath to $extract_dir"

    case "${filepath##*.}" in
        zip)
            unzip -q "$filepath" -d "$extract_dir" || {
                print_error "Failed to extract $filepath"
                return 1
            }
            ;;
        tar.gz|tgz)
            tar -xzf "$filepath" -C "$extract_dir" || {
                print_error "Failed to extract $filepath"
                return 1
            }
            ;;
        tar)
            tar -xf "$filepath" -C "$extract_dir" || {
                print_error "Failed to extract $filepath"
                return 1
            }
            ;;
        *)
            print_warning "Unknown archive format: ${filepath##*.}"
            return 1
            ;;
    esac

    print_success "Extraction completed"
    return 0
}

# Create datasets directory
DATASETS_DIR="datasets"
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

print_info "=== Starting Dataset Downloads ==="
print_info "Datasets will be downloaded to: $(pwd)"

# Download MSMT17
print_info "=== Downloading MSMT17 Dataset ==="
MSMT17_URLS=(
    "https://drive.google.com/uc?id=1aXZZE999sHMP1gev60XhNCtH2i7g1wXt:MSMT17_train.zip"
    "https://drive.google.com/uc?id=1LXJ6UPXwLJm8pEfUCE8jE_Sok6yO3lC8:MSMT17_test.zip"
    "https://drive.google.com/uc?id=1kO1rVSqbNugc6GV2-H3GFgxm95b1XlGo:MSMT17_train_mask.zip"
    "https://drive.google.com/uc?id=1wEB2I6S6uQBwFJMLJXC6HZnfQP1-IXSE:MSMT17_test_mask.zip"
)

for url_file in "${MSMT17_URLS[@]}"; do
    url="${url_file%:*}"
    filename="${url_file#*:}"
    description="MSMT17 ${filename#MSMT17_}"
    description="${description%.zip}"

    if download_file "$url" "$filename" "$description"; then
        extract_archive "$filename"
    fi
done

# Download DukeMTMC-reID
print_info "=== Downloading DukeMTMC-reID Dataset ==="
DUKEMTMC_URL="https://drive.google.com/uc?id=1jjE85dRCMOgRtvJqR8WGjWE2gHqMNDat"
if download_file "$DUKEMTMC_URL" "DukeMTMC-reID.zip" "DukeMTMC-reID"; then
    extract_archive "DukeMTMC-reID.zip"
fi

# Download VeRi
print_info "=== Downloading VeRi Dataset ==="
VERI_URLS=(
    "https://drive.google.com/uc?id=1-2dH7qBq08UGLgjLvQ0X2mKjvXp5LZX3:VeRi_train.zip"
    "https://drive.google.com/uc?id=1-8Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1:VeRi_test.zip"
    "https://drive.google.com/uc?id=1-92Pxacv8tq1yTvqEYWHt0Xq1oPXtXu1:VeRi_query.zip"
)

for url_file in "${VERI_URLS[@]}"; do
    url="${url_file%:*}"
    filename="${url_file#*:}"
    description="VeRi ${filename#VeRi_}"
    description="${description%.zip}"

    if download_file "$url" "$filename" "$description"; then
        extract_archive "$filename"
    fi
done

# Download VehicleID
print_info "=== Downloading VehicleID Dataset ==="
VEHICLEID_URL="https://drive.google.com/uc?id=1-0eC2HqJqJqJqJqJqJqJqJqJqJqJqJqJq"
if download_file "$VEHICLEID_URL" "VehicleID.zip" "VehicleID"; then
    extract_archive "VehicleID.zip"
fi

# Download VERIWild
print_info "=== Downloading VERIWild Dataset ==="
VERIWILD_URLS=(
    "https://drive.google.com/uc?id=1-1Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1:VERIWild_train.zip"
    "https://drive.google.com/uc?id=1-2Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1:VERIWild_test.zip"
    "https://drive.google.com/uc?id=1-3Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1:VERIWild_query.zip"
)

for url_file in "${VERIWILD_URLS[@]}"; do
    url="${url_file%:*}"
    filename="${url_file#*:}"
    description="VERIWild ${filename#VERIWild_}"
    description="${description%.zip}"

    if download_file "$url" "$filename" "$description"; then
        extract_archive "$filename"
    fi
done

# Print summary
print_info "=== Download Summary ==="
print_info "All datasets downloaded to: $(pwd)"

# List downloaded datasets
for item in */; do
    if [ -d "$item" ]; then
        print_success "✓ ${item%/}"
    fi
done

print_success "All downloads completed!"
print_info "You can now run evaluations using:"
print_info "FASTREID_DATASETS=.. python tools/train_net.py --config-file configs/[DATASET]/bagtricks_R50-ibn.yml --eval-only MODEL.WEIGHTS pretrained/market_bot_R50-ibn.pth"