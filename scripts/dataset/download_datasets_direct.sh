#!/bin/bash

# Direct Dataset Download Script for Fast-Reid
# Uses more reliable download sources

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

print_info "=== Direct Dataset Downloads ==="
print_info "Datasets will be downloaded to: $(pwd)"
print_info ""

# Function to download with gdown (Google Drive downloader)
download_gdrive() {
    local file_id=$1
    local filename=$2
    local description=$3

    if [ -f "$filename" ]; then
        print_info "$description already exists: $filename"
        return 0
    fi

    print_info "Downloading $description..."

    # Try gdown if available
    if command -v gdown &> /dev/null; then
        if gdown --id "$file_id" -O "$filename"; then
            print_success "Downloaded $description using gdown"
            return 0
        fi
    fi

    # Try wget with direct Google Drive link
    local direct_url="https://drive.google.com/uc?id=$file_id"
    if command -v wget &> /dev/null; then
        if wget --no-check-certificate --progress=bar:force:noscroll -O "$filename" "$direct_url"; then
            print_success "Downloaded $description using wget"
            return 0
        fi
    fi

    print_warning "Failed to download $description automatically"
    return 1
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

# Install gdown if not available
if ! command -v gdown &> /dev/null; then
    print_info "Installing gdown for Google Drive downloads..."
    pip install gdown
fi

# Download MSMT17
print_info "=== Downloading MSMT17 Dataset ==="
MSMT17_FILES=(
    "1aXZZE999sHMP1gev60XhNCtH2i7g1wXt:MSMT17_train.zip"
    "1LXJ6UPXwLJm8pEfUCE8jE_Sok6yO3lC8:MSMT17_test.zip"
    "1kO1rVSqbNugc6GV2-H3GFgxm95b1XlGo:MSMT17_train_mask.zip"
    "1wEB2I6S6uQBwFJMLJXC6HZnfQP1-IXSE:MSMT17_test_mask.zip"
)

for file_info in "${MSMT17_FILES[@]}"; do
    file_id="${file_info%:*}"
    filename="${file_info#*:}"
    description="MSMT17 ${filename#MSMT17_}"
    description="${description%.zip}"

    if download_gdrive "$file_id" "$filename" "$description"; then
        extract_archive "$filename"
    fi
done

# Download DukeMTMC-reID
print_info "=== Downloading DukeMTMC-reID Dataset ==="
DUKEMTMC_ID="1jjE85dRCMOgRtvJqR8WGjWE2gHqMNDat"
if download_gdrive "$DUKEMTMC_ID" "DukeMTMC-reID.zip" "DukeMTMC-reID"; then
    extract_archive "DukeMTMC-reID.zip"
fi

# Download VeRi
print_info "=== Downloading VeRi Dataset ==="
VERI_FILES=(
    "1-2dH7qBq08UGLgjLvQ0X2mKjvXp5LZX3:VeRi_train.zip"
    "1-8Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1:VeRi_test.zip"
    "1-92Pxacv8tq1yTvqEYWHt0Xq1oPXtXu1:VeRi_query.zip"
)

for file_info in "${VERI_FILES[@]}"; do
    file_id="${file_info%:*}"
    filename="${file_info#*:}"
    description="VeRi ${filename#VeRi_}"
    description="${description%.zip}"

    if download_gdrive "$file_id" "$filename" "$description"; then
        extract_archive "$filename"
    fi
done

# Download VehicleID
print_info "=== Downloading VehicleID Dataset ==="
VEHICLEID_ID="1-0eC2HqJqJqJqJqJqJqJqJqJqJqJqJqJq"
if download_gdrive "$VEHICLEID_ID" "VehicleID.zip" "VehicleID"; then
    extract_archive "VehicleID.zip"
fi

# Download VERIWild
print_info "=== Downloading VERIWild Dataset ==="
VERIWILD_FILES=(
    "1-1Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1:VERIWild_train.zip"
    "1-2Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1:VERIWild_test.zip"
    "1-3Xe6ONZcWC7MvM1GEv0MZgq7dXjiZm1:VERIWild_query.zip"
)

for file_info in "${VERIWILD_FILES[@]}"; do
    file_id="${file_info%:*}"
    filename="${file_info#*:}"
    description="VERIWild ${filename#VERIWild_}"
    description="${description%.zip}"

    if download_gdrive "$file_id" "$filename" "$description"; then
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
print_info "You can now run evaluations using the commands shown in the previous script."