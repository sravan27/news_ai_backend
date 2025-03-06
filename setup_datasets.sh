#!/bin/bash

# Setup script for downloading and extracting MIND Large Dataset

set -e

# Configuration
MIND_LARGE_URL="https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip"
MIND_LARGE_DEV_URL="https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip"
MIND_LARGE_TEST_URL="https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip"
DOWNLOAD_DIR="downloads"
TARGET_DIR="MINDLarge"

# Create directories
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$TARGET_DIR"

# Function to download and extract
download_and_extract() {
    local url=$1
    local filename=$(basename "$url")
    local download_path="$DOWNLOAD_DIR/$filename"
    
    echo "Downloading $filename..."
    if [ ! -f "$download_path" ]; then
        curl -L "$url" -o "$download_path"
    else
        echo "File already exists, skipping download."
    fi
    
    echo "Extracting $filename..."
    unzip -q -o "$download_path" -d "$TARGET_DIR"
}

# Download and extract all datasets
download_and_extract "$MIND_LARGE_URL"
download_and_extract "$MIND_LARGE_DEV_URL"
download_and_extract "$MIND_LARGE_TEST_URL"

echo "Creating .gitkeep files for empty directories..."
find "$TARGET_DIR" -type d -empty -exec touch {}/.gitkeep \;

echo "Dataset setup complete! Data is available in $TARGET_DIR directory"
echo "Run the pipeline with: python ml_pipeline/workflows/run_optimized_pipeline.sh"