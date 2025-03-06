#!/bin/bash

# Run Silicon Layer Processing Script
# This script runs the optimized silicon layer processing pipeline

# Set script directory
SCRIPT_DIR=$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")

# Set environment variable to enable MPS (Apple Silicon) acceleration
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Check if a specific metric was requested
if [ "$1" != "" ]; then
    METRIC_ARG="--metrics $1"
    echo "Running silicon layer processing for metric: $1"
else
    METRIC_ARG="--metrics all"
    echo "Running silicon layer processing for all metrics"
fi

# Check if a sample size was provided
if [ "$SAMPLE_SIZE" != "" ]; then
    echo "Using sample size: $SAMPLE_SIZE"
    export SAMPLE_SIZE
fi

# Enable Apple Silicon optimizations if available
IS_ARM64=$(uname -m)
if [ "$IS_ARM64" = "arm64" ]; then
    echo "🍎 Apple Silicon detected - Using optimized MPS acceleration"
fi

# Run the script
echo "================================================================================"
echo "🚀 STARTING OPTIMIZED SILICON LAYER PROCESSING 🚀"
echo "================================================================================"
python $SCRIPT_DIR/scripts/optimize_silicon_processing.py $METRIC_ARG

# Check if the script was successful
if [ $? -eq 0 ]; then
    echo "✅ Silicon layer processing completed successfully!"
    exit 0
else
    echo "❌ Silicon layer processing failed with error code $?"
    exit 1
fi