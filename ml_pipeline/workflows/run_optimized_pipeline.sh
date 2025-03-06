#!/bin/bash

# Optimized News AI Pipeline Runner
# This script runs the entire News AI pipeline with optimized components

# Set script directory
SCRIPT_DIR=$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")

# Set default sample size if not provided
if [ -z "$SAMPLE_SIZE" ]; then
    SAMPLE_SIZE=5000
    export SAMPLE_SIZE
fi

echo "Running optimized pipeline with SAMPLE_SIZE=$SAMPLE_SIZE"

# Step 1: Run Silver Layer Processing
echo "================================================================================"
echo "🚀 STEP 1: SILVER LAYER PROCESSING 🚀"
echo "================================================================================"
$SCRIPT_DIR/scripts/optimize_silver_processing.py

# Check if silver layer processing succeeded
if [ $? -ne 0 ]; then
    echo "❌ Silver layer processing failed! Aborting pipeline."
    exit 1
fi

echo "✅ Silver layer processing completed successfully!"

# Step 2: Run Silicon Layer Processing
echo "================================================================================"
echo "🚀 STEP 2: SILICON LAYER PROCESSING 🚀"
echo "================================================================================"
$SCRIPT_DIR/workflows/run_silicon_processing.sh

# Check if silicon layer processing succeeded
if [ $? -ne 0 ]; then
    echo "❌ Silicon layer processing failed! Aborting pipeline."
    exit 1
fi

echo "✅ Silicon layer processing completed successfully!"

# Pipeline completed
echo "================================================================================"
echo "🎉 NEWS AI PIPELINE COMPLETED SUCCESSFULLY 🎉"
echo "================================================================================"
echo "The optimized pipeline processed data with the following components:"
echo "- Silver Layer: Feature engineering and data transformation"
echo "- Silicon Layer: Advanced metric models"
echo ""
echo "Data outputs:"
echo "- Silver data: $SCRIPT_DIR/data/silver/"
echo "- Silicon data: $SCRIPT_DIR/data/silicon/"
echo "- Deployed models: $SCRIPT_DIR/models/deployed/"

# Print execution time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Total execution time: $(($ELAPSED/60)) minutes and $(($ELAPSED%60)) seconds"

exit 0