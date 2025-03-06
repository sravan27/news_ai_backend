#!/bin/bash
# Install optimization dependencies for News AI ML Pipeline
# This script installs the packages needed for the optimized pipeline

echo "Installing optimization dependencies for News AI ML Pipeline..."

# Ensure pip is up to date
python -m pip install --upgrade pip

# Install core optimization libraries
echo "Installing DuckDB, PyArrow, Polars..."
python -m pip install duckdb pyarrow polars

# Install parallel processing libraries
echo "Installing Ray, tqdm..."
python -m pip install ray[default] tqdm

# Install memory optimization libraries
echo "Installing psutil, memory_profiler..."
python -m pip install psutil memory_profiler

# Install PyTorch MPS support (for Apple Silicon)
echo "Installing PyTorch with MPS support..."
python -m pip install torch torchvision

# Install HuggingFace optimizations
echo "Installing HuggingFace accelerate for faster inference..."
python -m pip install accelerate

# Make the optimization script executable
chmod +x /Users/sravansridhar/Documents/news_ai/ml_pipeline/scripts/optimize_silver_processing.py

echo "Installation complete! You can now run the optimized pipeline with:"
echo "python /Users/sravansridhar/Documents/news_ai/ml_pipeline/scripts/optimize_silver_processing.py"