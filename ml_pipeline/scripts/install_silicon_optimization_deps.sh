#!/bin/bash

# Install dependencies for optimized silicon layer processing

echo "Installing dependencies for optimized silicon layer processing..."

# Basic data science packages
pip install numpy pandas pyarrow tqdm

# Machine learning packages
pip install scikit-learn xgboost lightgbm catboost

# PyTorch with MPS support for Apple Silicon
pip install torch torchvision

# Dask for distributed computing
pip install "dask[complete]" distributed

# Optional: Install Rust package for faster data processing
# This requires Rust to be installed on the system
echo "Would you like to install Rust for faster data processing? (y/n)"
read -r install_rust
if [[ "$install_rust" =~ ^[Yy]$ ]]; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source "$HOME/.cargo/env"
    pip install polars
    echo "Rust installed and polars library added for faster data processing."
else
    echo "Skipping Rust installation."
fi

echo "Dependencies installed successfully!"