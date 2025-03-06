#!/usr/bin/env python
"""
Setup script for the News AI application.

This script performs the following actions:
1. Creates the necessary directories
2. Downloads pretrained models
3. Loads the MIND dataset
4. Sets up the environment

Usage:
    python setup_app.py
"""
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from news_ai_app.utils.pretrained_model_downloader import download_pretrained_models


def setup_directories():
    """Create necessary directories."""
    # Create models directory
    os.makedirs("models/pretrained", exist_ok=True)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)


def setup_logging():
    """Set up logging configuration."""
    # Make sure logs directory exists first
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/setup.log"),
        ],
    )


def check_mind_dataset():
    """Check if the MIND dataset is available."""
    # Check the path from .env
    mind_path = os.getenv("MIND_DATASET_PATH", "./MINDLarge")
    mind_path = Path(mind_path)
    
    # Check if the directory exists
    if not mind_path.exists():
        logging.warning(f"MIND dataset directory {mind_path} not found.")
        return False
        
    # Check if the necessary files exist
    train_path = mind_path / "MINDlarge_train"
    dev_path = mind_path / "MINDlarge_dev"
    test_path = mind_path / "MINDlarge_test"
    
    if not (train_path.exists() and dev_path.exists() and test_path.exists()):
        logging.warning("MIND dataset splits not found.")
        return False
        
    # Check if the necessary files exist in each split
    for split_path in [train_path, dev_path, test_path]:
        if not all((split_path / f).exists() for f in ["behaviors.tsv", "news.tsv", "entity_embedding.vec", "relation_embedding.vec"]):
            logging.warning(f"MIND dataset files missing in {split_path}.")
            return False
    
    # All checks passed
    logging.info("MIND dataset found and appears to be valid.")
    return True


def main():
    """Run the setup script."""
    parser = argparse.ArgumentParser(description="Setup the News AI application.")
    parser.add_argument("--skip-downloads", action="store_true", help="Skip downloading pretrained models.")
    parser.add_argument("--force", action="store_true", help="Force download of pretrained models even if they already exist.")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logging.info("Starting News AI application setup...")
    
    # Create directories
    logging.info("Creating directories...")
    setup_directories()
    
    # Check MIND dataset
    logging.info("Checking MIND dataset...")
    if not check_mind_dataset():
        logging.warning("MIND dataset check failed. Please ensure the dataset is available.")
        logging.info("Setup will continue, but some features may not work without the dataset.")
    
    # Download pretrained models
    if not args.skip_downloads:
        logging.info("Downloading pretrained models...")
        download_pretrained_models(force=args.force)
    else:
        logging.info("Skipping pretrained model downloads.")
    
    logging.info("Setup completed successfully!")
    

if __name__ == "__main__":
    main()