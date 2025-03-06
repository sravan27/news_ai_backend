#!/usr/bin/env python
"""
Train advanced machine learning models for news metrics.

This script runs the advanced model training pipeline for all metrics:
1. Political Influence
2. Rhetoric Intensity
3. Information Depth
4. Sentiment

Usage:
    python train_advanced_models.py [--metric METRIC_NAME]

Options:
    --metric: Specific metric to train (political_influence, rhetoric_intensity, 
              information_depth, sentiment, or all)
"""

import os
import argparse
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/advanced_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Metrics to train
METRICS = ["political_influence", "rhetoric_intensity", "information_depth", "sentiment"]

def ensure_directories():
    """Ensure necessary directories exist."""
    # Create log directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create ml_pipeline/data/silicon directory if it doesn't exist
    os.makedirs("ml_pipeline/data/silicon", exist_ok=True)
    
    # Create ml_pipeline/models/deployed directory if it doesn't exist
    os.makedirs("ml_pipeline/models/deployed", exist_ok=True)
    
    # Create directory for each metric
    for metric in METRICS:
        os.makedirs(f"ml_pipeline/data/silicon/{metric}", exist_ok=True)
        os.makedirs(f"ml_pipeline/models/deployed/{metric}", exist_ok=True)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "numpy", "pandas", "sklearn", "torch", "lightgbm", 
        "xgboost", "catboost", "pyarrow"
    ]
    
    missing_packages = []
    package_versions = {}
    
    for package in required_packages:
        try:
            if package == "sklearn":
                # Special handling for scikit-learn
                import sklearn
                package_versions[package] = sklearn.__version__
            else:
                mod = __import__(package)
                version = getattr(mod, "__version__", "unknown")
                package_versions[package] = version
        except ImportError as e:
            logger.error(f"Error importing {package}: {e}")
            missing_packages.append(package)
    
    # Print found packages and their versions
    logger.info("Found packages:")
    for package, version in package_versions.items():
        logger.info(f"  - {package}: {version}")
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install them using pip install -r requirements.txt")
        return False
    
    return True

def get_optimal_core_count():
    """Determine the optimal number of CPU cores to use."""
    cpu_count = os.cpu_count()
    if cpu_count is None:
        # Default to 4 if we can't determine the core count
        return 4
    
    # Use 75% of available cores (rounded up) to leave resources for system
    return max(1, int(cpu_count * 0.75))

def run_training(metric="all"):
    """
    Run the advanced model training.
    
    Args:
        metric: Which metric to train ('all' for all metrics)
    """
    # Path to training script
    script_path = Path("ml_pipeline/scripts/advanced_metrics_modeling.py")
    
    if not script_path.exists():
        logger.error(f"Training script not found at {script_path}")
        return False
    
    # Determine optimal core count
    num_cores = get_optimal_core_count()
    logger.info(f"Using {num_cores} CPU cores for training")
    
    # Run the training script with verbose output
    cmd = [sys.executable, str(script_path), "--verbose", "--num_cores", str(num_cores)]
    
    if metric != "all":
        cmd.extend(["--metric", metric])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Stream output in real-time
        for line in process.stdout:
            line = line.strip()
            if line:
                logger.info(line)
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"Successfully trained model(s) for {metric}")
            return True
        else:
            logger.error(f"Training failed with return code {process.returncode}")
            return False
    
    except Exception as e:
        logger.error(f"Error running training script: {e}")
        return False

def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train advanced ML models for news metrics")
    parser.add_argument(
        "--metric", 
        choices=METRICS + ["all"], 
        default="all",
        help="Which metric to train models for"
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Run training
    success = run_training(args.metric)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())