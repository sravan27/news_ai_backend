#\!/usr/bin/env python3
"""
Deployment script for News AI application.
This script starts the full production deployment of the News AI application.
"""
import logging
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/deployment.log"),
    ],
)

logger = logging.getLogger(__name__)

def verify_model_files():
    """Verify that all required model files exist."""
    logger.info("Verifying model files...")
    
    required_models = [
        "models/pretrained/recommender.pt",
        "models/pretrained/political_influence.pt",
        "models/pretrained/rhetoric_intensity.pt",
        "models/pretrained/information_depth.pt",
        "models/pretrained/sentiment.pt"
    ]
    
    missing_models = [m for m in required_models if not os.path.exists(m)]
    
    if missing_models:
        logger.error(f"Missing model files: {missing_models}")
        logger.error("Please run train_full_models.py first")
        return False
    
    logger.info("All model files verified successfully")
    return True

def verify_environment_variables():
    """Verify that all required environment variables are set."""
    logger.info("Verifying environment variables...")
    
    required_vars = [
        "NEWS_API_KEY",
        "OPENAI_API_KEY",
        "HUGGING_FACE_API_KEY"
    ]
    
    missing_vars = [v for v in required_vars if not os.getenv(v)]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        logger.error("Please make sure these are set in your .env file")
        return False
    
    logger.info("All environment variables verified successfully")
    return True

def verify_mind_dataset():
    """Verify that the MIND dataset is available."""
    logger.info("Verifying MIND dataset...")
    
    mind_path = Path(os.getenv("MIND_DATASET_PATH", "./MINDLarge"))
    
    if not mind_path.exists():
        logger.error(f"MIND dataset directory {mind_path} not found")
        return False
    
    # Check for train/dev/test splits
    splits = ["MINDlarge_train", "MINDlarge_dev", "MINDlarge_test"]
    missing_splits = [s for s in splits if not (mind_path / s).exists()]
    
    if missing_splits:
        logger.error(f"Missing MIND dataset splits: {missing_splits}")
        return False
    
    logger.info("MIND dataset verified successfully")
    return True

def run_backend(port=8000):
    """Run the FastAPI backend server."""
    logger.info(f"Starting backend server on port {port}...")
    return subprocess.Popen(
        ["uvicorn", "news_ai_app.main:app", "--host", "0.0.0.0", "--port", str(port), "--workers", "4"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

def run_frontend(port=8501):
    """Run the Streamlit frontend app."""
    logger.info(f"Starting frontend app on port {port}...")
    return subprocess.Popen(
        ["streamlit", "run", "news_ai_app/frontend/streamlit_app.py", "--server.port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

def stream_logs(process, prefix):
    """Stream logs from a subprocess."""
    for line in iter(process.stdout.readline, ""):
        logger.info(f"{prefix}: {line.strip()}")
        print(f"{prefix}: {line.strip()}")

def handle_sigint(sig, frame):
    """Handle Ctrl+C by terminating both processes."""
    logger.info("Shutting down servers...")
    for process in active_processes:
        if process.poll() is None:  # If process is still running
            process.terminate()
    sys.exit(0)

def main():
    """Run the deployment script."""
    logger.info("Starting News AI deployment...")
    
    # Verify prerequisites
    if not verify_model_files():
        sys.exit(1)
    
    if not verify_environment_variables():
        sys.exit(1)
    
    if not verify_mind_dataset():
        sys.exit(1)
    
    # Set up processes list
    global active_processes
    active_processes = []
    
    # Register signal handler for Ctrl+C, but only in the main thread
    try:
        signal.signal(signal.SIGINT, handle_sigint)
    except ValueError:
        # We're not in the main thread, so we can't use signal handlers
        logger.warning("Signal handlers can't be set (not in main thread)")
        # In this case, we'll rely on the parent process to clean up
    
    # Start backend
    backend_process = run_backend()
    active_processes.append(backend_process)
    
    # Wait a bit for the backend to start
    time.sleep(2)
    
    # Start frontend
    frontend_process = run_frontend()
    active_processes.append(frontend_process)
    
    # Stream logs from both processes
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(stream_logs, backend_process, "BACKEND")
        executor.submit(stream_logs, frontend_process, "FRONTEND")
    
    # Wait for processes to complete (shouldn't normally happen)
    for process in active_processes:
        process.wait()

if __name__ == "__main__":
    main()
