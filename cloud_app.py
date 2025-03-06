#!/usr/bin/env python
"""
Simplified entry point for News AI cloud deployment.
This file is used as the main entry point for Streamlit Cloud.
"""
import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Set environment variables for cloud
os.environ["IS_STREAMLIT_CLOUD"] = "true"

# Ensure API keys are available
if not os.environ.get("NEWS_API_KEY"):
    os.environ["NEWS_API_KEY"] = "e9b0f7a3c3004651a35b5f0b042e1828"

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "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"

if not os.environ.get("HUGGING_FACE_API_KEY"):
    os.environ["HUGGING_FACE_API_KEY"] = "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"

# Start the API in a background process
logger.info("Starting News AI backend server...")
try:
    from standalone_api import app
    import uvicorn
    import threading

    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    # Start API in a background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    logger.info("Backend API started in background thread")
except Exception as e:
    logger.error(f"Failed to start backend API: {e}")

# Wait for API to initialize
logger.info("Waiting for API to initialize...")
time.sleep(5)
logger.info("Starting Streamlit frontend...")

# The Streamlit app will now run automatically
# Import the streamlit app directly
import news_ai_app.frontend.streamlit_app