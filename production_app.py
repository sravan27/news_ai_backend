#!/usr/bin/env python3
"""
Production-ready News AI Application
This script provides a fully containerized, production-grade deployment
of the News AI backend and frontend using the medallion architecture.
"""
import os
import sys
import time
import logging
import subprocess
import threading
import signal
import atexit
from pathlib import Path

# Set up robust logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("NewsAI")

# Configure environment
def setup_environment():
    """Prepare the environment with necessary variables and settings."""
    logger.info("Setting up News AI production environment")
    
    # Environment variables
    os.environ["IS_PRODUCTION"] = "true"
    
    # Set API keys
    api_keys = {
        "NEWS_API_KEY": "e9b0f7a3c3004651a35b5f0b042e1828",
        "OPENAI_API_KEY": "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",
        "HUGGING_FACE_API_KEY": "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"
    }
    
    for key, value in api_keys.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set {key} environment variable")
    
    # Detect path for MIND dataset
    if os.path.exists("/mount/src/news_ai_backend/MINDLarge"):
        # Streamlit Cloud path
        os.environ["MIND_DATASET_PATH"] = "/mount/src/news_ai_backend/MINDLarge"
    elif os.path.exists("/app/MINDLarge"):
        # Docker container path
        os.environ["MIND_DATASET_PATH"] = "/app/MINDLarge"
    else:
        # Local development path - relative to script location
        os.environ["MIND_DATASET_PATH"] = str(Path(__file__).parent / "MINDLarge")
    
    logger.info(f"MIND dataset path: {os.environ['MIND_DATASET_PATH']}")
    
    # Other configuration settings
    os.environ["PYTHONUNBUFFERED"] = "1"  # Ensure unbuffered output
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable parallel tokenization
    os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
    os.environ["MKL_NUM_THREADS"] = "4"  # Limit MKL threads
    
    logger.info("Environment setup complete")

# Backend API Server
class APIServer:
    """Manages the News AI backend API server."""
    
    def __init__(self, port=8000):
        self.port = port
        self.process = None
        self.running = False
    
    def start(self):
        """Start the API server."""
        from standalone_api import app
        import uvicorn
        
        def run_server():
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=self.port,
                log_level="info",
                workers=1
            )
        
        logger.info(f"Starting API server on port {self.port}")
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        self.running = True
        
        # Wait for server to start
        max_retries = 5
        for i in range(max_retries):
            try:
                import requests
                response = requests.get(f"http://localhost:{self.port}/api/health")
                if response.status_code == 200:
                    logger.info("API server started successfully")
                    return True
            except:
                pass
            
            logger.info(f"Waiting for API server to start (attempt {i+1}/{max_retries})...")
            time.sleep(2)
        
        logger.warning("API server may not have started properly")
        return False
    
    def stop(self):
        """Stop the API server."""
        if self.running:
            logger.info("Shutting down API server")
            self.running = False
            # The thread will terminate when the process exits

# Main application class
class NewsAIApp:
    """Main News AI application manager."""
    
    def __init__(self):
        self.api_server = APIServer()
    
    def start(self):
        """Start the entire application stack."""
        logger.info("Starting News AI Production Application")
        
        # Start the API server
        api_started = self.api_server.start()
        if not api_started:
            logger.warning("API server didn't start properly, but continuing...")
        
        # Return control to Streamlit
        logger.info("Backend services started, handing control to Streamlit")
    
    def stop(self):
        """Stop all application components."""
        logger.info("Shutting down News AI Application")
        self.api_server.stop()
        logger.info("Shutdown complete")

# Initialize app
app_instance = None

def main():
    """Main entry point."""
    # Set up environment
    setup_environment()
    
    # Create and start application
    global app_instance
    app_instance = NewsAIApp()
    app_instance.start()
    
    # Register cleanup handlers
    def cleanup():
        if app_instance:
            app_instance.stop()
    
    atexit.register(cleanup)
    
    # The actual Streamlit app is imported here, so it can use our API
    try:
        import streamlit as st
        from news_ai_app.frontend.streamlit_app import main as streamlit_main
        
        # Override the API base URL in the global environment
        st.session_state["API_BASE_URL"] = "http://localhost:8000/api"
        
        # Run the Streamlit app
        streamlit_main()
    except Exception as e:
        logger.error(f"Error running Streamlit app: {e}")
        
        # Fallback to simple interface if the main app fails
        import streamlit as st
        st.title("News AI Dashboard")
        st.write("Welcome to the News AI platform!")
        st.error("The full application could not be loaded. Please check the logs for details.")
        
        # Show basic news feed
        st.subheader("Sample News")
        import httpx
        try:
            response = httpx.get("http://localhost:8000/api/news?limit=5")
            if response.status_code == 200:
                news = response.json()
                for article in news:
                    st.write(f"**{article['title']}**")
                    st.write(article.get('abstract', ''))
                    st.write(f"Category: {article.get('category', '')}")
                    st.write("---")
            else:
                st.warning("Could not fetch news articles from API.")
        except Exception as e:
            st.warning(f"API connection failed: {e}")
            st.write("Displaying sample news articles:")
            for i in range(5):
                st.write(f"**Sample Article {i+1}**")
                st.write(f"This is a placeholder article {i+1}.")
                st.write("---")

if __name__ == "__main__":
    main()