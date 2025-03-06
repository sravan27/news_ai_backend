#!/usr/bin/env python
"""
Standalone Streamlit entry point for News AI app.
This is designed to be the main file for Streamlit Cloud.
"""
import os
import sys
import time
import logging
import subprocess
from pathlib import Path

# Set environment variable to indicate Streamlit Cloud environment
os.environ["IS_STREAMLIT_CLOUD"] = "true"

# Configure API keys
API_KEYS = {
    "NEWS_API_KEY": "e9b0f7a3c3004651a35b5f0b042e1828",
    "OPENAI_API_KEY": "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",
    "HUGGING_FACE_API_KEY": "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"
}

# Set API keys if not already set
for key, value in API_KEYS.items():
    if not os.environ.get(key):
        os.environ[key] = value

# Start the API server in a separate process
def start_api_server():
    subprocess.Popen(
        ["python", "-m", "standalone_api"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

# Start the API server first
try:
    print("Starting backend API server...")
    start_api_server()
    print("API server started in background")
    
    # Wait for the API to initialize
    print("Waiting for API to initialize...")
    time.sleep(3)
except Exception as e:
    print(f"Error starting API server: {e}")

# Now import the Streamlit app
import streamlit as st
import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from typing import Dict, List, Optional, Tuple
import datetime
import json

# Set page configuration
st.set_page_config(
    page_title="News AI Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL for Streamlit Cloud
API_BASE_URL = "/api"

# Include all the functions from the original streamlit_app.py
# Helper Functions
def fetch_news(category: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """Fetch news articles from the API."""
    params = {"limit": limit}
    if category:
        params["category"] = category
        
    try:
        response = httpx.get(f"{API_BASE_URL}/news", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching news: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        # Return dummy news for demo purposes
        return [
            {
                "news_id": f"N{i}",
                "title": f"Sample News Article {i}",
                "abstract": f"This is a sample news article for demonstration purposes. #{i}",
                "category": "news",
                "subcategory": "general"
            }
            for i in range(1, limit+1)
        ]

# Main app
st.title("News AI Dashboard")
st.write("Welcome to News AI! This platform provides advanced news analytics and recommendations.")

# Display some sample data 
st.header("Sample News")
news = fetch_news(limit=5)
for article in news:
    st.subheader(article["title"])
    st.write(article.get("abstract", ""))
    st.write(f"Category: {article.get('category', '')}")
    st.write("---")