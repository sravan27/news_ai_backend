#!/usr/bin/env python
"""
Run script for the Advanced EDA Streamlit apps.

This script:
1. Provides command line options to run different dashboards
2. Adds the necessary paths to sys.path
3. Runs the selected Streamlit app
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Get the directory of this script
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Add paths
REPO_ROOT = SCRIPT_DIR.parent
STREAMLIT_DIR = SCRIPT_DIR / 'streamlit'
EDA_APP_PATH = STREAMLIT_DIR / 'eda_app.py'
METRICS_APP_PATH = STREAMLIT_DIR / 'silicon_metrics_dashboard.py'
MAIN_APP_PATH = STREAMLIT_DIR / 'app.py'

# Add repository root to Python path
sys.path.append(str(REPO_ROOT))
sys.path.append(str(SCRIPT_DIR))

def run_streamlit(app_path, port=8501):
    """Run the Streamlit app."""
    try:
        subprocess.run(["streamlit", "run", str(app_path), "--server.port", str(port)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Streamlit command not found. Make sure Streamlit is installed.")
        print("Install with: pip install streamlit")
        sys.exit(1)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the Advanced EDA Streamlit apps")
    parser.add_argument(
        "--dashboard", 
        choices=["eda", "metrics", "all"], 
        default="all",
        help="Which dashboard to run (eda=news analysis, metrics=model metrics, all=both)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501,
        help="Port to run the Streamlit app on"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine which app to run
    if args.dashboard == "eda":
        app_path = EDA_APP_PATH
        app_name = "News EDA Dashboard"
    elif args.dashboard == "metrics":
        app_path = METRICS_APP_PATH
        app_name = "Silicon Metrics Dashboard"
    else:  # Run main app that includes both
        app_path = MAIN_APP_PATH
        app_name = "News AI Multi-Dashboard"
    
    # Check if app exists
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"{app_name} - Streamlit App")
    print("=" * 80)
    print(f"Running Streamlit app from: {app_path}")
    print(f"Port: {args.port}")
    print("=" * 80)
    
    # Run streamlit
    run_streamlit(app_path, args.port)

if __name__ == "__main__":
    main()