#!/usr/bin/env python
"""
Script to run both the backend API and frontend Streamlit app.
"""
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

def run_backend():
    """Run the FastAPI backend."""
    print("Starting News AI backend server...")
    return subprocess.Popen(
        ["uvicorn", "news_ai_app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

def run_frontend():
    """Run the Streamlit frontend."""
    print("Starting News AI frontend app...")
    return subprocess.Popen(
        ["streamlit", "run", "news_ai_app/frontend/streamlit_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

def stream_logs(process, prefix):
    """Stream logs from a subprocess."""
    for line in iter(process.stdout.readline, ""):
        print(f"{prefix}: {line.strip()}")

def handle_sigint(sig, frame):
    """Handle Ctrl+C by terminating both processes."""
    print("\nShutting down servers...")
    for process in active_processes:
        if process.poll() is None:  # If process is still running
            process.terminate()
    sys.exit(0)

if __name__ == "__main__":
    active_processes = []
    
    # Start both servers
    backend_process = run_backend()
    active_processes.append(backend_process)
    
    # Wait a bit for the backend to start before launching the frontend
    time.sleep(2)
    
    frontend_process = run_frontend()
    active_processes.append(frontend_process)
    
    # Only register signal handler if running in main thread (not in Streamlit cloud)
    try:
        signal.signal(signal.SIGINT, handle_sigint)
    except ValueError:
        # Skip signal handling when running in non-main thread (e.g., Streamlit Cloud)
        print("Signal handling disabled - not running in main thread")
    
    # Stream logs from both processes
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(stream_logs, backend_process, "BACKEND")
        executor.submit(stream_logs, frontend_process, "FRONTEND")
    
    # Wait for processes to complete (shouldn't normally happen)
    for process in active_processes:
        process.wait()