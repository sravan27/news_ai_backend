FROM python:3.10-slim

LABEL maintainer="NewsAI Team <info@newsai.com>"
LABEL description="Production Docker image for News AI Application"

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV NVIDIA_VISIBLE_DEVICES=all
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    libcurl4-openssl-dev \
    libssl-dev \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories
RUN mkdir -p logs
RUN mkdir -p MINDLarge
RUN mkdir -p models/pretrained

# Download MIND dataset if not present (can be mounted instead for faster startup)
RUN chmod +x setup_datasets.sh

# Use production entry point
ENTRYPOINT ["python", "production_app.py"]

# Default port for Streamlit
EXPOSE 8501
# API port
EXPOSE 8000