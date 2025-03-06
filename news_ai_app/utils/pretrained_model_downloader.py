"""
Utility for downloading pretrained models.
"""
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import requests
from tqdm import tqdm

from news_ai_app.config import settings

logger = logging.getLogger(__name__)

# URLs of pretrained models
PRETRAINED_MODEL_URLS = {
    "recommender": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/pytorch_model.bin",
    "political_influence": "https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin",
    "rhetoric_intensity": "https://huggingface.co/distilbert-base-uncased/resolve/main/pytorch_model.bin",
    "information_depth": "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin",
    "sentiment": "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/pytorch_model.bin"
}

# Local paths for pretrained models
PRETRAINED_MODEL_PATHS = {
    "recommender": "models/pretrained/recommender.pt",
    "political_influence": "models/pretrained/political_influence.pt",
    "rhetoric_intensity": "models/pretrained/rhetoric_intensity.pt",
    "information_depth": "models/pretrained/information_depth.pt",
    "sentiment": "models/pretrained/sentiment.pt"
}


def download_file(url: str, destination: Union[str, Path], chunk_size: int = 8192) -> None:
    """
    Download a file from a URL to a local destination.
    
    Args:
        url: URL to download from
        destination: Local destination path
        chunk_size: Size of chunks to download
    """
    destination = Path(destination)
    
    # Create directory if it doesn't exist
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Download file
    try:
        # Get Hugging Face API token from environment
        huggingface_token = os.environ.get("HUGGING_FACE_API_KEY")
        headers = {}
        if huggingface_token:
            headers["Authorization"] = f"Bearer {huggingface_token}"
        
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        with open(destination, "wb") as f, tqdm(
            desc=f"Downloading {destination.name}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
                    
        logger.info(f"Downloaded {url} to {destination}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        if destination.exists():
            destination.unlink()
        raise


def download_pretrained_models(models: Optional[List[str]] = None, force: bool = False) -> None:
    """
    Download pretrained models.
    
    Args:
        models: List of models to download. If None, download all models.
        force: If True, download even if models already exist.
    """
    # Set default models list if not provided
    if models is None:
        models = list(PRETRAINED_MODEL_URLS.keys())
    
    # Validate models
    invalid_models = [m for m in models if m not in PRETRAINED_MODEL_URLS]
    if invalid_models:
        logger.error(f"Invalid models: {invalid_models}")
        logger.info(f"Available models: {list(PRETRAINED_MODEL_URLS.keys())}")
        return
    
    # Create models directory
    models_dir = Path("models/pretrained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download models
    for model_name in models:
        url = PRETRAINED_MODEL_URLS[model_name]
        path = Path(PRETRAINED_MODEL_PATHS[model_name])
        
        if path.exists() and not force:
            logger.info(f"Model {model_name} already exists at {path}. Skipping.")
            continue
        
        logger.info(f"Downloading model {model_name} from {url}...")
        download_file(url, path)
    
    logger.info("All models downloaded successfully.")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Download pretrained models.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(PRETRAINED_MODEL_URLS.keys()),
        help="Models to download. If not specified, download all models."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download models even if they already exist."
    )
    
    args = parser.parse_args()
    
    # Download models
    download_pretrained_models(models=args.models, force=args.force)