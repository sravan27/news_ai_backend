"""
Helper functions for Jupyter notebooks in the News AI project.

These utilities help with common data processing tasks in notebooks.
"""
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging for notebooks
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def safe_parse_impressions(impressions_str: str, is_test: bool = False) -> List[Dict[str, Any]]:
    """
    Safely parse impressions string to handle malformed data.
    
    Args:
        impressions_str: String of impressions in format "news_id-clicked news_id-clicked ..." 
                         or just space-separated news IDs for test data
        is_test: Whether this is test data (no click information)
        
    Returns:
        List of dictionaries with news_id and clicked flag
    """
    if not impressions_str or not isinstance(impressions_str, str):
        return []
    
    impressions = []
    for item in impressions_str.split():
        try:
            if is_test:
                # Test data has no click information, just news IDs
                impressions.append({
                    "news_id": item.strip(),
                    "clicked": None  # None indicates test data with unknown clicks
                })
                continue
                
            parts = item.split("-")
            if len(parts) != 2:
                logger.warning(f"Malformed impression item: {item}, using default values")
                impressions.append({
                    "news_id": parts[0] if parts else "unknown", 
                    "clicked": 0
                })
                continue
                
            impressions.append({
                "news_id": parts[0],
                "clicked": int(parts[1])
            })
        except (IndexError, ValueError) as e:
            logger.warning(f"Error parsing impression item '{item}': {str(e)}")
            impressions.append({"news_id": "unknown", "clicked": 0})
    
    return impressions

def process_behaviors_file(file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, is_test: bool = False) -> pd.DataFrame:
    """
    Process a behaviors.tsv file with error handling to prevent any failures.
    
    Args:
        file_path: Path to the behaviors.tsv file
        output_path: Optional path to save the processed data as parquet
        is_test: Whether this is test data (with no click information)
        
    Returns:
        Processed DataFrame
    """
    logger.info(f"Processing behaviors file: {file_path} (is_test={is_test})")
    
    # Load the behaviors data
    columns = ["impression_id", "user_id", "time", "history", "impressions"]
    
    try:
        behaviors_df = pd.read_csv(file_path, sep="\t", names=columns)
        logger.info(f"Loaded {len(behaviors_df)} records from {file_path}")
    except Exception as e:
        logger.error(f"Error loading behaviors file: {str(e)}")
        raise
    
    # Process history with error handling
    logger.info("Processing history data...")
    behaviors_df["history"] = behaviors_df["history"].apply(
        lambda x: x.split() if pd.notna(x) and x else []
    )
    
    # Process impressions with error handling
    logger.info("Processing impressions data...")
    behaviors_df["impressions"] = behaviors_df["impressions"].apply(
        lambda x: safe_parse_impressions(x, is_test=is_test) if pd.notna(x) else []
    )
    
    # Add derived features
    logger.info("Adding derived features...")
    behaviors_df["history_length"] = behaviors_df["history"].apply(len)
    behaviors_df["impressions_count"] = behaviors_df["impressions"].apply(len)
    
    # Only calculate clicked_count for non-test data (test data has no click information)
    if not is_test:
        behaviors_df["clicked_count"] = behaviors_df["impressions"].apply(
            lambda x: sum(1 for item in x if item["clicked"] == 1)
        )
    else:
        # For test data, set clicked_count to None
        behaviors_df["clicked_count"] = None
        logger.info("Test data detected: clicked_count set to None")
    
    # Save to parquet if output path is provided
    if output_path:
        logger.info(f"Saving processed data to {output_path}")
        behaviors_df.to_parquet(output_path)
    
    return behaviors_df

def process_news_file(file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Process a news.tsv file with error handling.
    
    Args:
        file_path: Path to the news.tsv file
        output_path: Optional path to save the processed data as parquet
        
    Returns:
        Processed DataFrame
    """
    logger.info(f"Processing news file: {file_path}")
    
    # Load the news data
    columns = [
        "news_id", "category", "subcategory", "title", "abstract", 
        "url", "title_entities", "abstract_entities"
    ]
    
    try:
        news_df = pd.read_csv(file_path, sep="\t", names=columns, quoting=3)
        logger.info(f"Loaded {len(news_df)} records from {file_path}")
    except Exception as e:
        logger.error(f"Error loading news file: {str(e)}")
        raise
    
    # Parse entity JSON data with error handling
    logger.info("Parsing entity data...")
    
    def safe_parse_json(x):
        if not pd.notna(x) or not x:
            return []
        try:
            return pd.json_normalize(x)
        except Exception:
            try:
                import json
                return json.loads(x)
            except Exception as e:
                logger.warning(f"Error parsing JSON: {str(e)}, returning empty list")
                return []
    
    news_df["title_entities"] = news_df["title_entities"].apply(safe_parse_json)
    news_df["abstract_entities"] = news_df["abstract_entities"].apply(safe_parse_json)
    
    # Add derived features
    logger.info("Adding derived features...")
    news_df["title_length"] = news_df["title"].fillna("").apply(len)
    news_df["abstract_length"] = news_df["abstract"].fillna("").apply(len)
    news_df["has_abstract"] = news_df["abstract"].notna() & (news_df["abstract"] != "")
    news_df["entity_count"] = news_df["title_entities"].apply(len) + news_df["abstract_entities"].apply(len)
    
    # Save to parquet if output path is provided
    if output_path:
        logger.info(f"Saving processed data to {output_path}")
        news_df.to_parquet(output_path)
    
    return news_df

def process_mind_dataset_to_parquet(base_path: Union[str, Path], split: str, output_dir: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process both news and behaviors files from a MIND dataset split.
    
    Args:
        base_path: Base path to the MIND dataset
        split: Dataset split ('train', 'dev', 'test')
        output_dir: Optional directory to save processed files
        
    Returns:
        Tuple of (news_df, behaviors_df)
    """
    base_path = Path(base_path)
    split_path = base_path / f"MINDlarge_{split}"
    
    if not split_path.exists():
        raise ValueError(f"Split directory not found: {split_path}")
    
    # Define file paths
    news_file = split_path / "news.tsv"
    behaviors_file = split_path / "behaviors.tsv"
    
    # Define output paths if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        news_output = output_dir / f"{split}_news.parquet"
        behaviors_output = output_dir / f"{split}_behaviors.parquet"
    else:
        news_output = None
        behaviors_output = None
    
    # Determine if this is test data (which has a different format)
    is_test = split == "test"
    
    # Process files
    logger.info(f"Processing {split} split...")
    news_df = process_news_file(news_file, news_output)
    behaviors_df = process_behaviors_file(behaviors_file, behaviors_output, is_test=is_test)
    
    return news_df, behaviors_df