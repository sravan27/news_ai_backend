#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions for robust notebook execution.
These functions handle common error cases and ensure smooth processing.
"""

import os
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("notebook_helpers")


def safe_json_loads(json_str, default=None):
    """Safely load JSON string with error handling."""
    if not isinstance(json_str, str) or not json_str:
        return default
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON: {json_str[:50]}...")
        return default


def robust_file_reader(file_path, read_func, **kwargs):
    """
    Generic robust file reader that handles errors gracefully.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
    read_func : callable
        Function to read the file (e.g., pd.read_csv)
    **kwargs : 
        Additional arguments to pass to read_func
    
    Returns:
    --------
    data : object or None
        The data read from the file, or None if an error occurred
    error : str or None
        Error message if an error occurred, or None if successful
    """
    try:
        if not os.path.exists(file_path):
            return None, f"File not found: {file_path}"
        
        data = read_func(file_path, **kwargs)
        return data, None
    except Exception as e:
        error_msg = f"Error reading {file_path}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None, error_msg


def robust_tsv_reader(file_path, delimiter='\t', **kwargs):
    """Robustly read a TSV file with error handling."""
    return robust_file_reader(
        file_path,
        pd.read_csv,
        sep=delimiter,
        **kwargs
    )


def robust_parquet_reader(file_path, **kwargs):
    """Robustly read a Parquet file with error handling."""
    try:
        import pyarrow.parquet as pq
        return robust_file_reader(
            file_path,
            pq.read_table,
            **kwargs
        )
    except ImportError:
        return None, "pyarrow not installed"


def robust_csv_reader(file_path, **kwargs):
    """Robustly read a CSV file with error handling."""
    return robust_file_reader(
        file_path,
        pd.read_csv,
        **kwargs
    )


def safe_split(text, separator=None, default=None):
    """Safely split text with error handling."""
    if not isinstance(text, str):
        return default or []
    
    try:
        return text.split(separator)
    except Exception:
        return default or []


def parse_impressions_safely(impressions_str):
    """
    Parse impressions string to list of dicts in a robust way.
    
    Parameters:
    -----------
    impressions_str : str
        String of impressions in the format "item1-1 item2-0 ..."
    
    Returns:
    --------
    list
        List of dicts with news_id and clicked keys
    """
    if not isinstance(impressions_str, str):
        return []
    
    result = []
    for item in safe_split(impressions_str):
        try:
            if '-' in item:
                parts = item.split('-')
                news_id = parts[0]
                clicked = int(parts[1]) if len(parts) > 1 else 0
                # Ensure clicked is either 0 or 1
                clicked = min(max(clicked, 0), 1)
            else:
                news_id = item
                clicked = 0
            
            result.append({
                'news_id': news_id,
                'clicked': clicked
            })
        except Exception as e:
            logger.warning(f"Error parsing impression item '{item}': {e}")
            # Add a fallback entry with clicked=0
            result.append({
                'news_id': item,
                'clicked': 0
            })
    
    return result


def parse_history_safely(history_str):
    """Parse history string to list in a robust way."""
    if not isinstance(history_str, str):
        return []
    
    return safe_split(history_str)


def safe_to_datetime(time_str):
    """Safely convert string to datetime with error handling."""
    try:
        return pd.to_datetime(time_str)
    except Exception:
        warnings.warn(f"Failed to convert '{time_str}' to datetime")
        return pd.NaT


def load_news_data_safely(file_path, **kwargs):
    """
    Load news data with robust error handling.
    
    Parameters:
    -----------
    file_path : str
        Path to the news.tsv file
    **kwargs : dict
        Additional arguments to pass to pd.read_csv
    
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with news data, or None if an error occurred
    """
    columns = [
        "news_id", 
        "category", 
        "subcategory", 
        "title", 
        "abstract", 
        "url", 
        "title_entities", 
        "abstract_entities"
    ]
    
    news_df, error = robust_tsv_reader(
        file_path,
        names=columns,
        quoting=3,
        **kwargs
    )
    
    if error is not None:
        return None, error
    
    # Process entities with robust handling
    try:
        news_df["title_entities"] = news_df["title_entities"].apply(
            lambda x: safe_json_loads(x, default=[])
        )
        news_df["abstract_entities"] = news_df["abstract_entities"].apply(
            lambda x: safe_json_loads(x, default=[])
        )
        
        # Fill missing values safely
        news_df['abstract'] = news_df['abstract'].fillna('')
        news_df['url'] = news_df['url'].fillna('')
        
        return news_df, None
    except Exception as e:
        error_msg = f"Error processing news data: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None, error_msg


def load_behaviors_data_safely(file_path, **kwargs):
    """
    Load behaviors data with robust error handling.
    
    Parameters:
    -----------
    file_path : str
        Path to the behaviors.tsv file
    **kwargs : dict
        Additional arguments to pass to pd.read_csv
    
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with behaviors data, or None if an error occurred
    """
    columns = [
        "impression_id", 
        "user_id", 
        "time", 
        "history", 
        "impressions"
    ]
    
    behaviors_df, error = robust_tsv_reader(
        file_path,
        names=columns,
        **kwargs
    )
    
    if error is not None:
        return None, error
    
    try:
        # Save original string format
        behaviors_df['history_str'] = behaviors_df['history']
        behaviors_df['impressions_str'] = behaviors_df['impressions']
        
        # Parse history and impressions with robust error handling
        behaviors_df['history'] = behaviors_df['history'].apply(parse_history_safely)
        behaviors_df['impressions'] = behaviors_df['impressions'].apply(parse_impressions_safely)
        
        # Extract additional features
        behaviors_df['history_length'] = behaviors_df['history'].apply(len)
        behaviors_df['impressions_count'] = behaviors_df['impressions'].apply(len)
        
        # Extract impression clicks
        behaviors_df['click_count'] = behaviors_df['impressions'].apply(
            lambda x: sum(1 for item in x if item.get('clicked', 0) == 1)
        )
        
        behaviors_df['click_ratio'] = behaviors_df.apply(
            lambda row: row['click_count'] / row['impressions_count'] if row['impressions_count'] > 0 else 0,
            axis=1
        )
        
        # Convert time string to timestamp
        behaviors_df['timestamp'] = behaviors_df['time'].apply(safe_to_datetime)
        
        # Extract time components
        behaviors_df['day_of_week'] = behaviors_df['timestamp'].dt.dayofweek
        behaviors_df['hour_of_day'] = behaviors_df['timestamp'].dt.hour
        
        return behaviors_df, None
    except Exception as e:
        error_msg = f"Error processing behaviors data: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None, error_msg


def process_behaviors_to_parquet_safely(split, mind_path, output_path, **kwargs):
    """
    Process behaviors data to Parquet format with robust error handling.
    
    Parameters:
    -----------
    split : str
        Split name (train, dev, test)
    mind_path : str
        Path to the MIND dataset
    output_path : str
        Path to save the processed Parquet file
    **kwargs : dict
        Additional arguments to pass to pd.read_csv
    
    Returns:
    --------
    str or None
        Path to the output file, or None if an error occurred
    """
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    behaviors_path = Path(mind_path) / f"MINDlarge_{split}" / "behaviors.tsv"
    parquet_path = Path(output_path) / f"behaviors_{split}.parquet"
    
    logger.info(f"Processing {split} behaviors dataset...")
    
    # Load and process behaviors data
    behaviors_df, error = load_behaviors_data_safely(behaviors_path, **kwargs)
    
    if error is not None:
        logger.error(f"Failed to load behaviors data: {error}")
        return None
    
    # Convert to Parquet
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        table = pa.Table.from_pandas(behaviors_df)
        pq.write_table(table, parquet_path, compression='snappy')
        
        duration = time.time() - start_time
        logger.info(f"Processed {len(behaviors_df)} behavior records in {duration:.2f} seconds")
        logger.info(f"Output saved to: {parquet_path}")
        
        return str(parquet_path)
    except Exception as e:
        error_msg = f"Error writing to Parquet: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None


def process_news_to_parquet_safely(split, mind_path, output_path, **kwargs):
    """
    Process news data to Parquet format with robust error handling.
    
    Parameters:
    -----------
    split : str
        Split name (train, dev, test)
    mind_path : str
        Path to the MIND dataset
    output_path : str
        Path to save the processed Parquet file
    **kwargs : dict
        Additional arguments to pass to pd.read_csv
    
    Returns:
    --------
    str or None
        Path to the output file, or None if an error occurred
    """
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    news_path = Path(mind_path) / f"MINDlarge_{split}" / "news.tsv"
    parquet_path = Path(output_path) / f"news_{split}.parquet"
    
    logger.info(f"Processing {split} news dataset...")
    
    # Load and process news data
    news_df, error = load_news_data_safely(news_path, **kwargs)
    
    if error is not None:
        logger.error(f"Failed to load news data: {error}")
        return None
    
    try:
        # Add text length features
        news_df['title_length'] = news_df['title'].str.len()
        news_df['abstract_length'] = news_df['abstract'].str.len()
        
        # Extract entity counts
        news_df['title_entity_count'] = news_df['title_entities'].apply(len)
        news_df['abstract_entity_count'] = news_df['abstract_entities'].apply(len)
        
        # Convert to Parquet
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        table = pa.Table.from_pandas(news_df)
        pq.write_table(table, parquet_path, compression='snappy')
        
        duration = time.time() - start_time
        logger.info(f"Processed {len(news_df)} news articles in {duration:.2f} seconds")
        logger.info(f"Output saved to: {parquet_path}")
        
        return str(parquet_path)
    except Exception as e:
        error_msg = f"Error writing to Parquet: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None


if __name__ == "__main__":
    print("Testing notebook helpers...")
    # Add tests here if needed