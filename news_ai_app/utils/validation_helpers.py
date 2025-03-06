"""
Validation helpers for data processing.

These utilities help validate and safely process data from external sources.
"""
import logging
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_split_impression(impression_item: str, is_test: bool = False) -> Dict[str, Any]:
    """
    Safely split an impression item with error handling.
    
    Args:
        impression_item: A string in the format "news_id-clicked_flag" or just "news_id" for test data
        is_test: Whether this is a test dataset (which only contains news IDs without click labels)
        
    Returns:
        Dictionary with news_id and clicked flag, or a default value if parsing fails
    """
    try:
        if is_test:
            # Test data has no click information, just news IDs
            return {
                "news_id": impression_item.strip(),
                "clicked": None  # None indicates test data with unknown clicks
            }
        
        parts = impression_item.split("-")
        if len(parts) != 2:
            logger.warning(f"Malformed impression item: {impression_item}, using default values")
            return {"news_id": parts[0] if parts else "unknown", "clicked": 0}
        
        return {
            "news_id": parts[0],
            "clicked": int(parts[1])
        }
    except (IndexError, ValueError) as e:
        logger.warning(f"Error parsing impression item '{impression_item}': {str(e)}")
        return {"news_id": "unknown", "clicked": 0}

def parse_impressions(impressions_str: str, is_test: bool = False) -> List[Dict[str, Any]]:
    """
    Safely parse a string of impressions into a list of dictionaries.
    
    Args:
        impressions_str: String containing impression items separated by spaces
        is_test: Whether this is a test dataset (which only contains news IDs without click labels)
        
    Returns:
        List of dictionaries with news_id and clicked flag
    """
    if not impressions_str or not isinstance(impressions_str, str):
        return []
    
    impressions = []
    for item in impressions_str.split():
        impressions.append(safe_split_impression(item, is_test))
    
    return impressions