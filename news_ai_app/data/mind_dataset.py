"""
Utilities for loading and processing the MIND dataset.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from news_ai_app.config import settings
from news_ai_app.utils.validation_helpers import parse_impressions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MINDDataset:
    """
    Wrapper for the MIND (Microsoft News Dataset) for news recommendation.
    
    This class provides methods to load and preprocess the dataset for training and evaluation.
    """
    
    def __init__(
        self, 
        dataset_path: Optional[Union[str, Path]] = None,
        split: str = "train",
    ):
        """
        Initialize the MIND dataset.
        
        Args:
            dataset_path: Path to the MIND dataset. If None, uses the path from settings.
            split: Dataset split to load. One of "train", "dev", or "test".
        """
        self.dataset_path = Path(dataset_path or settings.model.mind_dataset_path)
        self.split = split
        self._validate_split()
        
        # Initialize empty dataframes
        self.news_df = None
        self.behaviors_df = None
        self.entity_embeddings = None
        self.relation_embeddings = None
        
    def _validate_split(self) -> None:
        """Validate the requested split exists."""
        split_path = self.dataset_path / f"MINDlarge_{self.split}"
        if not split_path.exists():
            raise ValueError(f"Split '{self.split}' not found at {split_path}")
    
    def load_news(self) -> pd.DataFrame:
        """
        Load news articles from the news.tsv file.
        
        Returns:
            DataFrame with news articles data.
        """
        news_path = self.dataset_path / f"MINDlarge_{self.split}" / "news.tsv"
        
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
        
        self.news_df = pd.read_csv(
            news_path, 
            sep="\t", 
            names=columns, 
            quoting=3
        )
        
        # Parse entities JSON strings
        self.news_df["title_entities"] = self.news_df["title_entities"].apply(
            lambda x: json.loads(x) if pd.notna(x) and x else []
        )
        self.news_df["abstract_entities"] = self.news_df["abstract_entities"].apply(
            lambda x: json.loads(x) if pd.notna(x) and x else []
        )
        
        return self.news_df
    
    def load_behaviors(self) -> pd.DataFrame:
        """
        Load user behaviors from the behaviors.tsv file.
        
        Returns:
            DataFrame with user behavior data.
        """
        behaviors_path = self.dataset_path / f"MINDlarge_{self.split}" / "behaviors.tsv"
        
        columns = [
            "impression_id", 
            "user_id", 
            "time", 
            "history", 
            "impressions"
        ]
        
        self.behaviors_df = pd.read_csv(
            behaviors_path, 
            sep="\t", 
            names=columns
        )
        
        # Parse history and impressions
        self.behaviors_df["history"] = self.behaviors_df["history"].apply(
            lambda x: x.split() if pd.notna(x) and x else []
        )
        
        # Use safe parsing for impressions to handle malformed data
        # Note: test set has different format (no click information)
        is_test = self.split == "test"
        self.behaviors_df["impressions"] = self.behaviors_df["impressions"].apply(
            lambda x: parse_impressions(x, is_test=is_test) if pd.notna(x) else []
        )
        
        # Log statistics about the loaded data
        logger.info(f"Loaded {len(self.behaviors_df)} behavior records from {self.split} split")
        
        # Add special handling for clicked_count feature in test data
        if is_test:
            logger.info("Test dataset detected: no click information available in impressions")
        
        return self.behaviors_df
    
    def load_entity_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load entity embeddings from the entity_embedding.vec file.
        
        Returns:
            Dictionary mapping entity IDs to their embeddings.
        """
        embeddings_path = self.dataset_path / f"MINDlarge_{self.split}" / "entity_embedding.vec"
        
        embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in tqdm(f, desc="Loading entity embeddings"):
                parts = line.strip().split('\t')
                entity_id = parts[0]
                entity_vector = np.array([float(x) for x in parts[1:]])
                embeddings[entity_id] = entity_vector
        
        self.entity_embeddings = embeddings
        return embeddings
    
    def load_relation_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load relation embeddings from the relation_embedding.vec file.
        
        Returns:
            Dictionary mapping relation IDs to their embeddings.
        """
        embeddings_path = self.dataset_path / f"MINDlarge_{self.split}" / "relation_embedding.vec"
        
        embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in tqdm(f, desc="Loading relation embeddings"):
                parts = line.strip().split('\t')
                relation_id = parts[0]
                relation_vector = np.array([float(x) for x in parts[1:]])
                embeddings[relation_id] = relation_vector
        
        self.relation_embeddings = embeddings
        return embeddings
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Load all dataset components: news, behaviors, entity embeddings, and relation embeddings.
        
        Returns:
            Tuple containing news DataFrame, behaviors DataFrame, entity embeddings dict, and relation embeddings dict.
        """
        self.load_news()
        self.load_behaviors()
        self.load_entity_embeddings()
        self.load_relation_embeddings()
        
        return self.news_df, self.behaviors_df, self.entity_embeddings, self.relation_embeddings
    
    def get_user_histories(self) -> Dict[str, List[str]]:
        """
        Extract user reading histories from behaviors DataFrame.
        
        Returns:
            Dictionary mapping user IDs to lists of news IDs they've read.
        """
        if self.behaviors_df is None:
            self.load_behaviors()
            
        user_histories = {}
        for _, row in self.behaviors_df.iterrows():
            user_id = row['user_id']
            if user_id not in user_histories:
                user_histories[user_id] = []
            
            user_histories[user_id].extend(row['history'])
            
        return user_histories
    
    def get_user_clicks(self) -> Dict[str, List[str]]:
        """
        Extract user click data from behaviors DataFrame.
        
        Returns:
            Dictionary mapping user IDs to lists of news IDs they've clicked.
        """
        if self.behaviors_df is None:
            self.load_behaviors()
            
        user_clicks = {}
        for _, row in self.behaviors_df.iterrows():
            user_id = row['user_id']
            if user_id not in user_clicks:
                user_clicks[user_id] = []
            
            for impression in row['impressions']:
                if impression['clicked'] == 1:
                    user_clicks[user_id].append(impression['news_id'])
                    
        return user_clicks
    
    def create_user_item_matrix(self) -> pd.DataFrame:
        """
        Create a user-item interaction matrix from behaviors data.
        
        Returns:
            DataFrame with users as rows, news items as columns, and values indicating interactions.
        """
        if self.behaviors_df is None:
            self.load_behaviors()
            
        # Extract all user-item interactions
        interactions = []
        for _, row in self.behaviors_df.iterrows():
            user_id = row['user_id']
            
            # Add history items (implicit feedback)
            for news_id in row['history']:
                interactions.append({
                    'user_id': user_id,
                    'news_id': news_id,
                    'interaction': 1  # Viewed/read
                })
            
            # Add impression items (explicit feedback)
            for impression in row['impressions']:
                interactions.append({
                    'user_id': user_id,
                    'news_id': impression['news_id'],
                    'interaction': 2 if impression['clicked'] == 1 else 0  # 2 for clicked, 0 for not clicked
                })
        
        # Create DataFrame from interactions
        interactions_df = pd.DataFrame(interactions)
        
        # Create pivot table
        user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='news_id',
            values='interaction',
            fill_value=0
        )
        
        return user_item_matrix


def preprocess_news_text(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess news text data for NLP tasks.
    
    Args:
        news_df: DataFrame containing news articles.
        
    Returns:
        DataFrame with additional preprocessed text columns.
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = news_df.copy()
    
    # Combine title and abstract
    df['full_text'] = df['title'] + ' ' + df['abstract'].fillna('')
    
    # Convert to lowercase
    df['title_lower'] = df['title'].str.lower()
    df['abstract_lower'] = df['abstract'].fillna('').str.lower()
    df['full_text_lower'] = df['full_text'].str.lower()
    
    return df