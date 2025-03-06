#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIND Dataset Loader for Advanced EDA
-----------------------------------
Utility functions to load and preprocess the MIND dataset for EDA purposes.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MINDDatasetLoader:
    """
    Wrapper for the MIND (Microsoft News Dataset) specifically for EDA purposes.
    
    Provides methods to load, preprocess, and explore the dataset with extended capabilities.
    """
    
    def __init__(
        self, 
        dataset_path: Optional[Union[str, Path]] = None,
        split: str = "train",
        dataset_size: str = "large",  # "large" or "small"
    ):
        """
        Initialize the MIND dataset loader.
        
        Args:
            dataset_path: Path to the MIND dataset
            split: Dataset split to load. One of "train", "dev", or "test"
            dataset_size: Size of the dataset. One of "large" or "small"
        """
        if dataset_path is None:
            # Try to find dataset in standard locations
            potential_paths = [
                Path("/Users/sravansridhar/Documents/news_ai/MINDLarge"),
                Path("/Users/sravansridhar/Documents/news_ai/MINDSmall"),
                Path("./MINDLarge"),
                Path("./MINDSmall")
            ]
            for path in potential_paths:
                if path.exists():
                    dataset_path = path
                    break
            if dataset_path is None:
                raise ValueError("Dataset path not specified and couldn't find default locations.")
        
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.dataset_size = dataset_size
        self._validate_split()
        
        # Initialize empty dataframes
        self.news_df = None
        self.behaviors_df = None
        self.entity_embeddings = None
        self.relation_embeddings = None
        
    def _validate_split(self) -> None:
        """Validate the requested split exists."""
        split_path = self.dataset_path / f"MIND{self.dataset_size}_{self.split}"
        if not split_path.exists():
            raise ValueError(f"Split '{self.split}' not found at {split_path}")
    
    def load_news(self) -> pd.DataFrame:
        """
        Load news articles from the news.tsv file.
        
        Returns:
            DataFrame with news articles data.
        """
        news_path = self.dataset_path / f"MIND{self.dataset_size}_{self.split}" / "news.tsv"
        logger.info(f"Loading news from {news_path}")
        
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
            quoting=3  # QUOTE_NONE
        )
        
        # Parse entities JSON strings
        self.news_df["title_entities"] = self.news_df["title_entities"].apply(
            lambda x: json.loads(x) if pd.notna(x) and x else []
        )
        self.news_df["abstract_entities"] = self.news_df["abstract_entities"].apply(
            lambda x: json.loads(x) if pd.notna(x) and x else []
        )
        
        logger.info(f"Loaded {len(self.news_df)} news articles")
        return self.news_df
    
    def load_behaviors(self) -> pd.DataFrame:
        """
        Load user behaviors from the behaviors.tsv file.
        
        Returns:
            DataFrame with user behavior data.
        """
        behaviors_path = self.dataset_path / f"MIND{self.dataset_size}_{self.split}" / "behaviors.tsv"
        logger.info(f"Loading behaviors from {behaviors_path}")
        
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
        
        # Parse impressions and handle different formats between test and train/dev
        is_test = self.split == "test"
        self.behaviors_df["impressions"] = self.behaviors_df["impressions"].apply(
            lambda x: self._parse_impressions(x, is_test=is_test) if pd.notna(x) else []
        )
        
        # Convert time to datetime
        self.behaviors_df["time"] = pd.to_datetime(self.behaviors_df["time"])
        
        logger.info(f"Loaded {len(self.behaviors_df)} behavior records")
        return self.behaviors_df
    
    def _parse_impressions(self, impressions_str: str, is_test: bool = False) -> List[Dict]:
        """
        Parse impressions string into a structured format.
        
        Args:
            impressions_str: String containing impressions data
            is_test: Whether the data is from the test set (no click information)
            
        Returns:
            List of dictionaries with impression data
        """
        if not impressions_str:
            return []
        
        result = []
        items = impressions_str.split()
        
        for item in items:
            if is_test:
                # Test set format: news_id
                news_id = item
                clicked = None
            else:
                # Train/dev set format: news_id-click
                parts = item.split('-')
                if len(parts) != 2:
                    continue
                news_id, clicked = parts
                clicked = int(clicked)
            
            result.append({
                "news_id": news_id,
                "clicked": clicked
            })
            
        return result
    
    def load_entity_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load entity embeddings from the entity_embedding.vec file.
        
        Returns:
            Dictionary mapping entity IDs to their embeddings.
        """
        embeddings_path = self.dataset_path / f"MIND{self.dataset_size}_{self.split}" / "entity_embedding.vec"
        logger.info(f"Loading entity embeddings from {embeddings_path}")
        
        embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in tqdm(f, desc="Loading entity embeddings"):
                parts = line.strip().split('\t')
                entity_id = parts[0]
                entity_vector = np.array([float(x) for x in parts[1:]])
                embeddings[entity_id] = entity_vector
        
        self.entity_embeddings = embeddings
        logger.info(f"Loaded {len(embeddings)} entity embeddings")
        return embeddings
    
    def load_relation_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load relation embeddings from the relation_embedding.vec file.
        
        Returns:
            Dictionary mapping relation IDs to their embeddings.
        """
        embeddings_path = self.dataset_path / f"MIND{self.dataset_size}_{self.split}" / "relation_embedding.vec"
        logger.info(f"Loading relation embeddings from {embeddings_path}")
        
        embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in tqdm(f, desc="Loading relation embeddings"):
                parts = line.strip().split('\t')
                relation_id = parts[0]
                relation_vector = np.array([float(x) for x in parts[1:]])
                embeddings[relation_id] = relation_vector
        
        self.relation_embeddings = embeddings
        logger.info(f"Loaded {len(embeddings)} relation embeddings")
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
    
    def process_entities_long_format(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process entities in long format (row expansion) for detailed analysis.
        
        Returns:
            Tuple of (title_entities_long, abstract_entities_long) DataFrames
        """
        if self.news_df is None:
            self.load_news()
        
        # Process title entities
        logger.info("Processing title entities in long format")
        title_entities_df = self.news_df.copy()
        title_entities_df = title_entities_df.explode('title_entities')
        title_entities_df.loc[title_entities_df['title_entities'].apply(lambda x: x == [] or pd.isna(x)), 'title_entities'] = None
        
        # Extract entity dictionaries into columns
        title_entities_long = pd.concat(
            [title_entities_df.drop(columns=['title_entities']), 
             title_entities_df['title_entities'].apply(pd.Series)], 
            axis=1
        )
        
        # Process abstract entities
        logger.info("Processing abstract entities in long format")
        abstract_entities_df = self.news_df.copy()
        abstract_entities_df = abstract_entities_df.explode('abstract_entities')
        abstract_entities_df.loc[abstract_entities_df['abstract_entities'].apply(lambda x: x == [] or pd.isna(x)), 'abstract_entities'] = None
        
        # Extract entity dictionaries into columns
        abstract_entities_long = pd.concat(
            [abstract_entities_df.drop(columns=['abstract_entities']), 
             abstract_entities_df['abstract_entities'].apply(pd.Series)], 
            axis=1
        )
        
        logger.info(f"Created entity long format with {len(title_entities_long)} title entities and {len(abstract_entities_long)} abstract entities")
        
        return title_entities_long, abstract_entities_long
    
    def get_entity_type_mapping(self) -> Dict[str, str]:
        """Return the mapping of entity type codes to descriptions."""
        return {
            "P": "Person",
            "O": "Organization",
            "L": "Location",
            "G": "Geo-political entity",
            "C": "Concept",
            "M": "Medical",
            "F": "Facility",
            "N": "Natural features",
            "U": "Unknown / Uncategorized",
            "S": "Event",
            "W": "Work of art",
            "B": "Brand",
            "H": "Historical event/person",
            "K": "Book",
            "V": "Video",
            "J": "Journal",
            "R": "Research/Scientific term",
            "A": "Astronomical object",
            "I": "Invention/Technology"
        }
    
    def get_wikidata_info(self, wikidata_ids: List[str]) -> Dict[str, Dict]:
        """
        Retrieve information from Wikidata API for a list of entity IDs.
        
        Args:
            wikidata_ids: List of Wikidata IDs (e.g., "Q123")
            
        Returns:
            Dictionary mapping Wikidata IDs to entity information
        """
        import requests
        
        # Chunk requests to avoid overly long URLs
        chunk_size = 50
        all_results = {}
        
        for i in range(0, len(wikidata_ids), chunk_size):
            chunk = wikidata_ids[i:i+chunk_size]
            ids_param = "|".join(chunk)
            
            url = f"https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbgetentities",
                "ids": ids_param,
                "format": "json",
                "languages": "en",
                "props": "labels|descriptions|sitelinks"
            }
            
            try:
                response = requests.get(url, params=params)
                data = response.json()
                
                if "entities" in data:
                    for entity_id, entity_data in data["entities"].items():
                        label = entity_data.get("labels", {}).get("en", {}).get("value", "")
                        description = entity_data.get("descriptions", {}).get("en", {}).get("value", "")
                        wikipedia = entity_data.get("sitelinks", {}).get("enwiki", {}).get("title", "")
                        
                        all_results[entity_id] = {
                            "label": label,
                            "description": description,
                            "wikipedia": wikipedia
                        }
            except Exception as e:
                logger.error(f"Error fetching Wikidata info: {e}")
                
        return all_results

    def enrich_entities_with_wikidata(self, entities_df: pd.DataFrame, wikidata_column: str = "WikidataId") -> pd.DataFrame:
        """
        Enrich entities DataFrame with additional information from Wikidata.
        
        Args:
            entities_df: DataFrame containing entities with a WikidataId column
            wikidata_column: Name of the column containing Wikidata IDs
            
        Returns:
            Enriched DataFrame with Wikidata information
        """
        if wikidata_column not in entities_df.columns:
            logger.error(f"Column {wikidata_column} not found in DataFrame")
            return entities_df
            
        # Extract unique Wikidata IDs
        unique_ids = entities_df[wikidata_column].dropna().unique().tolist()
        logger.info(f"Fetching Wikidata information for {len(unique_ids)} unique entities")
        
        # Get Wikidata information
        wikidata_info = self.get_wikidata_info(unique_ids)
        
        # Create mapping DataFrames
        wikidata_df = pd.DataFrame.from_dict(wikidata_info, orient='index')
        wikidata_df.index.name = wikidata_column
        wikidata_df = wikidata_df.reset_index()
        
        # Merge with original DataFrame
        enriched_df = entities_df.merge(wikidata_df, on=wikidata_column, how='left')
        
        return enriched_df


def preprocess_news_text(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess news text data for NLP tasks.
    
    Args:
        news_df: DataFrame containing news articles.
        
    Returns:
        DataFrame with additional preprocessed text columns.
    """
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Create a copy of the dataframe to avoid modifying the original
    df = news_df.copy()
    
    # Combine title and abstract
    df['full_text'] = df['title'] + ' ' + df['abstract'].fillna('')
    
    # Text cleaning - remove special characters
    df['cleaned_text'] = df['full_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    
    # Convert to lowercase
    df['title_lower'] = df['title'].str.lower()
    df['abstract_lower'] = df['abstract'].fillna('').str.lower()
    df['full_text_lower'] = df['full_text'].str.lower()
    df['cleaned_text_lower'] = df['cleaned_text'].str.lower()
    
    # Calculate word counts
    df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))
    df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    df['total_word_count'] = df['title_word_count'] + df['abstract_word_count']
    
    # Advanced preprocessing (tokenization, stopword removal, lemmatization) - optional
    try:
        # Ensure NLTK data is available
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        # Apply tokenization
        df['tokenized_text'] = df['cleaned_text_lower'].apply(word_tokenize)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        df['filtered_tokens'] = df['tokenized_text'].apply(
            lambda tokens: [token for token in tokens if token not in stop_words]
        )
        
        # Lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        df['lemmatized_tokens'] = df['filtered_tokens'].apply(
            lambda tokens: [lemmatizer.lemmatize(token) for token in tokens]
        )
        
        # Join tokens into a string
        df['processed_text'] = df['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))
        
    except Exception as e:
        logger.warning(f"Advanced text preprocessing failed: {e}")
        logger.warning("Skipping tokenization, stopword removal, and lemmatization")
    
    return df


def calculate_reading_level(text_df: pd.DataFrame, text_column: str = 'full_text') -> pd.DataFrame:
    """
    Calculate reading level metrics for text data.
    
    Args:
        text_df: DataFrame containing text data
        text_column: Column containing the text to analyze
        
    Returns:
        DataFrame with reading level metrics added
    """
    try:
        import textstat
    except ImportError:
        logger.error("textstat package required for reading level calculation")
        logger.error("Install with: pip install textstat")
        return text_df
    
    df = text_df.copy()
    
    # Apply reading level metrics to non-empty text
    df['flesch_kincaid_grade'] = df[text_column].apply(
        lambda x: textstat.flesch_kincaid_grade(x) if pd.notna(x) and x else None
    )
    df['flesch_reading_ease'] = df[text_column].apply(
        lambda x: textstat.flesch_reading_ease(x) if pd.notna(x) and x else None
    )
    df['smog_index'] = df[text_column].apply(
        lambda x: textstat.smog_index(x) if pd.notna(x) and x else None
    )
    df['dale_chall_readability'] = df[text_column].apply(
        lambda x: textstat.dale_chall_readability_score(x) if pd.notna(x) and x else None
    )
    
    # Categorize reading levels
    def categorize_reading_level(grade):
        if pd.isna(grade):
            return "Unknown"
        elif grade <= 5:
            return "Elementary"
        elif grade <= 8:
            return "Middle School"
        elif grade <= 12:
            return "High School"
        else:
            return "College Level"
            
    df['reading_level_category'] = df['flesch_kincaid_grade'].apply(categorize_reading_level)
    
    return df


def analyze_sentiment(text_df: pd.DataFrame, text_column: str = 'full_text', method: str = 'textblob') -> pd.DataFrame:
    """
    Analyze sentiment in text data.
    
    Args:
        text_df: DataFrame containing text data
        text_column: Column containing the text to analyze
        method: Sentiment analysis method ('textblob', 'vader', or 'transformer')
        
    Returns:
        DataFrame with sentiment analysis added
    """
    df = text_df.copy()
    
    if method == 'textblob':
        try:
            from textblob import TextBlob
        except ImportError:
            logger.error("textblob package required for sentiment analysis")
            logger.error("Install with: pip install textblob")
            return text_df
            
        # Calculate sentiment polarity and subjectivity
        df['sentiment_polarity'] = df[text_column].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) and x else None
        )
        df['sentiment_subjectivity'] = df[text_column].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) and x else None
        )
        
        # Calculate objectivity (inverse of subjectivity)
        df['objectivity'] = df['sentiment_subjectivity'].apply(
            lambda x: 1 - x if pd.notna(x) else None
        )
        
        # Categorize sentiment
        def categorize_sentiment(polarity):
            if pd.isna(polarity):
                return "Unknown"
            elif polarity > 0.1:
                return "Positive"
            elif polarity < -0.1:
                return "Negative"
            else:
                return "Neutral"
                
        df['sentiment_category'] = df['sentiment_polarity'].apply(categorize_sentiment)
        
    elif method == 'vader':
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
        except ImportError:
            logger.error("nltk package with vader required for sentiment analysis")
            logger.error("Install with: pip install nltk")
            logger.error("Then run: nltk.download('vader_lexicon')")
            return text_df
            
        # Initialize VADER
        sid = SentimentIntensityAnalyzer()
        
        # Calculate VADER scores
        def get_vader_scores(text):
            if pd.isna(text) or text == "":
                return pd.Series([None, None, None, None])
                
            scores = sid.polarity_scores(str(text))
            return pd.Series([
                scores['neg'],
                scores['neu'],
                scores['pos'],
                scores['compound']
            ])
            
        vader_scores = df[text_column].apply(get_vader_scores)
        df[['vader_negative', 'vader_neutral', 'vader_positive', 'vader_compound']] = vader_scores
        
        # Categorize VADER sentiment
        def categorize_vader_sentiment(compound):
            if pd.isna(compound):
                return "Unknown"
            elif compound >= 0.05:
                return "Positive"
            elif compound <= -0.05:
                return "Negative"
            else:
                return "Neutral"
                
        df['vader_sentiment_category'] = df['vader_compound'].apply(categorize_vader_sentiment)
        
    elif method == 'transformer':
        try:
            from transformers import pipeline
        except ImportError:
            logger.error("transformers package required for sentiment analysis")
            logger.error("Install with: pip install transformers torch")
            return text_df
            
        # Load the sentiment analysis pipeline
        sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        
        # Process in batches to avoid memory issues
        batch_size = 100
        all_sentiments = []
        
        for i in range(0, len(df), batch_size):
            batch = df[text_column].iloc[i:i+batch_size].fillna("").tolist()
            
            # Skip empty texts to avoid errors
            sentiments = []
            for text in batch:
                if text.strip():
                    try:
                        result = sentiment_pipeline(text)[0]
                        sentiments.append(result)
                    except Exception as e:
                        logger.warning(f"Error analyzing text: {e}")
                        sentiments.append({'label': 'LABEL_1', 'score': 0.5})  # Neutral fallback
                else:
                    sentiments.append({'label': 'LABEL_1', 'score': 0.5})  # Neutral for empty text
                    
            all_sentiments.extend(sentiments)
        
        # Extract labels and scores
        transformer_labels = [s['label'] for s in all_sentiments]
        transformer_scores = [s['score'] for s in all_sentiments]
        
        # Map labels to readable categories
        label_map = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral',
            'LABEL_2': 'Positive'
        }
        
        transformer_categories = [label_map.get(label, 'Neutral') for label in transformer_labels]
        
        # Add to dataframe
        df['transformer_sentiment'] = transformer_categories
        df['transformer_confidence'] = transformer_scores
    
    return df


if __name__ == "__main__":
    # Example usage
    loader = MINDDatasetLoader()
    news_df = loader.load_news()
    behaviors_df = loader.load_behaviors()
    
    print(f"Loaded {len(news_df)} news articles and {len(behaviors_df)} behavior records")