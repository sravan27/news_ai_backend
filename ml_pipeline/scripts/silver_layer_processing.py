#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Silver layer processing script that transforms bronze layer data
into feature-engineered data ready for modeling.
"""

import os
import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import time
from pathlib import Path
from tqdm import tqdm
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set up paths - adjust these to match your actual paths
project_root = Path("/Users/sravansridhar/Documents/news_ai")
BRONZE_DATA_PATH = project_root / "ml_pipeline" / "data" / "bronze"
SILVER_DATA_PATH = project_root / "ml_pipeline" / "data" / "silver"

# Create necessary directories
os.makedirs(SILVER_DATA_PATH, exist_ok=True)

def load_bronze_data(split="train"):
    """Load bronze layer data for a specific split."""
    behaviors_path = BRONZE_DATA_PATH / f"behaviors_{split}.parquet"
    news_path = BRONZE_DATA_PATH / f"news_{split}.parquet"
    
    # Load behaviors data
    behaviors_df = pd.read_parquet(behaviors_path)
    logger.info(f"Loaded {len(behaviors_df)} behavior records from {behaviors_path}")
    
    # Load news data
    news_df = pd.read_parquet(news_path)
    logger.info(f"Loaded {len(news_df)} news articles from {news_path}")
    
    return behaviors_df, news_df

def process_news_features(news_df):
    """Generate features from news content."""
    start_time = time.time()
    logger.info("Processing news features...")
    
    # Create a working copy
    df = news_df.copy()
    
    # 1. Text length features
    df['title_length'] = df['title'].str.len()
    df['abstract_length'] = df['abstract'].fillna('').str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    df['abstract_word_count'] = df['abstract'].fillna('').str.split().str.len()
    
    # 2. Entity count features
    df['title_entity_count'] = df['title_entities'].apply(len)
    df['abstract_entity_count'] = df['abstract_entities'].apply(len)
    df['total_entity_count'] = df['title_entity_count'] + df['abstract_entity_count']
    
    # 3. Category encoding (one-hot)
    category_encoder = OneHotEncoder(sparse_output=False)
    category_onehot = category_encoder.fit_transform(df[['category']])
    category_cols = [f'category_{cat}' for cat in category_encoder.categories_[0]]
    category_df = pd.DataFrame(category_onehot, columns=category_cols, index=df.index)
    
    # 4. Subcategory encoding
    # First, create a label encoder
    subcategory_encoder = LabelEncoder()
    df['subcategory_encoded'] = subcategory_encoder.fit_transform(df['subcategory'])
    
    # Add subcategory-related features
    subcategory_counts = df['subcategory'].value_counts()
    df['subcategory_frequency'] = df['subcategory'].map(subcategory_counts)
    
    # 5. Combine with original data
    result_df = pd.concat([df, category_df], axis=1)
    
    # Store encoders metadata for later use
    encoder_metadata = {
        'categories': category_encoder.categories_[0].tolist(),
        'subcategories': subcategory_encoder.classes_.tolist()
    }
    
    duration = time.time() - start_time
    logger.info(f"Processed news features in {duration:.2f} seconds")
    
    return result_df, encoder_metadata

def process_user_features(behaviors_df, news_features_df):
    """Generate user-based features from behaviors data."""
    start_time = time.time()
    logger.info("Processing user features...")
    
    # Create a working copy
    df = behaviors_df.copy()
    
    # 1. Extract basic user activity metrics
    user_stats = df.groupby('user_id').agg(
        history_count=('history_length', 'mean'),
        impressions_count=('impressions_count', 'mean'),
        total_actions=('impression_id', 'count')
    )
    
    # 2. Calculate category preferences for each user
    user_categories = {}
    
    # Process each user's history to get category preferences
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing user histories"):
        user_id = row['user_id']
        history_news_ids = row['history']
        
        if user_id not in user_categories:
            user_categories[user_id] = {}
        
        # Get categories for history items
        for news_id in history_news_ids:
            if news_id in news_features_df.index:
                category = news_features_df.loc[news_id, 'category']
                if category not in user_categories[user_id]:
                    user_categories[user_id][category] = 0
                user_categories[user_id][category] += 1
    
    # Convert category preferences to DataFrame
    categories = set()
    for user_prefs in user_categories.values():
        categories.update(user_prefs.keys())
    
    # Create DataFrame with user category preferences
    user_category_df = pd.DataFrame(index=user_categories.keys(), columns=list(categories))
    user_category_df = user_category_df.fillna(0)
    
    for user_id, prefs in user_categories.items():
        for category, count in prefs.items():
            user_category_df.loc[user_id, category] = count
    
    # Normalize to get preferences (proportions)
    for user_id in user_category_df.index:
        total = user_category_df.loc[user_id].sum()
        if total > 0:  # Avoid division by zero
            user_category_df.loc[user_id] = user_category_df.loc[user_id] / total
    
    # Rename columns to indicate they are preferences
    user_category_df.columns = [f'pref_{col}' for col in user_category_df.columns]
    
    # 3. Merge with user statistics
    user_features_df = user_stats.join(user_category_df)
    
    duration = time.time() - start_time
    logger.info(f"Processed user features in {duration:.2f} seconds")
    
    return user_features_df

def create_interaction_features(behaviors_df, user_features_df, news_features_df):
    """Create features for user-news interactions."""
    start_time = time.time()
    logger.info("Creating interaction features...")
    
    # Create lists to store interaction data
    interactions = []
    
    # Process each behavior record
    for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), desc="Processing interactions"):
        user_id = row['user_id']
        
        # Process each impression
        for impression in row['impressions']:
            news_id = impression['news_id']
            
            # For test data, clicked might be None
            clicked = impression.get('clicked')
            
            # Create interaction record
            interaction = {
                'user_id': user_id,
                'news_id': news_id,
                'impression_id': row['impression_id'],
                'clicked': clicked
            }
            
            interactions.append(interaction)
    
    # Create DataFrame from interactions
    interactions_df = pd.DataFrame(interactions)
    
    # Join with user features
    interactions_df = interactions_df.merge(user_features_df, on='user_id', how='left')
    
    # Join with news features
    interactions_df = interactions_df.merge(news_features_df, on='news_id', how='left')
    
    # Create additional interaction features
    # 1. Match between user's preferred category and news category
    categories = [col for col in news_features_df.columns if col.startswith('category_')]
    for cat in categories:
        # Get the pure category name (remove 'category_' prefix)
        cat_name = cat[9:]
        pref_col = f'pref_{cat_name}'
        
        if pref_col in interactions_df.columns:
            # Calculate the preference match score
            interactions_df[f'match_{cat_name}'] = interactions_df[cat] * interactions_df[pref_col]
    
    duration = time.time() - start_time
    logger.info(f"Created interaction features in {duration:.2f} seconds")
    
    return interactions_df

def save_to_silver(df, name, split="train"):
    """Save DataFrame to silver layer in Parquet format."""
    output_path = SILVER_DATA_PATH / f"{name}_{split}.parquet"
    
    # Convert to PyArrow Table
    table = pa.Table.from_pandas(df)
    
    # Write to Parquet with compression
    pq.write_table(table, output_path, compression='snappy')
    
    logger.info(f"Saved {len(df)} records to {output_path}")
    return output_path

def process_split(split):
    """Process a single split (train, dev, or test)."""
    logger.info(f"Processing {split} split...")
    
    # Load bronze data
    behaviors_df, news_df = load_bronze_data(split)
    
    # Process news features
    news_features_df, encoder_metadata = process_news_features(news_df)
    
    # Save encoders metadata
    encoders_path = SILVER_DATA_PATH / f"encoders_{split}.json"
    with open(encoders_path, 'w') as f:
        json.dump(encoder_metadata, f)
    logger.info(f"Saved encoders metadata to {encoders_path}")
    
    # Process user features
    news_features_indexed = news_features_df.set_index('news_id')
    user_features_df = process_user_features(behaviors_df, news_features_indexed)
    
    # Create interaction features
    news_features_df_reset = news_features_df.reset_index()  # Reset index for merge
    interactions_df = create_interaction_features(
        behaviors_df, 
        user_features_df, 
        news_features_df_reset
    )
    
    # Save to silver layer
    save_to_silver(news_features_df, "news_features", split)
    save_to_silver(user_features_df, "user_features", split)
    save_to_silver(interactions_df, "interactions", split)
    
    logger.info(f"Completed processing {split} split")
    return True

def main():
    """Process all splits of the dataset."""
    logger.info("Starting silver layer processing...")
    
    success = True
    for split in ["train", "dev", "test"]:
        try:
            result = process_split(split)
            if not result:
                logger.error(f"Failed to process {split} split")
                success = False
        except Exception as e:
            logger.error(f"Error processing {split} split: {str(e)}")
            success = False
    
    if success:
        logger.info("Silver layer processing completed successfully!")
    else:
        logger.error("Silver layer processing completed with errors.")
    
    return success

if __name__ == "__main__":
    main()