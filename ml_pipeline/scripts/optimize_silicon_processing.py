#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Silicon Layer Processing for News AI Pipeline

This script provides performance optimizations for the Silicon layer processing
in the News AI ML pipeline. It uses Dask for distributed processing and provides
fast implementations of the specialized metrics models.
"""

import os
import sys
import time
import json
import logging
import argparse
import warnings
from pathlib import Path
import multiprocessing

# Add parent directory to Python path to resolve imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import basic dependencies with error handling
try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
except ImportError as e:
    print(f"Error importing basic data libraries: {e}")
    print("Please install these dependencies with: pip install pandas numpy pyarrow")
    sys.exit(1)

# Import dask for distributed computing
try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster
    dask_available = True
except ImportError:
    print("WARNING: Dask not available. Using pandas fallback methods.")
    dask_available = False

# Try importing PyTorch for machine learning
try:
    import torch
    torch_available = True
except ImportError:
    print("WARNING: PyTorch not available. Will use CPU fallback methods.")
    torch_available = False

# Try importing scikit-learn for ML processing
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, mean_squared_error, mean_absolute_error
    )
    sklearn_available = True
except ImportError:
    print("WARNING: Scikit-learn not available. ML functionality limited.")
    sklearn_available = False

# Try importing progress bar
try:
    from tqdm import tqdm
except ImportError:
    # Create a simple tqdm replacement
    def tqdm(iterable, **kwargs):
        if 'total' in kwargs and kwargs.get('desc'):
            print(f"{kwargs.get('desc')}: Processing {kwargs.get('total')} items...")
        return iterable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
CONFIG_PATH = BASE_DIR / "config" / "pipeline_config.yaml"
BRONZE_PATH = BASE_DIR / "data" / "bronze"
SILVER_PATH = BASE_DIR / "data" / "silver"
SILICON_PATH = BASE_DIR / "data" / "silicon"
MODEL_PATH = BASE_DIR / "models"
DEPLOYED_MODEL_PATH = MODEL_PATH / "deployed"

# Create necessary directories
os.makedirs(SILICON_PATH, exist_ok=True)
os.makedirs(DEPLOYED_MODEL_PATH, exist_ok=True)

# Global variables for tracking performance
start_time = None
metric_timings = {}

def setup_dask_client():
    """Set up a Dask distributed client optimized for M2 Max."""
    if not dask_available:
        return None
    
    # Get the number of cores, reserving 2 for system tasks
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    # Check for Apple Silicon hardware
    is_apple_silicon = sys.platform == 'darwin' and os.uname().machine == 'arm64'
    
    if is_apple_silicon:
        print(f"🍎 Detected Apple Silicon (M-series chip)")
        # Use more memory per worker on Apple Silicon due to better unified memory
        memory_limit = '7GiB'  # Using about 7GiB per worker for a 32GB system
    else:
        memory_limit = '4GiB'  # Standard memory limit per worker
    
    try:
        # Set up a LocalCluster with optimized settings
        cluster = LocalCluster(
            n_workers=num_workers,
            threads_per_worker=2,  # Use 2 threads per worker for better balance
            memory_limit=memory_limit,
            processes=True,  # Use processes for true parallelism
            silence_logs=logging.INFO
        )
        
        # Create a client connected to the cluster
        client = Client(cluster)
        
        print(f"✅ Dask distributed client running with {num_workers} workers")
        print(f"   Dashboard link: {client.dashboard_link}")
        
        return client
    except Exception as e:
        print(f"Failed to start Dask client: {e}")
        return None

def load_silver_data():
    """Load processed data from the silver layer using Dask for efficiency."""
    logger.info("Loading silver layer data...")
    print("Loading processed data from silver layer...")
    
    # Find available silver layer files
    silver_files = {}
    expected_files = [
        'news_base.parquet',
        'text_features.parquet',
        'user_features.parquet',
        'interactions.parquet',
        'embeddings_metadata.txt'
    ]
    
    for file in expected_files:
        file_path = SILVER_PATH / file
        if file_path.exists():
            silver_files[file] = file_path
            print(f"✓ Found {file}")
        else:
            print(f"✗ Missing {file}")
    
    # Load news data with Dask if available
    if dask_available and 'news_base.parquet' in silver_files:
        try:
            print("Using Dask for efficient data loading...")
            news_df = dd.read_parquet(silver_files['news_base.parquet']).compute()
            print(f"✅ Loaded {len(news_df)} news articles with Dask")
        except Exception as e:
            print(f"Error loading with Dask: {e}")
            print("Falling back to pandas/pyarrow...")
            news_df = pd.read_parquet(silver_files['news_base.parquet'])
            print(f"✅ Loaded {len(news_df)} news articles with pandas")
    elif 'news_base.parquet' in silver_files:
        news_df = pd.read_parquet(silver_files['news_base.parquet'])
        print(f"✅ Loaded {len(news_df)} news articles")
    else:
        print("❌ No news data found in silver layer")
        news_df = None
    
    # Load text features
    if 'text_features.parquet' in silver_files:
        text_features = pd.read_parquet(silver_files['text_features.parquet'])
        print(f"✅ Loaded text features for {len(text_features)} articles")
    else:
        print("❌ No text features found in silver layer")
        text_features = None
    
    # Load user features
    if 'user_features.parquet' in silver_files:
        user_features = pd.read_parquet(silver_files['user_features.parquet'])
        print(f"✅ Loaded features for {len(user_features)} users")
    else:
        print("❌ No user features found in silver layer")
        user_features = None
    
    # Load interaction data
    if 'interactions.parquet' in silver_files:
        interactions = pd.read_parquet(silver_files['interactions.parquet'])
        print(f"✅ Loaded {len(interactions)} interactions")
    else:
        print("❌ No interactions found in silver layer")
        interactions = None
    
    # Load embeddings if available
    embeddings = {}
    try:
        if (SILVER_PATH / 'title_embeddings.npy').exists():
            embeddings['title'] = np.load(SILVER_PATH / 'title_embeddings.npy')
            embeddings['abstract'] = np.load(SILVER_PATH / 'abstract_embeddings.npy')
            embeddings['full_text'] = np.load(SILVER_PATH / 'full_text_embeddings.npy')
            print(f"✅ Loaded embeddings: {embeddings['title'].shape[0]} articles with {embeddings['title'].shape[1]} dimensions")
        else:
            print("❌ No embeddings found in silver layer")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        embeddings = None
    
    return news_df, text_features, user_features, interactions, embeddings

def prepare_features(news_df, text_features, metric_type):
    """Prepare features for a specific metric model."""
    logger.info(f"Preparing features for {metric_type} model...")
    print(f"\n📋 Preparing features for {metric_type.replace('_', ' ').title()} model")
    
    # Validate inputs
    if news_df is None:
        print("❌ No news data available. Cannot proceed.")
        return None, None
    
    # If we have text features, join with news_df
    if text_features is not None:
        print(f"Joining {len(news_df)} news articles with {len(text_features)} text features...")
        
        # Prepare for join
        if text_features.index.name == 'news_id':
            text_features = text_features.reset_index()
            
        # Perform join
        feature_df = pd.merge(news_df, text_features, on='news_id', how='inner')
        print(f"✅ Created feature dataframe with {len(feature_df)} articles and {feature_df.shape[1]} features")
    else:
        print("No text features available. Using only news data.")
        feature_df = news_df.copy()
    
    # Generate metric-specific features and labels based on metric type
    if metric_type == "political_influence":
        print("Generating political influence features...")
        
        # Safely convert category to numeric code first to avoid string processing issues
        if 'category' in feature_df.columns:
            if pd.api.types.is_categorical_dtype(feature_df['category']):
                # If it's already categorical, get the codes
                feature_df['category_code'] = feature_df['category'].cat.codes
            else:
                # Convert to categorical first
                feature_df['category_code'] = pd.Categorical(feature_df['category']).codes
                
            # Create is_politics flag - code for 'politics' needs to be determined
            # Let's set a default "politics" score
            feature_df['is_politics'] = 0
            
            # Check if we can find politics in the categories
            if not pd.api.types.is_numeric_dtype(feature_df['category']):
                # Only do this for non-numeric original categories
                if 'politics' in feature_df['category'].values:
                    politics_code = feature_df.loc[feature_df['category'] == 'politics', 'category_code'].iloc[0]
                    feature_df['is_politics'] = np.where(feature_df['category_code'] == politics_code, 1, 0)
                else:
                    # No politics category found, use news as proxy if available
                    if 'news' in feature_df['category'].values:
                        news_code = feature_df.loc[feature_df['category'] == 'news', 'category_code'].iloc[0]
                        feature_df['is_politics'] = np.where(feature_df['category_code'] == news_code, 1, 0)
        else:
            # If no category column exists, create a default
            feature_df['category_code'] = 0
            feature_df['is_politics'] = 0
        
        # Generate synthetic labels for demonstration
        # In a real system, these would come from expert annotations or derived data
        feature_df['label'] = np.where(
            feature_df['is_politics'] == 1,
            0.5 + 0.4 * np.random.random(len(feature_df)),  # Higher influence for politics (0.5-0.9)
            0.1 + 0.3 * np.random.random(len(feature_df))   # Lower influence for non-politics (0.1-0.4)
        )
    
    elif metric_type == "rhetoric_intensity":
        print("Generating rhetoric intensity features...")
        
        # Add engineered features
        # Check if category is already numeric
        if pd.api.types.is_numeric_dtype(feature_df['category']):
            # Just copy the category as category_code
            feature_df['category_code'] = feature_df['category']
        else:
            # Map string categories to numeric codes
            feature_df['category_code'] = feature_df['category'].map({
                'politics': 3,
                'entertainment': 2, 
                'sports': 1,
                'technology': 0,
                'health': 2  # Adding health, similar level as entertainment
            }).fillna(0)
        
        if 'title_length' in feature_df.columns:
            feature_df['title_length_norm'] = feature_df['title_length'] / feature_df['title_length'].max()
        
        # Generate synthetic labels
        if pd.api.types.is_numeric_dtype(feature_df['category']):
            # Use category_code for logic when category is numeric
            feature_df['label'] = np.where(
                feature_df['category_code'] >= 2,  # Politics and entertainment/health have higher codes
                0.5 + 0.5 * np.random.random(len(feature_df)),  # Higher rhetoric (0.5-1.0)
                0.1 + 0.4 * np.random.random(len(feature_df))   # Lower rhetoric (0.1-0.5)
            )
        else:
            # Use string category when available
            feature_df['label'] = np.where(
                feature_df['category'].isin(['politics', 'entertainment', 'health']),
                0.5 + 0.5 * np.random.random(len(feature_df)),  # Higher rhetoric (0.5-1.0)
                0.1 + 0.4 * np.random.random(len(feature_df))   # Lower rhetoric (0.1-0.5)
            )
    
    elif metric_type == "information_depth":
        print("Generating information depth features...")
        
        # Add engineered features related to information depth
        if 'abstract_length' in feature_df.columns:
            feature_df['abstract_length_norm'] = feature_df['abstract_length'] / feature_df['abstract_length'].max()
            
        if 'title_length' in feature_df.columns:
            feature_df['title_length_norm'] = feature_df['title_length'] / feature_df['title_length'].max()
        
        # Generate synthetic labels - higher depth for longer articles with more entities
        norm_abstract_length = feature_df['abstract_length'] / 300 if 'abstract_length' in feature_df.columns else 0.5
        if 'title_entity_count' in feature_df.columns:
            norm_entity_count = feature_df['title_entity_count'] / 10
        else:
            norm_entity_count = pd.Series(0.5, index=feature_df.index)
            
        feature_df['label'] = 0.4 * norm_abstract_length + 0.3 * norm_entity_count + 0.3 * np.random.random(len(feature_df))
        feature_df['label'] = feature_df['label'].clip(0, 1)  # Clip to [0,1]
        
    elif metric_type == "sentiment":
        print("Generating sentiment features...")
        
        # Add sentiment-related features
        if pd.api.types.is_numeric_dtype(feature_df['category']):
            # Create a default mapping for numeric categories
            sentiment_map = {
                0: 0.5,  # Default/unknown
                1: 0.55, # Sports equivalent 
                2: 0.6,  # Entertainment equivalent
                3: 0.4,  # Politics equivalent
                4: 0.5,  # Technology equivalent
                5: 0.6   # Health - positive sentiment similar to entertainment
            }
            # Apply mapping or default to neutral (0.5)
            feature_df['category_sentiment'] = feature_df['category'].map(
                lambda x: sentiment_map.get(x, 0.5)
            )
        else:
            # Use string category mapping
            feature_df['category_sentiment'] = feature_df['category'].map({
                'entertainment': 0.6,  # Entertainment tends more positive
                'sports': 0.55,        # Sports slightly positive
                'politics': 0.4,       # Politics tends negative
                'technology': 0.5,     # Technology neutral
                'health': 0.6          # Health tends positive
            }).fillna(0.5)
        
        # Generate synthetic sentiment scores
        np.random.seed(42)  # For reproducibility
        random_variation = np.random.normal(0, 0.15, len(feature_df))
        feature_df['label'] = (feature_df['category_sentiment'] + random_variation).clip(0, 1)
    
    else:
        print(f"Unknown metric type: {metric_type}")
        return None, None
    
    # Split features and target
    X = feature_df.drop(['news_id', 'label'], axis=1, errors='ignore')
    y = feature_df['label']
    
    # Remove text and categorical columns that aren't useful for modeling
    text_columns = ['title', 'abstract', 'url', 'body', 'category']
    X = X.drop([col for col in text_columns if col in X.columns], axis=1, errors='ignore')
    
    # Print column data types for debugging
    print("Column data types before conversion:")
    for col, dtype in X.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # Convert all object/string columns to numeric
    for col in X.select_dtypes(include=['object']).columns:
        try:
            print(f"Converting column {col} to numeric")
            # If the column has any NaN values, fillna first
            if X[col].isna().any():
                X[col] = X[col].fillna('unknown')
            X[col] = pd.Categorical(X[col]).codes
        except Exception as e:
            # If conversion fails, drop the column
            print(f"Dropping column {col} because it couldn't be converted to numeric: {str(e)}")
            X = X.drop(col, axis=1)
            
    # Convert categorical columns to numeric codes
    for col in X.select_dtypes(include=['category']).columns:
        try:
            print(f"Converting categorical column {col} to numeric codes")
            X[col] = X[col].cat.codes
        except Exception as e:
            print(f"Error converting categorical column {col}: {str(e)}")
            X = X.drop(col, axis=1)
            
    # Verify no string columns remain
    if len(X.select_dtypes(include=['object', 'category']).columns) > 0:
        string_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Warning: Non-numeric columns still present, dropping: {string_columns}")
        # Force drop any remaining string columns
        X = X.drop(string_columns, axis=1)
    
    print(f"✅ Prepared {X.shape[1]} features for {len(X)} articles")
    
    # Save features for future use
    os.makedirs(SILICON_PATH / metric_type, exist_ok=True)
    feature_df.to_parquet(SILICON_PATH / metric_type / "features.parquet")
    print(f"✅ Saved features to {SILICON_PATH / metric_type / 'features.parquet'}")
    
    return X, y

def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage by downcasting numeric columns and category types."""
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        
        elif col_type == object:
            # Check if column contains list values before trying nunique()
            try:
                df[col] = pd.Categorical(df[col])
            except:
                pass
    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    logger.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df

def train_model_with_dask(X, y, metric_type):
    """Train a model using Dask for distributed processing."""
    if not dask_available or not sklearn_available:
        print("Dask or scikit-learn not available. Using pandas fallback.")
        return train_model(X, y, metric_type)
    
    print(f"\n📊 Training {metric_type.replace('_', ' ').title()} model with Dask")
    start_time = time.time()
    
    # First check for any remaining non-numeric columns and handle them
    print("Checking for and handling any remaining non-numeric columns before Dask processing...")
    string_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    if string_cols:
        print(f"Found string columns that need to be removed or converted: {string_cols}")
        for col in string_cols:
            print(f"Dropping text column that can't be used for modeling: {col}")
            X = X.drop(col, axis=1)
    
    # Convert to Dask dataframes
    try:
        # Attempt to ensure X is numeric-only
        if 'text' in X.columns:
            print(f"Dropping text column before Dask processing")
            X = X.drop('text', axis=1, errors='ignore')
            
        for col in X.select_dtypes(include=['object', 'string']).columns:
            print(f"Dropping non-numeric column before Dask processing: {col}")
            X = X.drop(col, axis=1, errors='ignore')
            
        X_dask = dd.from_pandas(X, npartitions=multiprocessing.cpu_count())
        y_dask = dd.from_pandas(y, npartitions=multiprocessing.cpu_count())
    except Exception as e:
        print(f"Error converting to Dask dataframes: {e}")
        print("Falling back to pandas implementation...")
        return train_model(X, y, metric_type)
    
    # Determine if this is a classification or regression task
    unique_labels = len(np.unique(y))
    is_classification = unique_labels <= 10  # Assume classification if fewer than 10 unique values
    
    # Split data - need to compute Dask DataFrames
    try:
        X_np = X_dask.compute().values
        y_np = y_dask.compute().values
    except Exception as e:
        print(f"Error computing Dask arrays: {e}")
        print("This might be due to non-numeric data. Falling back to pandas implementation...")
        return train_model(X, y, metric_type)
        
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
    
    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    with open(SILICON_PATH / metric_type / "scaler.pkl", "wb") as f:
        import pickle
        pickle.dump(scaler, f)
    
    # Train the appropriate model
    if is_classification:
        # Set up a hyperparameter-optimized model
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=None, 
            min_samples_split=2,
            n_jobs=-1,  # Use all available cores
            random_state=42
        )
    else:
        # Regression model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,  # Use all available cores
            random_state=42
        )
    
    print(f"Training {type(model).__name__}...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    if is_classification:
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            y_pred_proba = model.predict_proba(X_test_scaled)
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            metrics = {
                'accuracy': accuracy, 
                'auc': auc
            }
            print(f"Model metrics - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        except:
            metrics = {'accuracy': accuracy}
            print(f"Model metrics - Accuracy: {accuracy:.4f}")
    else:
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        metrics = {'mse': mse, 'mae': mae}
        print(f"Model metrics - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # Save the trained model
    with open(SILICON_PATH / metric_type / "model.pkl", "wb") as f:
        import pickle
        pickle.dump(model, f)
    
    # Save model metadata
    model_metadata = {
        'model_type': type(model).__name__,
        'feature_columns': X.columns.tolist(),
        'num_features': X.shape[1],
        'metrics': metrics,
        'training_time': time.time() - start_time,
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'is_classification': is_classification
    }
    
    with open(SILICON_PATH / metric_type / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"✅ {metric_type.replace('_', ' ').title()} model training completed in {time.time() - start_time:.2f} seconds")
    print(f"✅ Model and metadata saved to {SILICON_PATH / metric_type}")
    
    return model, metrics

def _convert_numpy_types(obj):
    """Helper function to convert numpy types to python native types for JSON serialization."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj

def train_model(X, y, metric_type):
    """Train a model using standard scikit-learn."""
    print(f"\n📊 Training {metric_type.replace('_', ' ').title()} model")
    start_time = time.time()
    
    # Check if scikit-learn is available
    if not sklearn_available:
        print("❌ scikit-learn not available. Cannot train model.")
        return None, {}
    
    # Additional safety check - handle any remaining text/string columns
    # that might cause numpy conversion issues
    if len(X.select_dtypes(include=['object', 'string']).columns) > 0:
        for col in X.select_dtypes(include=['object', 'string']).columns:
            print(f"Dropping string column before model training: {col}")
            X = X.drop(col, axis=1)
    
    # Determine if this is a classification or regression task
    unique_labels = len(np.unique(y))
    is_classification = unique_labels <= 10  # Assume classification if fewer than 10 unique values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    with open(SILICON_PATH / metric_type / "scaler.pkl", "wb") as f:
        import pickle
        pickle.dump(scaler, f)
    
    # Train the appropriate model
    if is_classification:
        # Binary classification model
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=None, 
            min_samples_split=2,
            n_jobs=-1,  # Use all available cores
            random_state=42
        )
    else:
        # Regression model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,  # Use all available cores
            random_state=42
        )
    
    print(f"Training {type(model).__name__}...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    if is_classification:
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            y_pred_proba = model.predict_proba(X_test_scaled)
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            metrics = {
                'accuracy': accuracy, 
                'auc': auc
            }
            print(f"Model metrics - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        except:
            metrics = {'accuracy': accuracy}
            print(f"Model metrics - Accuracy: {accuracy:.4f}")
    else:
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        metrics = {'mse': mse, 'mae': mae}
        print(f"Model metrics - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # Save the trained model
    with open(SILICON_PATH / metric_type / "model.pkl", "wb") as f:
        import pickle
        pickle.dump(model, f)
    
    # Save model metadata
    model_metadata = {
        'model_type': type(model).__name__,
        'feature_columns': X.columns.tolist(),
        'num_features': X.shape[1],
        'metrics': metrics,
        'training_time': time.time() - start_time,
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'is_classification': is_classification
    }
    
    with open(SILICON_PATH / metric_type / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"✅ {metric_type.replace('_', ' ').title()} model training completed in {time.time() - start_time:.2f} seconds")
    print(f"✅ Model and metadata saved to {SILICON_PATH / metric_type}")
    
    return model, metrics

def train_neural_network_model(X, y, metric_type):
    """Train a neural network model using PyTorch with MPS acceleration."""
    if not torch_available:
        print("PyTorch not available. Skipping neural network training.")
        return None, {}
    
    print(f"\n🧠 Training neural network for {metric_type.replace('_', ' ').title()}")
    start_time = time.time()
    
    # Additional safety check - handle any remaining text/string columns
    # that might cause numpy conversion issues
    if len(X.select_dtypes(include=['object', 'string']).columns) > 0:
        for col in X.select_dtypes(include=['object', 'string']).columns:
            print(f"Dropping string column before neural network training: {col}")
            X = X.drop(col, axis=1)
    
    # Determine if this is a classification or regression task
    unique_labels = len(np.unique(y))
    is_classification = unique_labels <= 10  # Assume classification if fewer than 10 unique values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    with open(SILICON_PATH / metric_type / "nn_scaler.pkl", "wb") as f:
        import pickle
        pickle.dump(scaler, f)
    
    # Convert to PyTorch tensors
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else 
                          "cpu")
    print(f"Using device: {device}")
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    if is_classification:
        y_train_tensor = torch.FloatTensor(y_train.values.astype(np.float32)).reshape(-1, 1).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values.astype(np.float32)).reshape(-1, 1).to(device)
        y_test_tensor = torch.FloatTensor(y_test.values.astype(np.float32)).reshape(-1, 1).to(device)
    else:
        y_train_tensor = torch.FloatTensor(y_train.values.astype(np.float32)).reshape(-1, 1).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values.astype(np.float32)).reshape(-1, 1).to(device)
        y_test_tensor = torch.FloatTensor(y_test.values.astype(np.float32)).reshape(-1, 1).to(device)
    
    # Create dataset and dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Define neural network
    class MetricNet(torch.nn.Module):
        def __init__(self, input_size):
            super(MetricNet, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, 128)
            self.bn1 = torch.nn.BatchNorm1d(128)
            self.dropout1 = torch.nn.Dropout(0.3)
            self.fc2 = torch.nn.Linear(128, 64)
            self.bn2 = torch.nn.BatchNorm1d(64)
            self.dropout2 = torch.nn.Dropout(0.3)
            self.fc3 = torch.nn.Linear(64, 32)
            self.bn3 = torch.nn.BatchNorm1d(32)
            self.fc4 = torch.nn.Linear(32, 1)
            
        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout1(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout2(x)
            x = torch.relu(self.bn3(self.fc3(x)))
            
            if is_classification:
                x = torch.sigmoid(self.fc4(x))  # For binary classification
            else:
                x = self.fc4(x)  # For regression
            
            return x
    
    # Initialize model
    input_size = X_train_scaled.shape[1]
    nn_model = MetricNet(input_size).to(device)
    
    if is_classification:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 50
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stopping_rounds = 5
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        nn_model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = nn_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Calculate validation loss
        nn_model.eval()
        with torch.no_grad():
            val_outputs = nn_model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        # Track losses
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss.item())
        
        # Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            # Save the best model
            torch.save(nn_model.state_dict(), SILICON_PATH / metric_type / "nn_model.pt")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_rounds:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}')
    
    # Load the best model for evaluation
    nn_model.load_state_dict(torch.load(SILICON_PATH / metric_type / "nn_model.pt"))
    
    # Evaluate the model
    nn_model.eval()
    with torch.no_grad():
        test_outputs = nn_model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        
        # Convert tensors to numpy for evaluation
        test_preds = test_outputs.cpu().numpy()
        test_true = y_test_tensor.cpu().numpy()
        
        if is_classification:
            # Convert probabilities to class labels
            test_preds_class = (test_preds > 0.5).astype(int)
            accuracy = accuracy_score(test_true, test_preds_class)
            
            try:
                auc = roc_auc_score(test_true, test_preds)
                metrics = {'accuracy': accuracy, 'auc': auc, 'test_loss': test_loss.item()}
                print(f"Neural network metrics - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            except:
                metrics = {'accuracy': accuracy, 'test_loss': test_loss.item()}
                print(f"Neural network metrics - Accuracy: {accuracy:.4f}")
        else:
            mse = mean_squared_error(test_true, test_preds)
            mae = mean_absolute_error(test_true, test_preds)
            metrics = {'mse': mse, 'mae': mae, 'test_loss': test_loss.item()}
            print(f"Neural network metrics - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # Save model metadata - use helper function to convert numpy types
    model_metadata = {
        'model_type': 'PyTorch Neural Network',
        'architecture': 'FC: input->128->64->32->1',
        'feature_columns': X.columns.tolist(),
        'num_features': X.shape[1],
        'metrics': _convert_numpy_types(metrics),
        'training_time': float(time.time() - start_time),
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'is_classification': is_classification,
        'device': str(device),
        'epochs_trained': int(epoch + 1),
        'best_val_loss': float(best_val_loss)
    }
    
    with open(SILICON_PATH / metric_type / "nn_model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)
    
    # Save feature names
    with open(SILICON_PATH / metric_type / "feature_names.json", "w") as f:
        json.dump({'feature_names': X.columns.tolist()}, f)
    
    print(f"✅ Neural network training completed in {time.time() - start_time:.2f} seconds")
    print(f"✅ Model and metadata saved to {SILICON_PATH / metric_type}")
    
    return nn_model, metrics

def save_model_for_deployment(metric_type):
    """Copy trained models to the deployment directory."""
    print(f"\n📦 Preparing {metric_type.replace('_', ' ').title()} model for deployment")
    
    # Create deployment directory
    deploy_dir = DEPLOYED_MODEL_PATH / metric_type
    os.makedirs(deploy_dir, exist_ok=True)
    
    # Check if models exist
    model_files = [
        (SILICON_PATH / metric_type / "model.pkl", deploy_dir / "model.pkl"),
        (SILICON_PATH / metric_type / "nn_model.pt", deploy_dir / "nn_model.pt"),
        (SILICON_PATH / metric_type / "scaler.pkl", deploy_dir / "scaler.pkl"),
        (SILICON_PATH / metric_type / "nn_scaler.pkl", deploy_dir / "nn_scaler.pkl"),
        (SILICON_PATH / metric_type / "feature_names.json", deploy_dir / "feature_names.json"),
        (SILICON_PATH / metric_type / "model_metadata.json", deploy_dir / "model_metadata.json"),
        (SILICON_PATH / metric_type / "nn_model_metadata.json", deploy_dir / "nn_model_metadata.json")
    ]
    
    # Copy files
    import shutil
    for src, dst in model_files:
        if src.exists():
            shutil.copy(src, dst)
            print(f"✓ Copied {src.name} to deployment directory")
    
    # Create model card
    model_card = {
        "model_name": f"{metric_type.replace('_', ' ').title()} Predictor",
        "version": "1.0.0",
        "description": f"Model for predicting {metric_type.replace('_', ' ')} of news articles",
        "models": [],
        "deployment_date": time.strftime('%Y-%m-%d'),
        "authors": ["News AI Team"],
        "license": "Proprietary",
        "ethical_considerations": [
            "Model should be regularly monitored for bias",
            "Prediction should be used as one signal among many for decision making",
            "Human oversight is recommended for critical applications"
        ]
    }
    
    # Add standard model info if available
    if (SILICON_PATH / metric_type / "model_metadata.json").exists():
        with open(SILICON_PATH / metric_type / "model_metadata.json", "r") as f:
            model_metadata = json.load(f)
        
        # Ensure metrics are serializable
        metrics_dict = _convert_numpy_types(model_metadata.get("metrics", {}))
        
        model_card["models"].append({
            "name": "standard_model",
            "type": model_metadata.get("model_type", "RandomForest"),
            "metrics": metrics_dict,
            "training_date": model_metadata.get("training_date", "unknown")
        })
    
    # Add neural network info if available
    if (SILICON_PATH / metric_type / "nn_model_metadata.json").exists():
        with open(SILICON_PATH / metric_type / "nn_model_metadata.json", "r") as f:
            nn_metadata = json.load(f)
            
        # Ensure metrics are serializable
        nn_metrics_dict = _convert_numpy_types(nn_metadata.get("metrics", {}))
        
        model_card["models"].append({
            "name": "neural_network",
            "type": nn_metadata.get("model_type", "Neural Network"),
            "metrics": nn_metrics_dict,
            "training_date": nn_metadata.get("training_date", "unknown")
        })
    
    # Save model card
    with open(deploy_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)
    
    print(f"✅ Created model card for {metric_type.replace('_', ' ').title()} model")
    print(f"✅ Model ready for deployment at {deploy_dir}")
    
    return True

def create_inference_module(metric_type):
    """Create a Python module for model inference."""
    print(f"\n📝 Creating inference module for {metric_type.replace('_', ' ').title()} model")
    
    # Define the Python module content
    module_content = f'''"""
{metric_type.replace('_', ' ').title()} Inference Module

This module provides a simple interface for making predictions with the
{metric_type.replace('_', ' ')} model.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch

# Define paths
DEPLOY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{metric_type}")

# Load feature names
with open(os.path.join(DEPLOY_DIR, "feature_names.json"), "r") as f:
    FEATURE_NAMES = json.load(f).get("feature_names", [])

# Check for model type
has_standard_model = os.path.exists(os.path.join(DEPLOY_DIR, "model.pkl"))
has_neural_network = os.path.exists(os.path.join(DEPLOY_DIR, "nn_model.pt"))

# Set up device for PyTorch
device = None
if has_neural_network:
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                           "cuda" if torch.cuda.is_available() else 
                           "cpu")

# Define neural network class (needs to match training)
class MetricNet(torch.nn.Module):
    def __init__(self, input_size):
        super(MetricNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(128, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fc3 = torch.nn.Linear(64, 32)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.fc4 = torch.nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))  # For binary classification
        return x

def load_models():
    """Load the trained models and scalers."""
    models = {{}}
    
    # Load standard model if available
    if has_standard_model:
        try:
            with open(os.path.join(DEPLOY_DIR, "model.pkl"), "rb") as f:
                models["standard"] = pickle.load(f)
            
            with open(os.path.join(DEPLOY_DIR, "scaler.pkl"), "rb") as f:
                models["standard_scaler"] = pickle.load(f)
        except Exception as e:
            print(f"Error loading standard model: {{e}}")
    
    # Load neural network if available
    if has_neural_network:
        try:
            # Load model metadata to get input size
            with open(os.path.join(DEPLOY_DIR, "nn_model_metadata.json"), "r") as f:
                nn_metadata = json.load(f)
            
            input_size = len(FEATURE_NAMES)
            
            # Initialize neural network
            nn_model = MetricNet(input_size).to(device)
            nn_model.load_state_dict(torch.load(
                os.path.join(DEPLOY_DIR, "nn_model.pt"),
                map_location=device
            ))
            nn_model.eval()
            
            models["neural_network"] = nn_model
            
            # Load neural network scaler
            with open(os.path.join(DEPLOY_DIR, "nn_scaler.pkl"), "rb") as f:
                models["nn_scaler"] = pickle.load(f)
        except Exception as e:
            print(f"Error loading neural network model: {{e}}")
    
    return models

# Load models at module import time
MODELS = load_models()

def predict(features, model_type="standard"):
    """
    Make a prediction with the {metric_type.replace('_', ' ')} model.
    
    Parameters:
    -----------
    features : pandas.DataFrame
        Features to use for prediction. Must contain required feature columns.
    model_type : str
        Either "standard" for scikit-learn model or "neural_network" for PyTorch model.
        
    Returns:
    --------
    numpy.ndarray
        Predicted {metric_type.replace('_', ' ')} scores.
    """
    if model_type not in ["standard", "neural_network"]:
        raise ValueError(f"Unknown model type: {{model_type}}. Use 'standard' or 'neural_network'.")
    
    if model_type == "standard" and "standard" not in MODELS:
        raise ValueError("Standard model not available.")
    
    if model_type == "neural_network" and "neural_network" not in MODELS:
        raise ValueError("Neural network model not available.")
    
    # Ensure features dataframe has required columns
    missing_features = [col for col in FEATURE_NAMES if col not in features.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {{missing_features}}")
    
    # Select and order features
    features = features[FEATURE_NAMES]
    
    # Apply appropriate scaler
    if model_type == "standard":
        scaler = MODELS["standard_scaler"]
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predictions = MODELS["standard"].predict(features_scaled)
        
        # If model has predict_proba, use it
        if hasattr(MODELS["standard"], "predict_proba"):
            try:
                # Get probability of positive class
                predictions_proba = MODELS["standard"].predict_proba(features_scaled)[:, 1]
                return predictions_proba
            except:
                pass
            
        return predictions
    
    else:  # Neural network
        scaler = MODELS["nn_scaler"]
        features_scaled = scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        
        # Make prediction
        with torch.no_grad():
            predictions = MODELS["neural_network"](features_tensor).cpu().numpy()
        
        return predictions.reshape(-1)

def batch_predict(features_list, model_type="standard"):
    """
    Make predictions for a batch of feature sets.
    
    Parameters:
    -----------
    features_list : list of pandas.DataFrame
        List of feature dataframes.
    model_type : str
        Either "standard" for scikit-learn model or "neural_network" for PyTorch model.
        
    Returns:
    --------
    list
        List of predictions.
    """
    return [predict(features, model_type) for features in features_list]
'''
    
    # Create the module directory
    os.makedirs(DEPLOYED_MODEL_PATH / "inference", exist_ok=True)
    
    # Save the module
    with open(DEPLOYED_MODEL_PATH / "inference" / f"{metric_type}_inference.py", "w") as f:
        f.write(module_content)
    
    print(f"✅ Created inference module at {DEPLOYED_MODEL_PATH / 'inference' / f'{metric_type}_inference.py'}")
    
    # Create an __init__.py file if it doesn't exist
    init_path = DEPLOYED_MODEL_PATH / "inference" / "__init__.py"
    if not init_path.exists():
        with open(init_path, "w") as f:
            f.write(f'''"""
News AI Inference Modules

This package provides inference modules for News AI silicon layer models.
"""

__version__ = "1.0.0"
''')
    
    return True

def process_metric(metric_type, news_df, text_features, use_dask=True):
    """Process a specific metric from data preparation to deployment."""
    global metric_timings
    
    metric_start = time.time()
    print(f"\n{'='*80}")
    print(f"🔍 PROCESSING {metric_type.upper()} METRIC")
    print(f"{'='*80}")
    
    # Prepare features
    X, y = prepare_features(news_df, text_features, metric_type)
    if X is None or y is None:
        print(f"❌ Failed to prepare features for {metric_type}")
        return False
    
    # Train standard model
    if use_dask and dask_available:
        model, metrics = train_model_with_dask(X, y, metric_type)
    else:
        model, metrics = train_model(X, y, metric_type)
    
    # Train neural network model
    nn_model, nn_metrics = train_neural_network_model(X, y, metric_type)
    
    # Save model for deployment
    save_model_for_deployment(metric_type)
    
    # Create inference module
    create_inference_module(metric_type)
    
    metric_time = time.time() - metric_start
    metric_timings[metric_type] = metric_time
    
    print(f"✅ {metric_type.replace('_', ' ').title()} processing completed in {metric_time:.2f} seconds")
    
    return True

def main():
    """Main function to run the optimized silicon layer processing."""
    global start_time
    start_time = time.time()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimized Silicon Layer Processing for News AI")
    parser.add_argument("--no-dask", action="store_true", help="Disable Dask distributed processing")
    parser.add_argument("--metrics", nargs="+", 
                        choices=["political_influence", "rhetoric_intensity", "information_depth", "sentiment", "all"],
                        default=["all"],
                        help="Metrics to process (default: all)")
    parser.add_argument("--silver-path", type=str, help="Path to silver layer data")
    parser.add_argument("--silicon-path", type=str, help="Path to save silicon layer data")
    
    args = parser.parse_args()
    
    # Make SILVER_PATH and SILICON_PATH global since they're used at module level
    global SILVER_PATH, SILICON_PATH
    
    # Override default paths if provided
    if args.silver_path:
        SILVER_PATH = Path(args.silver_path)
    if args.silicon_path:
        SILICON_PATH = Path(args.silicon_path)
    
    # Set up Dask client if enabled - but force no-dask for now due to issues
    client = None
    use_dask = False  # Force no-dask until dask-related issues are fixed
    if not args.no_dask and use_dask:
        client = setup_dask_client()
    
    logger.info("Starting optimized silicon layer processing")
    print("\n" + "="*80)
    print("🚀 STARTING OPTIMIZED SILICON LAYER PROCESSING 🚀")
    print("="*80)
    
    # Load silver data
    news_df, text_features, user_features, interactions, embeddings = load_silver_data()
    
    # Determine which metrics to process
    if "all" in args.metrics:
        metrics_to_process = ["political_influence", "rhetoric_intensity", "information_depth", "sentiment"]
    else:
        metrics_to_process = args.metrics
    
    print(f"\nProcessing {len(metrics_to_process)} metrics: {', '.join(metrics_to_process)}")
    
    # Process each metric using Dask for distributed processing
    force_no_dask = False  # Enable Dask for distributed processing
    for metric_type in metrics_to_process:
        process_metric(metric_type, news_df, text_features, use_dask=not force_no_dask)
    
    # Create a summary report
    summary = {
        "processed_metrics": len(metrics_to_process),
        "metrics": metrics_to_process,
        "processing_times": {metric: f"{time:.2f} seconds" for metric, time in metric_timings.items()},
        "total_time": time.time() - start_time,
        "processing_date": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save summary
    with open(SILICON_PATH / "processing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("🎉 SILICON LAYER PROCESSING COMPLETED SUCCESSFULLY 🎉")
    print("="*80)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds ({(time.time() - start_time)/60:.2f} minutes)")
    
    # Print individual metric timings
    print("\nMetric processing times:")
    for metric, timing in metric_timings.items():
        print(f"   - {metric.replace('_', ' ').title()}: {timing:.2f} seconds ({timing/60:.2f} minutes)")
    
    # Cleanup dask client if it was created
    if client:
        client.close()
    
    return 0

if __name__ == "__main__":
    main()