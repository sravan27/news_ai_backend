"""
Advanced ML models for news content metrics using ensemble methods and neural networks.

This script implements sophisticated machine learning models for four key news metrics:
1. Political Influence
2. Rhetoric Intensity
3. Information Depth
4. Sentiment Analysis

Using bagging, boosting, stacking, and neural networks to achieve optimal performance
on constrained resources (M3 Max MacBook Pro).
"""
import os
import time
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ML libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, 
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier,
    StackingRegressor, StackingClassifier,
    BaggingRegressor, BaggingClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# Efficient gradient boosting libraries
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost, Pool

# Removed DuckDB for compatibility with pandas version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Use Apple Metal Performance Shaders if available
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Paths
DATA_PATH = Path("../data")
BRONZE_PATH = DATA_PATH / "bronze"
SILVER_PATH = DATA_PATH / "silver"
SILICON_PATH = DATA_PATH / "silicon"
MODEL_PATH = Path("../models")
DEPLOYED_PATH = MODEL_PATH / "deployed"

# List of metrics to model
METRICS = ["political_influence", "rhetoric_intensity", "information_depth", "sentiment"]

# Set random seeds for reproducibility
def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
# Neural network architectures
class DeepMetricsRegressor(nn.Module):
    """Advanced neural network for news metrics regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())  # Output in [0,1] range for all metrics
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# ResNet-style blocks for deeper networks
class ResidualBlock(nn.Module):
    """Residual block for neural networks."""
    
    def __init__(self, dim, dropout_rate=0.3, use_batch_norm=True):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity()
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        return self.relu(out)

class DeepResidualRegressor(nn.Module):
    """Deep residual network for metrics regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0,1] range
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output(x)

def load_data(metric_name: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Load data for a specific metric.
    
    Args:
        metric_name: Name of the metric to load data for
        
    Returns:
        Tuple containing:
        - DataFrame of features
        - Series of labels if available, otherwise None
    """
    logger.info(f"Loading data for {metric_name}...")
    
    # Look for existing features in silicon layer
    features_path = SILICON_PATH / metric_name / "features.parquet"
    feature_names_path = SILICON_PATH / metric_name / "feature_names.json"
    
    if features_path.exists() and feature_names_path.exists():
        # Load existing features
        logger.info(f"Found existing features for {metric_name}")
        try:
            df = pd.read_parquet(features_path)
            
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f).get('feature_names', [])
            
            if 'label' in df.columns:
                X = df.drop(['label', 'news_id'] if 'news_id' in df.columns else ['label'], axis=1)
                y = df['label']
                return X, y
            else:
                X = df.drop(['news_id'] if 'news_id' in df.columns else [], axis=1)
                return X, None
        except Exception as e:
            logger.error(f"Error loading existing features: {e}")
    
    # If not found, use silver layer data to create features
    logger.info(f"No existing features found, creating new features from silver layer")
    
    # Load news features from silver layer
    news_features_paths = [
        SILVER_PATH / "news_features_train.parquet",
        SILVER_PATH / "news_features_dev.parquet",
        SILVER_PATH / "news_features_test.parquet",
        SILVER_PATH / "news_base.parquet"  # Fallback to news_base if features not found
    ]
    
    dfs = []
    for path in news_features_paths:
        if path.exists():
            try:
                logger.info(f"Loading news features from {path}")
                df = pd.read_parquet(path)
                logger.info(f"Loaded {len(df)} records with columns: {df.columns.tolist()}")
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
    
    if not dfs:
        logger.warning("No news features found in silver layer. Creating synthetic data for demonstration.")
        # Create synthetic data for demonstration
        synthetic_df = pd.DataFrame({
            'news_id': [f'N{i}' for i in range(1000)],
            'category': np.random.choice(['politics', 'sports', 'entertainment', 'technology'], 1000),
            'title': [f'Synthetic Title {i}' for i in range(1000)],
            'abstract': [f'Synthetic Abstract {i} with some more text for demonstration purposes.' for i in range(1000)],
            'subcategory': np.random.choice(['world', 'us', 'business', 'health', 'science'], 1000),
        })
        dfs.append(synthetic_df)
    
    # Combine all dataframes
    news_df = pd.concat(dfs, ignore_index=True)
    
    # Create features relevant to the specific metric
    feature_df = pd.DataFrame()
    feature_df['news_id'] = news_df['news_id']
    
    # Basic features
    if 'category' in news_df.columns:
        # One-hot encode categories
        categories = pd.get_dummies(news_df['category'], prefix='category')
        feature_df = pd.concat([feature_df, categories], axis=1)
    
    if 'subcategory' in news_df.columns:
        # Count subcategories, but limit to avoid high dimensionality
        top_subcats = news_df['subcategory'].value_counts().nlargest(10).index
        for subcat in top_subcats:
            feature_df[f'subcategory_{subcat}'] = (news_df['subcategory'] == subcat).astype(int)
    
    # Text length features
    if 'title' in news_df.columns:
        feature_df['title_length'] = news_df['title'].apply(lambda x: len(str(x)))
        feature_df['title_word_count'] = news_df['title'].apply(lambda x: len(str(x).split()))
    
    if 'abstract' in news_df.columns:
        feature_df['abstract_length'] = news_df['abstract'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        feature_df['abstract_word_count'] = news_df['abstract'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    
    # Entity features
    if 'title_entities' in news_df.columns:
        feature_df['title_entity_count'] = news_df['title_entities'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    if 'abstract_entities' in news_df.columns:
        feature_df['abstract_entity_count'] = news_df['abstract_entities'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Skip DuckDB for compatibility
    logger.info("Skipping DuckDB for compatibility with current pandas version.")
    
    # Metric-specific features
    if metric_name == "political_influence":
        # Add features specific to political influence
        political_terms = ['president', 'government', 'congress', 'law', 'policy', 'election',
                          'vote', 'democrat', 'republican', 'senator', 'representative']
        
        # Count political terms in title and abstract
        for term in political_terms:
            feature_df[f'title_contains_{term}'] = news_df['title'].apply(
                lambda x: 1 if term.lower() in str(x).lower() else 0
            )
            
            feature_df[f'abstract_contains_{term}'] = news_df['abstract'].apply(
                lambda x: 1 if pd.notna(x) and term.lower() in str(x).lower() else 0
            )
        
        # Count political entities
        if 'title_entities' in news_df.columns and 'abstract_entities' in news_df.columns:
            feature_df['political_entity_count'] = news_df.apply(
                lambda row: sum(
                    1 for entity in (row['title_entities'] + row['abstract_entities']) 
                    if isinstance(entity, dict) and entity.get('Type', '') == 'P'
                ),
                axis=1
            )
    
    elif metric_name == "rhetoric_intensity":
        # Add features specific to rhetoric intensity
        rhetoric_markers = ['!', '?', '...', 'very', 'extremely', 'incredibly', 'absolutely', 'utterly']
        
        # Count rhetoric markers
        for marker in rhetoric_markers:
            feature_df[f'title_contains_{marker}'] = news_df['title'].apply(
                lambda x: 1 if marker in str(x) else 0
            )
            
            feature_df[f'abstract_contains_{marker}'] = news_df['abstract'].apply(
                lambda x: 1 if pd.notna(x) and marker in str(x) else 0
            )
        
        # Count questions and exclamations
        feature_df['title_exclamation_count'] = news_df['title'].apply(
            lambda x: str(x).count('!')
        )
        feature_df['title_question_count'] = news_df['title'].apply(
            lambda x: str(x).count('?')
        )
        
        if 'abstract' in news_df.columns:
            feature_df['abstract_exclamation_count'] = news_df['abstract'].apply(
                lambda x: str(x).count('!') if pd.notna(x) else 0
            )
            feature_df['abstract_question_count'] = news_df['abstract'].apply(
                lambda x: str(x).count('?') if pd.notna(x) else 0
            )
    
    elif metric_name == "information_depth":
        # Add features specific to information depth
        
        # Unique word ratio
        feature_df['title_unique_word_ratio'] = news_df['title'].apply(
            lambda x: len(set(str(x).lower().split())) / len(str(x).lower().split()) if len(str(x).split()) > 0 else 0
        )
        
        if 'abstract' in news_df.columns:
            feature_df['abstract_unique_word_ratio'] = news_df['abstract'].apply(
                lambda x: len(set(str(x).lower().split())) / len(str(x).lower().split()) 
                if pd.notna(x) and len(str(x).split()) > 0 else 0
            )
        
        # Count numbers in text (often indicates data/statistics)
        feature_df['title_number_count'] = news_df['title'].apply(
            lambda x: sum(c.isdigit() for c in str(x))
        )
        
        if 'abstract' in news_df.columns:
            feature_df['abstract_number_count'] = news_df['abstract'].apply(
                lambda x: sum(c.isdigit() for c in str(x)) if pd.notna(x) else 0
            )
    
    elif metric_name == "sentiment":
        # Add features specific to sentiment analysis
        
        # Positive and negative term counts
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", 
                         "happy", "joy", "success", "beautiful", "love", "positive",
                         "achieve", "benefit", "win", "victory", "celebrate"]
        
        negative_words = ["bad", "terrible", "awful", "horrible", "poor", 
                         "sad", "hate", "failure", "ugly", "angry", "negative",
                         "loss", "damage", "threat", "crisis", "problem"]
        
        # Count positive words
        for word in positive_words[:5]:  # Limit to avoid too many features
            feature_df[f'title_contains_{word}'] = news_df['title'].apply(
                lambda x: 1 if word.lower() in str(x).lower() else 0
            )
            
            feature_df[f'abstract_contains_{word}'] = news_df['abstract'].apply(
                lambda x: 1 if pd.notna(x) and word.lower() in str(x).lower() else 0
            )
        
        # Count negative words
        for word in negative_words[:5]:  # Limit to avoid too many features
            feature_df[f'title_contains_{word}'] = news_df['title'].apply(
                lambda x: 1 if word.lower() in str(x).lower() else 0
            )
            
            feature_df[f'abstract_contains_{word}'] = news_df['abstract'].apply(
                lambda x: 1 if pd.notna(x) and word.lower() in str(x).lower() else 0
            )
        
        # Overall counts
        feature_df['title_positive_count'] = news_df['title'].apply(
            lambda x: sum(1 for word in positive_words if word.lower() in str(x).lower())
        )
        
        feature_df['title_negative_count'] = news_df['title'].apply(
            lambda x: sum(1 for word in negative_words if word.lower() in str(x).lower())
        )
        
        if 'abstract' in news_df.columns:
            feature_df['abstract_positive_count'] = news_df['abstract'].apply(
                lambda x: sum(1 for word in positive_words if pd.notna(x) and word.lower() in str(x).lower())
            )
            
            feature_df['abstract_negative_count'] = news_df['abstract'].apply(
                lambda x: sum(1 for word in negative_words if pd.notna(x) and word.lower() in str(x).lower())
            )
        
        # Sentiment ratio
        feature_df['title_sentiment_ratio'] = (
            feature_df['title_positive_count'] - feature_df['title_negative_count']
        ) / (feature_df['title_positive_count'] + feature_df['title_negative_count'] + 1)
        
        if 'abstract' in news_df.columns:
            feature_df['abstract_sentiment_ratio'] = (
                feature_df['abstract_positive_count'] - feature_df['abstract_negative_count']
            ) / (feature_df['abstract_positive_count'] + feature_df['abstract_negative_count'] + 1)
    
    # Create synthetic labels for demonstration/training
    # In a real scenario, you would use expert-annotated data
    labels = pd.Series(np.random.random(len(feature_df)), index=feature_df.index)
    
    if metric_name == "political_influence":
        # Higher political influence for political categories
        if 'category_politics' in feature_df.columns:
            labels = labels * 0.3 + 0.5 * feature_df['category_politics'] + 0.2 * np.random.random(len(feature_df))
    
    elif metric_name == "rhetoric_intensity":
        # Higher rhetoric intensity with more exclamation points, questions
        if 'title_exclamation_count' in feature_df.columns and 'title_question_count' in feature_df.columns:
            labels = labels * 0.3 + 0.3 * (
                feature_df['title_exclamation_count'] + feature_df['title_question_count']
            ).clip(0, 1) + 0.4 * np.random.random(len(feature_df))
    
    elif metric_name == "information_depth":
        # Higher information depth with more unique words and longer text
        if 'abstract_unique_word_ratio' in feature_df.columns and 'abstract_length' in feature_df.columns:
            labels = labels * 0.3 + 0.3 * feature_df['abstract_unique_word_ratio'] + 0.2 * (
                feature_df['abstract_length'] / 500
            ).clip(0, 1) + 0.2 * np.random.random(len(feature_df))
    
    elif metric_name == "sentiment":
        # Sentiment based on positive/negative word counts
        if 'title_sentiment_ratio' in feature_df.columns and 'abstract_sentiment_ratio' in feature_df.columns:
            labels = 0.5 + 0.25 * feature_df['title_sentiment_ratio'] + 0.25 * feature_df['abstract_sentiment_ratio']
            # Clip to [0,1] range
            labels = labels.clip(0, 1)
    
    # Normalize labels to [0,1] range
    labels = (labels - labels.min()) / (labels.max() - labels.min())
    
    # Add labels to feature dataframe
    feature_df['label'] = labels
    
    # Save the features
    os.makedirs(SILICON_PATH / metric_name, exist_ok=True)
    feature_df.to_parquet(SILICON_PATH / metric_name / "features.parquet", index=False)
    
    # Save feature names
    with open(SILICON_PATH / metric_name / "feature_names.json", 'w') as f:
        json.dump({'feature_names': feature_df.drop(['news_id', 'label'], axis=1).columns.tolist()}, f)
    
    # Return features and labels
    X = feature_df.drop(['news_id', 'label'], axis=1)
    y = feature_df['label']
    
    return X, y

def train_boosting_models(X_train, y_train, X_val, y_val, metric_name: str):
    """
    Train and evaluate boosting models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric_name: Name of the metric
        
    Returns:
        Dictionary of models and their validation scores
    """
    logger.info(f"Training boosting models for {metric_name}...")
    
    models = {}
    scores = {}
    
    # LightGBM
    start = time.time()
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mse',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    lgb_val_pred = lgb_model.predict(X_val)
    lgb_val_mse = mean_squared_error(y_val, lgb_val_pred)
    lgb_val_mae = mean_absolute_error(y_val, lgb_val_pred)
    lgb_val_r2 = r2_score(y_val, lgb_val_pred)
    
    models['lgb'] = lgb_model
    scores['lgb'] = {
        'mse': lgb_val_mse,
        'mae': lgb_val_mae,
        'r2': lgb_val_r2,
        'time': time.time() - start
    }
    
    logger.info(f"LightGBM - MSE: {lgb_val_mse:.4f}, MAE: {lgb_val_mae:.4f}, R²: {lgb_val_r2:.4f}")
    
    # XGBoost
    start = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        early_stopping_rounds=50
    )
    
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_val_mse = mean_squared_error(y_val, xgb_val_pred)
    xgb_val_mae = mean_absolute_error(y_val, xgb_val_pred)
    xgb_val_r2 = r2_score(y_val, xgb_val_pred)
    
    models['xgb'] = xgb_model
    scores['xgb'] = {
        'mse': xgb_val_mse,
        'mae': xgb_val_mae,
        'r2': xgb_val_r2,
        'time': time.time() - start
    }
    
    logger.info(f"XGBoost - MSE: {xgb_val_mse:.4f}, MAE: {xgb_val_mae:.4f}, R²: {xgb_val_r2:.4f}")
    
    # CatBoost
    start = time.time()
    cat_model = CatBoost(
        params={
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'RMSE',
            'verbose': False
        }
    )
    
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50
    )
    
    cat_val_pred = cat_model.predict(X_val)
    cat_val_mse = mean_squared_error(y_val, cat_val_pred)
    cat_val_mae = mean_absolute_error(y_val, cat_val_pred)
    cat_val_r2 = r2_score(y_val, cat_val_pred)
    
    models['cat'] = cat_model
    scores['cat'] = {
        'mse': cat_val_mse,
        'mae': cat_val_mae,
        'r2': cat_val_r2,
        'time': time.time() - start
    }
    
    logger.info(f"CatBoost - MSE: {cat_val_mse:.4f}, MAE: {cat_val_mae:.4f}, R²: {cat_val_r2:.4f}")
    
    return models, scores

def train_ensemble_models(base_models, X_train, y_train, X_val, y_val, metric_name: str):
    """
    Train and evaluate ensemble models.
    
    Args:
        base_models: Dictionary of base models
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric_name: Name of the metric
        
    Returns:
        Dictionary of models and their validation scores
    """
    logger.info(f"Training ensemble models for {metric_name}...")
    
    models = {}
    scores = {}
    
    # Voting Regressor
    start = time.time()
    voting_model = VotingRegressor(
        estimators=[
            ('lgb', base_models['lgb']),
            ('xgb', base_models['xgb']),
            ('cat', base_models['cat'])
        ]
    )
    
    voting_model.fit(X_train, y_train)
    
    voting_val_pred = voting_model.predict(X_val)
    voting_val_mse = mean_squared_error(y_val, voting_val_pred)
    voting_val_mae = mean_absolute_error(y_val, voting_val_pred)
    voting_val_r2 = r2_score(y_val, voting_val_pred)
    
    models['voting'] = voting_model
    scores['voting'] = {
        'mse': voting_val_mse,
        'mae': voting_val_mae,
        'r2': voting_val_r2,
        'time': time.time() - start
    }
    
    logger.info(f"Voting Ensemble - MSE: {voting_val_mse:.4f}, MAE: {voting_val_mae:.4f}, R²: {voting_val_r2:.4f}")
    
    # Stacking Regressor
    start = time.time()
    stacking_model = StackingRegressor(
        estimators=[
            ('lgb', base_models['lgb']),
            ('xgb', base_models['xgb']),
            ('cat', base_models['cat'])
        ],
        final_estimator=Ridge()
    )
    
    stacking_model.fit(X_train, y_train)
    
    stacking_val_pred = stacking_model.predict(X_val)
    stacking_val_mse = mean_squared_error(y_val, stacking_val_pred)
    stacking_val_mae = mean_absolute_error(y_val, stacking_val_pred)
    stacking_val_r2 = r2_score(y_val, stacking_val_pred)
    
    models['stacking'] = stacking_model
    scores['stacking'] = {
        'mse': stacking_val_mse,
        'mae': stacking_val_mae,
        'r2': stacking_val_r2,
        'time': time.time() - start
    }
    
    logger.info(f"Stacking Ensemble - MSE: {stacking_val_mse:.4f}, MAE: {stacking_val_mae:.4f}, R²: {stacking_val_r2:.4f}")
    
    # Bagging with Random Forest
    start = time.time()
    bagging_model = BaggingRegressor(
        base_estimator=RandomForestRegressor(n_estimators=100, random_state=42),
        n_estimators=10,
        random_state=42
    )
    
    bagging_model.fit(X_train, y_train)
    
    bagging_val_pred = bagging_model.predict(X_val)
    bagging_val_mse = mean_squared_error(y_val, bagging_val_pred)
    bagging_val_mae = mean_absolute_error(y_val, bagging_val_pred)
    bagging_val_r2 = r2_score(y_val, bagging_val_pred)
    
    models['bagging'] = bagging_model
    scores['bagging'] = {
        'mse': bagging_val_mse,
        'mae': bagging_val_mae,
        'r2': bagging_val_r2,
        'time': time.time() - start
    }
    
    logger.info(f"Bagging Ensemble - MSE: {bagging_val_mse:.4f}, MAE: {bagging_val_mae:.4f}, R²: {bagging_val_r2:.4f}")
    
    return models, scores

def train_neural_network(X_train, y_train, X_val, y_val, metric_name: str):
    """
    Train and evaluate neural network models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric_name: Name of the metric
        
    Returns:
        Dictionary of models and their validation scores
    """
    logger.info(f"Training neural network models for {metric_name}...")
    
    models = {}
    scores = {}
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(device)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Deep Neural Network
    start = time.time()
    input_dim = X_train.shape[1]
    
    # Standard Deep Network
    deep_model = DeepMetricsRegressor(input_dim=input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(deep_model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    epochs = 100
    patience = 10
    best_val_loss = float('inf')
    counter = 0
    
    # Use torch.compile for PyTorch 2.0+ acceleration if available
    if hasattr(torch, 'compile'):
        try:
            deep_model = torch.compile(deep_model)
            logger.info("Using torch.compile() for accelerated training")
        except Exception as e:
            logger.warning(f"Could not use torch.compile(): {e}")
    
    # Enable cuDNN benchmark mode for faster training
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark mode")
    
    for epoch in range(epochs):
        # Training
        deep_model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = deep_model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        deep_model.eval()
        with torch.no_grad():
            val_outputs = deep_model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Save best model
                torch.save(deep_model.state_dict(), 
                          SILICON_PATH / metric_name / "nn_model.pt")
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    
    # Evaluation
    deep_model.eval()
    with torch.no_grad():
        val_pred = deep_model(X_val_tensor).cpu().numpy().flatten()
        val_mse = mean_squared_error(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
    
    models['dnn'] = deep_model
    scores['dnn'] = {
        'mse': val_mse,
        'mae': val_mae,
        'r2': val_r2,
        'time': time.time() - start
    }
    
    logger.info(f"Deep Neural Network - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    # Residual Neural Network
    start = time.time()
    res_model = DeepResidualRegressor(input_dim=input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(res_model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    best_val_loss = float('inf')
    counter = 0
    
    # Use torch.compile for PyTorch 2.0+ acceleration if available
    if hasattr(torch, 'compile'):
        try:
            res_model = torch.compile(res_model)
            logger.info("Using torch.compile() for accelerated training")
        except Exception as e:
            logger.warning(f"Could not use torch.compile(): {e}")
    
    for epoch in range(epochs):
        # Training
        res_model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = res_model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        res_model.eval()
        with torch.no_grad():
            val_outputs = res_model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Save best model (residual)
                torch.save(res_model.state_dict(), 
                          SILICON_PATH / metric_name / "res_nn_model.pt")
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    
    # Evaluation
    res_model.eval()
    with torch.no_grad():
        val_pred = res_model(X_val_tensor).cpu().numpy().flatten()
        val_mse = mean_squared_error(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
    
    models['resnet'] = res_model
    scores['resnet'] = {
        'mse': val_mse,
        'mae': val_mae,
        'r2': val_r2,
        'time': time.time() - start
    }
    
    logger.info(f"Residual Neural Network - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    # Save scaler for inference
    with open(SILICON_PATH / metric_name / "nn_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save neural network metadata
    nn_metadata = {
        'input_dim': input_dim,
        'device': str(device),
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val_loss,
        'architecture': 'DeepMetricsRegressor',
        'layers': [input_dim, 256, 128, 64, 1]
    }
    
    with open(SILICON_PATH / metric_name / "nn_model_metadata.json", 'w') as f:
        json.dump(nn_metadata, f, indent=2)
    
    return models, scores

def select_best_model(all_scores, base_models, ensemble_models, nn_models, X_test, y_test, metric_name: str):
    """
    Select the best model based on validation scores and test it.
    
    Args:
        all_scores: Dictionary of all model scores
        base_models: Dictionary of base models
        ensemble_models: Dictionary of ensemble models
        nn_models: Dictionary of neural network models
        X_test: Test features
        y_test: Test labels
        metric_name: Name of the metric
        
    Returns:
        Best model and test metrics
    """
    logger.info(f"Selecting best model for {metric_name}...")
    
    # Combine all scores
    all_model_scores = {}
    for model_type, scores_dict in all_scores.items():
        for model_name, score_values in scores_dict.items():
            all_model_scores[f"{model_type}_{model_name}"] = score_values
    
    # Find best model based on validation MAE
    best_model_name = min(all_model_scores, key=lambda x: all_model_scores[x]['mae'])
    best_val_score = all_model_scores[best_model_name]['mae']
    
    logger.info(f"Best model: {best_model_name} with validation MAE: {best_val_score:.4f}")
    
    # Get the best model
    model_type, model_name = best_model_name.split('_')
    if model_type == 'base':
        best_model = base_models[model_name]
    elif model_type == 'ensemble':
        best_model = ensemble_models[model_name]
    elif model_type == 'nn':
        best_model = nn_models[model_name]
    
    # Evaluate on test set
    if model_type == 'nn':
        # Scale test data
        with open(SILICON_PATH / metric_name / "nn_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        X_test_scaled = scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        best_model.eval()
        with torch.no_grad():
            y_pred = best_model(X_test_tensor).cpu().numpy().flatten()
    else:
        y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test metrics for {best_model_name}:")
    logger.info(f"MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # Save the best model
    if model_type != 'nn':  # NN models are already saved during training
        with open(SILICON_PATH / metric_name / "model.pkl", 'wb') as f:
            pickle.dump(best_model, f)
    
    # Save model metadata
    model_metadata = {
        'model_type': model_type,
        'model_name': model_name,
        'validation_metrics': all_model_scores[best_model_name],
        'test_metrics': {
            'mse': test_mse,
            'mae': test_mae,
            'r2': test_r2
        },
        'feature_count': X_test.shape[1],
        'training_date': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(SILICON_PATH / metric_name / "model_metadata.json", 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Save a sample of test predictions for visual inspection
    sample_size = min(100, len(y_test))
    sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    sample_df = pd.DataFrame({
        'actual': y_test.iloc[sample_indices].values,
        'predicted': y_pred[sample_indices]
    })
    
    sample_df.to_csv(SILICON_PATH / metric_name / "prediction_samples.csv", index=False)
    
    # Create plots for visual evaluation
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of predictions vs actuals
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{metric_name.replace('_', ' ').title()} - Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    
    # Save the plot
    plt.savefig(SILICON_PATH / metric_name / "prediction_plot.png")
    plt.close()
    
    return best_model, {
        'mse': test_mse,
        'mae': test_mae,
        'r2': test_r2
    }

def train_metric_model(metric_name: str):
    """
    Train models for a specific metric.
    
    Args:
        metric_name: Name of the metric to train models for
    """
    start_time = time.time()
    logger.info(f"Starting model training for {metric_name}...")
    
    # Create the metric directory
    os.makedirs(SILICON_PATH / metric_name, exist_ok=True)
    
    # Load and prepare data
    data_load_start = time.time()
    X, y = load_data(metric_name)
    logger.info(f"Data loading took {time.time() - data_load_start:.2f} seconds")
    
    if X is None or y is None:
        logger.error(f"Failed to load data for {metric_name}")
        return
    
    logger.info(f"Loaded data with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Split data
    split_start = time.time()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    logger.info(f"Data splitting took {time.time() - split_start:.2f} seconds")
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # 1. Train boosting models
    boost_start = time.time()
    base_models, base_scores = train_boosting_models(X_train, y_train, X_val, y_val, metric_name)
    logger.info(f"Boosting models training took {time.time() - boost_start:.2f} seconds")
    
    # 2. Train ensemble models
    ensemble_start = time.time()
    ensemble_models, ensemble_scores = train_ensemble_models(
        base_models, X_train, y_train, X_val, y_val, metric_name
    )
    logger.info(f"Ensemble models training took {time.time() - ensemble_start:.2f} seconds")
    
    # 3. Train neural networks
    nn_start = time.time()
    nn_models, nn_scores = train_neural_network(X_train, y_train, X_val, y_val, metric_name)
    logger.info(f"Neural network training took {time.time() - nn_start:.2f} seconds")
    
    # 4. Select best model
    all_scores = {
        'base': base_scores,
        'ensemble': ensemble_scores,
        'nn': nn_scores
    }
    
    select_start = time.time()
    best_model, test_metrics = select_best_model(
        all_scores, base_models, ensemble_models, nn_models,
        X_test, y_test, metric_name
    )
    logger.info(f"Best model selection took {time.time() - select_start:.2f} seconds")
    
    # 5. Final summary
    logger.info(f"Model training for {metric_name} completed.")
    logger.info(f"Final test metrics: MSE={test_metrics['mse']:.4f}, MAE={test_metrics['mae']:.4f}, R²={test_metrics['r2']:.4f}")
    
    # Calculate total time
    total_training_time = time.time() - start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total training time for {metric_name}: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    return best_model, test_metrics

def deploy_best_models():
    """Deploy the best models for all metrics to the deployment directory."""
    logger.info("Deploying best models...")
    
    # Create the deployment directory
    os.makedirs(DEPLOYED_PATH, exist_ok=True)
    
    all_metrics = {}
    
    for metric_name in METRICS:
        metric_path = SILICON_PATH / metric_name
        deploy_path = DEPLOYED_PATH / metric_name
        
        if not metric_path.exists():
            logger.warning(f"No model found for {metric_name}")
            continue
        
        # Create deployment directory
        os.makedirs(deploy_path, exist_ok=True)
        
        # Copy files
        files_to_copy = [
            "model.pkl",
            "nn_model.pt",
            "scaler.pkl",
            "nn_scaler.pkl",
            "feature_names.json",
            "model_metadata.json",
        ]
        
        for file in files_to_copy:
            src = metric_path / file
            dst = deploy_path / file
            if src.exists():
                import shutil
                shutil.copy(src, dst)
                logger.info(f"Copied {file} for {metric_name}")
        
        # Load and store metrics
        if (metric_path / "model_metadata.json").exists():
            with open(metric_path / "model_metadata.json", 'r') as f:
                metadata = json.load(f)
                test_metrics = metadata.get('test_metrics', {})
                all_metrics[metric_name] = test_metrics
    
    # Save a summary of all metrics
    with open(SILICON_PATH / "processing_summary.json", 'w') as f:
        summary = {
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": all_metrics
        }
        json.dump(summary, f, indent=2)
    
    # Also copy to deployment
    shutil.copy(SILICON_PATH / "processing_summary.json", DEPLOYED_PATH / "processing_summary.json")
    
    logger.info("Deployment completed")

def main():
    """Main entry point."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train advanced ML models for news metrics")
    parser.add_argument('--metric', choices=METRICS + ['all'], default='all', 
                       help='Which metric to train models for')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--num_cores', type=int, default=8,
                      help='Number of CPU cores to use')
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Set random seed
    set_seeds(42)
    
    # Set number of CPU cores to use for parallel processing
    import os
    os.environ["OMP_NUM_THREADS"] = str(args.num_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.num_cores)
    os.environ["MKL_NUM_THREADS"] = str(args.num_cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.num_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.num_cores)
    
    logger.info(f"Using {args.num_cores} CPU cores for parallel processing")
    
    # Create necessary directories
    os.makedirs(SILICON_PATH, exist_ok=True)
    os.makedirs(DEPLOYED_PATH, exist_ok=True)
    
    # Train models
    if args.metric == 'all':
        logger.info("Training models for all metrics...")
        
        all_metrics = {}
        
        for metric_name in METRICS:
            _, test_metrics = train_metric_model(metric_name)
            all_metrics[metric_name] = test_metrics
        
        # Deploy all models
        deploy_best_models()
    else:
        logger.info(f"Training models for {args.metric}...")
        train_metric_model(args.metric)
    
    logger.info("All done!")

if __name__ == "__main__":
    main()