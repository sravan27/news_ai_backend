"""
User Feature Extractor for News AI application.

This module provides advanced user feature extraction capabilities:
- Behavioral features (click patterns, read time, etc.)
- Temporal engagement features
- Category and topic preferences
- Content diversity metrics
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import Counter, defaultdict
import yaml
import os
import pickle
import json
import time
from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class UserFeatureExtractor:
    """Advanced user feature extractor for News AI pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the user feature extractor with configuration settings.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.config = config['features']['user_features']
        else:
            # Default configuration if not provided
            self.config = {
                'min_history_length': 5,
                'temporal_window_days': 7,
                'recency_weight_decay': 0.85
            }
        
        self.min_history_length = self.config.get('min_history_length', 5)
        self.temporal_window_days = self.config.get('temporal_window_days', 7)
        self.recency_weight_decay = self.config.get('recency_weight_decay', 0.85)
        
        # Initialize scalers
        self.feature_scaler = None
    
    def extract_basic_engagement_features(self, behaviors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic engagement features from user behaviors.
        
        Args:
            behaviors_df: DataFrame containing user behaviors
            
        Returns:
            DataFrame with user engagement features
        """
        print("Extracting basic engagement features...")
        
        # Group by user_id
        user_features = behaviors_df.groupby('user_id').agg({
            'impression_id': 'count',
            'history_length': ['mean', 'min', 'max', 'std'],
            'impressions_count': ['mean', 'min', 'max', 'sum'],
            'click_count': ['mean', 'min', 'max', 'sum'],
            'click_ratio': ['mean', 'min', 'max']
        })
        
        # Flatten the multi-level columns
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
        
        # Rename columns for clarity
        user_features = user_features.rename(columns={
            'impression_id_count': 'behavior_count',
            'history_length_mean': 'avg_history_length',
            'impressions_count_sum': 'total_impressions',
            'click_count_sum': 'total_clicks'
        })
        
        # Calculate additional metrics
        user_features['overall_ctr'] = user_features['total_clicks'] / user_features['total_impressions']
        
        # Handle potential division by zero
        user_features['overall_ctr'] = user_features['overall_ctr'].fillna(0)
        
        return user_features
    
    def extract_temporal_features(self, behaviors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal engagement features from user behaviors.
        
        Args:
            behaviors_df: DataFrame containing user behaviors with timestamps
            
        Returns:
            DataFrame with user temporal features
        """
        print("Extracting temporal engagement features...")
        import multiprocessing as mp
        
        # Ensure the timestamp column is datetime
        if 'timestamp' not in behaviors_df.columns:
            if 'time' in behaviors_df.columns:
                behaviors_df['timestamp'] = pd.to_datetime(behaviors_df['time'])
            else:
                raise ValueError("No timestamp column found in behaviors DataFrame")
        
        # Create a function to process each user in parallel
        def process_user_temporal(user_data):
            user_id, user_df = user_data
            
            if user_df.empty:
                return None
            
            # Sort by timestamp
            user_df = user_df.sort_values('timestamp')
            
            # Extract temporal features
            first_activity = user_df['timestamp'].min()
            last_activity = user_df['timestamp'].max()
            activity_duration = (last_activity - first_activity).total_seconds() / 3600  # in hours
            
            # Count activity by day of week - use numpy for performance
            day_values = user_df['day_of_week'].values
            day_counts = {day: np.sum(day_values == day) for day in range(7)}
            days_active = len([c for c in day_counts.values() if c > 0])
            
            # Find most active day
            if not user_df.empty:
                day_count_values = np.array(list(day_counts.values()))
                if len(day_count_values) > 0 and day_count_values.max() > 0:
                    most_active_day = list(day_counts.keys())[day_count_values.argmax()]
                else:
                    most_active_day = -1
            else:
                most_active_day = -1
            
            # Count activity by hour of day
            hour_values = user_df['hour_of_day'].values
            hour_counts = {hour: np.sum(hour_values == hour) for hour in range(24)}
            hours_active = len([c for c in hour_counts.values() if c > 0])
            
            # Find most active hour
            if not user_df.empty:
                hour_count_values = np.array(list(hour_counts.values()))
                if len(hour_count_values) > 0 and hour_count_values.max() > 0:
                    most_active_hour = list(hour_counts.keys())[hour_count_values.argmax()]
                else:
                    most_active_hour = -1
            else:
                most_active_hour = -1
            
            # Calculate activity frequency (sessions per day)
            days_span = max(1, (last_activity - first_activity).days)
            activity_frequency = len(user_df) / days_span
            
            # Calculate time between sessions - vectorized
            user_df_len = len(user_df)
            if user_df_len > 1:
                # Convert to numpy for faster operations
                timestamps = user_df['timestamp'].values
                time_diffs = np.diff(timestamps) / np.timedelta64(1, 'h')  # in hours
                
                avg_time_between_sessions = np.mean(time_diffs)
                min_time_between_sessions = np.min(time_diffs)
                max_time_between_sessions = np.max(time_diffs)
                std_time_between_sessions = np.std(time_diffs)
                
                # Detect bursty behavior (many activities in short time)
                bursty_threshold = 0.5  # hours
                bursty_session_count = np.sum(time_diffs < bursty_threshold)
                bursty_session_ratio = bursty_session_count / len(time_diffs) if len(time_diffs) > 0 else 0
            else:
                avg_time_between_sessions = 0
                min_time_between_sessions = 0
                max_time_between_sessions = 0
                std_time_between_sessions = 0
                bursty_session_ratio = 0
            
            # Check recency of activity
            now = behaviors_df['timestamp'].max()
            days_since_last_activity = (now - last_activity).total_seconds() / (3600 * 24)
            is_recent_active = days_since_last_activity < self.temporal_window_days
            
            # Pre-calculate total user records for ratios
            total_records = len(user_df)
            
            # Compile features
            user_features = {
                'user_id': user_id,
                'first_activity': first_activity,
                'last_activity': last_activity,
                'activity_duration_hours': activity_duration,
                'days_active': days_active,
                'most_active_day': most_active_day,
                'hours_active': hours_active,
                'most_active_hour': most_active_hour,
                'activity_frequency': activity_frequency,
                'avg_time_between_sessions': avg_time_between_sessions,
                'min_time_between_sessions': min_time_between_sessions,
                'max_time_between_sessions': max_time_between_sessions,
                'std_time_between_sessions': std_time_between_sessions,
                'bursty_session_ratio': bursty_session_ratio,
                'days_since_last_activity': days_since_last_activity,
                'is_recent_active': is_recent_active
            }
            
            # Add day of week distribution - vectorized operation
            for day in range(7):
                user_features[f'day_{day}_ratio'] = day_counts.get(day, 0) / total_records
            
            # Add hour of day groups - vectorized operations
            morning_hours = set(range(6, 12))
            afternoon_hours = set(range(12, 18))
            evening_hours = set(range(18, 24))
            night_hours = set(range(0, 6))
            
            user_features['morning_ratio'] = sum(hour_counts.get(h, 0) for h in morning_hours) / total_records
            user_features['afternoon_ratio'] = sum(hour_counts.get(h, 0) for h in afternoon_hours) / total_records
            user_features['evening_ratio'] = sum(hour_counts.get(h, 0) for h in evening_hours) / total_records
            user_features['night_ratio'] = sum(hour_counts.get(h, 0) for h in night_hours) / total_records
            
            return user_features
        
        # Group data by user
        user_groups = list(behaviors_df.groupby('user_id'))
        
        # Use parallel processing with process pool
        num_cores = min(mp.cpu_count(), 8)  # Use up to 8 cores for M2 Max
        print(f"Using {num_cores} cores for temporal feature extraction")
        
        # Process users in parallel
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(process_user_temporal, user_groups)
        
        # Filter out None values and convert to DataFrame
        results = [r for r in results if r is not None]
        temporal_features = pd.DataFrame(results)
        
        if temporal_features.empty:
            return pd.DataFrame()
            
        # Set user_id as index
        temporal_features.set_index('user_id', inplace=True)
        
        return temporal_features
    
    def extract_content_preference_features(self, behaviors_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract content preference features from user behaviors.
        
        Args:
            behaviors_df: DataFrame containing user behaviors
            news_df: DataFrame containing news articles metadata
            
        Returns:
            DataFrame with user content preference features
        """
        print("Extracting content preference features...")
        import multiprocessing as mp
        from functools import partial
        
        # Create a lookup dictionary for news metadata (convert to dict once)
        news_lookup = news_df.set_index('news_id').to_dict(orient='index')
        
        # Create a function to process each user in parallel
        def process_user(user_data):
            user_id, user_df = user_data
            
            # Extract all clicked news IDs - vectorized
            clicked_news_ids = []
            for impressions in user_df['impressions']:
                clicked_news_ids.extend([imp['news_id'] for imp in impressions if imp['clicked'] == 1])
            
            # Extract all history news IDs - vectorized
            history_news_ids = []
            for history in user_df['history']:
                history_news_ids.extend(history)
            
            # Combine clicked and history news IDs
            all_news_ids = clicked_news_ids + history_news_ids
            
            # Skip users with insufficient history
            if len(all_news_ids) < self.min_history_length:
                return None
            
            # Create lookup arrays for faster access
            valid_news_ids = [nid for nid in all_news_ids if nid in news_lookup]
            
            # Get categories and subcategories in one pass
            categories = [news_lookup[nid]['category'] for nid in valid_news_ids]
            subcategories = [news_lookup[nid]['subcategory'] for nid in valid_news_ids]
            
            # Count categories and subcategories
            category_counts = Counter(categories)
            subcategory_counts = Counter(subcategories)
            
            # Calculate category preferences
            total_articles = len(categories)
            if total_articles == 0:
                return None
                
            category_ratios = {cat: count / total_articles for cat, count in category_counts.items()}
            
            # Find most preferred categories
            top_categories = [cat for cat, _ in category_counts.most_common(3)]
            top_subcategories = [subcat for subcat, _ in subcategory_counts.most_common(3)]
            
            # Calculate category diversity (entropy)
            if category_counts:
                category_probs = np.array([count / total_articles for count in category_counts.values()])
                # Use numpy vectorized operations for entropy calculation
                category_entropy = -np.sum(category_probs * np.log2(category_probs))
            else:
                category_entropy = 0
            
            # Initialize features
            user_features = {
                'user_id': user_id,
                'total_content_interactions': total_articles,
                'category_diversity': category_entropy,
                'top_category': top_categories[0] if top_categories else None,
                'top_subcategory': top_subcategories[0] if top_subcategories else None
            }
            
            # Add category ratios - pre-allocate in dictionary
            for cat, ratio in category_ratios.items():
                user_features[f'category_{cat}_ratio'] = ratio
                
            return user_features
        
        # Group data by user (only once)
        user_groups = list(behaviors_df.groupby('user_id'))
        
        # Use parallel processing with process pool
        num_cores = min(mp.cpu_count(), 8)  # Use up to 8 cores
        print(f"Using {num_cores} cores for parallel processing")
        
        # Process users in chunks
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(process_user, user_groups)
        
        # Filter out None values and convert to DataFrame
        results = [r for r in results if r is not None]
        content_features = pd.DataFrame(results)
        
        if content_features.empty:
            return pd.DataFrame()
            
        # Set user_id as index
        content_features.set_index('user_id', inplace=True)
        
        # Handle categorical columns efficiently
        content_features = pd.get_dummies(content_features, columns=['top_category', 'top_subcategory'], dummy_na=True, sparse=True)
        
        return content_features
    
    def extract_engagement_pattern_features(self, behaviors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract engagement pattern features from user behaviors.
        
        Args:
            behaviors_df: DataFrame containing user behaviors
            
        Returns:
            DataFrame with user engagement pattern features
        """
        print("Extracting engagement pattern features...")
        import multiprocessing as mp
        
        # Create a function to process each user in parallel
        def process_user_engagement(user_data):
            user_id, user_df = user_data
            
            if user_df.empty:
                return None
                
            user_features = {'user_id': user_id}
            
            # Calculate click-through rate over time
            if 'timestamp' in user_df.columns:
                user_df = user_df.sort_values('timestamp')
                
                # Calculate CTR in different time windows
                ctr_7d = ctr_14d = ctr_30d = 0
                
                if len(user_df) > 0:
                    last_timestamp = user_df['timestamp'].max()
                    
                    # Use numpy for faster filtering
                    timestamps = user_df['timestamp'].values
                    impressions = user_df['impressions_count'].values
                    clicks = user_df['click_count'].values
                    
                    # 7-day window - vectorized
                    window_7d_mask = timestamps >= (last_timestamp - np.timedelta64(7, 'D'))
                    imps_7d = np.sum(impressions[window_7d_mask]) if any(window_7d_mask) else 0
                    clicks_7d = np.sum(clicks[window_7d_mask]) if any(window_7d_mask) else 0
                    ctr_7d = clicks_7d / imps_7d if imps_7d > 0 else 0
                    
                    # 14-day window - vectorized
                    window_14d_mask = timestamps >= (last_timestamp - np.timedelta64(14, 'D'))
                    imps_14d = np.sum(impressions[window_14d_mask]) if any(window_14d_mask) else 0
                    clicks_14d = np.sum(clicks[window_14d_mask]) if any(window_14d_mask) else 0
                    ctr_14d = clicks_14d / imps_14d if imps_14d > 0 else 0
                    
                    # 30-day window - vectorized
                    window_30d_mask = timestamps >= (last_timestamp - np.timedelta64(30, 'D'))
                    imps_30d = np.sum(impressions[window_30d_mask]) if any(window_30d_mask) else 0
                    clicks_30d = np.sum(clicks[window_30d_mask]) if any(window_30d_mask) else 0
                    ctr_30d = clicks_30d / imps_30d if imps_30d > 0 else 0
            
            # Calculate engagement consistency - vectorized
            if len(user_df) > 1 and 'timestamp' in user_df.columns:
                # Numpy operations for faster processing
                timestamps = user_df['timestamp'].values
                time_diffs = np.diff(timestamps) / np.timedelta64(1, 'D')  # in days
                
                engagement_consistency = 1.0 / (1.0 + np.std(time_diffs))
            else:
                engagement_consistency = 0
            
            # Calculate recency-weighted engagement - vectorized
            if len(user_df) > 0 and 'timestamp' in user_df.columns:
                most_recent = user_df['timestamp'].max()
                
                # Calculate days from most recent for each session - vectorized
                days_from_recent = (most_recent - user_df['timestamp'].values) / np.timedelta64(1, 'D')
                
                # Apply exponential decay weight - vectorized
                recency_weights = np.power(self.recency_weight_decay, days_from_recent)
                
                # Calculate weighted engagement metrics - vectorized
                weighted_clicks = np.sum(user_df['click_count'].values * recency_weights)
                weighted_impressions = np.sum(user_df['impressions_count'].values * recency_weights)
                
                recency_weighted_ctr = weighted_clicks / weighted_impressions if weighted_impressions > 0 else 0
            else:
                recency_weighted_ctr = 0
            
            # Compile features
            user_features.update({
                'ctr_7d': ctr_7d,
                'ctr_14d': ctr_14d,
                'ctr_30d': ctr_30d,
                'engagement_consistency': engagement_consistency,
                'recency_weighted_ctr': recency_weighted_ctr
            })
            
            return user_features
        
        # Group data by user
        user_groups = list(behaviors_df.groupby('user_id'))
        
        # Use parallel processing with process pool
        num_cores = min(mp.cpu_count(), 8)  # Use up to 8 cores
        print(f"Using {num_cores} cores for engagement feature extraction")
        
        # Process users in parallel
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(process_user_engagement, user_groups)
        
        # Filter out None values and convert to DataFrame
        results = [r for r in results if r is not None]
        engagement_features = pd.DataFrame(results)
        
        if engagement_features.empty:
            return pd.DataFrame()
            
        # Set user_id as index
        engagement_features.set_index('user_id', inplace=True)
        
        return engagement_features
    
    def combine_features(self, user_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple user feature DataFrames.
        
        Args:
            user_dfs: List of DataFrames containing user features
            
        Returns:
            Combined DataFrame with all user features
        """
        print("Combining all user features...")
        
        # Start with the first DataFrame
        if not user_dfs:
            return pd.DataFrame()
        
        result = user_dfs[0].copy()
        
        # Join with other DataFrames
        for df in user_dfs[1:]:
            result = result.join(df, how='outer')
        
        # Fill missing values
        result = result.fillna(0)
        
        return result
    
    def extract_all_features(self, behaviors_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all user features.
        
        Args:
            behaviors_df: DataFrame containing user behaviors
            news_df: DataFrame containing news articles metadata
            
        Returns:
            DataFrame with all user features
        """
        print("Extracting all user features in parallel...")
        import concurrent.futures
        
        # Define a function to preprocess behaviors_df first to avoid duplicating work
        def preprocess_behaviors(df):
            if 'timestamp' not in df.columns and 'time' in df.columns:
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['time'])
                
            if 'day_of_week' not in df.columns and 'timestamp' in df.columns:
                df = df.copy()
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['hour_of_day'] = df['timestamp'].dt.hour
                
            return df
        
        # Preprocess data once
        behaviors_df = preprocess_behaviors(behaviors_df)
        
        # Extract all features in parallel using ThreadPoolExecutor
        # Use ThreadPoolExecutor since these operations are I/O bound and GIL won't be an issue
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all extraction tasks
            basic_future = executor.submit(self.extract_basic_engagement_features, behaviors_df)
            temporal_future = executor.submit(self.extract_temporal_features, behaviors_df)
            content_future = executor.submit(self.extract_content_preference_features, behaviors_df, news_df)
            engagement_future = executor.submit(self.extract_engagement_pattern_features, behaviors_df)
            
            # Get results as they complete
            basic_features = basic_future.result()
            temporal_features = temporal_future.result()
            content_features = content_future.result()
            engagement_features = engagement_future.result()
        
        # Combine all features
        all_features = self.combine_features([
            basic_features, 
            temporal_features, 
            content_features, 
            engagement_features
        ])
        
        print(f"Generated {all_features.shape[1]} features for {all_features.shape[0]} users")
        
        return all_features
    
    def scale_features(self, features_df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale user features.
        
        Args:
            features_df: DataFrame containing user features
            method: Scaling method ('standard' or 'minmax')
            
        Returns:
            DataFrame with scaled user features
        """
        print(f"Scaling features using {method} scaling...")
        
        # Identify numeric columns
        numeric_cols = features_df.select_dtypes(include=['int64', 'float64']).columns
        
        # Initialize scaler
        if method == 'standard':
            self.feature_scaler = StandardScaler()
        elif method == 'minmax':
            self.feature_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Scale numeric features
        scaled_features = features_df.copy()
        scaled_features[numeric_cols] = self.feature_scaler.fit_transform(features_df[numeric_cols])
        
        return scaled_features
    
    def save(self, output_dir: str) -> None:
        """
        Save the fitted feature extractors to disk.
        
        Args:
            output_dir: Directory to save the feature extractors
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature scaler
        if self.feature_scaler is not None:
            with open(output_dir / 'feature_scaler.pkl', 'wb') as f:
                pickle.dump(self.feature_scaler, f)
        
        # Save configuration
        with open(output_dir / 'user_feature_extractor_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"User feature extractor saved to {output_dir}")
    
    def load(self, model_dir: str) -> 'UserFeatureExtractor':
        """
        Load saved feature extractors from disk.
        
        Args:
            model_dir: Directory containing saved feature extractors
            
        Returns:
            Self for method chaining
        """
        model_dir = Path(model_dir)
        
        # Load feature scaler
        if (model_dir / 'feature_scaler.pkl').exists():
            with open(model_dir / 'feature_scaler.pkl', 'rb') as f:
                self.feature_scaler = pickle.load(f)
        
        # Load configuration
        if (model_dir / 'user_feature_extractor_config.json').exists():
            with open(model_dir / 'user_feature_extractor_config.json', 'r') as f:
                self.config = json.load(f)
                
                # Update instance variables
                self.min_history_length = self.config.get('min_history_length', 5)
                self.temporal_window_days = self.config.get('temporal_window_days', 7)
                self.recency_weight_decay = self.config.get('recency_weight_decay', 0.85)
        
        print(f"User feature extractor loaded from {model_dir}")
        
        return self


# Example usage
if __name__ == "__main__":
    # Load config from file
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "../../config/pipeline_config.yaml"
    
    # Initialize feature extractor
    feature_extractor = UserFeatureExtractor(config_path)
    
    # Sample behavior data (would typically come from Parquet files)
    behaviors_data = {
        'user_id': ['U1', 'U1', 'U1', 'U2', 'U2', 'U3'],
        'impression_id': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6'],
        'time': ['2023-01-01 08:00:00', '2023-01-02 12:30:00', '2023-01-03 18:15:00', 
                '2023-01-01 09:45:00', '2023-01-03 14:20:00', '2023-01-02 22:10:00'],
        'history_length': [3, 5, 6, 2, 4, 1],
        'impressions_count': [10, 8, 12, 6, 9, 5],
        'click_count': [2, 3, 4, 1, 2, 0],
        'click_ratio': [0.2, 0.375, 0.333, 0.167, 0.222, 0.0],
        'history': [['N1', 'N2', 'N3'], ['N1', 'N2', 'N3', 'N4', 'N5'], ['N1', 'N2', 'N3', 'N4', 'N5', 'N6'],
                   ['N7', 'N8'], ['N7', 'N8', 'N9', 'N10'], ['N11']],
        'impressions': [[{'news_id': 'N4', 'clicked': 1}, {'news_id': 'N5', 'clicked': 0}],
                       [{'news_id': 'N6', 'clicked': 1}, {'news_id': 'N7', 'clicked': 1}],
                       [{'news_id': 'N8', 'clicked': 1}, {'news_id': 'N9', 'clicked': 0}],
                       [{'news_id': 'N9', 'clicked': 0}, {'news_id': 'N10', 'clicked': 1}],
                       [{'news_id': 'N11', 'clicked': 1}, {'news_id': 'N12', 'clicked': 0}],
                       [{'news_id': 'N13', 'clicked': 0}, {'news_id': 'N14', 'clicked': 0}]]
    }
    
    behaviors_df = pd.DataFrame(behaviors_data)
    behaviors_df['timestamp'] = pd.to_datetime(behaviors_df['time'])
    behaviors_df['day_of_week'] = behaviors_df['timestamp'].dt.dayofweek
    behaviors_df['hour_of_day'] = behaviors_df['timestamp'].dt.hour
    
    # Sample news data
    news_data = {
        'news_id': ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14'],
        'category': ['sports', 'tech', 'politics', 'entertainment', 'health', 'tech', 'sports', 
                    'politics', 'entertainment', 'health', 'tech', 'sports', 'politics', 'entertainment'],
        'subcategory': ['football', 'ai', 'elections', 'movies', 'wellness', 'gadgets', 'basketball', 
                       'policy', 'music', 'nutrition', 'software', 'tennis', 'government', 'tv']
    }
    
    news_df = pd.DataFrame(news_data)
    
    # Extract basic engagement features
    basic_features = feature_extractor.extract_basic_engagement_features(behaviors_df)
    print("\nBasic engagement features:")
    print(basic_features.head())
    
    # Extract temporal features
    temporal_features = feature_extractor.extract_temporal_features(behaviors_df)
    print("\nTemporal features:")
    print(temporal_features.head())
    
    # Extract content preference features
    content_features = feature_extractor.extract_content_preference_features(behaviors_df, news_df)
    print("\nContent preference features:")
    print(content_features.head())
    
    # Extract all features
    all_features = feature_extractor.extract_all_features(behaviors_df, news_df)
    print("\nAll features combined:")
    print(all_features.head())
    
    # Scale features
    scaled_features = feature_extractor.scale_features(all_features)
    print("\nScaled features:")
    print(scaled_features.head())
