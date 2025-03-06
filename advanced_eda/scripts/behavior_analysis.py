#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Behavioral Analysis for MIND Dataset
----------------------------------
Functions to analyze and visualize user behaviors and engagement patterns.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_time_features(behaviors_df: pd.DataFrame, time_column: str = "time") -> pd.DataFrame:
    """
    Extract time-related features from behaviors.
    
    Args:
        behaviors_df: DataFrame containing user behaviors
        time_column: Column containing datetime information
        
    Returns:
        DataFrame with time features added
    """
    if time_column not in behaviors_df.columns:
        logger.warning(f"Time column '{time_column}' not found in DataFrame")
        return behaviors_df
    
    df = behaviors_df.copy()
    
    # Ensure time column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        try:
            df[time_column] = pd.to_datetime(df[time_column])
        except Exception as e:
            logger.error(f"Error converting time to datetime: {e}")
            return behaviors_df
    
    # Extract time features
    df['hour'] = df[time_column].dt.hour
    df['day'] = df[time_column].dt.day
    df['day_of_week'] = df[time_column].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_name'] = df[time_column].dt.day_name()
    df['week'] = df[time_column].dt.isocalendar().week
    df['month'] = df[time_column].dt.month
    df['year'] = df[time_column].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday or Sunday
    
    # Time of day categories
    def categorize_time_of_day(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 22:
            return "Evening"
        else:
            return "Night"
            
    df['time_of_day'] = df['hour'].apply(categorize_time_of_day)
    
    return df


def calculate_engagement_metrics(behaviors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate user engagement metrics from behaviors.
    
    Args:
        behaviors_df: DataFrame containing user behaviors
        
    Returns:
        DataFrame with engagement metrics added
    """
    df = behaviors_df.copy()
    
    # Calculate history length (articles previously read)
    if 'history' in df.columns:
        df['history_length'] = df['history'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Calculate impression metrics
    if 'impressions' in df.columns:
        # Count impressions
        df['impression_count'] = df['impressions'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Count clicks (if available)
        df['clicked_count'] = df['impressions'].apply(
            lambda x: sum(1 for imp in x if isinstance(imp, dict) and imp.get('clicked') == 1) 
            if isinstance(x, list) else 0
        )
        
        # Calculate click-through rate
        df['click_rate'] = df.apply(
            lambda row: row['clicked_count'] / row['impression_count'] if row['impression_count'] > 0 else 0,
            axis=1
        )
    
    # Categorize engagement level
    if 'history_length' in df.columns:
        def categorize_engagement(length):
            if length == 0:
                return "No Engagement"
            elif length <= 5:
                return "Low"
            elif length <= 15:
                return "Medium"
            elif length <= 30:
                return "High"
            else:
                return "Very High"
                
        df['engagement_level'] = df['history_length'].apply(categorize_engagement)
    
    return df


def analyze_user_activity(behaviors_df: pd.DataFrame, 
                        time_column: str = "time",
                        user_id_column: str = "user_id") -> pd.DataFrame:
    """
    Analyze user activity patterns over time.
    
    Args:
        behaviors_df: DataFrame containing user behaviors
        time_column: Column containing datetime information
        user_id_column: Column containing user IDs
        
    Returns:
        DataFrame with activity counts aggregated by time periods
    """
    if time_column not in behaviors_df.columns or user_id_column not in behaviors_df.columns:
        logger.warning(f"Required columns not found in DataFrame")
        return pd.DataFrame()
    
    # Ensure time column is in datetime format
    df = behaviors_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        try:
            df[time_column] = pd.to_datetime(df[time_column])
        except Exception as e:
            logger.error(f"Error converting time to datetime: {e}")
            return pd.DataFrame()
    
    # Activity by hour
    hourly_activity = df.groupby(df[time_column].dt.hour).size().reset_index()
    hourly_activity.columns = ['hour', 'activity_count']
    
    # Activity by day of week
    daily_activity = df.groupby(df[time_column].dt.dayofweek).size().reset_index()
    daily_activity.columns = ['day_of_week', 'activity_count']
    daily_activity['day_name'] = daily_activity['day_of_week'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    
    # Unique users by day
    unique_users_by_day = df.groupby(df[time_column].dt.date)[user_id_column].nunique().reset_index()
    unique_users_by_day.columns = ['date', 'unique_users']
    
    # Return as dictionary of DataFrames
    return {
        'hourly_activity': hourly_activity,
        'daily_activity': daily_activity,
        'unique_users_by_day': unique_users_by_day
    }


def analyze_click_patterns(behaviors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze patterns in article clicks from impressions.
    
    Args:
        behaviors_df: DataFrame containing user behaviors with impressions
        
    Returns:
        Dictionary of DataFrames with click pattern analysis
    """
    if 'impressions' not in behaviors_df.columns:
        logger.warning("Impressions column not found in DataFrame")
        return {}
    
    # Extract all impressions
    all_impressions = []
    
    for idx, row in behaviors_df.iterrows():
        impressions = row['impressions']
        if not isinstance(impressions, list):
            continue
            
        for i, impression in enumerate(impressions):
            if not isinstance(impression, dict):
                continue
                
            # Add impression with position and user info
            impression_dict = impression.copy()
            impression_dict.update({
                'position': i,  # Position in impression list
                'user_id': row.get('user_id', 'unknown'),
                'impression_id': row.get('impression_id', idx)
            })
            all_impressions.append(impression_dict)
    
    # Convert to DataFrame
    impressions_df = pd.DataFrame(all_impressions)
    
    if len(impressions_df) == 0:
        logger.warning("No valid impressions found")
        return {}
        
    # Click rate by position
    click_by_position = impressions_df.groupby('position').agg(
        total=('news_id', 'count'),
        clicked=('clicked', lambda x: (x == 1).sum())
    ).reset_index()
    
    click_by_position['click_rate'] = click_by_position['clicked'] / click_by_position['total']
    
    # Most clicked news articles
    top_clicks = impressions_df[impressions_df['clicked'] == 1].groupby('news_id').size().reset_index()
    top_clicks.columns = ['news_id', 'click_count']
    top_clicks = top_clicks.sort_values('click_count', ascending=False)
    
    # Click rate by user (active users)
    user_clicks = impressions_df.groupby('user_id').agg(
        total=('news_id', 'count'),
        clicked=('clicked', lambda x: (x == 1).sum())
    ).reset_index()
    
    user_clicks['click_rate'] = user_clicks['clicked'] / user_clicks['total']
    
    return {
        'click_by_position': click_by_position,
        'top_clicked_news': top_clicks,
        'user_click_rates': user_clicks
    }


def analyze_user_interests(behaviors_df: pd.DataFrame, news_df: pd.DataFrame,
                         user_id_column: str = "user_id") -> Dict[str, pd.DataFrame]:
    """
    Analyze user interests based on history and clicks.
    
    Args:
        behaviors_df: DataFrame containing user behaviors
        news_df: DataFrame containing news articles
        user_id_column: Column containing user IDs
        
    Returns:
        Dictionary of DataFrames with user interest analysis
    """
    if user_id_column not in behaviors_df.columns:
        logger.warning(f"User ID column not found in DataFrame")
        return {}
    
    if 'history' not in behaviors_df.columns:
        logger.warning("History column not found in DataFrame")
        return {}
        
    # Get news ID to category mapping
    if 'news_id' in news_df.columns and 'category' in news_df.columns:
        news_categories = news_df.set_index('news_id')['category'].to_dict()
    else:
        logger.warning("Required columns not found in news DataFrame")
        return {}
    
    # Extract user interests from history
    user_interests = {}
    
    for _, row in behaviors_df.iterrows():
        user_id = row[user_id_column]
        history = row['history']
        
        if not isinstance(history, list):
            continue
            
        if user_id not in user_interests:
            user_interests[user_id] = []
            
        # Add categories from history
        for news_id in history:
            category = news_categories.get(news_id)
            if category:
                user_interests[user_id].append(category)
    
    # Convert to DataFrame for each user
    user_category_counts = {}
    
    for user_id, categories in user_interests.items():
        category_counts = pd.Series(categories).value_counts().reset_index()
        category_counts.columns = ['category', 'count']
        category_counts['frequency'] = category_counts['count'] / category_counts['count'].sum()
        user_category_counts[user_id] = category_counts
    
    # Aggregate across all users
    all_categories = []
    
    for user_id, df in user_category_counts.items():
        for _, row in df.iterrows():
            all_categories.append({
                'user_id': user_id,
                'category': row['category'],
                'count': row['count'],
                'frequency': row['frequency']
            })
    
    # Convert to DataFrame
    all_categories_df = pd.DataFrame(all_categories)
    
    # Category popularity across all users
    category_popularity = all_categories_df.groupby('category')['count'].sum().reset_index()
    category_popularity.columns = ['category', 'total_count']
    category_popularity = category_popularity.sort_values('total_count', ascending=False)
    
    return {
        'user_category_counts': user_category_counts,
        'category_popularity': category_popularity,
        'all_categories': all_categories_df
    }


def create_user_item_matrix(behaviors_df: pd.DataFrame, 
                          user_id_column: str = "user_id") -> pd.DataFrame:
    """
    Create a user-item interaction matrix from behaviors.
    
    Args:
        behaviors_df: DataFrame containing user behaviors
        user_id_column: Column containing user IDs
        
    Returns:
        DataFrame with users as rows, news items as columns, and values indicating interactions
    """
    if user_id_column not in behaviors_df.columns:
        logger.warning(f"User ID column not found in DataFrame")
        return pd.DataFrame()
        
    if 'history' not in behaviors_df.columns and 'impressions' not in behaviors_df.columns:
        logger.warning("Neither history nor impressions columns found in DataFrame")
        return pd.DataFrame()
    
    # Extract all user-item interactions
    interactions = []
    
    for _, row in behaviors_df.iterrows():
        user_id = row[user_id_column]
        
        # Add history items (implicit feedback - read)
        if 'history' in behaviors_df.columns and isinstance(row['history'], list):
            for news_id in row['history']:
                interactions.append({
                    'user_id': user_id,
                    'news_id': news_id,
                    'interaction': 1  # 1 = history (read)
                })
        
        # Add impression items (explicit feedback - clicked or not)
        if 'impressions' in behaviors_df.columns and isinstance(row['impressions'], list):
            for impression in row['impressions']:
                if isinstance(impression, dict) and 'news_id' in impression:
                    clicked = impression.get('clicked')
                    if clicked is not None:
                        interactions.append({
                            'user_id': user_id,
                            'news_id': impression['news_id'],
                            'interaction': 2 if clicked == 1 else 0  # 2 = clicked, 0 = not clicked
                        })
    
    if not interactions:
        logger.warning("No valid interactions found")
        return pd.DataFrame()
        
    # Create DataFrame from interactions
    interactions_df = pd.DataFrame(interactions)
    
    # Create pivot table (user-item matrix)
    user_item_matrix = interactions_df.pivot_table(
        index='user_id',
        columns='news_id',
        values='interaction',
        fill_value=0
    )
    
    return user_item_matrix


def identify_user_segments(behaviors_df: pd.DataFrame, 
                         user_id_column: str = "user_id") -> pd.DataFrame:
    """
    Identify user segments based on engagement patterns.
    
    Args:
        behaviors_df: DataFrame containing user behaviors with engagement metrics
        user_id_column: Column containing user IDs
        
    Returns:
        DataFrame with user segments
    """
    if user_id_column not in behaviors_df.columns:
        logger.warning(f"User ID column not found in DataFrame")
        return pd.DataFrame()
    
    # Calculate engagement metrics if not already present
    df = behaviors_df.copy()
    if 'history_length' not in df.columns or 'click_rate' not in df.columns:
        df = calculate_engagement_metrics(df)
    
    # Aggregate metrics by user
    user_metrics = df.groupby(user_id_column).agg({
        'history_length': 'max',
        'impression_count': 'sum',
        'clicked_count': 'sum',
        'click_rate': 'mean',
        'time': 'count'  # Count of sessions
    }).reset_index()
    
    user_metrics.rename(columns={'time': 'session_count'}, inplace=True)
    
    # Standardize metrics for segmentation
    from sklearn.preprocessing import StandardScaler
    
    feature_columns = ['history_length', 'impression_count', 'clicked_count', 'click_rate', 'session_count']
    features = user_metrics[feature_columns].fillna(0)
    
    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_scaled_df = pd.DataFrame(features_scaled, columns=feature_columns)
    except Exception as e:
        logger.error(f"Error scaling features: {e}")
        features_scaled_df = features
    
    # Apply clustering (K-means)
    try:
        from sklearn.cluster import KMeans
        
        # Determine optimal number of clusters (between 2 and 6)
        from sklearn.metrics import silhouette_score
        
        silhouette_scores = []
        k_values = range(2, 7)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled_df)
            silhouette_avg = silhouette_score(features_scaled_df, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find optimal k
        optimal_k = k_values[np.argmax(silhouette_scores)]
        
        # Apply K-means with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        user_metrics['segment'] = kmeans.fit_predict(features_scaled_df)
        
        # Add segment labels
        segment_profiles = user_metrics.groupby('segment').agg({
            'history_length': 'mean',
            'click_rate': 'mean',
            'session_count': 'mean'
        })
        
        # Create meaningful segment names
        segment_names = {}
        
        for segment_id, profile in segment_profiles.iterrows():
            if profile['history_length'] > segment_profiles['history_length'].median():
                if profile['click_rate'] > segment_profiles['click_rate'].median():
                    name = "Highly Engaged Clickers"
                else:
                    name = "Content Explorers"
            else:
                if profile['click_rate'] > segment_profiles['click_rate'].median():
                    name = "Selective Browsers"
                else:
                    name = "Casual Visitors"
                    
            # Add session frequency modifier
            if profile['session_count'] > segment_profiles['session_count'].median():
                name = f"Frequent {name}"
            else:
                name = f"Occasional {name}"
                
            segment_names[segment_id] = name
        
        user_metrics['segment_name'] = user_metrics['segment'].map(segment_names)
        
    except Exception as e:
        logger.error(f"Error during clustering: {e}")
        user_metrics['segment'] = 0
        user_metrics['segment_name'] = "All Users"
    
    return user_metrics


# Visualization Functions
def plot_time_patterns(activity_data: Dict[str, pd.DataFrame],
                     title_prefix: str = "User Activity",
                     figsize: Tuple[int, int] = (10, 6)) -> Dict[str, plt.Figure]:
    """
    Plot time-based activity patterns.
    
    Args:
        activity_data: Dictionary of DataFrames with time activity data
        title_prefix: Prefix for plot titles
        figsize: Figure size
        
    Returns:
        Dictionary of Matplotlib figures
    """
    figures = {}
    
    # Hourly activity
    if 'hourly_activity' in activity_data:
        hourly_df = activity_data['hourly_activity']
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x='hour', y='activity_count', data=hourly_df, palette='viridis', ax=ax)
        
        ax.set_title(f"{title_prefix} by Hour of Day")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Activity Count")
        ax.set_xticks(range(0, 24, 2))
        
        figures['hourly_activity'] = fig
    
    # Daily activity
    if 'daily_activity' in activity_data:
        daily_df = activity_data['daily_activity']
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            x='day_name', 
            y='activity_count', 
            data=daily_df,
            order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            palette='viridis',
            ax=ax
        )
        
        ax.set_title(f"{title_prefix} by Day of Week")
        ax.set_xlabel("Day")
        ax.set_ylabel("Activity Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        figures['daily_activity'] = fig
    
    # Unique users by day
    if 'unique_users_by_day' in activity_data:
        users_df = activity_data['unique_users_by_day']
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(x='date', y='unique_users', data=users_df, marker='o', ax=ax)
        
        ax.set_title(f"Unique Users by Date")
        ax.set_xlabel("Date")
        ax.set_ylabel("Unique Users")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        figures['unique_users'] = fig
    
    return figures


def plot_engagement_metrics(behaviors_df: pd.DataFrame,
                          figsize: Tuple[int, int] = (10, 6)) -> Dict[str, plt.Figure]:
    """
    Plot engagement metrics distribution.
    
    Args:
        behaviors_df: DataFrame containing user behaviors with engagement metrics
        figsize: Figure size
        
    Returns:
        Dictionary of Matplotlib figures
    """
    figures = {}
    
    # Calculate engagement metrics if not already present
    df = behaviors_df.copy()
    if 'history_length' not in df.columns or 'click_rate' not in df.columns:
        df = calculate_engagement_metrics(df)
    
    # History length distribution
    if 'history_length' in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(df['history_length'], bins=20, kde=True, ax=ax)
        
        ax.set_title("Distribution of Articles Read Per User")
        ax.set_xlabel("Number of Articles in History")
        ax.set_ylabel("Frequency")
        
        figures['history_length'] = fig
    
    # Click rate distribution
    if 'click_rate' in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(df['click_rate'], bins=20, kde=True, ax=ax)
        
        ax.set_title("Distribution of User Click Rates")
        ax.set_xlabel("Click Rate")
        ax.set_ylabel("Frequency")
        
        figures['click_rate'] = fig
    
    # Engagement level distribution
    if 'engagement_level' in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        order = ["No Engagement", "Low", "Medium", "High", "Very High"]
        sns.countplot(
            data=df, 
            x='engagement_level', 
            order=[level for level in order if level in df['engagement_level'].unique()],
            palette='viridis',
            ax=ax
        )
        
        ax.set_title("User Engagement Level Distribution")
        ax.set_xlabel("Engagement Level")
        ax.set_ylabel("Count")
        
        figures['engagement_level'] = fig
    
    # Correlation between history length and click rate
    if 'history_length' in df.columns and 'click_rate' in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(x='history_length', y='click_rate', data=df, alpha=0.6, ax=ax)
        
        ax.set_title("Relationship Between Reading History and Click Rate")
        ax.set_xlabel("Number of Articles in History")
        ax.set_ylabel("Click Rate")
        
        figures['history_vs_clicks'] = fig
    
    return figures


def plot_click_patterns(click_data: Dict[str, pd.DataFrame],
                      figsize: Tuple[int, int] = (10, 6)) -> Dict[str, plt.Figure]:
    """
    Plot click pattern analysis.
    
    Args:
        click_data: Dictionary of DataFrames with click pattern analysis
        figsize: Figure size
        
    Returns:
        Dictionary of Matplotlib figures
    """
    figures = {}
    
    # Click rate by position
    if 'click_by_position' in click_data:
        position_df = click_data['click_by_position']
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(x='position', y='click_rate', data=position_df, marker='o', ax=ax)
        
        ax.set_title("Click Rate by Position in Impression List")
        ax.set_xlabel("Position")
        ax.set_ylabel("Click Rate")
        
        figures['click_by_position'] = fig
    
    # Top clicked news
    if 'top_clicked_news' in click_data:
        top_clicks_df = click_data['top_clicked_news'].head(15)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x='click_count', y='news_id', data=top_clicks_df, palette='viridis', ax=ax)
        
        ax.set_title("Top 15 Most Clicked News Articles")
        ax.set_xlabel("Click Count")
        ax.set_ylabel("News ID")
        
        figures['top_clicks'] = fig
    
    # User click rate distribution
    if 'user_click_rates' in click_data:
        user_clicks_df = click_data['user_click_rates']
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(user_clicks_df['click_rate'], bins=20, kde=True, ax=ax)
        
        ax.set_title("Distribution of User Click Rates")
        ax.set_xlabel("Click Rate")
        ax.set_ylabel("Frequency")
        
        figures['user_click_rates'] = fig
    
    return figures


def plot_user_interests(interest_data: Dict[str, pd.DataFrame],
                      figsize: Tuple[int, int] = (10, 6)) -> Dict[str, plt.Figure]:
    """
    Plot user interest analysis.
    
    Args:
        interest_data: Dictionary of DataFrames with user interest analysis
        figsize: Figure size
        
    Returns:
        Dictionary of Matplotlib figures
    """
    figures = {}
    
    # Category popularity
    if 'category_popularity' in interest_data:
        category_df = interest_data['category_popularity']
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x='total_count', y='category', data=category_df, palette='viridis', ax=ax)
        
        ax.set_title("Category Popularity Across All Users")
        ax.set_xlabel("Total Count")
        ax.set_ylabel("Category")
        
        figures['category_popularity'] = fig
    
    # Category diversity by user
    if 'user_category_counts' in interest_data:
        user_counts = interest_data['user_category_counts']
        
        # Get diversity metrics
        diversity_metrics = []
        
        for user_id, df in user_counts.items():
            diversity_metrics.append({
                'user_id': user_id,
                'category_count': len(df),
                'entropy': -sum(row['frequency'] * np.log(row['frequency']) for _, row in df.iterrows())
            })
        
        diversity_df = pd.DataFrame(diversity_metrics)
        
        # Plot category count distribution
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(diversity_df['category_count'], bins=10, kde=True, ax=ax)
        
        ax.set_title("Distribution of Category Diversity per User")
        ax.set_xlabel("Number of Different Categories")
        ax.set_ylabel("Frequency")
        
        figures['category_diversity'] = fig
    
    # Plot all categories heatmap
    if 'all_categories' in interest_data:
        all_cat_df = interest_data['all_categories']
        
        # Get top users and categories for visualization
        top_users = all_cat_df.groupby('user_id')['count'].sum().nlargest(20).index
        top_categories = all_cat_df.groupby('category')['count'].sum().nlargest(10).index
        
        # Filter data for heatmap
        heatmap_data = all_cat_df[
            all_cat_df['user_id'].isin(top_users) & 
            all_cat_df['category'].isin(top_categories)
        ]
        
        # Pivot for heatmap
        heatmap_pivot = heatmap_data.pivot_table(
            index='user_id',
            columns='category',
            values='count',
            fill_value=0
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_pivot, cmap="YlGnBu", ax=ax)
        
        ax.set_title("Category Interest Heatmap (Top 20 Users)")
        ax.set_xlabel("Category")
        ax.set_ylabel("User ID")
        
        figures['category_heatmap'] = fig
    
    return figures


def plot_user_segments(user_segments: pd.DataFrame,
                     figsize: Tuple[int, int] = (10, 6)) -> Dict[str, plt.Figure]:
    """
    Plot user segment analysis.
    
    Args:
        user_segments: DataFrame containing user segment information
        figsize: Figure size
        
    Returns:
        Dictionary of Matplotlib figures
    """
    figures = {}
    
    if 'segment_name' not in user_segments.columns:
        logger.warning("Segment name column not found in DataFrame")
        return figures
    
    # Segment distribution
    fig, ax = plt.subplots(figsize=figsize)
    segment_counts = user_segments['segment_name'].value_counts()
    
    # Pie chart
    ax.pie(
        segment_counts, 
        labels=segment_counts.index, 
        autopct='%1.1f%%', 
        startangle=90,
        colors=plt.cm.tab10(np.arange(len(segment_counts)) % 10)
    )
    ax.axis('equal')
    ax.set_title("User Segments Distribution")
    
    figures['segment_distribution'] = fig
    
    # Segment profiles
    segment_profiles = user_segments.groupby('segment_name').agg({
        'history_length': 'mean',
        'click_rate': 'mean',
        'session_count': 'mean',
    }).reset_index()
    
    # Radar chart for segment profiles
    try:
        fig = plt.figure(figsize=(10, 8))
        
        # Standardize metrics for radar chart
        metrics = ['history_length', 'click_rate', 'session_count']
        min_max_scaler = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
        
        for col in metrics:
            segment_profiles[f'{col}_scaled'] = min_max_scaler(segment_profiles[col])
        
        # Number of variables
        N = len(metrics)
        
        # Create angle values
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create subplot with polar projection
        ax = fig.add_subplot(111, polar=True)
        
        # For each segment
        for i, (_, row) in enumerate(segment_profiles.iterrows()):
            values = [row[f'{col}_scaled'] for col in metrics]
            values += values[:1]  # Close the loop
            
            # Plot segment
            ax.plot(angles, values, linewidth=2, label=row['segment_name'])
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        ax.set_title("Segment Profiles Comparison")
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        figures['segment_profiles'] = fig
    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")
    
    return figures


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('/Users/sravansridhar/Documents/news_ai/advanced_eda/scripts')
    from mind_dataset_loader import MINDDatasetLoader
    
    # Load data
    loader = MINDDatasetLoader()
    news_df = loader.load_news()
    behaviors_df = loader.load_behaviors()
    
    # Calculate engagement metrics
    behaviors_df = calculate_engagement_metrics(behaviors_df)
    
    # Extract time features
    behaviors_df = extract_time_features(behaviors_df)
    
    # Analyze user activity
    activity_data = analyze_user_activity(behaviors_df)
    
    # Plot time patterns
    figures = plot_time_patterns(activity_data)
    plt.show()