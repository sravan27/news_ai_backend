"""
User Behavior Analysis module for MIND dataset.

This module provides functions to analyze user behaviors in the MIND dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import Counter
import calendar
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_clicks_and_impressions(behaviors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract clicks and impressions from the behaviors data.
    
    Args:
        behaviors_df: Behaviors DataFrame
        
    Returns:
        DataFrame with processed behaviors data
    """
    logger.info("Extracting clicks and impressions")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        behaviors_df_copy = behaviors_df.copy()
        
        # Process impressions
        def process_impressions(impressions_str):
            """Extract clicked and non-clicked news from impressions string."""
            if pd.isna(impressions_str):
                return [], []
                
            impressions = impressions_str.split()
            clicked = []
            non_clicked = []
            
            for impression in impressions:
                parts = impression.split('-')
                news_id = parts[0]
                click = int(parts[1])
                
                if click == 1:
                    clicked.append(news_id)
                else:
                    non_clicked.append(news_id)
                    
            return clicked, non_clicked
        
        # Apply the function
        behaviors_df_copy['Processed_Impressions'] = behaviors_df_copy['Impressions'].apply(process_impressions)
        
        # Extract clicked and non-clicked news
        behaviors_df_copy['Clicked_News'] = behaviors_df_copy['Processed_Impressions'].apply(lambda x: x[0])
        behaviors_df_copy['Non_Clicked_News'] = behaviors_df_copy['Processed_Impressions'].apply(lambda x: x[1])
        
        # Count clicks and impressions
        behaviors_df_copy['Click_Count'] = behaviors_df_copy['Clicked_News'].apply(len)
        behaviors_df_copy['Impression_Count'] = behaviors_df_copy['Impressions'].apply(
            lambda x: len(x.split()) if pd.notna(x) else 0
        )
        
        # Calculate click-through rate (CTR)
        behaviors_df_copy['CTR'] = behaviors_df_copy['Click_Count'] / behaviors_df_copy['Impression_Count']
        
        # Process history
        behaviors_df_copy['History_List'] = behaviors_df_copy['History'].apply(
            lambda x: x.split() if pd.notna(x) else []
        )
        behaviors_df_copy['History_Length'] = behaviors_df_copy['History_List'].apply(len)
        
        # Drop intermediate column
        behaviors_df_copy = behaviors_df_copy.drop(columns=['Processed_Impressions'])
        
        logger.info("Extraction complete")
        return behaviors_df_copy
    except Exception as e:
        logger.error(f"Error extracting clicks and impressions: {e}")
        raise

def analyze_user_engagement(behaviors_df: pd.DataFrame) -> Dict:
    """
    Analyze user engagement statistics.
    
    Args:
        behaviors_df: Behaviors DataFrame with processed clicks and impressions
        
    Returns:
        Dictionary with user engagement statistics
    """
    logger.info("Analyzing user engagement")
    
    try:
        # Check if clicks and impressions have been extracted
        if 'Click_Count' not in behaviors_df.columns:
            behaviors_df = extract_clicks_and_impressions(behaviors_df)
            
        # Calculate user-level statistics
        user_stats = behaviors_df.groupby('User_ID').agg({
            'Impression_Count': 'sum',
            'Click_Count': 'sum',
            'History_Length': 'mean'
        }).reset_index()
        
        # Calculate user-level CTR
        user_stats['User_CTR'] = user_stats['Click_Count'] / user_stats['Impression_Count']
        
        # Calculate overall statistics
        stats = {
            'total_users': len(user_stats),
            'total_impressions': user_stats['Impression_Count'].sum(),
            'total_clicks': user_stats['Click_Count'].sum(),
            'average_impressions_per_user': user_stats['Impression_Count'].mean(),
            'median_impressions_per_user': user_stats['Impression_Count'].median(),
            'average_clicks_per_user': user_stats['Click_Count'].mean(),
            'median_clicks_per_user': user_stats['Click_Count'].median(),
            'average_history_length': user_stats['History_Length'].mean(),
            'median_history_length': user_stats['History_Length'].median(),
            'average_ctr': user_stats['User_CTR'].mean(),
            'median_ctr': user_stats['User_CTR'].median()
        }
        
        # Calculate engagement segments
        user_stats['Engagement_Level'] = pd.cut(
            user_stats['History_Length'],
            bins=[-1, 0, 5, 15, float('inf')],
            labels=['None', 'Low', 'Medium', 'High']
        )
        
        # Count users by engagement level
        engagement_counts = user_stats['Engagement_Level'].value_counts().sort_index()
        
        # Add engagement level counts to stats
        for level, count in engagement_counts.items():
            stats[f'users_{level.lower()}_engagement'] = count
            stats[f'users_{level.lower()}_engagement_pct'] = (count / len(user_stats) * 100).round(2)
            
        logger.info("User engagement analysis complete")
        return stats
    except Exception as e:
        logger.error(f"Error analyzing user engagement: {e}")
        raise

def plot_engagement_distribution(behaviors_df: pd.DataFrame) -> plt.Figure:
    """
    Plot the distribution of user engagement levels.
    
    Args:
        behaviors_df: Behaviors DataFrame with processed clicks and impressions
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting user engagement distribution")
    
    try:
        # Check if clicks and impressions have been extracted
        if 'Click_Count' not in behaviors_df.columns:
            behaviors_df = extract_clicks_and_impressions(behaviors_df)
            
        # Calculate user-level statistics
        user_stats = behaviors_df.groupby('User_ID').agg({
            'Impression_Count': 'sum',
            'Click_Count': 'sum',
            'History_Length': 'mean'
        }).reset_index()
        
        # Calculate engagement levels
        user_stats['Engagement_Level'] = pd.cut(
            user_stats['History_Length'],
            bins=[-1, 0, 5, 15, float('inf')],
            labels=['None', 'Low', 'Medium', 'High']
        )
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count users by engagement level
        engagement_counts = user_stats['Engagement_Level'].value_counts().sort_index()
        
        # Create bar chart
        sns.barplot(
            x=engagement_counts.index,
            y=engagement_counts.values,
            palette='viridis',
            ax=ax1
        )
        
        # Set labels and title
        ax1.set_xlabel("Engagement Level")
        ax1.set_ylabel("Number of Users")
        ax1.set_title("Distribution of User Engagement Levels")
        
        # Add count and percentage text
        total = engagement_counts.sum()
        for i, count in enumerate(engagement_counts):
            percentage = (count / total * 100).round(1)
            ax1.text(i, count + 5, f"{count} ({percentage}%)", ha='center')
            
        # Create pie chart
        ax2.pie(
            engagement_counts,
            labels=engagement_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette('viridis', n_colors=len(engagement_counts))
        )
        ax2.set_title('User Engagement Level Proportions')
        
        plt.tight_layout()
        logger.info("Engagement distribution plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting engagement distribution: {e}")
        raise

def analyze_temporal_patterns(behaviors_df: pd.DataFrame) -> Dict:
    """
    Analyze temporal patterns in user behaviors.
    
    Args:
        behaviors_df: Behaviors DataFrame
        
    Returns:
        Dictionary with temporal pattern statistics
    """
    logger.info("Analyzing temporal patterns")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        behaviors_df_copy = behaviors_df.copy()
        
        # Ensure time is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(behaviors_df_copy['Time']):
            behaviors_df_copy['Time'] = pd.to_datetime(behaviors_df_copy['Time'])
            
        # Extract time components
        behaviors_df_copy['Hour'] = behaviors_df_copy['Time'].dt.hour
        behaviors_df_copy['Day'] = behaviors_df_copy['Time'].dt.day
        behaviors_df_copy['Weekday'] = behaviors_df_copy['Time'].dt.dayofweek
        behaviors_df_copy['Weekday_Name'] = behaviors_df_copy['Time'].dt.day_name()
        behaviors_df_copy['Month'] = behaviors_df_copy['Time'].dt.month
        behaviors_df_copy['Year'] = behaviors_df_copy['Time'].dt.year
        
        # Determine time of day
        def get_time_of_day(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
                
        behaviors_df_copy['Time_of_Day'] = behaviors_df_copy['Hour'].apply(get_time_of_day)
        
        # Check if clicks and impressions have been extracted
        if 'Click_Count' not in behaviors_df_copy.columns:
            behaviors_df_copy = extract_clicks_and_impressions(behaviors_df_copy)
            
        # Calculate statistics by time components
        hourly_stats = behaviors_df_copy.groupby('Hour').agg({
            'Impression_ID': 'count',
            'Click_Count': 'sum',
            'Impression_Count': 'sum'
        })
        hourly_stats['CTR'] = hourly_stats['Click_Count'] / hourly_stats['Impression_Count']
        
        weekday_stats = behaviors_df_copy.groupby('Weekday_Name').agg({
            'Impression_ID': 'count',
            'Click_Count': 'sum',
            'Impression_Count': 'sum'
        })
        weekday_stats['CTR'] = weekday_stats['Click_Count'] / weekday_stats['Impression_Count']
        
        time_of_day_stats = behaviors_df_copy.groupby('Time_of_Day').agg({
            'Impression_ID': 'count',
            'Click_Count': 'sum',
            'Impression_Count': 'sum'
        })
        time_of_day_stats['CTR'] = time_of_day_stats['Click_Count'] / time_of_day_stats['Impression_Count']
        
        # Find peak hours and days
        peak_hour = hourly_stats['Impression_ID'].idxmax()
        peak_weekday = weekday_stats['Impression_ID'].idxmax()
        peak_time_of_day = time_of_day_stats['Impression_ID'].idxmax()
        
        # Calculate overall statistics
        stats = {
            'peak_hour': int(peak_hour),
            'peak_weekday': peak_weekday,
            'peak_time_of_day': peak_time_of_day,
            'hourly_stats': hourly_stats.to_dict(),
            'weekday_stats': weekday_stats.to_dict(),
            'time_of_day_stats': time_of_day_stats.to_dict()
        }
        
        logger.info("Temporal pattern analysis complete")
        return stats
    except Exception as e:
        logger.error(f"Error analyzing temporal patterns: {e}")
        raise

def plot_hourly_activity(behaviors_df: pd.DataFrame) -> plt.Figure:
    """
    Plot user activity by hour of day.
    
    Args:
        behaviors_df: Behaviors DataFrame
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting hourly activity")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        behaviors_df_copy = behaviors_df.copy()
        
        # Ensure time is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(behaviors_df_copy['Time']):
            behaviors_df_copy['Time'] = pd.to_datetime(behaviors_df_copy['Time'])
            
        # Extract hour
        behaviors_df_copy['Hour'] = behaviors_df_copy['Time'].dt.hour
        
        # Count impressions by hour
        hourly_counts = behaviors_df_copy.groupby('Hour').size()
        
        # Check if clicks and impressions have been extracted
        if 'Click_Count' not in behaviors_df_copy.columns:
            behaviors_df_copy = extract_clicks_and_impressions(behaviors_df_copy)
            
        # Calculate CTR by hour
        hourly_stats = behaviors_df_copy.groupby('Hour').agg({
            'Click_Count': 'sum',
            'Impression_Count': 'sum'
        })
        hourly_stats['CTR'] = hourly_stats['Click_Count'] / hourly_stats['Impression_Count']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot impression counts
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette='viridis', ax=ax1)
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Number of Impressions")
        ax1.set_title("User Activity by Hour of Day")
        ax1.set_xticks(range(24))
        
        # Add horizontal grid lines
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Find peak hour
        peak_hour = hourly_counts.idxmax()
        peak_count = hourly_counts.max()
        
        # Highlight peak hour
        ax1.get_children()[peak_hour].set_facecolor('salmon')
        ax1.annotate(
            f'Peak: {peak_hour}:00-{peak_hour+1}:00\n{peak_count} impressions',
            xy=(peak_hour, peak_count),
            xytext=(peak_hour, peak_count + peak_count * 0.1),
            ha='center',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
        )
        
        # Plot CTR
        sns.lineplot(x=hourly_stats.index, y=hourly_stats['CTR'], marker='o', color='darkblue', ax=ax2)
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Click-Through Rate (CTR)")
        ax2.set_title("Click-Through Rate by Hour of Day")
        ax2.set_xticks(range(24))
        
        # Add horizontal grid lines
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Find peak CTR hour
        peak_ctr_hour = hourly_stats['CTR'].idxmax()
        peak_ctr = hourly_stats['CTR'].max()
        
        # Highlight peak CTR hour
        ax2.annotate(
            f'Peak CTR: {peak_ctr_hour}:00-{peak_ctr_hour+1}:00\n{peak_ctr:.2%}',
            xy=(peak_ctr_hour, peak_ctr),
            xytext=(peak_ctr_hour, peak_ctr + peak_ctr * 0.1),
            ha='center',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
        )
        
        plt.tight_layout()
        logger.info("Hourly activity plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting hourly activity: {e}")
        raise

def plot_weekday_activity(behaviors_df: pd.DataFrame) -> plt.Figure:
    """
    Plot user activity by day of week.
    
    Args:
        behaviors_df: Behaviors DataFrame
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting weekday activity")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        behaviors_df_copy = behaviors_df.copy()
        
        # Ensure time is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(behaviors_df_copy['Time']):
            behaviors_df_copy['Time'] = pd.to_datetime(behaviors_df_copy['Time'])
            
        # Extract weekday
        behaviors_df_copy['Weekday'] = behaviors_df_copy['Time'].dt.dayofweek
        behaviors_df_copy['Weekday_Name'] = behaviors_df_copy['Time'].dt.day_name()
        
        # Count impressions by weekday
        weekday_counts = behaviors_df_copy.groupby('Weekday_Name').size()
        
        # Order weekdays correctly
        days_order = list(calendar.day_name)
        if len(weekday_counts) > 0:
            weekday_counts = weekday_counts.reindex(days_order, fill_value=0)
        
        # Check if clicks and impressions have been extracted
        if 'Click_Count' not in behaviors_df_copy.columns:
            behaviors_df_copy = extract_clicks_and_impressions(behaviors_df_copy)
            
        # Calculate CTR by weekday
        weekday_stats = behaviors_df_copy.groupby('Weekday_Name').agg({
            'Click_Count': 'sum',
            'Impression_Count': 'sum'
        })
        
        if len(weekday_stats) > 0:
            weekday_stats = weekday_stats.reindex(days_order, fill_value=0)
            weekday_stats['CTR'] = weekday_stats['Click_Count'] / weekday_stats['Impression_Count'].replace(0, np.nan)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot impression counts
        sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette='viridis', ax=ax1)
        ax1.set_xlabel("Day of Week")
        ax1.set_ylabel("Number of Impressions")
        ax1.set_title("User Activity by Day of Week")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Add horizontal grid lines
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Find peak day
        if len(weekday_counts) > 0:
            peak_day_idx = weekday_counts.values.argmax()
            peak_day = weekday_counts.index[peak_day_idx]
            peak_count = weekday_counts.max()
            
            # Highlight peak day
            ax1.get_children()[peak_day_idx].set_facecolor('salmon')
            ax1.annotate(
                f'Peak: {peak_day}\n{peak_count} impressions',
                xy=(peak_day_idx, peak_count),
                xytext=(peak_day_idx, peak_count + peak_count * 0.1),
                ha='center',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
            )
        
        # Plot CTR
        if len(weekday_stats) > 0 and not weekday_stats['CTR'].isna().all():
            sns.lineplot(x=weekday_stats.index, y=weekday_stats['CTR'], marker='o', color='darkblue', ax=ax2)
            ax2.set_xlabel("Day of Week")
            ax2.set_ylabel("Click-Through Rate (CTR)")
            ax2.set_title("Click-Through Rate by Day of Week")
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            
            # Add horizontal grid lines
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Find peak CTR day
            peak_ctr_day_idx = weekday_stats['CTR'].fillna(0).values.argmax()
            peak_ctr_day = weekday_stats.index[peak_ctr_day_idx]
            peak_ctr = weekday_stats['CTR'].max()
            
            # Highlight peak CTR day
            ax2.annotate(
                f'Peak CTR: {peak_ctr_day}\n{peak_ctr:.2%}',
                xy=(peak_ctr_day_idx, peak_ctr),
                xytext=(peak_ctr_day_idx, peak_ctr + peak_ctr * 0.1),
                ha='center',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
            )
        else:
            ax2.text(0.5, 0.5, "Insufficient data for CTR by weekday", 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        logger.info("Weekday activity plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting weekday activity: {e}")
        raise

def plot_time_of_day_activity(behaviors_df: pd.DataFrame) -> plt.Figure:
    """
    Plot user activity by time of day.
    
    Args:
        behaviors_df: Behaviors DataFrame
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting time of day activity")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        behaviors_df_copy = behaviors_df.copy()
        
        # Ensure time is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(behaviors_df_copy['Time']):
            behaviors_df_copy['Time'] = pd.to_datetime(behaviors_df_copy['Time'])
            
        # Extract hour
        behaviors_df_copy['Hour'] = behaviors_df_copy['Time'].dt.hour
        
        # Determine time of day
        def get_time_of_day(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
                
        behaviors_df_copy['Time_of_Day'] = behaviors_df_copy['Hour'].apply(get_time_of_day)
        
        # Count impressions by time of day
        time_of_day_counts = behaviors_df_copy.groupby('Time_of_Day').size()
        
        # Order by time of day
        time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
        if len(time_of_day_counts) > 0:
            time_of_day_counts = time_of_day_counts.reindex(time_order, fill_value=0)
        
        # Check if clicks and impressions have been extracted
        if 'Click_Count' not in behaviors_df_copy.columns:
            behaviors_df_copy = extract_clicks_and_impressions(behaviors_df_copy)
            
        # Calculate CTR by time of day
        time_of_day_stats = behaviors_df_copy.groupby('Time_of_Day').agg({
            'Click_Count': 'sum',
            'Impression_Count': 'sum'
        })
        
        if len(time_of_day_stats) > 0:
            time_of_day_stats = time_of_day_stats.reindex(time_order, fill_value=0)
            time_of_day_stats['CTR'] = time_of_day_stats['Click_Count'] / time_of_day_stats['Impression_Count'].replace(0, np.nan)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot impression counts
        sns.barplot(x=time_of_day_counts.index, y=time_of_day_counts.values, palette='viridis', ax=ax1)
        ax1.set_xlabel("Time of Day")
        ax1.set_ylabel("Number of Impressions")
        ax1.set_title("User Activity by Time of Day")
        
        # Add horizontal grid lines
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Find peak time of day
        if len(time_of_day_counts) > 0:
            peak_time_idx = time_of_day_counts.values.argmax()
            peak_time = time_of_day_counts.index[peak_time_idx]
            peak_count = time_of_day_counts.max()
            
            # Highlight peak time
            ax1.get_children()[peak_time_idx].set_facecolor('salmon')
            ax1.annotate(
                f'Peak: {peak_time}\n{peak_count} impressions',
                xy=(peak_time_idx, peak_count),
                xytext=(peak_time_idx, peak_count + peak_count * 0.1),
                ha='center',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
            )
        
        # Plot CTR
        if len(time_of_day_stats) > 0 and not time_of_day_stats['CTR'].isna().all():
            sns.lineplot(x=time_of_day_stats.index, y=time_of_day_stats['CTR'], marker='o', color='darkblue', ax=ax2)
            ax2.set_xlabel("Time of Day")
            ax2.set_ylabel("Click-Through Rate (CTR)")
            ax2.set_title("Click-Through Rate by Time of Day")
            
            # Add horizontal grid lines
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Find peak CTR time
            peak_ctr_time_idx = time_of_day_stats['CTR'].fillna(0).values.argmax()
            peak_ctr_time = time_of_day_stats.index[peak_ctr_time_idx]
            peak_ctr = time_of_day_stats['CTR'].max()
            
            # Highlight peak CTR time
            ax2.annotate(
                f'Peak CTR: {peak_ctr_time}\n{peak_ctr:.2%}',
                xy=(peak_ctr_time_idx, peak_ctr),
                xytext=(peak_ctr_time_idx, peak_ctr + peak_ctr * 0.1),
                ha='center',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
            )
        else:
            ax2.text(0.5, 0.5, "Insufficient data for CTR by time of day", 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        logger.info("Time of day activity plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting time of day activity: {e}")
        raise

def analyze_click_through_rate(behaviors_df: pd.DataFrame) -> Dict:
    """
    Analyze click-through rate (CTR) statistics.
    
    Args:
        behaviors_df: Behaviors DataFrame
        
    Returns:
        Dictionary with CTR statistics
    """
    logger.info("Analyzing click-through rate")
    
    try:
        # Check if clicks and impressions have been extracted
        if 'Click_Count' not in behaviors_df.columns:
            behaviors_df = extract_clicks_and_impressions(behaviors_df)
            
        # Calculate overall CTR
        total_clicks = behaviors_df['Click_Count'].sum()
        total_impressions = behaviors_df['Impression_Count'].sum()
        overall_ctr = total_clicks / total_impressions if total_impressions > 0 else 0
        
        # Calculate user-level CTR
        user_ctr = behaviors_df.groupby('User_ID').agg({
            'Click_Count': 'sum',
            'Impression_Count': 'sum'
        })
        user_ctr['CTR'] = user_ctr['Click_Count'] / user_ctr['Impression_Count']
        
        # Calculate CTR statistics
        stats = {
            'overall_ctr': overall_ctr,
            'average_user_ctr': user_ctr['CTR'].mean(),
            'median_user_ctr': user_ctr['CTR'].median(),
            'min_user_ctr': user_ctr['CTR'].min(),
            'max_user_ctr': user_ctr['CTR'].max(),
            'std_user_ctr': user_ctr['CTR'].std(),
            'total_clicks': total_clicks,
            'total_impressions': total_impressions
        }
        
        # Calculate CTR distribution
        ctr_bins = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        user_ctr['CTR_Bin'] = pd.cut(user_ctr['CTR'], bins=ctr_bins)
        ctr_distribution = user_ctr['CTR_Bin'].value_counts().sort_index()
        
        # Add CTR distribution to stats
        stats['ctr_distribution'] = {
            str(bin_): count for bin_, count in zip(ctr_distribution.index.astype(str), ctr_distribution.values)
        }
        
        logger.info("CTR analysis complete")
        return stats
    except Exception as e:
        logger.error(f"Error analyzing CTR: {e}")
        raise

def plot_ctr_distribution(behaviors_df: pd.DataFrame) -> plt.Figure:
    """
    Plot the distribution of user click-through rates.
    
    Args:
        behaviors_df: Behaviors DataFrame
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting CTR distribution")
    
    try:
        # Check if clicks and impressions have been extracted
        if 'Click_Count' not in behaviors_df.columns:
            behaviors_df = extract_clicks_and_impressions(behaviors_df)
            
        # Calculate user-level CTR
        user_ctr = behaviors_df.groupby('User_ID').agg({
            'Click_Count': 'sum',
            'Impression_Count': 'sum'
        })
        user_ctr['CTR'] = user_ctr['Click_Count'] / user_ctr['Impression_Count']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot histogram
        sns.histplot(user_ctr['CTR'], bins=20, kde=True, ax=ax1)
        ax1.set_xlabel("Click-Through Rate (CTR)")
        ax1.set_ylabel("Number of Users")
        ax1.set_title("Distribution of User Click-Through Rates")
        
        # Add mean and median lines
        mean_ctr = user_ctr['CTR'].mean()
        median_ctr = user_ctr['CTR'].median()
        
        ax1.axvline(mean_ctr, color='red', linestyle='--', 
                   label=f'Mean: {mean_ctr:.2%}')
        ax1.axvline(median_ctr, color='green', linestyle=':', 
                   label=f'Median: {median_ctr:.2%}')
        
        ax1.legend()
        
        # Calculate CTR distribution
        ctr_bins = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        user_ctr['CTR_Bin'] = pd.cut(user_ctr['CTR'], bins=ctr_bins)
        ctr_distribution = user_ctr['CTR_Bin'].value_counts().sort_index()
        
        # Create bar chart
        sns.barplot(
            x=[str(bin_) for bin_ in ctr_distribution.index],
            y=ctr_distribution.values,
            palette='viridis',
            ax=ax2
        )
        ax2.set_xlabel("Click-Through Rate Range")
        ax2.set_ylabel("Number of Users")
        ax2.set_title("Distribution of User CTR Ranges")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # Add count and percentage text
        total = ctr_distribution.sum()
        for i, count in enumerate(ctr_distribution):
            percentage = (count / total * 100).round(1)
            ax2.text(i, count + 5, f"{count} ({percentage}%)", ha='center')
            
        plt.tight_layout()
        logger.info("CTR distribution plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting CTR distribution: {e}")
        raise

def analyze_history_patterns(behaviors_df: pd.DataFrame) -> Dict:
    """
    Analyze patterns in user browsing history.
    
    Args:
        behaviors_df: Behaviors DataFrame
        
    Returns:
        Dictionary with history pattern statistics
    """
    logger.info("Analyzing history patterns")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        behaviors_df_copy = behaviors_df.copy()
        
        # Process history
        behaviors_df_copy['History_List'] = behaviors_df_copy['History'].apply(
            lambda x: x.split() if pd.notna(x) else []
        )
        behaviors_df_copy['History_Length'] = behaviors_df_copy['History_List'].apply(len)
        
        # Calculate overall statistics
        history_stats = {
            'average_history_length': behaviors_df_copy['History_Length'].mean(),
            'median_history_length': behaviors_df_copy['History_Length'].median(),
            'min_history_length': behaviors_df_copy['History_Length'].min(),
            'max_history_length': behaviors_df_copy['History_Length'].max(),
            'std_history_length': behaviors_df_copy['History_Length'].std(),
            'users_with_no_history': (behaviors_df_copy['History_Length'] == 0).sum(),
            'users_with_no_history_pct': ((behaviors_df_copy['History_Length'] == 0).sum() / len(behaviors_df_copy) * 100).round(2)
        }
        
        # Collect all history items
        all_history_items = []
        for history_list in behaviors_df_copy['History_List']:
            all_history_items.extend(history_list)
            
        # Count most common news items in history
        history_counts = Counter(all_history_items)
        top_history_items = history_counts.most_common(10)
        
        # Add top history items to stats
        history_stats['top_history_items'] = [
            {'news_id': item[0], 'count': item[1]} for item in top_history_items
        ]
        
        # Calculate history length distribution
        history_bins = [0, 1, 5, 10, 20, 50, 100, float('inf')]
        behaviors_df_copy['History_Length_Bin'] = pd.cut(behaviors_df_copy['History_Length'], bins=history_bins)
        history_distribution = behaviors_df_copy['History_Length_Bin'].value_counts().sort_index()
        
        # Add history distribution to stats
        history_stats['history_length_distribution'] = {
            str(bin_): count for bin_, count in zip(history_distribution.index.astype(str), history_distribution.values)
        }
        
        logger.info("History pattern analysis complete")
        return history_stats
    except Exception as e:
        logger.error(f"Error analyzing history patterns: {e}")
        raise

def plot_history_length_distribution(behaviors_df: pd.DataFrame) -> plt.Figure:
    """
    Plot the distribution of user history lengths.
    
    Args:
        behaviors_df: Behaviors DataFrame
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting history length distribution")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        behaviors_df_copy = behaviors_df.copy()
        
        # Process history
        behaviors_df_copy['History_List'] = behaviors_df_copy['History'].apply(
            lambda x: x.split() if pd.notna(x) else []
        )
        behaviors_df_copy['History_Length'] = behaviors_df_copy['History_List'].apply(len)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot histogram
        sns.histplot(behaviors_df_copy['History_Length'], bins=20, kde=True, ax=ax1)
        ax1.set_xlabel("History Length (Number of Articles)")
        ax1.set_ylabel("Number of Users")
        ax1.set_title("Distribution of User History Lengths")
        
        # Add mean and median lines
        mean_length = behaviors_df_copy['History_Length'].mean()
        median_length = behaviors_df_copy['History_Length'].median()
        
        ax1.axvline(mean_length, color='red', linestyle='--', 
                   label=f'Mean: {mean_length:.2f}')
        ax1.axvline(median_length, color='green', linestyle=':', 
                   label=f'Median: {median_length:.2f}')
        
        ax1.legend()
        
        # Calculate history length distribution
        history_bins = [0, 1, 5, 10, 20, 50, 100, float('inf')]
        behaviors_df_copy['History_Length_Bin'] = pd.cut(behaviors_df_copy['History_Length'], bins=history_bins)
        history_distribution = behaviors_df_copy['History_Length_Bin'].value_counts().sort_index()
        
        # Create bar chart
        sns.barplot(
            x=[str(bin_) for bin_ in history_distribution.index],
            y=history_distribution.values,
            palette='viridis',
            ax=ax2
        )
        ax2.set_xlabel("History Length Range")
        ax2.set_ylabel("Number of Users")
        ax2.set_title("Distribution of User History Length Ranges")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # Add count and percentage text
        total = history_distribution.sum()
        for i, count in enumerate(history_distribution):
            percentage = (count / total * 100).round(1)
            ax2.text(i, count + 5, f"{count} ({percentage}%)", ha='center')
            
        plt.tight_layout()
        logger.info("History length distribution plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting history length distribution: {e}")
        raise

def analyze_user_categories(news_df: pd.DataFrame, behaviors_df: pd.DataFrame) -> Dict:
    """
    Analyze user preferences for news categories.
    
    Args:
        news_df: News DataFrame
        behaviors_df: Behaviors DataFrame
        
    Returns:
        Dictionary with user category preference statistics
    """
    logger.info("Analyzing user category preferences")
    
    try:
        # Create copies to avoid modifying the original DataFrames
        news_df_copy = news_df.copy()
        behaviors_df_copy = behaviors_df.copy()
        
        # Process history
        behaviors_df_copy['History_List'] = behaviors_df_copy['History'].apply(
            lambda x: x.split() if pd.notna(x) else []
        )
        
        # Check if clicks and impressions have been extracted
        if 'Clicked_News' not in behaviors_df_copy.columns:
            behaviors_df_copy = extract_clicks_and_impressions(behaviors_df_copy)
            
        # Create a mapping from news ID to category
        news_categories = dict(zip(news_df_copy['News_ID'], news_df_copy['Category']))
        
        # Function to get categories for a list of news IDs
        def get_categories(news_ids):
            return [news_categories.get(news_id) for news_id in news_ids if news_id in news_categories]
            
        # Get categories from history and clicks
        behaviors_df_copy['History_Categories'] = behaviors_df_copy['History_List'].apply(get_categories)
        behaviors_df_copy['Clicked_Categories'] = behaviors_df_copy['Clicked_News'].apply(get_categories)
        
        # Count categories in history
        all_history_categories = []
        for categories in behaviors_df_copy['History_Categories']:
            all_history_categories.extend(categories)
            
        history_category_counts = Counter(all_history_categories)
        
        # Count categories in clicks
        all_clicked_categories = []
        for categories in behaviors_df_copy['Clicked_Categories']:
            all_clicked_categories.extend(categories)
            
        clicked_category_counts = Counter(all_clicked_categories)
        
        # Calculate overall statistics
        total_history = sum(history_category_counts.values())
        total_clicks = sum(clicked_category_counts.values())
        
        history_category_pct = {
            category: (count / total_history * 100).round(2) 
            for category, count in history_category_counts.items()
        }
        
        clicked_category_pct = {
            category: (count / total_clicks * 100).round(2) 
            for category, count in clicked_category_counts.items()
        }
        
        # Calculate category preference score (ratio of clicks to history)
        preference_score = {}
        for category in set(list(history_category_counts.keys()) + list(clicked_category_counts.keys())):
            history_count = history_category_counts.get(category, 0)
            clicked_count = clicked_category_counts.get(category, 0)
            
            if history_count > 0:
                preference_score[category] = clicked_count / history_count
            else:
                preference_score[category] = 0
                
        # Calculate statistics
        stats = {
            'history_category_counts': dict(history_category_counts),
            'history_category_pct': history_category_pct,
            'clicked_category_counts': dict(clicked_category_counts),
            'clicked_category_pct': clicked_category_pct,
            'category_preference_score': preference_score
        }
        
        logger.info("User category preference analysis complete")
        return stats
    except Exception as e:
        logger.error(f"Error analyzing user category preferences: {e}")
        raise

def plot_category_preferences(news_df: pd.DataFrame, behaviors_df: pd.DataFrame) -> plt.Figure:
    """
    Plot user preferences for news categories.
    
    Args:
        news_df: News DataFrame
        behaviors_df: Behaviors DataFrame
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting category preferences")
    
    try:
        # Analyze user categories
        category_stats = analyze_user_categories(news_df, behaviors_df)
        
        # Convert to DataFrames
        history_df = pd.DataFrame({
            'Category': list(category_stats['history_category_counts'].keys()),
            'Count': list(category_stats['history_category_counts'].values()),
            'Percentage': list(category_stats['history_category_pct'].values())
        }).sort_values('Count', ascending=False)
        
        clicked_df = pd.DataFrame({
            'Category': list(category_stats['clicked_category_counts'].keys()),
            'Count': list(category_stats['clicked_category_counts'].values()),
            'Percentage': list(category_stats['clicked_category_pct'].values())
        }).sort_values('Count', ascending=False)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot history categories
        sns.barplot(x='Percentage', y='Category', data=history_df.head(10), palette='viridis', ax=ax1)
        ax1.set_xlabel("Percentage")
        ax1.set_ylabel("Category")
        ax1.set_title("Top Categories in User History")
        
        # Add percentage text
        for i, row in enumerate(history_df.head(10).itertuples()):
            ax1.text(row.Percentage + 0.5, i, f"{row.Percentage}%", va='center')
            
        # Plot clicked categories
        sns.barplot(x='Percentage', y='Category', data=clicked_df.head(10), palette='viridis', ax=ax2)
        ax2.set_xlabel("Percentage")
        ax2.set_ylabel("Category")
        ax2.set_title("Top Categories in User Clicks")
        
        # Add percentage text
        for i, row in enumerate(clicked_df.head(10).itertuples()):
            ax2.text(row.Percentage + 0.5, i, f"{row.Percentage}%", va='center')
            
        plt.tight_layout()
        logger.info("Category preferences plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting category preferences: {e}")
        raise

def analyze_user_segments(behaviors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment users based on their behaviors.
    
    Args:
        behaviors_df: Behaviors DataFrame
        
    Returns:
        DataFrame with user segments
    """
    logger.info("Analyzing user segments")
    
    try:
        # Check if clicks and impressions have been extracted
        if 'Click_Count' not in behaviors_df.columns:
            behaviors_df = extract_clicks_and_impressions(behaviors_df)
            
        # Calculate user-level statistics
        user_stats = behaviors_df.groupby('User_ID').agg({
            'Impression_Count': 'sum',
            'Click_Count': 'sum',
            'History_Length': 'mean',
            'Time': 'count'  # Number of sessions
        }).reset_index()
        
        # Calculate CTR
        user_stats['CTR'] = user_stats['Click_Count'] / user_stats['Impression_Count']
        
        # Rename column
        user_stats = user_stats.rename(columns={'Time': 'Session_Count'})
        
        # Define engagement segments
        user_stats['Engagement_Level'] = pd.cut(
            user_stats['History_Length'],
            bins=[-1, 0, 5, 15, float('inf')],
            labels=['None', 'Low', 'Medium', 'High']
        )
        
        # Define activity segments
        user_stats['Activity_Level'] = pd.cut(
            user_stats['Session_Count'],
            bins=[-1, 1, 3, 7, float('inf')],
            labels=['Inactive', 'Low', 'Medium', 'High']
        )
        
        # Define CTR segments
        user_stats['CTR_Level'] = pd.cut(
            user_stats['CTR'],
            bins=[-0.001, 0.05, 0.1, 0.2, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Calculate segment combinations
        user_stats['Segment'] = user_stats.apply(
            lambda row: f"{row['Engagement_Level']} Engagement, {row['Activity_Level']} Activity, {row['CTR_Level']} CTR",
            axis=1
        )
        
        # Calculate segment counts
        segment_counts = user_stats['Segment'].value_counts()
        
        # Add segment percentages
        total_users = len(user_stats)
        user_stats['Segment_Percentage'] = user_stats['Segment'].map(
            segment_counts.apply(lambda x: (x / total_users * 100).round(2))
        )
        
        logger.info("User segmentation complete")
        return user_stats
    except Exception as e:
        logger.error(f"Error analyzing user segments: {e}")
        raise

def plot_user_segments(user_segments: pd.DataFrame) -> plt.Figure:
    """
    Plot user segment distribution.
    
    Args:
        user_segments: DataFrame with user segments
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting user segments")
    
    try:
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot engagement levels
        engagement_counts = user_segments['Engagement_Level'].value_counts().sort_index()
        sns.barplot(x=engagement_counts.index, y=engagement_counts.values, palette='viridis', ax=ax1)
        ax1.set_xlabel("Engagement Level")
        ax1.set_ylabel("Number of Users")
        ax1.set_title("Distribution of User Engagement Levels")
        
        # Add count and percentage text
        total = len(user_segments)
        for i, count in enumerate(engagement_counts):
            percentage = (count / total * 100).round(1)
            ax1.text(i, count + 5, f"{count} ({percentage}%)", ha='center')
            
        # Plot activity levels
        activity_counts = user_segments['Activity_Level'].value_counts().sort_index()
        sns.barplot(x=activity_counts.index, y=activity_counts.values, palette='viridis', ax=ax2)
        ax2.set_xlabel("Activity Level")
        ax2.set_ylabel("Number of Users")
        ax2.set_title("Distribution of User Activity Levels")
        
        # Add count and percentage text
        for i, count in enumerate(activity_counts):
            percentage = (count / total * 100).round(1)
            ax2.text(i, count + 5, f"{count} ({percentage}%)", ha='center')
            
        # Plot CTR levels
        ctr_counts = user_segments['CTR_Level'].value_counts().sort_index()
        sns.barplot(x=ctr_counts.index, y=ctr_counts.values, palette='viridis', ax=ax3)
        ax3.set_xlabel("CTR Level")
        ax3.set_ylabel("Number of Users")
        ax3.set_title("Distribution of User CTR Levels")
        
        # Add count and percentage text
        for i, count in enumerate(ctr_counts):
            percentage = (count / total * 100).round(1)
            ax3.text(i, count + 5, f"{count} ({percentage}%)", ha='center')
            
        plt.tight_layout()
        logger.info("User segments plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting user segments: {e}")
        raise

def plot_top_segments(user_segments: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Plot the top N user segments.
    
    Args:
        user_segments: DataFrame with user segments
        top_n: Number of top segments to display
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting top {top_n} user segments")
    
    try:
        # Count segments
        segment_counts = user_segments['Segment'].value_counts().head(top_n)
        
        # Calculate percentages
        total_users = len(user_segments)
        segment_pct = (segment_counts / total_users * 100).round(1)
        
        # Create DataFrame
        top_segments = pd.DataFrame({
            'Segment': segment_counts.index,
            'Count': segment_counts.values,
            'Percentage': segment_pct.values
        })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot horizontal bar chart
        sns.barplot(x='Count', y='Segment', data=top_segments, palette='viridis', ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Number of Users")
        ax.set_ylabel("Segment")
        ax.set_title(f"Top {top_n} User Segments")
        
        # Add count and percentage text
        for i, row in enumerate(top_segments.itertuples()):
            ax.text(row.Count + 5, i, f"{row.Count} ({row.Percentage}%)", va='center')
            
        plt.tight_layout()
        logger.info("Top segments plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting top segments: {e}")
        raise