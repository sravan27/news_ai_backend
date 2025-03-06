"""
News Analysis module for MIND dataset.

This module provides functions to analyze news articles in the MIND dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import Counter
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {e}")

def analyze_news_categories(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the distribution of news categories.
    
    Args:
        news_df: News DataFrame
        
    Returns:
        DataFrame with category counts
    """
    logger.info("Analyzing news categories")
    
    try:
        # Count categories
        category_counts = news_df['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        # Calculate percentage
        total = category_counts['Count'].sum()
        category_counts['Percentage'] = (category_counts['Count'] / total * 100).round(2)
        
        logger.info(f"Analysis complete: found {len(category_counts)} unique categories")
        return category_counts
    except Exception as e:
        logger.error(f"Error analyzing news categories: {e}")
        raise

def plot_news_categories(category_counts: pd.DataFrame, title: str = None) -> plt.Figure:
    """
    Plot news category distribution.
    
    Args:
        category_counts: DataFrame with category counts
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting news category distribution")
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by frequency
        sorted_counts = category_counts.sort_values('Count', ascending=False)
        
        # Create bar plot
        sns.barplot(x='Category', y='Count', data=sorted_counts, palette="viridis", ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.set_title(title or "Distribution of News Categories")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add count and percentage text
        for i, row in enumerate(sorted_counts.itertuples()):
            ax.text(i, row.Count + 0.5, f"{row.Count} ({row.Percentage}%)", ha='center')
            
        plt.tight_layout()
        logger.info("Category plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting news categories: {e}")
        raise

def analyze_news_subcategories(news_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Analyze the distribution of news subcategories.
    
    Args:
        news_df: News DataFrame
        top_n: Number of top subcategories to include
        
    Returns:
        DataFrame with subcategory counts
    """
    logger.info(f"Analyzing news subcategories (top {top_n})")
    
    try:
        # Count subcategories
        subcategory_counts = news_df['Subcategory'].value_counts().reset_index()
        subcategory_counts.columns = ['Subcategory', 'Count']
        
        # Calculate percentage
        total = subcategory_counts['Count'].sum()
        subcategory_counts['Percentage'] = (subcategory_counts['Count'] / total * 100).round(2)
        
        # Get top N subcategories
        top_subcategories = subcategory_counts.head(top_n)
        
        logger.info(f"Analysis complete: found {len(subcategory_counts)} unique subcategories")
        return top_subcategories
    except Exception as e:
        logger.error(f"Error analyzing news subcategories: {e}")
        raise

def plot_news_subcategories(subcategory_counts: pd.DataFrame, title: str = None) -> plt.Figure:
    """
    Plot news subcategory distribution.
    
    Args:
        subcategory_counts: DataFrame with subcategory counts
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting news subcategory distribution")
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create bar plot
        sns.barplot(x='Subcategory', y='Count', data=subcategory_counts, palette="viridis", ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Subcategory")
        ax.set_ylabel("Count")
        ax.set_title(title or f"Top {len(subcategory_counts)} News Subcategories")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add count and percentage text
        for i, row in enumerate(subcategory_counts.itertuples()):
            ax.text(i, row.Count + 0.5, f"{row.Count} ({row.Percentage}%)", ha='center')
            
        plt.tight_layout()
        logger.info("Subcategory plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting news subcategories: {e}")
        raise

def analyze_category_subcategory_relationship(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the relationship between categories and subcategories.
    
    Args:
        news_df: News DataFrame
        
    Returns:
        DataFrame with category-subcategory counts
    """
    logger.info("Analyzing category-subcategory relationship")
    
    try:
        # Group by category and subcategory
        grouped = news_df.groupby(['Category', 'Subcategory']).size().reset_index()
        grouped.columns = ['Category', 'Subcategory', 'Count']
        
        # Sort by count within each category
        grouped = grouped.sort_values(['Category', 'Count'], ascending=[True, False])
        
        logger.info("Category-subcategory analysis complete")
        return grouped
    except Exception as e:
        logger.error(f"Error analyzing category-subcategory relationship: {e}")
        raise

def plot_category_subcategory_heatmap(grouped_df: pd.DataFrame, 
                                     max_subcategories: int = 5,
                                     title: str = None) -> plt.Figure:
    """
    Plot a heatmap of the category-subcategory relationship.
    
    Args:
        grouped_df: DataFrame with category-subcategory counts
        max_subcategories: Maximum number of subcategories to show per category
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting category-subcategory heatmap (max_subcategories={max_subcategories})")
    
    try:
        # Get the top subcategories for each category
        top_subcategories = {}
        for category in grouped_df['Category'].unique():
            category_data = grouped_df[grouped_df['Category'] == category]
            top_subcategories[category] = category_data.nlargest(max_subcategories, 'Count')
            
        # Combine all top subcategories
        filtered_df = pd.concat(top_subcategories.values())
        
        # Pivot to create a matrix for the heatmap
        pivot_df = filtered_df.pivot(index='Category', columns='Subcategory', values='Count')
        pivot_df = pivot_df.fillna(0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='g', ax=ax)
        
        # Set title
        ax.set_title(title or f"Category-Subcategory Relationship (Top {max_subcategories} per Category)")
        
        plt.tight_layout()
        logger.info("Heatmap created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting category-subcategory heatmap: {e}")
        raise

def analyze_title_length(news_df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of title lengths.
    
    Args:
        news_df: News DataFrame
        
    Returns:
        Dictionary with title length statistics
    """
    logger.info("Analyzing news title lengths")
    
    try:
        # Calculate title length (character count)
        news_df_copy = news_df.copy()
        news_df_copy['Title_Length'] = news_df_copy['Title'].str.len()
        
        # Calculate title word count
        news_df_copy['Title_Word_Count'] = news_df_copy['Title'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        # Calculate statistics
        title_length_stats = {
            'char_mean': news_df_copy['Title_Length'].mean(),
            'char_median': news_df_copy['Title_Length'].median(),
            'char_min': news_df_copy['Title_Length'].min(),
            'char_max': news_df_copy['Title_Length'].max(),
            'char_std': news_df_copy['Title_Length'].std(),
            'word_mean': news_df_copy['Title_Word_Count'].mean(),
            'word_median': news_df_copy['Title_Word_Count'].median(),
            'word_min': news_df_copy['Title_Word_Count'].min(),
            'word_max': news_df_copy['Title_Word_Count'].max(),
            'word_std': news_df_copy['Title_Word_Count'].std()
        }
        
        logger.info("Title length analysis complete")
        return title_length_stats
    except Exception as e:
        logger.error(f"Error analyzing title length: {e}")
        raise

def plot_title_length_distribution(news_df: pd.DataFrame, 
                                  by_characters: bool = True,
                                  title: str = None) -> plt.Figure:
    """
    Plot the distribution of title lengths.
    
    Args:
        news_df: News DataFrame
        by_characters: If True, plot character count, else plot word count
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting title length distribution (by_characters={by_characters})")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        news_df_copy = news_df.copy()
        
        if by_characters:
            # Calculate title length (character count)
            news_df_copy['Title_Length'] = news_df_copy['Title'].str.len()
            x_label = "Title Length (Characters)"
            column = 'Title_Length'
        else:
            # Calculate title word count
            news_df_copy['Title_Word_Count'] = news_df_copy['Title'].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
            x_label = "Title Length (Words)"
            column = 'Title_Word_Count'
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        sns.histplot(news_df_copy[column].dropna(), bins=30, kde=True, ax=ax)
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel("Frequency")
        ax.set_title(title or f"Distribution of Title Lengths")
        
        # Add mean and median lines
        mean_val = news_df_copy[column].mean()
        median_val = news_df_copy[column].median()
        
        ax.axvline(mean_val, color='red', linestyle='--', 
                   label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle=':', 
                   label=f'Median: {median_val:.1f}')
        
        ax.legend()
        
        logger.info("Title length plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting title length distribution: {e}")
        raise

def analyze_abstract_length(news_df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of abstract lengths.
    
    Args:
        news_df: News DataFrame
        
    Returns:
        Dictionary with abstract length statistics
    """
    logger.info("Analyzing news abstract lengths")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        news_df_copy = news_df.copy()
        
        # Calculate abstract length (character count)
        news_df_copy['Abstract_Length'] = news_df_copy['Abstract'].str.len()
        
        # Calculate abstract word count
        news_df_copy['Abstract_Word_Count'] = news_df_copy['Abstract'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        # Calculate statistics
        abstract_length_stats = {
            'char_mean': news_df_copy['Abstract_Length'].mean(),
            'char_median': news_df_copy['Abstract_Length'].median(),
            'char_min': news_df_copy['Abstract_Length'].min(),
            'char_max': news_df_copy['Abstract_Length'].max(),
            'char_std': news_df_copy['Abstract_Length'].std(),
            'word_mean': news_df_copy['Abstract_Word_Count'].mean(),
            'word_median': news_df_copy['Abstract_Word_Count'].median(),
            'word_min': news_df_copy['Abstract_Word_Count'].min(),
            'word_max': news_df_copy['Abstract_Word_Count'].max(),
            'word_std': news_df_copy['Abstract_Word_Count'].std(),
            'missing_count': news_df_copy['Abstract'].isna().sum(),
            'missing_percentage': (news_df_copy['Abstract'].isna().sum() / len(news_df_copy) * 100).round(2)
        }
        
        logger.info("Abstract length analysis complete")
        return abstract_length_stats
    except Exception as e:
        logger.error(f"Error analyzing abstract length: {e}")
        raise

def plot_abstract_length_distribution(news_df: pd.DataFrame, 
                                     by_characters: bool = True,
                                     title: str = None) -> plt.Figure:
    """
    Plot the distribution of abstract lengths.
    
    Args:
        news_df: News DataFrame
        by_characters: If True, plot character count, else plot word count
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting abstract length distribution (by_characters={by_characters})")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        news_df_copy = news_df.copy()
        
        if by_characters:
            # Calculate abstract length (character count)
            news_df_copy['Abstract_Length'] = news_df_copy['Abstract'].str.len()
            x_label = "Abstract Length (Characters)"
            column = 'Abstract_Length'
        else:
            # Calculate abstract word count
            news_df_copy['Abstract_Word_Count'] = news_df_copy['Abstract'].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
            x_label = "Abstract Length (Words)"
            column = 'Abstract_Word_Count'
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        sns.histplot(news_df_copy[column].dropna(), bins=30, kde=True, ax=ax)
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel("Frequency")
        ax.set_title(title or f"Distribution of Abstract Lengths")
        
        # Add mean and median lines
        mean_val = news_df_copy[column].mean()
        median_val = news_df_copy[column].median()
        
        ax.axvline(mean_val, color='red', linestyle='--', 
                   label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle=':', 
                   label=f'Median: {median_val:.1f}')
        
        ax.legend()
        
        logger.info("Abstract length plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting abstract length distribution: {e}")
        raise

def compare_title_abstract_lengths(news_df: pd.DataFrame, 
                                  by_characters: bool = False) -> plt.Figure:
    """
    Compare title and abstract lengths.
    
    Args:
        news_df: News DataFrame
        by_characters: If True, use character count, else use word count
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Comparing title and abstract lengths (by_characters={by_characters})")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        news_df_copy = news_df.copy()
        
        if by_characters:
            # Calculate lengths
            news_df_copy['Title_Length'] = news_df_copy['Title'].str.len()
            news_df_copy['Abstract_Length'] = news_df_copy['Abstract'].str.len()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create box plot
            data = [
                news_df_copy['Title_Length'].dropna(),
                news_df_copy['Abstract_Length'].dropna()
            ]
            
            ax.boxplot(data, labels=['Title', 'Abstract'])
            
            # Set labels and title
            ax.set_ylabel("Length (Characters)")
            ax.set_title("Comparison of Title and Abstract Lengths (Characters)")
            
        else:
            # Calculate word counts
            news_df_copy['Title_Word_Count'] = news_df_copy['Title'].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
            news_df_copy['Abstract_Word_Count'] = news_df_copy['Abstract'].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create box plot
            data = [
                news_df_copy['Title_Word_Count'].dropna(),
                news_df_copy['Abstract_Word_Count'].dropna()
            ]
            
            ax.boxplot(data, labels=['Title', 'Abstract'])
            
            # Set labels and title
            ax.set_ylabel("Length (Words)")
            ax.set_title("Comparison of Title and Abstract Lengths (Words)")
            
        logger.info("Length comparison plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error comparing title and abstract lengths: {e}")
        raise

def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing special characters and lowercasing.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    if pd.isna(text):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters, keeping only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from text.
    
    Args:
        text: Input text
        
    Returns:
        Text without stopwords
    """
    if not text:
        return ""
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    return ' '.join(filtered_tokens)

def lemmatize_text(text: str) -> str:
    """
    Lemmatize text.
    
    Args:
        text: Input text
        
    Returns:
        Lemmatized text
    """
    if not text:
        return ""
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(lemmatized_tokens)

def analyze_top_words(news_df: pd.DataFrame, 
                     column: str = 'Title', 
                     top_n: int = 20,
                     remove_stop: bool = True,
                     lemmatize: bool = True) -> pd.DataFrame:
    """
    Analyze the most frequent words in news titles or abstracts.
    
    Args:
        news_df: News DataFrame
        column: Column to analyze ('Title' or 'Abstract')
        top_n: Number of top words to return
        remove_stop: Whether to remove stopwords
        lemmatize: Whether to lemmatize words
        
    Returns:
        DataFrame with word frequency counts
    """
    logger.info(f"Analyzing top words in {column} (remove_stop={remove_stop}, lemmatize={lemmatize})")
    
    try:
        # Preprocess text
        processed_texts = news_df[column].apply(preprocess_text)
        
        if remove_stop:
            processed_texts = processed_texts.apply(remove_stopwords)
            
        if lemmatize:
            processed_texts = processed_texts.apply(lemmatize_text)
            
        # Combine all text
        all_text = ' '.join(processed_texts.dropna())
        
        # Tokenize
        words = all_text.split()
        
        # Count frequencies
        word_counts = Counter(words)
        
        # Convert to DataFrame
        word_df = pd.DataFrame({
            'Word': list(word_counts.keys()),
            'Count': list(word_counts.values())
        }).sort_values('Count', ascending=False)
        
        # Calculate percentage
        total = word_df['Count'].sum()
        word_df['Percentage'] = (word_df['Count'] / total * 100).round(3)
        
        logger.info(f"Analysis complete: found {len(word_df)} unique words")
        return word_df.head(top_n)
    except Exception as e:
        logger.error(f"Error analyzing top words: {e}")
        raise

def plot_top_words(word_counts: pd.DataFrame, 
                  title: str = None) -> plt.Figure:
    """
    Plot the most frequent words.
    
    Args:
        word_counts: DataFrame with word counts
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting top words")
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        sns.barplot(x='Count', y='Word', data=word_counts, palette="viridis", ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Count")
        ax.set_ylabel("Word")
        ax.set_title(title or f"Top {len(word_counts)} Most Frequent Words")
        
        # Add count and percentage text
        for i, row in enumerate(word_counts.itertuples()):
            ax.text(row.Count + 0.5, i, f"{row.Count} ({row.Percentage}%)", va='center')
            
        logger.info("Word frequency plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting top words: {e}")
        raise

def create_wordcloud(news_df: pd.DataFrame, 
                    column: str = 'Title',
                    remove_stop: bool = True,
                    lemmatize: bool = True,
                    title: str = None) -> plt.Figure:
    """
    Create a word cloud visualization.
    
    Args:
        news_df: News DataFrame
        column: Column to analyze ('Title' or 'Abstract')
        remove_stop: Whether to remove stopwords
        lemmatize: Whether to lemmatize words
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Creating word cloud for {column}")
    
    try:
        # Preprocess text
        processed_texts = news_df[column].apply(preprocess_text)
        
        if remove_stop:
            processed_texts = processed_texts.apply(remove_stopwords)
            
        if lemmatize:
            processed_texts = processed_texts.apply(lemmatize_text)
            
        # Combine all text
        all_text = ' '.join(processed_texts.dropna())
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=200,
            contour_width=3,
            contour_color='steelblue'
        ).generate(all_text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Display word cloud
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        # Set title
        plt.title(title or f"Word Cloud of News {column}s")
        
        logger.info("Word cloud created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error creating word cloud: {e}")
        raise

def analyze_news_with_entities(news_df: pd.DataFrame) -> Dict:
    """
    Analyze the presence of entities in news articles.
    
    Args:
        news_df: News DataFrame
        
    Returns:
        Dictionary with entity presence statistics
    """
    logger.info("Analyzing presence of entities in news articles")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        news_df_copy = news_df.copy()
        
        # Check if Title_Entities and Abstract_Entities are already parsed
        if isinstance(news_df_copy['Title_Entities'].iloc[0], str):
            # Import the parse_entities function
            from .data_loader import parse_entities
            
            # Parse entities
            news_df_copy['Title_Entities'] = news_df_copy['Title_Entities'].apply(parse_entities)
            news_df_copy['Abstract_Entities'] = news_df_copy['Abstract_Entities'].apply(parse_entities)
            
        # Count entities per article
        news_df_copy['Title_Entity_Count'] = news_df_copy['Title_Entities'].apply(len)
        news_df_copy['Abstract_Entity_Count'] = news_df_copy['Abstract_Entities'].apply(len)
        news_df_copy['Total_Entity_Count'] = news_df_copy['Title_Entity_Count'] + news_df_copy['Abstract_Entity_Count']
        
        # Calculate presence flags
        news_df_copy['Has_Title_Entities'] = news_df_copy['Title_Entity_Count'] > 0
        news_df_copy['Has_Abstract_Entities'] = news_df_copy['Abstract_Entity_Count'] > 0
        news_df_copy['Has_Any_Entities'] = news_df_copy['Total_Entity_Count'] > 0
        
        # Calculate statistics
        stats = {
            'title_entity_mean': news_df_copy['Title_Entity_Count'].mean(),
            'title_entity_median': news_df_copy['Title_Entity_Count'].median(),
            'title_entity_max': news_df_copy['Title_Entity_Count'].max(),
            'abstract_entity_mean': news_df_copy['Abstract_Entity_Count'].mean(),
            'abstract_entity_median': news_df_copy['Abstract_Entity_Count'].median(),
            'abstract_entity_max': news_df_copy['Abstract_Entity_Count'].max(),
            'total_entity_mean': news_df_copy['Total_Entity_Count'].mean(),
            'total_entity_median': news_df_copy['Total_Entity_Count'].median(),
            'total_entity_max': news_df_copy['Total_Entity_Count'].max(),
            'articles_with_title_entities': news_df_copy['Has_Title_Entities'].sum(),
            'articles_with_title_entities_pct': (news_df_copy['Has_Title_Entities'].mean() * 100).round(2),
            'articles_with_abstract_entities': news_df_copy['Has_Abstract_Entities'].sum(),
            'articles_with_abstract_entities_pct': (news_df_copy['Has_Abstract_Entities'].mean() * 100).round(2),
            'articles_with_any_entities': news_df_copy['Has_Any_Entities'].sum(),
            'articles_with_any_entities_pct': (news_df_copy['Has_Any_Entities'].mean() * 100).round(2)
        }
        
        logger.info("Entity presence analysis complete")
        return stats
    except Exception as e:
        logger.error(f"Error analyzing entity presence: {e}")
        raise

def plot_entity_count_distribution(news_df: pd.DataFrame) -> plt.Figure:
    """
    Plot the distribution of entity counts per article.
    
    Args:
        news_df: News DataFrame
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting entity count distribution")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        news_df_copy = news_df.copy()
        
        # Check if Title_Entities and Abstract_Entities are already parsed
        if isinstance(news_df_copy['Title_Entities'].iloc[0], str):
            # Import the parse_entities function
            from .data_loader import parse_entities
            
            # Parse entities
            news_df_copy['Title_Entities'] = news_df_copy['Title_Entities'].apply(parse_entities)
            news_df_copy['Abstract_Entities'] = news_df_copy['Abstract_Entities'].apply(parse_entities)
            
        # Count entities per article
        news_df_copy['Title_Entity_Count'] = news_df_copy['Title_Entities'].apply(len)
        news_df_copy['Abstract_Entity_Count'] = news_df_copy['Abstract_Entities'].apply(len)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot histogram for title entities
        sns.histplot(news_df_copy['Title_Entity_Count'], bins=10, kde=True, ax=ax1)
        ax1.set_xlabel("Number of Entities")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Entities in Titles")
        
        # Add mean and median lines
        title_mean = news_df_copy['Title_Entity_Count'].mean()
        title_median = news_df_copy['Title_Entity_Count'].median()
        
        ax1.axvline(title_mean, color='red', linestyle='--', 
                   label=f'Mean: {title_mean:.2f}')
        ax1.axvline(title_median, color='green', linestyle=':', 
                   label=f'Median: {title_median:.2f}')
        
        ax1.legend()
        
        # Plot histogram for abstract entities
        sns.histplot(news_df_copy['Abstract_Entity_Count'], bins=10, kde=True, ax=ax2)
        ax2.set_xlabel("Number of Entities")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Entities in Abstracts")
        
        # Add mean and median lines
        abstract_mean = news_df_copy['Abstract_Entity_Count'].mean()
        abstract_median = news_df_copy['Abstract_Entity_Count'].median()
        
        ax2.axvline(abstract_mean, color='red', linestyle='--', 
                   label=f'Mean: {abstract_mean:.2f}')
        ax2.axvline(abstract_median, color='green', linestyle=':', 
                   label=f'Median: {abstract_median:.2f}')
        
        ax2.legend()
        
        plt.tight_layout()
        logger.info("Entity count plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting entity count distribution: {e}")
        raise

def plot_entity_presence_pie(news_df: pd.DataFrame) -> plt.Figure:
    """
    Plot a pie chart showing the proportion of articles with and without entities.
    
    Args:
        news_df: News DataFrame
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting entity presence pie chart")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        news_df_copy = news_df.copy()
        
        # Check if Title_Entities and Abstract_Entities are already parsed
        if isinstance(news_df_copy['Title_Entities'].iloc[0], str):
            # Import the parse_entities function
            from .data_loader import parse_entities
            
            # Parse entities
            news_df_copy['Title_Entities'] = news_df_copy['Title_Entities'].apply(parse_entities)
            news_df_copy['Abstract_Entities'] = news_df_copy['Abstract_Entities'].apply(parse_entities)
            
        # Count entities per article
        news_df_copy['Title_Entity_Count'] = news_df_copy['Title_Entities'].apply(len)
        news_df_copy['Abstract_Entity_Count'] = news_df_copy['Abstract_Entities'].apply(len)
        news_df_copy['Total_Entity_Count'] = news_df_copy['Title_Entity_Count'] + news_df_copy['Abstract_Entity_Count']
        
        # Calculate presence flags
        news_df_copy['Has_Title_Entities'] = news_df_copy['Title_Entity_Count'] > 0
        news_df_copy['Has_Abstract_Entities'] = news_df_copy['Abstract_Entity_Count'] > 0
        news_df_copy['Has_Any_Entities'] = news_df_copy['Total_Entity_Count'] > 0
        
        # Create figure with two subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot pie chart for title entities
        title_counts = news_df_copy['Has_Title_Entities'].value_counts()
        ax1.pie(
            title_counts, 
            labels=['No Entities', 'Has Entities'] if title_counts.index[0] == False else ['Has Entities', 'No Entities'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['lightcoral', 'lightgreen'] if title_counts.index[0] == False else ['lightgreen', 'lightcoral']
        )
        ax1.set_title('Articles with Entities in Title')
        
        # Plot pie chart for abstract entities
        abstract_counts = news_df_copy['Has_Abstract_Entities'].value_counts()
        ax2.pie(
            abstract_counts, 
            labels=['No Entities', 'Has Entities'] if abstract_counts.index[0] == False else ['Has Entities', 'No Entities'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['lightcoral', 'lightgreen'] if abstract_counts.index[0] == False else ['lightgreen', 'lightcoral']
        )
        ax2.set_title('Articles with Entities in Abstract')
        
        # Plot pie chart for any entities
        any_counts = news_df_copy['Has_Any_Entities'].value_counts()
        ax3.pie(
            any_counts, 
            labels=['No Entities', 'Has Entities'] if any_counts.index[0] == False else ['Has Entities', 'No Entities'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['lightcoral', 'lightgreen'] if any_counts.index[0] == False else ['lightgreen', 'lightcoral']
        )
        ax3.set_title('Articles with Any Entities')
        
        plt.tight_layout()
        logger.info("Entity presence pie chart created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting entity presence pie chart: {e}")
        raise

def perform_advanced_sentiment_analysis(news_df: pd.DataFrame, 
                                      column: str = 'Title',
                                      method: str = 'vader') -> pd.DataFrame:
    """
    Perform sentiment analysis on news titles or abstracts.
    
    Args:
        news_df: News DataFrame
        column: Column to analyze ('Title' or 'Abstract')
        method: Sentiment analysis method ('vader', 'textblob', or 'transformers')
        
    Returns:
        DataFrame with sentiment scores
    """
    logger.info(f"Performing sentiment analysis on {column} using {method}")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        news_df_copy = news_df.copy()
        
        if method == 'vader':
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                nltk.download('vader_lexicon', quiet=True)
                
                # Initialize VADER
                sia = SentimentIntensityAnalyzer()
                
                # Calculate sentiment scores
                news_df_copy[f'{column}_Sentiment_Scores'] = news_df_copy[column].apply(
                    lambda x: sia.polarity_scores(str(x)) if pd.notna(x) else None
                )
                
                # Extract individual scores
                news_df_copy[f'{column}_Negative'] = news_df_copy[f'{column}_Sentiment_Scores'].apply(
                    lambda x: x['neg'] if x is not None else None
                )
                news_df_copy[f'{column}_Neutral'] = news_df_copy[f'{column}_Sentiment_Scores'].apply(
                    lambda x: x['neu'] if x is not None else None
                )
                news_df_copy[f'{column}_Positive'] = news_df_copy[f'{column}_Sentiment_Scores'].apply(
                    lambda x: x['pos'] if x is not None else None
                )
                news_df_copy[f'{column}_Compound'] = news_df_copy[f'{column}_Sentiment_Scores'].apply(
                    lambda x: x['compound'] if x is not None else None
                )
                
                # Determine sentiment category
                news_df_copy[f'{column}_Sentiment'] = news_df_copy[f'{column}_Compound'].apply(
                    lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral') if pd.notna(x) else None
                )
                
                # Drop the scores dictionary
                news_df_copy = news_df_copy.drop(columns=[f'{column}_Sentiment_Scores'])
                
            except Exception as e:
                logger.error(f"Error using VADER: {e}")
                # Fallback to TextBlob
                logger.info("Falling back to TextBlob for sentiment analysis")
                method = 'textblob'
        
        if method == 'textblob':
            from textblob import TextBlob
            
            # Calculate sentiment polarity
            news_df_copy[f'{column}_Polarity'] = news_df_copy[column].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else None
            )
            
            # Calculate sentiment subjectivity
            news_df_copy[f'{column}_Subjectivity'] = news_df_copy[column].apply(
                lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else None
            )
            
            # Determine sentiment category
            news_df_copy[f'{column}_Sentiment'] = news_df_copy[f'{column}_Polarity'].apply(
                lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral') if pd.notna(x) else None
            )
            
        elif method == 'transformers':
            try:
                from transformers import pipeline
                
                # Initialize sentiment pipeline
                sentiment_pipeline = pipeline("sentiment-analysis")
                
                # Apply sentiment analysis
                results = []
                for text in news_df_copy[column].fillna("").tolist():
                    if text:
                        try:
                            result = sentiment_pipeline(text)[0]
                            results.append({
                                'label': result['label'],
                                'score': result['score']
                            })
                        except Exception as e:
                            logger.warning(f"Error analyzing text: {e}")
                            results.append(None)
                    else:
                        results.append(None)
                        
                # Add results to DataFrame
                news_df_copy[f'{column}_Transformer_Label'] = [r['label'] if r else None for r in results]
                news_df_copy[f'{column}_Transformer_Score'] = [r['score'] if r else None for r in results]
                
                # Map labels to consistent format
                label_map = {
                    'LABEL_0': 'Negative',
                    'LABEL_1': 'Neutral',
                    'LABEL_2': 'Positive',
                    'NEGATIVE': 'Negative',
                    'NEUTRAL': 'Neutral',
                    'POSITIVE': 'Positive'
                }
                
                news_df_copy[f'{column}_Sentiment'] = news_df_copy[f'{column}_Transformer_Label'].map(
                    lambda x: label_map.get(x, x) if pd.notna(x) else None
                )
                
            except Exception as e:
                logger.error(f"Error using transformers: {e}")
                # Fallback to TextBlob
                logger.info("Falling back to TextBlob for sentiment analysis")
                return perform_advanced_sentiment_analysis(news_df, column, 'textblob')
                
        logger.info("Sentiment analysis complete")
        return news_df_copy
    except Exception as e:
        logger.error(f"Error performing sentiment analysis: {e}")
        raise

def plot_sentiment_distribution(sentiment_df: pd.DataFrame,
                               column: str = 'Title',
                               title: str = None) -> plt.Figure:
    """
    Plot the distribution of sentiment categories.
    
    Args:
        sentiment_df: DataFrame with sentiment scores
        column: Column that was analyzed ('Title' or 'Abstract')
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting sentiment distribution for {column}")
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Count sentiment categories
        sentiment_counts = sentiment_df[f'{column}_Sentiment'].value_counts()
        
        # Create bar chart
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='RdYlGn', ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title(title or f"Distribution of {column} Sentiment")
        
        # Add count and percentage text
        total = sentiment_counts.sum()
        for i, count in enumerate(sentiment_counts):
            percentage = (count / total * 100).round(1)
            ax.text(i, count + 5, f"{count} ({percentage}%)", ha='center')
            
        # Create pie chart as an inset
        ax_inset = fig.add_axes([0.65, 0.6, 0.25, 0.25])
        ax_inset.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=['red', 'gold', 'green'] if 'Negative' in sentiment_counts.index else ['gold', 'green', 'red']
        )
        ax_inset.set_title('Sentiment Proportions')
        
        logger.info("Sentiment distribution plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting sentiment distribution: {e}")
        raise

def plot_sentiment_by_category(sentiment_df: pd.DataFrame,
                              column: str = 'Title',
                              title: str = None) -> plt.Figure:
    """
    Plot sentiment distribution by news category.
    
    Args:
        sentiment_df: DataFrame with sentiment scores
        column: Column that was analyzed ('Title' or 'Abstract')
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting sentiment by category for {column}")
    
    try:
        # Create a crosstab of Category vs Sentiment
        sentiment_by_category = pd.crosstab(
            sentiment_df['Category'], 
            sentiment_df[f'{column}_Sentiment'],
            normalize='index'
        ) * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create stacked bar chart
        sentiment_by_category.plot(
            kind='bar',
            stacked=True,
            colormap='RdYlGn',
            ax=ax
        )
        
        # Set labels and title
        ax.set_xlabel("Category")
        ax.set_ylabel("Percentage")
        ax.set_title(title or f"Sentiment Distribution by Category ({column})")
        
        # Add percentage annotations
        for i, row in enumerate(sentiment_by_category.itertuples()):
            cumulative = 0
            for j, percentage in enumerate(row[1:]):
                # Skip small slices
                if percentage < 5:
                    continue
                    
                # Calculate position
                position = cumulative + percentage / 2
                ax.text(i, position, f"{percentage:.1f}%", ha='center')
                cumulative += percentage
                
        # Add legend
        ax.legend(title="Sentiment")
        
        plt.tight_layout()
        logger.info("Sentiment by category plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting sentiment by category: {e}")
        raise

def analyze_reading_level(news_df: pd.DataFrame,
                         column: str = 'Title') -> pd.DataFrame:
    """
    Analyze the reading level of news titles or abstracts.
    
    Args:
        news_df: News DataFrame
        column: Column to analyze ('Title' or 'Abstract')
        
    Returns:
        DataFrame with reading level scores
    """
    logger.info(f"Analyzing reading level of {column}")
    
    try:
        # Import textstat
        import textstat
        
        # Create a copy to avoid modifying the original DataFrame
        news_df_copy = news_df.copy()
        
        # Calculate Flesch-Kincaid Grade Level
        news_df_copy[f'{column}_FK_Grade'] = news_df_copy[column].apply(
            lambda x: textstat.flesch_kincaid_grade(str(x)) if pd.notna(x) else None
        )
        
        # Calculate Dale-Chall Readability Score
        news_df_copy[f'{column}_Dale_Chall'] = news_df_copy[column].apply(
            lambda x: textstat.dale_chall_readability_score(str(x)) if pd.notna(x) else None
        )
        
        # Calculate SMOG Index
        news_df_copy[f'{column}_SMOG'] = news_df_copy[column].apply(
            lambda x: textstat.smog_index(str(x)) if pd.notna(x) else None
        )
        
        # Calculate Gunning Fog Index
        news_df_copy[f'{column}_Gunning_Fog'] = news_df_copy[column].apply(
            lambda x: textstat.gunning_fog(str(x)) if pd.notna(x) else None
        )
        
        # Categorize by reading level
        news_df_copy[f'{column}_Reading_Level'] = news_df_copy[f'{column}_FK_Grade'].apply(
            lambda grade: (
                "Elementary" if grade <= 5 else
                "Middle School" if grade <= 8 else
                "High School" if grade <= 12 else
                "College Level"
            ) if pd.notna(grade) else None
        )
        
        logger.info("Reading level analysis complete")
        return news_df_copy
    except Exception as e:
        logger.error(f"Error analyzing reading level: {e}")
        raise

def plot_reading_level_distribution(reading_df: pd.DataFrame,
                                   column: str = 'Title',
                                   metric: str = 'FK_Grade',
                                   title: str = None) -> plt.Figure:
    """
    Plot the distribution of reading level scores.
    
    Args:
        reading_df: DataFrame with reading level scores
        column: Column that was analyzed ('Title' or 'Abstract')
        metric: Reading level metric to plot ('FK_Grade', 'Dale_Chall', 'SMOG', or 'Gunning_Fog')
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting reading level distribution for {column} ({metric})")
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        sns.histplot(
            reading_df[f'{column}_{metric}'].dropna(),
            bins=20,
            kde=True,
            ax=ax
        )
        
        # Set labels and title
        ax.set_xlabel(f"{metric.replace('_', ' ')} Score")
        ax.set_ylabel("Frequency")
        ax.set_title(title or f"Distribution of {column} Reading Level ({metric.replace('_', ' ')})")
        
        # Add mean and median lines
        mean_val = reading_df[f'{column}_{metric}'].mean()
        median_val = reading_df[f'{column}_{metric}'].median()
        
        ax.axvline(mean_val, color='red', linestyle='--', 
                   label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle=':', 
                   label=f'Median: {median_val:.2f}')
        
        ax.legend()
        
        logger.info("Reading level distribution plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting reading level distribution: {e}")
        raise

def plot_reading_level_categories(reading_df: pd.DataFrame,
                                 column: str = 'Title',
                                 title: str = None) -> plt.Figure:
    """
    Plot the distribution of reading level categories.
    
    Args:
        reading_df: DataFrame with reading level scores
        column: Column that was analyzed ('Title' or 'Abstract')
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting reading level categories for {column}")
    
    try:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count reading level categories
        level_counts = reading_df[f'{column}_Reading_Level'].value_counts()
        
        # Create bar chart
        sns.barplot(x=level_counts.index, y=level_counts.values, palette='viridis', ax=ax1)
        
        # Set labels and title
        ax1.set_xlabel("Reading Level")
        ax1.set_ylabel("Count")
        ax1.set_title(f"Distribution of {column} Reading Level Categories")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count and percentage text
        total = level_counts.sum()
        for i, count in enumerate(level_counts):
            percentage = (count / total * 100).round(1)
            ax1.text(i, count + 5, f"{count} ({percentage}%)", ha='center')
            
        # Create pie chart
        ax2.pie(
            level_counts,
            labels=level_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette('viridis', n_colors=len(level_counts))
        )
        ax2.set_title('Reading Level Proportions')
        
        plt.tight_layout()
        logger.info("Reading level categories plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting reading level categories: {e}")
        raise

def analyze_political_content(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze political content in news articles.
    
    Args:
        news_df: News DataFrame
        
    Returns:
        DataFrame with political content flags
    """
    logger.info("Analyzing political content in news articles")
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        news_df_copy = news_df.copy()
        
        # List of political keywords
        political_keywords = [
            'president', 'congress', 'senate', 'election', 'vote', 'democrat', 'republican',
            'politics', 'policy', 'government', 'biden', 'trump', 'obama', 'white house',
            'parliament', 'prime minister', 'supreme court', 'brexit', 'eu', 'nato',
            'party', 'campaign', 'candidate', 'ballot', 'poll', 'debate', 'legislation',
            'law', 'court', 'judge', 'justice', 'ruling', 'constitution', 'amendment'
        ]
        
        # Function to check if text contains political keywords
        def contains_political_keywords(text):
            if pd.isna(text):
                return False
                
            text = str(text).lower()
            return any(keyword in text for keyword in political_keywords)
            
        # Check for political content in title and abstract
        news_df_copy['Title_Political'] = news_df_copy['Title'].apply(contains_political_keywords)
        news_df_copy['Abstract_Political'] = news_df_copy['Abstract'].apply(contains_political_keywords)
        news_df_copy['Any_Political'] = news_df_copy['Title_Political'] | news_df_copy['Abstract_Political']
        
        # Check if the article is in a political category or subcategory
        political_categories = ['news', 'politics']
        political_subcategories = ['politics', 'worldpolitics', 'uspolitics', 'elections', 'government']
        
        news_df_copy['Category_Political'] = news_df_copy['Category'].str.lower().isin(political_categories)
        news_df_copy['Subcategory_Political'] = news_df_copy['Subcategory'].str.lower().isin(political_subcategories)
        
        # Combine all flags
        news_df_copy['Is_Political'] = (
            news_df_copy['Title_Political'] | 
            news_df_copy['Abstract_Political'] | 
            news_df_copy['Category_Political'] | 
            news_df_copy['Subcategory_Political']
        )
        
        logger.info("Political content analysis complete")
        return news_df_copy
    except Exception as e:
        logger.error(f"Error analyzing political content: {e}")
        raise

def plot_political_content_distribution(political_df: pd.DataFrame,
                                       title: str = None) -> plt.Figure:
    """
    Plot the distribution of political content.
    
    Args:
        political_df: DataFrame with political content flags
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting political content distribution")
    
    try:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count political and non-political articles
        political_counts = political_df['Is_Political'].value_counts().sort_index()
        
        # Create bar chart
        sns.barplot(
            x=['Non-Political', 'Political'],
            y=political_counts.values,
            palette=['skyblue', 'salmon'],
            ax=ax1
        )
        
        # Set labels and title
        ax1.set_xlabel("Content Type")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution of Political vs. Non-Political Content")
        
        # Add count and percentage text
        total = political_counts.sum()
        for i, count in enumerate(political_counts):
            percentage = (count / total * 100).round(1)
            ax1.text(i, count + 5, f"{count} ({percentage}%)", ha='center')
            
        # Create pie chart
        ax2.pie(
            political_counts,
            labels=['Non-Political', 'Political'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['skyblue', 'salmon']
        )
        ax2.set_title('Political Content Proportions')
        
        plt.tight_layout()
        logger.info("Political content distribution plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting political content distribution: {e}")
        raise

def plot_political_content_by_category(political_df: pd.DataFrame,
                                      title: str = None) -> plt.Figure:
    """
    Plot political content distribution by news category.
    
    Args:
        political_df: DataFrame with political content flags
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting political content by category")
    
    try:
        # Create a crosstab of Category vs Political Content
        political_by_category = pd.crosstab(
            political_df['Category'], 
            political_df['Is_Political'],
            normalize='index'
        ) * 100
        
        # Rename columns
        political_by_category.columns = ['Non-Political', 'Political']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create stacked bar chart
        political_by_category.plot(
            kind='bar',
            stacked=True,
            color=['skyblue', 'salmon'],
            ax=ax
        )
        
        # Set labels and title
        ax.set_xlabel("Category")
        ax.set_ylabel("Percentage")
        ax.set_title(title or "Political Content Distribution by Category")
        
        # Add percentage annotations
        for i, row in enumerate(political_by_category.itertuples()):
            # Non-political percentage (at the bottom)
            non_political_pct = row[1]
            if non_political_pct >= 5:
                ax.text(i, non_political_pct / 2, f"{non_political_pct:.1f}%", ha='center')
                
            # Political percentage (at the top)
            political_pct = row[2]
            if political_pct >= 5:
                ax.text(i, non_political_pct + political_pct / 2, f"{political_pct:.1f}%", ha='center')
                
        # Add legend
        ax.legend(title="Content Type")
        
        plt.tight_layout()
        logger.info("Political content by category plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting political content by category: {e}")
        raise