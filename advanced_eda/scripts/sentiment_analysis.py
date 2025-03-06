#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Analysis for MIND Dataset
---------------------------------
Functions to analyze and visualize sentiment in news articles.
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


def analyze_sentiment_textblob(news_df: pd.DataFrame, 
                             text_column: str = "title",
                             result_prefix: str = "title") -> pd.DataFrame:
    """
    Analyze sentiment using TextBlob.
    
    Args:
        news_df: DataFrame containing news articles
        text_column: Column containing text to analyze
        result_prefix: Prefix for result columns
        
    Returns:
        DataFrame with sentiment analysis results added
    """
    try:
        from textblob import TextBlob
    except ImportError:
        logger.error("TextBlob not installed. Install with: pip install textblob")
        return news_df
    
    if text_column not in news_df.columns:
        logger.warning(f"Text column '{text_column}' not found in DataFrame")
        return news_df
    
    # Create a copy of the dataframe
    df = news_df.copy()
    
    # Calculate polarity and subjectivity
    logger.info(f"Analyzing sentiment for {text_column} using TextBlob...")
    
    # Helper function to analyze a single text
    def analyze_text(text):
        if pd.isna(text) or text == "":
            return pd.Series([None, None])
        
        blob = TextBlob(text)
        return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity])
    
    # Apply to all texts
    tqdm.pandas(desc=f"Analyzing {text_column}")
    sentiment_df = df[text_column].progress_apply(analyze_text)
    df[f"{result_prefix}_polarity"] = sentiment_df[0]
    df[f"{result_prefix}_subjectivity"] = sentiment_df[1]
    
    # Calculate objectivity (inverse of subjectivity)
    df[f"{result_prefix}_objectivity"] = 1 - df[f"{result_prefix}_subjectivity"]
    
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
    
    df[f"{result_prefix}_sentiment"] = df[f"{result_prefix}_polarity"].apply(categorize_sentiment)
    
    logger.info(f"TextBlob sentiment analysis complete for {text_column}")
    return df


def analyze_sentiment_vader(news_df: pd.DataFrame, 
                          text_column: str = "title",
                          result_prefix: str = "title") -> pd.DataFrame:
    """
    Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    
    Args:
        news_df: DataFrame containing news articles
        text_column: Column containing text to analyze
        result_prefix: Prefix for result columns
        
    Returns:
        DataFrame with sentiment analysis results added
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            logger.info("Downloading VADER lexicon...")
            nltk.download('vader_lexicon')
    except ImportError:
        logger.error("NLTK not installed. Install with: pip install nltk")
        return news_df
    
    if text_column not in news_df.columns:
        logger.warning(f"Text column '{text_column}' not found in DataFrame")
        return news_df
    
    # Create a copy of the dataframe
    df = news_df.copy()
    
    # Initialize VADER
    sid = SentimentIntensityAnalyzer()
    
    # Helper function to analyze a single text
    def analyze_text(text):
        if pd.isna(text) or text == "":
            return pd.Series([None, None, None, None])
        
        scores = sid.polarity_scores(text)
        return pd.Series([scores['neg'], scores['neu'], scores['pos'], scores['compound']])
    
    # Apply to all texts
    logger.info(f"Analyzing sentiment for {text_column} using VADER...")
    tqdm.pandas(desc=f"Analyzing {text_column}")
    sentiment_df = df[text_column].progress_apply(analyze_text)
    
    df[f"{result_prefix}_vader_negative"] = sentiment_df[0]
    df[f"{result_prefix}_vader_neutral"] = sentiment_df[1]
    df[f"{result_prefix}_vader_positive"] = sentiment_df[2]
    df[f"{result_prefix}_vader_compound"] = sentiment_df[3]
    
    # Categorize sentiment
    def categorize_sentiment(compound):
        if pd.isna(compound):
            return "Unknown"
        elif compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    df[f"{result_prefix}_vader_sentiment"] = df[f"{result_prefix}_vader_compound"].apply(categorize_sentiment)
    
    logger.info(f"VADER sentiment analysis complete for {text_column}")
    return df


def analyze_sentiment_transformer(news_df: pd.DataFrame, 
                                text_column: str = "title",
                                result_prefix: str = "title",
                                model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
                                batch_size: int = 32) -> pd.DataFrame:
    """
    Analyze sentiment using transformer models from Hugging Face.
    
    Args:
        news_df: DataFrame containing news articles
        text_column: Column containing text to analyze
        result_prefix: Prefix for result columns
        model_name: Hugging Face model to use
        batch_size: Batch size for inference
        
    Returns:
        DataFrame with sentiment analysis results added
    """
    try:
        from transformers import pipeline
        import torch
    except ImportError:
        logger.error("Transformers not installed. Install with: pip install transformers torch")
        return news_df
    
    if text_column not in news_df.columns:
        logger.warning(f"Text column '{text_column}' not found in DataFrame")
        return news_df
    
    # Create a copy of the dataframe
    df = news_df.copy()
    
    # Check for GPU
    device = 0 if torch.cuda.is_available() else -1
    
    # Initialize the pipeline
    logger.info(f"Loading transformer model: {model_name}")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=device)
    
    # Process in batches
    logger.info(f"Analyzing sentiment for {text_column} using transformer model...")
    all_texts = df[text_column].fillna("").tolist()
    all_results = []
    
    for i in tqdm(range(0, len(all_texts), batch_size), desc=f"Processing batches"):
        batch_texts = all_texts[i:i + batch_size]
        
        # Skip empty texts
        batch_results = []
        for text in batch_texts:
            if text.strip():
                try:
                    result = sentiment_pipeline(text)[0]
                    batch_results.append(result)
                except Exception as e:
                    logger.warning(f"Error analyzing text: {e}")
                    batch_results.append({"label": "LABEL_1", "score": 0.5})  # Default to neutral
            else:
                batch_results.append({"label": "LABEL_1", "score": 0.5})  # Default to neutral for empty text
        
        all_results.extend(batch_results)
    
    # Extract results
    if model_name == "cardiffnlp/twitter-roberta-base-sentiment":
        # Map labels to sentiment categories
        label_map = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        }
        
        df[f"{result_prefix}_transformer_sentiment"] = [label_map.get(r["label"], "Neutral") for r in all_results]
    else:
        # Generic mapping for other models
        df[f"{result_prefix}_transformer_sentiment"] = [r["label"] for r in all_results]
    
    df[f"{result_prefix}_transformer_score"] = [r["score"] for r in all_results]
    
    logger.info(f"Transformer sentiment analysis complete for {text_column}")
    return df


def calculate_reading_level(news_df: pd.DataFrame, 
                          text_column: str = "title",
                          result_prefix: str = "title") -> pd.DataFrame:
    """
    Calculate reading level metrics for text.
    
    Args:
        news_df: DataFrame containing news articles
        text_column: Column containing text to analyze
        result_prefix: Prefix for result columns
        
    Returns:
        DataFrame with reading level metrics added
    """
    try:
        import textstat
    except ImportError:
        logger.error("Textstat not installed. Install with: pip install textstat")
        return news_df
    
    if text_column not in news_df.columns:
        logger.warning(f"Text column '{text_column}' not found in DataFrame")
        return news_df
    
    # Create a copy of the dataframe
    df = news_df.copy()
    
    # Helper function to calculate reading levels
    def calculate_levels(text):
        if pd.isna(text) or text == "":
            return pd.Series([None, None, None, None])
        
        fk_grade = textstat.flesch_kincaid_grade(text)
        flesch_ease = textstat.flesch_reading_ease(text)
        smog = textstat.smog_index(text)
        dale_chall = textstat.dale_chall_readability_score(text)
        
        return pd.Series([fk_grade, flesch_ease, smog, dale_chall])
    
    # Apply to all texts
    logger.info(f"Calculating reading levels for {text_column}...")
    tqdm.pandas(desc=f"Processing {text_column}")
    reading_df = df[text_column].progress_apply(calculate_levels)
    
    df[f"{result_prefix}_fk_grade"] = reading_df[0]
    df[f"{result_prefix}_flesch_ease"] = reading_df[1]
    df[f"{result_prefix}_smog_index"] = reading_df[2]
    df[f"{result_prefix}_dale_chall"] = reading_df[3]
    
    # Categorize reading level
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
    
    df[f"{result_prefix}_reading_level"] = df[f"{result_prefix}_fk_grade"].apply(categorize_reading_level)
    
    logger.info(f"Reading level calculation complete for {text_column}")
    return df


def identify_political_content(news_df: pd.DataFrame,
                             text_column: str = "title",
                             result_prefix: str = "title") -> pd.DataFrame:
    """
    Identify politically-charged content in text.
    
    Args:
        news_df: DataFrame containing news articles
        text_column: Column containing text to analyze
        result_prefix: Prefix for result columns
        
    Returns:
        DataFrame with political content flags added
    """
    if text_column not in news_df.columns:
        logger.warning(f"Text column '{text_column}' not found in DataFrame")
        return news_df
    
    # Create a copy of the dataframe
    df = news_df.copy()
    
    # Political keywords
    political_keywords = [
        "president", "minister", "congress", "senate", "parliament", "government", 
        "democrat", "republican", "party", "election", "campaign", "vote", "ballot",
        "politics", "political", "policy", "politician", "governor", "mayor", 
        "white house", "capitol", "federal", "administration", "trump", "biden", "obama"
    ]
    
    # Check for political content
    def is_political(text):
        if pd.isna(text) or text == "":
            return False
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in political_keywords)
    
    logger.info(f"Identifying political content in {text_column}...")
    df[f"{result_prefix}_is_political"] = df[text_column].apply(is_political)
    
    return df


def calculate_rhetoric_intensity(news_df: pd.DataFrame,
                               text_column: str = "title",
                               result_prefix: str = "title") -> pd.DataFrame:
    """
    Calculate rhetoric intensity in text (based on word usage).
    
    Args:
        news_df: DataFrame containing news articles
        text_column: Column containing text to analyze
        result_prefix: Prefix for result columns
        
    Returns:
        DataFrame with rhetoric intensity scores added
    """
    if text_column not in news_df.columns:
        logger.warning(f"Text column '{text_column}' not found in DataFrame")
        return news_df
    
    # Create a copy of the dataframe
    df = news_df.copy()
    
    # Rhetoric intensifier words
    intensifiers = [
        "very", "extremely", "incredibly", "absolutely", "completely", "totally",
        "utterly", "entirely", "definitely", "certainly", "undoubtedly", "unquestionably",
        "enormously", "tremendously", "exceedingly", "immensely", "thoroughly", "decidedly",
        "positively", "strongly", "highly", "amazingly", "astonishingly", "extraordinarily"
    ]
    
    # Emotionally charged words
    emotional_words = [
        "outrage", "scandal", "shocking", "devastating", "horrible", "terrible",
        "disastrous", "catastrophic", "crisis", "emergency", "alarming", "tragedy",
        "nightmare", "horrific", "frightening", "terrifying", "threatening", "dangerous",
        "violence", "attack", "destruction", "controversy", "conflict", "fight", "battle"
    ]
    
    # Extreme claims
    extreme_claims = [
        "never", "always", "every", "all", "none", "best", "worst", "only", "greatest",
        "perfect", "impossible", "must", "essential", "vital", "critical", "crucial", 
        "unprecedented", "historic", "revolutionary", "groundbreaking", "radical", "dramatic"
    ]
    
    # Calculate rhetoric intensity
    def calculate_intensity(text):
        if pd.isna(text) or text == "":
            return 0.0
        
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        # Count occurrences of rhetoric words
        intensifier_count = sum(1 for word in words if word in intensifiers)
        emotional_count = sum(1 for word in words if word in emotional_words)
        extreme_count = sum(1 for word in words if word in extreme_claims)
        
        # Total rhetoric words
        rhetoric_count = intensifier_count + emotional_count + extreme_count
        
        # Calculate intensity score (0-1 range)
        if word_count == 0:
            return 0.0
        
        # Weight different types of rhetoric words
        weighted_count = (intensifier_count * 0.8) + (emotional_count * 1.2) + (extreme_count * 1.0)
        
        # Scale the score (max at around 30% rhetoric words)
        intensity = min(1.0, weighted_count / (word_count * 0.3))
        
        return intensity
    
    logger.info(f"Calculating rhetoric intensity for {text_column}...")
    df[f"{result_prefix}_rhetoric_intensity"] = df[text_column].apply(calculate_intensity)
    
    # Categorize rhetoric intensity
    def categorize_intensity(score):
        if score < 0.2:
            return "Low"
        elif score < 0.5:
            return "Medium"
        elif score < 0.8:
            return "High"
        else:
            return "Very High"
    
    df[f"{result_prefix}_rhetoric_level"] = df[f"{result_prefix}_rhetoric_intensity"].apply(categorize_intensity)
    
    return df


def analyze_entity_sentiment(entities_df: pd.DataFrame, 
                           label_column: str = "Label",
                           method: str = "textblob") -> pd.DataFrame:
    """
    Analyze sentiment of entities extracted from news articles.
    
    Args:
        entities_df: DataFrame containing entities
        label_column: Column containing entity labels/names
        method: Sentiment analysis method ('textblob' or 'vader')
        
    Returns:
        DataFrame with entity sentiment analysis added
    """
    if label_column not in entities_df.columns:
        logger.warning(f"Label column '{label_column}' not found in DataFrame")
        return entities_df
    
    # Create a copy of the dataframe
    df = entities_df.copy()
    
    logger.info(f"Analyzing sentiment for {len(df)} entities using {method}...")
    
    if method == "textblob":
        try:
            from textblob import TextBlob
        except ImportError:
            logger.error("TextBlob not installed. Install with: pip install textblob")
            return entities_df
        
        # Helper function for TextBlob sentiment
        def analyze_entity_textblob(label):
            if pd.isna(label) or label == "":
                return pd.Series([None, None])
            
            blob = TextBlob(label)
            return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity])
        
        # Apply to all entities
        tqdm.pandas(desc="Analyzing entities")
        sentiment_df = df[label_column].progress_apply(analyze_entity_textblob)
        df["entity_polarity"] = sentiment_df[0]
        df["entity_subjectivity"] = sentiment_df[1]
        
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
        
        df["entity_sentiment"] = df["entity_polarity"].apply(categorize_sentiment)
        
    elif method == "vader":
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon')
        except ImportError:
            logger.error("NLTK not installed. Install with: pip install nltk")
            return entities_df
        
        # Initialize VADER
        sid = SentimentIntensityAnalyzer()
        
        # Helper function for VADER sentiment
        def analyze_entity_vader(label):
            if pd.isna(label) or label == "":
                return pd.Series([None, None, None, None])
            
            scores = sid.polarity_scores(label)
            return pd.Series([scores['neg'], scores['neu'], scores['pos'], scores['compound']])
        
        # Apply to all entities
        tqdm.pandas(desc="Analyzing entities")
        sentiment_df = df[label_column].progress_apply(analyze_entity_vader)
        df["entity_vader_negative"] = sentiment_df[0]
        df["entity_vader_neutral"] = sentiment_df[1]
        df["entity_vader_positive"] = sentiment_df[2]
        df["entity_vader_compound"] = sentiment_df[3]
        
        # Categorize sentiment
        def categorize_vader_sentiment(compound):
            if pd.isna(compound):
                return "Unknown"
            elif compound >= 0.05:
                return "Positive"
            elif compound <= -0.05:
                return "Negative"
            else:
                return "Neutral"
        
        df["entity_sentiment"] = df["entity_vader_compound"].apply(categorize_vader_sentiment)
    
    logger.info("Entity sentiment analysis complete")
    return df


def aggregate_article_sentiment(news_df: pd.DataFrame, 
                              title_sentiment_column: str = "title_sentiment",
                              abstract_sentiment_column: str = "abstract_sentiment") -> pd.DataFrame:
    """
    Aggregate sentiment from title and abstract to get overall article sentiment.
    
    Args:
        news_df: DataFrame containing news articles with sentiment analysis
        title_sentiment_column: Column containing title sentiment
        abstract_sentiment_column: Column containing abstract sentiment
        
    Returns:
        DataFrame with overall article sentiment added
    """
    if title_sentiment_column not in news_df.columns or abstract_sentiment_column not in news_df.columns:
        logger.warning(f"Required sentiment columns not found in DataFrame")
        return news_df
    
    # Create a copy of the dataframe
    df = news_df.copy()
    
    # Helper function to combine sentiments
    def combine_sentiments(row):
        title_sentiment = row[title_sentiment_column]
        abstract_sentiment = row[abstract_sentiment_column]
        
        # If either is missing, use the other
        if pd.isna(title_sentiment) or title_sentiment == "Unknown":
            return abstract_sentiment
        if pd.isna(abstract_sentiment) or abstract_sentiment == "Unknown":
            return title_sentiment
        
        # If both are the same, use that sentiment
        if title_sentiment == abstract_sentiment:
            return title_sentiment
        
        # If one is neutral and one is not, use the non-neutral one
        if title_sentiment == "Neutral":
            return abstract_sentiment
        if abstract_sentiment == "Neutral":
            return title_sentiment
        
        # If one is positive and one is negative, return "Mixed"
        if (title_sentiment == "Positive" and abstract_sentiment == "Negative") or \
           (title_sentiment == "Negative" and abstract_sentiment == "Positive"):
            return "Mixed"
        
        # Fallback
        return "Neutral"
    
    logger.info("Aggregating article sentiment...")
    df["article_sentiment"] = df.apply(combine_sentiments, axis=1)
    
    return df


# Visualization Functions
def plot_sentiment_distribution(news_df: pd.DataFrame, 
                              sentiment_column: str = "title_sentiment",
                              title: str = "Sentiment Distribution in Titles",
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the distribution of sentiment categories.
    
    Args:
        news_df: DataFrame containing news articles with sentiment analysis
        sentiment_column: Column containing sentiment categories
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if sentiment_column not in news_df.columns:
        logger.warning(f"Sentiment column '{sentiment_column}' not found in DataFrame")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get sentiment counts (excluding Unknown)
    sentiment_counts = news_df[news_df[sentiment_column] != "Unknown"][sentiment_column].value_counts()
    
    # Determine order and colors
    order = ["Positive", "Neutral", "Negative", "Mixed"]
    order = [o for o in order if o in sentiment_counts.index]
    colors = ["green", "gray", "red", "purple"]
    
    # Create countplot
    sns.countplot(
        data=news_df[news_df[sentiment_column] != "Unknown"],
        x=sentiment_column,
        order=order,
        palette=dict(zip(order, colors)),
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    
    # Add percentage labels
    total = sentiment_counts.sum()
    for i, count in enumerate(sentiment_counts[order]):
        ax.text(
            i, 
            count + 5, 
            f"{count/total:.1%}", 
            ha="center"
        )
    
    return fig


def plot_sentiment_pie(news_df: pd.DataFrame, 
                      sentiment_column: str = "title_sentiment",
                      title: str = "Sentiment Distribution in Titles",
                      figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Plot a pie chart of sentiment distribution.
    
    Args:
        news_df: DataFrame containing news articles with sentiment analysis
        sentiment_column: Column containing sentiment categories
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if sentiment_column not in news_df.columns:
        logger.warning(f"Sentiment column '{sentiment_column}' not found in DataFrame")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get sentiment counts (excluding Unknown)
    sentiment_counts = news_df[news_df[sentiment_column] != "Unknown"][sentiment_column].value_counts()
    
    # Determine colors
    colors = {
        "Positive": "green",
        "Neutral": "gray",
        "Negative": "red",
        "Mixed": "purple"
    }
    
    # Create pie chart
    ax.pie(
        sentiment_counts, 
        labels=sentiment_counts.index, 
        autopct='%1.1f%%',
        colors=[colors.get(s, "blue") for s in sentiment_counts.index],
        startangle=90
    )
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_title(title)
    
    return fig


def plot_sentiment_by_category(news_df: pd.DataFrame, 
                             sentiment_column: str = "title_sentiment",
                             category_column: str = "category",
                             title: str = "Sentiment Distribution by Category",
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot sentiment distribution across different categories.
    
    Args:
        news_df: DataFrame containing news articles with sentiment analysis
        sentiment_column: Column containing sentiment categories
        category_column: Column containing article categories
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if sentiment_column not in news_df.columns or category_column not in news_df.columns:
        logger.warning(f"Required columns not found in DataFrame")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out Unknown sentiment
    filtered_df = news_df[news_df[sentiment_column] != "Unknown"]
    
    # Determine order and colors
    sentiment_order = ["Positive", "Neutral", "Negative", "Mixed"]
    sentiment_order = [o for o in sentiment_order if o in filtered_df[sentiment_column].unique()]
    
    colors = ["green", "gray", "red", "purple"]
    color_palette = dict(zip(sentiment_order, colors[:len(sentiment_order)]))
    
    # For large number of categories, focus on top N
    if filtered_df[category_column].nunique() > 10:
        top_categories = filtered_df[category_column].value_counts().nlargest(8).index
        filtered_df = filtered_df[filtered_df[category_column].isin(top_categories)]
    
    # Create countplot
    sns.countplot(
        data=filtered_df,
        x=category_column,
        hue=sentiment_column,
        hue_order=sentiment_order,
        palette=color_palette,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.legend(title="Sentiment")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_sentiment_heatmap(news_df: pd.DataFrame, 
                         sentiment_column: str = "title_sentiment",
                         category_column: str = "category",
                         title: str = "Sentiment Heatmap by Category",
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot a heatmap of sentiment distribution across categories.
    
    Args:
        news_df: DataFrame containing news articles with sentiment analysis
        sentiment_column: Column containing sentiment categories
        category_column: Column containing article categories
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if sentiment_column not in news_df.columns or category_column not in news_df.columns:
        logger.warning(f"Required columns not found in DataFrame")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out Unknown sentiment
    filtered_df = news_df[news_df[sentiment_column] != "Unknown"]
    
    # Determine order
    sentiment_order = ["Positive", "Neutral", "Negative", "Mixed"]
    sentiment_order = [o for o in sentiment_order if o in filtered_df[sentiment_column].unique()]
    
    # For large number of categories, focus on top N
    if filtered_df[category_column].nunique() > 10:
        top_categories = filtered_df[category_column].value_counts().nlargest(8).index
        filtered_df = filtered_df[filtered_df[category_column].isin(top_categories)]
    
    # Create cross-tabulation
    sentiment_counts = pd.crosstab(
        filtered_df[category_column], 
        filtered_df[sentiment_column],
        normalize="index"
    ) * 100  # Convert to percentage
    
    # Reorder columns if possible
    available_sentiments = [s for s in sentiment_order if s in sentiment_counts.columns]
    sentiment_counts = sentiment_counts[available_sentiments]
    
    # Create heatmap
    sns.heatmap(
        sentiment_counts,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        vmin=0,
        vmax=100,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_ylabel("Category")
    ax.set_xlabel("Sentiment")
    
    return fig


def plot_subjectivity_distribution(news_df: pd.DataFrame, 
                                 subjectivity_column: str = "title_subjectivity",
                                 title: str = "Subjectivity Distribution in Titles",
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the distribution of subjectivity scores.
    
    Args:
        news_df: DataFrame containing news articles with sentiment analysis
        subjectivity_column: Column containing subjectivity scores
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if subjectivity_column not in news_df.columns:
        logger.warning(f"Subjectivity column '{subjectivity_column}' not found in DataFrame")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out missing values
    filtered_df = news_df[news_df[subjectivity_column].notna()]
    
    # Create histogram
    sns.histplot(
        data=filtered_df,
        x=subjectivity_column,
        bins=20,
        kde=True,
        color="purple",
        ax=ax
    )
    
    # Add vertical lines for reference
    mean = filtered_df[subjectivity_column].mean()
    median = filtered_df[subjectivity_column].median()
    ax.axvline(x=mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    ax.axvline(x=median, color='green', linestyle='--', label=f'Median: {median:.2f}')
    ax.axvline(x=0.5, color='black', linestyle=':', label='Midpoint')
    
    ax.set_title(title)
    ax.set_xlabel("Subjectivity Score (0=Objective, 1=Subjective)")
    ax.set_ylabel("Count")
    ax.legend()
    
    return fig


def plot_reading_level_distribution(news_df: pd.DataFrame, 
                                  reading_level_column: str = "title_reading_level",
                                  title: str = "Reading Level Distribution in Titles",
                                  figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the distribution of reading level categories.
    
    Args:
        news_df: DataFrame containing news articles with reading level analysis
        reading_level_column: Column containing reading level categories
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if reading_level_column not in news_df.columns:
        logger.warning(f"Reading level column '{reading_level_column}' not found in DataFrame")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out Unknown category
    filtered_df = news_df[news_df[reading_level_column] != "Unknown"]
    
    # Determine order
    order = ["Elementary", "Middle School", "High School", "College Level"]
    order = [o for o in order if o in filtered_df[reading_level_column].unique()]
    
    # Create countplot
    sns.countplot(
        data=filtered_df,
        x=reading_level_column,
        order=order,
        palette="viridis",
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel("Reading Level")
    ax.set_ylabel("Count")
    
    # Add percentage labels
    counts = filtered_df[reading_level_column].value_counts()
    total = counts.sum()
    for i, level in enumerate(order):
        if level in counts:
            count = counts[level]
            ax.text(
                i, 
                count + 5, 
                f"{count/total:.1%}", 
                ha="center"
            )
    
    return fig


def plot_rhetoric_intensity_distribution(news_df: pd.DataFrame, 
                                       intensity_column: str = "title_rhetoric_intensity",
                                       title: str = "Rhetoric Intensity Distribution in Titles",
                                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the distribution of rhetoric intensity scores.
    
    Args:
        news_df: DataFrame containing news articles with rhetoric analysis
        intensity_column: Column containing rhetoric intensity scores
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if intensity_column not in news_df.columns:
        logger.warning(f"Rhetoric intensity column '{intensity_column}' not found in DataFrame")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out missing values
    filtered_df = news_df[news_df[intensity_column].notna()]
    
    # Create histogram
    sns.histplot(
        data=filtered_df,
        x=intensity_column,
        bins=20,
        kde=True,
        color="orange",
        ax=ax
    )
    
    # Add vertical lines for reference
    mean = filtered_df[intensity_column].mean()
    median = filtered_df[intensity_column].median()
    ax.axvline(x=mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    ax.axvline(x=median, color='green', linestyle='--', label=f'Median: {median:.2f}')
    
    # Add intensity level thresholds
    ax.axvline(x=0.2, color='blue', linestyle=':', label='Low/Medium')
    ax.axvline(x=0.5, color='purple', linestyle=':', label='Medium/High')
    ax.axvline(x=0.8, color='darkred', linestyle=':', label='High/Very High')
    
    ax.set_title(title)
    ax.set_xlabel("Rhetoric Intensity Score (0-1)")
    ax.set_ylabel("Count")
    ax.legend()
    
    return fig


def plot_entity_sentiment_distribution(entities_df: pd.DataFrame, 
                                     sentiment_column: str = "entity_sentiment",
                                     type_column: str = "Type",
                                     title: str = "Entity Sentiment Distribution",
                                     figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot the distribution of entity sentiment across entity types.
    
    Args:
        entities_df: DataFrame containing entities with sentiment analysis
        sentiment_column: Column containing sentiment categories
        type_column: Column containing entity types
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if sentiment_column not in entities_df.columns or type_column not in entities_df.columns:
        logger.warning(f"Required columns not found in DataFrame")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out Unknown sentiment
    filtered_df = entities_df[entities_df[sentiment_column] != "Unknown"]
    
    # Determine sentiment order and colors
    sentiment_order = ["Positive", "Neutral", "Negative"]
    sentiment_order = [o for o in sentiment_order if o in filtered_df[sentiment_column].unique()]
    
    colors = ["green", "gray", "red"]
    color_palette = dict(zip(sentiment_order, colors[:len(sentiment_order)]))
    
    # For large number of entity types, focus on top N
    if filtered_df[type_column].nunique() > 10:
        top_types = filtered_df[type_column].value_counts().nlargest(8).index
        filtered_df = filtered_df[filtered_df[type_column].isin(top_types)]
    
    # Create countplot
    sns.countplot(
        data=filtered_df,
        x=type_column,
        hue=sentiment_column,
        hue_order=sentiment_order,
        palette=color_palette,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Count")
    ax.legend(title="Sentiment")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('/Users/sravansridhar/Documents/news_ai/advanced_eda/scripts')
    from mind_dataset_loader import MINDDatasetLoader
    
    # Load data
    loader = MINDDatasetLoader()
    news_df = loader.load_news()
    
    # Analyze sentiment using TextBlob
    print("Analyzing sentiment with TextBlob...")
    news_df = analyze_sentiment_textblob(news_df, text_column="title", result_prefix="title")
    news_df = analyze_sentiment_textblob(news_df, text_column="abstract", result_prefix="abstract")
    
    # Plot sentiment distribution
    fig = plot_sentiment_distribution(news_df, sentiment_column="title_sentiment")
    plt.show()
    
    # Plot sentiment by category
    fig = plot_sentiment_by_category(news_df, category_column="category")
    plt.show()