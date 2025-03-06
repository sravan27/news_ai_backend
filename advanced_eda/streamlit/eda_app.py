#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit app for interactive exploration of the MIND dataset.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / "scripts"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import custom modules
from scripts.mind_dataset_loader import MINDDatasetLoader, preprocess_news_text
from scripts.entity_analysis import (
    analyze_entity_distribution, analyze_entity_types, analyze_entity_confidence,
    analyze_entity_occurrences, identify_political_entities
)
from scripts.behavior_analysis import (
    extract_time_features, calculate_engagement_metrics, analyze_user_activity,
    analyze_click_patterns, analyze_user_interests
)
from scripts.sentiment_analysis import (
    analyze_sentiment_textblob, calculate_reading_level, calculate_rhetoric_intensity
)

# Set page configuration
st.set_page_config(
    page_title="MIND Dataset Explorer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize dataset loader
@st.cache_resource
def load_dataset_loader(dataset_size="large", split="train"):
    """Initialize and cache the dataset loader."""
    return MINDDatasetLoader(dataset_size=dataset_size, split=split)

# Cache data loading functions
@st.cache_data
def load_news_data(loader):
    """Load and cache news data."""
    return loader.load_news()

@st.cache_data
def load_behaviors_data(loader):
    """Load and cache behaviors data."""
    return loader.load_behaviors()

@st.cache_data
def process_entities(loader):
    """Process and cache entity data."""
    return loader.process_entities_long_format()

@st.cache_data
def load_entity_embeddings(loader):
    """Load and cache entity embeddings."""
    return loader.load_entity_embeddings()

@st.cache_data
def preprocess_texts(news_df):
    """Preprocess and cache text data."""
    return preprocess_news_text(news_df)

@st.cache_data
def analyze_sentiment(news_df, sample_size=1000):
    """Analyze and cache sentiment data."""
    # Take a sample for faster processing
    if len(news_df) > sample_size:
        news_sample = news_df.sample(sample_size, random_state=42)
    else:
        news_sample = news_df
        
    # Analyze title sentiment
    news_with_sentiment = analyze_sentiment_textblob(news_sample, text_column="title", result_prefix="title")
    # Analyze abstract sentiment
    news_with_sentiment = analyze_sentiment_textblob(news_with_sentiment, text_column="abstract", result_prefix="abstract")
    
    return news_with_sentiment

@st.cache_data
def calculate_text_metrics(news_with_sentiment):
    """Calculate and cache text metrics."""
    # Calculate reading level
    news_with_reading = calculate_reading_level(news_with_sentiment, text_column="title", result_prefix="title")
    news_with_reading = calculate_reading_level(news_with_reading, text_column="abstract", result_prefix="abstract")
    
    # Calculate rhetoric intensity
    news_with_metrics = calculate_rhetoric_intensity(news_with_reading, text_column="title", result_prefix="title")
    news_with_metrics = calculate_rhetoric_intensity(news_with_metrics, text_column="abstract", result_prefix="abstract")
    
    return news_with_metrics

@st.cache_data
def analyze_user_behaviors(behaviors_df):
    """Analyze and cache user behavior data."""
    # Process time features
    behaviors_with_time = extract_time_features(behaviors_df)
    
    # Calculate engagement metrics
    behaviors_with_metrics = calculate_engagement_metrics(behaviors_with_time)
    
    return behaviors_with_metrics

# Visualization functions
def plot_category_distribution(news_df):
    """Plot news category distribution."""
    category_counts = news_df['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']
    category_counts['percentage'] = category_counts['count'] / category_counts['count'].sum() * 100
    
    fig = px.bar(
        category_counts, 
        x='category', 
        y='count',
        text=category_counts['percentage'].apply(lambda x: f'{x:.1f}%'),
        title='News Categories Distribution',
        labels={'category': 'Category', 'count': 'Count'},
        color='count',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_sentiment_distribution(news_with_sentiment, column="title_sentiment"):
    """Plot sentiment distribution."""
    sentiment_counts = news_with_sentiment[column].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    sentiment_counts['percentage'] = sentiment_counts['count'] / sentiment_counts['count'].sum() * 100
    
    # Define color map
    color_map = {
        'Positive': 'green',
        'Neutral': 'gray',
        'Negative': 'red',
        'Mixed': 'purple',
        'Unknown': 'lightgray'
    }
    
    # Add colors based on sentiment
    sentiment_counts['color'] = sentiment_counts['sentiment'].map(color_map)
    
    title_prefix = "Title" if column.startswith("title") else "Abstract"
    
    fig = px.bar(
        sentiment_counts, 
        x='sentiment', 
        y='count',
        text=sentiment_counts['percentage'].apply(lambda x: f'{x:.1f}%'),
        title=f'{title_prefix} Sentiment Distribution',
        labels={'sentiment': 'Sentiment', 'count': 'Count'},
        color='sentiment',
        color_discrete_map=color_map
    )
    
    return fig

def plot_entity_distribution(entities_df, entity_column="Label", top_n=20):
    """Plot entity distribution."""
    entity_counts = analyze_entity_distribution(entities_df, entity_column=entity_column)
    top_entities = entity_counts.nlargest(top_n, 'count')
    
    fig = px.bar(
        top_entities, 
        x='count', 
        y=entity_column,
        title=f'Top {top_n} Entities',
        labels={entity_column: 'Entity', 'count': 'Count'},
        color='count',
        color_continuous_scale='viridis',
        orientation='h'
    )
    
    return fig

def plot_entity_types_pie(entities_df, type_column="Type", type_mapping=None):
    """Plot entity types as pie chart."""
    # Analyze entity types
    type_counts = analyze_entity_types(entities_df, type_column=type_column, type_mapping=type_mapping)
    
    if type_mapping:
        type_column = f"{type_column}_desc"
    
    fig = px.pie(
        type_counts, 
        values='count', 
        names=type_column,
        title='Entity Types Distribution',
        hole=0.4
    )
    
    return fig

def plot_user_activity_by_hour(behaviors_df):
    """Plot user activity by hour."""
    # Extract hour from time
    if pd.api.types.is_datetime64_any_dtype(behaviors_df['time']):
        hour_counts = behaviors_df['time'].dt.hour.value_counts().reset_index()
    else:
        # Convert to datetime first
        time_as_dt = pd.to_datetime(behaviors_df['time'])
        hour_counts = time_as_dt.dt.hour.value_counts().reset_index()
        
    hour_counts.columns = ['hour', 'count']
    hour_counts = hour_counts.sort_values('hour')
    
    fig = px.line(
        hour_counts, 
        x='hour', 
        y='count',
        markers=True,
        title='User Activity by Hour of Day',
        labels={'hour': 'Hour', 'count': 'Activity Count'}
    )
    
    # Add reference lines for morning, afternoon, evening
    fig.add_vline(x=6, line_dash="dash", line_color="gray", annotation_text="Morning")
    fig.add_vline(x=12, line_dash="dash", line_color="gray", annotation_text="Noon")
    fig.add_vline(x=18, line_dash="dash", line_color="gray", annotation_text="Evening")
    
    return fig

def plot_user_engagement_histogram(behaviors_with_metrics):
    """Plot user engagement metrics histogram."""
    if 'history_length' not in behaviors_with_metrics.columns:
        return None
        
    fig = px.histogram(
        behaviors_with_metrics,
        x='history_length',
        nbins=30,
        title='Distribution of Articles Read Per User',
        labels={'history_length': 'Number of Articles in History', 'count': 'Frequency'},
        marginal='box'
    )
    
    return fig

def plot_click_rate_by_position(behaviors_with_metrics):
    """Plot click rate by position."""
    # Analyze click patterns
    click_patterns = analyze_click_patterns(behaviors_with_metrics)
    
    if 'click_by_position' not in click_patterns:
        return None
        
    position_df = click_patterns['click_by_position']
    
    fig = px.line(
        position_df,
        x='position',
        y='click_rate',
        markers=True,
        title='Click Rate by Position in Impression List',
        labels={'position': 'Position', 'click_rate': 'Click Rate'}
    )
    
    return fig

def plot_entity_embeddings_pca(entity_embeddings, entity_to_label=None, sample_size=1000):
    """Plot entity embeddings using PCA."""
    # Take a sample of entity embeddings
    sample_entity_ids = list(entity_embeddings.keys())[:sample_size]
    sample_embeddings = {eid: entity_embeddings[eid] for eid in sample_entity_ids}
    
    # Apply PCA
    entity_matrix = np.stack([sample_embeddings[eid] for eid in sample_entity_ids])
    pca = PCA(n_components=2)
    entity_2d = pca.fit_transform(entity_matrix)
    
    # Create DataFrame for plotting
    viz_df = pd.DataFrame({
        'entity_id': sample_entity_ids,
        'x': entity_2d[:, 0],
        'y': entity_2d[:, 1]
    })
    
    # Add labels if available
    if entity_to_label:
        viz_df['label'] = viz_df['entity_id'].map(lambda x: entity_to_label.get(x, x))
    else:
        viz_df['label'] = viz_df['entity_id']
    
    # Create scatter plot
    fig = px.scatter(
        viz_df,
        x='x',
        y='y',
        hover_data=['entity_id', 'label'],
        title=f'Entity Embeddings (PCA) - {sample_size} entities'
    )
    
    # Add variance explained annotation
    variance_explained = pca.explained_variance_ratio_.sum()
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=f"Variance explained: {variance_explained:.2%}",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig

def plot_text_metrics_by_category(news_with_metrics):
    """Plot text metrics by category."""
    if 'title_polarity' not in news_with_metrics.columns:
        return None
        
    # Group by category and calculate mean metrics
    category_metrics = news_with_metrics.groupby('category').agg({
        'title_polarity': 'mean',
        'title_subjectivity': 'mean', 
        'title_fk_grade': 'mean',
        'title_rhetoric_intensity': 'mean'
    }).reset_index()
    
    # Rename columns for better display
    category_metrics.columns = ['Category', 'Sentiment', 'Subjectivity', 'Reading Grade', 'Rhetoric']
    
    # Melt the dataframe for better plotting
    melted_metrics = pd.melt(
        category_metrics, 
        id_vars=['Category'],
        value_vars=['Sentiment', 'Subjectivity', 'Reading Grade', 'Rhetoric'],
        var_name='Metric', 
        value_name='Value'
    )
    
    # Normalize reading grade to 0-1 scale (assuming max grade is 20)
    melted_metrics.loc[melted_metrics['Metric'] == 'Reading Grade', 'Value'] = \
        melted_metrics.loc[melted_metrics['Metric'] == 'Reading Grade', 'Value'] / 20
    
    # Normalize sentiment to 0-1 scale (from -1 to 1)
    melted_metrics.loc[melted_metrics['Metric'] == 'Sentiment', 'Value'] = \
        (melted_metrics.loc[melted_metrics['Metric'] == 'Sentiment', 'Value'] + 1) / 2
    
    # Create heatmap
    fig = px.density_heatmap(
        melted_metrics,
        x='Category',
        y='Metric',
        z='Value',
        title='News Content Metrics by Category',
        labels={'Value': 'Normalized Value'},
        color_continuous_scale='viridis'
    )
    
    return fig

# Main app function
def main():
    """Main Streamlit application."""
    st.title("MIND Dataset Explorer")
    st.write("Interactive exploration of the Microsoft News Dataset (MIND) for news recommendation.")
    
    # Sidebar for dataset selection
    st.sidebar.header("Dataset Settings")
    dataset_size = st.sidebar.selectbox("Dataset Size", ["large", "small"], index=0)
    split = st.sidebar.selectbox("Dataset Split", ["train", "dev", "test"], index=0)
    
    # Initialize loader
    try:
        loader = load_dataset_loader(dataset_size=dataset_size, split=split)
        st.sidebar.success(f"Using MIND{dataset_size.capitalize()} {split} dataset")
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {e}")
        st.stop()
    
    # Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Section",
        ["Overview", "News Content", "Entity Analysis", "User Behavior", "Text Analysis", "Entity Embeddings"]
    )
    
    # Load data based on selected page
    if page in ["Overview", "News Content", "Text Analysis"]:
        with st.spinner("Loading news data..."):
            news_df = load_news_data(loader)
    
    if page in ["Overview", "User Behavior"]:
        with st.spinner("Loading behavior data..."):
            behaviors_df = load_behaviors_data(loader)
    
    if page in ["Overview", "Entity Analysis", "Entity Embeddings"]:
        with st.spinner("Processing entities..."):
            title_entities_df, abstract_entities_df = process_entities(loader)
            
    if page == "Entity Embeddings":
        with st.spinner("Loading entity embeddings..."):
            entity_embeddings = load_entity_embeddings(loader)
    
    # Render selected page
    if page == "Overview":
        render_overview(news_df, behaviors_df, title_entities_df, abstract_entities_df)
    elif page == "News Content":
        render_news_content(news_df)
    elif page == "Entity Analysis":
        render_entity_analysis(title_entities_df, abstract_entities_df, loader)
    elif page == "User Behavior":
        render_user_behavior(behaviors_df)
    elif page == "Text Analysis":
        render_text_analysis(news_df)
    elif page == "Entity Embeddings":
        render_entity_embeddings(entity_embeddings, title_entities_df)

def render_overview(news_df, behaviors_df, title_entities_df, abstract_entities_df):
    """Render overview page."""
    st.header("Dataset Overview")
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("News Articles", f"{len(news_df):,}")
    with col2:
        st.metric("User Behaviors", f"{len(behaviors_df):,}")
    with col3:
        st.metric("Unique Entities", f"{len(set(title_entities_df['WikidataId'].dropna()) | set(abstract_entities_df['WikidataId'].dropna())):,}")
    
    # Dataset overview tabs
    tab1, tab2, tab3 = st.tabs(["News", "Behaviors", "Entities"])
    
    with tab1:
        st.subheader("News Categories")
        fig = plot_category_distribution(news_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("News Sample")
        st.dataframe(news_df.head(5))
    
    with tab2:
        st.subheader("User Activity by Hour")
        behaviors_with_time = extract_time_features(behaviors_df)
        fig = plot_user_activity_by_hour(behaviors_with_time)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Behaviors Sample")
        st.dataframe(behaviors_df.head(5))
    
    with tab3:
        st.subheader("Entity Types Distribution")
        entity_type_mapping = loader.get_entity_type_mapping()
        fig = plot_entity_types_pie(title_entities_df, type_mapping=entity_type_mapping)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Top Entities")
        fig = plot_entity_distribution(title_entities_df, top_n=10)
        st.plotly_chart(fig, use_container_width=True)

def render_news_content(news_df):
    """Render news content page."""
    st.header("News Content Analysis")
    
    # Process news text
    with st.spinner("Processing news text..."):
        news_df_processed = preprocess_news_text(news_df)
    
    # News content analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("News Categories")
        fig = plot_category_distribution(news_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Word count statistics
        st.subheader("Word Count Statistics")
        word_count_stats = pd.DataFrame({
            "Metric": ["Mean", "Median", "Min", "Max"],
            "Title": [
                news_df_processed["title_word_count"].mean(),
                news_df_processed["title_word_count"].median(),
                news_df_processed["title_word_count"].min(),
                news_df_processed["title_word_count"].max()
            ],
            "Abstract": [
                news_df_processed["abstract_word_count"].mean(),
                news_df_processed["abstract_word_count"].median(),
                news_df_processed["abstract_word_count"].min(),
                news_df_processed["abstract_word_count"].max()
            ]
        })
        st.dataframe(word_count_stats)
    
    with col2:
        st.subheader("Top Subcategories")
        top_subcategories = news_df['subcategory'].value_counts().head(15).reset_index()
        top_subcategories.columns = ['subcategory', 'count']
        
        fig = px.bar(
            top_subcategories,
            x='subcategory',
            y='count',
            title='Top 15 News Subcategories',
            labels={'subcategory': 'Subcategory', 'count': 'Count'},
            color='count',
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Word count distributions
    st.subheader("Word Count Distributions")
    
    tab1, tab2, tab3 = st.tabs(["Title", "Abstract", "Title vs Abstract"])
    
    with tab1:
        fig = px.histogram(
            news_df_processed,
            x="title_word_count",
            nbins=30,
            title="Title Word Count Distribution",
            labels={"title_word_count": "Word Count", "count": "Frequency"},
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.histogram(
            news_df_processed,
            x="abstract_word_count",
            nbins=30,
            title="Abstract Word Count Distribution",
            labels={"abstract_word_count": "Word Count", "count": "Frequency"},
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.scatter(
            news_df_processed,
            x="title_word_count",
            y="abstract_word_count",
            title="Title Word Count vs Abstract Word Count",
            labels={"title_word_count": "Title Word Count", "abstract_word_count": "Abstract Word Count"},
            opacity=0.6,
            color_discrete_sequence=["blue"]
        )
        
        # Add mean lines
        fig.add_hline(y=news_df_processed["abstract_word_count"].mean(), line_dash="dash", line_color="red", annotation_text="Mean Abstract")
        fig.add_vline(x=news_df_processed["title_word_count"].mean(), line_dash="dash", line_color="green", annotation_text="Mean Title")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive news explorer
    st.subheader("Interactive News Explorer")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        selected_category = st.selectbox("Select Category", ["All"] + news_df["category"].unique().tolist())
    with col2:
        min_words = st.slider("Minimum Abstract Words", 0, 100, 0)
    
    # Filter data
    filtered_df = news_df_processed.copy()
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]
    if min_words > 0:
        filtered_df = filtered_df[filtered_df["abstract_word_count"] >= min_words]
    
    # Display filtered data
    st.write(f"Showing {len(filtered_df)} news articles")
    st.dataframe(filtered_df[["news_id", "category", "subcategory", "title", "abstract", "title_word_count", "abstract_word_count"]])

def render_entity_analysis(title_entities_df, abstract_entities_df, loader):
    """Render entity analysis page."""
    st.header("Entity Analysis")
    
    # Entity type mapping
    entity_type_mapping = loader.get_entity_type_mapping()
    
    # Title vs Abstract selection
    entity_source = st.radio("Entity Source", ["Title", "Abstract"], horizontal=True)
    
    if entity_source == "Title":
        entities_df = title_entities_df
        title_prefix = "Title"
    else:
        entities_df = abstract_entities_df
        title_prefix = "Abstract"
    
    # Entity analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Entity Distribution", "Entity Types", "Entity Confidence", "Political Entities"])
    
    with tab1:
        st.subheader(f"Top Entities in {title_prefix}")
        top_n = st.slider("Number of top entities to show", 5, 50, 20)
        fig = plot_entity_distribution(entities_df, top_n=top_n)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader(f"Entity Types in {title_prefix}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_entity_types_pie(entities_df, type_mapping=entity_type_mapping)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Get entity type counts
            type_counts = analyze_entity_types(entities_df, type_column="Type", type_mapping=entity_type_mapping)
            type_column = "Type_desc" if entity_type_mapping else "Type"
            
            # Display type counts
            st.write("Entity Type Counts")
            st.dataframe(type_counts)
    
    with tab3:
        st.subheader(f"Entity Confidence in {title_prefix}")
        
        # Analyze entity confidence
        entities_with_occurrences = analyze_entity_occurrences(entities_df)
        confidence_stats = analyze_entity_confidence(entities_with_occurrences)
        
        if confidence_stats:
            col1, col2 = st.columns(2)
            
            with col1:
                # Display confidence statistics
                st.write("Confidence Statistics")
                stats_df = pd.DataFrame({
                    "Metric": ["Mean", "Median", "Min", "Max", "Std Dev"],
                    "Value": [
                        confidence_stats["mean"],
                        confidence_stats["median"],
                        confidence_stats["min"],
                        confidence_stats["max"],
                        confidence_stats["std"]
                    ]
                })
                st.dataframe(stats_df)
            
            with col2:
                # Confidence distribution
                fig = px.histogram(
                    entities_with_occurrences,
                    x="Confidence",
                    nbins=30,
                    title=f"Entity Detection Confidence in {title_prefix}",
                    labels={"Confidence": "Confidence Score", "count": "Frequency"},
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Number of occurrences 
            st.subheader("Number of Entity Occurrences")
            fig = px.histogram(
                entities_with_occurrences,
                x="num_occurrences",
                nbins=20,
                title=f"Number of Entity Occurrences in {title_prefix}",
                labels={"num_occurrences": "Number of Occurrences", "count": "Frequency"},
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader(f"Political Entities in {title_prefix}")
        
        # Identify political entities
        political_entities = identify_political_entities(entities_df)
        
        # Calculate percentage
        political_percent = political_entities['is_political_entity'].mean() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Political vs non-political pie chart
            labels = ['Political', 'Non-Political']
            values = [political_percent, 100 - political_percent]
            
            fig = px.pie(
                values=values,
                names=labels,
                title=f"Political vs Non-Political Entities in {title_prefix}",
                color_discrete_sequence=['red', 'blue']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show political entity stats
            st.write("Political Entity Statistics")
            stats_df = pd.DataFrame({
                "Metric": ["Political Entities", "Non-Political Entities", "Political Percentage"],
                "Value": [
                    f"{political_entities['is_political_entity'].sum():,}",
                    f"{(~political_entities['is_political_entity']).sum():,}",
                    f"{political_percent:.2f}%"
                ]
            })
            st.dataframe(stats_df)
        
        # Show political entities
        st.write("Sample of Political Entities")
        political_sample = political_entities[political_entities['is_political_entity']].head(20)
        st.dataframe(political_sample[['Label', 'Type', 'WikidataId', 'Confidence']])

def render_user_behavior(behaviors_df):
    """Render user behavior page."""
    st.header("User Behavior Analysis")
    
    # Process user behaviors
    with st.spinner("Analyzing user behaviors..."):
        behaviors_with_metrics = analyze_user_behaviors(behaviors_df)
    
    # User behavior tabs
    tab1, tab2, tab3 = st.tabs(["Engagement", "Time Patterns", "Click Behavior"])
    
    with tab1:
        st.subheader("User Engagement Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Engagement metrics statistics
            engagement_stats = pd.DataFrame({
                "Metric": ["Mean", "Median", "Min", "Max"],
                "History Length": [
                    behaviors_with_metrics["history_length"].mean(),
                    behaviors_with_metrics["history_length"].median(),
                    behaviors_with_metrics["history_length"].min(),
                    behaviors_with_metrics["history_length"].max()
                ],
                "Click Rate": [
                    behaviors_with_metrics["click_rate"].mean(),
                    behaviors_with_metrics["click_rate"].median(),
                    behaviors_with_metrics["click_rate"].min(),
                    behaviors_with_metrics["click_rate"].max()
                ]
            })
            st.dataframe(engagement_stats)
        
        with col2:
            # Engagement level distribution
            if 'engagement_level' in behaviors_with_metrics.columns:
                engagement_counts = behaviors_with_metrics['engagement_level'].value_counts().reset_index()
                engagement_counts.columns = ['level', 'count']
                
                # Ensure correct order
                level_order = ["No Engagement", "Low", "Medium", "High", "Very High"]
                level_order = [level for level in level_order if level in engagement_counts['level'].values]
                
                engagement_counts['order'] = engagement_counts['level'].map({level: i for i, level in enumerate(level_order)})
                engagement_counts = engagement_counts.sort_values('order')
                
                fig = px.bar(
                    engagement_counts,
                    x='level',
                    y='count',
                    title='User Engagement Level Distribution',
                    labels={'level': 'Engagement Level', 'count': 'Count'},
                    color='level',
                    category_orders={'level': level_order}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # History length distribution
        fig = plot_user_engagement_histogram(behaviors_with_metrics)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Click rate distribution
        if 'click_rate' in behaviors_with_metrics.columns:
            fig = px.histogram(
                behaviors_with_metrics,
                x='click_rate',
                nbins=30,
                title='Click Rate Distribution',
                labels={'click_rate': 'Click Rate', 'count': 'Frequency'},
                marginal='box'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("User Activity Time Patterns")
        
        # Activity by hour
        fig = plot_user_activity_by_hour(behaviors_with_metrics)
        st.plotly_chart(fig, use_container_width=True)
        
        # Activity by day of week
        if 'day_name' in behaviors_with_metrics.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = behaviors_with_metrics['day_name'].value_counts().reset_index()
            day_counts.columns = ['day', 'count']
            
            fig = px.bar(
                day_counts,
                x='day',
                y='count',
                title='User Activity by Day of Week',
                labels={'day': 'Day', 'count': 'Activity Count'},
                color='count',
                color_continuous_scale='viridis',
                category_orders={'day': day_order}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Activity by time of day
        if 'time_of_day' in behaviors_with_metrics.columns:
            time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
            time_counts = behaviors_with_metrics['time_of_day'].value_counts().reset_index()
            time_counts.columns = ['time_of_day', 'count']
            
            fig = px.pie(
                time_counts,
                values='count',
                names='time_of_day',
                title='User Activity by Time of Day',
                color_discrete_sequence=px.colors.sequential.Viridis,
                category_orders={'time_of_day': time_order}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Click Patterns Analysis")
        
        # Click rate by position
        fig = plot_click_rate_by_position(behaviors_with_metrics)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click rate by position data not available.")
        
        # Analyze user interests if news_df is already loaded
        try:
            news_df = load_news_data(load_dataset_loader())
            user_interests = analyze_user_interests(behaviors_with_metrics, news_df)
            
            if 'category_popularity' in user_interests:
                st.subheader("Category Popularity")
                category_df = user_interests['category_popularity']
                
                fig = px.bar(
                    category_df,
                    x='category',
                    y='total_count',
                    title='Category Popularity Across All Users',
                    labels={'category': 'Category', 'total_count': 'Total Count'},
                    color='total_count',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not analyze user interests: {e}")

def render_text_analysis(news_df):
    """Render text analysis page."""
    st.header("Text Analysis")
    
    # Process sentiment and text metrics
    with st.spinner("Analyzing sentiment and text metrics..."):
        news_with_sentiment = analyze_sentiment(news_df)
        news_with_metrics = calculate_text_metrics(news_with_sentiment)
    
    # Text analysis tabs
    tab1, tab2, tab3 = st.tabs(["Sentiment", "Reading Level", "Text Metrics"])
    
    with tab1:
        st.subheader("Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Title sentiment
            fig = plot_sentiment_distribution(news_with_sentiment, column="title_sentiment")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Abstract sentiment
            fig = plot_sentiment_distribution(news_with_sentiment, column="abstract_sentiment")
            st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment by category
        st.subheader("Sentiment by Category")
        
        # Create crosstab
        sentiment_by_category = pd.crosstab(
            news_with_sentiment['category'], 
            news_with_sentiment['title_sentiment'],
            normalize='index'
        ) * 100
        
        # Ensure all sentiment columns exist
        for col in ['Positive', 'Neutral', 'Negative']:
            if col not in sentiment_by_category.columns:
                sentiment_by_category[col] = 0
        
        # Keep only Positive, Neutral, Negative columns
        sentiment_by_category = sentiment_by_category[['Positive', 'Neutral', 'Negative']]
        
        # Create heatmap
        fig = px.imshow(
            sentiment_by_category,
            labels=dict(x="Sentiment", y="Category", color="Percentage"),
            x=['Positive', 'Neutral', 'Negative'],
            text_auto='.1f',
            aspect="auto",
            title="Sentiment Distribution by Category (%)",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Reading Level Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Title reading level
            if 'title_reading_level' in news_with_metrics.columns:
                reading_counts = news_with_metrics['title_reading_level'].value_counts().reset_index()
                reading_counts.columns = ['level', 'count']
                
                # Ensure correct order
                level_order = ["Elementary", "Middle School", "High School", "College Level"]
                level_order = [level for level in level_order if level in reading_counts['level'].values]
                
                reading_counts['order'] = reading_counts['level'].map({level: i for i, level in enumerate(level_order)})
                reading_counts = reading_counts.sort_values('order')
                
                fig = px.bar(
                    reading_counts,
                    x='level',
                    y='count',
                    title='Title Reading Level Distribution',
                    labels={'level': 'Reading Level', 'count': 'Count'},
                    color='level',
                    category_orders={'level': level_order}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Abstract reading level
            if 'abstract_reading_level' in news_with_metrics.columns:
                reading_counts = news_with_metrics['abstract_reading_level'].value_counts().reset_index()
                reading_counts.columns = ['level', 'count']
                
                # Ensure correct order
                level_order = ["Elementary", "Middle School", "High School", "College Level"]
                level_order = [level for level in level_order if level in reading_counts['level'].values]
                
                reading_counts['order'] = reading_counts['level'].map({level: i for i, level in enumerate(level_order)})
                reading_counts = reading_counts.sort_values('order')
                
                fig = px.bar(
                    reading_counts,
                    x='level',
                    y='count',
                    title='Abstract Reading Level Distribution',
                    labels={'level': 'Reading Level', 'count': 'Count'},
                    color='level',
                    category_orders={'level': level_order}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Reading grade distribution
        st.subheader("Reading Grade Level Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Title Flesch-Kincaid grade
            if 'title_fk_grade' in news_with_metrics.columns:
                fig = px.histogram(
                    news_with_metrics,
                    x='title_fk_grade',
                    nbins=20,
                    title='Title Flesch-Kincaid Grade Distribution',
                    labels={'title_fk_grade': 'Grade Level', 'count': 'Frequency'},
                    marginal='box'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Abstract Flesch-Kincaid grade
            if 'abstract_fk_grade' in news_with_metrics.columns:
                fig = px.histogram(
                    news_with_metrics,
                    x='abstract_fk_grade',
                    nbins=20,
                    title='Abstract Flesch-Kincaid Grade Distribution',
                    labels={'abstract_fk_grade': 'Grade Level', 'count': 'Frequency'},
                    marginal='box'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Content Metrics Analysis")
        
        # Rhetoric intensity
        st.subheader("Rhetoric Intensity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Title rhetoric intensity
            if 'title_rhetoric_intensity' in news_with_metrics.columns:
                fig = px.histogram(
                    news_with_metrics,
                    x='title_rhetoric_intensity',
                    nbins=20,
                    title='Title Rhetoric Intensity Distribution',
                    labels={'title_rhetoric_intensity': 'Rhetoric Intensity (0-1)', 'count': 'Frequency'},
                    marginal='box'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Abstract rhetoric intensity
            if 'abstract_rhetoric_intensity' in news_with_metrics.columns:
                fig = px.histogram(
                    news_with_metrics,
                    x='abstract_rhetoric_intensity',
                    nbins=20,
                    title='Abstract Rhetoric Intensity Distribution',
                    labels={'abstract_rhetoric_intensity': 'Rhetoric Intensity (0-1)', 'count': 'Frequency'},
                    marginal='box'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Subjectivity
        st.subheader("Subjectivity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Title subjectivity
            if 'title_subjectivity' in news_with_metrics.columns:
                fig = px.histogram(
                    news_with_metrics,
                    x='title_subjectivity',
                    nbins=20,
                    title='Title Subjectivity Distribution',
                    labels={'title_subjectivity': 'Subjectivity (0=Objective, 1=Subjective)', 'count': 'Frequency'},
                    marginal='box'
                )
                
                # Add vertical line for midpoint
                fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Midpoint")
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Abstract subjectivity
            if 'abstract_subjectivity' in news_with_metrics.columns:
                fig = px.histogram(
                    news_with_metrics,
                    x='abstract_subjectivity',
                    nbins=20,
                    title='Abstract Subjectivity Distribution',
                    labels={'abstract_subjectivity': 'Subjectivity (0=Objective, 1=Subjective)', 'count': 'Frequency'},
                    marginal='box'
                )
                
                # Add vertical line for midpoint
                fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Midpoint")
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Text metrics by category
        st.subheader("Text Metrics by Category")
        
        fig = plot_text_metrics_by_category(news_with_metrics)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

def render_entity_embeddings(entity_embeddings, title_entities_df):
    """Render entity embeddings page."""
    st.header("Entity Embeddings Visualization")
    
    # Entity embedding stats
    embedding_sizes = []
    for entity_id, embedding in list(entity_embeddings.items())[:100]:  # Sample first 100 for speed
        embedding_sizes.append(len(embedding))
    
    st.write(f"Entity embeddings available: {len(entity_embeddings):,}")
    st.write(f"Embedding dimension: {np.mean(embedding_sizes):.0f}")
    
    # Create entity to label mapping
    entity_to_label = dict(zip(title_entities_df['WikidataId'], title_entities_df['Label']))
    
    # Visualization options
    st.subheader("Embedding Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        viz_method = st.radio("Visualization Method", ["PCA", "t-SNE"])
    with col2:
        sample_size = st.slider("Number of entities to visualize", 100, 2000, 500, step=100)
    
    # Create visualization
    with st.spinner(f"Generating {viz_method} visualization for {sample_size} entities..."):
        if viz_method == "PCA":
            fig = plot_entity_embeddings_pca(entity_embeddings, entity_to_label, sample_size)
            st.plotly_chart(fig, use_container_width=True)
        else:  # t-SNE
            # Take a sample of entity embeddings
            sample_entity_ids = list(entity_embeddings.keys())[:sample_size]
            sample_embeddings = {eid: entity_embeddings[eid] for eid in sample_entity_ids}
            
            # Apply t-SNE
            try:
                entity_matrix = np.stack([sample_embeddings[eid] for eid in sample_entity_ids])
                tsne = TSNE(n_components=2, perplexity=min(30, sample_size-1), random_state=42)
                entity_2d = tsne.fit_transform(entity_matrix)
                
                # Create DataFrame for plotting
                viz_df = pd.DataFrame({
                    'entity_id': sample_entity_ids,
                    'x': entity_2d[:, 0],
                    'y': entity_2d[:, 1]
                })
                
                # Add labels if available
                viz_df['label'] = viz_df['entity_id'].map(lambda x: entity_to_label.get(x, x))
                
                # Create scatter plot
                fig = px.scatter(
                    viz_df,
                    x='x',
                    y='y',
                    hover_data=['entity_id', 'label'],
                    title=f'Entity Embeddings (t-SNE) - {sample_size} entities'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating t-SNE visualization: {e}")
    
    # Entity similarity search
    st.subheader("Entity Similarity Search")
    
    # Get entities with labels for selection
    entities_with_labels = {}
    for entity_id, label in entity_to_label.items():
        if entity_id in entity_embeddings:
            entities_with_labels[entity_id] = f"{label} ({entity_id})"
    
    # Sort by label for easier selection
    sorted_entities = sorted(entities_with_labels.items(), key=lambda x: x[1])
    entity_options = [f"{label}" for entity_id, label in sorted_entities[:1000]]  # Limit to 1000 for performance
    
    selected_entity = st.selectbox("Select an entity to find similar entities", [""] + entity_options)
    
    if selected_entity:
        # Extract entity_id from selection
        entity_id = selected_entity.split("(")[-1].strip(")")
        
        if entity_id in entity_embeddings:
            with st.spinner(f"Finding entities similar to {selected_entity}..."):
                # Get entity embedding
                query_embedding = entity_embeddings[entity_id]
                
                try:
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    # Calculate similarity to all other entities
                    similarities = {}
                    for eid, embedding in entity_embeddings.items():
                        if eid != entity_id:
                            # Calculate cosine similarity
                            sim = cosine_similarity(
                                query_embedding.reshape(1, -1),
                                embedding.reshape(1, -1)
                            )[0][0]
                            similarities[eid] = sim
                    
                    # Sort by similarity (descending)
                    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                    
                    # Display top 10 similar entities
                    st.write(f"Top 10 entities most similar to {selected_entity}:")
                    
                    similar_entities = []
                    for eid, sim in sorted_similarities[:10]:
                        label = entity_to_label.get(eid, eid)
                        similar_entities.append({
                            "Entity ID": eid,
                            "Label": label,
                            "Similarity": sim
                        })
                    
                    st.dataframe(pd.DataFrame(similar_entities))
                except Exception as e:
                    st.error(f"Error calculating entity similarities: {e}")
        else:
            st.error(f"Entity {entity_id} not found in embeddings")

if __name__ == "__main__":
    main()