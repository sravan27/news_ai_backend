"""
Interactive Streamlit app for exploring MIND dataset.

This app provides interactive visualizations and insights from 
the Microsoft News Dataset (MIND).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

# Set up paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../..'))
SCRIPTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'scripts'))

# Add directories to the Python path
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(CURRENT_DIR))  # advanced_eda directory
sys.path.insert(0, SCRIPTS_DIR)

# Import custom modules
from advanced_eda.scripts.data_loader import (
    load_news_data, load_behaviors_data, load_entity_embeddings, load_relation_embeddings,
    parse_entities, process_entities_to_long_format, extract_entity_wikidata_ids,
    merge_news_with_embeddings, fetch_wikidata_info
)

from advanced_eda.scripts.entity_analysis import (
    analyze_entity_distribution, analyze_entity_types,
    analyze_entity_confidence, analyze_entity_co_occurrence
)

from advanced_eda.scripts.news_analysis import (
    analyze_news_categories, analyze_news_subcategories,
    analyze_category_subcategory_relationship,
    analyze_title_length, analyze_abstract_length, analyze_top_words,
    analyze_news_with_entities, perform_advanced_sentiment_analysis
)

from advanced_eda.scripts.user_analysis import (
    extract_clicks_and_impressions, analyze_user_engagement,
    analyze_temporal_patterns, analyze_click_through_rate, 
    analyze_history_patterns, analyze_user_categories, analyze_user_segments
)

# Set page configuration
st.set_page_config(
    page_title="MIND Dataset Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure styles
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0D47A1;
        border-bottom: 2px solid #0D47A1;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        color: #1976D2;
        margin-top: 1.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_dataset(split="MINDlarge_train", sample_size=10000):
    """Load and sample dataset."""
    data_dir = os.path.join(REPO_ROOT, 'MINDLarge')
    news_path = os.path.join(data_dir, split, 'news.tsv')
    behaviors_path = os.path.join(data_dir, split, 'behaviors.tsv')
    
    news_df = load_news_data(news_path)
    behaviors_df = load_behaviors_data(behaviors_path)
    
    # Sample for faster processing
    news_df_sample = news_df.sample(min(sample_size, len(news_df)), random_state=42)
    behaviors_df_sample = behaviors_df.sample(min(sample_size, len(behaviors_df)), random_state=42)
    
    return news_df, behaviors_df, news_df_sample, behaviors_df_sample

@st.cache_data
def load_embeddings(split="MINDlarge_train"):
    """Load entity embeddings."""
    data_dir = os.path.join(REPO_ROOT, 'MINDLarge')
    entity_embeddings_path = os.path.join(data_dir, split, 'entity_embedding.vec')
    
    try:
        entity_embeddings = load_entity_embeddings(entity_embeddings_path)
        return entity_embeddings
    except Exception as e:
        st.error(f"Error loading entity embeddings: {e}")
        return None

@st.cache_data
def process_entities(news_df_sample):
    """Process entities into long format."""
    title_entities_long = process_entities_to_long_format(news_df_sample, 'Title_Entities')
    abstract_entities_long = process_entities_to_long_format(news_df_sample, 'Abstract_Entities')
    all_entities = pd.concat([title_entities_long, abstract_entities_long])
    return title_entities_long, abstract_entities_long, all_entities

@st.cache_data
def process_behaviors(behaviors_df_sample):
    """Process user behaviors."""
    return extract_clicks_and_impressions(behaviors_df_sample)

# Create sidebar
st.sidebar.image("https://www.microsoft.com/en-us/research/uploads/prod/2020/05/MIND_overview.png", width=300)
st.sidebar.title("MIND Dataset Explorer")

# Dataset selection
dataset_split = st.sidebar.selectbox(
    "Select Dataset Split",
    ["MINDlarge_train", "MINDlarge_dev", "MINDlarge_test"],
    index=0
)

# Sample size selection
sample_size = st.sidebar.slider(
    "Sample Size",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000
)

# Analysis type selection
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Overview", "Entity Analysis", "News Content Analysis", "User Behavior Analysis", "Entity Embeddings", "Silicon Metrics Dashboard"]
)

# Load data
with st.spinner("Loading and processing data..."):
    news_df, behaviors_df, news_df_sample, behaviors_df_sample = load_dataset(dataset_split, sample_size)
    title_entities_long, abstract_entities_long, all_entities = process_entities(news_df_sample)
    processed_behaviors = process_behaviors(behaviors_df_sample)

# Main content
if analysis_type == "Overview":
    st.markdown('<h1 class="main-header">MIND Dataset Explorer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="info-text">
    This app provides interactive visualizations and insights from the Microsoft News Dataset (MIND).
    MIND is a large-scale English news recommendation dataset with user click behaviors, news articles, 
    and entity information.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("""
    This interactive application allows you to:
    - Explore entities mentioned in news articles
    - Analyze news content and categories
    - Understand user behavior patterns
    - Visualize entity embeddings
    
    Use the sidebar to navigate between different analysis types.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Dataset Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("News Articles", f"{len(news_df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Users", f"{behaviors_df['User_ID'].nunique():,}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Categories", f"{news_df['Category'].nunique():,}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Subcategories", f"{news_df['Subcategory'].nunique():,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Sample Data</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-header">News Articles</h3>', unsafe_allow_html=True)
    st.dataframe(news_df_sample.head(5))
    
    st.markdown('<h3 class="subsection-header">User Behaviors</h3>', unsafe_allow_html=True)
    st.dataframe(behaviors_df_sample.head(5))
    
    # Quick insights
    st.markdown('<h2 class="section-header">Quick Insights</h2>', unsafe_allow_html=True)
    
    # Top categories
    category_counts = analyze_news_categories(news_df_sample)
    
    fig = px.bar(
        category_counts.head(10),
        x='Category',
        y='Count',
        color='Count',
        color_continuous_scale='viridis',
        labels={'Count': 'Number of Articles'},
        title='Top News Categories'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # User engagement
    engagement_stats = analyze_user_engagement(processed_behaviors)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="subsection-header">User Engagement Levels</h3>', unsafe_allow_html=True)
        
        engagement_data = [
            {"level": "None", "count": engagement_stats.get('users_none_engagement', 0)},
            {"level": "Low", "count": engagement_stats.get('users_low_engagement', 0)},
            {"level": "Medium", "count": engagement_stats.get('users_medium_engagement', 0)},
            {"level": "High", "count": engagement_stats.get('users_high_engagement', 0)}
        ]
        
        fig = px.pie(
            engagement_data,
            values='count',
            names='level',
            title='User Engagement Distribution',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="subsection-header">Engagement Statistics</h3>', unsafe_allow_html=True)
        
        metrics = {
            "Average Impressions per User": engagement_stats['average_impressions_per_user'],
            "Average Clicks per User": engagement_stats['average_clicks_per_user'],
            "Average CTR": engagement_stats['average_ctr'],
            "Average History Length": engagement_stats['average_history_length']
        }
        
        for metric, value in metrics.items():
            st.markdown(f"**{metric}:** {value:.2f}")

elif analysis_type == "Entity Analysis":
    st.markdown('<h1 class="main-header">Entity Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="info-text">
    The MIND dataset includes rich entity information extracted from news titles and abstracts. 
    Each entity includes details such as the entity type, WikidataId, confidence score, and more.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Entity Distribution</h2>', unsafe_allow_html=True)
    
    # Entity counts
    entity_counts = analyze_entity_distribution(all_entities, entity_column='Label')
    
    top_entities = entity_counts.head(20)
    fig = px.bar(
        top_entities,
        x='Count',
        y='Label',
        orientation='h',
        color='Count',
        color_continuous_scale='viridis',
        labels={'Count': 'Frequency', 'Label': 'Entity'},
        title='Top 20 Most Common Entities'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Entity types
    st.markdown('<h2 class="section-header">Entity Types</h2>', unsafe_allow_html=True)
    
    type_counts = analyze_entity_types(all_entities)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            type_counts,
            x='Type',
            y='Count',
            color='Count',
            color_continuous_scale='viridis',
            hover_data=['Type_Full', 'Percentage'],
            labels={'Count': 'Frequency', 'Type': 'Entity Type'},
            title='Entity Type Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            type_counts,
            values='Count',
            names='Type',
            hover_data=['Type_Full', 'Percentage'],
            title='Entity Type Proportions',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Type legend
    st.markdown('<h3 class="subsection-header">Entity Type Legend</h3>', unsafe_allow_html=True)
    
    type_legend = [
        {"Code": "P", "Meaning": "Person (e.g., 'Barack Obama')"},
        {"Code": "O", "Meaning": "Organization (e.g., 'Microsoft', 'NASA')"},
        {"Code": "L", "Meaning": "Location (e.g., 'New York', 'Paris')"},
        {"Code": "G", "Meaning": "Geo-political entity (e.g., 'United States', 'European Union')"},
        {"Code": "C", "Meaning": "Concept (e.g., 'Artificial Intelligence', 'Exercise')"},
        {"Code": "M", "Meaning": "Medical (e.g., 'COVID-19', 'Diabetes')"},
        {"Code": "F", "Meaning": "Facility (e.g., 'Madison Square Garden', 'Stanford University')"},
        {"Code": "N", "Meaning": "Natural features (e.g., 'Mount Everest', 'Amazon River')"},
        {"Code": "U", "Meaning": "Unknown / Uncategorized entity"},
        {"Code": "S", "Meaning": "Event (e.g., 'World Cup 2022', 'Olympics')"},
        {"Code": "W", "Meaning": "Work of art (e.g., 'The Mona Lisa', 'Inception')"},
        {"Code": "B", "Meaning": "Brand (e.g., 'Apple', 'Nike')"},
        {"Code": "J", "Meaning": "Journal (e.g., 'Nature', 'The New England Journal of Medicine')"}
    ]
    
    st.table(pd.DataFrame(type_legend))
    
    # Entity confidence
    st.markdown('<h2 class="section-header">Entity Recognition Confidence</h2>', unsafe_allow_html=True)
    
    confidence_stats = analyze_entity_confidence(all_entities)
    
    # Convert bin counts to DataFrame for visualization
    bin_counts = pd.DataFrame({
        'Bin': list(confidence_stats['bin_counts'].keys()),
        'Count': list(confidence_stats['bin_counts'].values())
    })
    
    fig = px.bar(
        bin_counts,
        x='Bin',
        y='Count',
        color='Count',
        color_continuous_scale='viridis',
        labels={'Count': 'Number of Entities', 'Bin': 'Confidence Range'},
        title='Entity Recognition Confidence Score Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Mean Confidence", f"{confidence_stats['mean']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Median Confidence", f"{confidence_stats['median']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Min Confidence", f"{confidence_stats['min']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Max Confidence", f"{confidence_stats['max']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Entity co-occurrence
    st.markdown('<h2 class="section-header">Entity Co-occurrence</h2>', unsafe_allow_html=True)
    
    try:
        # Parse entities for co-occurrence analysis
        news_sample_parsed = news_df_sample.copy()
        news_sample_parsed['Title_Entities'] = news_sample_parsed['Title_Entities'].apply(parse_entities)
        news_sample_parsed['Abstract_Entities'] = news_sample_parsed['Abstract_Entities'].apply(parse_entities)
        
        co_occurrence_df = analyze_entity_co_occurrence(news_sample_parsed)
        
        st.markdown('<h3 class="subsection-header">Top Co-occurring Entity Pairs</h3>', unsafe_allow_html=True)
        st.dataframe(co_occurrence_df.head(10))
        
        top_co_occurrence = co_occurrence_df.head(20)
        fig = px.bar(
            top_co_occurrence,
            x='Count',
            y=[f"{row.Entity1} + {row.Entity2}" for _, row in top_co_occurrence.iterrows()],
            orientation='h',
            color='Count',
            color_continuous_scale='viridis',
            labels={'Count': 'Co-occurrence Frequency', 'y': 'Entity Pair'},
            title='Top 20 Co-occurring Entity Pairs'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error analyzing entity co-occurrence: {e}")
    
    # Entity sources comparison
    st.markdown('<h2 class="section-header">Entity Sources Comparison</h2>', unsafe_allow_html=True)
    
    title_entity_counts = analyze_entity_distribution(title_entities_long, entity_column='Label')
    abstract_entity_counts = analyze_entity_distribution(abstract_entities_long, entity_column='Label')
    
    # Get top entities from both sources
    top_title_entities = set(title_entity_counts.head(20)['Label'])
    top_abstract_entities = set(abstract_entity_counts.head(20)['Label'])
    common_entities = top_title_entities.intersection(top_abstract_entities)
    
    # Create comparison data
    comparison_data = []
    for entity in common_entities:
        title_count = title_entity_counts[title_entity_counts['Label'] == entity]['Count'].values[0]
        title_pct = title_entity_counts[title_entity_counts['Label'] == entity]['Percentage'].values[0]
        
        abstract_count = abstract_entity_counts[abstract_entity_counts['Label'] == entity]['Count'].values[0]
        abstract_pct = abstract_entity_counts[abstract_entity_counts['Label'] == entity]['Percentage'].values[0]
        
        comparison_data.append({
            'Entity': entity,
            'Title_Count': title_count,
            'Title_Percentage': title_pct,
            'Abstract_Count': abstract_count,
            'Abstract_Percentage': abstract_pct
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if len(comparison_df) > 0:
        # Plot comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=comparison_df['Entity'],
            y=comparison_df['Title_Percentage'],
            name='Title',
            marker_color='royalblue'
        ))
        
        fig.add_trace(go.Bar(
            x=comparison_df['Entity'],
            y=comparison_df['Abstract_Percentage'],
            name='Abstract',
            marker_color='firebrick'
        ))
        
        fig.update_layout(
            title='Entity Presence Comparison: Title vs Abstract',
            xaxis_title='Entity',
            yaxis_title='Percentage (%)',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No common entities found in the top 20 of both titles and abstracts.")

elif analysis_type == "News Content Analysis":
    st.markdown('<h1 class="main-header">News Content Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="info-text">
    This section provides insights into the news content, including categories, text properties,
    and sentiment analysis.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">News Categories</h2>', unsafe_allow_html=True)
    
    # News categories
    category_counts = analyze_news_categories(news_df_sample)
    
    fig = px.bar(
        category_counts,
        x='Category',
        y='Count',
        color='Count',
        color_continuous_scale='viridis',
        labels={'Count': 'Number of Articles'},
        title='News Category Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # News subcategories
    st.markdown('<h2 class="section-header">News Subcategories</h2>', unsafe_allow_html=True)
    
    subcategory_counts = analyze_news_subcategories(news_df_sample, top_n=20)
    
    fig = px.bar(
        subcategory_counts,
        x='Subcategory',
        y='Count',
        color='Count',
        color_continuous_scale='viridis',
        labels={'Count': 'Number of Articles'},
        title='Top 20 News Subcategories'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Category-Subcategory relationship
    st.markdown('<h2 class="section-header">Category-Subcategory Relationship</h2>', unsafe_allow_html=True)
    
    grouped_df = analyze_category_subcategory_relationship(news_df_sample)
    
    # Get top 5 subcategories for each category
    top_subcategories = {}
    for category in grouped_df['Category'].unique():
        category_data = grouped_df[grouped_df['Category'] == category]
        top_subcategories[category] = category_data.nlargest(5, 'Count')
    
    # Combine all top subcategories
    top_data = pd.concat(top_subcategories.values())
    
    fig = px.sunburst(
        top_data,
        path=['Category', 'Subcategory'],
        values='Count',
        color='Count',
        color_continuous_scale='viridis',
        title='Category-Subcategory Hierarchy (Top 5 per Category)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Text length analysis
    st.markdown('<h2 class="section-header">Text Length Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="subsection-header">Title Length</h3>', unsafe_allow_html=True)
        
        title_stats = analyze_title_length(news_df_sample)
        
        title_length_df = news_df_sample.copy()
        title_length_df['Title_Word_Count'] = title_length_df['Title'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        fig = px.histogram(
            title_length_df,
            x='Title_Word_Count',
            nbins=30,
            color_discrete_sequence=['royalblue'],
            labels={'Title_Word_Count': 'Word Count'},
            title='Distribution of Title Lengths (Words)'
        )
        
        fig.add_vline(x=title_stats['word_mean'], line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {title_stats['word_mean']:.1f}", annotation_position="top right")
        
        fig.add_vline(x=title_stats['word_median'], line_dash="dot", line_color="green", 
                     annotation_text=f"Median: {title_stats['word_median']:.1f}", annotation_position="top left")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Title length statistics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"""
        **Title Length Statistics (Words)**
        
        Mean: {title_stats['word_mean']:.2f}
        
        Median: {title_stats['word_median']:.2f}
        
        Min: {title_stats['word_min']:.0f}
        
        Max: {title_stats['word_max']:.0f}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 class="subsection-header">Abstract Length</h3>', unsafe_allow_html=True)
        
        abstract_stats = analyze_abstract_length(news_df_sample)
        
        abstract_length_df = news_df_sample.copy()
        abstract_length_df['Abstract_Word_Count'] = abstract_length_df['Abstract'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        fig = px.histogram(
            abstract_length_df,
            x='Abstract_Word_Count',
            nbins=30,
            color_discrete_sequence=['firebrick'],
            labels={'Abstract_Word_Count': 'Word Count'},
            title='Distribution of Abstract Lengths (Words)'
        )
        
        fig.add_vline(x=abstract_stats['word_mean'], line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {abstract_stats['word_mean']:.1f}", annotation_position="top right")
        
        fig.add_vline(x=abstract_stats['word_median'], line_dash="dot", line_color="green", 
                     annotation_text=f"Median: {abstract_stats['word_median']:.1f}", annotation_position="top left")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Abstract length statistics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"""
        **Abstract Length Statistics (Words)**
        
        Mean: {abstract_stats['word_mean']:.2f}
        
        Median: {abstract_stats['word_median']:.2f}
        
        Min: {abstract_stats['word_min']:.0f}
        
        Max: {abstract_stats['word_max']:.0f}
        
        Missing: {abstract_stats['missing_count']} ({abstract_stats['missing_percentage']:.2f}%)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Word frequency analysis
    st.markdown('<h2 class="section-header">Word Frequency Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="subsection-header">Top Words in Titles</h3>', unsafe_allow_html=True)
        
        # Analyze top words in titles
        title_word_counts = analyze_top_words(news_df_sample, column='Title', top_n=20, remove_stop=True, lemmatize=True)
        
        fig = px.bar(
            title_word_counts,
            x='Count',
            y='Word',
            orientation='h',
            color='Count',
            color_continuous_scale='blues',
            title='Top 20 Words in News Titles'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="subsection-header">Top Words in Abstracts</h3>', unsafe_allow_html=True)
        
        # Analyze top words in abstracts
        abstract_word_counts = analyze_top_words(news_df_sample, column='Abstract', top_n=20, remove_stop=True, lemmatize=True)
        
        fig = px.bar(
            abstract_word_counts,
            x='Count',
            y='Word',
            orientation='h',
            color='Count',
            color_continuous_scale='reds',
            title='Top 20 Words in News Abstracts'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Entity presence in news
    st.markdown('<h2 class="section-header">Entity Presence in News</h2>', unsafe_allow_html=True)
    
    entity_presence_stats = analyze_news_with_entities(news_df_sample)
    
    # Create data for entity presence
    entity_presence_data = [
        {"Source": "Title", "Has_Entities": entity_presence_stats['articles_with_title_entities'], 
         "No_Entities": len(news_df_sample) - entity_presence_stats['articles_with_title_entities']},
        {"Source": "Abstract", "Has_Entities": entity_presence_stats['articles_with_abstract_entities'], 
         "No_Entities": len(news_df_sample) - entity_presence_stats['articles_with_abstract_entities']},
        {"Source": "Any", "Has_Entities": entity_presence_stats['articles_with_any_entities'], 
         "No_Entities": len(news_df_sample) - entity_presence_stats['articles_with_any_entities']}
    ]
    
    entity_presence_df = pd.DataFrame(entity_presence_data)
    entity_presence_df = pd.melt(entity_presence_df, id_vars=['Source'], value_vars=['Has_Entities', 'No_Entities'],
                               var_name='Entity_Status', value_name='Count')
    
    fig = px.bar(
        entity_presence_df,
        x='Source',
        y='Count',
        color='Entity_Status',
        color_discrete_map={'Has_Entities': 'forestgreen', 'No_Entities': 'lightgray'},
        barmode='stack',
        title='Entity Presence in News Articles'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Entity count statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg. Entities per Title", f"{entity_presence_stats['title_entity_mean']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg. Entities per Abstract", f"{entity_presence_stats['abstract_entity_mean']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg. Entities per Article", f"{entity_presence_stats['total_entity_mean']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sentiment analysis
    st.markdown('<h2 class="section-header">Sentiment Analysis</h2>', unsafe_allow_html=True)
    
    try:
        # Perform sentiment analysis
        sentiment_df = perform_advanced_sentiment_analysis(news_df_sample, column='Title', method='vader')
        
        # Count sentiment categories
        sentiment_counts = sentiment_df['Title_Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        sentiment_counts['Percentage'] = (sentiment_counts['Count'] / sentiment_counts['Count'].sum() * 100).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                sentiment_counts,
                x='Sentiment',
                y='Count',
                color='Sentiment',
                color_discrete_map={'Positive': 'forestgreen', 'Neutral': 'gold', 'Negative': 'crimson'},
                text='Percentage',
                title='Title Sentiment Distribution'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                sentiment_counts,
                values='Count',
                names='Sentiment',
                color='Sentiment',
                color_discrete_map={'Positive': 'forestgreen', 'Neutral': 'gold', 'Negative': 'crimson'},
                title='Title Sentiment Proportions'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment by category
        st.markdown('<h3 class="subsection-header">Sentiment by Category</h3>', unsafe_allow_html=True)
        
        # Create a crosstab of Category vs Sentiment
        sentiment_by_category = pd.crosstab(
            sentiment_df['Category'], 
            sentiment_df['Title_Sentiment'],
            normalize='index'
        ) * 100
        
        sentiment_by_category_df = sentiment_by_category.reset_index()
        sentiment_by_category_df = pd.melt(sentiment_by_category_df, id_vars=['Category'], 
                                         value_vars=sentiment_by_category.columns.tolist(),
                                         var_name='Sentiment', value_name='Percentage')
        
        fig = px.bar(
            sentiment_by_category_df,
            x='Category',
            y='Percentage',
            color='Sentiment',
            color_discrete_map={'Positive': 'forestgreen', 'Neutral': 'gold', 'Negative': 'crimson'},
            barmode='stack',
            title='Sentiment Distribution by Category'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error performing sentiment analysis: {e}")
        st.info("Try installing NLTK and downloading the vader_lexicon: `pip install nltk` and then `import nltk; nltk.download('vader_lexicon')`")

elif analysis_type == "User Behavior Analysis":
    st.markdown('<h1 class="main-header">User Behavior Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="info-text">
    This section analyzes user behaviors, including engagement patterns, temporal patterns,
    click-through rates, and content preferences.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">User Engagement</h2>', unsafe_allow_html=True)
    
    # Analyze user engagement
    engagement_stats = analyze_user_engagement(processed_behaviors)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create data for engagement levels
        engagement_data = [
            {"Level": "None", "Count": engagement_stats.get('users_none_engagement', 0)},
            {"Level": "Low", "Count": engagement_stats.get('users_low_engagement', 0)},
            {"Level": "Medium", "Count": engagement_stats.get('users_medium_engagement', 0)},
            {"Level": "High", "Count": engagement_stats.get('users_high_engagement', 0)}
        ]
        
        engagement_df = pd.DataFrame(engagement_data)
        
        fig = px.pie(
            engagement_df,
            values='Count',
            names='Level',
            title='User Engagement Levels',
            color='Level',
            color_discrete_map={
                'None': 'lightgray',
                'Low': 'lightblue',
                'Medium': 'royalblue',
                'High': 'darkblue'
            },
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Engagement statistics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        **Engagement Metrics**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg. Impressions", f"{engagement_stats['average_impressions_per_user']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg. History Length", f"{engagement_stats['average_history_length']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg. Clicks", f"{engagement_stats['average_clicks_per_user']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg. CTR", f"{engagement_stats['average_ctr']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Temporal patterns
    st.markdown('<h2 class="section-header">Temporal Patterns</h2>', unsafe_allow_html=True)
    
    # Analyze temporal patterns
    temporal_stats = analyze_temporal_patterns(processed_behaviors)
    
    tab1, tab2, tab3 = st.tabs(["Hourly Activity", "Weekday Activity", "Time of Day"])
    
    with tab1:
        # Hourly activity
        hourly_stats = pd.DataFrame({
            'Hour': range(24),
            'Sessions': [temporal_stats['hourly_stats']['Impression_ID'].get(hour, 0) for hour in range(24)]
        })
        
        fig = px.line(
            hourly_stats,
            x='Hour',
            y='Sessions',
            markers=True,
            line_shape='spline',
            title='User Activity by Hour of Day',
            color_discrete_sequence=['royalblue']
        )
        
        # Highlight peak hour
        peak_hour = temporal_stats['peak_hour']
        peak_count = hourly_stats.loc[hourly_stats['Hour'] == peak_hour, 'Sessions'].values[0]
        
        fig.add_annotation(
            x=peak_hour,
            y=peak_count,
            text=f"Peak: {peak_hour}:00",
            showarrow=True,
            arrowhead=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Weekday activity
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_stats = pd.DataFrame({
            'Weekday': weekday_order,
            'Sessions': [temporal_stats['weekday_stats']['Impression_ID'].get(day, 0) for day in weekday_order]
        })
        
        fig = px.bar(
            weekday_stats,
            x='Weekday',
            y='Sessions',
            title='User Activity by Day of Week',
            color='Sessions',
            color_continuous_scale='viridis'
        )
        
        # Highlight peak day
        peak_day = temporal_stats['peak_weekday']
        peak_idx = weekday_order.index(peak_day)
        peak_count = weekday_stats.loc[weekday_stats['Weekday'] == peak_day, 'Sessions'].values[0]
        
        fig.add_annotation(
            x=peak_day,
            y=peak_count,
            text=f"Peak: {peak_day}",
            showarrow=True,
            arrowhead=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Time of day activity
        tod_order = ['Morning', 'Afternoon', 'Evening', 'Night']
        tod_stats = pd.DataFrame({
            'TimeOfDay': tod_order,
            'Sessions': [temporal_stats['time_of_day_stats']['Impression_ID'].get(tod, 0) for tod in tod_order]
        })
        
        fig = px.bar(
            tod_stats,
            x='TimeOfDay',
            y='Sessions',
            title='User Activity by Time of Day',
            color='Sessions',
            color_continuous_scale='viridis'
        )
        
        # Highlight peak time of day
        peak_tod = temporal_stats['peak_time_of_day']
        peak_count = tod_stats.loc[tod_stats['TimeOfDay'] == peak_tod, 'Sessions'].values[0]
        
        fig.add_annotation(
            x=peak_tod,
            y=peak_count,
            text=f"Peak: {peak_tod}",
            showarrow=True,
            arrowhead=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Click-through rate
    st.markdown('<h2 class="section-header">Click-Through Rate (CTR) Analysis</h2>', unsafe_allow_html=True)
    
    # Analyze CTR
    ctr_stats = analyze_click_through_rate(processed_behaviors)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CTR distribution
        ctr_bins_str = list(ctr_stats['ctr_distribution'].keys())
        ctr_counts = list(ctr_stats['ctr_distribution'].values())
        
        ctr_dist_df = pd.DataFrame({
            'CTR_Range': ctr_bins_str,
            'Count': ctr_counts
        })
        
        fig = px.bar(
            ctr_dist_df,
            x='CTR_Range',
            y='Count',
            title='Distribution of User CTR',
            color='Count',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CTR statistics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        **CTR Statistics**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Overall CTR", f"{ctr_stats['overall_ctr']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Min User CTR", f"{ctr_stats['min_user_ctr']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg. User CTR", f"{ctr_stats['average_user_ctr']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Max User CTR", f"{ctr_stats['max_user_ctr']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f"**Total Clicks:** {ctr_stats['total_clicks']:,}")
        st.markdown(f"**Total Impressions:** {ctr_stats['total_impressions']:,}")
    
    # History patterns
    st.markdown('<h2 class="section-header">User History Patterns</h2>', unsafe_allow_html=True)
    
    # Analyze history patterns
    history_stats = analyze_history_patterns(processed_behaviors)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # History length distribution
        history_bins_str = list(history_stats['history_length_distribution'].keys())
        history_counts = list(history_stats['history_length_distribution'].values())
        
        history_dist_df = pd.DataFrame({
            'History_Length': history_bins_str,
            'Count': history_counts
        })
        
        fig = px.bar(
            history_dist_df,
            x='History_Length',
            y='Count',
            title='Distribution of User History Lengths',
            color='Count',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # History statistics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        **History Statistics**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg. History Length", f"{history_stats['average_history_length']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Min History Length", f"{history_stats['min_history_length']:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Median History Length", f"{history_stats['median_history_length']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Max History Length", f"{history_stats['max_history_length']:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Users with no history
        st.markdown(f"**Users with No History:** {history_stats['users_with_no_history']} ({history_stats['users_with_no_history_pct']:.2f}%)")
        
        # Top history items
        if 'top_history_items' in history_stats:
            st.markdown("**Top News Items in User History:**")
            top_history_df = pd.DataFrame(history_stats['top_history_items'])
            st.dataframe(top_history_df)
    
    # User category preferences
    st.markdown('<h2 class="section-header">User Category Preferences</h2>', unsafe_allow_html=True)
    
    try:
        # Analyze user category preferences
        category_stats = analyze_user_categories(news_df_sample, processed_behaviors)
        
        # Create DataFrame for visualization
        categories = list(set(list(category_stats['history_category_counts'].keys()) + 
                             list(category_stats['clicked_category_counts'].keys())))
        
        category_pref_data = []
        for category in categories:
            history_count = category_stats['history_category_counts'].get(category, 0)
            history_pct = category_stats['history_category_pct'].get(category, 0)
            
            clicked_count = category_stats['clicked_category_counts'].get(category, 0)
            clicked_pct = category_stats['clicked_category_pct'].get(category, 0)
            
            preference_score = category_stats['category_preference_score'].get(category, 0)
            
            category_pref_data.append({
                'Category': category,
                'History_Count': history_count,
                'History_Percentage': history_pct,
                'Clicked_Count': clicked_count,
                'Clicked_Percentage': clicked_pct,
                'Preference_Score': preference_score
            })
        
        category_pref_df = pd.DataFrame(category_pref_data)
        category_pref_df = category_pref_df.sort_values('History_Count', ascending=False)
        
        # Display top categories
        top_categories = category_pref_df.head(10).copy()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_categories['Category'],
            y=top_categories['History_Percentage'],
            name='In History',
            marker_color='royalblue'
        ))
        
        fig.add_trace(go.Bar(
            x=top_categories['Category'],
            y=top_categories['Clicked_Percentage'],
            name='Clicked',
            marker_color='firebrick'
        ))
        
        fig.update_layout(
            title='Top 10 Categories: History vs. Clicks',
            xaxis_title='Category',
            yaxis_title='Percentage (%)',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot preference scores
        top_pref = category_pref_df.sort_values('Preference_Score', ascending=False).head(10).copy()
        
        fig = px.bar(
            top_pref,
            x='Category',
            y='Preference_Score',
            title='Top 10 Categories by Preference Score (Clicks ÷ History)',
            color='Preference_Score',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error analyzing user category preferences: {e}")
    
    # User segmentation
    st.markdown('<h2 class="section-header">User Segmentation</h2>', unsafe_allow_html=True)
    
    try:
        # Segment users
        user_segments = analyze_user_segments(processed_behaviors)
        
        # Plot segment distributions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Engagement levels
            engagement_counts = user_segments['Engagement_Level'].value_counts().reset_index()
            engagement_counts.columns = ['Level', 'Count']
            
            fig = px.pie(
                engagement_counts,
                values='Count',
                names='Level',
                title='Engagement Level Distribution',
                color='Level',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Activity levels
            activity_counts = user_segments['Activity_Level'].value_counts().reset_index()
            activity_counts.columns = ['Level', 'Count']
            
            fig = px.pie(
                activity_counts,
                values='Count',
                names='Level',
                title='Activity Level Distribution',
                color='Level',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # CTR levels
            ctr_counts = user_segments['CTR_Level'].value_counts().reset_index()
            ctr_counts.columns = ['Level', 'Count']
            
            fig = px.pie(
                ctr_counts,
                values='Count',
                names='Level',
                title='CTR Level Distribution',
                color='Level',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top segments
        segment_counts = user_segments['Segment'].value_counts().reset_index().head(10)
        segment_counts.columns = ['Segment', 'Count']
        segment_counts['Percentage'] = (segment_counts['Count'] / len(user_segments) * 100).round(2)
        
        fig = px.bar(
            segment_counts,
            x='Count',
            y='Segment',
            orientation='h',
            color='Count',
            color_continuous_scale='viridis',
            title='Top 10 User Segments',
            text='Percentage'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error segmenting users: {e}")

elif analysis_type == "Entity Embeddings":
    st.markdown('<h1 class="main-header">Entity Embeddings Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="info-text">
    This section provides analysis and visualization of entity embeddings. Entity embeddings are
    vector representations of entities mentioned in news articles, capturing semantic relationships.
    </p>
    """, unsafe_allow_html=True)
    
    # Load entity embeddings
    entity_embeddings = load_embeddings(dataset_split)
    
    if entity_embeddings is None:
        st.error("Failed to load entity embeddings. Please check if the embedding files exist.")
    else:
        st.markdown('<h2 class="section-header">Embedding Overview</h2>', unsafe_allow_html=True)
        
        # Display embedding statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Number of Entities", f"{len(entity_embeddings):,}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Embedding Dimension", f"{entity_embeddings.shape[1] - 1}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            # Extract WikidataIds from entities
            news_with_wikidata = extract_entity_wikidata_ids(news_df_sample)
            
            # Get unique WikidataIds
            all_wikidata_ids = [
                wid for sublist in news_with_wikidata['All_WikidataIds'] 
                for wid in sublist if wid is not None
            ]
            unique_wikidata_ids = list(set(all_wikidata_ids))
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Unique WikidataIds in Sample", f"{len(unique_wikidata_ids):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Sample of embeddings
        st.markdown('<h2 class="section-header">Sample Embeddings</h2>', unsafe_allow_html=True)
        
        # Display a few rows of entity embeddings
        st.dataframe(entity_embeddings.head(5))
        
        # Dimensionality reduction for visualization
        st.markdown('<h2 class="section-header">Embedding Visualization</h2>', unsafe_allow_html=True)
        
        # Options for visualization
        vis_method = st.radio(
            "Select Visualization Method",
            ["PCA", "t-SNE"],
            horizontal=True
        )
        
        # Number of entities to visualize
        max_vis_entities = min(1000, len(entity_embeddings))
        num_entities = st.slider(
            "Number of Entities to Visualize",
            min_value=100,
            max_value=max_vis_entities,
            value=500,
            step=100
        )
        
        # Sample entities for visualization
        sampled_embeddings = entity_embeddings.sample(num_entities, random_state=42)
        
        # Merge with entity metadata if available
        if len(all_entities) > 0:
            # Extract unique WikidataIds from sampled embeddings
            sampled_wikidata_ids = sampled_embeddings['WikidataId'].tolist()
            
            # Extract entity metadata
            entity_metadata = all_entities[all_entities['WikidataId'].isin(sampled_wikidata_ids)][
                ['WikidataId', 'Label', 'Type']
            ].drop_duplicates()
            
            # Merge with embeddings
            vis_df = sampled_embeddings.merge(entity_metadata, on='WikidataId', how='left')
            
            # Fill NA values
            vis_df['Label'] = vis_df['Label'].fillna('Unknown')
            vis_df['Type'] = vis_df['Type'].fillna('U')
            
            with st.spinner("Performing dimensionality reduction..."):
                if vis_method == "PCA":
                    # Extract embedding columns
                    embedding_cols = [col for col in vis_df.columns if col not in ['WikidataId', 'Label', 'Type']]
                    
                    # Use scikit-learn for PCA
                    from sklearn.decomposition import PCA
                    
                    pca = PCA(n_components=2)
                    embedding_values = vis_df[embedding_cols].values
                    reduced_embeddings = pca.fit_transform(embedding_values)
                    
                    # Create DataFrame for visualization
                    vis_data = pd.DataFrame({
                        'PC1': reduced_embeddings[:, 0],
                        'PC2': reduced_embeddings[:, 1],
                        'WikidataId': vis_df['WikidataId'],
                        'Label': vis_df['Label'],
                        'Type': vis_df['Type']
                    })
                    
                    # Create scatter plot
                    fig = px.scatter(
                        vis_data,
                        x='PC1',
                        y='PC2',
                        color='Type',
                        hover_data=['Label', 'WikidataId'],
                        title=f'PCA Visualization of {num_entities} Entity Embeddings',
                        color_discrete_sequence=px.colors.qualitative.Vivid
                    )
                    
                    # Add variance explained
                    variance_explained = pca.explained_variance_ratio_
                    fig.update_xaxes(title=f"PC1 ({variance_explained[0]:.2%} variance)")
                    fig.update_yaxes(title=f"PC2 ({variance_explained[1]:.2%} variance)")
                    
                elif vis_method == "t-SNE":
                    # Extract embedding columns
                    embedding_cols = [col for col in vis_df.columns if col not in ['WikidataId', 'Label', 'Type']]
                    
                    # Use scikit-learn for t-SNE
                    from sklearn.manifold import TSNE
                    
                    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                    embedding_values = vis_df[embedding_cols].values
                    reduced_embeddings = tsne.fit_transform(embedding_values)
                    
                    # Create DataFrame for visualization
                    vis_data = pd.DataFrame({
                        'TSNE1': reduced_embeddings[:, 0],
                        'TSNE2': reduced_embeddings[:, 1],
                        'WikidataId': vis_df['WikidataId'],
                        'Label': vis_df['Label'],
                        'Type': vis_df['Type']
                    })
                    
                    # Create scatter plot
                    fig = px.scatter(
                        vis_data,
                        x='TSNE1',
                        y='TSNE2',
                        color='Type',
                        hover_data=['Label', 'WikidataId'],
                        title=f't-SNE Visualization of {num_entities} Entity Embeddings',
                        color_discrete_sequence=px.colors.qualitative.Vivid
                    )
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Type legend
                st.markdown('<h3 class="subsection-header">Entity Type Legend</h3>', unsafe_allow_html=True)
                
                type_legend = [
                    {"Code": "P", "Meaning": "Person (e.g., 'Barack Obama')"},
                    {"Code": "O", "Meaning": "Organization (e.g., 'Microsoft', 'NASA')"},
                    {"Code": "L", "Meaning": "Location (e.g., 'New York', 'Paris')"},
                    {"Code": "G", "Meaning": "Geo-political entity (e.g., 'United States', 'European Union')"},
                    {"Code": "C", "Meaning": "Concept (e.g., 'Artificial Intelligence', 'Exercise')"},
                    {"Code": "M", "Meaning": "Medical (e.g., 'COVID-19', 'Diabetes')"},
                    {"Code": "F", "Meaning": "Facility (e.g., 'Madison Square Garden', 'Stanford University')"},
                    {"Code": "N", "Meaning": "Natural features (e.g., 'Mount Everest', 'Amazon River')"},
                    {"Code": "U", "Meaning": "Unknown / Uncategorized entity"},
                    {"Code": "S", "Meaning": "Event (e.g., 'World Cup 2022', 'Olympics')"},
                    {"Code": "W", "Meaning": "Work of art (e.g., 'The Mona Lisa', 'Inception')"},
                    {"Code": "B", "Meaning": "Brand (e.g., 'Apple', 'Nike')"},
                    {"Code": "J", "Meaning": "Journal (e.g., 'Nature', 'The New England Journal of Medicine')"}
                ]
                
                st.table(pd.DataFrame(type_legend))
        else:
            st.warning("Entity metadata not available. Visualization will not include entity types and labels.")
            
            # Perform dimensionality reduction on raw embeddings
            with st.spinner("Performing dimensionality reduction..."):
                # Extract embedding columns
                embedding_cols = [col for col in sampled_embeddings.columns if col != 'WikidataId']
                
                if vis_method == "PCA":
                    from sklearn.decomposition import PCA
                    
                    pca = PCA(n_components=2)
                    embedding_values = sampled_embeddings[embedding_cols].values
                    reduced_embeddings = pca.fit_transform(embedding_values)
                    
                    # Create DataFrame for visualization
                    vis_data = pd.DataFrame({
                        'PC1': reduced_embeddings[:, 0],
                        'PC2': reduced_embeddings[:, 1],
                        'WikidataId': sampled_embeddings['WikidataId']
                    })
                    
                    # Create scatter plot
                    fig = px.scatter(
                        vis_data,
                        x='PC1',
                        y='PC2',
                        hover_data=['WikidataId'],
                        title=f'PCA Visualization of {num_entities} Entity Embeddings'
                    )
                    
                    # Add variance explained
                    variance_explained = pca.explained_variance_ratio_
                    fig.update_xaxes(title=f"PC1 ({variance_explained[0]:.2%} variance)")
                    fig.update_yaxes(title=f"PC2 ({variance_explained[1]:.2%} variance)")
                    
                elif vis_method == "t-SNE":
                    from sklearn.manifold import TSNE
                    
                    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                    embedding_values = sampled_embeddings[embedding_cols].values
                    reduced_embeddings = tsne.fit_transform(embedding_values)
                    
                    # Create DataFrame for visualization
                    vis_data = pd.DataFrame({
                        'TSNE1': reduced_embeddings[:, 0],
                        'TSNE2': reduced_embeddings[:, 1],
                        'WikidataId': sampled_embeddings['WikidataId']
                    })
                    
                    # Create scatter plot
                    fig = px.scatter(
                        vis_data,
                        x='TSNE1',
                        y='TSNE2',
                        hover_data=['WikidataId'],
                        title=f't-SNE Visualization of {num_entities} Entity Embeddings'
                    )
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
        
        # Wikidata integration
        st.markdown('<h2 class="section-header">Wikidata Integration</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <p class="info-text">
        The MIND dataset includes WikidataId for entities, which allows integration with Wikidata
        to fetch additional information about entities.
        </p>
        """, unsafe_allow_html=True)
        
        # Sample entities for Wikidata lookup
        st.markdown('<h3 class="subsection-header">Entity Information Lookup</h3>', unsafe_allow_html=True)
        
        # Extract WikidataIds from entities
        news_with_wikidata = extract_entity_wikidata_ids(news_df_sample)
        
        # Get unique WikidataIds
        all_wikidata_ids = [
            wid for sublist in news_with_wikidata['All_WikidataIds'] 
            for wid in sublist if wid is not None
        ]
        unique_wikidata_ids = list(set(all_wikidata_ids))
        
        # Allow user to select entities
        selected_entities = st.multiselect(
            "Select Entities to Look Up",
            options=unique_wikidata_ids,
            default=unique_wikidata_ids[:5] if len(unique_wikidata_ids) >= 5 else unique_wikidata_ids
        )
        
        if selected_entities:
            with st.spinner("Fetching entity information from Wikidata..."):
                try:
                    # Fetch Wikidata info
                    wikidata_info = fetch_wikidata_info(selected_entities)
                    
                    # Convert to DataFrame
                    wikidata_df = pd.DataFrame([
                        {
                            'WikidataId': wid,
                            'Label': info['label'],
                            'Description': info['description'],
                            'Wikipedia_URL': info['wikipedia_url']
                        }
                        for wid, info in wikidata_info.items()
                    ])
                    
                    # Display the results
                    st.dataframe(wikidata_df)
                    
                    # Display details for a specific entity
                    if len(wikidata_info) > 0:
                        st.markdown('<h3 class="subsection-header">Entity Details</h3>', unsafe_allow_html=True)
                        
                        selected_entity = st.selectbox(
                            "Select Entity to View Details",
                            options=list(wikidata_info.keys())
                        )
                        
                        if selected_entity:
                            entity_info = wikidata_info[selected_entity]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**Wikidata ID:** {selected_entity}")
                                st.markdown(f"**Label:** {entity_info['label']}")
                                st.markdown(f"**Description:** {entity_info['description']}")
                                
                                if entity_info['wikipedia_url']:
                                    st.markdown(f"**Wikipedia:** [Link]({entity_info['wikipedia_url']})")
                            
                            with col2:
                                # If the entity is in the embeddings, show its nearest neighbors
                                if selected_entity in entity_embeddings['WikidataId'].values:
                                    st.markdown("**Top 5 Most Similar Entities:**")
                                    
                                    # Extract the embedding for the selected entity
                                    selected_embedding = entity_embeddings[entity_embeddings['WikidataId'] == selected_entity].iloc[0, 1:].values
                                    
                                    # Compute cosine similarity with all other embeddings
                                    from sklearn.metrics.pairwise import cosine_similarity
                                    
                                    # Convert embeddings to numpy array
                                    embedding_cols = [col for col in entity_embeddings.columns if col != 'WikidataId']
                                    embeddings_array = entity_embeddings[embedding_cols].values
                                    
                                    # Reshape the selected embedding for cosine_similarity
                                    selected_embedding = selected_embedding.reshape(1, -1)
                                    
                                    # Compute similarities
                                    similarities = cosine_similarity(selected_embedding, embeddings_array)[0]
                                    
                                    # Get top 6 similar entities (including self)
                                    top_indices = similarities.argsort()[::-1][:6]
                                    
                                    # Display similar entities (excluding self)
                                    similar_entities = []
                                    for idx in top_indices[1:]:  # Skip the first one (self)
                                        wid = entity_embeddings.iloc[idx]['WikidataId']
                                        similarity = similarities[idx]
                                        similar_entities.append({'WikidataId': wid, 'Similarity': similarity})
                                    
                                    st.table(pd.DataFrame(similar_entities))
                                else:
                                    st.info("Entity not found in embeddings.")
                except Exception as e:
                    st.error(f"Error fetching Wikidata info: {e}")
        else:
            st.info("Select entities to look up their information.")

elif analysis_type == "Silicon Metrics Dashboard":
    st.markdown('<h1 class="main-header">Silicon Metrics Dashboard</h1>', unsafe_allow_html=True)
    
    # Use streamlit's component system to embed the silicon dashboard
    from streamlit.components.v1 import html
    
    # Path to silicon dashboard
    silicon_dashboard_path = os.path.join(CURRENT_DIR, 'silicon_metrics_dashboard.py')
    
    # Check if the silicon dashboard exists
    if os.path.exists(silicon_dashboard_path):
        # Provide a button to open the silicon dashboard in a new tab/window
        if st.button("Launch Silicon Metrics Dashboard", type="primary"):
            # Run the silicon dashboard in a subprocess
            import subprocess
            subprocess.Popen(["streamlit", "run", silicon_dashboard_path, "--server.port", "8502"])
            
            # Display the link
            st.success("Silicon Metrics Dashboard launched! Click the link below:")
            st.markdown("[Open Silicon Metrics Dashboard](http://localhost:8502)")
            
        # Display introduction
        st.markdown("""
        <div class="highlight">
        <h3>About Silicon Metrics Dashboard</h3>
        <p>
        The Silicon Metrics Dashboard provides comprehensive visualization and evaluation for our advanced machine learning models.
        </p>
        <p>
        These models analyze news content across four key metrics:
        <ul>
        <li><strong>Political Influence:</strong> Measures political content and bias in news articles</li>
        <li><strong>Rhetoric Intensity:</strong> Quantifies use of rhetorical devices and emotional language</li>
        <li><strong>Information Depth:</strong> Evaluates information density and complexity</li>
        <li><strong>Sentiment:</strong> Analyzes the emotional tone of the news article</li>
        </ul>
        </p>
        <p>
        The dashboard provides detailed model performance metrics, feature importance analysis, and model comparison functionality.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Preview the dashboard features
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Performance Analysis")
            st.markdown("""
            - Regression and classification metrics
            - Prediction vs actual visualization
            - Error distribution analysis
            - Model comparison across metrics
            """)
            
        with col2:
            st.markdown("### Feature Analysis")
            st.markdown("""
            - Feature importance visualization
            - Feature distribution analysis
            - Feature correlation analysis
            - Top predictive features
            """)
            
        # Display WikiData integration
        st.markdown("### WikiData Integration")
        st.markdown("""
        Our models utilize the WikiData knowledge graph to enhance entity understanding:
        - Entity type classification
        - Entity relationship analysis
        - Entity disambiguation
        - Cross-lingual support
        """)
        
    else:
        st.warning(f"Silicon Metrics Dashboard not found at {silicon_dashboard_path}. Please make sure it exists.")

# Add footer
st.markdown("""
---
<p style="text-align: center;">
    <b>Advanced EDA on MIND Dataset</b><br/>
    This app provides interactive exploration of the Microsoft News Dataset (MIND).<br/>
    Created for the advanced EDA project.
</p>
""", unsafe_allow_html=True)