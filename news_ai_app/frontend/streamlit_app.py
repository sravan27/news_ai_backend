"""
Streamlit frontend for News AI application.
"""
import datetime
import json
import time
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="News AI Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL - automatically detect local vs deployed
import os

# Check if API URL is already in session state (set by production_app.py)
if "API_BASE_URL" in st.session_state:
    API_BASE_URL = st.session_state["API_BASE_URL"]
else:
    # Check environment
    is_streamlit_cloud = os.environ.get("IS_STREAMLIT_CLOUD", "false").lower() == "true"
    is_production = os.environ.get("IS_PRODUCTION", "false").lower() == "true"
    
    # Set API URL based on environment
    if is_streamlit_cloud:
        # Use the same domain for API in cloud deployment
        API_BASE_URL = "/api"
    elif is_production:
        # In production Docker environment
        API_BASE_URL = "http://localhost:8000/api"
    else:
        # Use localhost when running locally
        API_BASE_URL = "http://localhost:8000/api"


# Helper Functions
def fetch_news(category: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """Fetch news articles from the API."""
    params = {"limit": limit}
    if category:
        params["category"] = category
        
    try:
        response = httpx.get(f"{API_BASE_URL}/news", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching news: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return []


def fetch_news_by_id(news_id: str) -> Optional[Dict]:
    """Fetch a specific news article by ID."""
    try:
        response = httpx.get(f"{API_BASE_URL}/news/{news_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching news by ID: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None


def analyze_news_content(url: Optional[str] = None, text: Optional[str] = None, title: Optional[str] = None) -> Optional[Dict]:
    """Analyze news content using the API."""
    data = {}
    if url:
        data["url"] = url
    if text:
        data["text"] = text
    if title:
        data["title"] = title
        
    try:
        response = httpx.post(f"{API_BASE_URL}/analyze", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error analyzing content: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None


def get_recommendations(user_id: Optional[str] = None, history_news_ids: List[str] = None) -> List[Dict]:
    """Get news recommendations from the API."""
    if history_news_ids is None:
        history_news_ids = []
        
    data = {
        "user_id": user_id,
        "history_news_ids": history_news_ids,
        "top_k": 10
    }
    
    try:
        response = httpx.post(f"{API_BASE_URL}/recommendations", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error getting recommendations: {response.status_code}")
            return {}
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return {}


def generate_podcast(news_ids: List[str]) -> Optional[Dict]:
    """Generate a podcast from selected news articles."""
    data = {
        "news_ids": news_ids,
        "include_intro": True,
        "include_transitions": True
    }
    
    try:
        response = httpx.post(f"{API_BASE_URL}/podcast/generate", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error generating podcast: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None


# Visualization Functions
def create_metrics_gauge(value: float, title: str, color_scale: List[str]) -> go.Figure:
    """Create a radial gauge visualization for metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": "royalblue"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 0.3], "color": color_scale[0]},
                {"range": [0.3, 0.7], "color": color_scale[1]},
                {"range": [0.7, 1], "color": color_scale[2]}
            ]
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig


def create_sentiment_emoji(sentiment: Dict) -> str:
    """Create sentiment emoji based on sentiment analysis."""
    label = sentiment.get("label", "").lower()
    score = sentiment.get("score", 0.5)
    
    if label == "positive":
        if score > 0.8:
            return "😄"  # Very positive
        else:
            return "🙂"  # Positive
    elif label == "negative":
        if score > 0.8:
            return "😠"  # Very negative
        else:
            return "🙁"  # Negative
    else:
        return "😐"  # Neutral


# UI Components
def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("News AI Dashboard")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["News Feed", "Article Analysis", "Recommendations", "Podcast Generator"]
    )
    
    # Categories filter for News Feed
    if page == "News Feed":
        st.sidebar.subheader("Categories")
        categories = ["All", "news", "sports", "entertainment", "health", "lifestyle", "politics", "technology"]
        selected_category = st.sidebar.selectbox("Select Category", categories)
        if selected_category == "All":
            selected_category = None
    else:
        selected_category = None
        
    # User ID for recommendations
    if page == "Recommendations":
        st.sidebar.subheader("User")
        user_id = st.sidebar.text_input("User ID (optional)")
    else:
        user_id = None
        
    return page, selected_category, user_id


def render_news_feed(category: Optional[str] = None):
    """Render the news feed page."""
    st.title("News Feed")
    
    # Fetch news articles
    news_articles = fetch_news(category=category, limit=20)
    
    if not news_articles:
        st.warning("No news articles found.")
        return
    
    # Display articles in a grid
    cols = st.columns(3)
    for i, article in enumerate(news_articles):
        col = cols[i % 3]
        with col:
            st.subheader(article["title"])
            st.caption(f"Category: {article['category']} / {article.get('subcategory', '')}")
            
            if article.get("abstract"):
                st.write(article["abstract"])
                
            st.button(
                "Analyze",
                key=f"analyze_{article['news_id']}",
                on_click=lambda article_id=article["news_id"]: st.session_state.update({
                    "current_page": "Article Analysis",
                    "article_id": article_id
                })
            )
            
            st.markdown("---")


def render_article_analysis():
    """Render the article analysis page."""
    st.title("Article Analysis")
    
    # Check if coming from news feed
    article_id = st.session_state.get("article_id")
    
    # Input modes
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Analyze by URL", "Analyze by Text", "Analyze from News Feed"],
        index=2 if article_id else 0
    )
    
    article_data = None
    url = None
    text = None
    title = None
    
    if analysis_mode == "Analyze by URL":
        url = st.text_input("Enter news article URL")
        if url and st.button("Analyze URL"):
            with st.spinner("Analyzing article..."):
                article_data = analyze_news_content(url=url)
                
    elif analysis_mode == "Analyze by Text":
        title = st.text_input("Article Title (optional)")
        text = st.text_area("Article Text", height=200)
        if text and st.button("Analyze Text"):
            with st.spinner("Analyzing text..."):
                article_data = analyze_news_content(text=text, title=title)
                
    elif analysis_mode == "Analyze from News Feed" and article_id:
        # Fetch article by ID
        article = fetch_news_by_id(article_id)
        if article:
            title = article["title"]
            text = article.get("abstract", "")
            
            st.subheader(title)
            st.write(text)
            
            if st.button("Analyze This Article"):
                with st.spinner("Analyzing article..."):
                    article_data = analyze_news_content(text=text, title=title)
    
    # Display analysis results if available
    if article_data:
        st.header("Analysis Results")
        
        # Display metrics in a grid
        metrics = article_data.get("metrics", {})
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Political Influence")
            pol_gauge = create_metrics_gauge(
                metrics.get("political_influence", 0),
                "Political Influence Level",
                ["lightblue", "royalblue", "darkblue"]
            )
            st.plotly_chart(pol_gauge, use_container_width=True)
            
        with col2:
            st.subheader("Rhetoric Intensity")
            rhet_gauge = create_metrics_gauge(
                metrics.get("rhetoric_intensity", 0),
                "Rhetoric Intensity Scale",
                ["lightgreen", "lime", "darkgreen"]
            )
            st.plotly_chart(rhet_gauge, use_container_width=True)
            
        with col3:
            st.subheader("Information Depth")
            info_gauge = create_metrics_gauge(
                metrics.get("information_depth", 0),
                "Information Depth Score",
                ["lightyellow", "yellow", "gold"]
            )
            st.plotly_chart(info_gauge, use_container_width=True)
        
        # Sentiment analysis
        st.subheader("Sentiment Analysis")
        sentiment = metrics.get("sentiment", {})
        sentiment_label = sentiment.get("label", "neutral")
        sentiment_score = sentiment.get("score", 0.5)
        sentiment_emoji = create_sentiment_emoji(sentiment)
        
        sentiment_col1, sentiment_col2 = st.columns([1, 3])
        with sentiment_col1:
            st.markdown(f"<h1 style='text-align: center;'>{sentiment_emoji}</h1>", unsafe_allow_html=True)
        with sentiment_col2:
            st.write(f"**Label:** {sentiment_label.capitalize()}")
            st.write(f"**Confidence:** {sentiment_score:.2f}")
            sentiment_bar = px.bar(
                x=[sentiment_score], 
                y=["Sentiment"],
                orientation="h",
                range_x=[0, 1],
                color_discrete_sequence=["lightcoral"] if sentiment_label == "negative" else ["lightgreen"]
            )
            sentiment_bar.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(sentiment_bar, use_container_width=True)
            
        # Entities
        st.subheader("Entities Detected")
        entities = article_data.get("entities", [])
        
        if entities:
            # Create dataframe from entities
            entities_df = pd.DataFrame(entities)
            
            # Map entity types to friendly names
            type_map = {
                "P": "Person",
                "O": "Organization",
                "G": "Geographic Location",
                "J": "Product",
                "C": "Concept",
                "M": "Media Source"
            }
            
            entities_df["Type"] = entities_df["Type"].map(lambda x: type_map.get(x, x))
            
            # Display entities table
            st.dataframe(
                entities_df[["Label", "Type", "Confidence"]],
                use_container_width=True
            )
        else:
            st.info("No entities detected in the article.")
            
        # Processing time
        st.caption(f"Analysis completed in {article_data.get('processing_time', 0):.2f} seconds")


def render_recommendations(user_id: Optional[str] = None):
    """Render the recommendations page."""
    st.title("News Recommendations")
    
    # Mock user history for demo purposes
    mock_history = [
        "N88753", "N45436", "N23144", "N86255", "N93187"
    ]
    
    history_source = st.radio(
        "Recommendation Source",
        ["Use User ID", "Select From News Feed"]
    )
    
    history_ids = []
    
    if history_source == "Use User ID":
        input_user_id = st.text_input("Enter User ID", value=user_id if user_id else "")
        if input_user_id:
            # In a real scenario, we'd get user history from the database
            # For demo, use mock data
            st.info(f"Using mock history for user {input_user_id}")
            history_ids = mock_history
    else:
        # Let user pick news items
        news_articles = fetch_news(limit=20)
        
        # Display as multiselect
        if news_articles:
            options = {f"{a['news_id']}: {a['title'][:50]}...": a["news_id"] for a in news_articles}
            selected = st.multiselect(
                "Select articles you've read or are interested in",
                options=list(options.keys())
            )
            history_ids = [options[s] for s in selected]
    
    # Get recommendations button
    if (history_ids or (history_source == "Use User ID" and input_user_id)) and st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            recommendations = get_recommendations(
                user_id=input_user_id if history_source == "Use User ID" else None,
                history_news_ids=history_ids
            )
            
            if recommendations and "recommendations" in recommendations:
                st.subheader("Recommended Articles")
                
                # Create expandable list of recommendations
                for i, (rec, score) in enumerate(zip(recommendations["recommendations"], recommendations["scores"])):
                    with st.expander(f"{i+1}. {rec['title']}"):
                        st.write(f"**Category:** {rec['category']}")
                        if rec.get('subcategory'):
                            st.write(f"**Subcategory:** {rec['subcategory']}")
                        st.write(f"**Match Score:** {score:.4f}")
                        
                        if rec.get('abstract'):
                            st.write(f"**Abstract:** {rec['abstract']}")
                            
                        st.button(
                            "Analyze This Article",
                            key=f"rec_analyze_{rec['news_id']}",
                            on_click=lambda article_id=rec["news_id"]: st.session_state.update({
                                "current_page": "Article Analysis",
                                "article_id": article_id
                            })
                        )
            else:
                st.error("Failed to get recommendations.")
    else:
        st.info("Please provide user information or select articles to get recommendations.")


def render_podcast_generator():
    """Render the podcast generator page."""
    st.title("Personalized News Podcast Generator")
    
    # Get news articles for selection
    news_articles = fetch_news(limit=50)
    
    if not news_articles:
        st.warning("No news articles available for podcast generation.")
        return
    
    # Create news selection interface
    st.subheader("Select News Articles")
    
    # Group articles by category
    articles_by_category = {}
    for article in news_articles:
        category = article["category"]
        if category not in articles_by_category:
            articles_by_category[category] = []
        articles_by_category[category].append(article)
    
    # Create tabs for categories
    tabs = st.tabs(list(articles_by_category.keys()))
    
    # Track selected articles
    if "selected_podcast_articles" not in st.session_state:
        st.session_state.selected_podcast_articles = {}
    
    # Show articles in tabs
    for i, (category, articles) in enumerate(articles_by_category.items()):
        with tabs[i]:
            for article in articles:
                col1, col2 = st.columns([9, 1])
                
                with col1:
                    st.write(f"**{article['title']}**")
                    
                with col2:
                    # Check if article is selected
                    article_id = article["news_id"]
                    is_selected = article_id in st.session_state.selected_podcast_articles
                    
                    if st.checkbox("", value=is_selected, key=f"podcast_select_{article_id}"):
                        # Add to selected articles
                        st.session_state.selected_podcast_articles[article_id] = article
                    elif article_id in st.session_state.selected_podcast_articles:
                        # Remove from selected articles
                        del st.session_state.selected_podcast_articles[article_id]
    
    # Display selected articles
    st.subheader("Selected Articles")
    
    if st.session_state.selected_podcast_articles:
        for article_id, article in st.session_state.selected_podcast_articles.items():
            st.write(f"- {article['title']}")
    else:
        st.info("No articles selected yet.")
    
    # Podcast settings
    st.subheader("Podcast Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        include_intro = st.checkbox("Include Introduction", value=True)
    with col2:
        include_transitions = st.checkbox("Include Transitions", value=True)
    
    # Voice selection (mock)
    voice_options = {
        "default": "Default Voice",
        "male1": "Male Voice 1",
        "female1": "Female Voice 1",
        "male2": "Male Voice 2",
        "female2": "Female Voice 2"
    }
    selected_voice = st.selectbox("Select Voice", list(voice_options.values()))
    voice_id = [k for k, v in voice_options.items() if v == selected_voice][0]
    
    # Generate podcast button
    if (
        st.session_state.selected_podcast_articles and 
        st.button("Generate Podcast", disabled=len(st.session_state.selected_podcast_articles) == 0)
    ):
        with st.spinner("Generating podcast..."):
            # Get selected article IDs
            article_ids = list(st.session_state.selected_podcast_articles.keys())
            
            # Generate podcast
            podcast_data = generate_podcast(article_ids)
            
            if podcast_data:
                st.success("Podcast generated successfully!")
                
                # Display podcast player (mock)
                st.subheader("Listen to Podcast")
                st.audio(podcast_data["podcast_url"], format="audio/mp3")
                
                # Display transcript
                with st.expander("View Transcript"):
                    st.write(podcast_data["transcript"])
                    
                # Display metadata
                st.caption(f"Duration: {podcast_data['duration_seconds'] / 60:.1f} minutes")
                st.caption(f"Generation time: {podcast_data['generation_time']:.2f} seconds")
            else:
                st.error("Failed to generate podcast.")


# Main App
def main():
    """Main application function."""
    # Initialize session state for navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = None
    
    # Render sidebar
    page, category, user_id = render_sidebar()
    
    # Override page if set in session state
    if st.session_state.current_page:
        page = st.session_state.current_page
        st.session_state.current_page = None
    
    # Render selected page
    if page == "News Feed":
        render_news_feed(category)
    elif page == "Article Analysis":
        render_article_analysis()
    elif page == "Recommendations":
        render_recommendations(user_id)
    elif page == "Podcast Generator":
        render_podcast_generator()


# Make the main function importable for production_app.py
if __name__ == "__main__":
    main()