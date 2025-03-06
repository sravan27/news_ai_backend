#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Silicon Metrics Dashboard - Advanced model metrics visualization for News AI project.
This dashboard provides comprehensive visualization for the silicon layer models:
1. Political Influence
2. Rhetoric Intensity 
3. Information Depth
4. Sentiment
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

# Add paths for imports
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = current_dir.parent.parent
ml_pipeline_dir = parent_dir / "ml_pipeline"
sys.path.append(str(parent_dir))
sys.path.append(str(ml_pipeline_dir))

# Set page configuration
st.set_page_config(
    page_title="Silicon Metrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
SILICON_PATH = ml_pipeline_dir / "data" / "silicon"
DEPLOYED_PATH = ml_pipeline_dir / "models" / "deployed"
METRICS = ["political_influence", "rhetoric_intensity", "information_depth", "sentiment"]

# Cache data loading functions
@st.cache_data
def load_model_metadata(metric_name):
    """Load model metadata for a specific metric."""
    silicon_meta_path = SILICON_PATH / metric_name / "model_metadata.json"
    deployed_meta_path = DEPLOYED_PATH / metric_name / "model_metadata.json"
    
    if silicon_meta_path.exists():
        with open(silicon_meta_path, 'r') as f:
            return json.load(f)
    elif deployed_meta_path.exists():
        with open(deployed_meta_path, 'r') as f:
            return json.load(f)
    else:
        return None

@st.cache_data
def load_model_card(metric_name):
    """Load model card for a specific metric."""
    model_card_path = DEPLOYED_PATH / metric_name / "model_card.json"
    
    if model_card_path.exists():
        with open(model_card_path, 'r') as f:
            return json.load(f)
    else:
        return None

@st.cache_data
def load_feature_names(metric_name):
    """Load feature names for a specific metric."""
    silicon_path = SILICON_PATH / metric_name / "feature_names.json"
    deployed_path = DEPLOYED_PATH / metric_name / "feature_names.json"
    
    if silicon_path.exists():
        with open(silicon_path, 'r') as f:
            return json.load(f)
    elif deployed_path.exists():
        with open(deployed_path, 'r') as f:
            return json.load(f)
    else:
        return None

@st.cache_data
def load_prediction_samples(metric_name):
    """Load prediction samples for a specific metric."""
    samples_path = SILICON_PATH / metric_name / "prediction_samples.csv"
    
    if samples_path.exists():
        return pd.read_csv(samples_path)
    else:
        return None

@st.cache_data
def load_features_data(metric_name):
    """Load feature data for a specific metric."""
    features_path = SILICON_PATH / metric_name / "features.parquet"
    
    if features_path.exists():
        return pd.read_parquet(features_path)
    else:
        return None

@st.cache_data
def load_processing_summary():
    """Load processing summary for all metrics."""
    summary_path = SILICON_PATH / "processing_summary.json"
    
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            return json.load(f)
    else:
        return None

# Visualization functions
def plot_prediction_vs_actual(samples_df):
    """Plot prediction vs actual values."""
    fig = px.scatter(
        samples_df, 
        x='actual', 
        y='predicted',
        opacity=0.6,
        title='Predicted vs Actual Values',
        labels={'actual': 'Actual Value', 'predicted': 'Predicted Value'}
    )
    
    # Add diagonal line (perfect predictions)
    min_val = min(samples_df['actual'].min(), samples_df['predicted'].min())
    max_val = max(samples_df['actual'].max(), samples_df['predicted'].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        )
    )
    
    return fig

def plot_error_distribution(samples_df):
    """Plot error distribution."""
    samples_df['error'] = samples_df['predicted'] - samples_df['actual']
    
    fig = px.histogram(
        samples_df,
        x='error',
        nbins=30,
        title='Prediction Error Distribution',
        labels={'error': 'Error (Predicted - Actual)', 'count': 'Frequency'},
        marginal='box'
    )
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    
    return fig

def plot_feature_importance(feature_names, model_path, model_type):
    """Plot feature importance if available."""
    try:
        feature_names_list = feature_names.get('feature_names', [])
        
        if not feature_names_list or not model_path.exists():
            return None
            
        if model_type == 'nn':
            # For neural networks, we don't have direct feature importance
            return None
            
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names_list,
                'importance': model.feature_importances_
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Take top 20 features
            top_n = min(20, len(importance_df))
            importance_df = importance_df.head(top_n)
            
            # Plot
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title=f'Top {top_n} Feature Importance',
                labels={'importance': 'Importance', 'feature': 'Feature'},
                color='importance'
            )
            
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"Error plotting feature importance: {e}")
        return None

def plot_feature_distributions(features_df, feature_names_list, top_n=5):
    """Plot distributions of top features."""
    if features_df is None or feature_names_list is None:
        return None
        
    # If features_df has a 'label' column, calculate feature correlations with label
    if 'label' in features_df.columns:
        correlations = {}
        for feature in feature_names_list:
            if feature in features_df.columns:
                correlations[feature] = abs(features_df[feature].corr(features_df['label']))
                
        # Sort by correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, corr in sorted_features[:top_n]]
    else:
        # If no label, just take the first top_n features
        top_features = feature_names_list[:top_n]
    
    # Create subplots for each feature
    figs = []
    for feature in top_features:
        if feature in features_df.columns:
            fig = px.histogram(
                features_df,
                x=feature,
                nbins=30,
                title=f'Distribution of {feature}',
                labels={feature: feature, 'count': 'Frequency'},
                marginal='box'
            )
            figs.append((feature, fig))
    
    return figs

def plot_model_comparison(processing_summary):
    """Plot model performance comparison across metrics."""
    if not processing_summary:
        return None
        
    # Extract metric names and performance data
    metrics_data = []
    for metric_name, details in processing_summary.items():
        if metric_name in METRICS and 'best_model' in details:
            model_type = details.get('best_model', {}).get('model_type', 'Unknown')
            model_name = details.get('best_model', {}).get('model_name', 'Unknown')
            model_full_name = f"{model_type}_{model_name}"
            
            test_metrics = details.get('best_model', {}).get('test_metrics', {})
            
            metrics_data.append({
                'metric_name': metric_name,
                'model': model_full_name,
                'mse': test_metrics.get('mse', 0),
                'mae': test_metrics.get('mae', 0),
                'r2': test_metrics.get('r2', 0)
            })
    
    if not metrics_data:
        return None
        
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create visualization for each performance metric
    figs = {}
    
    # MSE comparison
    fig_mse = px.bar(
        metrics_df,
        x='metric_name',
        y='mse',
        color='model',
        title='Mean Squared Error by Metric',
        labels={'metric_name': 'Metric', 'mse': 'Mean Squared Error', 'model': 'Model'}
    )
    figs['mse'] = fig_mse
    
    # MAE comparison
    fig_mae = px.bar(
        metrics_df,
        x='metric_name',
        y='mae',
        color='model',
        title='Mean Absolute Error by Metric',
        labels={'metric_name': 'Metric', 'mae': 'Mean Absolute Error', 'model': 'Model'}
    )
    figs['mae'] = fig_mae
    
    # R2 comparison
    fig_r2 = px.bar(
        metrics_df,
        x='metric_name',
        y='r2',
        color='model',
        title='R² Score by Metric',
        labels={'metric_name': 'Metric', 'r2': 'R² Score', 'model': 'Model'}
    )
    figs['r2'] = fig_r2
    
    return figs

def render_metric_evaluation(metric_name):
    """Render evaluation for a specific metric."""
    st.header(f"{metric_name.replace('_', ' ').title()} Evaluation")
    
    # Load data
    model_metadata = load_model_metadata(metric_name)
    model_card = load_model_card(metric_name)
    feature_names = load_feature_names(metric_name)
    samples_df = load_prediction_samples(metric_name)
    features_df = load_features_data(metric_name)
    
    # Display model information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        if model_metadata:
            st.write(f"**Model Type:** {model_metadata.get('model_type', 'Unknown')}")
            st.write(f"**Model Name:** {model_metadata.get('model_name', 'Unknown')}")
            
            # Display test metrics
            test_metrics = model_metadata.get('test_metrics', {})
            st.write("**Test Metrics:**")
            metrics_df = pd.DataFrame({
                'Metric': ['MSE', 'MAE', 'R²'],
                'Value': [
                    f"{test_metrics.get('mse', 0):.4f}",
                    f"{test_metrics.get('mae', 0):.4f}",
                    f"{test_metrics.get('r2', 0):.4f}"
                ]
            })
            st.dataframe(metrics_df)
            
            # Display feature count
            st.write(f"**Feature Count:** {model_metadata.get('feature_count', 0)}")
            st.write(f"**Training Date:** {model_metadata.get('training_date', 'Unknown')}")
        else:
            st.info(f"No model metadata available for {metric_name}")
    
    with col2:
        st.subheader("Model Card")
        if model_card:
            st.write(f"**Purpose:** {model_card.get('purpose', 'Unknown')}")
            st.write(f"**Description:** {model_card.get('description', 'Unknown')}")
            
            # Display performance summary
            st.write("**Performance Summary:**")
            st.write(model_card.get('performance_summary', 'No performance summary available'))
            
            # Display limitations
            st.write("**Limitations:**")
            limitations = model_card.get('limitations', [])
            if limitations:
                for limitation in limitations:
                    st.write(f"- {limitation}")
            else:
                st.write("No limitations specified")
        else:
            st.info(f"No model card available for {metric_name}")
    
    # Plot predictions vs actual values
    st.subheader("Prediction Analysis")
    
    tab1, tab2 = st.tabs(["Predictions vs Actual", "Error Analysis"])
    
    with tab1:
        if samples_df is not None:
            fig = plot_prediction_vs_actual(samples_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate metrics
            actual = samples_df['actual']
            predicted = samples_df['predicted']
            
            mse = mean_squared_error(actual, predicted)
            mae = mean_absolute_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            
            st.write(f"**Sample MSE:** {mse:.4f}")
            st.write(f"**Sample MAE:** {mae:.4f}")
            st.write(f"**Sample R²:** {r2:.4f}")
        else:
            st.info(f"No prediction samples available for {metric_name}")
    
    with tab2:
        if samples_df is not None:
            fig = plot_error_distribution(samples_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate error statistics
            samples_df['error'] = samples_df['predicted'] - samples_df['actual']
            
            st.write(f"**Mean Error:** {samples_df['error'].mean():.4f}")
            st.write(f"**Error Standard Deviation:** {samples_df['error'].std():.4f}")
            st.write(f"**Median Error:** {samples_df['error'].median():.4f}")
            st.write(f"**Min Error:** {samples_df['error'].min():.4f}")
            st.write(f"**Max Error:** {samples_df['error'].max():.4f}")
        else:
            st.info(f"No prediction samples available for {metric_name}")
    
    # Plot feature importance
    st.subheader("Feature Analysis")
    
    if model_metadata and feature_names:
        model_type = model_metadata.get('model_type', '')
        model_name = model_metadata.get('model_name', '')
        
        if model_type != 'nn':
            model_path = SILICON_PATH / metric_name / "model.pkl"
            fig = plot_feature_importance(feature_names, model_path, model_type)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model")
        else:
            st.info("Feature importance not available for neural network models")
            
        # Plot feature distributions
        feature_names_list = feature_names.get('feature_names', [])
        if feature_names_list and features_df is not None:
            st.subheader("Feature Distributions")
            
            top_n = min(5, len(feature_names_list))
            st.write(f"Showing distributions for top {top_n} features")
            
            feature_figs = plot_feature_distributions(features_df, feature_names_list, top_n)
            
            if feature_figs:
                for i, (feature, fig) in enumerate(feature_figs):
                    if i % 2 == 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        with col2:
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature distributions available")
    else:
        st.info(f"No feature information available for {metric_name}")

def render_model_comparison():
    """Render model comparison across metrics."""
    st.header("Model Comparison Across Metrics")
    
    # Load processing summary
    processing_summary = load_processing_summary()
    
    if processing_summary:
        # Display summary statistics
        metrics_info = {}
        for metric_name in METRICS:
            if metric_name in processing_summary:
                metric_details = processing_summary[metric_name]
                best_model = metric_details.get('best_model', {})
                metrics_info[metric_name] = {
                    'model_type': best_model.get('model_type', 'Unknown'),
                    'model_name': best_model.get('model_name', 'Unknown'),
                    'mse': best_model.get('test_metrics', {}).get('mse', 0),
                    'mae': best_model.get('test_metrics', {}).get('mae', 0),
                    'r2': best_model.get('test_metrics', {}).get('r2', 0)
                }
        
        # Create summary table
        if metrics_info:
            summary_df = pd.DataFrame.from_dict(metrics_info, orient='index')
            summary_df['model'] = summary_df['model_type'] + '_' + summary_df['model_name']
            summary_df = summary_df.reset_index().rename(columns={'index': 'metric_name'})
            
            st.subheader("Model Performance Summary")
            st.dataframe(summary_df[['metric_name', 'model', 'mse', 'mae', 'r2']])
            
            # Plot comparisons
            figs = plot_model_comparison(processing_summary)
            
            if figs:
                # Display each performance metric in tabs
                tab1, tab2, tab3 = st.tabs(["Mean Squared Error", "Mean Absolute Error", "R² Score"])
                
                with tab1:
                    st.plotly_chart(figs['mse'], use_container_width=True)
                
                with tab2:
                    st.plotly_chart(figs['mae'], use_container_width=True)
                
                with tab3:
                    st.plotly_chart(figs['r2'], use_container_width=True)
        else:
            st.info("No model summary information available")
    else:
        st.info("No processing summary available")

def render_wiki_integration():
    """Render WikiData integration insights."""
    st.header("WikiData Integration")
    
    st.subheader("Entity Enhancement")
    
    # Demo WikiData integration
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        Our metrics are enhanced with WikiData entity information, providing rich context for news content analysis:
        
        - **Entity Recognition**: Named entities are detected in news titles and abstracts
        - **Entity Linking**: Entities are linked to their WikiData IDs
        - **Entity Types**: We extract entity types from WikiData (Person, Location, Organization, etc.)
        - **Entity Relationships**: We analyze relationships between entities
        - **Domain Knowledge**: WikiData provides domain-specific information for politics, business, etc.
        """)
        
        st.info("Entity information improves accuracy of all four metrics, especially Political Influence.")
    
    with col2:
        # Display example entity graph
        st.write("**Example Entity Graph**")
        
        # Create sample entity graph visualization
        nodes = [
            {"id": "Q30", "label": "United States", "type": "Country", "size": 15},
            {"id": "Q76", "label": "Barack Obama", "type": "Person", "size": 10},
            {"id": "Q6279", "label": "The White House", "type": "Location", "size": 8},
            {"id": "Q7099", "label": "U.S. Congress", "type": "Organization", "size": 12},
            {"id": "Q11696", "label": "Joe Biden", "type": "Person", "size": 10}
        ]
        
        edges = [
            {"source": "Q76", "target": "Q30", "relation": "President of"},
            {"source": "Q76", "target": "Q6279", "relation": "worked at"},
            {"source": "Q11696", "target": "Q76", "relation": "Vice President to"},
            {"source": "Q7099", "target": "Q30", "relation": "legislature of"}
        ]
        
        # Create DataFrame for nodes and edges
        nodes_df = pd.DataFrame(nodes)
        edges_df = pd.DataFrame(edges)
        
        # Create network diagram
        import networkx as nx
        G = nx.DiGraph()
        
        # Add nodes
        for _, node in nodes_df.iterrows():
            G.add_node(node['id'], label=node['label'], type=node['type'], size=node['size'])
        
        # Add edges
        for _, edge in edges_df.iterrows():
            G.add_edge(edge['source'], edge['target'], relation=edge['relation'])
        
        # Create positions
        pos = nx.spring_layout(G, seed=42)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=[G.nodes[node]['size'] * 100 for node in G.nodes],
            node_color=['#1f77b4' if G.nodes[node]['type'] == 'Person' else 
                       '#ff7f0e' if G.nodes[node]['type'] == 'Location' else
                       '#2ca02c' if G.nodes[node]['type'] == 'Organization' else
                       '#d62728' for node in G.nodes],
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels={node: G.nodes[node]['label'] for node in G.nodes})
        
        plt.axis('off')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    st.subheader("WikiData Query Examples")
    
    # Display some example WikiData queries
    query_examples = [
        {
            "title": "Get all political parties in the US",
            "query": """
            SELECT ?party ?partyLabel
            WHERE {
              ?party wdt:P31 wd:Q7278.
              ?party wdt:P17 wd:Q30.
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
            }
            """,
            "description": "This query returns all entities that are political parties (P31 = instance of, Q7278 = political party) in the United States (P17 = country, Q30 = USA)"
        },
        {
            "title": "Find politicians that are also business people",
            "query": """
            SELECT ?person ?personLabel
            WHERE {
              ?person wdt:P106 wd:Q82955.
              ?person wdt:P106 wd:Q131524.
              SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
            }
            LIMIT 10
            """,
            "description": "This query finds people who have both politician (Q82955) and business person (Q131524) as their occupation (P106)"
        }
    ]
    
    for i, example in enumerate(query_examples):
        with st.expander(f"Example {i+1}: {example['title']}"):
            st.code(example['query'], language='sparql')
            st.write(example['description'])
    
    st.write("""
    ## Benefits of WikiData Integration
    
    1. **Enhanced Entity Understanding**: WikiData provides rich context about entities
    2. **Reduced Ambiguity**: Entity disambiguation through WikiData identifiers
    3. **Cross-lingual Support**: WikiData is multilingual, enabling future cross-language analysis
    4. **Dynamic Knowledge**: WikiData is constantly updated, keeping our analysis current
    5. **Open Standards**: Using WikiData aligns with open data standards
    """)

def render_mind_dataset_info():
    """Render information about the MIND dataset."""
    st.header("MIND Dataset Information")
    
    st.write("""
    The Microsoft News Dataset (MIND) is a large-scale dataset for news recommendation research.
    It contains anonymized behavior logs from Microsoft News website.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Statistics")
        
        stats_df = pd.DataFrame({
            'Split': ['Train', 'Dev', 'Test'],
            'Users': ['1,000,000', '50,000', '50,000'],
            'News Articles': ['161,013', '15,835', '14,322'],
            'Impressions': ['15,777,377', '718,770', '705,022'],
            'Clicks': ['24,155,470', '750,561', 'N/A']
        })
        
        st.dataframe(stats_df)
    
    with col2:
        st.subheader("Data Format")
        
        st.write("""
        - **News**: articles with title, abstract, body, category, etc.
        - **Behaviors**: user interactions with news articles
        - **Embeddings**: entity and relation embeddings
        """)
        
        st.code("""
# news.tsv format
news_id \t category \t subcategory \t title \t abstract \t url \t title_entities \t abstract_entities

# behaviors.tsv format
impression_id \t user_id \t time \t history \t impressions
        """)
    
    st.subheader("Data Processing Pipeline")
    
    # Create pipeline diagram
    pipeline_steps = [
        "Raw MIND Data",
        "Bronze Layer\n(Raw Parquet)",
        "Silver Layer\n(Features)",
        "Silicon Layer\n(Advanced Models)",
        "Gold Layer\n(Evaluation)"
    ]
    
    # Create a horizontal pipeline diagram
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Add boxes for each step
    box_width = 1.5
    box_height = 1
    
    for i, step in enumerate(pipeline_steps):
        rect = plt.Rectangle((i*2, 0), box_width, box_height, 
                           fc='lightblue', ec='blue', alpha=0.7)
        ax.add_patch(rect)
        ax.text(i*2 + box_width/2, box_height/2, step, 
               ha='center', va='center', fontweight='bold')
        
        # Add arrow
        if i < len(pipeline_steps) - 1:
            ax.arrow(i*2 + box_width + 0.1, box_height/2, 0.8, 0, 
                   head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax.set_xlim(-0.5, len(pipeline_steps)*2)
    ax.set_ylim(-0.5, box_height + 0.5)
    ax.axis('off')
    
    st.pyplot(fig)
    
    st.write("""
    Our data processing pipeline follows the medallion architecture with an additional Silicon layer:
    
    1. **Bronze Layer**: Raw data ingestion from MIND dataset TSV files to parquet
    2. **Silver Layer**: Feature extraction and preprocessing
    3. **Silicon Layer**: Advanced model training for the four key metrics
    4. **Gold Layer**: Model evaluation and deployment
    """)

def main():
    """Main function for the Streamlit app."""
    st.title("Silicon Metrics Dashboard")
    st.write("Advanced model metrics visualization and evaluation for News AI")
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Section",
        ["Overview", "Model Comparison", "Political Influence", "Rhetoric Intensity", 
         "Information Depth", "Sentiment", "WikiData Integration", "MIND Dataset"]
    )
    
    # Render selected page
    if page == "Overview":
        st.header("Silicon Metrics Overview")
        
        st.write("""
        This dashboard provides comprehensive visualization and evaluation for our silicon layer models.
        The silicon layer contains advanced machine learning models for four key news metrics:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Metrics")
            st.write("""
            1. **Political Influence**: Measures political content and bias in news articles
            2. **Rhetoric Intensity**: Quantifies use of rhetorical devices and emotional language
            3. **Information Depth**: Evaluates information density and complexity
            4. **Sentiment**: Analyzes the emotional tone of the news article
            """)
        
        with col2:
            st.subheader("Model Types")
            st.write("""
            Our system evaluates multiple model types for each metric:
            
            - **Gradient Boosting**: LightGBM, XGBoost, CatBoost
            - **Ensemble Methods**: Voting, Stacking, Bagging
            - **Neural Networks**: Deep networks, Residual networks
            """)
        
        # Display architecture diagram
        st.subheader("Silicon Layer Architecture")
        
        # Create architecture diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define boxes
        components = [
            {"name": "MIND Dataset", "position": (1, 5), "width": 2, "height": 1, "color": "lightblue"},
            {"name": "WikiData", "position": (1, 3), "width": 2, "height": 1, "color": "lightgreen"},
            {"name": "Feature Extraction", "position": (4, 4), "width": 2, "height": 1, "color": "lightyellow"},
            {"name": "Model Training", "position": (7, 4), "width": 2, "height": 3, "color": "lightcoral"},
            {"name": "Political Influence", "position": (10, 6), "width": 2, "height": 0.8, "color": "lightblue"},
            {"name": "Rhetoric Intensity", "position": (10, 5), "width": 2, "height": 0.8, "color": "lightgreen"},
            {"name": "Information Depth", "position": (10, 4), "width": 2, "height": 0.8, "color": "lightyellow"},
            {"name": "Sentiment", "position": (10, 3), "width": 2, "height": 0.8, "color": "lightcoral"}
        ]
        
        # Draw the components
        for component in components:
            rect = plt.Rectangle(
                component["position"], 
                component["width"], 
                component["height"],
                fc=component["color"],
                ec="black",
                alpha=0.7
            )
            ax.add_patch(rect)
            ax.text(
                component["position"][0] + component["width"]/2,
                component["position"][1] + component["height"]/2,
                component["name"],
                ha='center',
                va='center',
                fontweight='bold'
            )
        
        # Add arrows
        arrows = [
            {"start": (3, 5.5), "end": (4, 4.5), "color": "blue"},
            {"start": (3, 3.5), "end": (4, 4.2), "color": "green"},
            {"start": (6, 4.5), "end": (7, 4.5), "color": "black"},
            {"start": (9, 6), "end": (10, 6.4), "color": "blue"},
            {"start": (9, 5.5), "end": (10, 5.4), "color": "green"},
            {"start": (9, 5), "end": (10, 4.4), "color": "orange"},
            {"start": (9, 4.5), "end": (10, 3.4), "color": "red"}
        ]
        
        for arrow in arrows:
            ax.arrow(
                arrow["start"][0], arrow["start"][1],
                arrow["end"][0] - arrow["start"][0], arrow["end"][1] - arrow["start"][1],
                head_width=0.2, head_length=0.2, fc=arrow["color"], ec=arrow["color"],
                length_includes_head=True
            )
        
        ax.set_xlim(0, 13)
        ax.set_ylim(2, 7)
        ax.axis('off')
        
        st.pyplot(fig)
        
        # Quick navigation
        st.subheader("Quick Navigation")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Political Influence"):
                st.switch_page(f"{current_dir}/silicon_metrics_dashboard.py#Political Influence")
        
        with col2:
            if st.button("Rhetoric Intensity"):
                st.switch_page(f"{current_dir}/silicon_metrics_dashboard.py#Rhetoric Intensity")
        
        with col3:
            if st.button("Information Depth"):
                st.switch_page(f"{current_dir}/silicon_metrics_dashboard.py#Information Depth")
        
        with col4:
            if st.button("Sentiment"):
                st.switch_page(f"{current_dir}/silicon_metrics_dashboard.py#Sentiment")
        
    elif page == "Model Comparison":
        render_model_comparison()
    elif page == "Political Influence":
        render_metric_evaluation("political_influence")
    elif page == "Rhetoric Intensity":
        render_metric_evaluation("rhetoric_intensity")
    elif page == "Information Depth":
        render_metric_evaluation("information_depth")
    elif page == "Sentiment":
        render_metric_evaluation("sentiment")
    elif page == "WikiData Integration":
        render_wiki_integration()
    elif page == "MIND Dataset":
        render_mind_dataset_info()

if __name__ == "__main__":
    main()