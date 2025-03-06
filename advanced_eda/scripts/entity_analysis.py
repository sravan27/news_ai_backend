"""
Entity Analysis module for MIND dataset.

This module provides functions to analyze entities in the MIND dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_entity_distribution(entities_df: pd.DataFrame, entity_column: str = 'Label') -> pd.DataFrame:
    """
    Analyze the distribution of entities.
    
    Args:
        entities_df: DataFrame in long format with one row per entity
        entity_column: Column name to analyze (e.g., 'Label', 'Type', 'WikidataId')
        
    Returns:
        DataFrame with entity counts
    """
    logger.info(f"Analyzing entity distribution for {entity_column}")
    
    try:
        # Count entities
        entity_counts = entities_df[entity_column].value_counts().reset_index()
        entity_counts.columns = [entity_column, 'Count']
        
        # Calculate percentage
        total = entity_counts['Count'].sum()
        entity_counts['Percentage'] = (entity_counts['Count'] / total * 100).round(2)
        
        logger.info(f"Analysis complete: found {len(entity_counts)} unique {entity_column}s")
        return entity_counts
    except Exception as e:
        logger.error(f"Error analyzing entity distribution: {e}")
        raise

def plot_top_entities(entity_counts: pd.DataFrame, 
                     entity_column: str, 
                     title: str = None,
                     top_n: int = 10) -> plt.Figure:
    """
    Plot the top N most common entities.
    
    Args:
        entity_counts: DataFrame with entity counts
        entity_column: Column name containing entities
        title: Plot title
        top_n: Number of top entities to display
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting top {top_n} entities for {entity_column}")
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get top N entities
        top_entities = entity_counts.nlargest(top_n, 'Count')
        
        # Create bar plot
        sns.barplot(x='Count', y=entity_column, data=top_entities, palette="viridis", ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Count")
        ax.set_ylabel(entity_column)
        ax.set_title(title or f"Top {top_n} {entity_column}")
        
        # Add count and percentage text
        for i, row in enumerate(top_entities.itertuples()):
            ax.text(row.Count + 0.5, i, f"{row.Count} ({row.Percentage}%)", va='center')
            
        logger.info(f"Plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting top entities: {e}")
        raise

def analyze_entity_types(entities_df: pd.DataFrame, type_column: str = 'Type', type_mapping: Dict[str, str] = None) -> pd.DataFrame:
    """
    Analyze the distribution of entity types.
    
    Args:
        entities_df: DataFrame in long format with one row per entity
        type_column: Column name containing entity types (default: 'Type')
        type_mapping: Dictionary mapping type codes to full type names (optional)
        
    Returns:
        DataFrame with entity type counts
    """
    logger.info(f"Analyzing entity types using column: {type_column}")
    
    try:
        # Default type mapping for better readability if not provided
        if type_mapping is None:
            type_mapping = {
                'P': 'Person',
                'O': 'Organization',
                'L': 'Location',
                'G': 'Geo-political entity',
                'C': 'Concept',
                'M': 'Medical',
                'F': 'Facility',
                'N': 'Natural feature',
                'U': 'Unknown',
                'S': 'Event',
                'W': 'Work of art',
                'B': 'Brand',
                'H': 'Historical event/person',
                'K': 'Book',
                'V': 'Video',
                'J': 'Journal',
                'R': 'Research/Scientific term',
                'A': 'Astronomical object',
                'I': 'Invention/Technology'
            }
        
        # Create a copy to avoid modifying the original DataFrame
        df_copy = entities_df.copy()
        
        # Add full type name
        df_copy['Type_Full'] = df_copy[type_column].map(type_mapping)
        
        # Count entity types
        type_counts = df_copy[type_column].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        
        # Add full type name
        type_counts['Type_Full'] = type_counts['Type'].map(type_mapping)
        
        # Calculate percentage
        total = type_counts['Count'].sum()
        type_counts['Percentage'] = (type_counts['Count'] / total * 100).round(2)
        
        logger.info(f"Analysis complete: found {len(type_counts)} unique entity types")
        return type_counts
    except Exception as e:
        logger.error(f"Error analyzing entity types: {e}")
        raise

def plot_entity_types(type_counts: pd.DataFrame, 
                  type_column: str = "Type",
                  plot_type: str = "bar",
                  title: str = None) -> plt.Figure:
    """
    Plot entity type distribution as a bar or pie chart.
    
    Args:
        type_counts: DataFrame with entity type counts
        type_column: Column name containing entity types
        plot_type: Type of plot ('bar' or 'pie')
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting entity type distribution as {plot_type} chart")
    
    try:
        if plot_type.lower() == 'bar':
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create bar plot
            sns.barplot(x=type_column, y='Count', data=type_counts, palette="viridis", ax=ax)
            
            # Set labels and title
            ax.set_xlabel("Entity Type")
            ax.set_ylabel("Count")
            ax.set_title(title or "Distribution of Entity Types")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            
            # Add count and percentage text
            for i, row in enumerate(type_counts.itertuples()):
                ax.text(i, row.Count + 0.5, f"{row.Count} ({row.Percentage}%)", ha='center')
                
            # Add type description in the tooltip or as annotation
            type_desc_column = "Type_Full" if "Type_Full" in type_counts.columns else "Type_desc" if "Type_desc" in type_counts.columns else None
            if type_desc_column:
                for i, row in enumerate(type_counts.itertuples()):
                    if hasattr(row, type_desc_column) and getattr(row, type_desc_column) is not None:
                        ax.annotate(
                            getattr(row, type_desc_column),
                            xy=(i, 0),
                            xytext=(0, -20),
                            textcoords='offset points',
                            ha='center',
                            va='top',
                            fontsize=8,
                            color='gray',
                            rotation=45
                        )
                
            plt.tight_layout()
            logger.info("Bar chart created successfully")
            
        elif plot_type.lower() == 'pie':
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                type_counts['Count'], 
                labels=type_counts[type_column],
                autopct='%1.1f%%',
                startangle=140,
                colors=sns.color_palette('pastel', n_colors=len(type_counts))
            )
            
            # Set title and legend
            ax.set_title(title or "Distribution of Entity Types")
            
            # Add legend with full type names if available
            type_desc_column = "Type_Full" if "Type_Full" in type_counts.columns else "Type_desc" if "Type_desc" in type_counts.columns else None
            if type_desc_column:
                legend_labels = [f"{getattr(row, type_column)}: {getattr(row, type_desc_column)}" for row in type_counts.itertuples()]
                ax.legend(wedges, legend_labels, title="Entity Types", 
                        loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
                
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            logger.info("Pie chart created successfully")
        else:
            raise ValueError(f"Invalid plot type: {plot_type}. Must be 'bar' or 'pie'.")
            
        return fig
    except Exception as e:
        logger.error(f"Error plotting entity types: {e}")
        raise

def plot_entity_types_bar(type_counts: pd.DataFrame, title: str = None) -> plt.Figure:
    """
    Plot entity type distribution as a bar chart.
    
    Args:
        type_counts: DataFrame with entity type counts
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting entity type distribution (bar chart)")
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        sns.barplot(x='Type', y='Count', data=type_counts, palette="viridis", ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Entity Type")
        ax.set_ylabel("Count")
        ax.set_title(title or "Distribution of Entity Types")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        # Add count and percentage text
        for i, row in enumerate(type_counts.itertuples()):
            ax.text(i, row.Count + 0.5, f"{row.Count} ({row.Percentage}%)", ha='center')
            
        # Add type description in the tooltip or as annotation
        for i, row in enumerate(type_counts.itertuples()):
            if hasattr(row, 'Type_Full') and row.Type_Full is not None:
                ax.annotate(
                    row.Type_Full,
                    xy=(i, 0),
                    xytext=(0, -20),
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    fontsize=8,
                    color='gray',
                    rotation=45
                )
            
        plt.tight_layout()
        logger.info("Bar chart created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting entity types bar chart: {e}")
        raise

def plot_entity_types_pie(type_counts: pd.DataFrame, title: str = None) -> plt.Figure:
    """
    Plot entity type distribution as a pie chart.
    
    Args:
        type_counts: DataFrame with entity type counts
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting entity type distribution (pie chart)")
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            type_counts['Count'], 
            labels=type_counts['Type'],
            autopct='%1.1f%%',
            startangle=140,
            colors=sns.color_palette('pastel', n_colors=len(type_counts))
        )
        
        # Set title and legend
        ax.set_title(title or "Distribution of Entity Types")
        
        # Add legend with full type names if available
        if 'Type_Full' in type_counts.columns:
            legend_labels = [f"{row.Type}: {row.Type_Full}" for row in type_counts.itertuples()]
            ax.legend(wedges, legend_labels, title="Entity Types", 
                      loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
            
        logger.info("Pie chart created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting entity types pie chart: {e}")
        raise

def analyze_entity_confidence(entities_df: pd.DataFrame) -> Dict:
    """
    Analyze the confidence scores of entity recognition.
    
    Args:
        entities_df: DataFrame in long format with one row per entity
        
    Returns:
        Dictionary with confidence score statistics
    """
    logger.info("Analyzing entity confidence scores")
    
    try:
        # Calculate confidence statistics
        confidence_stats = {
            'mean': entities_df['Confidence'].mean(),
            'median': entities_df['Confidence'].median(),
            'min': entities_df['Confidence'].min(),
            'max': entities_df['Confidence'].max(),
            'std': entities_df['Confidence'].std()
        }
        
        # Create confidence bins
        confidence_bins = pd.cut(
            entities_df['Confidence'],
            bins=[0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
            labels=['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-0.9', '0.9-0.95', '0.95-0.99', '0.99-1.0']
        )
        
        # Count entities by confidence bin
        confidence_counts = confidence_bins.value_counts().sort_index()
        
        # Add to results
        confidence_stats['bin_counts'] = confidence_counts.to_dict()
        
        logger.info("Confidence analysis complete")
        return confidence_stats
    except Exception as e:
        logger.error(f"Error analyzing entity confidence: {e}")
        raise

def plot_confidence_distribution(entities_df: pd.DataFrame, title: str = None) -> plt.Figure:
    """
    Plot the distribution of confidence scores.
    
    Args:
        entities_df: DataFrame in long format with one row per entity
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting confidence score distribution")
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        sns.histplot(entities_df['Confidence'].dropna(), bins=20, kde=True, color="blue", ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Frequency")
        ax.set_title(title or "Distribution of Entity Confidence Scores")
        
        # Add mean and median lines
        mean_confidence = entities_df['Confidence'].mean()
        median_confidence = entities_df['Confidence'].median()
        
        ax.axvline(mean_confidence, color='red', linestyle='--', 
                  label=f'Mean: {mean_confidence:.3f}')
        ax.axvline(median_confidence, color='green', linestyle=':', 
                  label=f'Median: {median_confidence:.3f}')
        
        ax.legend()
        
        logger.info("Confidence distribution plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting confidence distribution: {e}")
        raise

def analyze_entity_co_occurrence(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which entities frequently co-occur in the same news articles.
    
    Args:
        news_df: News DataFrame with parsed entities
        
    Returns:
        DataFrame with co-occurrence counts
    """
    logger.info("Analyzing entity co-occurrence")
    
    try:
        # Function to extract entity pairs from a list of entities
        def get_entity_pairs(entity_list):
            entity_labels = [entity.get('Label') for entity in entity_list if entity and 'Label' in entity]
            pairs = []
            for i in range(len(entity_labels)):
                for j in range(i + 1, len(entity_labels)):
                    # Sort to ensure consistent ordering
                    pair = tuple(sorted([entity_labels[i], entity_labels[j]]))
                    pairs.append(pair)
            return pairs
        
        # Extract all entity pairs from titles and abstracts
        all_pairs = []
        
        # Process Title_Entities
        title_pairs = news_df['Title_Entities'].apply(get_entity_pairs)
        all_pairs.extend([pair for pairs_list in title_pairs for pair in pairs_list])
        
        # Process Abstract_Entities
        abstract_pairs = news_df['Abstract_Entities'].apply(get_entity_pairs)
        all_pairs.extend([pair for pairs_list in abstract_pairs for pair in pairs_list])
        
        # Count co-occurrences
        pair_counts = Counter(all_pairs)
        
        # Convert to DataFrame
        co_occurrence_df = pd.DataFrame([
            {'Entity1': pair[0], 'Entity2': pair[1], 'Count': count}
            for pair, count in pair_counts.most_common()
        ])
        
        logger.info(f"Co-occurrence analysis complete: found {len(co_occurrence_df)} entity pairs")
        return co_occurrence_df
    except Exception as e:
        logger.error(f"Error analyzing entity co-occurrence: {e}")
        raise

def plot_entity_network(co_occurrence_df: pd.DataFrame, 
                       min_count: int = 3, 
                       max_entities: int = 30) -> plt.Figure:
    """
    Plot a network graph of entity co-occurrences.
    
    Args:
        co_occurrence_df: DataFrame with entity co-occurrence counts
        min_count: Minimum co-occurrence count to include
        max_entities: Maximum number of entities to include
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Plotting entity network graph (min_count={min_count}, max_entities={max_entities})")
    
    try:
        import networkx as nx
        
        # Filter co-occurrences by minimum count
        filtered_df = co_occurrence_df[co_occurrence_df['Count'] >= min_count]
        
        # Calculate top entities by occurrence frequency
        all_entities = pd.concat([
            filtered_df['Entity1'], filtered_df['Entity2']
        ]).value_counts()
        
        top_entities = set(all_entities.nlargest(max_entities).index)
        
        # Filter co-occurrences to top entities
        filtered_df = filtered_df[
            filtered_df['Entity1'].isin(top_entities) & 
            filtered_df['Entity2'].isin(top_entities)
        ]
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for entity in top_entities:
            count = all_entities[entity]
            G.add_node(entity, count=count)
            
        # Add edges
        for row in filtered_df.itertuples():
            G.add_edge(row.Entity1, row.Entity2, weight=row.Count)
            
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Calculate node sizes based on frequency
        node_sizes = [G.nodes[node]['count'] * 50 for node in G.nodes]
        
        # Calculate edge widths based on co-occurrence count
        edge_widths = [G[u][v]['weight'] for u, v in G.edges]
        
        # Calculate node colors based on frequency
        node_colors = [G.nodes[node]['count'] for node in G.nodes]
        
        # Calculate positions using spring layout
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                              alpha=0.8, cmap='viridis', ax=ax)
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray', ax=ax)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', ax=ax)
        
        # Set title
        plt.title(f"Entity Co-occurrence Network (min. {min_count} co-occurrences)")
        
        # Remove axis
        plt.axis('off')
        
        logger.info("Network graph created successfully")
        return fig
    except ImportError:
        logger.error("NetworkX library not found. Install it with 'pip install networkx'")
        raise
    except Exception as e:
        logger.error(f"Error plotting entity network: {e}")
        raise

def analyze_entity_embedding_similarities(entity_df: pd.DataFrame,
                                        embeddings_df: pd.DataFrame,
                                        top_n: int = 5) -> Dict:
    """
    Analyze entity embedding similarities.
    
    Args:
        entity_df: DataFrame with entities
        embeddings_df: DataFrame with entity embeddings
        top_n: Number of most similar entities to return for each entity
        
    Returns:
        Dictionary with similarity results
    """
    logger.info("Analyzing entity embedding similarities")
    
    try:
        # Extract columns that contain embeddings
        embedding_cols = [col for col in embeddings_df.columns if col != 'WikidataId']
        
        # Convert embeddings to numpy array
        embeddings = embeddings_df[embedding_cols].values
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Convert to DataFrame with WikidataIds as index and columns
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=embeddings_df['WikidataId'],
            columns=embeddings_df['WikidataId']
        )
        
        # Extract top N most similar entities for each entity
        results = {}
        
        # Get unique entities from entity_df
        unique_entities = entity_df[['WikidataId', 'Label']].drop_duplicates()
        
        # For each entity, find most similar entities
        for _, row in unique_entities.iterrows():
            wikidata_id = row['WikidataId']
            label = row['Label']
            
            if wikidata_id in similarity_df.index:
                # Get similarities for this entity
                similarities = similarity_df[wikidata_id].sort_values(ascending=False)
                
                # Exclude self (which would be the most similar)
                similarities = similarities[similarities.index != wikidata_id]
                
                # Get top N most similar
                top_similar = similarities.head(top_n)
                
                # Convert to dictionary with WikidataId -> similarity
                top_similar_dict = top_similar.to_dict()
                
                # Store results
                results[wikidata_id] = {
                    'label': label,
                    'similar_entities': top_similar_dict
                }
        
        logger.info(f"Similarity analysis complete for {len(results)} entities")
        return results
    except Exception as e:
        logger.error(f"Error analyzing embedding similarities: {e}")
        raise

def visualize_embeddings_pca(embeddings_df: pd.DataFrame, 
                            entity_df: pd.DataFrame,
                            n_components: int = 2) -> plt.Figure:
    """
    Visualize entity embeddings using PCA.
    
    Args:
        embeddings_df: DataFrame with entity embeddings
        entity_df: DataFrame with entity metadata
        n_components: Number of PCA components
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Visualizing embeddings using PCA (n_components={n_components})")
    
    try:
        # Extract embedding columns
        embedding_cols = [col for col in embeddings_df.columns if col != 'WikidataId']
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings_df[embedding_cols].values)
        
        # Create DataFrame with reduced dimensions
        reduced_df = pd.DataFrame(
            reduced_embeddings,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        reduced_df['WikidataId'] = embeddings_df['WikidataId'].values
        
        # Merge with entity metadata
        entity_metadata = entity_df[['WikidataId', 'Label', 'Type']].drop_duplicates()
        vis_df = reduced_df.merge(entity_metadata, on='WikidataId', how='left')
        
        # Create figure based on number of components
        if n_components == 2:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create scatter plot, coloring by entity type
            scatter = sns.scatterplot(
                x='PC1', y='PC2', hue='Type', data=vis_df,
                palette='viridis', alpha=0.7, s=30, ax=ax
            )
            
            # Set labels and title
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            ax.set_title("PCA Projection of Entity Embeddings")
            
            # Add legend with entity type descriptions
            type_mapping = {
                'P': 'Person',
                'O': 'Organization',
                'L': 'Location',
                'G': 'Geo-political entity',
                'C': 'Concept',
                'M': 'Medical',
                'F': 'Facility',
                'N': 'Natural feature',
                'U': 'Unknown',
                'S': 'Event',
                'W': 'Work of art',
                'B': 'Brand',
                'H': 'Historical event/person',
                'K': 'Book',
                'V': 'Video',
                'J': 'Journal',
                'R': 'Research/Scientific term',
                'A': 'Astronomical object',
                'I': 'Invention/Technology'
            }
            
            # Create custom legend with type descriptions
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [f"{label}: {type_mapping.get(label, 'Unknown')}" for label in labels]
            ax.legend(handles, new_labels, title="Entity Type")
            
        elif n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create 3D scatter plot
            scatter = ax.scatter(
                vis_df['PC1'], vis_df['PC2'], vis_df['PC3'],
                c=pd.factorize(vis_df['Type'])[0],
                cmap='viridis', alpha=0.7, s=30
            )
            
            # Set labels and title
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)")
            ax.set_title("3D PCA Projection of Entity Embeddings")
            
            # Add legend
            type_labels = vis_df['Type'].unique()
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                         markersize=10, label=f"{label}")
                              for i, label in enumerate(type_labels)]
            ax.legend(handles=legend_elements, title="Entity Type")
            
        else:
            raise ValueError("Only 2 or 3 components are supported for visualization")
            
        logger.info("PCA visualization created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error visualizing embeddings with PCA: {e}")
        raise

def visualize_embeddings_tsne(embeddings_df: pd.DataFrame, 
                             entity_df: pd.DataFrame,
                             n_components: int = 2,
                             perplexity: int = 30,
                             n_iter: int = 1000) -> plt.Figure:
    """
    Visualize entity embeddings using t-SNE.
    
    Args:
        embeddings_df: DataFrame with entity embeddings
        entity_df: DataFrame with entity metadata
        n_components: Number of t-SNE components
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        
    Returns:
        Matplotlib figure
    """
    logger.info(f"Visualizing embeddings using t-SNE (n_components={n_components}, perplexity={perplexity})")
    
    try:
        # Extract embedding columns
        embedding_cols = [col for col in embeddings_df.columns if col != 'WikidataId']
        
        # Apply t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42
        )
        reduced_embeddings = tsne.fit_transform(embeddings_df[embedding_cols].values)
        
        # Create DataFrame with reduced dimensions
        reduced_df = pd.DataFrame(
            reduced_embeddings,
            columns=[f'TSNE{i+1}' for i in range(n_components)]
        )
        reduced_df['WikidataId'] = embeddings_df['WikidataId'].values
        
        # Merge with entity metadata
        entity_metadata = entity_df[['WikidataId', 'Label', 'Type']].drop_duplicates()
        vis_df = reduced_df.merge(entity_metadata, on='WikidataId', how='left')
        
        # Create figure based on number of components
        if n_components == 2:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create scatter plot, coloring by entity type
            scatter = sns.scatterplot(
                x='TSNE1', y='TSNE2', hue='Type', data=vis_df,
                palette='viridis', alpha=0.7, s=30, ax=ax
            )
            
            # Set labels and title
            ax.set_xlabel("t-SNE Component 1")
            ax.set_ylabel("t-SNE Component 2")
            ax.set_title("t-SNE Projection of Entity Embeddings")
            
            # Add legend with entity type descriptions
            type_mapping = {
                'P': 'Person',
                'O': 'Organization',
                'L': 'Location',
                'G': 'Geo-political entity',
                'C': 'Concept',
                'M': 'Medical',
                'F': 'Facility',
                'N': 'Natural feature',
                'U': 'Unknown',
                'S': 'Event',
                'W': 'Work of art',
                'B': 'Brand',
                'H': 'Historical event/person',
                'K': 'Book',
                'V': 'Video',
                'J': 'Journal',
                'R': 'Research/Scientific term',
                'A': 'Astronomical object',
                'I': 'Invention/Technology'
            }
            
            # Create custom legend with type descriptions
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [f"{label}: {type_mapping.get(label, 'Unknown')}" for label in labels]
            ax.legend(handles, new_labels, title="Entity Type")
            
        elif n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create 3D scatter plot
            scatter = ax.scatter(
                vis_df['TSNE1'], vis_df['TSNE2'], vis_df['TSNE3'],
                c=pd.factorize(vis_df['Type'])[0],
                cmap='viridis', alpha=0.7, s=30
            )
            
            # Set labels and title
            ax.set_xlabel("t-SNE Component 1")
            ax.set_ylabel("t-SNE Component 2")
            ax.set_zlabel("t-SNE Component 3")
            ax.set_title("3D t-SNE Projection of Entity Embeddings")
            
            # Add legend
            type_labels = vis_df['Type'].unique()
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                         markersize=10, label=f"{label}")
                              for i, label in enumerate(type_labels)]
            ax.legend(handles=legend_elements, title="Entity Type")
            
        else:
            raise ValueError("Only 2 or 3 components are supported for visualization")
            
        logger.info("t-SNE visualization created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error visualizing embeddings with t-SNE: {e}")
        raise
        
def analyze_entity_occurrences(entity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze entity occurrences in the text.
    
    Args:
        entity_df: DataFrame with entities
        
    Returns:
        DataFrame with added occurrence information
    """
    logger.info("Analyzing entity occurrences")
    
    try:
        # Add occurrence count and index information
        result_df = entity_df.copy()
        
        # Group by document ID and entity Label to analyze occurrence patterns
        grouped = result_df.groupby(['NewsID', 'Label'])
        
        # Calculate occurrences
        occurrences = grouped.size().reset_index(name='OccurrenceCount')
        first_occurrences = grouped['Offset'].min().reset_index(name='FirstOccurrencePosition')
        
        # Merge results back
        result_df = result_df.merge(occurrences, on=['NewsID', 'Label'], how='left')
        result_df = result_df.merge(first_occurrences, on=['NewsID', 'Label'], how='left')
        
        # Calculate normalized position (0-1 scale)
        text_lengths = result_df.groupby('NewsID')['TextLength'].first()
        result_df['NormalizedPosition'] = result_df.apply(
            lambda x: x['Offset'] / text_lengths[x['NewsID']] if x['NewsID'] in text_lengths else None, 
            axis=1
        )
        
        logger.info(f"Entity occurrence analysis complete for {len(result_df)} entities")
        return result_df
    except Exception as e:
        logger.error(f"Error analyzing entity occurrences: {e}")
        raise

def plot_occurrences_distribution(entity_df: pd.DataFrame, title: str = None) -> plt.Figure:
    """
    Plot the distribution of entity occurrences.
    
    Args:
        entity_df: DataFrame with entity occurrences
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting entity occurrences distribution")
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histogram of occurrence counts
        sns.histplot(entity_df['OccurrenceCount'].dropna(), bins=10, kde=True, ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Number of Occurrences")
        ax.set_ylabel("Frequency")
        ax.set_title(title or "Distribution of Entity Occurrences")
        
        # Add mean and median lines
        mean_occurrences = entity_df['OccurrenceCount'].mean()
        median_occurrences = entity_df['OccurrenceCount'].median()
        
        ax.axvline(mean_occurrences, color='red', linestyle='--', 
                  label=f'Mean: {mean_occurrences:.2f}')
        ax.axvline(median_occurrences, color='green', linestyle=':', 
                  label=f'Median: {median_occurrences:.2f}')
        
        ax.legend()
        
        logger.info("Occurrences distribution plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting occurrences distribution: {e}")
        raise

def plot_first_occurrence_position(entity_df: pd.DataFrame, title: str = None) -> plt.Figure:
    """
    Plot the distribution of first occurrence positions.
    
    Args:
        entity_df: DataFrame with entity occurrences
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    logger.info("Plotting first occurrence position distribution")
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histogram of normalized positions
        sns.histplot(entity_df['NormalizedPosition'].dropna(), bins=20, kde=True, ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Normalized Position in Text (0-1)")
        ax.set_ylabel("Frequency")
        ax.set_title(title or "Distribution of Entity First Occurrence Positions")
        
        # Add mean and median lines
        mean_pos = entity_df['NormalizedPosition'].mean()
        median_pos = entity_df['NormalizedPosition'].median()
        
        ax.axvline(mean_pos, color='red', linestyle='--', 
                  label=f'Mean: {mean_pos:.2f}')
        ax.axvline(median_pos, color='green', linestyle=':', 
                  label=f'Median: {median_pos:.2f}')
        
        ax.legend()
        
        logger.info("First occurrence position plot created successfully")
        return fig
    except Exception as e:
        logger.error(f"Error plotting first occurrence positions: {e}")
        raise

def identify_political_entities(entity_df: pd.DataFrame, political_keywords: List[str] = None) -> pd.DataFrame:
    """
    Identify entities that are likely political in nature.
    
    Args:
        entity_df: DataFrame with entities
        political_keywords: List of keywords that indicate political entities
        
    Returns:
        DataFrame with political entity indicators
    """
    logger.info("Identifying political entities")
    
    # Default political keywords if none provided
    if political_keywords is None:
        political_keywords = [
            'president', 'senator', 'congress', 'parliament', 'minister', 'governor',
            'election', 'campaign', 'vote', 'ballot', 'democrat', 'republican', 'party',
            'government', 'administration', 'policy', 'political', 'candidate',
            'legislation', 'senate', 'representative', 'constitution', 'court',
            'justice', 'federal', 'law', 'regulation', 'treaty', 'diplomat'
        ]
    
    try:
        # Create a copy to avoid modifying the original DataFrame
        result_df = entity_df.copy()
        
        # Flag entities with political labels
        result_df['is_political_entity'] = result_df['Label'].str.lower().apply(
            lambda x: any(keyword in str(x).lower() for keyword in political_keywords)
        )
        
        # Flag entities with political types (G: Geo-political entity)
        result_df['is_political_entity'] = result_df['is_political_entity'] | (result_df['Type'] == 'G')
        
        # Count political entities
        political_count = result_df['is_political_entity'].sum()
        logger.info(f"Identified {political_count} political entities out of {len(result_df)} total entities")
        
        return result_df
    except Exception as e:
        logger.error(f"Error identifying political entities: {e}")
        raise

def wikidata_enrichment(entity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich entity data with additional information from Wikidata.
    
    Args:
        entity_df: DataFrame with entities and WikidataIds
        
    Returns:
        DataFrame with enriched entity data
    """
    logger.info("Enriching entity data with Wikidata information")
    
    try:
        # Get unique WikidataIds
        unique_ids = entity_df['WikidataId'].dropna().unique().tolist()
        
        # Fetch additional information from Wikidata
        from .data_loader import fetch_wikidata_info
        wikidata_info = fetch_wikidata_info(unique_ids)
        
        # Convert to DataFrame
        wikidata_df = pd.DataFrame([
            {
                'WikidataId': wid,
                'Wikidata_Label': info['label'],
                'Wikidata_Description': info['description'],
                'Wikipedia_URL': info['wikipedia_url']
            }
            for wid, info in wikidata_info.items()
        ])
        
        # Merge with entity DataFrame
        enriched_df = entity_df.merge(wikidata_df, on='WikidataId', how='left')
        
        logger.info(f"Entity data enriched successfully with Wikidata info for {len(wikidata_info)} entities")
        return enriched_df
    except Exception as e:
        logger.error(f"Error enriching entity data: {e}")
        return entity_df  # Return original DataFrame on error