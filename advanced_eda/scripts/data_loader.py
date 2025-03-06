"""
Data Loader module for MIND dataset.

This module provides functions to load and preprocess the MIND dataset files.
"""

import os
import json
import pandas as pd
import ast
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_news_data(news_file_path: str) -> pd.DataFrame:
    """
    Load news data from a TSV file.
    
    Args:
        news_file_path: Path to the news TSV file
        
    Returns:
        DataFrame containing news data
    """
    logger.info(f"Loading news data from {news_file_path}")
    
    try:
        news = pd.read_csv(
            news_file_path, 
            sep='\t', 
            header=None,
            names=['News_ID', 'Category', 'Subcategory', 'Title', 'Abstract', 'URL', 'Title_Entities', 'Abstract_Entities']
        )
        logger.info(f"Successfully loaded news data with shape {news.shape}")
        return news
    except Exception as e:
        logger.error(f"Error loading news data: {e}")
        raise

def load_behaviors_data(behaviors_file_path: str) -> pd.DataFrame:
    """
    Load user behaviors data from a TSV file.
    
    Args:
        behaviors_file_path: Path to the behaviors TSV file
        
    Returns:
        DataFrame containing user behaviors data
    """
    logger.info(f"Loading behaviors data from {behaviors_file_path}")
    
    try:
        behaviors = pd.read_csv(
            behaviors_file_path, 
            sep='\t', 
            header=None,
            names=['Impression_ID', 'User_ID', 'Time', 'History', 'Impressions']
        )
        
        # Convert time to datetime
        behaviors['Time'] = pd.to_datetime(behaviors['Time'])
        
        logger.info(f"Successfully loaded behaviors data with shape {behaviors.shape}")
        return behaviors
    except Exception as e:
        logger.error(f"Error loading behaviors data: {e}")
        raise

def load_entity_embeddings(embedding_file_path: str) -> pd.DataFrame:
    """
    Load entity embeddings from a .vec file.
    
    Args:
        embedding_file_path: Path to the entity_embedding.vec file
        
    Returns:
        DataFrame containing entity embeddings
    """
    logger.info(f"Loading entity embeddings from {embedding_file_path}")
    
    try:
        # Read the file as raw text
        with open(embedding_file_path, "r") as f:
            lines = f.readlines()
            
        # Process each line
        data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 1:
                wikidata_id = parts[0]
                embedding = [float(val) for val in parts[1:]]
                data.append([wikidata_id] + embedding)
                
        # Create column names
        embedding_dim = len(data[0]) - 1
        columns = ['WikidataId'] + [f'dim_{i}' for i in range(embedding_dim)]
        
        # Create DataFrame
        embeddings_df = pd.DataFrame(data, columns=columns)
        
        logger.info(f"Successfully loaded entity embeddings with shape {embeddings_df.shape}")
        return embeddings_df
    except Exception as e:
        logger.error(f"Error loading entity embeddings: {e}")
        raise

def load_relation_embeddings(embedding_file_path: str) -> pd.DataFrame:
    """
    Load relation embeddings from a .vec file.
    
    Args:
        embedding_file_path: Path to the relation_embedding.vec file
        
    Returns:
        DataFrame containing relation embeddings
    """
    logger.info(f"Loading relation embeddings from {embedding_file_path}")
    
    try:
        # Read the file as raw text
        with open(embedding_file_path, "r") as f:
            lines = f.readlines()
            
        # Process each line
        data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 1:
                relation_id = parts[0]
                embedding = [float(val) for val in parts[1:]]
                data.append([relation_id] + embedding)
                
        # Create column names
        embedding_dim = len(data[0]) - 1
        columns = ['RelationId'] + [f'dim_{i}' for i in range(embedding_dim)]
        
        # Create DataFrame
        embeddings_df = pd.DataFrame(data, columns=columns)
        
        logger.info(f"Successfully loaded relation embeddings with shape {embeddings_df.shape}")
        return embeddings_df
    except Exception as e:
        logger.error(f"Error loading relation embeddings: {e}")
        raise
        
def parse_entities(entities_str: str) -> List[Dict]:
    """
    Parse entity JSON strings into Python dictionaries.
    
    Args:
        entities_str: JSON string representing entities
        
    Returns:
        List of entity dictionaries
    """
    if pd.isna(entities_str) or entities_str == '[]':
        return []
    
    try:
        return ast.literal_eval(entities_str)
    except Exception as e:
        logger.error(f"Error parsing entity string: {e}")
        return []

def process_entities_to_long_format(news_df: pd.DataFrame, entity_column: str) -> pd.DataFrame:
    """
    Process entities into long format (one row per entity).
    
    Args:
        news_df: News DataFrame
        entity_column: Column name containing entity data ('Title_Entities' or 'Abstract_Entities')
        
    Returns:
        DataFrame in long format with one row per entity
    """
    logger.info(f"Processing {entity_column} to long format")
    
    try:
        # Create a copy of the original DataFrame
        df_long = news_df.copy()
        
        # Parse entities
        df_long[entity_column] = df_long[entity_column].apply(parse_entities)
        
        # Remove rows with empty entity lists
        df_long = df_long[df_long[entity_column].apply(len) > 0]
        
        # Explode entities to create one row per entity
        df_long = df_long.explode(entity_column)
        
        # Convert entity dictionaries to columns
        entity_df = pd.json_normalize(df_long[entity_column])
        
        # Concatenate original columns with entity columns
        result_df = pd.concat(
            [df_long.drop(columns=[entity_column]).reset_index(drop=True), 
             entity_df.reset_index(drop=True)], 
            axis=1
        )
        
        logger.info(f"Successfully processed to long format with shape {result_df.shape}")
        return result_df
    except Exception as e:
        logger.error(f"Error processing to long format: {e}")
        raise

def process_entities_to_wide_format(news_df: pd.DataFrame, entity_column: str) -> pd.DataFrame:
    """
    Process entities into wide format (multiple columns per entity).
    
    Args:
        news_df: News DataFrame
        entity_column: Column name containing entity data ('Title_Entities' or 'Abstract_Entities')
        
    Returns:
        DataFrame in wide format with multiple columns per entity
    """
    logger.info(f"Processing {entity_column} to wide format")
    
    try:
        # Create a copy of the original DataFrame
        df_wide = news_df.copy()
        
        # Parse entities
        df_wide[entity_column] = df_wide[entity_column].apply(parse_entities)
        
        # Get maximum number of entities per row
        max_entities = df_wide[entity_column].apply(len).max()
        
        # Create entity columns
        for i in range(max_entities):
            # Create a column for each entity
            df_wide[f'Entity_{i+1}'] = df_wide[entity_column].apply(
                lambda x: x[i] if i < len(x) else None
            )
            
            # For each entity, expand its properties into columns
            entity_col = f'Entity_{i+1}'
            non_null_mask = df_wide[entity_col].notna()
            
            if non_null_mask.any():
                # Convert dictionaries to DataFrame
                entity_props = pd.json_normalize(df_wide.loc[non_null_mask, entity_col])
                
                # Add prefix to column names
                entity_props = entity_props.add_prefix(f'{entity_col}_')
                
                # Add to original DataFrame
                for col in entity_props.columns:
                    df_wide.loc[non_null_mask, col] = entity_props[col].values
        
        # Drop original entities column and temporary entity columns
        cols_to_drop = [entity_column] + [f'Entity_{i+1}' for i in range(max_entities)]
        df_wide = df_wide.drop(columns=cols_to_drop)
        
        logger.info(f"Successfully processed to wide format with shape {df_wide.shape}")
        return df_wide
    except Exception as e:
        logger.error(f"Error processing to wide format: {e}")
        raise

def process_entities_to_compact_format(news_df: pd.DataFrame, entity_column: str) -> pd.DataFrame:
    """
    Process entities into compact format (comma-separated values).
    
    Args:
        news_df: News DataFrame
        entity_column: Column name containing entity data ('Title_Entities' or 'Abstract_Entities')
        
    Returns:
        DataFrame with compact entity representations
    """
    logger.info(f"Processing {entity_column} to compact format")
    
    try:
        # Create a copy of the original DataFrame
        df_compact = news_df.copy()
        
        # Parse entities
        df_compact[entity_column] = df_compact[entity_column].apply(parse_entities)
        
        # Extract relevant properties
        df_compact[f'Entity_Labels'] = df_compact[entity_column].apply(
            lambda x: ", ".join([entity.get('Label', '') for entity in x]) if x else ""
        )
        
        df_compact[f'Entity_Types'] = df_compact[entity_column].apply(
            lambda x: ", ".join([entity.get('Type', '') for entity in x]) if x else ""
        )
        
        df_compact[f'Entity_WikidataIds'] = df_compact[entity_column].apply(
            lambda x: ", ".join([entity.get('WikidataId', '') for entity in x]) if x else ""
        )
        
        df_compact[f'Entity_Confidence'] = df_compact[entity_column].apply(
            lambda x: ", ".join([str(entity.get('Confidence', '')) for entity in x]) if x else ""
        )
        
        # Drop original entities column
        df_compact = df_compact.drop(columns=[entity_column])
        
        logger.info(f"Successfully processed to compact format with shape {df_compact.shape}")
        return df_compact
    except Exception as e:
        logger.error(f"Error processing to compact format: {e}")
        raise

def extract_entity_wikidata_ids(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract WikidataIds from Title_Entities and Abstract_Entities.
    
    Args:
        news_df: News DataFrame
        
    Returns:
        DataFrame with extracted WikidataIds
    """
    logger.info("Extracting WikidataIds from entities")
    
    try:
        # Create a copy of the DataFrame
        result_df = news_df.copy()
        
        # Parse entities and extract WikidataIds
        result_df['Title_Entities'] = result_df['Title_Entities'].apply(parse_entities)
        result_df['Abstract_Entities'] = result_df['Abstract_Entities'].apply(parse_entities)
        
        # Extract WikidataIds
        result_df['Title_WikidataIds'] = result_df['Title_Entities'].apply(
            lambda entities: [entity.get('WikidataId') for entity in entities if 'WikidataId' in entity]
        )
        
        result_df['Abstract_WikidataIds'] = result_df['Abstract_Entities'].apply(
            lambda entities: [entity.get('WikidataId') for entity in entities if 'WikidataId' in entity]
        )
        
        # Combine all WikidataIds
        result_df['All_WikidataIds'] = result_df.apply(
            lambda row: list(set(row['Title_WikidataIds'] + row['Abstract_WikidataIds'])), axis=1
        )
        
        logger.info("Successfully extracted WikidataIds")
        return result_df
    except Exception as e:
        logger.error(f"Error extracting WikidataIds: {e}")
        raise

def merge_news_with_embeddings(news_df: pd.DataFrame, 
                               embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge news data with entity embeddings.
    
    Args:
        news_df: News DataFrame with extracted WikidataIds
        embeddings_df: Entity embeddings DataFrame
        
    Returns:
        DataFrame with merged embeddings
    """
    logger.info("Merging news with embeddings")
    
    try:
        # Ensure WikidataIds are extracted
        if 'All_WikidataIds' not in news_df.columns:
            news_df = extract_entity_wikidata_ids(news_df)
        
        # Explode the WikidataIds column to have one row per ID
        news_exploded = news_df.explode('All_WikidataIds')
        
        # Merge with embeddings
        merged_df = news_exploded.merge(
            embeddings_df, 
            left_on='All_WikidataIds',
            right_on='WikidataId',
            how='left'
        )
        
        logger.info(f"Successfully merged news with embeddings, shape: {merged_df.shape}")
        return merged_df
    except Exception as e:
        logger.error(f"Error merging with embeddings: {e}")
        raise

def fetch_wikidata_info(wikidata_ids: List[str]) -> Dict:
    """
    Fetch additional information from Wikidata for given entity IDs.
    
    Args:
        wikidata_ids: List of Wikidata IDs (e.g., ['Q312', 'Q43274'])
        
    Returns:
        Dictionary mapping Wikidata IDs to additional information
    """
    logger.info(f"Fetching Wikidata info for {len(wikidata_ids)} entities")
    
    try:
        import requests
        
        # Initialize results dictionary
        results = {}
        
        # Process in batches to avoid hitting API limits
        batch_size = 50
        for i in range(0, len(wikidata_ids), batch_size):
            batch = wikidata_ids[i:i+batch_size]
            
            # Prepare query
            ids_str = ' '.join([f'wd:{wid}' for wid in batch])
            query = f"""
            SELECT ?item ?itemLabel ?description ?article WHERE {{
              VALUES ?item {{ {ids_str} }}
              OPTIONAL {{ ?item schema:description ?description. FILTER(LANG(?description) = "en") }}
              OPTIONAL {{ ?article schema:about ?item . 
                        ?article schema:inLanguage "en" . 
                        ?article schema:isPartOf <https://en.wikipedia.org/> }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            """
            
            # Query Wikidata SPARQL endpoint
            response = requests.get(
                'https://query.wikidata.org/sparql',
                params={'query': query, 'format': 'json'},
                headers={'User-Agent': 'MIND_Dataset_Analysis/1.0'}
            )
            
            if response.ok:
                data = response.json()
                
                # Process results
                for item in data['results']['bindings']:
                    wid = item['item']['value'].split('/')[-1]
                    
                    # Extract data
                    label = item.get('itemLabel', {}).get('value', '')
                    description = item.get('description', {}).get('value', '')
                    wikipedia = item.get('article', {}).get('value', '')
                    
                    # Store in results
                    results[wid] = {
                        'label': label,
                        'description': description,
                        'wikipedia_url': wikipedia
                    }
            else:
                logger.warning(f"Failed to fetch data for batch {i//batch_size + 1}: {response.status_code}")
        
        logger.info(f"Successfully fetched data for {len(results)} Wikidata entities")
        return results
    except Exception as e:
        logger.error(f"Error fetching Wikidata info: {e}")
        return {}

def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed DataFrame to a file.
    
    Args:
        df: DataFrame to save
        output_path: Path where the file should be saved
        
    Returns:
        None
    """
    logger.info(f"Saving processed data to {output_path}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Determine file format based on extension
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.parquet'):
            df.to_parquet(output_path, index=False)
        elif output_path.endswith('.json'):
            df.to_json(output_path, orient='records', lines=True)
        else:
            df.to_csv(output_path, index=False)
            
        logger.info(f"Successfully saved data to {output_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise