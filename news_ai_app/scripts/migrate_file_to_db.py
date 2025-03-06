"""
Data migration script for transitioning from file-based to database storage.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from database.models import Base, User, NewsArticle, Embedding, NewsMetrics, Interaction
from database.session import engine, get_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=parent_dir / 'logs' / 'db_migration.log'
)
logger = logging.getLogger("db_migration")

def setup_database():
    """Create tables if they don't exist."""
    logger.info("Setting up database tables")
    Base.metadata.create_all(bind=engine)

def load_news_data():
    """Load news data from parquet files."""
    try:
        silver_path = Path(parent_dir.parent) / "ml_pipeline" / "data" / "silver"
        news_base = pd.read_parquet(silver_path / "news_base.parquet")
        
        # Try to load text features if they exist
        try:
            text_features = pd.read_parquet(silver_path / "text_features.parquet")
        except Exception:
            text_features = None
            
        return news_base, text_features
    except Exception as e:
        logger.error(f"Error loading news data: {e}")
        return None, None

def load_embeddings():
    """Load embeddings from numpy files."""
    try:
        silver_path = Path(parent_dir.parent) / "ml_pipeline" / "data" / "silver"
        
        # Check if embeddings exist
        title_emb_path = silver_path / "title_embeddings.npy"
        abstract_emb_path = silver_path / "abstract_embeddings.npy"
        
        title_embeddings = np.load(title_emb_path) if title_emb_path.exists() else None
        abstract_embeddings = np.load(abstract_emb_path) if abstract_emb_path.exists() else None
        
        return title_embeddings, abstract_embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        return None, None

def load_interactions():
    """Load interaction data."""
    try:
        silver_path = Path(parent_dir.parent) / "ml_pipeline" / "data" / "silver"
        
        # Try different potential interaction filenames
        for filename in ["interactions.parquet", "interactions_train.parquet"]:
            path = silver_path / filename
            if path.exists():
                interactions = pd.read_parquet(path)
                return interactions
                
        logger.warning("No interaction files found")
        return None
    except Exception as e:
        logger.error(f"Error loading interactions: {e}")
        return None

def migrate_news_articles(session, news_df, text_features_df, title_emb, abstract_emb):
    """Migrate news articles and their features to the database."""
    total = len(news_df)
    logger.info(f"Migrating {total} news articles")
    
    # Join news and features
    if text_features_df is not None and "news_id" in text_features_df.columns:
        try:
            news_with_features = pd.merge(news_df, text_features_df, on="news_id", how="left")
        except Exception as e:
            logger.error(f"Error merging news with features: {e}")
            news_with_features = news_df
    else:
        news_with_features = news_df
    
    # Get silicon metrics if available
    silicon_path = Path(parent_dir.parent) / "ml_pipeline" / "data" / "silicon" / "processing_summary.json"
    metrics = {}
    if silicon_path.exists():
        try:
            with open(silicon_path, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading silicon metrics: {e}")
    
    # Batch process for better performance
    batch_size = 1000
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = news_with_features.iloc[start_idx:end_idx]
        
        try:
            for i, row in batch.iterrows():
                idx = i if title_emb is None else start_idx + (i - batch.index[0])
                
                # Extract entities if they exist
                entities = None
                if "entities" in row:
                    entities = row["entities"]
                elif "title_entities" in row and "abstract_entities" in row:
                    entities = {
                        "title": row["title_entities"],
                        "abstract": row["abstract_entities"]
                    }
                
                # Create news article
                article = NewsArticle(
                    news_id=row['news_id'],
                    category=row['category'],
                    subcategory=row.get('subcategory', None),
                    title=row['title'],
                    abstract=row.get('abstract', None),
                    url=row.get('url', None),
                    entities=entities,
                    publication_date=None  # Add date parsing if available
                )
                session.add(article)
                
                # Add embeddings if available
                if title_emb is not None and idx < len(title_emb):
                    title_vector = title_emb[idx].tolist() if idx < len(title_emb) else None
                    abstract_vector = abstract_emb[idx].tolist() if abstract_emb is not None and idx < len(abstract_emb) else None
                    
                    if title_vector is not None:
                        embedding = Embedding(
                            news_id=row['news_id'],
                            title_embedding=title_vector,
                            abstract_embedding=abstract_vector
                        )
                        session.add(embedding)
                
                # Add metrics if available for this article
                if row['news_id'] in metrics:
                    article_metrics = metrics[row['news_id']]
                    news_metrics = NewsMetrics(
                        news_id=row['news_id'],
                        political_influence=article_metrics.get('political_influence', None),
                        rhetoric_intensity=article_metrics.get('rhetoric_intensity', None),
                        information_depth=article_metrics.get('information_depth', None),
                        sentiment=article_metrics.get('sentiment', None)
                    )
                    session.add(news_metrics)
            
            # Commit batch
            session.commit()
            logger.info(f"Migrated articles {start_idx} to {end_idx}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error in batch {start_idx}-{end_idx}: {e}")

def migrate_interactions(session, interactions_df):
    """Migrate interaction data to the database."""
    if interactions_df is None:
        return
    
    total = len(interactions_df)
    logger.info(f"Migrating {total} interactions")
    
    # Create a set of user IDs for faster lookup
    user_ids = set()
    
    # Batch process for better performance
    batch_size = 1000
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = interactions_df.iloc[start_idx:end_idx]
        
        try:
            for _, row in batch.iterrows():
                user_id = row['user_id']
                
                # Create user if not exists
                if user_id not in user_ids:
                    user = User(id=user_id)
                    session.add(user)
                    user_ids.add(user_id)
                
                # Create interaction
                interaction = Interaction(
                    user_id=user_id,
                    news_id=row['news_id'],
                    clicked=row.get('clicked', False),
                    timestamp=datetime.fromisoformat(row['time']) if isinstance(row.get('time'), str) else datetime.now(),
                    impression_id=row.get('impression_id', None)
                )
                session.add(interaction)
            
            # Commit batch
            session.commit()
            logger.info(f"Migrated interactions {start_idx} to {end_idx}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error in batch {start_idx}-{end_idx}: {e}")

def main():
    """Main migration function."""
    logger.info("Starting data migration")
    
    # Setup database
    setup_database()
    
    # Load data
    news_df, text_features_df = load_news_data()
    title_emb, abstract_emb = load_embeddings()
    interactions_df = load_interactions()
    
    # Check if data was loaded successfully
    if news_df is None:
        logger.error("Failed to load news data. Aborting migration.")
        return
    
    # Perform migration within a session
    with get_db() as session:
        # Migrate news articles
        migrate_news_articles(session, news_df, text_features_df, title_emb, abstract_emb)
        
        # Migrate interactions
        migrate_interactions(session, interactions_df)
    
    logger.info("Data migration completed")

if __name__ == "__main__":
    main()