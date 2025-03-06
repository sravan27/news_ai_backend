"""
Feature Store implementation for ML features.
"""
from typing import Dict, List, Union, Any, Optional
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select

from .models import FeatureStore
from .session import get_db

logger = logging.getLogger("feature_store")

class FeatureStoreManager:
    """Manager for Feature Store operations."""
    
    def __init__(self):
        """Initialize Feature Store manager."""
        pass
    
    def store_feature(self, feature_name: str, entity_id: str, entity_type: str, 
                     value: Any, timestamp: Optional[datetime] = None):
        """Store a feature in the feature store."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(value, np.ndarray):
                value = value.tolist()
            
            # Convert pandas Series or DataFrame to records
            elif isinstance(value, pd.Series):
                value = value.to_dict()
            elif isinstance(value, pd.DataFrame):
                value = value.to_dict(orient='records')
            
            with get_db() as session:
                feature = FeatureStore(
                    feature_name=feature_name,
                    entity_id=entity_id,
                    entity_type=entity_type,
                    value=value,
                    timestamp=timestamp or datetime.utcnow()
                )
                session.add(feature)
                session.commit()
                
            logger.info(f"Stored feature '{feature_name}' for {entity_type} {entity_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing feature: {e}")
            return False
    
    def get_feature(self, feature_name: str, entity_id: str, entity_type: str) -> Any:
        """Get a feature from the feature store."""
        try:
            with get_db() as session:
                query = select(FeatureStore).where(
                    FeatureStore.feature_name == feature_name,
                    FeatureStore.entity_id == entity_id,
                    FeatureStore.entity_type == entity_type
                ).order_by(FeatureStore.timestamp.desc())
                
                result = session.execute(query).scalars().first()
                
                if result:
                    return result.value
                return None
        except Exception as e:
            logger.error(f"Error getting feature: {e}")
            return None
    
    def get_features_for_entity(self, entity_id: str, entity_type: str) -> Dict[str, Any]:
        """Get all features for an entity."""
        try:
            with get_db() as session:
                query = select(FeatureStore).where(
                    FeatureStore.entity_id == entity_id,
                    FeatureStore.entity_type == entity_type
                )
                
                results = session.execute(query).scalars().all()
                
                features = {}
                for result in results:
                    features[result.feature_name] = result.value
                
                return features
        except Exception as e:
            logger.error(f"Error getting features for entity: {e}")
            return {}
    
    def store_batch_features(self, features: List[Dict[str, Any]]):
        """Store multiple features in a batch."""
        try:
            with get_db() as session:
                for feature_data in features:
                    value = feature_data.get('value')
                    
                    # Convert numpy arrays to lists for JSON serialization
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    
                    # Convert pandas Series or DataFrame to records
                    elif isinstance(value, pd.Series):
                        value = value.to_dict()
                    elif isinstance(value, pd.DataFrame):
                        value = value.to_dict(orient='records')
                    
                    feature = FeatureStore(
                        feature_name=feature_data.get('feature_name'),
                        entity_id=feature_data.get('entity_id'),
                        entity_type=feature_data.get('entity_type'),
                        value=value,
                        timestamp=feature_data.get('timestamp') or datetime.utcnow()
                    )
                    session.add(feature)
                
                session.commit()
                
            logger.info(f"Stored {len(features)} features in batch")
            return True
        except Exception as e:
            logger.error(f"Error storing batch features: {e}")
            return False