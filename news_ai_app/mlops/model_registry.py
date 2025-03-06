"""
Model Registry for managing model versions and deployments.
"""
import os
from datetime import datetime
import logging
import json
import shutil
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import select

from database.models import ModelVersion
from database.session import get_db

logger = logging.getLogger("model_registry")

class ModelRegistry:
    """Model Registry for model management."""
    
    def __init__(self, models_dir=None):
        """Initialize Model Registry."""
        self.models_dir = models_dir or Path(__file__).resolve().parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.models_dir / "versions").mkdir(exist_ok=True)
        (self.models_dir / "deployed").mkdir(exist_ok=True)
    
    def register_model(self, model_name: str, model_path: str, 
                      metrics: Dict[str, float] = None, 
                      parameters: Dict[str, Any] = None) -> Optional[str]:
        """Register a model in the registry."""
        try:
            # Generate version
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            version = f"{timestamp}"
            
            # Create version directory
            version_dir = self.models_dir / "versions" / model_name / version
            version_dir.mkdir(exist_ok=True, parents=True)
            
            # Copy model files
            source_path = Path(model_path)
            if source_path.is_dir():
                # Copy all files in directory
                for item in source_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, version_dir)
            else:
                # Copy single file
                shutil.copy2(source_path, version_dir)
            
            # Save metadata
            metadata = {
                "model_name": model_name,
                "version": version,
                "timestamp": timestamp,
                "metrics": metrics or {},
                "parameters": parameters or {}
            }
            
            with open(version_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Store in database
            with get_db() as session:
                model_version = ModelVersion(
                    model_name=model_name,
                    version=version,
                    uri=str(version_dir),
                    metrics=metrics,
                    parameters=parameters,
                    is_active=False
                )
                session.add(model_version)
                session.commit()
            
            logger.info(f"Registered model {model_name} version {version}")
            return version
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None
    
    def deploy_model(self, model_name: str, version: str) -> bool:
        """Deploy a model version."""
        try:
            # Check if version exists
            version_dir = self.models_dir / "versions" / model_name / version
            if not version_dir.exists():
                logger.error(f"Version {version} of model {model_name} not found")
                return False
            
            # Create deployed directory
            deployed_dir = self.models_dir / "deployed" / model_name
            deployed_dir.mkdir(exist_ok=True, parents=True)
            
            # Clear existing deployed files
            for item in deployed_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            
            # Copy version files to deployed
            for item in version_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, deployed_dir)
                elif item.is_dir():
                    shutil.copytree(item, deployed_dir / item.name)
            
            # Update metadata
            metadata_path = deployed_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                metadata["deployed_at"] = datetime.now().isoformat()
                metadata["is_active"] = True
                
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            
            # Update database
            with get_db() as session:
                # Deactivate all versions of this model
                session.execute(
                    f"UPDATE model_versions SET is_active = 0 WHERE model_name = '{model_name}'"
                )
                
                # Activate this version
                session.execute(
                    f"UPDATE model_versions SET is_active = 1 WHERE model_name = '{model_name}' AND version = '{version}'"
                )
                
                session.commit()
            
            logger.info(f"Deployed model {model_name} version {version}")
            return True
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            return False
    
    def get_active_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the active model version."""
        try:
            with get_db() as session:
                query = select(ModelVersion).where(
                    ModelVersion.model_name == model_name,
                    ModelVersion.is_active == True
                )
                
                result = session.execute(query).scalars().first()
                
                if result:
                    return {
                        "model_name": result.model_name,
                        "version": result.version,
                        "uri": result.uri,
                        "metrics": result.metrics,
                        "parameters": result.parameters,
                        "created_at": result.created_at
                    }
                
                return None
        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """List all models in the registry."""
        try:
            versions_dir = self.models_dir / "versions"
            if not versions_dir.exists():
                return []
            
            models = [d.name for d in versions_dir.iterdir() if d.is_dir()]
            return sorted(models)
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        try:
            with get_db() as session:
                query = select(ModelVersion).where(
                    ModelVersion.model_name == model_name
                ).order_by(ModelVersion.created_at.desc())
                
                results = session.execute(query).scalars().all()
                
                versions = []
                for result in results:
                    versions.append({
                        "version": result.version,
                        "uri": result.uri,
                        "metrics": result.metrics,
                        "is_active": result.is_active,
                        "created_at": result.created_at
                    })
                
                return versions
        except Exception as e:
            logger.error(f"Error listing model versions: {e}")
            return []