"""
MLflow integration for experiment tracking and model registry.
"""
import os
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.pytorch
import logging
import json
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("mlflow_tracker")

class MLflowTracker:
    """MLflow tracking and model registry integration."""
    
    def __init__(self, tracking_uri=None, experiment_name=None):
        """Initialize MLflow tracker."""
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", 
            "file://" + str(Path(__file__).resolve().parent.parent / "mlruns")
        )
        self.experiment_name = experiment_name or os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", 
            "news_ai_experiments"
        )
        self.experiment_id = None
    
    def setup(self):
        """Set up MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
            else:
                self.experiment_id = mlflow.create_experiment(name=self.experiment_name)
            
            logger.info(f"MLflow tracking set up with experiment: {self.experiment_name} (ID: {self.experiment_id})")
            return True
        except Exception as e:
            logger.error(f"Error setting up MLflow tracking: {e}")
            return False
    
    def start_run(self, run_name=None, tags=None):
        """Start a new MLflow run."""
        try:
            if self.experiment_id is None:
                self.setup()
            
            active_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=tags
            )
            
            logger.info(f"Started MLflow run: {active_run.info.run_id}")
            return active_run
        except Exception as e:
            logger.error(f"Error starting MLflow run: {e}")
            return None
    
    def end_run(self):
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
            return True
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")
            return False
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to the current run."""
        try:
            for key, value in params.items():
                # Convert non-string values appropriately
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                mlflow.log_param(key, value)
            
            logger.info(f"Logged {len(params)} parameters")
            return True
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")
            return False
    
    def log_metrics(self, metrics: Dict[str, float], step=None):
        """Log metrics to the current run."""
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            
            logger.info(f"Logged {len(metrics)} metrics")
            return True
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
            return False
    
    def log_artifact(self, local_path):
        """Log an artifact to the current run."""
        try:
            mlflow.log_artifact(local_path)
            logger.info(f"Logged artifact: {local_path}")
            return True
        except Exception as e:
            logger.error(f"Error logging artifact: {e}")
            return False
    
    def log_sklearn_model(self, model, artifact_path="model", register_name=None):
        """Log a scikit-learn model."""
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path
            )
            
            logger.info(f"Logged scikit-learn model to {artifact_path}")
            
            # Register the model if a name is provided
            if register_name:
                self._register_model(artifact_path, register_name)
            
            return True
        except Exception as e:
            logger.error(f"Error logging scikit-learn model: {e}")
            return False
    
    def log_pytorch_model(self, model, artifact_path="model", register_name=None):
        """Log a PyTorch model."""
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path
            )
            
            logger.info(f"Logged PyTorch model to {artifact_path}")
            
            # Register the model if a name is provided
            if register_name:
                self._register_model(artifact_path, register_name)
            
            return True
        except Exception as e:
            logger.error(f"Error logging PyTorch model: {e}")
            return False
    
    def _register_model(self, artifact_path, register_name):
        """Register a model with the MLflow model registry."""
        try:
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/{artifact_path}"
            
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=register_name
            )
            
            logger.info(f"Registered model: {register_name} (version: {registered_model.version})")
            return registered_model
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None
    
    def load_model(self, run_id=None, model_name=None, version="production"):
        """Load a model from MLflow."""
        try:
            if run_id:
                # Load from a specific run
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.pyfunc.load_model(model_uri)
            elif model_name:
                # Load a registered model
                model_uri = f"models:/{model_name}/{version}"
                model = mlflow.pyfunc.load_model(model_uri)
            else:
                raise ValueError("Either run_id or model_name must be provided")
            
            logger.info(f"Loaded model from {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def promote_model_to_production(self, model_name, version):
        """Promote a model version to production."""
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            
            logger.info(f"Promoted {model_name} version {version} to Production")
            return True
        except Exception as e:
            logger.error(f"Error promoting model to production: {e}")
            return False