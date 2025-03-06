"""
Rhetoric Intensity Inference Module

This module provides a simple interface for making predictions with the
rhetoric intensity model.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch

# Define paths
DEPLOY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rhetoric_intensity")

# Load feature names
with open(os.path.join(DEPLOY_DIR, "feature_names.json"), "r") as f:
    FEATURE_NAMES = json.load(f).get("feature_names", [])

# Check for model type
has_standard_model = os.path.exists(os.path.join(DEPLOY_DIR, "model.pkl"))
has_neural_network = os.path.exists(os.path.join(DEPLOY_DIR, "nn_model.pt"))

# Set up device for PyTorch
device = None
if has_neural_network:
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                           "cuda" if torch.cuda.is_available() else 
                           "cpu")

# Define neural network class (needs to match training)
class MetricNet(torch.nn.Module):
    def __init__(self, input_size):
        super(MetricNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(128, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fc3 = torch.nn.Linear(64, 32)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.fc4 = torch.nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))  # For binary classification
        return x

def load_models():
    """Load the trained models and scalers."""
    models = {}
    
    # Load standard model if available
    if has_standard_model:
        try:
            with open(os.path.join(DEPLOY_DIR, "model.pkl"), "rb") as f:
                models["standard"] = pickle.load(f)
            
            with open(os.path.join(DEPLOY_DIR, "scaler.pkl"), "rb") as f:
                models["standard_scaler"] = pickle.load(f)
        except Exception as e:
            print(f"Error loading standard model: {e}")
    
    # Load neural network if available
    if has_neural_network:
        try:
            # Load model metadata to get input size
            with open(os.path.join(DEPLOY_DIR, "nn_model_metadata.json"), "r") as f:
                nn_metadata = json.load(f)
            
            input_size = len(FEATURE_NAMES)
            
            # Initialize neural network
            nn_model = MetricNet(input_size).to(device)
            nn_model.load_state_dict(torch.load(
                os.path.join(DEPLOY_DIR, "nn_model.pt"),
                map_location=device
            ))
            nn_model.eval()
            
            models["neural_network"] = nn_model
            
            # Load neural network scaler
            with open(os.path.join(DEPLOY_DIR, "nn_scaler.pkl"), "rb") as f:
                models["nn_scaler"] = pickle.load(f)
        except Exception as e:
            print(f"Error loading neural network model: {e}")
    
    return models

# Load models at module import time
MODELS = load_models()

def predict(features, model_type="standard"):
    """
    Make a prediction with the rhetoric intensity model.
    
    Parameters:
    -----------
    features : pandas.DataFrame
        Features to use for prediction. Must contain required feature columns.
    model_type : str
        Either "standard" for scikit-learn model or "neural_network" for PyTorch model.
        
    Returns:
    --------
    numpy.ndarray
        Predicted rhetoric intensity scores.
    """
    if model_type not in ["standard", "neural_network"]:
        raise ValueError(f"Unknown model type: {model_type}. Use 'standard' or 'neural_network'.")
    
    if model_type == "standard" and "standard" not in MODELS:
        raise ValueError("Standard model not available.")
    
    if model_type == "neural_network" and "neural_network" not in MODELS:
        raise ValueError("Neural network model not available.")
    
    # Ensure features dataframe has required columns
    missing_features = [col for col in FEATURE_NAMES if col not in features.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select and order features
    features = features[FEATURE_NAMES]
    
    # Apply appropriate scaler
    if model_type == "standard":
        scaler = MODELS["standard_scaler"]
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predictions = MODELS["standard"].predict(features_scaled)
        
        # If model has predict_proba, use it
        if hasattr(MODELS["standard"], "predict_proba"):
            try:
                # Get probability of positive class
                predictions_proba = MODELS["standard"].predict_proba(features_scaled)[:, 1]
                return predictions_proba
            except:
                pass
            
        return predictions
    
    else:  # Neural network
        scaler = MODELS["nn_scaler"]
        features_scaled = scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        
        # Make prediction
        with torch.no_grad():
            predictions = MODELS["neural_network"](features_tensor).cpu().numpy()
        
        return predictions.reshape(-1)

def batch_predict(features_list, model_type="standard"):
    """
    Make predictions for a batch of feature sets.
    
    Parameters:
    -----------
    features_list : list of pandas.DataFrame
        List of feature dataframes.
    model_type : str
        Either "standard" for scikit-learn model or "neural_network" for PyTorch model.
        
    Returns:
    --------
    list
        List of predictions.
    """
    return [predict(features, model_type) for features in features_list]
