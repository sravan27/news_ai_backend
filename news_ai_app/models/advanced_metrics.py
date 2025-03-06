"""
Advanced metrics calculation using optimized machine learning models.

This module provides high-performance implementations of news metrics:
1. Political Influence Level
2. Rhetoric Intensity Scale
3. Information Depth Score
4. Sentiment Analysis

Using models trained with bagging, boosting, stacking, and neural networks.
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from news_ai_app.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Auto-detect device for PyTorch
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")

# Paths
DEPLOYED_PATH = Path(settings.model.silicon_models_path)
if not DEPLOYED_PATH.exists():
    # Fallback to default paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DEPLOYED_PATH = PROJECT_ROOT / "ml_pipeline" / "models" / "deployed"

# Neural network architecture definition
class DeepMetricsRegressor(nn.Module):
    """Advanced neural network for news metrics regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())  # Output in [0,1] range for all metrics
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ResidualBlock(nn.Module):
    """Residual block for neural networks."""
    
    def __init__(self, dim, dropout_rate=0.3, use_batch_norm=True):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity()
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        return self.relu(out)

class DeepResidualRegressor(nn.Module):
    """Deep residual network for metrics regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0,1] range
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output(x)

class MetricModel:
    """Base class for advanced metric models."""
    
    def __init__(self, metric_name: str):
        """
        Initialize a metric model.
        
        Args:
            metric_name: Name of the metric
        """
        self.metric_name = metric_name
        self.model_path = DEPLOYED_PATH / metric_name
        
        # Check if model exists
        if not self.model_path.exists():
            raise ValueError(f"Model for {metric_name} not found at {self.model_path}")
        
        # Load feature names
        try:
            with open(self.model_path / "feature_names.json", 'r') as f:
                self.feature_names = json.load(f).get('feature_names', [])
        except Exception as e:
            logger.error(f"Error loading feature names for {metric_name}: {e}")
            self.feature_names = []
        
        # Load model metadata
        try:
            with open(self.model_path / "model_metadata.json", 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading model metadata for {metric_name}: {e}")
            self.metadata = {}
        
        # Determine model type
        self.model_type = self.metadata.get('model_type', 'unknown')
        
        # Load model
        if self.model_type == 'nn':
            self._load_nn_model()
        else:
            self._load_sklearn_model()
    
    def _load_nn_model(self):
        """Load a neural network model."""
        nn_model_path = self.model_path / "nn_model.pt"
        nn_scaler_path = self.model_path / "nn_scaler.pkl"
        
        if not nn_model_path.exists() or not nn_scaler_path.exists():
            logger.error(f"Neural network model files for {self.metric_name} not found")
            return
        
        # Load scaler
        with open(nn_scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Determine architecture
        architecture = self.metadata.get('architecture', 'DeepMetricsRegressor')
        if 'input_dim' not in self.metadata:
            input_dim = len(self.feature_names)
        else:
            input_dim = self.metadata['input_dim']
        
        # Create model
        if architecture == 'DeepResidualRegressor':
            self.model = DeepResidualRegressor(input_dim=input_dim).to(device)
        else:
            self.model = DeepMetricsRegressor(input_dim=input_dim).to(device)
        
        # Load weights
        self.model.load_state_dict(torch.load(nn_model_path, map_location=device))
        self.model.eval()
    
    def _load_sklearn_model(self):
        """Load a scikit-learn model."""
        model_path = self.model_path / "model.pkl"
        scaler_path = self.model_path / "scaler.pkl"
        
        if not model_path.exists():
            logger.error(f"Scikit-learn model file for {self.metric_name} not found")
            return
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler if exists
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None
    
    def create_features(self, title: str, abstract: str = "", category: str = "") -> pd.DataFrame:
        """
        Create features for prediction.
        
        Args:
            title: Article title
            abstract: Article abstract (optional)
            category: Article category (optional)
            
        Returns:
            DataFrame with features
        """
        # Create feature dataframe
        features = {}
        
        # Basic features
        if category:
            # One-hot encode categories
            category = category.lower()
            for cat in ['politics', 'sports', 'entertainment', 'technology', 'health', 'business']:
                features[f'category_{cat}'] = 1 if category == cat else 0
        
        # Text length features
        features['title_length'] = len(title)
        features['title_word_count'] = len(title.split())
        
        if abstract:
            features['abstract_length'] = len(abstract)
            features['abstract_word_count'] = len(abstract.split())
        else:
            features['abstract_length'] = 0
            features['abstract_word_count'] = 0
        
        # Metric-specific features
        if self.metric_name == "political_influence":
            # Political terms
            political_terms = ['president', 'government', 'congress', 'law', 'policy', 'election',
                            'vote', 'democrat', 'republican', 'senator', 'representative']
            
            # Count political terms in title and abstract
            for term in political_terms:
                features[f'title_contains_{term}'] = 1 if term.lower() in title.lower() else 0
                features[f'abstract_contains_{term}'] = 1 if abstract and term.lower() in abstract.lower() else 0
        
        elif self.metric_name == "rhetoric_intensity":
            # Rhetoric markers
            rhetoric_markers = ['!', '?', '...', 'very', 'extremely', 'incredibly', 'absolutely', 'utterly']
            
            # Count rhetoric markers
            for marker in rhetoric_markers:
                features[f'title_contains_{marker}'] = 1 if marker in title else 0
                features[f'abstract_contains_{marker}'] = 1 if abstract and marker in abstract else 0
            
            # Count questions and exclamations
            features['title_exclamation_count'] = title.count('!')
            features['title_question_count'] = title.count('?')
            
            if abstract:
                features['abstract_exclamation_count'] = abstract.count('!')
                features['abstract_question_count'] = abstract.count('?')
            else:
                features['abstract_exclamation_count'] = 0
                features['abstract_question_count'] = 0
        
        elif self.metric_name == "information_depth":
            # Unique word ratio
            title_words = title.lower().split()
            features['title_unique_word_ratio'] = len(set(title_words)) / len(title_words) if title_words else 0
            
            if abstract:
                abstract_words = abstract.lower().split()
                features['abstract_unique_word_ratio'] = len(set(abstract_words)) / len(abstract_words) if abstract_words else 0
            else:
                features['abstract_unique_word_ratio'] = 0
            
            # Count numbers
            features['title_number_count'] = sum(c.isdigit() for c in title)
            features['abstract_number_count'] = sum(c.isdigit() for c in abstract) if abstract else 0
        
        elif self.metric_name == "sentiment":
            # Positive and negative terms
            positive_words = ["good", "great", "excellent", "amazing", "wonderful", 
                            "happy", "joy", "success", "beautiful", "love", "positive",
                            "achieve", "benefit", "win", "victory", "celebrate"]
            
            negative_words = ["bad", "terrible", "awful", "horrible", "poor", 
                            "sad", "hate", "failure", "ugly", "angry", "negative",
                            "loss", "damage", "threat", "crisis", "problem"]
            
            # Count positive words
            for word in positive_words[:5]:  # Limit to top 5
                features[f'title_contains_{word}'] = 1 if word.lower() in title.lower() else 0
                features[f'abstract_contains_{word}'] = 1 if abstract and word.lower() in abstract.lower() else 0
            
            # Count negative words
            for word in negative_words[:5]:  # Limit to top 5
                features[f'title_contains_{word}'] = 1 if word.lower() in title.lower() else 0
                features[f'abstract_contains_{word}'] = 1 if abstract and word.lower() in abstract.lower() else 0
            
            # Overall counts
            features['title_positive_count'] = sum(1 for word in positive_words if word.lower() in title.lower())
            features['title_negative_count'] = sum(1 for word in negative_words if word.lower() in title.lower())
            
            if abstract:
                features['abstract_positive_count'] = sum(1 for word in positive_words if word.lower() in abstract.lower())
                features['abstract_negative_count'] = sum(1 for word in negative_words if word.lower() in abstract.lower())
            else:
                features['abstract_positive_count'] = 0
                features['abstract_negative_count'] = 0
            
            # Sentiment ratio
            features['title_sentiment_ratio'] = (
                features['title_positive_count'] - features['title_negative_count']
            ) / (features['title_positive_count'] + features['title_negative_count'] + 1)
            
            if abstract:
                features['abstract_sentiment_ratio'] = (
                    features['abstract_positive_count'] - features['abstract_negative_count']
                ) / (features['abstract_positive_count'] + features['abstract_negative_count'] + 1)
            else:
                features['abstract_sentiment_ratio'] = 0
        
        # Create dataframe
        feature_df = pd.DataFrame([features])
        
        # Keep only the features required by the model
        missing_features = [col for col in self.feature_names if col not in feature_df.columns]
        
        # Add missing features with zeros
        for col in missing_features:
            feature_df[col] = 0
        
        # Reorder columns to match feature names
        feature_df = feature_df[self.feature_names]
        
        return feature_df
    
    def predict(self, title: str, abstract: str = "", category: str = "") -> float:
        """
        Make a prediction for the metric.
        
        Args:
            title: Article title
            abstract: Article abstract (optional)
            category: Article category (optional)
            
        Returns:
            Predicted metric value (0 to 1)
        """
        # Create features
        feature_df = self.create_features(title, abstract, category)
        
        # Make prediction
        if self.model_type == 'nn':
            # Scale features
            X_scaled = self.scaler.transform(feature_df)
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            
            # Get prediction
            with torch.no_grad():
                self.model.eval()
                pred = self.model(X_tensor).cpu().numpy().item()
        else:
            # Scale if needed
            if hasattr(self, 'scaler') and self.scaler is not None:
                X_scaled = self.scaler.transform(feature_df)
                pred = self.model.predict(X_scaled)[0]
            else:
                pred = self.model.predict(feature_df)[0]
        
        # Ensure result is in [0, 1] range
        pred = max(0.0, min(1.0, pred))
        
        return pred

class AdvancedMetricsCalculator:
    """
    Calculate advanced metrics for news content analysis using machine learning models.
    """
    
    def __init__(self):
        """Initialize the metrics calculator with optimized models."""
        # Try to load advanced models
        try:
            self.political_model = MetricModel("political_influence")
            self.rhetoric_model = MetricModel("rhetoric_intensity")
            self.info_depth_model = MetricModel("information_depth")
            self.sentiment_model = MetricModel("sentiment")
            
            # Using advanced models
            self.using_advanced_models = True
            logger.info("Using advanced ML models for metrics calculation")
        except Exception as e:
            # Fallback to simple model wrappers
            logger.warning(f"Error loading advanced models: {e}")
            logger.warning("Falling back to simple models for metrics calculation")
            
            # Use simplified model wrappers
            from .metrics import SimpleModelWrapper
            
            self.political_model = SimpleModelWrapper(num_classes=5)
            self.rhetoric_model = SimpleModelWrapper(num_classes=10)
            self.info_depth_model = SimpleModelWrapper(num_classes=10)
            
            # Simplified tokenizer simulator
            self.tokenizer = self._create_dummy_tokenizer()
            
            # Not using advanced models
            self.using_advanced_models = False
    
    def _create_dummy_tokenizer(self):
        """Create a dummy tokenizer function for simple models."""
        def tokenizer_func(text, return_tensors="pt", truncation=True, max_length=512):
            # Convert text to a simple token ID representation
            token_ids = [hash(word) % 50000 for word in text.split()]
            
            # Truncate if needed
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                
            # Create tensor
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor([token_ids]),
                    "attention_mask": torch.ones(1, len(token_ids))
                }
            return token_ids
        
        return tokenizer_func
        
    def calculate_political_influence(self, text: str, category: str = "") -> float:
        """
        Calculate the political influence level of a text.
        
        Args:
            text: The text to analyze.
            category: The news category (optional).
            
        Returns:
            A float between 0 and 1 representing the political influence level.
        """
        if self.using_advanced_models:
            # Split text into title and abstract (if possible)
            parts = text.split("\n", 1)
            title = parts[0]
            abstract = parts[1] if len(parts) > 1 else ""
            
            return self.political_model.predict(title, abstract, category)
        else:
            # Fallback to simple model
            # Generate a deterministic but varied value based on text content
            seed_value = sum(ord(c) for c in text[:100])
            torch.manual_seed(seed_value)
            
            # Tokenize text
            inputs = self.tokenizer(text)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.political_model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1).squeeze().numpy()
            
            # For now, use a simplified approach - map the highest class probability to [0,1]
            political_influence = float(scores.max())
            
            # Normalize to [0, 1] range and make more deterministic based on text
            base_value = (seed_value % 80) / 100.0 + 0.1  # Range 0.1-0.9
            return (base_value + political_influence) / 2
    
    def calculate_rhetoric_intensity(self, text: str, category: str = "") -> float:
        """
        Calculate the rhetoric intensity of a text.
        
        Args:
            text: The text to analyze.
            category: The news category (optional).
            
        Returns:
            A float between 0 and 1 representing the rhetoric intensity.
        """
        if self.using_advanced_models:
            # Split text into title and abstract (if possible)
            parts = text.split("\n", 1)
            title = parts[0]
            abstract = parts[1] if len(parts) > 1 else ""
            
            return self.rhetoric_model.predict(title, abstract, category)
        else:
            # Fallback to simple model
            # Generate a deterministic but varied value based on text content
            seed_value = sum(ord(c) for c in text[:100])
            torch.manual_seed(seed_value + 42)  # Different seed from political influence
            
            # Tokenize text
            inputs = self.tokenizer(text)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.rhetoric_model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1).squeeze().numpy()
            
            # Calculate rhetoric intensity as weighted average of class scores
            weights = np.linspace(0.1, 1.0, len(scores))
            rhetoric_intensity = float(np.sum(scores * weights))
            
            # Make more deterministic based on text properties
            # Look for rhetoric markers like exclamation marks, question marks, etc.
            rhetoric_markers = text.count('!') + text.count('?') + text.count(':') + text.count(';')
            rhetoric_factor = min(rhetoric_markers / 10.0, 1.0)  # Cap at 1.0
            
            # Combine model output with text properties
            return min(max((rhetoric_intensity + rhetoric_factor) / 2.0, 0.0), 1.0)
    
    def calculate_information_depth(self, text: str, category: str = "") -> float:
        """
        Calculate the information depth score of a text.
        
        Args:
            text: The text to analyze.
            category: The news category (optional).
            
        Returns:
            A float between 0 and 1 representing the information depth.
        """
        if self.using_advanced_models:
            # Split text into title and abstract (if possible)
            parts = text.split("\n", 1)
            title = parts[0]
            abstract = parts[1] if len(parts) > 1 else ""
            
            return self.info_depth_model.predict(title, abstract, category)
        else:
            # Fallback to simple model
            # Generate a deterministic but varied value based on text content
            seed_value = sum(ord(c) for c in text[:100])
            torch.manual_seed(seed_value + 100)  # Different seed from other metrics
            
            # Tokenize text
            inputs = self.tokenizer(text)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.info_depth_model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1).squeeze().numpy()
            
            # Calculate information depth as weighted average of class scores
            weights = np.linspace(0.1, 1.0, len(scores))
            info_depth = float(np.sum(scores * weights))
            
            # Use text properties for deterministic components
            # Check text length, unique words, etc. as proxies for information density
            words = text.split()
            unique_words = len(set(words))
            unique_ratio = unique_words / len(words) if words else 0
            
            # Longer texts with more unique words tend to have more information
            text_factor = min(len(words) / 200.0, 1.0) * 0.5 + unique_ratio * 0.5
            
            # Combine model output with text properties
            return min(max((info_depth + text_factor) / 2.0, 0.0), 1.0)
    
    def calculate_sentiment(self, text: str, category: str = "") -> Dict[str, Union[str, float]]:
        """
        Calculate the sentiment of a text.
        
        Args:
            text: The text to analyze.
            category: The news category (optional).
            
        Returns:
            A dictionary with sentiment label and score.
        """
        if self.using_advanced_models:
            # Split text into title and abstract (if possible)
            parts = text.split("\n", 1)
            title = parts[0]
            abstract = parts[1] if len(parts) > 1 else ""
            
            # Get sentiment score (0-1)
            score = self.sentiment_model.predict(title, abstract, category)
            
            # Convert to label
            if score < 0.4:
                label = "negative"
            elif score > 0.6:
                label = "positive"
            else:
                label = "neutral"
            
            return {
                "label": label,
                "score": float(score)
            }
        else:
            # Fallback to simple model
            # Generate deterministic sentiment based on text characteristics
            seed_value = sum(ord(c) for c in text[:100])
            np.random.seed(seed_value)
            
            # Simple sentiment analysis based on text properties
            # Check for positive and negative words
            positive_words = ["good", "great", "excellent", "amazing", "wonderful", 
                             "happy", "joy", "success", "beautiful", "love"]
            negative_words = ["bad", "terrible", "awful", "horrible", "poor", 
                             "sad", "hate", "failure", "ugly", "angry"]
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            # Calculate sentiment score
            if pos_count > neg_count:
                label = "positive"
                score = 0.5 + min(pos_count / 20.0, 0.5)  # 0.5-1.0 range
            elif neg_count > pos_count:
                label = "negative"
                score = 0.5 + min(neg_count / 20.0, 0.5)  # 0.5-1.0 range
            else:
                # If tie or no sentiment words, use a random but deterministic value
                label = "neutral" if np.random.random() < 0.5 else ("positive" if np.random.random() < 0.5 else "negative")
                score = 0.4 + np.random.random() * 0.3  # 0.4-0.7 range
            
            return {
                "label": label,
                "score": float(score)
            }
    
    def calculate_all_metrics(self, text: str, category: str = "") -> Dict[str, Union[float, Dict]]:
        """
        Calculate all metrics for a given text.
        
        Args:
            text: The text to analyze.
            category: The news category (optional).
            
        Returns:
            Dictionary with all calculated metrics.
        """
        return {
            "political_influence": self.calculate_political_influence(text, category),
            "rhetoric_intensity": self.calculate_rhetoric_intensity(text, category),
            "information_depth": self.calculate_information_depth(text, category),
            "sentiment": self.calculate_sentiment(text, category)
        }
    
    def batch_calculate_metrics(self, texts: List[str], categories: List[str] = None) -> List[Dict[str, Union[float, Dict]]]:
        """
        Calculate metrics for a batch of texts.
        
        Args:
            texts: List of texts to analyze.
            categories: List of news categories (optional).
            
        Returns:
            List of dictionaries with calculated metrics for each text.
        """
        if categories is None:
            categories = [""] * len(texts)
        
        return [self.calculate_all_metrics(text, category) 
                for text, category in zip(texts, categories)]


# Factory function to get the metrics calculator
def get_advanced_metrics_calculator() -> AdvancedMetricsCalculator:
    """Get an instance of the AdvancedMetricsCalculator."""
    return AdvancedMetricsCalculator()