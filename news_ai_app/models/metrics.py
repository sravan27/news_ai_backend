"""
Advanced metrics calculation for news content analysis.

This module provides functionality to compute various metrics:
1. Political Influence Level
2. Rhetoric Intensity Scale  
3. Information Depth Score
4. Sentiment Analysis
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import os
import json

# Handle models directly without transformers dependencies
class SimpleModelWrapper:
    """Simple wrapper to simulate transformer models for demo purposes."""
    
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
    
    def __call__(self, **kwargs):
        """Simulate model forward pass."""
        batch_size = 1
        if "input_ids" in kwargs:
            batch_size = kwargs["input_ids"].shape[0]
        
        # Create dummy logits
        logits = torch.randn(batch_size, self.num_classes)
        
        # Return object with logits attribute
        class DummyOutput:
            def __init__(self, logits):
                self.logits = logits
                
        return DummyOutput(logits)

class AdvancedMetricsCalculator:
    """
    Calculate advanced metrics for news content analysis using simplified models.
    """
    
    def __init__(self):
        """Initialize the metrics calculator with necessary models."""
        # Use simplified model wrappers instead of loading from HuggingFace
        self.political_model = SimpleModelWrapper(num_classes=5)
        self.rhetoric_model = SimpleModelWrapper(num_classes=10)
        self.info_depth_model = SimpleModelWrapper(num_classes=10)
        
        # Simplified tokenizer simulator
        self.tokenizer = self._create_dummy_tokenizer()
    
    def _create_dummy_tokenizer(self):
        """Create a dummy tokenizer function."""
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
        
    def calculate_political_influence(self, text: str) -> float:
        """
        Calculate the political influence level of a text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A float between 0 and 1 representing the political influence level.
        """
        # Generate a deterministic but varied value based on text content
        # Using simplified approach for demo
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
    
    def calculate_rhetoric_intensity(self, text: str) -> float:
        """
        Calculate the rhetoric intensity of a text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A float between 0 and 1 representing the rhetoric intensity.
        """
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
    
    def calculate_information_depth(self, text: str) -> float:
        """
        Calculate the information depth score of a text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A float between 0 and 1 representing the information depth.
        """
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
    
    def calculate_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Calculate the sentiment of a text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A dictionary with sentiment label and score.
        """
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
    
    def calculate_all_metrics(self, text: str) -> Dict[str, Union[float, Dict]]:
        """
        Calculate all metrics for a given text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            Dictionary with all calculated metrics.
        """
        return {
            "political_influence": self.calculate_political_influence(text),
            "rhetoric_intensity": self.calculate_rhetoric_intensity(text),
            "information_depth": self.calculate_information_depth(text),
            "sentiment": self.calculate_sentiment(text)
        }
    
    def batch_calculate_metrics(self, texts: List[str]) -> List[Dict[str, Union[float, Dict]]]:
        """
        Calculate metrics for a batch of texts.
        
        Args:
            texts: List of texts to analyze.
            
        Returns:
            List of dictionaries with calculated metrics for each text.
        """
        return [self.calculate_all_metrics(text) for text in texts]


# Factory function to get the metrics calculator
def get_metrics_calculator() -> AdvancedMetricsCalculator:
    """Get an instance of the AdvancedMetricsCalculator."""
    return AdvancedMetricsCalculator()