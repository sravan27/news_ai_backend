"""
Text Feature Extractor for News AI application.

This module provides advanced text feature extraction capabilities:
- TF-IDF features
- Word embeddings (Word2Vec, GloVe)
- Transformer-based embeddings (BERT, RoBERTa)
- Traditional NLP features (readability scores, sentiment, etc.)
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import Counter
import re
import yaml
import os
import pickle
import json
import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, hstack, vstack

# For advanced features
import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# For transformer-based embeddings
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset

# Initialize NLTK resources
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


class TextFeaturesDataset(Dataset):
    """Optimized PyTorch dataset for efficient batch processing of text features."""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Cache mechanism to avoid duplicate tokenization
        self._cache = {}
        
        # Pre-tokenize a subset if not too large to warm up the tokenizer
        # This helps the tokenizer optimize its internal state
        if len(texts) < 1000:
            self.tokenizer.encode_batch(texts[:100])
            
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Use cache to avoid re-tokenizing the same text
        if idx in self._cache:
            return self._cache[idx]
        
        # Truncate very long texts first for efficiency
        if len(text) > self.max_length * 10:  # Rough estimate of char-to-token ratio
            text = text[:self.max_length * 20]  # Truncate while leaving margin for tokenization
        
        # Tokenize text with optimized settings
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False,  # Not needed for embedding generation
            return_attention_mask=True
        )
        
        # Remove batch dimension and extract only needed tensors
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        
        # Cache result for repeated accesses
        if len(self._cache) < 1000:  # Limit cache size to avoid memory issues
            self._cache[idx] = result
            
        return result


class TextFeatureExtractor:
    """Advanced text feature extractor for News AI pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the text feature extractor with configuration settings.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.config = config['features']['text_embeddings']
        else:
            # Default configuration if not provided
            self.config = {
                'model': 'sentence-transformers/all-MiniLM-L6-v2',
                'batch_size': 32,
                'max_length': 128,
                'use_gpu': True
            }
        
        # Initialize feature extractors
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.svd = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize transformer components
        self.transformer_tokenizer = None
        self.transformer_model = None
        self.device = self._get_device()
        
        print(f"Using device: {self.device}")
    
    def _get_device(self):
        """Determine the appropriate device for computations."""
        if self.config['use_gpu']:
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # For Apple Silicon (M1/M2)
                return torch.device('mps')
        
        return torch.device('cpu')
    
    def fit_tfidf(self, texts: List[str], max_features: int = 10000, min_df: int = 5) -> 'TextFeatureExtractor':
        """
        Fit TF-IDF vectorizer on text corpus.
        
        Args:
            texts: List of texts to fit on
            max_features: Maximum number of features (vocabulary size)
            min_df: Minimum document frequency
            
        Returns:
            Self for method chaining
        """
        print(f"Fitting TF-IDF vectorizer with {max_features} features...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.tfidf_vectorizer.fit(texts)
        print(f"Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return self
    
    def transform_tfidf(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts: List of texts to transform
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        
        return self.tfidf_vectorizer.transform(texts)
    
    def fit_bow(self, texts: List[str], max_features: int = 10000, min_df: int = 5) -> 'TextFeatureExtractor':
        """
        Fit Bag-of-Words vectorizer on text corpus.
        
        Args:
            texts: List of texts to fit on
            max_features: Maximum number of features (vocabulary size)
            min_df: Minimum document frequency
            
        Returns:
            Self for method chaining
        """
        print(f"Fitting BoW vectorizer with {max_features} features...")
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.count_vectorizer.fit(texts)
        print(f"Vocabulary size: {len(self.count_vectorizer.vocabulary_)}")
        
        return self
    
    def transform_bow(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts to Bag-of-Words features.
        
        Args:
            texts: List of texts to transform
            
        Returns:
            Sparse matrix of BoW features
        """
        if self.count_vectorizer is None:
            raise ValueError("BoW vectorizer not fitted. Call fit_bow first.")
        
        return self.count_vectorizer.transform(texts)
    
    def fit_lsa(self, tfidf_matrix: csr_matrix, n_components: int = 100) -> 'TextFeatureExtractor':
        """
        Fit LSA (Latent Semantic Analysis) on TF-IDF matrix.
        
        Args:
            tfidf_matrix: TF-IDF features to fit on
            n_components: Number of LSA components
            
        Returns:
            Self for method chaining
        """
        print(f"Fitting LSA with {n_components} components...")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf_matrix)
        print(f"Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}")
        
        return self
    
    def transform_lsa(self, tfidf_matrix: csr_matrix) -> np.ndarray:
        """
        Transform TF-IDF matrix to LSA features.
        
        Args:
            tfidf_matrix: TF-IDF features to transform
            
        Returns:
            Array of LSA features
        """
        if self.svd is None:
            raise ValueError("LSA not fitted. Call fit_lsa first.")
        
        return self.svd.transform(tfidf_matrix)
    
    def extract_readability_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract readability scores from texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with readability features
        """
        features = []
        
        for text in texts:
            text_features = {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'smog_index': textstat.smog_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
                'syllable_count': textstat.syllable_count(text),
                'lexicon_count': textstat.lexicon_count(text),
                'sentence_count': textstat.sentence_count(text),
                'avg_sentence_length': textstat.avg_sentence_length(text),
                'avg_syllables_per_word': textstat.avg_syllables_per_word(text),
                'avg_letter_per_word': textstat.avg_letter_per_word(text)
            }
            features.append(text_features)
        
        return pd.DataFrame(features)
    
    def extract_sentiment_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract sentiment features from texts using VADER.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with sentiment features
        """
        features = []
        
        for text in texts:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
            features.append(sentiment_scores)
        
        return pd.DataFrame(features)
    
    def extract_statistical_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract statistical features from texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with statistical features
        """
        features = []
        
        for text in texts:
            # Count characters, words, sentences
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.\!?]+', text))
            
            # Count capital letters, digits, special characters
            capital_count = sum(1 for c in text if c.isupper())
            digit_count = sum(1 for c in text if c.isdigit())
            special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
            
            # Calculate ratios
            capital_ratio = capital_count / char_count if char_count > 0 else 0
            digit_ratio = digit_count / char_count if char_count > 0 else 0
            special_char_ratio = special_char_count / char_count if char_count > 0 else 0
            
            # Average word length
            avg_word_length = char_count / word_count if word_count > 0 else 0
            
            # Extract unique words and their count
            words = text.lower().split()
            unique_words = set(words)
            unique_word_count = len(unique_words)
            unique_word_ratio = unique_word_count / word_count if word_count > 0 else 0
            
            text_features = {
                'char_count': char_count,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'capital_count': capital_count,
                'digit_count': digit_count,
                'special_char_count': special_char_count,
                'capital_ratio': capital_ratio,
                'digit_ratio': digit_ratio,
                'special_char_ratio': special_char_ratio,
                'avg_word_length': avg_word_length,
                'unique_word_count': unique_word_count,
                'unique_word_ratio': unique_word_ratio
            }
            features.append(text_features)
        
        return pd.DataFrame(features)
    
    def load_transformer_model(self, model_name: Optional[str] = None) -> None:
        """
        Load transformer model and tokenizer for embedding extraction.
        
        Args:
            model_name: Name of the pre-trained model to load (default: from config)
        """
        model_name = model_name or self.config['model']
        print(f"Loading transformer model: {model_name}")
        
        # Load tokenizer and model
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer_model = AutoModel.from_pretrained(model_name)
        
        # Move model to device
        self.transformer_model.to(self.device)
        
        # Set model to evaluation mode
        self.transformer_model.eval()
    
    def extract_transformer_embeddings(self, texts: List[str], batch_size: Optional[int] = None, 
                                      max_length: Optional[int] = None, 
                                      pooling_strategy: str = 'mean') -> np.ndarray:
        """
        Extract embeddings from texts using a transformer model.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            pooling_strategy: Strategy for pooling token embeddings ('mean', 'cls', or 'max')
            
        Returns:
            Array of embeddings
        """
        import multiprocessing
        import os
        from functools import partial
        
        if self.transformer_model is None or self.transformer_tokenizer is None:
            self.load_transformer_model()
        
        # Use parameters from config if not specified - optimize for M2 Max
        batch_size = batch_size or min(128, self.config.get('batch_size', 32))  # Larger batch size for M2 Max
        max_length = max_length or self.config.get('max_length', 128)
        
        # Set num_workers based on available CPU cores
        num_workers = min(4, max(1, multiprocessing.cpu_count() // 2))
        
        # Optimize performance with environment variables
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        
        print(f"Processing embeddings with batch_size={batch_size}, workers={num_workers}, device={self.device}")
        
        # Create dataset and optimized dataloader with more workers
        dataset = TextFeaturesDataset(texts, self.transformer_tokenizer, max_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type != 'cpu' else False
        )
        
        # Function to apply pooling (for cleaner code)
        def apply_pooling(last_hidden_state, attention_mask, strategy):
            if strategy == 'mean':
                # Mean pooling: average over tokens, considering attention mask
                token_embeddings = last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            
            elif strategy == 'cls':
                # CLS token pooling: use the first token representation
                return last_hidden_state[:, 0]
            
            elif strategy == 'max':
                # Max pooling: take maximum over tokens, considering attention mask
                token_embeddings = last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                token_embeddings[mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                return torch.max(token_embeddings, 1)[0]
            
            else:
                raise ValueError(f"Unknown pooling strategy: {strategy}")
                
        # Extract embeddings with memory optimization
        # Use a list for accumulating without repeatedly copying large arrays
        total_samples = len(texts)
        embeddings = []
        
        # Add progress tracking
        print(f"Generating embeddings for {total_samples} texts")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass with optimizations for speed
                outputs = self.transformer_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Get last hidden states
                last_hidden_state = outputs.last_hidden_state
                
                # Apply pooling strategy
                batch_embeddings = apply_pooling(last_hidden_state, attention_mask, pooling_strategy)
                
                # Move to CPU and convert to numpy efficiently (avoid unnecessary copies)
                # Move to CPU in chunks to avoid OOM on transfer
                embeddings.append(batch_embeddings.cpu().numpy())
                
                # Report progress
                if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
                    processed = min((batch_idx + 1) * batch_size, total_samples)
                    print(f"Processed {processed}/{total_samples} texts ({processed/total_samples:.1%})")
        
        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)
        
        # Clean up to prevent memory leaks
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        return all_embeddings
    
    def save(self, output_dir: str) -> None:
        """
        Save the fitted feature extractors to disk.
        
        Args:
            output_dir: Directory to save the feature extractors
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vectorizers and transformers
        if self.tfidf_vectorizer is not None:
            with open(output_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        if self.count_vectorizer is not None:
            with open(output_dir / 'count_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.count_vectorizer, f)
        
        if self.svd is not None:
            with open(output_dir / 'svd.pkl', 'wb') as f:
                pickle.dump(self.svd, f)
        
        # Save configuration
        with open(output_dir / 'feature_extractor_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Feature extractors saved to {output_dir}")
    
    def load(self, model_dir: str) -> 'TextFeatureExtractor':
        """
        Load saved feature extractors from disk.
        
        Args:
            model_dir: Directory containing saved feature extractors
            
        Returns:
            Self for method chaining
        """
        model_dir = Path(model_dir)
        
        # Load vectorizers and transformers
        if (model_dir / 'tfidf_vectorizer.pkl').exists():
            with open(model_dir / 'tfidf_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        
        if (model_dir / 'count_vectorizer.pkl').exists():
            with open(model_dir / 'count_vectorizer.pkl', 'rb') as f:
                self.count_vectorizer = pickle.load(f)
        
        if (model_dir / 'svd.pkl').exists():
            with open(model_dir / 'svd.pkl', 'rb') as f:
                self.svd = pickle.load(f)
        
        # Load configuration
        if (model_dir / 'feature_extractor_config.json').exists():
            with open(model_dir / 'feature_extractor_config.json', 'r') as f:
                self.config = json.load(f)
        
        print(f"Feature extractors loaded from {model_dir}")
        
        return self


# Example usage
if __name__ == "__main__":
    # Load config from file
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "../../config/pipeline_config.yaml"
    
    # Initialize feature extractor
    feature_extractor = TextFeatureExtractor(config_path)
    
    # Sample texts
    sample_texts = [
        "This is a sample article about AI news recommendation. It contains important information.",
        "Breaking news: New advances in machine learning have revolutionized the industry!",
        "Sports update: Local team wins championship in dramatic fashion."
    ]
    
    # Extract transformer embeddings
    embeddings = feature_extractor.extract_transformer_embeddings(sample_texts)
    print(f"\nTransformer embeddings shape: {embeddings.shape}")
    
    # Extract readability features
    readability_features = feature_extractor.extract_readability_features(sample_texts)
    print("\nReadability features:")
    print(readability_features.head())
    
    # Extract sentiment features
    sentiment_features = feature_extractor.extract_sentiment_features(sample_texts)
    print("\nSentiment features:")
    print(sentiment_features.head())
    
    # Extract statistical features
    statistical_features = feature_extractor.extract_statistical_features(sample_texts)
    print("\nStatistical features:")
    print(statistical_features.head())
