"""
Advanced text preprocessing module for News AI application.

This module provides comprehensive text preprocessing functionalities for NLP tasks:
- Tokenization and normalization
- Stop word removal
- Stemming and lemmatization
- Named entity recognition
- Text cleaning and standardization
"""
import re
import unicodedata
import string
import nltk
import spacy
import contractions
from typing import List, Dict, Tuple, Optional, Union, Any
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import yaml
import os
from pathlib import Path

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextProcessor:
    """Advanced text processor for NLP preprocessing in the News AI pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the text processor with configuration settings.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)['preprocessing']
        else:
            # Default configuration if not provided
            self.config = {
                'max_title_length': 64,
                'max_abstract_length': 512,
                'min_word_freq': 5,
                'max_vocab_size': 100000,
                'use_stemming': True,
                'use_lemmatization': True,
                'remove_stopwords': True,
                'use_spacy': True,
                'spacy_model': "en_core_web_sm"
            }
        
        # Initialize tools
        self.stemmer = PorterStemmer() if self.config['use_stemming'] else None
        self.lemmatizer = WordNetLemmatizer() if self.config['use_lemmatization'] else None
        self.stop_words = set(stopwords.words('english')) if self.config['remove_stopwords'] else set()
        
        # Load spaCy model if specified
        self.nlp = None
        if self.config['use_spacy']:
            try:
                self.nlp = spacy.load(self.config['spacy_model'])
                print(f"Loaded spaCy model: {self.config['spacy_model']}")
            except OSError:
                print(f"spaCy model {self.config['spacy_model']} not found. Downloading...")
                spacy.cli.download(self.config['spacy_model'])
                self.nlp = spacy.load(self.config['spacy_model'])
    
    def preprocess_text(self, text: str, 
                        remove_stopwords: bool = None, 
                        stem: bool = None, 
                        lemmatize: bool = None,
                        max_length: int = None) -> str:
        """
        Preprocess text with full pipeline of cleaning and normalization.
        
        Args:
            text: The input text to preprocess
            remove_stopwords: Whether to remove stopwords (overrides config)
            stem: Whether to apply stemming (overrides config)
            lemmatize: Whether to apply lemmatization (overrides config)
            max_length: Maximum text length (overrides config)
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Use parameter values if provided, otherwise fall back to config
        remove_stopwords = remove_stopwords if remove_stopwords is not None else self.config['remove_stopwords']
        stem = stem if stem is not None else self.config['use_stemming']
        lemmatize = lemmatize if lemmatize is not None else self.config['use_lemmatization']
        
        # Apply text cleaning
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords if specified
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming or lemmatization
        if stem and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Rejoin tokens
        processed_text = ' '.join(tokens)
        
        # Truncate if max_length is specified
        if max_length:
            processed_text = ' '.join(processed_text.split()[:max_length])
        
        return processed_text
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unnecessary characters and normalizing.
        
        Args:
            text: The input text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Expand contractions (e.g., "don't" -> "do not")
        text = contractions.fix(text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        return text
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text: The input text to analyze
            
        Returns:
            List of dictionaries containing entity information
        """
        if not text or not isinstance(text, str) or not self.nlp:
            return []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = [
            {
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'type': ent.label_,
                'type_description': spacy.explain(ent.label_)
            }
            for ent in doc.ents
        ]
        
        return entities
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text using spaCy.
        
        Args:
            text: The input text to analyze
            
        Returns:
            List of noun phrases
        """
        if not text or not isinstance(text, str) or not self.nlp:
            return []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        return noun_phrases
    
    def tokenize_for_model(self, text: str, max_length: int = None) -> List[str]:
        """
        Tokenize text for model input, with minimal preprocessing.
        
        Args:
            text: The input text to tokenize
            max_length: Maximum number of tokens to return
            
        Returns:
            List of tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        # Clean text with minimal processing
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Truncate if max_length is specified
        if max_length:
            tokens = tokens[:max_length]
        
        return tokens
    
    def preprocess_batch(self, texts: List[str], **kwargs) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts to preprocess
            **kwargs: Additional arguments to pass to preprocess_text
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text, **kwargs) for text in texts]


# Example usage
if __name__ == "__main__":
    # Load config from file
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "../../config/pipeline_config.yaml"
    
    processor = TextProcessor(config_path)
    
    sample_text = "This is a sample article about AI news recommendation\! It might contain some contractions like don't and can't."
    
    # Test preprocessing
    print("\nOriginal text:")
    print(sample_text)
    
    print("\nPreprocessed text:")
    print(processor.preprocess_text(sample_text))
    
    # Test entity extraction
    print("\nExtracted entities:")
    entities = processor.extract_entities(sample_text)
    for entity in entities:
        print(f"  - {entity['text']} ({entity['type']})")
    
    # Test noun phrase extraction
    print("\nNoun phrases:")
    noun_phrases = processor.extract_noun_phrases(sample_text)
    for phrase in noun_phrases:
        print(f"  - {phrase}")
