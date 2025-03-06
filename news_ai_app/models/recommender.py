"""
News recommendation models.

This module implements a hybrid recommendation system for news articles
combining collaborative filtering, content-based filtering, and knowledge graph embeddings.
"""
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class NewsEncoder(nn.Module):
    """
    News content encoder using transformer architecture.
    """
    
    def __init__(
        self,
        pretrained_model: str = "microsoft/mpnet-base",
        embedding_dim: int = 768,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the news encoder.
        
        Args:
            pretrained_model: Name of the pretrained transformer model to use.
            embedding_dim: Dimension of the output embeddings.
            dropout_rate: Dropout rate to use for regularization.
        """
        super().__init__()
        
        # Load pretrained transformer model
        self.transformer = AutoModel.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        
        # Additional layers
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass through the news encoder."""
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the hidden states
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Apply attention mechanism
        attention_weights = self.attention(hidden_states).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention to hidden states
        attended_output = torch.bmm(
            attention_weights.unsqueeze(1),
            hidden_states
        ).squeeze(1)
        
        # Apply dropout and final linear layer
        output = self.fc(self.dropout(attended_output))
        
        return output
    
    def encode_news(self, titles, abstracts):
        """
        Encode news articles into embedding vectors.
        
        Args:
            titles: List of news titles.
            abstracts: List of news abstracts.
            
        Returns:
            Tensor of news embeddings.
        """
        # Combine title and abstract
        full_texts = [f"{title} {abstract}" for title, abstract in zip(titles, abstracts)]
        
        # Tokenize
        inputs = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        
        return embeddings


class UserEncoder(nn.Module):
    """
    User encoder that aggregates users' reading history.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the user encoder.
        
        Args:
            embedding_dim: Dimension of the input and output embeddings.
            dropout_rate: Dropout rate to use for regularization.
        """
        super().__init__()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Additional layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, news_embeddings, history_mask=None):
        """
        Forward pass through the user encoder.
        
        Args:
            news_embeddings: Tensor of news embeddings from user's history.
                Shape: (batch_size, history_length, embedding_dim)
            history_mask: Boolean mask indicating which news are valid in the history.
                Shape: (batch_size, history_length)
                
        Returns:
            User embeddings tensor of shape (batch_size, embedding_dim)
        """
        # Apply attention to get weighted sum of news embeddings
        attention_weights = self.attention(news_embeddings).squeeze(-1)  # (batch_size, history_length)
        
        # Apply mask if provided
        if history_mask is not None:
            attention_weights = attention_weights.masked_fill(~history_mask, -1e9)
        
        # Apply softmax to get normalized weights
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights to news embeddings
        user_embedding = torch.bmm(
            attention_weights.unsqueeze(1),
            news_embeddings
        ).squeeze(1)  # (batch_size, embedding_dim)
        
        # Apply dropout and final transformation
        user_embedding = self.fc(self.dropout(user_embedding))
        
        return user_embedding


class KnowledgeGraphEncoder(nn.Module):
    """
    Knowledge graph encoder for incorporating entity and relation information.
    """
    
    def __init__(
        self,
        entity_embeddings: Dict[str, np.ndarray],
        relation_embeddings: Dict[str, np.ndarray],
        embedding_dim: int = 100,
        output_dim: int = 768
    ):
        """
        Initialize the knowledge graph encoder.
        
        Args:
            entity_embeddings: Dictionary mapping entity IDs to embedding vectors.
            relation_embeddings: Dictionary mapping relation IDs to embedding vectors.
            embedding_dim: Dimension of the input entity and relation embeddings.
            output_dim: Dimension of the output embeddings.
        """
        super().__init__()
        
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        
        # Transformation layers
        self.entity_transform = nn.Linear(embedding_dim, output_dim)
        self.relation_transform = nn.Linear(embedding_dim, output_dim)
        
        # Attention mechanism for combining entity embeddings
        self.attention = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
    
    def encode_entities(self, entity_ids: List[str]) -> torch.Tensor:
        """
        Encode a list of entity IDs into a single embedding vector.
        
        Args:
            entity_ids: List of entity IDs to encode.
            
        Returns:
            Tensor of entity embedding.
        """
        if not entity_ids:
            # Return zero vector if no entities
            return torch.zeros(1, self.entity_transform.out_features)
            
        # Get entity embeddings
        entity_vectors = [
            self.entity_embeddings.get(entity_id, np.zeros(self.entity_transform.in_features))
            for entity_id in entity_ids
        ]
        
        # Convert to tensor
        entity_tensors = torch.tensor(np.array(entity_vectors), dtype=torch.float32)
        
        # Apply transformation
        transformed_entities = self.entity_transform(entity_tensors)
        
        # Apply attention
        attention_weights = self.attention(transformed_entities).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Weighted sum
        entity_embedding = (transformed_entities * attention_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
        
        return entity_embedding
    
    def encode_relations(self, relation_ids: List[str]) -> torch.Tensor:
        """
        Encode a list of relation IDs into a single embedding vector.
        
        Args:
            relation_ids: List of relation IDs to encode.
            
        Returns:
            Tensor of relation embedding.
        """
        if not relation_ids:
            # Return zero vector if no relations
            return torch.zeros(1, self.relation_transform.out_features)
            
        # Get relation embeddings
        relation_vectors = [
            self.relation_embeddings.get(relation_id, np.zeros(self.relation_transform.in_features))
            for relation_id in relation_ids
        ]
        
        # Convert to tensor
        relation_tensors = torch.tensor(np.array(relation_vectors), dtype=torch.float32)
        
        # Apply transformation
        transformed_relations = self.relation_transform(relation_tensors)
        
        # Apply attention
        attention_weights = self.attention(transformed_relations).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Weighted sum
        relation_embedding = (transformed_relations * attention_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
        
        return relation_embedding


class HybridNewsRecommender(nn.Module):
    """
    Hybrid news recommendation model combining content, collaborative, and knowledge graph approaches.
    """
    
    def __init__(
        self,
        entity_embeddings: Dict[str, np.ndarray],
        relation_embeddings: Dict[str, np.ndarray],
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the hybrid news recommender.
        
        Args:
            entity_embeddings: Dictionary mapping entity IDs to embedding vectors.
            relation_embeddings: Dictionary mapping relation IDs to embedding vectors.
            embedding_dim: Dimension of the embeddings.
            hidden_dim: Dimension of the hidden layers.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        # Encoders
        self.news_encoder = NewsEncoder(embedding_dim=embedding_dim, dropout_rate=dropout_rate)
        self.user_encoder = UserEncoder(embedding_dim=embedding_dim, dropout_rate=dropout_rate)
        self.kg_encoder = KnowledgeGraphEncoder(
            entity_embeddings=entity_embeddings,
            relation_embeddings=relation_embeddings,
            output_dim=embedding_dim
        )
        
        # Fusion layers
        self.content_kg_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
    
    def encode_news_batch(
        self, 
        titles: List[str], 
        abstracts: List[str], 
        title_entities: List[List[Dict]], 
        abstract_entities: List[List[Dict]]
    ) -> torch.Tensor:
        """
        Encode a batch of news articles.
        
        Args:
            titles: List of news titles.
            abstracts: List of news abstracts.
            title_entities: List of lists of entity dictionaries from titles.
            abstract_entities: List of lists of entity dictionaries from abstracts.
            
        Returns:
            Tensor of encoded news articles.
        """
        # Get content embeddings
        content_embeddings = self.news_encoder.encode_news(titles, abstracts)
        
        # Process each news article's entities
        kg_embeddings = []
        for i in range(len(titles)):
            # Extract entity IDs from title and abstract
            title_entity_ids = [
                entity["WikidataId"] for entity in title_entities[i]
                if "WikidataId" in entity
            ]
            abstract_entity_ids = [
                entity["WikidataId"] for entity in abstract_entities[i]
                if "WikidataId" in entity
            ]
            
            # Combine entity IDs
            entity_ids = list(set(title_entity_ids + abstract_entity_ids))
            
            # Encode entities
            entity_embedding = self.kg_encoder.encode_entities(entity_ids)
            kg_embeddings.append(entity_embedding)
        
        # Stack KG embeddings
        kg_embeddings = torch.cat(kg_embeddings, dim=0)
        
        # Fuse content and KG embeddings
        combined_input = torch.cat([content_embeddings, kg_embeddings], dim=1)
        news_embeddings = self.content_kg_fusion(combined_input)
        
        return news_embeddings
    
    def encode_user(self, history_embeddings: torch.Tensor, history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a user based on their reading history.
        
        Args:
            history_embeddings: Tensor of news embeddings from user's history.
            history_mask: Boolean mask indicating which news are valid in the history.
            
        Returns:
            Tensor of user embeddings.
        """
        return self.user_encoder(history_embeddings, history_mask)
    
    def forward(
        self,
        user_history_embeddings: torch.Tensor,
        candidate_news_embeddings: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to calculate recommendation scores.
        
        Args:
            user_history_embeddings: Tensor of news embeddings from user's history.
                Shape: (batch_size, history_length, embedding_dim)
            candidate_news_embeddings: Tensor of candidate news embeddings.
                Shape: (batch_size, num_candidates, embedding_dim)
            history_mask: Boolean mask indicating which news are valid in the history.
                Shape: (batch_size, history_length)
                
        Returns:
            Tensor of recommendation scores.
        """
        # Encode user
        user_embedding = self.encode_user(user_history_embeddings, history_mask)  # (batch_size, embedding_dim)
        
        # Reshape for batch processing
        batch_size, num_candidates, embedding_dim = candidate_news_embeddings.shape
        user_embedding_expanded = user_embedding.unsqueeze(1).expand(-1, num_candidates, -1)
        
        # Concatenate user and news embeddings
        combined = torch.cat([
            user_embedding_expanded,
            candidate_news_embeddings
        ], dim=2)  # (batch_size, num_candidates, embedding_dim*2)
        
        # Reshape for prediction
        combined = combined.view(-1, embedding_dim * 2)  # (batch_size*num_candidates, embedding_dim*2)
        
        # Predict scores
        scores = self.predictor(combined).view(batch_size, num_candidates)  # (batch_size, num_candidates)
        
        return scores
    
    def get_recommendations(
        self,
        user_history: List[Dict],
        candidate_news: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Get recommendations for a user.
        
        Args:
            user_history: List of news articles from user's history.
            candidate_news: List of candidate news articles.
            top_k: Number of top recommendations to return.
            
        Returns:
            List of recommended news articles with scores.
        """
        # Prepare history data
        history_titles = [article["title"] for article in user_history]
        history_abstracts = [article.get("abstract", "") for article in user_history]
        history_title_entities = [article.get("title_entities", []) for article in user_history]
        history_abstract_entities = [article.get("abstract_entities", []) for article in user_history]
        
        # Prepare candidate data
        candidate_titles = [article["title"] for article in candidate_news]
        candidate_abstracts = [article.get("abstract", "") for article in candidate_news]
        candidate_title_entities = [article.get("title_entities", []) for article in candidate_news]
        candidate_abstract_entities = [article.get("abstract_entities", []) for article in candidate_news]
        
        # Encode history
        with torch.no_grad():
            history_embeddings = self.encode_news_batch(
                history_titles, 
                history_abstracts, 
                history_title_entities, 
                history_abstract_entities
            )
            
            # Encode candidates
            candidate_embeddings = self.encode_news_batch(
                candidate_titles, 
                candidate_abstracts, 
                candidate_title_entities, 
                candidate_abstract_entities
            )
            
            # Reshape tensors for model
            history_embeddings = history_embeddings.unsqueeze(0)  # (1, history_length, embedding_dim)
            candidate_embeddings = candidate_embeddings.unsqueeze(0)  # (1, num_candidates, embedding_dim)
            
            # Get scores
            scores = self.forward(history_embeddings, candidate_embeddings).squeeze(0)
            
            # Get top-k recommendations
            top_indices = torch.topk(scores, min(top_k, len(candidate_news))).indices.cpu().numpy()
            
        # Create recommendation results
        recommendations = []
        for i, idx in enumerate(top_indices):
            recommendations.append({
                **candidate_news[idx],
                "score": float(scores[idx]),
                "rank": i + 1
            })
        
        return recommendations


def load_pretrained_recommender(
    entity_embeddings: Dict[str, np.ndarray],
    relation_embeddings: Dict[str, np.ndarray],
    model_path: Optional[str] = None
) -> HybridNewsRecommender:
    """
    Load a pretrained recommender model.
    
    Args:
        entity_embeddings: Dictionary mapping entity IDs to embedding vectors.
        relation_embeddings: Dictionary mapping relation IDs to embedding vectors.
        model_path: Path to the pretrained model weights. If None, returns a new model.
        
    Returns:
        HybridNewsRecommender model.
    """
    # Create model
    model = HybridNewsRecommender(
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings
    )
    
    # Load weights if provided
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            logger.info(f"Loaded pretrained model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
    
    return model