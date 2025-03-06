"""
Unit tests for news recommender.
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from news_ai_app.models.recommender import (HybridNewsRecommender,
                                           KnowledgeGraphEncoder,
                                           NewsEncoder, UserEncoder,
                                           load_pretrained_recommender)


class TestNewsEncoder(unittest.TestCase):
    """Test cases for NewsEncoder."""
    
    def setUp(self):
        """Set up test case."""
        # Mock the transformer model and tokenizer
        self.mock_transformer = MagicMock()
        self.mock_tokenizer = MagicMock()
        
        # Create a patch for AutoModel and AutoTokenizer
        self.auto_model_patcher = patch("news_ai_app.models.recommender.AutoModel")
        self.auto_tokenizer_patcher = patch("news_ai_app.models.recommender.AutoTokenizer")
        
        # Start the patches
        self.mock_auto_model = self.auto_model_patcher.start()
        self.mock_auto_tokenizer = self.auto_tokenizer_patcher.start()
        
        # Set up return values for the mocks
        self.mock_auto_model.from_pretrained.return_value = self.mock_transformer
        self.mock_auto_tokenizer.from_pretrained.return_value = self.mock_tokenizer
        
        # Create the encoder
        self.encoder = NewsEncoder(embedding_dim=768)
    
    def tearDown(self):
        """Tear down test case."""
        # Stop the patches
        self.auto_model_patcher.stop()
        self.auto_tokenizer_patcher.stop()
    
    def test_init(self):
        """Test initialization."""
        # Check that the transformer and tokenizer were initialized correctly
        self.mock_auto_model.from_pretrained.assert_called_once()
        self.mock_auto_tokenizer.from_pretrained.assert_called_once()
        
        # Check that the encoder has the correct attributes
        self.assertEqual(self.encoder.transformer, self.mock_transformer)
        self.assertEqual(self.encoder.tokenizer, self.mock_tokenizer)
        self.assertIsInstance(self.encoder.attention, nn.Sequential)
        self.assertIsInstance(self.encoder.dropout, nn.Dropout)
        self.assertIsInstance(self.encoder.fc, nn.Linear)
    
    def test_forward(self):
        """Test forward pass."""
        # Mock input tensors
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10, dtype=torch.bool)
        
        # Mock transformer output
        output = MagicMock()
        output.last_hidden_state = torch.randn(2, 10, 768)
        self.mock_transformer.return_value = output
        
        # Call forward
        result = self.encoder.forward(input_ids, attention_mask)
        
        # Check that transformer was called with the correct arguments
        self.mock_transformer.assert_called_once_with(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Check that the result has the correct shape
        self.assertEqual(result.shape, (2, 768))


class TestUserEncoder(unittest.TestCase):
    """Test cases for UserEncoder."""
    
    def setUp(self):
        """Set up test case."""
        self.encoder = UserEncoder(embedding_dim=768)
    
    def test_init(self):
        """Test initialization."""
        # Check that the encoder has the correct attributes
        self.assertIsInstance(self.encoder.attention, nn.Sequential)
        self.assertIsInstance(self.encoder.dropout, nn.Dropout)
        self.assertIsInstance(self.encoder.fc, nn.Linear)
    
    def test_forward(self):
        """Test forward pass."""
        # Create input tensor
        news_embeddings = torch.randn(2, 5, 768)
        
        # Call forward
        result = self.encoder.forward(news_embeddings)
        
        # Check that the result has the correct shape
        self.assertEqual(result.shape, (2, 768))
    
    def test_forward_with_mask(self):
        """Test forward pass with mask."""
        # Create input tensor and mask
        news_embeddings = torch.randn(2, 5, 768)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, False, False]
        ])
        
        # Call forward
        result = self.encoder.forward(news_embeddings, mask)
        
        # Check that the result has the correct shape
        self.assertEqual(result.shape, (2, 768))


class TestKnowledgeGraphEncoder(unittest.TestCase):
    """Test cases for KnowledgeGraphEncoder."""
    
    def setUp(self):
        """Set up test case."""
        # Create mock entity and relation embeddings
        self.entity_embeddings = {
            "Q1": np.random.rand(100),
            "Q2": np.random.rand(100),
            "Q3": np.random.rand(100)
        }
        
        self.relation_embeddings = {
            "P1": np.random.rand(100),
            "P2": np.random.rand(100)
        }
        
        # Create the encoder
        self.encoder = KnowledgeGraphEncoder(
            entity_embeddings=self.entity_embeddings,
            relation_embeddings=self.relation_embeddings,
            embedding_dim=100,
            output_dim=768
        )
    
    def test_init(self):
        """Test initialization."""
        # Check that the encoder has the correct attributes
        self.assertEqual(self.encoder.entity_embeddings, self.entity_embeddings)
        self.assertEqual(self.encoder.relation_embeddings, self.relation_embeddings)
        self.assertIsInstance(self.encoder.entity_transform, nn.Linear)
        self.assertIsInstance(self.encoder.relation_transform, nn.Linear)
        self.assertIsInstance(self.encoder.attention, nn.Sequential)
    
    def test_encode_entities(self):
        """Test encode_entities function."""
        # Create entity IDs
        entity_ids = ["Q1", "Q2"]
        
        # Call encode_entities
        result = self.encoder.encode_entities(entity_ids)
        
        # Check that the result has the correct shape
        self.assertEqual(result.shape, (1, 768))
    
    def test_encode_entities_empty(self):
        """Test encode_entities function with empty input."""
        # Call encode_entities with empty list
        result = self.encoder.encode_entities([])
        
        # Check that the result has the correct shape
        self.assertEqual(result.shape, (1, 768))
    
    def test_encode_relations(self):
        """Test encode_relations function."""
        # Create relation IDs
        relation_ids = ["P1", "P2"]
        
        # Call encode_relations
        result = self.encoder.encode_relations(relation_ids)
        
        # Check that the result has the correct shape
        self.assertEqual(result.shape, (1, 768))
    
    def test_encode_relations_empty(self):
        """Test encode_relations function with empty input."""
        # Call encode_relations with empty list
        result = self.encoder.encode_relations([])
        
        # Check that the result has the correct shape
        self.assertEqual(result.shape, (1, 768))


class TestHybridNewsRecommender(unittest.TestCase):
    """Test cases for HybridNewsRecommender."""
    
    def setUp(self):
        """Set up test case."""
        # Create mock entity and relation embeddings
        self.entity_embeddings = {
            "Q1": np.random.rand(100),
            "Q2": np.random.rand(100),
            "Q3": np.random.rand(100)
        }
        
        self.relation_embeddings = {
            "P1": np.random.rand(100),
            "P2": np.random.rand(100)
        }
        
        # Create patches for the encoder classes
        self.news_encoder_patcher = patch("news_ai_app.models.recommender.NewsEncoder")
        self.user_encoder_patcher = patch("news_ai_app.models.recommender.UserEncoder")
        self.kg_encoder_patcher = patch("news_ai_app.models.recommender.KnowledgeGraphEncoder")
        
        # Start the patches
        self.mock_news_encoder = self.news_encoder_patcher.start()
        self.mock_user_encoder = self.user_encoder_patcher.start()
        self.mock_kg_encoder = self.kg_encoder_patcher.start()
        
        # Create the recommender
        self.recommender = HybridNewsRecommender(
            entity_embeddings=self.entity_embeddings,
            relation_embeddings=self.relation_embeddings
        )
    
    def tearDown(self):
        """Tear down test case."""
        # Stop the patches
        self.news_encoder_patcher.stop()
        self.user_encoder_patcher.stop()
        self.kg_encoder_patcher.stop()
    
    def test_init(self):
        """Test initialization."""
        # Check that the encoder classes were initialized correctly
        self.mock_news_encoder.assert_called_once()
        self.mock_user_encoder.assert_called_once()
        self.mock_kg_encoder.assert_called_once_with(
            entity_embeddings=self.entity_embeddings,
            relation_embeddings=self.relation_embeddings,
            output_dim=768
        )
        
        # Check that the recommender has the correct attributes
        self.assertIsInstance(self.recommender.content_kg_fusion, nn.Sequential)
        self.assertIsInstance(self.recommender.predictor, nn.Sequential)
    
    @patch("torch.cat")
    def test_forward(self, mock_cat):
        """Test forward pass."""
        # Mock input tensors
        user_history_embeddings = torch.randn(2, 5, 768)
        candidate_news_embeddings = torch.randn(2, 3, 768)
        
        # Mock user encoder output
        self.recommender.user_encoder.return_value = torch.randn(2, 768)
        
        # Mock torch.cat output
        mock_cat.return_value = torch.randn(2, 3, 1536)
        
        # Call forward
        result = self.recommender.forward(
            user_history_embeddings,
            candidate_news_embeddings
        )
        
        # Check that the result has the correct shape
        self.assertEqual(result.shape, (2, 3))


class TestLoadPretrainedRecommender(unittest.TestCase):
    """Test cases for load_pretrained_recommender function."""
    
    def setUp(self):
        """Set up test case."""
        # Create mock entity and relation embeddings
        self.entity_embeddings = {
            "Q1": np.random.rand(100),
            "Q2": np.random.rand(100),
            "Q3": np.random.rand(100)
        }
        
        self.relation_embeddings = {
            "P1": np.random.rand(100),
            "P2": np.random.rand(100)
        }
        
        # Create a patch for HybridNewsRecommender
        self.recommender_patcher = patch("news_ai_app.models.recommender.HybridNewsRecommender")
        
        # Start the patch
        self.mock_recommender = self.recommender_patcher.start()
    
    def tearDown(self):
        """Tear down test case."""
        # Stop the patch
        self.recommender_patcher.stop()
    
    def test_load_pretrained_recommender(self):
        """Test load_pretrained_recommender function."""
        # Call the function
        result = load_pretrained_recommender(
            entity_embeddings=self.entity_embeddings,
            relation_embeddings=self.relation_embeddings
        )
        
        # Check that the recommender was initialized correctly
        self.mock_recommender.assert_called_once_with(
            entity_embeddings=self.entity_embeddings,
            relation_embeddings=self.relation_embeddings
        )
        
        # Check that the result is the mock recommender
        self.assertEqual(result, self.mock_recommender.return_value)
    
    @patch("torch.load")
    @patch.object(Path, "exists")
    def test_load_pretrained_recommender_with_model_path(self, mock_exists, mock_load):
        """Test load_pretrained_recommender function with model path."""
        # Set up mocks
        mock_exists.return_value = True
        mock_load.return_value = {"state": "dict"}
        
        # Call the function
        result = load_pretrained_recommender(
            entity_embeddings=self.entity_embeddings,
            relation_embeddings=self.relation_embeddings,
            model_path="models/pretrained/recommender.pt"
        )
        
        # Check that the recommender was initialized correctly
        self.mock_recommender.assert_called_once_with(
            entity_embeddings=self.entity_embeddings,
            relation_embeddings=self.relation_embeddings
        )
        
        # Check that torch.load was called with the correct path
        mock_load.assert_called_once_with("models/pretrained/recommender.pt")
        
        # Check that load_state_dict was called on the recommender
        self.mock_recommender.return_value.load_state_dict.assert_called_once_with(
            {"state": "dict"}
        )
        
        # Check that eval was called on the recommender
        self.mock_recommender.return_value.eval.assert_called_once()
        
        # Check that the result is the mock recommender
        self.assertEqual(result, self.mock_recommender.return_value)


if __name__ == "__main__":
    unittest.main()