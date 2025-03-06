"""
Unit tests for metrics calculation.
"""
import unittest

import torch

from news_ai_app.models.metrics import AdvancedMetricsCalculator, get_metrics_calculator


class TestMetrics(unittest.TestCase):
    """Test cases for metrics calculation."""
    
    def setUp(self):
        """Set up test case."""
        self.metrics_calculator = get_metrics_calculator()
        self.sample_text = """
        Climate change is one of the most pressing issues of our time, requiring immediate action 
        from governments, businesses, and individuals. The Intergovernmental Panel on Climate Change (IPCC) 
        has warned that without significant reductions in carbon emissions, global warming will exceed 
        1.5 degrees Celsius above pre-industrial levels, leading to devastating consequences.
        """
    
    def test_political_influence(self):
        """Test political influence calculation."""
        result = self.metrics_calculator.calculate_political_influence(self.sample_text)
        
        # Check that result is a float
        self.assertIsInstance(result, float)
        
        # Check that result is between 0 and 1
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_rhetoric_intensity(self):
        """Test rhetoric intensity calculation."""
        result = self.metrics_calculator.calculate_rhetoric_intensity(self.sample_text)
        
        # Check that result is a float
        self.assertIsInstance(result, float)
        
        # Check that result is between 0 and 1
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_information_depth(self):
        """Test information depth calculation."""
        result = self.metrics_calculator.calculate_information_depth(self.sample_text)
        
        # Check that result is a float
        self.assertIsInstance(result, float)
        
        # Check that result is between 0 and 1
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_sentiment(self):
        """Test sentiment calculation."""
        result = self.metrics_calculator.calculate_sentiment(self.sample_text)
        
        # Check that result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check that result contains label and score
        self.assertIn("label", result)
        self.assertIn("score", result)
        
        # Check that label is a string
        self.assertIsInstance(result["label"], str)
        
        # Check that score is a float between 0 and 1
        self.assertIsInstance(result["score"], float)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
    
    def test_calculate_all_metrics(self):
        """Test calculate_all_metrics function."""
        result = self.metrics_calculator.calculate_all_metrics(self.sample_text)
        
        # Check that result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check that result contains all metrics
        self.assertIn("political_influence", result)
        self.assertIn("rhetoric_intensity", result)
        self.assertIn("information_depth", result)
        self.assertIn("sentiment", result)
    
    def test_batch_calculate_metrics(self):
        """Test batch_calculate_metrics function."""
        texts = [self.sample_text, "This is a test."]
        result = self.metrics_calculator.batch_calculate_metrics(texts)
        
        # Check that result is a list
        self.assertIsInstance(result, list)
        
        # Check that result has correct length
        self.assertEqual(len(result), len(texts))
        
        # Check that each item is a dictionary with metrics
        for item in result:
            self.assertIsInstance(item, dict)
            self.assertIn("political_influence", item)
            self.assertIn("rhetoric_intensity", item)
            self.assertIn("information_depth", item)
            self.assertIn("sentiment", item)


if __name__ == "__main__":
    unittest.main()