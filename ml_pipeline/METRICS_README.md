# Advanced News Metrics Modeling

This document describes the sophisticated machine learning approach used to calculate news content metrics in our platform.

## Overview

Our system uses a combination of advanced machine learning techniques to calculate four key metrics:

1. **Political Influence**: Measures the political content and bias in news articles
2. **Rhetoric Intensity**: Quantifies the use of rhetorical devices and emotional language
3. **Information Depth**: Evaluates the information density and complexity of content
4. **Sentiment**: Analyzes the emotional tone of the news article

## Technical Approach

### Pipeline Architecture

The modeling pipeline follows a multi-layer approach:

1. **Bronze Layer**: Raw data ingestion and storage
2. **Silver Layer**: Feature extraction and intermediate processing
3. **Silicon Layer**: Advanced model training and selection
4. **Gold Layer**: Model deployment and inference

### Advanced Modeling Techniques

For each metric, we implement and evaluate multiple models:

#### Base Models
- LightGBM
- XGBoost
- CatBoost

#### Ensemble Methods
- Voting Ensemble
- Stacking Ensemble
- Bagging

#### Neural Networks
- Deep Neural Networks with BatchNorm and Dropout
- Residual Networks with skip connections

### Feature Engineering

We extract rich features from news articles for each metric:

#### Political Influence Features
- Political term frequency
- Political entity detection
- Topic indicators
- Category-based features

#### Rhetoric Intensity Features
- Punctuation patterns
- Rhetorical markers
- Question and exclamation density
- Adverb and adjective usage

#### Information Depth Features
- Text length and complexity metrics
- Unique word ratio
- Entity density
- Numeric content density

#### Sentiment Features
- Positive/negative word counts
- Emotion indicators
- Title vs. abstract sentiment contrast
- Category-adjusted sentiment baseline

### Model Optimization

Our models are optimized for:

1. **Performance**: High accuracy with low MSE and MAE
2. **Efficiency**: Fast inference on resource-constrained devices
3. **Generalization**: Robust performance across different news categories
4. **Interpretability**: Understanding feature importance for transparency

## M3 Max MacBook Pro Optimization

The models are specifically tuned for Apple Silicon:

- PyTorch MPS acceleration
- Batch size optimization for M3 Max memory bandwidth
- Model pruning to reduce memory footprint
- CPU-GPU workload balancing

## Training the Models

To train the advanced metric models:

```bash
python train_advanced_models.py
```

To train a specific metric:

```bash
python train_advanced_models.py --metric political_influence
```

## Model Evaluation

Each model is evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared
- When applicable, classification metrics (Accuracy, F1)

## Integration with News AI App

The models are automatically deployed to the `ml_pipeline/models/deployed` directory, where they are loaded by the News AI application for inference.

The application's API integrates these models to provide real-time analysis of news content through the following endpoints:

- `/api/metrics`: Calculate metrics for a single piece of text
- `/api/metrics/batch`: Calculate metrics for multiple texts
- `/api/analyze`: Comprehensive analysis including metrics and entities

## Next Steps

Future improvements planned:

1. Fine-tuning with additional datasets
2. Cross-validation for more robust evaluation
3. Distillation of large models to smaller, more efficient ones
4. Online learning to adapt to evolving news patterns

## References

- Microsoft MIND Dataset
- Learning to Rank for Content-Based News Recommendation
- Knowledge Graph-Enhanced News Recommendation