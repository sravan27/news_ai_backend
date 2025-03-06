# News AI Machine Learning Pipeline

This directory contains the complete data processing and model training pipeline for the News AI project. The pipeline follows a medallion architecture approach with the following layers:

## Medallion Architecture Layers

### 1. Raw Layer
- Original MIND dataset files (TSV, VEC)
- Located in `/MINDLarge`
- Contains: `behaviors.tsv`, `news.tsv`, `entity_embedding.vec`, `relation_embedding.vec`

### 2. Bronze Layer
- Standardized format data (Parquet)
- Located in `data/bronze`
- Contains: `behaviors_*.parquet`, `news_*.parquet`

### 3. Silver Layer
- Feature-engineered data ready for modeling
- Located in `data/silver`
- Contains: `news_features_*.parquet`, `user_features_*.parquet`, `interactions_*.parquet`

### 4. Silicon Layer (Specialized Models)
- Advanced metric-specific modeling
- Located in `data/silicon`
- Specialized metrics:
  - Political Influence
  - Rhetoric Intensity
  - Information Depth
  - Sentiment

### 5. Gold Layer
- Final model outputs and predictions
- Located in `data/gold`
- Contains: `news_recommendations.parquet`, `user_recommendations.json`, and model artifacts

## Directory Structure

```
ml_pipeline/
│
├── data/                  # Storage for pipeline outputs
│   ├── bronze/           # Standardized data from raw
│   ├── silver/           # Feature-engineered data
│   ├── silicon/          # Specialized model outputs
│   ├── gold/             # Final model outputs
│   └── raw/              # Metadata about raw data
│
├── notebooks/            # Jupyter notebooks for each processing stage
│   ├── 01_raw_data_acquisition.ipynb
│   ├── 02_bronze_layer_processing.ipynb
│   ├── 03_silver_layer_processing.ipynb
│   ├── 04_gold_layer_model_integration.ipynb
│   ├── political_influence_silicon.ipynb
│   ├── rhetoric_intensity_silicon.ipynb
│   ├── information_depth_silicon.ipynb
│   └── sentiment_silicon.ipynb
│
├── scripts/              # Standalone processing scripts
│   ├── fix_behaviors_processing.py
│   ├── fix_bronze_layer_processing.py
│   ├── silver_layer_processing.py
│   └── create_silicon_layer_notebook.py
│
└── workflows/            # Pipeline orchestration
    └── run_data_processing.sh
```

## Pipeline Process

1. **Raw to Bronze Layer**:
   - Fix behaviors and news data processing
   - Convert TSV files to Parquet format
   - Implement robust error handling for test data format differences

2. **Bronze to Silver Layer**:
   - Feature engineering for news content
   - User behavior analysis
   - Category preference calculation
   - Build interaction feature dataset

3. **Silver to Silicon Layer**:
   - Specialized model training for each metric
   - Political influence scoring
   - Rhetoric intensity analysis
   - Information depth evaluation
   - Sentiment analysis

4. **Silicon to Gold Layer**:
   - Integrate all specialized models
   - Train recommendation model
   - Generate user-specific recommendations
   - Export artifacts for News AI application

## Running the Pipeline

To run the complete pipeline, use:

```bash
bash ml_pipeline/workflows/run_data_processing.sh
```

This script will:
1. Create the necessary directory structure
2. Process each layer in sequence
3. Run the specialized silicon layer notebooks
4. Build the final gold layer model
5. Start the News AI application

## Handling Test Data Format

The pipeline includes special handling for the test dataset, which has a different format from train/dev:

- Train/dev: impressions have format "news_id-clicked" (e.g., "N12345-1")
- Test: impressions only have news IDs without click information (e.g., "N12345")

The processing scripts detect the format and use appropriate parsing logic.

## Model Integration with News AI App

The gold layer creates all necessary outputs for the News AI application:
- User recommendations
- News metadata with specialized scores
- Trained recommendation model

These artifacts are exported to `news_ai_app/data/app_data` for use by the API and frontend.