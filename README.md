# News AI - Intelligent News Analysis and Recommendation System

A comprehensive platform for analyzing news content across political influence, rhetoric intensity, information depth, and sentiment dimensions using a medallion architecture (Bronze → Silver → Gold → Silicon) data pipeline.

## Key Features

- **Multi-dimensional News Analysis**: Analyze news content across political influence, rhetoric intensity, information depth, and sentiment dimensions
- **Personalized Recommendations**: Hybrid recommendation system combining content-based, collaborative filtering, and knowledge graph approaches
- **Podcast Generation**: Create audio podcasts from selected news articles
- **Interactive Dashboard**: Explore news content, visualize metrics, and get recommendations

## Architecture

### Backend Components

- **Data Pipeline**:
  - Bronze Layer: Raw data ingestion and validation
  - Silver Layer: Feature engineering and embeddings
  - Gold Layer: Model-ready datasets
  - Silicon Layer: Advanced metrics models

- **Machine Learning**:
  - Hybrid recommendation system using transformer-based encoders
  - Specialized models for content analysis
  - Entity extraction and knowledge graph integration

### Frontend Components

- **Streamlit Dashboard**: Interactive exploration of news content and analysis
- **API**: RESTful endpoints for retrieving news, recommendations, and analysis

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/news_ai.git
cd news_ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up the environment and download pretrained models
python setup_app.py
```

### Running the Application

```bash
# Run the full application (API + Streamlit)
python run.py

# Run just the API
python standalone_api.py

# Run just the Streamlit frontend
cd news_ai_app/frontend && streamlit run streamlit_app.py
```

### Data Processing Pipeline

```bash
# Run the full pipeline (Bronze → Silver → Gold → Silicon)
cd ml_pipeline/workflows && ./run_optimized_pipeline.sh

# Run just the Silicon layer processing
cd ml_pipeline/workflows && ./run_silicon_processing.sh
```

## Data Requirements

This project requires the MIND (Microsoft News Dataset) to train and run models. You can set up the dataset using our script:

```bash
# Make the script executable
chmod +x setup_datasets.sh

# Run the dataset setup script
./setup_datasets.sh
```

This will download and extract the MINDLarge dataset (train, dev, test) to the `MINDLarge` directory.

Alternatively, you can manually download the dataset from [https://msnews.github.io/](https://msnews.github.io/) and extract it to the `MINDLarge` directory.

## Configuration

Edit `ml_pipeline/config/pipeline_config.yaml` to adjust:
- Data paths
- Model parameters
- System resources
- Hardware acceleration

## Performance Optimization

The project includes extensive optimizations for processing large datasets, including:
- Multiprocessing with Ray
- GPU/MPS acceleration
- Memory-efficient data structures
- Vectorized operations

See `OPTIMIZATION.md` and `SILICON_OPTIMIZATION.md` for details.

## License

[MIT License](LICENSE)