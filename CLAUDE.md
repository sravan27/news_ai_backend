# News AI Project Guide

## Common Commands

### Setup and Environment
```bash
# Set up the environment and download pretrained models
python setup_app.py

# Activate the virtual environment
source venv/bin/activate
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

### Training Models
```bash
# Train the full recommender model
python train_full_models.py

# Train individual silicon models
cd ml_pipeline/scripts && python optimize_silicon_processing.py --metric political_influence
cd ml_pipeline/scripts && python optimize_silicon_processing.py --metric rhetoric_intensity
cd ml_pipeline/scripts && python optimize_silicon_processing.py --metric information_depth
cd ml_pipeline/scripts && python optimize_silicon_processing.py --metric sentiment
```

## Project Structure
- **news_ai_app**: Main application code
  - **api**: FastAPI implementation
  - **frontend**: Streamlit dashboard
  - **models**: ML models implementation
  - **data**: Dataset loaders
  - **utils**: Utilities

- **ml_pipeline**: Data processing pipeline
  - **data**: Multi-layer data storage (Bronze → Silver → Gold → Silicon)
  - **notebooks**: Pipeline documentation notebooks
  - **scripts**: Processing scripts
  - **models**: Model artifacts

## Code Style Preferences
- Use snake_case for variables and functions
- Use PEP 8 style guidelines
- Type hints for function parameters and return values
- Clear docstrings in Google format
- Descriptive variable names

## Performance Optimization
- For large dataset processing (MINDLarge):
  - Set batch_size to 256 in config
  - Use parallel_jobs=8+ for multiprocessing
  - Enable use_amp=true for mixed precision
  - memory_limit_gb=64+ for large datasets