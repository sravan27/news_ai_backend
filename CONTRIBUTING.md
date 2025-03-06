# Contributing to News AI

Thank you for considering contributing to News AI! This document provides guidelines and instructions for contributing to the project.

## Repository Structure

The project follows a medallion architecture with the following data flow:
- **Bronze Layer**: Raw data ingestion and cleansing
- **Silver Layer**: Feature engineering and transformations
- **Gold Layer**: Model-ready datasets
- **Silicon Layer**: Advanced metrics models

### Key Directories

- `ml_pipeline/` - Core data processing pipeline
  - `data/` - Data storage for each layer
  - `scripts/` - Processing scripts
  - `config/` - Configuration files
  - `notebooks/` - Jupyter notebooks for documentation
- `news_ai_app/` - Application code
  - `api/` - FastAPI implementation
  - `frontend/` - Streamlit dashboard
  - `models/` - ML model implementations

## Setting Up Development Environment

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/news_ai.git
   cd news_ai
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the dataset
   ```bash
   ./setup_datasets.sh
   ```

## Working with Large Files

This project uses Git LFS (Large File Storage) for managing large files. To work with these files:

1. Install Git LFS
   ```bash
   # macOS
   brew install git-lfs
   
   # Ubuntu
   sudo apt-get install git-lfs
   
   # Windows
   Download from https://git-lfs.github.com/
   ```

2. Initialize Git LFS
   ```bash
   git lfs install
   ```

3. When you clone the repository, large files will be pulled automatically. If you need to force a pull:
   ```bash
   git lfs pull
   ```

## Running the Pipeline

The full pipeline can be run with:
```bash
cd ml_pipeline/workflows && ./run_optimized_pipeline.sh
```

For individual layers:
```bash
# Bronze to Silver
cd ml_pipeline/scripts && python silver_layer_processing.py

# Silicon layer
cd ml_pipeline/scripts && python optimize_silicon_processing.py
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Include docstrings in Google format
- Write unit tests for new functionality

## Pull Request Process

1. Create a new branch for your feature or bugfix
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with descriptive messages
   ```bash
   git commit -m "Add feature X that does Y"
   ```

3. Push to your branch
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a pull request to the main branch
   - Include a clear description of the changes
   - Reference any related issues
   - Ensure tests pass
   - Update documentation if necessary

## License

By contributing to News AI, you agree that your contributions will be licensed under the project's MIT License.