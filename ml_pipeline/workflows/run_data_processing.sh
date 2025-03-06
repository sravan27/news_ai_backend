#!/bin/bash

# News AI Data Processing Workflow Script
# This script runs the complete data processing pipeline in the correct order

echo "======================================================"
echo "     News AI Data Processing Pipeline                 "
echo "======================================================"

# Set working directory to project root
cd /Users/sravansridhar/Documents/news_ai

# 1. Create necessary directories
mkdir -p ml_pipeline/data/bronze
mkdir -p ml_pipeline/data/silver
mkdir -p ml_pipeline/data/silicon
mkdir -p ml_pipeline/data/gold
mkdir -p ml_pipeline/data/raw

echo "Directory structure prepared."
echo "------------------------------------------------------"

# 2. First run the fix for behaviors processing script
echo "Running behavior processing fix script..."
python ml_pipeline/scripts/fix_behaviors_processing.py

# 3. Run the bronze layer processing script
echo "------------------------------------------------------"
echo "Running bronze layer processing script..."
python ml_pipeline/scripts/fix_bronze_layer_processing.py

# 4. Process news data to bronze layer
# Run the news processing separately as it's less likely to have issues
echo "------------------------------------------------------"
echo "Processing news data to bronze layer..."
python -c "
import sys
sys.path.append('/Users/sravansridhar/Documents/news_ai')
from ml_pipeline.scripts.fix_bronze_layer_processing import process_news_to_parquet
for split in ['train', 'dev', 'test']:
    print(f'Processing {split} news data...')
    process_news_to_parquet(split)
"

# 5. Process silver layer (feature engineering)
echo "------------------------------------------------------"
echo "Processing silver layer..."
python ml_pipeline/scripts/silver_layer_processing.py

# 6. Run silicon layer notebooks for specialized scores
echo "------------------------------------------------------"
echo "Processing silicon layer notebooks..."

# Run each silicon layer notebook
for notebook in political_influence_silicon rhetoric_intensity_silicon information_depth_silicon sentiment_silicon; do
    echo "Running $notebook notebook..."
    jupyter nbconvert --to notebook --execute ml_pipeline/notebooks/${notebook}.ipynb --output ${notebook}_executed
    mv ${notebook}_executed.ipynb ml_pipeline/notebooks/${notebook}_executed.ipynb
done

# 7. Run the gold layer model integration
echo "------------------------------------------------------"
echo "Processing gold layer model integration..."
jupyter nbconvert --to notebook --execute ml_pipeline/notebooks/04_gold_layer_model_integration.ipynb --output gold_layer_executed
mv gold_layer_executed.ipynb ml_pipeline/notebooks/gold_layer_executed.ipynb

# 8. Final data validation and summary
echo "------------------------------------------------------"
echo "Validating final data..."
python -c "
import sys
import os
from pathlib import Path

project_root = Path('/Users/sravansridhar/Documents/news_ai')
data_paths = {
    'bronze': project_root / 'ml_pipeline' / 'data' / 'bronze',
    'silver': project_root / 'ml_pipeline' / 'data' / 'silver',
    'silicon': project_root / 'ml_pipeline' / 'data' / 'silicon',
    'gold': project_root / 'ml_pipeline' / 'data' / 'gold'
}

for layer, path in data_paths.items():
    if path.exists():
        files = list(path.glob('*'))
        print(f'{layer.capitalize()} layer: {len(files)} files')
        for file in files:
            print(f'  - {file.name} ({file.stat().st_size / (1024*1024):.2f} MB)')
    else:
        print(f'{layer.capitalize()} layer: Directory not found')
"

echo "------------------------------------------------------"
echo "Starting News AI application..."
python run.py

echo "------------------------------------------------------"
echo "Processing complete!"
echo "All data successfully processed and application started."
echo "======================================================"