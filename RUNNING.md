# Running Instructions

## Quick Start Guide

1. Set up the dataset:
   ```bash
   ./setup_datasets.sh
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python run.py
   ```

## Running the EDA Dashboard

```bash
cd advanced_eda
pip install -r requirements.txt
python run_streamlit.py
```

## Running the Pipeline

```bash
cd ml_pipeline/workflows
./run_optimized_pipeline.sh
```

## Processing Individual Layers

### Bronze Layer
```bash
cd ml_pipeline/scripts
python bronze_layer_processing.py
```

### Silver Layer
```bash
cd ml_pipeline/scripts
python silver_layer_processing.py
```

### Silicon Layer (Metrics Models)
```bash
cd ml_pipeline/scripts
python optimize_silicon_processing.py --metric information_depth
python optimize_silicon_processing.py --metric political_influence
python optimize_silicon_processing.py --metric rhetoric_intensity
python optimize_silicon_processing.py --metric sentiment
```

## Advanced Features

### Visualizing Model Performance
```bash
cd advanced_eda/streamlit
python silicon_metrics_dashboard.py
```

### Training Custom Models
```bash
python train_full_models.py
```