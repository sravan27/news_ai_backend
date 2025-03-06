# News AI Data Processing Pipeline Instructions

This document provides step-by-step instructions for running the data processing pipeline in the correct order to handle all edge cases, particularly with the test data format differences.

## Prerequisites

- Clone the repository and install dependencies
- MIND dataset downloaded and placed in the correct location

## Pipeline Steps

### 1. Fix Scripts Preparation

Make sure the fix scripts are in place:
- `/ml_pipeline/scripts/fix_behaviors_processing.py`
- `/ml_pipeline/scripts/fix_bronze_layer_processing.py`

### 2. Run the Automated Pipeline

For convenience, you can run the entire pipeline with a single command:

```bash
bash ml_pipeline/workflows/run_data_processing.sh
```

This script will:
1. Create necessary directories
2. Run the behavior processing fix
3. Run the bronze layer processing for all splits
4. Process news data to bronze layer

### 3. Manual Step-by-Step Process (if needed)

If you prefer to run steps individually or need more control:

#### Step 1: Process Behaviors Data

```bash
python ml_pipeline/scripts/fix_behaviors_processing.py
```

This script handles the different formats in train, dev, and test datasets:
- Train/dev: impressions with click information (e.g., "N12345-1")
- Test: impressions without click information (e.g., "N12345")

#### Step 2: Process News Data

```bash
# From Python script or notebook
from ml_pipeline.scripts.fix_bronze_layer_processing import process_news_to_parquet
for split in ["train", "dev", "test"]:
    process_news_to_parquet(split)
```

#### Step 3: Verify Processing Results

Check if the parquet files were created successfully:

```bash
ls -la ml_pipeline/data/bronze/
```

You should see:
- `behaviors_train.parquet`
- `behaviors_dev.parquet`
- `behaviors_test.parquet`
- `news_train.parquet`
- `news_dev.parquet`
- `news_test.parquet`

### 4. Notebook Processing (Optional)

For exploration and analysis, you can run these notebooks:

1. `notebooks/fix_behavior_processing.ipynb`: Demonstrates the fix for the IndexError
2. `notebooks/process_mind_dataset.ipynb`: End-to-end processing using the fixed helpers
3. `notebooks/test_data_format.ipynb`: Analysis of the test data format differences

## Troubleshooting

### Common Issues

1. **IndexError: list index out of range**
   - This occurs when processing test data with the original code
   - Solution: Use the fixed scripts that handle different formats

2. **Missing files in bronze layer**
   - Check paths in scripts match your environment
   - Ensure source MIND data exists in the correct location

3. **PyArrow serialization errors**
   - This can happen with complex data structures
   - Solution: Ensure data types are consistent and serializable

## Next Steps

After successfully processing the bronze layer:

1. Proceed to the silver layer for feature engineering
2. Run Silicon layer notebooks for advanced metrics models
3. Process to gold layer for model outputs and predictions