#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation helper functions for the ML pipeline.
Used to validate notebook code and data integrity before execution.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import traceback


def validate_file_exists(file_path):
    """Check if file exists and return True/False."""
    return os.path.exists(file_path)


def validate_directory_exists(dir_path):
    """Check if directory exists and return True/False."""
    return os.path.isdir(dir_path)


def validate_tsv_structure(file_path, expected_columns=None, delimiter='\t'):
    """
    Validate TSV file structure.
    Returns (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        # Try to read the file
        df = pd.read_csv(file_path, sep=delimiter, nrows=10, header=None)
        
        # Check column count if expected_columns is provided
        if expected_columns is not None:
            if len(df.columns) != len(expected_columns):
                return False, f"Expected {len(expected_columns)} columns, got {len(df.columns)}"
        
        return True, None
    except Exception as e:
        return False, f"Error validating TSV: {str(e)}"


def validate_behaviors_file(file_path, delimiter='\t'):
    """
    Specifically validate behaviors.tsv file structure and content.
    Returns (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        # Try to read the file
        df = pd.read_csv(file_path, sep=delimiter, nrows=100, header=None)
        
        # Check column count (behaviors should have 5 columns)
        if len(df.columns) != 5:
            return False, f"Expected 5 columns, got {len(df.columns)}"
        
        # Check for impression format in a sample of rows
        for idx, row in df.iterrows():
            impressions = row[4]
            if pd.notna(impressions) and isinstance(impressions, str):
                items = impressions.split()
                for item in items[:5]:  # Check first 5 items
                    if '-' in item:
                        parts = item.split('-')
                        if len(parts) != 2:
                            return False, f"Invalid impression format at row {idx}: {item}"
                        if parts[1] not in ['0', '1']:
                            return False, f"Invalid click value at row {idx}: {parts[1]}"
        
        return True, None
    except Exception as e:
        return False, f"Error validating behaviors file: {str(e)}"


def validate_news_file(file_path, delimiter='\t'):
    """
    Specifically validate news.tsv file structure and content.
    Returns (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        # Try to read the file
        df = pd.read_csv(file_path, sep=delimiter, nrows=100, header=None, quoting=3)
        
        # Check column count (news should have 8 columns)
        if len(df.columns) != 8:
            return False, f"Expected 8 columns, got {len(df.columns)}"
        
        # Check entities columns for JSON format
        entity_columns = [6, 7]  # title_entities and abstract_entities
        for idx, row in df.iterrows():
            for col in entity_columns:
                entities = row[col]
                if pd.notna(entities) and entities:
                    try:
                        json.loads(entities)
                    except json.JSONDecodeError:
                        return False, f"Invalid JSON in row {idx}, column {col}: {entities[:50]}..."
        
        return True, None
    except Exception as e:
        return False, f"Error validating news file: {str(e)}\n" + traceback.format_exc()


def validate_embedding_file(file_path, delimiter='\t'):
    """
    Validate entity/relation embedding file structure.
    Returns (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        # Try to read a few lines
        with open(file_path, 'r') as f:
            lines = [f.readline() for _ in range(5)]
        
        # Check format of each line
        dimension = None
        for i, line in enumerate(lines):
            parts = line.strip().split(delimiter)
            if len(parts) < 2:
                return False, f"Invalid format at line {i+1}: too few columns"
            
            # Check if all values after the first column are numbers
            try:
                values = [float(x) for x in parts[1:]]
            except ValueError:
                return False, f"Invalid embedding values at line {i+1}"
            
            # Check consistent dimensions
            if dimension is None:
                dimension = len(values)
            elif dimension != len(values):
                return False, f"Inconsistent dimensions: {dimension} vs {len(values)} at line {i+1}"
        
        return True, None
    except Exception as e:
        return False, f"Error validating embedding file: {str(e)}"


def validate_mind_dataset(mind_path):
    """
    Validate the MIND dataset structure and availability.
    Returns (is_valid, error_message)
    """
    mind_path = Path(mind_path)
    
    # Check if the directory exists
    if not mind_path.exists():
        return False, f"MIND dataset directory {mind_path} not found"
        
    # Check for train/dev/test splits
    splits = ["MINDlarge_train", "MINDlarge_dev", "MINDlarge_test"]
    missing_splits = [s for s in splits if not (mind_path / s).exists()]
    
    if missing_splits:
        return False, f"Missing MIND dataset splits: {missing_splits}"
    
    # Check for required files in each split
    required_files = ["behaviors.tsv", "news.tsv", "entity_embedding.vec", "relation_embedding.vec"]
    
    validation_results = []
    
    for split in splits:
        split_path = mind_path / split
        missing_files = [f for f in required_files if not (split_path / f).exists()]
        
        if missing_files:
            return False, f"Missing files in {split}: {missing_files}"
        
        # Validate behaviors file
        behaviors_path = split_path / "behaviors.tsv"
        is_valid, error = validate_behaviors_file(behaviors_path)
        if not is_valid:
            validation_results.append(f"Invalid behaviors file in {split}: {error}")
            
        # Validate news file
        news_path = split_path / "news.tsv"
        is_valid, error = validate_news_file(news_path)
        if not is_valid:
            validation_results.append(f"Invalid news file in {split}: {error}")
            
        # Validate embeddings
        entity_path = split_path / "entity_embedding.vec"
        is_valid, error = validate_embedding_file(entity_path)
        if not is_valid:
            validation_results.append(f"Invalid entity embeddings in {split}: {error}")
            
        relation_path = split_path / "relation_embedding.vec"
        is_valid, error = validate_embedding_file(relation_path)
        if not is_valid:
            validation_results.append(f"Invalid relation embeddings in {split}: {error}")
    
    if validation_results:
        return False, "\n".join(validation_results)
    
    return True, "MIND dataset validated successfully"


def validate_parquet_file(file_path):
    """Validate a Parquet file can be read."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(file_path, memory_map=True)
        return True, f"Valid parquet file with {table.num_rows} rows, {len(table.column_names)} columns"
    except Exception as e:
        return False, f"Error validating parquet file: {str(e)}"


def validate_arrow_file(file_path):
    """Validate an Arrow file can be read."""
    try:
        import pyarrow as pa
        with pa.OSFile(file_path, 'rb') as f:
            reader = pa.RecordBatchFileReader(f)
            schema = reader.schema
            num_batches = reader.num_record_batches
        return True, f"Valid arrow file with {num_batches} batches"
    except Exception as e:
        return False, f"Error validating arrow file: {str(e)}"


if __name__ == "__main__":
    print("Testing validation helpers...")
    # Add tests here if needed