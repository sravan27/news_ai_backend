"""
Data and model validation system.
"""
import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import re

logger = logging.getLogger("validation")

class DataValidator:
    """Data validation for input and output data."""
    
    def __init__(self):
        """Initialize DataValidator."""
        # Define common validation rules
        self.validation_rules = {
            "not_null": lambda x: x is not None,
            "positive": lambda x: x > 0,
            "between_0_1": lambda x: 0 <= x <= 1,
            "is_string": lambda x: isinstance(x, str),
            "is_numeric": lambda x: isinstance(x, (int, float)),
            "valid_email": lambda x: bool(re.match(r"[^@]+@[^@]+\.[^@]+", x)) if isinstance(x, str) else False,
            "min_length": lambda x, min_len: len(x) >= min_len if hasattr(x, "__len__") else False,
            "max_length": lambda x, max_len: len(x) <= max_len if hasattr(x, "__len__") else False
        }
    
    def validate_dataframe(self, df: pd.DataFrame, schema: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Validate a pandas DataFrame against a schema."""
        validation_results = {}
        
        # Check for required columns
        for column, rules in schema.items():
            if "required" in rules and rules["required"] and column not in df.columns:
                validation_results[column] = [f"Required column '{column}' is missing"]
        
        # Validate each column
        for column in df.columns:
            if column not in schema:
                continue
            
            column_rules = schema[column]
            column_errors = []
            
            # Check data type
            if "type" in column_rules:
                expected_type = column_rules["type"]
                if expected_type == "numeric":
                    if not pd.api.types.is_numeric_dtype(df[column]):
                        column_errors.append(f"Column '{column}' should be numeric")
                elif expected_type == "string":
                    if not pd.api.types.is_string_dtype(df[column]):
                        column_errors.append(f"Column '{column}' should be string type")
                elif expected_type == "boolean":
                    if not pd.api.types.is_bool_dtype(df[column]):
                        column_errors.append(f"Column '{column}' should be boolean type")
                elif expected_type == "datetime":
                    if not pd.api.types.is_datetime64_dtype(df[column]):
                        column_errors.append(f"Column '{column}' should be datetime type")
            
            # Check null values
            if "allow_null" in column_rules and not column_rules["allow_null"]:
                null_count = df[column].isnull().sum()
                if null_count > 0:
                    column_errors.append(f"Column '{column}' has {null_count} null values but nulls are not allowed")
            
            # Check range
            if "min" in column_rules and pd.api.types.is_numeric_dtype(df[column]):
                min_value = column_rules["min"]
                if df[column].min() < min_value:
                    column_errors.append(f"Column '{column}' has values below minimum of {min_value}")
            
            if "max" in column_rules and pd.api.types.is_numeric_dtype(df[column]):
                max_value = column_rules["max"]
                if df[column].max() > max_value:
                    column_errors.append(f"Column '{column}' has values above maximum of {max_value}")
            
            # Check unique constraint
            if "unique" in column_rules and column_rules["unique"]:
                if not df[column].is_unique:
                    column_errors.append(f"Column '{column}' should have unique values")
            
            # Check custom validation function
            if "custom_validator" in column_rules and callable(column_rules["custom_validator"]):
                validator = column_rules["custom_validator"]
                invalid_mask = ~df[column].apply(validator)
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    column_errors.append(f"Column '{column}' has {invalid_count} rows that fail custom validation")
            
            # Add errors if any
            if column_errors:
                validation_results[column] = column_errors
        
        return validation_results
    
    def validate_model_input(self, data: Any, model_name: str, schema: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate model input data."""
        try:
            # Convert to DataFrame if it's not already
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, dict):
                    data = pd.DataFrame([data])
                elif isinstance(data, list):
                    data = pd.DataFrame(data)
                else:
                    return {"_general": [f"Unsupported input data type for model {model_name}: {type(data)}"]}
            
            # Validate against schema
            return self.validate_dataframe(data, schema)
        except Exception as e:
            logger.error(f"Error validating model input: {e}")
            return {"_error": [f"Validation error: {str(e)}"]}
    
    def validate_model_output(self, output: Any, model_name: str, schema: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate model output data."""
        try:
            # For simple outputs like a single prediction
            if isinstance(output, (int, float, str, bool)):
                return self._validate_simple_value(output, schema)
            
            # For dictionary output
            elif isinstance(output, dict):
                return self._validate_dict(output, schema)
            
            # For DataFrame output
            elif isinstance(output, pd.DataFrame):
                return self.validate_dataframe(output, schema)
            
            # For list output
            elif isinstance(output, list):
                errors = {}
                for i, item in enumerate(output):
                    item_errors = self._validate_dict(item, schema) if isinstance(item, dict) else {}
                    if item_errors:
                        errors[f"item_{i}"] = [f"{key}: {', '.join(msgs)}" for key, msgs in item_errors.items()]
                return errors
            
            else:
                return {"_general": [f"Unsupported output data type for model {model_name}: {type(output)}"]}
        except Exception as e:
            logger.error(f"Error validating model output: {e}")
            return {"_error": [f"Validation error: {str(e)}"]}
    
    def _validate_simple_value(self, value: Any, schema: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate a simple value."""
        errors = []
        
        # Check type
        if "type" in schema:
            if schema["type"] == "numeric" and not isinstance(value, (int, float)):
                errors.append(f"Value should be numeric, got {type(value)}")
            elif schema["type"] == "string" and not isinstance(value, str):
                errors.append(f"Value should be string, got {type(value)}")
            elif schema["type"] == "boolean" and not isinstance(value, bool):
                errors.append(f"Value should be boolean, got {type(value)}")
        
        # Check range for numeric values
        if isinstance(value, (int, float)):
            if "min" in schema and value < schema["min"]:
                errors.append(f"Value {value} is below minimum of {schema['min']}")
            if "max" in schema and value > schema["max"]:
                errors.append(f"Value {value} is above maximum of {schema['max']}")
        
        # Check string length
        if isinstance(value, str):
            if "min_length" in schema and len(value) < schema["min_length"]:
                errors.append(f"String length {len(value)} is below minimum of {schema['min_length']}")
            if "max_length" in schema and len(value) > schema["max_length"]:
                errors.append(f"String length {len(value)} is above maximum of {schema['max_length']}")
        
        return {"value": errors} if errors else {}
    
    def _validate_dict(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate a dictionary against a schema."""
        validation_results = {}
        
        # Check for required fields
        for field, field_schema in schema.items():
            if "required" in field_schema and field_schema["required"] and field not in data:
                validation_results[field] = [f"Required field '{field}' is missing"]
        
        # Validate each field
        for field, value in data.items():
            if field not in schema:
                continue
            
            field_schema = schema[field]
            field_errors = []
            
            # Check type
            if "type" in field_schema:
                if field_schema["type"] == "numeric" and not isinstance(value, (int, float)):
                    field_errors.append(f"Field should be numeric, got {type(value)}")
                elif field_schema["type"] == "string" and not isinstance(value, str):
                    field_errors.append(f"Field should be string, got {type(value)}")
                elif field_schema["type"] == "boolean" and not isinstance(value, bool):
                    field_errors.append(f"Field should be boolean, got {type(value)}")
                elif field_schema["type"] == "array" and not isinstance(value, list):
                    field_errors.append(f"Field should be an array, got {type(value)}")
                elif field_schema["type"] == "object" and not isinstance(value, dict):
                    field_errors.append(f"Field should be an object, got {type(value)}")
            
            # Check range for numeric values
            if isinstance(value, (int, float)):
                if "min" in field_schema and value < field_schema["min"]:
                    field_errors.append(f"Value {value} is below minimum of {field_schema['min']}")
                if "max" in field_schema and value > field_schema["max"]:
                    field_errors.append(f"Value {value} is above maximum of {field_schema['max']}")
            
            # Check string length
            if isinstance(value, str):
                if "min_length" in field_schema and len(value) < field_schema["min_length"]:
                    field_errors.append(f"String length {len(value)} is below minimum of {field_schema['min_length']}")
                if "max_length" in field_schema and len(value) > field_schema["max_length"]:
                    field_errors.append(f"String length {len(value)} is above maximum of {field_schema['max_length']}")
            
            # Check enum
            if "enum" in field_schema and value not in field_schema["enum"]:
                field_errors.append(f"Value '{value}' not in allowed values: {field_schema['enum']}")
            
            # Add errors if any
            if field_errors:
                validation_results[field] = field_errors
        
        return validation_results

class ModelValidator:
    """Model validation for model quality and performance."""
    
    def __init__(self):
        """Initialize ModelValidator."""
        pass
    
    def validate_predictions(self, predictions: Any, ground_truth: Any) -> Dict[str, float]:
        """Validate model predictions against ground truth."""
        try:
            # If predictions and ground_truth are not DataFrames, convert them
            if not isinstance(predictions, pd.DataFrame):
                if isinstance(predictions, (list, np.ndarray)):
                    predictions = pd.DataFrame({"prediction": predictions})
                else:
                    predictions = pd.DataFrame({"prediction": [predictions]})
            
            if not isinstance(ground_truth, pd.DataFrame):
                if isinstance(ground_truth, (list, np.ndarray)):
                    ground_truth = pd.DataFrame({"actual": ground_truth})
                else:
                    ground_truth = pd.DataFrame({"actual": [ground_truth]})
            
            # Calculate validation metrics
            metrics = {}
            
            # For regression models
            if pd.api.types.is_numeric_dtype(predictions.iloc[:, 0]) and pd.api.types.is_numeric_dtype(ground_truth.iloc[:, 0]):
                y_pred = predictions.iloc[:, 0].values
                y_true = ground_truth.iloc[:, 0].values
                
                # Mean Absolute Error
                metrics["mae"] = np.mean(np.abs(y_pred - y_true))
                
                # Mean Squared Error
                metrics["mse"] = np.mean(np.square(y_pred - y_true))
                
                # Root Mean Squared Error
                metrics["rmse"] = np.sqrt(metrics["mse"])
                
                # R-squared (only if more than 1 sample)
                if len(y_true) > 1:
                    ss_total = np.sum(np.square(y_true - np.mean(y_true)))
                    ss_residual = np.sum(np.square(y_true - y_pred))
                    metrics["r2"] = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            # For classification models
            elif predictions.iloc[:, 0].dtype == bool or ground_truth.iloc[:, 0].dtype == bool:
                y_pred = predictions.iloc[:, 0].astype(bool).values
                y_true = ground_truth.iloc[:, 0].astype(bool).values
                
                # True positives, false positives, true negatives, false negatives
                tp = np.sum((y_pred == True) & (y_true == True))
                fp = np.sum((y_pred == True) & (y_true == False))
                tn = np.sum((y_pred == False) & (y_true == False))
                fn = np.sum((y_pred == False) & (y_true == True))
                
                # Accuracy
                metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                
                # Precision
                metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                # Recall
                metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # F1 Score
                metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0
            
            return metrics
        except Exception as e:
            logger.error(f"Error validating predictions: {e}")
            return {"error": str(e)}
    
    def check_data_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                       columns: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Check for data drift between reference and current data."""
        try:
            drift_metrics = {}
            
            # Use all columns if not specified
            if columns is None:
                columns = [col for col in reference_data.columns if col in current_data.columns]
            
            for column in columns:
                if column not in reference_data.columns or column not in current_data.columns:
                    continue
                
                col_metrics = {}
                
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(reference_data[column]) and pd.api.types.is_numeric_dtype(current_data[column]):
                    # Calculate statistics
                    ref_mean = reference_data[column].mean()
                    curr_mean = current_data[column].mean()
                    ref_std = reference_data[column].std()
                    curr_std = current_data[column].std()
                    
                    # Mean difference
                    col_metrics["mean_diff"] = abs(ref_mean - curr_mean)
                    
                    # Mean difference percentage
                    if ref_mean != 0:
                        col_metrics["mean_diff_pct"] = abs(ref_mean - curr_mean) / abs(ref_mean)
                    else:
                        col_metrics["mean_diff_pct"] = float("inf") if curr_mean != 0 else 0
                    
                    # Standard deviation difference
                    col_metrics["std_diff"] = abs(ref_std - curr_std)
                    
                    # Standard deviation difference percentage
                    if ref_std != 0:
                        col_metrics["std_diff_pct"] = abs(ref_std - curr_std) / ref_std
                    else:
                        col_metrics["std_diff_pct"] = float("inf") if curr_std != 0 else 0
                
                # For categorical columns
                elif pd.api.types.is_categorical_dtype(reference_data[column]) or pd.api.types.is_string_dtype(reference_data[column]):
                    # Calculate value distributions
                    ref_counts = reference_data[column].value_counts(normalize=True).to_dict()
                    curr_counts = current_data[column].value_counts(normalize=True).to_dict()
                    
                    # Population Stability Index (PSI)
                    psi = 0
                    for value in set(list(ref_counts.keys()) + list(curr_counts.keys())):
                        ref_pct = ref_counts.get(value, 0)
                        curr_pct = curr_counts.get(value, 0)
                        
                        # Add a small value to avoid division by zero
                        ref_pct = max(ref_pct, 0.0001)
                        curr_pct = max(curr_pct, 0.0001)
                        
                        psi += (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
                    
                    col_metrics["psi"] = psi
                    
                    # Jaccard similarity for categories
                    ref_set = set(reference_data[column].dropna().unique())
                    curr_set = set(current_data[column].dropna().unique())
                    
                    intersection = len(ref_set.intersection(curr_set))
                    union = len(ref_set.union(curr_set))
                    
                    col_metrics["jaccard_similarity"] = intersection / union if union > 0 else 1
                
                drift_metrics[column] = col_metrics
            
            return drift_metrics
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return {"error": str(e)}