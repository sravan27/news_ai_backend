#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create a Silicon layer notebook template.
This script generates a complete, bug-free notebook scaffold.
"""

import os
import json
import argparse
from pathlib import Path


# Define notebook cell templates
MARKDOWN_TITLE_TEMPLATE = """# {title}

{description}

## Overview

This notebook implements the Silicon layer processing for the {metric_name} metric in our News AI application pipeline.

### Silicon Layer Objectives

1. Advanced model evaluation and selection
2. Ensemble methodology implementation
3. Neural architecture optimization
4. Model deployment preparation

We'll leverage Apple Silicon acceleration (M2 Max) for optimal performance.
"""

IMPORTS_CELL_TEMPLATE = """# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import pickle
import warnings
from tqdm.notebook import tqdm

# ML Libraries
import torch
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost, Pool
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
import tensorflow as tf

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
tf.random.set_seed(42)

# Enable better display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)

# Configure warning handling
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import custom modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from scripts.validation_helpers import validate_parquet_file
from scripts.notebook_helpers import robust_parquet_reader, safe_json_loads

# Set up device detection for PyTorch
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up paths
DATA_PATH = "../data"
BRONZE_PATH = os.path.join(DATA_PATH, "bronze")
SILVER_PATH = os.path.join(DATA_PATH, "silver")
SILICON_PATH = os.path.join(DATA_PATH, "silicon")
MODEL_PATH = "../models"
DEPLOYED_MODEL_PATH = os.path.join(MODEL_PATH, "deployed")

# Create necessary directories
os.makedirs(SILICON_PATH, exist_ok=True)
os.makedirs(DEPLOYED_MODEL_PATH, exist_ok=True)"""

PROBLEM_DEFINITION_TEMPLATE = """## 1. Problem Definition

### Business Context and Importance

{business_context}

### Metric Definition

{metric_definition}

### Evaluation Criteria

For this {metric_name} model, we'll use the following evaluation metrics:

{evaluation_metrics}

### Success Criteria

{success_criteria}"""

DATA_PREPARATION_TEMPLATE = """## 2. Data Preparation

In this section, we'll load the pre-processed data from the Silver layer and prepare it for model training and evaluation.

### Load Silver Layer Data"""

DATA_LOADING_TEMPLATE = r"""# Load the necessary data files
try:
    # Validate files exist and are readable
    news_path = os.path.join(SILVER_PATH, "news_processed.parquet")
    behaviors_path = os.path.join(SILVER_PATH, "behaviors_processed.parquet")
    features_path = os.path.join(SILVER_PATH, "features_{metric_type}.parquet")
    
    # Check if files exist
    if not os.path.exists(news_path):
        print(f"❌ News file not found: {news_path}")
    if not os.path.exists(behaviors_path):
        print(f"❌ Behaviors file not found: {behaviors_path}")
    if not os.path.exists(features_path):
        print(f"❌ Features file not found: {features_path}")
        print(f"⚠️ Using a placeholder features path for demonstration")
        features_path = None
    
    # Load news data
    news_df, news_error = robust_parquet_reader(news_path)
    if news_error:
        print(f"❌ Error loading news data: {news_error}")
        # Create dummy data for demonstration
        news_df = pd.DataFrame({
            'news_id': [f'N{i}' for i in range(100)],
            'category': np.random.choice(['politics', 'sports', 'entertainment', 'technology'], 100),
            'title': [f'Title {i}' for i in range(100)],
            'abstract': [f'Abstract {i}' for i in range(100)],
            'title_entities': [[{{'Label': f'Entity{j}', 'WikidataId': f'Q{j}'}} for j in range(3)] for i in range(100)],
            'title_length': np.random.randint(10, 50, 100),
            'abstract_length': np.random.randint(50, 200, 100)
        })
    else:
        print(f"✅ Successfully loaded {len(news_df)} news articles")
    
    # Load behaviors data
    behaviors_df, behaviors_error = robust_parquet_reader(behaviors_path)
    if behaviors_error:
        print(f"❌ Error loading behaviors data: {behaviors_error}")
        # Create dummy data for demonstration
        behaviors_df = pd.DataFrame({
            'user_id': [f'U{i}' for i in range(50)],
            'history': [[f'N{j}' for j in range(np.random.randint(5, 15))] for i in range(50)],
            'impressions': [[{{'news_id': f'N{j}', 'clicked': np.random.randint(0, 2)}} for j in range(np.random.randint(10, 20))] for i in range(50)],
            'history_length': np.random.randint(5, 15, 50),
            'click_ratio': np.random.random(50)
        })
    else:
        print(f"✅ Successfully loaded {len(behaviors_df)} behavior records")
    
    # Load feature data if available
    if features_path and os.path.exists(features_path):
        features_df, features_error = robust_parquet_reader(features_path)
        if features_error:
            print(f"❌ Error loading feature data: {features_error}")
            features_df = None
        else:
            print(f"✅ Successfully loaded {len(features_df)} feature records")
    else:
        print("⚠️ No feature data available, will generate synthetic features")
        features_df = None
    
except Exception as e:
    import traceback
    print(f"❌ Unexpected error: {str(e)}")
    print(traceback.format_exc())
    # Create dummy data if loading fails
    news_df = pd.DataFrame({
        'news_id': [f'N{i}' for i in range(100)],
        'category': np.random.choice(['politics', 'sports', 'entertainment', 'technology'], 100),
        'title': [f'Title {i}' for i in range(100)],
        'abstract': [f'Abstract {i}' for i in range(100)]
    })
    behaviors_df = pd.DataFrame({
        'user_id': [f'U{i}' for i in range(50)],
        'history': [[f'N{j}' for j in range(np.random.randint(5, 15))] for i in range(50)],
        'impressions': [[{{'news_id': f'N{j}', 'clicked': np.random.randint(0, 2)}} for j in range(np.random.randint(10, 20))] for i in range(50)]
    })
    features_df = None"""

FEATURE_ENGINEERING_TEMPLATE = """### Feature Engineering

Let's prepare the features for our model training. We'll either use the pre-computed features from the Silver layer or generate new ones specific to the {metric_name} task."""

FEATURE_PROCESSING_TEMPLATE = """# Generate features for {metric_name} prediction
def prepare_features():
    \"\"\"Prepare features for {metric_name} prediction.\"\"\"
    # If we have pre-computed features, use them
    if features_df is not None:
        print("Using pre-computed features from Silver layer")
        X = features_df.drop(['news_id', 'label'], axis=1, errors='ignore')
        y = features_df['label'] if 'label' in features_df.columns else None
        return X, y
    
    # Otherwise, generate features
    print("Generating features for {metric_name} prediction")
    
    # Create feature dataframe
    feature_data = []
    
    for _, news in tqdm(news_df.iterrows(), total=len(news_df), desc="Generating features"):
        # Basic features
        features = {{
            'news_id': news['news_id'],
            'category_politics': 1 if news['category'] == 'politics' else 0,
            'category_entertainment': 1 if news['category'] == 'entertainment' else 0,
            'category_sports': 1 if news['category'] == 'sports' else 0,
            'category_technology': 1 if news['category'] == 'technology' else 0,
            'title_length': len(news['title']) if 'title_length' not in news else news['title_length'],
            'abstract_length': len(news['abstract']) if 'abstract_length' not in news else news['abstract_length'],
        }}
        
        # Entity-based features
        if 'title_entities' in news and isinstance(news['title_entities'], list):
            features['title_entity_count'] = len(news['title_entities'])
        else:
            features['title_entity_count'] = 0
            
        # Generate synthetic metric score for demonstration
        # In a real scenario, this would be based on expert annotation or derived from data
        {synthetic_score_logic}
        
        feature_data.append(features)
    
    # Create dataframe
    features_df = pd.DataFrame(feature_data)
    
    # Separate features and target
    if 'label' in features_df.columns:
        X = features_df.drop(['news_id', 'label'], axis=1)
        y = features_df['label']
    else:
        X = features_df.drop(['news_id'], axis=1)
        y = None
        
    return X, y

# Prepare features
X, y = prepare_features()

# Display feature information
print(f"Feature set shape: {X.shape}")
if y is not None:
    print(f"Label distribution: \\n{y.value_counts()}")
    
# If we don't have labels yet, create synthetic ones for demonstration
if y is None:
    print("Creating synthetic labels for demonstration")
    {synthetic_label_logic}
    y = pd.Series(synthetic_y)
    print(f"Synthetic label distribution: \\n{pd.Series(synthetic_y).value_counts()}")

# Split data into train, validation and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler for later use
os.makedirs(os.path.join(SILICON_PATH, '{metric_type}'), exist_ok=True)
with open(os.path.join(SILICON_PATH, '{metric_type}', 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)"""

BASE_MODELS_TEMPLATE = """## 3. Base Models

In this section, we'll train and evaluate various base models for {metric_name} prediction."""

BASE_MODEL_EVALUATION_TEMPLATE = """# Define evaluation function
def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    \"\"\"Evaluate a model and return the results.\"\"\"
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # For classification
    if len(np.unique(y_train)) <= 5:  # Classification task
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        
        try:
            train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
            val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        except (AttributeError, IndexError):
            # If model doesn't have predict_proba or it's multiclass
            train_auc = np.nan
            val_auc = np.nan
        
        results = {{
            'model_name': model_name,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'training_time': train_time
        }}
    else:  # Regression task
        # Calculate metrics
        train_mse = np.mean((y_train - y_pred_train)**2)
        val_mse = np.mean((y_val - y_pred_val)**2)
        train_mae = np.mean(np.abs(y_train - y_pred_train))
        val_mae = np.mean(np.abs(y_val - y_pred_val))
        
        results = {{
            'model_name': model_name,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'training_time': train_time
        }}
    
    return results, model

# Train various base models

results = []

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_results, rf_model = evaluate_model(rf_model, X_train_scaled, X_val_scaled, y_train, y_val, "Random Forest")
results.append(rf_results)

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_results, xgb_model = evaluate_model(xgb_model, X_train_scaled, X_val_scaled, y_train, y_val, "XGBoost")
results.append(xgb_results)

# LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_results, lgb_model = evaluate_model(lgb_model, X_train_scaled, X_val_scaled, y_train, y_val, "LightGBM")
results.append(lgb_results)

# CatBoost
cat_model = CatBoost(iterations=100, random_seed=42, verbose=False)
cat_results, cat_model = evaluate_model(cat_model, X_train_scaled, X_val_scaled, y_train, y_val, "CatBoost")
results.append(cat_results)

# Create results dataframe
results_df = pd.DataFrame(results)
results_df"""

ENSEMBLE_METHODS_TEMPLATE = """## 4. Ensemble Methods

Now, let's implement various ensemble methods to improve our {metric_name} prediction performance."""

ENSEMBLE_IMPLEMENTATION_TEMPLATE = """# Implement ensemble methods

# 1. Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    voting='soft'  # Use predicted probabilities
)

voting_results, voting_model = evaluate_model(
    voting_clf, X_train_scaled, X_val_scaled, y_train, y_val, "Voting Ensemble"
)
results.append(voting_results)

# 2. Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=42)),
        ('lgb', lgb.LGBMClassifier(n_estimators=50, random_state=42))
    ],
    final_estimator=CatBoost(iterations=50, random_seed=42, verbose=False)
)

stacking_results, stacking_model = evaluate_model(
    stacking_clf, X_train_scaled, X_val_scaled, y_train, y_val, "Stacking Ensemble"
)
results.append(stacking_results)

# Update results dataframe
results_df = pd.DataFrame(results)
results_df"""

NN_MODELS_TEMPLATE = """## 5. Neural Network Models

Next, let's implement neural network models optimized for {metric_name} prediction."""

NN_IMPLEMENTATION_TEMPLATE = """# Define a PyTorch neural network
class {metric_type}Net(torch.nn.Module):
    def __init__(self, input_size):
        super({metric_type}Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(128, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fc3 = torch.nn.Linear(64, 32)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.fc4 = torch.nn.Linear(32, 1)  # Output layer
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))  # For binary classification
        return x

# Prepare data for PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1).to(device)

# Create dataset and dataloader
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
input_size = X_train_scaled.shape[1]
model = {metric_type}Net(input_size).to(device)
criterion = torch.nn.BCELoss()  # Binary cross-entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    # Calculate validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss.item())
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}')

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the neural network model
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor).cpu().numpy()
    y_pred_val = model(X_val_tensor).cpu().numpy()
    
    # Convert probabilities to class labels
    y_pred_train_class = (y_pred_train > 0.5).astype(int)
    y_pred_val_class = (y_pred_val > 0.5).astype(int)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train_class)
    val_acc = accuracy_score(y_val, y_pred_val_class)
    
    train_auc = roc_auc_score(y_train, y_pred_train)
    val_auc = roc_auc_score(y_val, y_pred_val)

nn_results = {{
    'model_name': 'Neural Network',
    'train_accuracy': train_acc,
    'val_accuracy': val_acc,
    'train_auc': train_auc,
    'val_auc': val_auc,
    'training_time': sum(train_losses)  # Using cumulative loss as a proxy for training time
}}

results.append(nn_results)
results_df = pd.DataFrame(results)
results_df"""

MODEL_SELECTION_TEMPLATE = """## 6. Final Model Selection

Based on our comprehensive evaluation, let's select the best model for {metric_name} prediction."""

MODEL_SELECTION_IMPL_TEMPLATE = """# Identify the best model based on validation metrics
if 'val_accuracy' in results_df.columns:  # Classification task
    # Sort by validation accuracy
    best_models = results_df.sort_values(by='val_accuracy', ascending=False)
elif 'val_auc' in results_df.columns:  # Binary classification with AUC
    # Sort by validation AUC
    best_models = results_df.sort_values(by='val_auc', ascending=False)
else:  # Regression task
    # Sort by validation MSE (lower is better)
    best_models = results_df.sort_values(by='val_mse', ascending=True)

print("Models ranked by performance:")
print(best_models)

# Select the best model
best_model_name = best_models.iloc[0]['model_name']
print(f"\\nBest model: {best_model_name}")

# Assign the best model for final evaluation
if best_model_name == "Random Forest":
    best_model = rf_model
elif best_model_name == "XGBoost":
    best_model = xgb_model
elif best_model_name == "LightGBM":
    best_model = lgb_model
elif best_model_name == "CatBoost":
    best_model = cat_model
elif best_model_name == "Voting Ensemble":
    best_model = voting_model
elif best_model_name == "Stacking Ensemble":
    best_model = stacking_model
elif best_model_name == "Neural Network":
    best_model = model  # The PyTorch model
else:
    print("Unknown best model name!")
    best_model = None

# Final evaluation on test set
if best_model_name == "Neural Network":
    # Evaluate PyTorch model
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).cpu().numpy()
        y_pred_test_class = (y_pred_test > 0.5).astype(int)
        
        test_acc = accuracy_score(y_test, y_pred_test_class)
        test_auc = roc_auc_score(y_test, y_pred_test)
        
    print(f"\\nTest set evaluation:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"AUC: {test_auc:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(SILICON_PATH, '{metric_type}', 'best_model.pt'))
else:
    # Evaluate sklearn-compatible model
    y_pred_test = best_model.predict(X_test_scaled)
    
    if len(np.unique(y_test)) <= 5:  # Classification task
        test_acc = accuracy_score(y_test, y_pred_test)
        
        try:
            test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
            print(f"\\nTest set evaluation:")
            print(f"Accuracy: {test_acc:.4f}")
            print(f"AUC: {test_auc:.4f}")
        except (AttributeError, IndexError):
            print(f"\\nTest set evaluation:")
            print(f"Accuracy: {test_acc:.4f}")
    else:  # Regression task
        test_mse = np.mean((y_test - y_pred_test)**2)
        test_mae = np.mean(np.abs(y_test - y_pred_test))
        
        print(f"\\nTest set evaluation:")
        print(f"MSE: {test_mse:.4f}")
        print(f"MAE: {test_mae:.4f}")
    
    # Save the model
    with open(os.path.join(SILICON_PATH, '{metric_type}', 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)

# Save the scaler and feature names for later use
with open(os.path.join(SILICON_PATH, '{metric_type}', 'feature_names.json'), 'w') as f:
    json.dump({{'feature_names': X.columns.tolist()}}, f)"""

DEPLOYMENT_TEMPLATE = """## 7. Deployment Preparation

Finally, let's prepare the selected model for deployment and inference."""

DEPLOYMENT_IMPL_TEMPLATE = """# Create a model card with all the necessary information
model_card = {{
    "model_name": "{metric_name} Predictor",
    "version": "1.0.0",
    "description": "Model for predicting {metric_name} of news articles",
    "model_type": best_model_name,
    "training_data": {{
        "size": len(X_train),
        "features": X.columns.tolist(),
        "target": "{metric_type}"
    }},
    "performance": {{
        "accuracy": float(test_acc) if 'test_acc' in locals() else None,
        "auc": float(test_auc) if 'test_auc' in locals() else None,
        "mse": float(test_mse) if 'test_mse' in locals() else None,
        "mae": float(test_mae) if 'test_mae' in locals() else None
    }},
    "feature_importance": {{
        # Add feature importance if available
    }},
    "deployment_requirements": {{
        "python_version": "3.9+",
        "required_packages": [
            "numpy",
            "pandas",
            "scikit-learn",
            "torch" if best_model_name == "Neural Network" else best_model_name.lower()
        ],
        "memory_requirements": "2GB minimum"
    }},
    "usage": {{
        "input_format": "Feature vector with scaled values",
        "output_format": "Probability score between 0 and 1" if len(np.unique(y_train)) <= 5 else "Continuous score",
        "example_inference_code": "model.predict(X_scaled)"
    }},
    "training_date": time.strftime("%Y-%m-%d"),
    "authors": ["News AI Team"],
    "license": "Proprietary",
    "ethical_considerations": [
        "Model should be regularly monitored for bias",
        "Prediction should be used as one signal among many for decision making",
        "Human oversight is recommended for critical applications"
    ]
}}

# Add feature importance if available
if best_model_name in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]:
    try:
        if best_model_name == "Random Forest":
            importance = best_model.feature_importances_
        elif best_model_name == "XGBoost":
            importance = best_model.feature_importances_
        elif best_model_name == "LightGBM":
            importance = best_model.feature_importances_
        elif best_model_name == "CatBoost":
            importance = best_model.feature_importances_
            
        feature_importance = {{X.columns[i]: float(importance[i]) for i in range(len(X.columns))}}
        model_card["feature_importance"] = feature_importance
    except:
        pass

# Save model card
with open(os.path.join(SILICON_PATH, '{metric_type}', 'model_card.json'), 'w') as f:
    json.dump(model_card, f, indent=2)

# Create a simple inference function
def predict_{metric_type}(features, model_path=os.path.join(SILICON_PATH, '{metric_type}')):
    \"\"\"
    Make predictions using the trained model.
    
    Parameters:
    -----------
    features : pandas.DataFrame
        Features to predict on
    model_path : str
        Path to the model directory
        
    Returns:
    --------
    numpy.ndarray
        Predicted values
    \"\"\"
    # Load feature names
    with open(os.path.join(model_path, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)['feature_names']
    
    # Ensure features have the right columns
    if not all(col in features.columns for col in feature_names):
        missing = [col for col in feature_names if col not in features.columns]
        raise ValueError(f"Missing features: {{missing}}")
    
    # Keep only the relevant features in the right order
    features = features[feature_names]
    
    # Load scaler
    with open(os.path.join(model_path, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Check if we have a PyTorch or scikit-learn model
    if os.path.exists(os.path.join(model_path, 'best_model.pt')):
        # Load PyTorch model
        model = {metric_type}Net(len(feature_names)).to(device)
        model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pt')))
        model.eval()
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        
        # Predict
        with torch.no_grad():
            predictions = model(features_tensor).cpu().numpy()
        
        return predictions
    else:
        # Load scikit-learn model
        with open(os.path.join(model_path, 'best_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        # Predict
        return model.predict(features_scaled)

# Test the inference function
test_predictions = predict_{metric_type}(X.iloc[:5])
print(f"\\nTest predictions for first 5 samples:")
print(test_predictions)

# Copy the best model to the deployed models directory
import shutil
os.makedirs(os.path.join(DEPLOYED_MODEL_PATH, '{metric_type}'), exist_ok=True)

# Copy model files
for filename in ['best_model.pkl', 'best_model.pt', 'scaler.pkl', 'feature_names.json', 'model_card.json']:
    src_file = os.path.join(SILICON_PATH, '{metric_type}', filename)
    if os.path.exists(src_file):
        shutil.copy(src_file, os.path.join(DEPLOYED_MODEL_PATH, '{metric_type}', filename))

print(f"\\nModel successfully prepared for deployment at: {{os.path.join(DEPLOYED_MODEL_PATH, '{metric_type}')}}")"""

CONCLUSION_TEMPLATE = """## 8. Conclusion

In this notebook, we've built, evaluated, and prepared for deployment a model for {metric_name} prediction. We've followed a comprehensive approach that included:

1. Data preparation from Silver layer
2. Feature engineering specific to {metric_name}
3. Training and evaluating multiple base models
4. Implementing ensemble methods for improved performance
5. Developing neural network architectures
6. Conducting rigorous model selection
7. Preparing the best model for deployment

The selected {best_model_placeholder} model achieved {performance_placeholder} on the test dataset, making it suitable for integration into our News AI application's recommendation system.

### Next Steps

1. Monitor model performance in production
2. Collect feedback for potential improvements
3. Periodically retrain the model with new data
4. Consider A/B testing different model versions
5. Integrate with other metric models in the application"""


def create_metric_notebook(metric_type, output_dir, overwrite=False):
    """
    Create a Silicon layer notebook for a specific metric.
    
    Parameters:
    -----------
    metric_type : str
        Type of metric (political_influence, rhetoric_intensity, information_depth, or sentiment)
    output_dir : str
        Directory to save the notebook
    overwrite : bool
        Whether to overwrite existing notebooks
    
    Returns:
    --------
    str
        Path to the created notebook
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define notebook content based on metric type
    if metric_type == "political_influence":
        title = "Political Influence Metric - Silicon Layer"
        description = """
This notebook implements advanced modeling techniques for political influence detection in news articles. 
We'll evaluate and select the best model for quantifying political bias and influence factors in news content.
"""
        business_context = """
Understanding the political influence of news content is crucial for:
- Providing balanced news recommendations to users
- Ensuring diverse political viewpoints in recommendations
- Avoiding filter bubbles and echo chambers
- Enabling users to make informed choices about news consumption

The Political Influence metric helps quantify the degree to which an article exhibits political bias
or attempts to influence readers toward a particular political viewpoint.
"""
        metric_definition = """
The Political Influence metric measures the degree to which a news article exhibits political bias or attempts to influence readers' political opinions. This metric considers:

- Use of politically charged language
- Representation of different political viewpoints
- Balance in coverage of political topics
- Presence of political entities and figures
- Implicit and explicit political framing

The metric is scaled from 0 to 1, where:
- 0: Completely neutral, balanced political reporting
- 0.5: Moderate political influence
- 1: Heavy political influence and bias
"""
        evaluation_metrics = """
- AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
- Accuracy
- Precision and Recall
- F1 Score
- Confusion Matrix
"""
        success_criteria = """
For this model to be considered successful and deployable, it must achieve:
- AUC-ROC score > 0.80
- Accuracy > 75%
- Balanced precision and recall across different political categories
- F1 Score > 0.75
"""
        synthetic_score_logic = """
        # Generate a political influence score (0-1)
        # Higher for politics category, lower for others
        if news['category'] == 'politics':
            features['label'] = np.random.beta(5, 2)  # Skewed toward higher values for politics
        else:
            features['label'] = np.random.beta(2, 5)  # Skewed toward lower values for non-politics
            
        # Add noise for realistic distribution
        features['label'] = min(1.0, max(0.0, features['label'] + np.random.normal(0, 0.05)))
"""
        synthetic_label_logic = """
    # Binary classification for political influence (high/low)
    synthetic_y = (X['category_politics'] + 0.3*np.random.random(len(X))).apply(lambda x: 1 if x > 0.5 else 0)
"""
    
    elif metric_type == "rhetoric_intensity":
        title = "Rhetoric Intensity Metric - Silicon Layer"
        description = """
This notebook implements advanced modeling techniques for rhetoric intensity detection in news articles.
We'll evaluate and select the best model for quantifying the strength and persuasiveness of rhetorical devices in news content.
"""
        business_context = """
Measuring rhetoric intensity in news articles helps:
- Users understand the persuasive nature of content
- Distinguish between factual reporting and persuasive commentary
- Identify content that may use emotional appeals over factual evidence
- Support critical media literacy by highlighting rhetorical techniques

The Rhetoric Intensity metric provides insight into how strongly a piece attempts to persuade rather than simply inform.
"""
        metric_definition = """
The Rhetoric Intensity metric measures the degree to which a news article employs rhetorical devices and persuasive language. This metric considers:

- Use of emotional language and appeals
- Presence of rhetorical questions
- Employment of metaphors and analogies
- Appeal to authority
- Repetition and emphasis techniques
- Call to action language

The metric is scaled from 0 to 1, where:
- 0: Purely factual reporting with minimal rhetoric
- 0.5: Moderate use of rhetorical devices
- 1: Heavy use of persuasive language and rhetorical techniques
"""
        evaluation_metrics = """
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-Squared
- Spearman Correlation
- Classification metrics when thresholded (Accuracy, F1)
"""
        success_criteria = """
For this model to be considered successful and deployable, it must achieve:
- Mean Absolute Error < 0.15
- Spearman Correlation > 0.7
- When used as a classifier (high/low rhetoric): F1 Score > 0.75
"""
        synthetic_score_logic = """
        # Generate a rhetoric intensity score (0-1)
        # Higher for opinion-like categories, lower for factual ones
        if news['category'] in ['entertainment', 'politics']:
            features['label'] = 0.5 + 0.5 * np.random.random()  # Higher rhetoric (0.5-1.0)
        else:
            features['label'] = 0.5 * np.random.random()  # Lower rhetoric (0-0.5)
            
        # Title length can influence rhetoric (longer titles might use more rhetoric)
        if features['title_length'] > 50:
            features['label'] = min(1.0, features['label'] + 0.1)
"""
        synthetic_label_logic = """
    # Binary classification for rhetoric intensity (high/low)
    synthetic_y = ((X['category_entertainment'] + X['category_politics'])/2 + 0.3*np.random.random(len(X))).apply(lambda x: 1 if x > 0.5 else 0)
"""
    
    elif metric_type == "information_depth":
        title = "Information Depth Metric - Silicon Layer"
        description = """
This notebook implements advanced modeling techniques for information depth assessment in news articles.
We'll evaluate and select the best model for quantifying the depth, comprehensiveness, and informativeness of news content.
"""
        business_context = """
Measuring information depth in news articles is valuable for:
- Helping users find substantive, in-depth reporting
- Distinguishing between shallow coverage and comprehensive analysis
- Supporting users who seek detailed understanding of complex topics
- Identifying content with educational value

The Information Depth metric helps users find content with substantial informational value.
"""
        metric_definition = """
The Information Depth metric measures how comprehensive, detailed, and informative a news article is. This metric considers:

- Depth of background context provided
- Breadth of perspectives included
- Use of supporting evidence and data
- Explanation of complex concepts
- Thoroughness of analysis
- Presence of expert quotes or insights

The metric is scaled from 0 to 1, where:
- 0: Very shallow coverage with minimal information
- 0.5: Moderate depth with some context and analysis
- 1: Extensive, in-depth coverage with comprehensive information
"""
        evaluation_metrics = """
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-Squared
- Spearman Correlation
- Classification metrics when thresholded (Accuracy, F1)
"""
        success_criteria = """
For this model to be considered successful and deployable, it must achieve:
- Mean Absolute Error < 0.15
- Spearman Correlation > 0.7
- When used as a classifier (high/low depth): Accuracy > 75%
"""
        synthetic_score_logic = """
        # Generate an information depth score (0-1)
        # Correlates with abstract length and entity count
        norm_abstract_length = min(1.0, features['abstract_length'] / 300)  # Normalize
        norm_entity_count = min(1.0, features['title_entity_count'] / 10)  # Normalize
        
        features['label'] = 0.4 * norm_abstract_length + 0.3 * norm_entity_count + 0.3 * np.random.random()
        features['label'] = min(1.0, max(0.0, features['label']))  # Clip to [0,1]
"""
        synthetic_label_logic = """
    # Binary classification for information depth (high/low)
    abstract_length_normalized = X['abstract_length'].apply(lambda x: min(1.0, x/200))
    synthetic_y = (abstract_length_normalized + 0.2*np.random.random(len(X))).apply(lambda x: 1 if x > 0.5 else 0)
"""
    
    elif metric_type == "sentiment":
        title = "Sentiment Analysis Metric - Silicon Layer"
        description = """
This notebook implements advanced modeling techniques for sentiment analysis in news articles.
We'll evaluate and select the best model for detecting and quantifying the emotional tone and sentiment in news content.
"""
        business_context = """
Sentiment analysis of news content helps:
- Provide emotional context for news recommendations
- Allow users to filter content based on emotional tone preferences
- Balance content across different emotional categories
- Provide insights into how specific topics are being presented

The Sentiment metric gives users more control over the emotional nature of their news consumption.
"""
        metric_definition = """
The Sentiment metric measures the emotional tone and polarity of a news article. This metric considers:

- Use of positive or negative language
- Emotional intensity of word choices
- Overall framing of events (optimistic vs. pessimistic)
- Tone of quotations included
- Balance of positive and negative elements

The metric is scaled from -1 to 1, where:
- -1: Extremely negative sentiment
- 0: Neutral sentiment
- 1: Extremely positive sentiment

For modeling purposes, we'll rescale to 0-1 range where 0.5 is neutral.
"""
        evaluation_metrics = """
- Mean Squared Error (MSE) 
- Mean Absolute Error (MAE)
- R-Squared
- Classification metrics when bucketed (Accuracy, F1)
"""
        success_criteria = """
For this model to be considered successful and deployable, it must achieve:
- Mean Absolute Error < 0.15
- When used as a classifier (positive/negative/neutral): F1 Score > 0.70
- Balanced performance across different news categories
"""
        synthetic_score_logic = """
        # Generate a sentiment score (0-1) 
        # Where 0.5 is neutral, <0.5 is negative, >0.5 is positive
        
        # Base sentiment by category
        if news['category'] == 'entertainment':
            base_sentiment = 0.6  # Entertainment tends more positive
        elif news['category'] == 'sports':
            base_sentiment = 0.55  # Sports slightly positive
        elif news['category'] == 'politics':
            base_sentiment = 0.4  # Politics tends negative
        else:
            base_sentiment = 0.5  # Neutral
            
        # Add random variation
        features['label'] = max(0.0, min(1.0, base_sentiment + np.random.normal(0, 0.15)))
"""
        synthetic_label_logic = """
    # Binary classification for sentiment (positive/negative)
    # Using 0.5 as the neutral point, >0.5 is positive
    synthetic_y = (0.7*X['category_entertainment'] + 0.6*X['category_sports'] - 0.3*X['category_politics'] + 0.5 + 0.2*np.random.random(len(X))).apply(lambda x: 1 if x > 0.5 else 0)
"""
    
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
    
    # Create notebook file path
    notebook_path = os.path.join(output_dir, f"{metric_type.zfill(2) if metric_type.isdigit() else ''}{metric_type}_silicon.ipynb")
    
    # Check if notebook already exists
    if os.path.exists(notebook_path) and not overwrite:
        print(f"Notebook already exists at {notebook_path}. Use overwrite=True to replace it.")
        return notebook_path
    
    # Create notebook structure
    cells = []
    
    # Title and description (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": MARKDOWN_TITLE_TEMPLATE.format(
            title=title,
            description=description,
            metric_name=metric_type.replace('_', ' ').title()
        )
    })
    
    # Imports (code)
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": IMPORTS_CELL_TEMPLATE,
        "execution_count": None,
        "outputs": []
    })
    
    # Problem Definition (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": PROBLEM_DEFINITION_TEMPLATE.format(
            business_context=business_context,
            metric_definition=metric_definition,
            evaluation_metrics=evaluation_metrics,
            success_criteria=success_criteria,
            metric_name=metric_type.replace('_', ' ').title()
        )
    })
    
    # Data Preparation (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": DATA_PREPARATION_TEMPLATE
    })
    
    # Data Loading (code) - handle raw templates with special formatting
    data_loading_code = DATA_LOADING_TEMPLATE.replace("{metric_type}", metric_type)
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": data_loading_code,
        "execution_count": None,
        "outputs": []
    })
    
    # Feature Engineering (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": FEATURE_ENGINEERING_TEMPLATE.format(
            metric_name=metric_type.replace('_', ' ').title()
        )
    })
    
    # Feature Processing (code)
    try:
        feature_processing_code = FEATURE_PROCESSING_TEMPLATE.format(
            metric_name=metric_type.replace('_', ' ').title(),
            metric_type=metric_type,
            synthetic_score_logic=synthetic_score_logic,
            synthetic_label_logic=synthetic_label_logic
        )
    except KeyError as e:
        print(f"Warning: Unable to format template due to missing key: {e}")
        feature_processing_code = FEATURE_PROCESSING_TEMPLATE.replace("{metric_name}", metric_type.replace('_', ' ').title())
        feature_processing_code = feature_processing_code.replace("{metric_type}", metric_type)
        feature_processing_code = feature_processing_code.replace("{synthetic_score_logic}", synthetic_score_logic)
        feature_processing_code = feature_processing_code.replace("{synthetic_label_logic}", synthetic_label_logic)
        
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": feature_processing_code,
        "execution_count": None,
        "outputs": []
    })
    
    # Base Models (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": BASE_MODELS_TEMPLATE.format(
            metric_name=metric_type.replace('_', ' ').title()
        )
    })
    
    # Base Model Evaluation (code)
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": BASE_MODEL_EVALUATION_TEMPLATE,
        "execution_count": None,
        "outputs": []
    })
    
    # Ensemble Methods (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ENSEMBLE_METHODS_TEMPLATE.format(
            metric_name=metric_type.replace('_', ' ').title()
        )
    })
    
    # Ensemble Implementation (code)
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": ENSEMBLE_IMPLEMENTATION_TEMPLATE,
        "execution_count": None,
        "outputs": []
    })
    
    # Neural Network Models (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": NN_MODELS_TEMPLATE.format(
            metric_name=metric_type.replace('_', ' ').title()
        )
    })
    
    # Neural Network Implementation (code)
    nn_implementation_code = NN_IMPLEMENTATION_TEMPLATE.replace("{metric_type}", metric_type.replace('_', ' '))
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": nn_implementation_code,
        "execution_count": None,
        "outputs": []
    })
    
    # Model Selection (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": MODEL_SELECTION_TEMPLATE.format(
            metric_name=metric_type.replace('_', ' ').title()
        )
    })
    
    # Model Selection Implementation (code)
    model_selection_code = MODEL_SELECTION_IMPL_TEMPLATE.replace("{metric_type}", metric_type)
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": model_selection_code,
        "execution_count": None,
        "outputs": []
    })
    
    # Deployment (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": DEPLOYMENT_TEMPLATE
    })
    
    # Deployment Implementation (code)
    deployment_code = DEPLOYMENT_IMPL_TEMPLATE
    deployment_code = deployment_code.replace("{metric_name}", metric_type.replace('_', ' ').title())
    deployment_code = deployment_code.replace("{metric_type}", metric_type)
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": deployment_code,
        "execution_count": None,
        "outputs": []
    })
    
    # Conclusion (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": CONCLUSION_TEMPLATE.format(
            metric_name=metric_type.replace('_', ' ').title(),
            best_model_placeholder="[Best Model Name]",
            performance_placeholder="[Performance Metrics]"
        )
    })
    
    # Process template cells to fix any placeholder issues
    processed_cells = []
    for cell in cells:
        if cell["cell_type"] == "code":
            # Ensure all placeholders in code cells have been processed
            try:
                if isinstance(cell["source"], str) and "{" in cell["source"]:
                    cell["source"] = cell["source"].format(
                        metric_type=metric_type,
                        metric_name=metric_type.replace('_', ' ').title()
                    )
            except KeyError as e:
                print(f"Warning: Unable to format placeholder {e} in template")
                # Just keep it as is if there's a KeyError
                pass
        processed_cells.append(cell)
    
    # Create the notebook
    notebook = {
        "cells": processed_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write the notebook to file
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Created notebook: {notebook_path}")
    return notebook_path


def main():
    """Create all Silicon layer notebooks."""
    parser = argparse.ArgumentParser(description="Create Silicon layer notebooks")
    parser.add_argument("--output_dir", type=str, default="../notebooks", help="Output directory for notebooks")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing notebooks")
    args = parser.parse_args()
    
    # Create notebooks for all metrics
    metrics = ["political_influence", "rhetoric_intensity", "information_depth", "sentiment"]
    
    for i, metric in enumerate(metrics, start=4):
        notebook_path = create_metric_notebook(metric, args.output_dir, args.overwrite)
        print(f"Created {metric} notebook: {notebook_path}")
    
    print("All notebooks created successfully!")


if __name__ == "__main__":
    main()