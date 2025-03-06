"""
Training script for news recommendation models.
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from news_ai_app.config import settings
from news_ai_app.data import MINDDataset, preprocess_news_text
from news_ai_app.models.recommender import (HybridNewsRecommender,
                                            load_pretrained_recommender)

logger = logging.getLogger(__name__)


class NewsRecommendationDataset(Dataset):
    """Dataset for news recommendation model training."""
    
    def __init__(
        self,
        user_histories: Dict[str, List[str]],
        news_features: Dict[str, torch.Tensor],
        user_impressions: List[Dict],
        history_size: int = 50
    ):
        """
        Initialize dataset.
        
        Args:
            user_histories: Dictionary mapping user IDs to lists of news IDs they've read
            news_features: Dictionary mapping news IDs to feature tensors
            user_impressions: List of impression dictionaries with user_id, news_id, and label
            history_size: Maximum number of news items in user history
        """
        self.user_histories = user_histories
        self.news_features = news_features
        self.impressions = user_impressions
        self.history_size = history_size
        
    def __len__(self):
        return len(self.impressions)
    
    def __getitem__(self, idx):
        impression = self.impressions[idx]
        user_id = impression["user_id"]
        candidate_id = impression["news_id"]
        label = impression["label"]
        
        # Get user history
        history = self.user_histories.get(user_id, [])[:self.history_size]
        
        # Pad if necessary
        if len(history) < self.history_size:
            history = history + ["PAD"] * (self.history_size - len(history))
        
        # Convert to tensors
        history_tensors = []
        for news_id in history:
            if news_id == "PAD":
                # Use zero tensor for padding
                history_tensors.append(torch.zeros_like(next(iter(self.news_features.values()))))
            else:
                history_tensors.append(self.news_features.get(news_id, torch.zeros_like(next(iter(self.news_features.values())))))
        
        history_tensor = torch.stack(history_tensors)
        candidate_tensor = self.news_features.get(candidate_id, torch.zeros_like(next(iter(self.news_features.values()))))
        
        # Create mask for valid history items
        mask = torch.tensor([news_id != "PAD" for news_id in history], dtype=torch.bool)
        
        return {
            "history": history_tensor,
            "candidate": candidate_tensor,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.float)
        }


def prepare_impression_data(behaviors_df: pd.DataFrame) -> List[Dict]:
    """
    Prepare impression data for model training.
    
    Args:
        behaviors_df: DataFrame with user behaviors
        
    Returns:
        List of impression dictionaries with user_id, news_id, and label
    """
    impressions = []
    
    for _, row in behaviors_df.iterrows():
        user_id = row["user_id"]
        
        # Process impressions
        for impression in row["impressions"]:
            news_id = impression["news_id"]
            clicked = impression["clicked"]
            
            impressions.append({
                "user_id": user_id,
                "news_id": news_id,
                "label": clicked
            })
    
    return impressions


def encode_news_features(
    news_df: pd.DataFrame,
    recommender: HybridNewsRecommender
) -> Dict[str, torch.Tensor]:
    """
    Encode news articles into feature vectors.
    
    Args:
        news_df: DataFrame with news articles
        recommender: Recommender model to use for encoding
        
    Returns:
        Dictionary mapping news IDs to feature tensors
    """
    news_features = {}
    
    # Process in batches to avoid memory issues
    batch_size = 32
    for i in range(0, len(news_df), batch_size):
        batch_df = news_df.iloc[i:i+batch_size]
        
        titles = batch_df["title"].tolist()
        abstracts = batch_df["abstract"].fillna("").tolist()
        title_entities = batch_df["title_entities"].tolist()
        abstract_entities = batch_df["abstract_entities"].tolist()
        
        # Encode batch
        with torch.no_grad():
            embeddings = recommender.encode_news_batch(
                titles=titles,
                abstracts=abstracts,
                title_entities=title_entities,
                abstract_entities=abstract_entities
            )
        
        # Store results
        for j, news_id in enumerate(batch_df["news_id"].tolist()):
            news_features[news_id] = embeddings[j]
    
    return news_features


def train_epoch(
    model: HybridNewsRecommender,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: Recommender model to train
        train_dataloader: DataLoader with training data
        optimizer: Optimizer to use
        device: Device to use for training
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Training loop
    for batch in tqdm(train_dataloader, desc="Training"):
        # Move batch to device
        history = batch["history"].to(device)
        candidate = batch["candidate"].unsqueeze(1).to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        outputs = model(
            user_history_embeddings=history,
            candidate_news_embeddings=candidate,
            history_mask=mask
        ).squeeze(1)
        
        # Calculate loss
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_dataloader)


def evaluate(
    model: HybridNewsRecommender,
    eval_dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the model.
    
    Args:
        model: Recommender model to evaluate
        eval_dataloader: DataLoader with evaluation data
        device: Device to use for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to device
            history = batch["history"].to(device)
            candidate = batch["candidate"].unsqueeze(1).to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(
                user_history_embeddings=history,
                candidate_news_embeddings=candidate,
                history_mask=mask
            ).squeeze(1)
            
            # Convert to probabilities
            probs = torch.sigmoid(outputs)
            
            # Store predictions and labels
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_preds)
    
    # Accuracy
    predictions = (all_preds > 0.5).astype(int)
    accuracy = (predictions == all_labels).mean()
    
    return {
        "auc": auc,
        "accuracy": accuracy
    }


def train_recommender(
    mind_dataset_path: Union[str, Path] = None,
    output_path: Union[str, Path] = None,
    batch_size: int = 64,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: Optional[torch.device] = None
) -> HybridNewsRecommender:
    """
    Train the news recommender model.
    
    Args:
        mind_dataset_path: Path to MIND dataset
        output_path: Path to save trained model
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to use for training
        
    Returns:
        Trained recommender model
    """
    # Set default paths if not provided
    if mind_dataset_path is None:
        mind_dataset_path = settings.model.mind_dataset_path
    
    if output_path is None:
        output_path = Path("./models/recommender.pt")
    else:
        output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading MIND dataset...")
    
    # Load train split
    train_dataset = MINDDataset(dataset_path=mind_dataset_path, split="train")
    train_news_df, train_behaviors_df, entity_embeddings, relation_embeddings = train_dataset.load_all()
    
    # Load validation split
    val_dataset = MINDDataset(dataset_path=mind_dataset_path, split="dev")
    val_news_df, val_behaviors_df, _, _ = val_dataset.load_all()
    
    # Preprocess news text
    logger.info("Preprocessing news text...")
    train_news_df = preprocess_news_text(train_news_df)
    val_news_df = preprocess_news_text(val_news_df)
    
    # Initialize model
    logger.info("Initializing model...")
    model = HybridNewsRecommender(
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings
    )
    model.to(device)
    
    # Encode news features
    logger.info("Encoding news features...")
    train_news_features = encode_news_features(train_news_df, model)
    val_news_features = encode_news_features(val_news_df, model)
    
    # Prepare impression data
    logger.info("Preparing impression data...")
    train_impressions = prepare_impression_data(train_behaviors_df)
    val_impressions = prepare_impression_data(val_behaviors_df)
    
    # Get user histories
    train_user_histories = train_dataset.get_user_histories()
    val_user_histories = val_dataset.get_user_histories()
    
    # Create datasets
    logger.info("Creating datasets...")
    train_data = NewsRecommendationDataset(
        user_histories=train_user_histories,
        news_features=train_news_features,
        user_impressions=train_impressions
    )
    
    val_data = NewsRecommendationDataset(
        user_histories=val_user_histories,
        news_features=val_news_features,
        user_impressions=val_impressions
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=batch_size
    )
    
    val_dataloader = DataLoader(
        val_data,
        sampler=SequentialSampler(val_data),
        batch_size=batch_size
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    logger.info("Starting training...")
    best_auc = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        logger.info(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        metrics = evaluate(model, val_dataloader, device)
        logger.info(f"Validation metrics: {metrics}")
        
        # Save best model
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), output_path)
            logger.info(f"Saved new best model with AUC: {best_auc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(output_path))
    
    return model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Train the model
    model = train_recommender()