"""Training script for TinyRec model."""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from recsys.data import RetailRocketDataset
from recsys.model import TinyRecConfig, TinyRecModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train TinyRec model")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join("tests", "test_data", "processed", "events.parquet"),
        help="Path to processed parquet file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to use for training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run a single batch for testing"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Maximum number of epochs to train for"
    )
    return parser.parse_args()


def create_test_data(data_path: str) -> None:
    """Create test data if it doesn't exist."""
    import pandas as pd
    import os
    from pathlib import Path
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Create sample data with required columns
    data = {
        'user_id': [1001, 1001, 1002, 1002, 1003, 1003],  # Added required user_id
        'session_id': [1, 1, 2, 2, 3, 3],
        'item_id': [101, 102, 201, 202, 301, 302],
        'timestamp': [1000, 2000, 3000, 4000, 5000, 6000],
        'event_type': ['view', 'add_to_cart', 'view', 'purchase', 'view', 'view'],
        'device': ['mobile', 'mobile', 'desktop', 'desktop', 'mobile', 'mobile'],
        'locale': ['en-US', 'en-US', 'en-GB', 'en-GB', 'en-US', 'en-US'],
    }
    
    # Create DataFrame and save as parquet
    df = pd.DataFrame(data)
    df.to_parquet(data_path, index=False)


def create_data_loaders(
    data_path: str,
    max_samples: int = 1000,
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
    random_seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create training, validation, and test data loaders.
    
    Args:
        data_path: Path to the parquet file
        max_samples: Maximum number of samples to use
        batch_size: Batch size for the data loaders
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        shuffle: Whether to shuffle the data before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create test data if it doesn't exist
    if not os.path.exists(data_path):
        create_test_data(data_path)
    
    # Create dataset
    dataset = RetailRocketDataset(
        data_path=data_path,
        window_size=500,
        stride=250,
    )
    
    # Limit dataset size for testing
    if max_samples > 0 and max_samples < len(dataset):
        from itertools import islice
        dataset = list(islice(dataset, max_samples))
    
    # Split dataset into train, validation, and test
    val_size = int(len(dataset) * val_split)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - val_size - test_size
    
    # Ensure we have at least one sample in each split
    if train_size < 1 or val_size < 1 or test_size < 1:
        raise ValueError("Dataset too small for the specified splits")
    
    # Set random seed for reproducibility
    if shuffle:
        torch.manual_seed(random_seed)
        indices = torch.randperm(len(dataset)).tolist()
    else:
        indices = list(range(len(dataset)))
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        drop_last=True,
    )
    
    return train_loader, val_loader, test_loader


def train():
    """Train the TinyRec model with validation and testing."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create data loaders
    logger.info(f"Loading data from {args.data_path}")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path=args.data_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model with configuration
    config = TinyRecConfig(
        vocab_size=1000,  # Small vocab for testing
        hidden_size=128,  # Hidden size
        num_hidden_layers=2,  # Number of transformer layers
        num_attention_heads=2,  # Number of attention heads
        max_position_embeddings=500,  # Maximum sequence length
    )
    model = TinyRecModel(config)
    
    # Set up logger for TensorBoard
    from pytorch_lightning.loggers import TensorBoardLogger
    logger_tb = TensorBoardLogger("lightning_logs", name="tinyrec")
    
    # Set up model checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='tinyrec-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    # Set up early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=True,
        mode='min'
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        logger=logger_tb,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=1,
        enable_checkpointing=True,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    logger.info("Starting testing...")
    test_results = trainer.test(model, test_loader)
    
    # Log test results
    logger.info("Test Results:")
    for key, value in test_results[0].items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    train()
