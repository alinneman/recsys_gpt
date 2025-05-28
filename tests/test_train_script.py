"""Test the training script."""

import logging
import pytest
from unittest.mock import patch, MagicMock

from scripts.train import create_data_loaders, train


@pytest.fixture
def mock_retail_rocket_dataset():
    """Mock RetailRocketDataset for testing."""
    with patch('scripts.train.RetailRocketDataset') as mock_cls:
        # Create a mock dataset with 10 samples
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = [
            {
                'input_ids': MagicMock(),
                'attention_mask': MagicMock()
            }
            for _ in range(10)
        ]
        mock_cls.return_value = mock_dataset
        yield mock_dataset


def test_create_data_loaders(mock_retail_rocket_dataset, tmp_path):
    """Test that data loaders are created correctly."""
    # Create a dummy parquet file
    data_path = tmp_path / "test.parquet"
    data_path.touch()
    
    # Call the function
    train_loader, val_loader = create_data_loaders(
        str(data_path),
        max_samples=5,  # Limit to 5 samples
        batch_size=2
    )
    
    # Basic assertions
    assert train_loader is not None
    assert val_loader is not None
    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2


def test_train_fast_dev_run():
    """Test training with fast_dev_run=True."""
    with patch('scripts.train.parse_args') as mock_parse_args, \
         patch('scripts.train.create_data_loaders') as mock_create_loaders, \
         patch('pytorch_lightning.Trainer') as mock_trainer_cls, \
         patch('scripts.train.logging') as mock_logging:
        
        # Set up mocks
        mock_args = MagicMock()
        mock_args.data_path = "dummy.parquet"
        mock_args.max_samples = 10
        mock_args.batch_size = 2
        mock_args.fast_dev_run = True
        mock_args.max_epochs = 1
        mock_parse_args.return_value = mock_args
        
        # Mock data loaders
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_create_loaders.return_value = (mock_train_loader, mock_val_loader)
        
        # Mock trainer and its metrics
        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {'train_loss': MagicMock()}
        mock_trainer_cls.return_value = mock_trainer
        
        # Mock logger
        mock_logger = MagicMock()
        mock_logging.getLogger.return_value = mock_logger
        mock_logging.INFO = logging.INFO
        
        # Mock basicConfig to avoid side effects
        mock_logging.basicConfig = MagicMock()
        
        # Run the training function
        train()
        
        # Verify trainer was called with fast_dev_run
        mock_trainer_cls.assert_called_once()
        call_kwargs = mock_trainer_cls.call_args[1]
        assert call_kwargs['fast_dev_run'] is True
        
        # Verify fit was called
        mock_trainer.fit.assert_called_once()
