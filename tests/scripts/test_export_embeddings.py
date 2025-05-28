"""Tests for the export_embeddings script."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# Add the parent directory to the path so we can import from scripts
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'scripts'))

# Import the script functions
import export_embeddings


class MockDataset(Dataset):
    """Mock dataset for testing."""
    def __init__(self, num_samples=10, seq_length=100, vocab_size=1000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, self.vocab_size, (self.seq_length,)),
            'attention_mask': torch.ones(self.seq_length, dtype=torch.long)
        }


class MockModel(torch.nn.Module):
    """Mock model for testing."""
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.transformer = MagicMock()
        
    def eval(self):
        return self
    
    def to(self, device):
        return self


def test_load_model(tmp_path):
    """Test loading a model from checkpoint."""
    # Create a dummy checkpoint
    checkpoint = {
        'state_dict': {'dummy': torch.tensor(1.0)},
        'hyper_parameters': {}
    }
    checkpoint_path = tmp_path / 'checkpoint.ckpt'
    torch.save(checkpoint, checkpoint_path)
    
    # Mock the TinyRecConfig
    mock_config = MagicMock()
    mock_config.from_pretrained.return_value = MagicMock()
    
    with patch('export_embeddings.TinyRecConfig', mock_config):
        # Mock the TinyRecModel
        with patch('export_embeddings.TinyRecModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.load_state_dict.return_value = None
            mock_model.eval.return_value = mock_model
            mock_model_class.return_value = mock_model
            
            # Call the function
            model, _ = export_embeddings.load_model(str(checkpoint_path))
            
            # Check that the model was loaded and set to eval mode
            assert model == mock_model
            mock_model.load_state_dict.assert_called_once()
            mock_model.eval.assert_called_once()


def test_compute_embeddings():
    """Test computing embeddings from a model."""
    # Create a mock model
    model = MockModel(embedding_dim=64)
    
    # Mock the transformer output
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(2, 10, 64)  # batch_size=2, seq_len=10, hidden_dim=64
    model.transformer.return_value = mock_output
    
    # Create a mock data loader
    dataset = MockDataset(num_samples=4, seq_length=10)
    data_loader = DataLoader(dataset, batch_size=2)
    
    # Call the function
    embeddings = export_embeddings.compute_embeddings(model, data_loader, device='cpu')
    
    # Check the output shapes
    assert 'user_embeddings' in embeddings
    assert 'item_embeddings' in embeddings
    assert embeddings['user_embeddings'].shape == (4, 64)  # 4 samples, 64-dim embeddings
    assert embeddings['item_embeddings'].shape == (4, 64)  # 4 samples, 64-dim embeddings


def test_save_embeddings(tmp_path):
    """Test saving embeddings to disk."""
    # Create dummy embeddings
    embeddings = {
        'user_embeddings': np.random.rand(10, 64),
        'item_embeddings': np.random.rand(20, 64)
    }
    
    # Call the function
    output_dir = tmp_path / 'test_embeddings'
    export_embeddings.save_embeddings(embeddings, str(output_dir))
    
    # Check that the files were created
    assert (output_dir / 'users.parquet').exists()
    assert (output_dir / 'items.parquet').exists()
    
    # Check the contents
    user_df = pd.read_parquet(output_dir / 'users.parquet')
    item_df = pd.read_parquet(output_dir / 'items.parquet')
    
    assert len(user_df) == 10  # 10 users
    assert len(item_df) == 20  # 20 items
    assert all(f'vec_{i}' in user_df.columns for i in range(64))
    assert all(f'vec_{i}' in item_df.columns for i in range(64))
    assert 'user_id' in user_df.columns
    assert 'item_id' in item_df.columns


def test_push_to_qdrant():
    """Test pushing embeddings to Qdrant."""
    # Create dummy embeddings
    embeddings = {
        'user_embeddings': np.random.rand(3, 64),
        'item_embeddings': np.random.rand(5, 64)
    }
    
    # Mock the Qdrant client
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Collection not found")
    
    with patch('export_embeddings.QdrantClient', return_value=mock_client) as mock_qdrant_class, \
         patch.dict('os.environ', {'QDRANT_HOST': 'test-host'}):
        
        # Call the function
        export_embeddings.push_to_qdrant(
            embeddings,
            collection_name='test_collection',
            host=None,  # Should use the one from env var
            port=12345
        )
        
        # Check that the client was created with the right parameters
        mock_qdrant_class.assert_called_once_with(host='test-host', port=12345)
        
        # Check that the collection was created
        mock_client.create_collection.assert_called_once()
        
        # Check that upsert was called twice (once for users, once for items)
        assert mock_client.upsert.call_count == 2
        
        # Check that the first call was for items
        item_call = mock_client.upsert.call_args_list[0][1]
        assert item_call['collection_name'] == 'test_collection'
        assert len(item_call['points'].ids) == 5  # 5 items
        
        # Check that the second call was for users
        user_call = mock_client.upsert.call_args_list[1][1]
        assert user_call['collection_name'] == 'test_collection'
        assert len(user_call['points'].ids) == 3  # 3 users


@patch('export_embeddings.load_model')
@patch('export_embeddings.RetailRocketDataset')
@patch('export_embeddings.DataLoader')
@patch('export_embeddings.compute_embeddings')
@patch('export_embeddings.save_embeddings')
@patch('export_embeddings.push_to_qdrant')
def test_main(mock_push, mock_save, mock_compute, mock_loader, mock_dataset, mock_load, tmp_path):
    """Test the main function with various arguments."""
    # Setup mocks
    mock_model = MagicMock()
    mock_config = MagicMock()
    mock_load.return_value = (mock_model, mock_config)
    
    mock_dataset.return_value = MagicMock()
    mock_loader.return_value = MagicMock()
    
    mock_compute.return_value = {
        'user_embeddings': np.random.rand(10, 64),
        'item_embeddings': np.random.rand(20, 64)
    }
    
    # Create a temporary checkpoint file
    checkpoint_path = tmp_path / 'model.ckpt'
    checkpoint_path.touch()
    
    # Test without --push
    with patch('sys.argv', [
        'export_embeddings.py',
        '--checkpoint', str(checkpoint_path),
        '--data_path', 'dummy.parquet',
        '--output_dir', str(tmp_path / 'embeddings')
    ]):
        export_embeddings.main()
    
    # Check that save_embeddings was called
    mock_save.assert_called_once()
    
    # Check that push_to_qdrant was not called
    mock_push.assert_not_called()
    
    # Reset mocks
    mock_save.reset_mock()
    
    # Test with --push
    with patch('sys.argv', [
        'export_embeddings.py',
        '--checkpoint', str(checkpoint_path),
        '--data_path', 'dummy.parquet',
        '--output_dir', str(tmp_path / 'embeddings'),
        '--push',
        '--qdrant_host', 'test-host',
        '--qdrant_port', '12345',
        '--collection_name', 'test_collection'
    ]):
        export_embeddings.main()
    
    # Check that both save_embeddings and push_to_qdrant were called
    mock_save.assert_called_once()
    mock_push.assert_called_once_with(
        mock_compute.return_value,
        collection_name='test_collection',
        host='test-host',
        port=12345,
        batch_size=100
    )
