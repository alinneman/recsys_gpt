"""Tests for the RetailRocketDataset."""

import os
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from recsys.data import RetailRocketDataset


def create_test_parquet(file_path: Path, num_events: int = 1000) -> None:
    """Create a test parquet file with synthetic data."""
    # Create synthetic data
    np.random.seed(42)
    
    # Generate timestamps with some randomness
    timestamps = np.cumsum(np.random.randint(60, 3600, size=num_events))  # 1 min to 1 hour between events
    
    # Create a DataFrame with the test data
    df = pd.DataFrame({
        'user_id': np.ones(num_events, dtype=int),  # All events from same user for testing
        'item_id': np.random.randint(1, 100, size=num_events),
        'event_type': np.random.randint(1, 5, size=num_events),  # 1-4 for different event types
        'device': np.random.randint(1, 4, size=num_events),  # 3 device types
        'locale': np.random.randint(1, 5, size=num_events),  # 4 locales
        'daypart': np.random.randint(1, 5, size=num_events),  # 4 day parts
        'dwell_seconds': np.random.uniform(0, 300, size=num_events),  # 0-300 seconds
        'timestamp': timestamps,
    })
    
    # Write to parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)


class TestRetailRocketDataset:
    """Test the RetailRocketDataset class."""
    
    def test_dataset_length(self):
        """Test that the dataset generates the correct number of windows."""
        with TemporaryDirectory() as temp_dir:
            # Create a test parquet file with 1000 events
            parquet_path = Path(temp_dir) / "test.parquet"
            create_test_parquet(parquet_path, num_events=1000)
            
            # Create dataset with window_size=500, stride=250
            # Expected windows for 1000 events: 0-499, 250-749, 500-999
            dataset = RetailRocketDataset(
                data_path=parquet_path,
                window_size=500,
                stride=250,
                num_workers=1,
                worker_rank=0
            )
            
            # Convert to list and check length
            items = list(islice(dataset, 0, None))
            assert len(items) == 3, f"Expected 3 windows, got {len(items)}"
    
    def test_window_content(self):
        """Test that windows have the correct content."""
        with TemporaryDirectory() as temp_dir:
            # Create a test parquet file with 600 events
            parquet_path = Path(temp_dir) / "test.parquet"
            create_test_parquet(parquet_path, num_events=600)
            
            # Create dataset with window_size=200, stride=100
            dataset = RetailRocketDataset(
                data_path=parquet_path,
                window_size=200,
                stride=100,
                num_workers=1,
                worker_rank=0
            )
            
            # Get all windows
            windows = list(islice(dataset, 0, None))
            assert len(windows) == 5  # 0-199, 100-299, 200-399, 300-499, 400-599
            
            # Check window shapes
            for window in windows:
                assert window['input_ids'].shape == (200,)
                assert window['attention_mask'].shape == (200,)
                assert window['device'].shape == (200,)
                assert window['locale'].shape == (200,)
                assert window['daypart'].shape == (200,)
                assert window['dwell_seconds'].shape == (200,)
    
    def test_worker_sharding(self):
        """Test that dataset works with multiple workers."""
        with TemporaryDirectory() as temp_dir:
            # Create a test parquet file with 2000 events
            parquet_path = Path(temp_dir) / "test.parquet"
            create_test_parquet(parquet_path, num_events=2000)
            
            # Test with 2 workers
            num_workers = 2
            all_windows = []
            
            for worker_id in range(num_workers):
                dataset = RetailRocketDataset(
                    data_path=parquet_path,
                    window_size=500,
                    stride=250,
                    num_workers=num_workers,
                    worker_rank=worker_id
                )
                # Collect windows from this worker
                worker_windows = list(islice(dataset, 0, None))
                all_windows.extend(worker_windows)
            
            # Check that we got all windows
            assert len(all_windows) >= 6  # At least 6 windows (3 per worker)
    
    def test_empty_dataset(self):
        """Test that dataset handles empty parquet files gracefully."""
        with TemporaryDirectory() as temp_dir:
            # Create a parquet file with the right schema but no rows
            parquet_path = Path(temp_dir) / "empty.parquet"
            
            # Create a table with the expected schema but no data
            schema = pa.schema([
                ('user_id', pa.int64()),
                ('item_id', pa.int64()),
                ('event_type', pa.int64()),
                ('device', pa.int64()),
                ('locale', pa.int64()),
                ('daypart', pa.int64()),
                ('dwell_seconds', pa.float64()),
                ('timestamp', pa.int64())
            ])
            table = pa.Table.from_arrays([[]] * len(schema), schema=schema)
            pq.write_table(table, parquet_path)
            
            # Should work with no errors but return no items
            dataset = RetailRocketDataset(parquet_path)
            assert len(list(dataset)) == 0
    
    def test_invalid_window_size(self):
        """Test that invalid window sizes raise appropriate errors."""
        with TemporaryDirectory() as temp_dir, pytest.raises(ValueError):
            RetailRocketDataset(
                data_path=Path(temp_dir) / "dummy.parquet",
                window_size=0,  # Invalid window size
                stride=100
            )
        
        with pytest.raises(ValueError):
            RetailRocketDataset(
                data_path=Path(temp_dir) / "dummy.parquet",
                window_size=100,
                stride=0  # Invalid stride
            )
