"""Pytest configuration and fixtures."""

import os
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory with test data."""
    # Create a temporary directory
    test_dir = tmp_path_factory.mktemp("test_data")
    
    # Create a sample parquet file with test data
    data = {
        'session_id': [1, 1, 2, 2, 3, 3],
        'item_id': [101, 102, 201, 202, 301, 302],
        'timestamp': [1000, 2000, 3000, 4000, 5000, 6000],
        'event_type': ['view', 'add_to_cart', 'view', 'purchase', 'view', 'view'],
        'device': ['mobile', 'mobile', 'desktop', 'desktop', 'mobile', 'mobile'],
        'locale': ['en-US', 'en-US', 'en-GB', 'en-GB', 'en-US', 'en-US'],
    }
    
    # Create a DataFrame and save it as a parquet file
    df = pd.DataFrame(data)
    
    # Create the processed directory if it doesn't exist
    processed_dir = test_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    # Save the test data
    parquet_path = processed_dir / 'events.parquet'
    df.to_parquet(parquet_path, index=False)
    
    return test_dir
