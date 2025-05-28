"""Dataset implementation for RetailRocket recommendation system."""

import os
from itertools import islice
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


class RetailRocketDataset(IterableDataset):
    """PyTorch IterableDataset for the RetailRocket recommendation system.
    
    This dataset reads tokenized events from a Parquet file and yields fixed-length
    windows of tokens suitable for transformer-based recommendation models.
    
    Args:
        data_path: Path to the parquet file containing tokenized events.
        window_size: Number of tokens in each sequence window.
        stride: Number of tokens to move the window by for each sequence.
        worker_rank: Rank of the current worker (for distributed training).
        num_workers: Total number of workers (for sharding).
    """
    
    def __init__(
        self,
        data_path: Union[str, os.PathLike],
        window_size: int = 500,
        stride: int = 250,
        worker_rank: int = 0,
        num_workers: int = 1,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.window_size = window_size
        self.stride = stride
        self.worker_rank = worker_rank
        self.num_workers = num_workers
        
        # Validate inputs
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if worker_rank < 0 or worker_rank >= num_workers:
            raise ValueError(f"worker_rank must be in [0, {num_workers-1}]")
    
    def _get_worker_info(self) -> Tuple[int, int]:
        """Get worker info for sharding."""
        worker_info = get_worker_info()
        if worker_info is None:  # Single-process data loading
            return 0, 1
        return worker_info.id, worker_info.num_workers
    
    def _read_parquet(self) -> pd.DataFrame:
        """Read and sort the parquet file."""
        # Use pyarrow directly for more efficient reading
        table = pq.read_table(self.data_path)
        df = table.to_pandas()
        
        # Ensure we have the required columns
        required_columns = {'user_id', 'item_id', 'event_type', 'timestamp'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns in parquet file: {missing}")
        
        # Sort by user_id and timestamp
        df = df.sort_values(['user_id', 'timestamp'])
        return df
    
    def _get_user_windows(
        self, 
        user_events: pd.DataFrame,
        worker_id: int,
        num_workers: int
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Generate windows for a single user's events."""
        # Convert to numpy for faster processing
        item_ids = user_events['item_id'].values
        event_types = user_events['event_type'].values
        
        # Get auxiliary features if they exist
        aux_columns = [
            'device', 'locale', 'daypart', 'dwell_seconds'
        ]
        aux_features = {}
        for col in aux_columns:
            if col in user_events.columns:
                aux_features[col] = user_events[col].values
        
        # Generate windows
        for start_idx in range(0, len(item_ids) - self.window_size + 1, self.stride):
            # Skip windows that don't belong to this worker
            if (start_idx // self.stride) % num_workers != worker_id:
                continue
                
            end_idx = start_idx + self.window_size
            window_item_ids = item_ids[start_idx:end_idx]
            window_event_types = event_types[start_idx:end_idx]
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones(self.window_size, dtype=torch.long)
            
            # Create input IDs (combine item_id and event_type)
            input_ids = torch.tensor(
                window_item_ids * 10 + window_event_types,  # Simple way to combine
                dtype=torch.long
            )
            
            # Create auxiliary features tensor
            window_aux = {}
            for col, values in aux_features.items():
                window_aux[col] = torch.tensor(
                    values[start_idx:end_idx],
                    dtype=torch.long if col != 'dwell_seconds' else torch.float32
                )
            
            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                **window_aux
            }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset."""
        worker_id, num_workers = self._get_worker_info()
        
        # Read and process the parquet file
        df = self._read_parquet()
        
        # Group by user and process each user's events
        for _, user_events in df.groupby('user_id'):
            # Skip users with too few events
            if len(user_events) < 2:  # At least 2 events for meaningful sequences
                continue
                
            # Generate windows for this user
            yield from self._get_user_windows(
                user_events,
                worker_id=worker_id,
                num_workers=num_workers
            )
    
    def __len__(self) -> int:
        """Estimate the number of batches (approximate)."""
        # This is an expensive operation as it requires reading the parquet file
        # Consider caching this value if needed frequently
        table = pq.read_table(self.data_path)
        num_events = len(table)
        
        # This is an approximation since we don't know the exact number of users
        # and their sequence lengths without processing the entire dataset
        return (num_events // self.stride) * self.num_workers
