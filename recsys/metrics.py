"""Recommendation system metrics."""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np

def get_top_k(predictions: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get top-k predictions and their indices.
    
    Args:
        predictions: Tensor of shape (batch_size, num_items) with prediction scores
        k: Number of top items to select
        
    Returns:
        Tuple of (top_k_scores, top_k_indices) where:
        - top_k_scores: Tensor of shape (batch_size, k) with top-k scores
        - top_k_indices: Tensor of shape (batch_size, k) with indices of top-k items
    """
    top_k_scores, top_k_indices = torch.topk(predictions, k=k, dim=1)
    return top_k_scores, top_k_indices

def ndcg_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Calculate Normalized Discounted Cumulative Gain at K.
    
    Args:
        predictions: Tensor of shape (batch_size, num_items) with prediction scores
        targets: Tensor of shape (batch_size, num_items) with binary relevance scores
        k: Number of top items to consider
        
    Returns:
        Tensor of shape (batch_size,) with NDCG@k for each example in the batch
    """
    _, top_k_indices = get_top_k(predictions, k)
    batch_indices = torch.arange(predictions.size(0), device=predictions.device).unsqueeze(1)
    
    # Get relevance scores for top-k items
    rel = targets[batch_indices, top_k_indices]
    
    # Calculate DCG: sum(rel_i / log2(i + 1))
    discounts = torch.log2(torch.arange(2, k + 2, device=predictions.device).float())
    dcg = (rel / discounts).sum(dim=1)
    
    # Calculate ideal DCG (IDCG)
    ideal_rel, _ = torch.topk(targets, k=k, dim=1)
    idcg = (ideal_rel / discounts).sum(dim=1)
    
    # Avoid division by zero
    idcg = torch.max(idcg, torch.ones_like(idcg) * 1e-8)
    
    return dcg / idcg

def hit_rate_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Calculate Hit Rate at K.
    
    Args:
        predictions: Tensor of shape (batch_size, num_items) with prediction scores
        targets: Tensor of shape (batch_size, num_items) with binary relevance scores
        k: Number of top items to consider
        
    Returns:
        Tensor of shape (batch_size,) with HitRate@k for each example in the batch
    """
    _, top_k_indices = get_top_k(predictions, k)
    batch_indices = torch.arange(predictions.size(0), device=predictions.device).unsqueeze(1)
    
    # Check if any of the top-k items are relevant
    hits = targets[batch_indices, top_k_indices].sum(dim=1) > 0
    return hits.float()

def recall_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Calculate Recall at K.
    
    Args:
        predictions: Tensor of shape (batch_size, num_items) with prediction scores
        targets: Tensor of shape (batch_size, num_items) with binary relevance scores
        k: Number of top items to consider
        
    Returns:
        Tensor of shape (batch_size,) with Recall@k for each example in the batch
    """
    _, top_k_indices = get_top_k(predictions, k)
    batch_indices = torch.arange(predictions.size(0), device=predictions.device).unsqueeze(1)
    
    # Number of relevant items in top-k
    hits = targets[batch_indices, top_k_indices].sum(dim=1)
    # Total number of relevant items
    total_relevant = targets.sum(dim=1)
    
    # Avoid division by zero
    recall = hits / total_relevant.clamp(min=1e-8)
    return recall

class RecommendationMetrics:
    """Container for recommendation metrics."""
    
    def __init__(self, ks: List[int] = None):
        """Initialize with list of k values to compute metrics for."""
        self.ks = ks or [1, 5, 10]
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute all metrics for the given predictions and targets.
        
        Args:
            predictions: Tensor of shape (batch_size, num_items) with prediction scores
            targets: Tensor of shape (batch_size, num_items) with binary relevance scores
            
        Returns:
            Dictionary with metric names as keys and tensors of shape (batch_size,) as values
        """
        metrics = {}
        
        for k in self.ks:
            metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, targets, k)
            metrics[f'hit_rate@{k}'] = hit_rate_at_k(predictions, targets, k)
            metrics[f'recall@{k}'] = recall_at_k(predictions, targets, k)
        
        return metrics
