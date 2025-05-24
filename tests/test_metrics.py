"""Tests for recommendation metrics."""

import pytest
import torch
import numpy as np
from recsys.metrics import ndcg_at_k, hit_rate_at_k, recall_at_k, RecommendationMetrics


def test_ndcg_at_k_perfect_prediction():
    """Test that perfect predictions give NDCG@k of 1.0."""
    # Create test data with perfect predictions
    batch_size = 3
    num_items = 10
    k = 5
    
    # Create predictions where the highest scores match the most relevant items
    predictions = torch.randn(batch_size, num_items)
    targets = torch.zeros_like(predictions)
    
    # Set one relevant item per example
    for i in range(batch_size):
        targets[i, i] = 1.0
        # Make sure the prediction is highest for the relevant item
        predictions[i, i] = 10.0
    
    # Calculate metrics
    ndcg = ndcg_at_k(predictions, targets, k=k)
    
    # Check that NDCG is 1.0 for all examples
    assert torch.allclose(ndcg, torch.ones(batch_size), atol=1e-5)


def test_ndcg_at_k_worst_prediction():
    """Test that completely wrong predictions give NDCG@k of 0."""
    batch_size = 3
    num_items = 10
    k = 5
    
    # Create predictions where the highest scores are for irrelevant items
    predictions = torch.randn(batch_size, num_items)
    targets = torch.zeros_like(predictions)
    
    # Set one relevant item per example (at the end of the list)
    for i in range(batch_size):
        targets[i, -1] = 1.0  # Relevant item is last
        # Make sure the prediction is lowest for the relevant item
        predictions[i, -1] = -10.0
        # Set high scores for the first k-1 items
        predictions[i, :k-1] = 10.0
    
    # Calculate metrics
    ndcg = ndcg_at_k(predictions, targets, k=k)
    
    # Check that NDCG is 0 for all examples (since relevant item is not in top-k)
    assert torch.allclose(ndcg, torch.zeros(batch_size), atol=1e-5)


def test_hit_rate_at_k():
    """Test hit rate at k with various scenarios."""
    # Test case 1: Relevant item is in top-k
    predictions = torch.tensor([[10.0, 0.5, 0.1, 0.2, 0.3]])
    targets = torch.tensor([[0, 1, 0, 0, 0]]).float()
    hr = hit_rate_at_k(predictions, targets, k=2)
    assert torch.allclose(hr, torch.tensor([1.0]))
    
    # Test case 2: Relevant item is not in top-k
    hr = hit_rate_at_k(predictions, targets, k=1)
    assert torch.allclose(hr, torch.tensor([0.0]))


def test_recall_at_k():
    """Test recall at k with multiple relevant items."""
    # Test case 1: 2 relevant items, both in top-2
    predictions = torch.tensor([[10.0, 9.0, 0.1, 0.2, 0.3]])
    targets = torch.tensor([[1, 1, 0, 0, 0]]).float()  # 2 relevant items
    recall = recall_at_k(predictions, targets, k=2)
    # With 2 relevant items and both in top-2, recall is 1.0
    assert torch.allclose(recall, torch.tensor([1.0]), atol=1e-4)
    
    # Test case 2: Only 1 relevant item in top-3
    predictions = torch.tensor([[10.0, 0.1, 0.2, 9.0, 0.3]])
    recall = recall_at_k(predictions, targets, k=3)
    # With 2 relevant items and 1 in top-3, recall is 0.5
    assert torch.allclose(recall, torch.tensor([0.5]), atol=1e-4)


def test_recommendation_metrics():
    """Test the RecommendationMetrics container class."""
    # Create test data
    batch_size = 2
    num_items = 5
    predictions = torch.tensor([
        [10.0, 5.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0, 5.0, 10.0]
    ])
    targets = torch.tensor([
        [1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1]
    ]).float()
    
    # Initialize metrics
    metrics = RecommendationMetrics(ks=[1, 2, 3])
    results = metrics(predictions, targets)
    
    # Check that all metrics are computed
    expected_metrics = [
        'ndcg@1', 'hit_rate@1', 'recall@1',
        'ndcg@2', 'hit_rate@2', 'recall@2',
        'ndcg@3', 'hit_rate@3', 'recall@3'
    ]
    assert set(results.keys()) == set(expected_metrics)
    
    # Check shapes
    for metric in results.values():
        assert metric.shape == (batch_size,)
    
    # Check specific values for first example
    assert torch.allclose(results['ndcg@1'][0], torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(results['hit_rate@1'][0], torch.tensor(1.0), atol=1e-4)
    # First example has 2 relevant items, one is in top-1
    assert torch.allclose(results['recall@1'][0], torch.tensor(0.5), atol=1e-4)
    
    # Check specific values for second example
    # For the second example, the top-3 items are at indices [4, 3, 2] (sorted by score)
    # The relevant items are at indices [1, 4] (0-based)
    # So for k=3, one relevant item is in top-3
    assert torch.allclose(results['ndcg@3'][1], torch.tensor(0.6131), atol=1e-4)
    assert torch.allclose(results['hit_rate@3'][1], torch.tensor(1.0), atol=1e-4)
    # 1/2 relevant items in top-3 for the second example
    assert torch.allclose(results['recall@3'][1], torch.tensor(0.5), atol=1e-4)
