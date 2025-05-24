"""Tests for the recommendation system API."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pandas as pd
import pytest
from litestar import Litestar
from litestar.testing import TestClient
from qdrant_client.http.models import ScoredPoint

# Import the app after patching the environment
from recsys.api.main import app, EmbeddingService, RecommendationRequest, get_embedding_service


@pytest.fixture
def mock_embeddings(tmp_path):
    """Create a temporary embeddings directory with test data."""
    # Create a temporary directory for embeddings
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    
    # Create test item embeddings
    item_embeddings = pd.DataFrame({
        "item_id": ["1", "2", "3"],
        "embedding": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
    })
    item_embeddings.to_parquet(embeddings_dir / "item_embeddings.parquet")
    
    # Create test user embeddings
    user_embeddings = pd.DataFrame({
        "user_id": ["user1", "user2"],
        "embedding": [
            [0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7]
        ]
    })
    user_embeddings.to_parquet(embeddings_dir / "user_embeddings.parquet")
    
    return embeddings_dir


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = MagicMock()
    
    # Mock the search method
    mock_search_result = [
        ScoredPoint(id="1", score=0.95, payload={"item_id": "1"}, version=1),
        ScoredPoint(id="2", score=0.85, payload={"item_id": "2"}, version=1),
    ]
    client.search.return_value = mock_search_result
    
    return client


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    with patch('recsys.api.main.EmbeddingService') as mock_service:
        # Set up the mock service
        mock_instance = mock_service.return_value
        mock_instance.get_item_embedding.return_value = np.array([0.1, 0.2])
        mock_instance.get_recommendations.return_value = {
            "item_ids": ["1", "2"], 
            "scores": [0.95, 0.85]
        }
        yield mock_instance


@pytest.fixture
def test_app(mock_embedding_service):
    """Create a test app with mocked dependencies."""
    # The mock_embedding_service fixture already patches the EmbeddingService
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(app=test_app)


def test_health_check(client, mock_embedding_service):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_item_embedding(client, mock_embedding_service):
    """Test getting an item embedding."""
    # Test with existing item
    response = client.get("/embeddings/item/1")
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert data["embedding"] == [0.1, 0.2]
    mock_embedding_service.get_item_embedding.assert_called_once_with("1")
    
    # Test with non-existent item
    mock_embedding_service.get_item_embedding.return_value = None
    response = client.get("/embeddings/item/999")
    assert response.status_code == 404


def test_get_recommendations(client, mock_embedding_service):
    """Test getting recommendations."""
    # Test with valid user
    request_data = {"user_id": "user1", "k": 2}
    response = client.post("/recommendations", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "item_ids" in data
    assert "scores" in data
    assert data["item_ids"] == ["1", "2"]
    assert data["scores"] == [0.95, 0.85]
    
    # Verify the service was called correctly
    mock_embedding_service.get_recommendations.assert_called_once_with("user1", 2)
    
    # Test with non-existent user
    from litestar.exceptions import NotFoundException
    mock_embedding_service.get_recommendations.side_effect = NotFoundException("User not found")
    response = client.post("/recommendations", json={"user_id": "nonexistent", "k": 2})
    assert response.status_code == 404


def test_recommendation_validation(client, mock_embedding_service):
    """Test request validation for recommendations."""
    # Test missing user_id
    response = client.post("/recommendations", json={"k": 2})
    assert response.status_code == 400
    
    # Test invalid k value
    response = client.post("/recommendations", json={"user_id": "user1", "k": 0})
    assert response.status_code == 400
    
    response = client.post("/recommendations", json={"user_id": "user1", "k": 101})
    assert response.status_code == 400


def test_embedding_service_initialization(mock_embeddings, mock_qdrant_client):
    """Test the EmbeddingService initialization and methods."""
    with patch('recsys.api.main.QdrantClient', return_value=mock_qdrant_client):
        service = EmbeddingService(embeddings_dir=mock_embeddings)
    
        # Check that embeddings were loaded correctly
        assert service.item_embeddings is not None
        assert service.user_embeddings is not None
        
        # Check that Qdrant client was initialized
        assert service.qdrant_client is not None
        
        # Test getting item embedding
        embedding = service.get_item_embedding("1")
        assert embedding is not None
        assert len(embedding) > 0
        
        # Test getting user embedding
        user_embedding = service.get_user_embedding("user1")
        assert user_embedding is not None
        assert len(user_embedding) > 0
        
        # Test getting recommendations
        recommendations = service.get_recommendations("user1", 2)
        assert "item_ids" in recommendations
        assert "scores" in recommendations
        assert len(recommendations["item_ids"]) == 2
        assert len(recommendations["scores"]) == 2


def test_recommendation_request_model():
    """Test the RecommendationRequest model."""
    # Valid request
    request = RecommendationRequest(user_id="user1", k=5)
    assert request.user_id == "user1"
    assert request.k == 5
    
    # Test validation
    with pytest.raises(ValueError):
        RecommendationRequest(user_id="user1", k=0)  # k too small
    
    with pytest.raises(ValueError):
        RecommendationRequest(user_id="user1", k=101)  # k too large
