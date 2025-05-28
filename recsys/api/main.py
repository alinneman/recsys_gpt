"""Litestar API for recommendation system."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from litestar import Litestar, get, post
from litestar.config.cors import CORSConfig
from litestar.exceptions import NotFoundException
from litestar.params import Parameter
from litestar.status_codes import HTTP_200_OK
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
from litestar.di import Provide
from litestar import get, post, Request


class RecommendationRequest(BaseModel):
    """Request model for getting recommendations."""
    user_id: str = Field(..., description="ID of the user to get recommendations for")
    k: int = Field(5, description="Number of recommendations to return", ge=1, le=100)


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    item_ids: List[str] = Field(..., description="List of recommended item IDs")
    scores: List[float] = Field(..., description="Similarity scores for each recommended item")


class EmbeddingService:
    """Service for handling embedding operations."""
    
    def __init__(self, embeddings_dir: str = "embeddings"):
        """Initialize the embedding service.
        
        Args:
            embeddings_dir: Directory containing the embedding files
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.item_embeddings = None
        self.user_embeddings = None
        self._load_embeddings()
        
        # Initialize Qdrant client from environment variables
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        self.collection_name = "recsys_embeddings"
        self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
    
    def _load_embeddings(self):
        """Load the embeddings from disk."""
        # Load item embeddings
        item_path = self.embeddings_dir / "items.parquet"
        if item_path.exists():
            df = pd.read_parquet(item_path)
            vec_columns = [col for col in df.columns if col.startswith('vec_')]
            self.item_embeddings = {
                str(row['item_id']): row[vec_columns].values.astype(np.float32)
                for _, row in df.iterrows()
            }
        
        # Load user embeddings
        user_path = self.embeddings_dir / "users.parquet"
        if user_path.exists():
            df = pd.read_parquet(user_path)
            vec_columns = [col for col in df.columns if col.startswith('vec_')]
            self.user_embeddings = {
                str(row['user_id']): row[vec_columns].values.astype(np.float32)
                for _, row in df.iterrows()
            }
    
    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get the embedding for a specific item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            The embedding vector or None if not found
        """
        if self.item_embeddings is None:
            raise RuntimeError("Item embeddings not loaded")
        return self.item_embeddings.get(str(item_id))
    
    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get the embedding for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            The embedding vector or None if not found
        """
        if self.user_embeddings is None:
            raise RuntimeError("User embeddings not loaded")
        return self.user_embeddings.get(str(user_id))
    
    def get_recommendations(self, user_id: str, k: int = 5) -> RecommendationResponse:
        """Get recommendations for a user.
        
        Args:
            user_id: ID of the user to get recommendations for
            k: Number of recommendations to return
            
        Returns:
            RecommendationResponse with item IDs and scores
        """
        # Get the user's embedding
        user_embedding = self.get_user_embedding(user_id)
        if user_embedding is None:
            raise NotFoundException(f"User {user_id} not found")
        
        # Search for similar items in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=("user", user_embedding.tolist()),
            limit=k,
            with_vectors=False,
            with_payload=False,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="vector_name",
                        match=models.MatchValue(value="item"),
                    )
                ]
            )
        )
        
        # Extract item IDs and scores
        item_ids = [str(hit.id) for hit in search_results]
        scores = [hit.score for hit in search_results]
        
        return RecommendationResponse(item_ids=item_ids, scores=scores)


async def get_embedding_service() -> EmbeddingService:
    """Get the embedding service instance."""
    # In a real app, you might want to cache this or manage its lifecycle differently
    return EmbeddingService()


@get("/health")
async def health_check(
    request: Request,
    embedding_service: EmbeddingService = Provide(get_embedding_service)
) -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@get("/embeddings/item/{item_id:str}")
async def get_item_embedding(
    item_id: str = Parameter(title="Item ID", description="ID of the item to get the embedding for"),
    embedding_service: EmbeddingService = Provide(get_embedding_service),
) -> Dict[str, List[float]]:
    """Get the embedding vector for a specific item.
    
    Args:
        item_id: ID of the item
        
    Returns:
        Dictionary with the embedding vector
    """
    embedding = embedding_service.get_item_embedding(item_id)
    if embedding is None:
        raise NotFoundException(f"Item {item_id} not found")
    return {"embedding": embedding.tolist()}


@post("/recommendations")
async def get_recommendations(
    data: RecommendationRequest,
    embedding_service: EmbeddingService = Provide(get_embedding_service),
) -> RecommendationResponse:
    """Get recommendations for a user.
    
    Args:
        data: Request data containing user ID and number of recommendations
        
    Returns:
        List of recommended item IDs and their scores
    """
    return embedding_service.get_recommendations(data.user_id, data.k)


# CORS configuration
cors_config = CORSConfig(
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the Litestar app
app = Litestar(
    route_handlers=[health_check, get_item_embedding, get_recommendations],
    cors_config=cors_config,
    debug=True,
    dependencies={
        "embedding_service": get_embedding_service
    },
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
