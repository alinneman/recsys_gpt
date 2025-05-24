"""Script to export user and item embeddings from a trained model."""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from recsys.data import RetailRocketDataset
from recsys.model import TinyRecModel, TinyRecConfig


def load_model(checkpoint_path: str) -> Tuple[TinyRecModel, TinyRecConfig]:
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        
    Returns:
        Tuple of (model, config)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load the model config first
    config = TinyRecConfig.from_pretrained(os.path.dirname(checkpoint_path))
    
    # Initialize the model with the config
    model = TinyRecModel(config)
    
    # Load the weights
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model, config


def compute_embeddings(
    model: TinyRecModel,
    data_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, np.ndarray]:
    """Compute user and item embeddings from the model.
    
    Args:
        model: The trained model
        data_loader: DataLoader for the dataset
        device: Device to run the model on
        
    Returns:
        Dictionary with 'user_embeddings' and 'item_embeddings' as numpy arrays
    """
    model = model.to(device)
    
    all_user_embeddings = []
    all_item_embeddings = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Get user and item embeddings
            outputs = model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Use the last hidden state as item embeddings
            item_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            
            # Use the [CLS] token embedding as user embedding
            user_embeddings = outputs.last_hidden_state[:, 0, :]
            
            all_user_embeddings.append(user_embeddings.cpu().numpy())
            all_item_embeddings.append(item_embeddings.cpu().numpy())
    
    return {
        'user_embeddings': np.concatenate(all_user_embeddings, axis=0),
        'item_embeddings': np.concatenate(all_item_embeddings, axis=0)
    }


def save_embeddings(
    embeddings: Dict[str, np.ndarray],
    output_dir: str = 'embeddings',
    prefix: str = ''
) -> None:
    """Save embeddings to parquet files.
    
    Args:
        embeddings: Dictionary containing 'user_embeddings' and 'item_embeddings'
        output_dir: Directory to save the embeddings
        prefix: Prefix for the output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save item embeddings
    item_df = pd.DataFrame(embeddings['item_embeddings'])
    item_df.columns = [f'vec_{i}' for i in range(item_df.shape[1])]
    item_df['item_id'] = range(len(item_df))
    item_df.to_parquet(os.path.join(output_dir, f'{prefix}items.parquet'), index=False)
    
    # Save user embeddings
    user_df = pd.DataFrame(embeddings['user_embeddings'])
    user_df.columns = [f'vec_{i}' for i in range(user_df.shape[1])]
    user_df['user_id'] = range(len(user_df))
    user_df.to_parquet(os.path.join(output_dir, f'{prefix}users.parquet'), index=False)


def push_to_qdrant(
    embeddings: Dict[str, np.ndarray],
    collection_name: str = 'recsys_embeddings',
    host: Optional[str] = None,
    port: int = 6333,
    batch_size: int = 100
) -> None:
    """Push embeddings to Qdrant vector database.
    
    Args:
        embeddings: Dictionary containing 'user_embeddings' and 'item_embeddings'
        collection_name: Name of the Qdrant collection
        host: Qdrant server host
        port: Qdrant server port
        batch_size: Batch size for upsert operations
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    
    if host is None:
        host = os.getenv('QDRANT_HOST', 'localhost')
    
    client = QdrantClient(host=host, port=port)
    
    # Create collection if it doesn't exist
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        # Get embedding dimension from the first item
        dim = embeddings['item_embeddings'].shape[1]
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                'user': models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                ),
                'item': models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                )
            }
        )
    
    # Upload item embeddings
    item_vectors = embeddings['item_embeddings'].tolist()
    item_ids = list(range(len(item_vectors)))
    
    client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=item_ids,
            vectors={
                'item': item_vectors
            }
        )
    )
    
    # Upload user embeddings with offset to avoid ID collision
    user_vectors = embeddings['user_embeddings'].tolist()
    user_ids = [i + 1_000_000 for i in range(len(user_vectors))]  # Offset to avoid ID collision
    
    client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=user_ids,
            vectors={
                'user': user_vectors
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description='Export embeddings from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='embeddings',
                      help='Directory to save embeddings')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for inference')
    parser.add_argument('--push', action='store_true',
                      help='Push embeddings to Qdrant')
    parser.add_argument('--qdrant_host', type=str, default=None,
                      help='Qdrant server host')
    parser.add_argument('--qdrant_port', type=int, default=6333,
                      help='Qdrant server port')
    parser.add_argument('--collection_name', type=str, default='recsys_embeddings',
                      help='Qdrant collection name')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, _ = load_model(args.checkpoint)
    
    # Create data loader
    dataset = RetailRocketDataset(
        data_path=args.data_path,
        window_size=500,
        stride=500,  # No overlap for embedding generation
        max_length=500
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Compute embeddings
    print("Computing embeddings...")
    embeddings = compute_embeddings(model, data_loader)
    
    # Save embeddings
    print(f"Saving embeddings to {args.output_dir}")
    save_embeddings(embeddings, output_dir=args.output_dir)
    
    # Push to Qdrant if requested
    if args.push:
        print(f"Pushing embeddings to Qdrant at {args.qdrant_host or 'localhost'}:{args.qdrant_port}")
        push_to_qdrant(
            embeddings,
            collection_name=args.collection_name,
            host=args.qdrant_host,
            port=args.qdrant_port
        )
        print("Embeddings pushed to Qdrant successfully")
    
    print("Done!")


if __name__ == "__main__":
    main()
