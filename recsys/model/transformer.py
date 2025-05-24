"""Transformer-based recommendation model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import BigBirdConfig, BigBirdModel
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (
    RetrievalNormalizedDCG,
    RetrievalRecall,
    RetrievalHitRate
)

from recsys.metrics import RecommendationMetrics


@dataclass
class TinyRecConfig:
    """Configuration for the TinyRec model.
    
    Args:
        vocab_size: Size of the vocabulary (number of unique items)
        hidden_size: Dimension of the model embeddings
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: Size of the feed-forward intermediate layer
        hidden_dropout_prob: Dropout probability for hidden layers
        attention_probs_dropout_prob: Dropout probability for attention probs
        max_position_embeddings: Maximum sequence length
        num_global_tokens: Number of global tokens for BigBird
        alibi: Whether to use ALiBi attention
    """
    vocab_size: int = 50000
    max_position_embeddings: int = 1024
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    num_global_tokens: int = 10
    alibi: bool = True


class TinyRecModel(pl.LightningModule):
    """A tiny transformer-based recommendation model.
    
    This model consists of:
    1. Token and positional embeddings
    2. A single transformer encoder layer
    3. Tied output embeddings (LM head)
    """
    
    def __init__(self, config: Optional[TinyRecConfig] = None):
        super().__init__()
        self.config = config or TinyRecConfig()
        
        # Initialize BigBird model
        bb_config = BigBirdConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            num_random_blocks=3,  # Default for BigBird
            block_size=64,  # Default for BigBird
            use_bias=True,
            pad_token_id=0,
            attention_type="original_full",
            use_cache=True,
            alibi=config.alibi,
            num_global_tokens=config.num_global_tokens,
        )
        
        self.transformer = BigBirdModel(bb_config)
        
        # Language modeling head (tied to input embeddings)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        
        # Initialize metrics
        self.metrics = RecommendationMetrics(ks=[1, 5, 10])
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights and apply weight tying."""
        # Tie weights between input embeddings and output layer
        self.lm_head.weight = self.transformer.embeddings.word_embeddings.weight
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Target token IDs for language modeling
            
        Returns:
            Tuple of (logits, loss) where loss is None if labels not provided
        """
        # Pass through BigBird
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get sequence output
        sequence_output = outputs.last_hidden_state
        
        # Get logits from language modeling head
        logits = self.lm_head(sequence_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens and compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        labels = batch['labels'] if 'labels' in batch else None
        
        # Get model outputs
        logits, loss = self(input_ids, attention_mask, labels)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step with metrics computation."""
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        labels = batch['labels'] if 'labels' in batch else None
        
        # Get model outputs
        logits, loss = self(input_ids, attention_mask, labels)
        
        # Compute metrics if we have labels
        if labels is not None:
            # Convert logits to scores (softmax for multi-class)
            scores = F.softmax(logits, dim=-1)
            
            # Create binary targets (1 for positive, 0 for negative)
            # In a real scenario, you might have multi-label targets
            targets = (labels > 0).float()
            
            # Compute metrics
            metrics = self.metrics(scores, targets)
            
            # Log metrics
            for name, value in metrics.items():
                self.log(f'val_{name}', value.mean(), on_epoch=True, prog_bar=True, logger=True)
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
        
    def test_step(self, batch, batch_idx):
        """Test step with metrics computation."""
        # Same as validation step but with test prefix
        return self.validation_step(batch, batch_idx)
        
    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
    
    def predict_next_item(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Predict the next item in the sequence.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Predicted next item IDs of shape (batch_size,)
        """
        # Get logits for the last position
        logits, _ = self(input_ids, attention_mask)
        last_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        
        # Get the most likely next token
        next_items = last_logits.argmax(dim=-1)  # (batch_size,)
        return next_items
