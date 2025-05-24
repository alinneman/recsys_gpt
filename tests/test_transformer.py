"""Tests for the transformer model with BigBird."""

import pytest
import torch

from recsys.model import TinyRecConfig, TinyRecModel


def test_model_initialization():
    """Test model initialization and parameter sharing."""
    config = TinyRecConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=512,
        num_global_tokens=5,
        alibi=True
    )
    
    model = TinyRecModel(config)
    
    # Test parameter sharing between input embeddings and output layer
    assert model.lm_head.weight.data_ptr() == \
           model.transformer.embeddings.word_embeddings.weight.data_ptr()


def test_forward_pass():
    """Test the forward pass of the TinyRec model with BigBird."""
    config = TinyRecConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=512,
        num_global_tokens=5,
        alibi=True
    )
    
    model = TinyRecModel(config)
    model.eval()
    
    # Test forward pass
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    
    # Test without labels
    logits, _ = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (batch_size, seq_length, config.vocab_size)
    
    # Test with labels
    labels = input_ids.clone()
    logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert logits.shape == (batch_size, seq_length, config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # Scalar loss


def test_model_config():
    """Test the model configuration."""
    # Test default config
    default_config = TinyRecConfig()
    assert default_config.vocab_size == 50000
    assert default_config.hidden_size == 512
    assert default_config.num_attention_heads == 8
    assert default_config.num_hidden_layers == 8
    assert default_config.alibi is True
    
    # Test custom config
    custom_config = TinyRecConfig(
        vocab_size=10000,
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=6,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=2048,
        num_global_tokens=20,
        alibi=False
    )
    assert custom_config.vocab_size == 10000
    assert custom_config.hidden_size == 256
    assert custom_config.num_attention_heads == 4
    assert custom_config.num_hidden_layers == 6
    assert custom_config.hidden_dropout_prob == 0.2
    assert custom_config.attention_probs_dropout_prob == 0.1
    assert custom_config.max_position_embeddings == 2048
    assert custom_config.num_global_tokens == 20
    assert custom_config.alibi is False
