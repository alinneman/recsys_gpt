"""Tests for BigBird model parameter count and performance."""

import pytest
import torch
from recsys.model import TinyRecConfig, TinyRecModel


def test_parameter_count():
    """Test that the model has the expected number of parameters."""
    config = TinyRecConfig(
        vocab_size=50000,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        num_global_tokens=10,
        alibi=True
    )
    
    model = TinyRecModel(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Expected parameters: ~51.6M (actual BigBird model size)
    expected_params = 51_608_064
    
    assert num_params == expected_params, \
        f"Expected {expected_params/1e6:.1f}M parameters, got {num_params/1e6:.1f}M"


def test_forward_pass_benchmark(benchmark):
    """Benchmark the forward pass of the model."""
    config = TinyRecConfig(
        vocab_size=50000,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        num_global_tokens=10,
        alibi=True
    )
    
    model = TinyRecModel(config)
    model.eval()
    
    # Create dummy input
    batch_size = 8
    seq_length = 500
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    
    # Benchmark forward pass
    def forward_pass():
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
    
    benchmark(forward_pass)
    
    # Verify output shapes
    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask=attention_mask)
        assert logits.shape == (batch_size, seq_length, config.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__])
