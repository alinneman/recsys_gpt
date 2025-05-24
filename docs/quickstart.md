# Quick Start Guide

## Overview

RecSys GPT is a foundation model for recommender systems that leverages transformer-based architectures to provide high-quality recommendations. This guide will help you get started with the basic usage of the system.

## Prerequisites

- Python 3.10+
- Poetry (for dependency management)
- Docker and Docker Compose (for containerized deployment)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/recsys_gpt.git
   cd recsys_gpt
   ```

2. Install dependencies:
   ```bash
   pip install poetry
   poetry install
   ```

## Basic Commands

### Training

Train the recommendation model:
```bash
# Basic training with default parameters
python scripts/train.py

# Train with custom parameters
python scripts/train.py --epochs 10 --batch_size 32 --learning_rate 0.001
```

### Inference

Get recommendations using the trained model:
```bash
# Generate recommendations for a user
python scripts/inference.py --user_id 123 --num_recommendations 10
```

### Exporting Embeddings

Export model embeddings to Qdrant vector database:
```bash
# Export embeddings
python scripts/export_embeddings.py

# Export and push to Qdrant
python scripts/export_embeddings.py --push
```

### API Server

Start the recommendation API:
```bash
# Start the API server
uvicorn recsys.api.main:app --reload

# Or using Docker Compose
docker compose up -d
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py -v
```

### Code Quality

```bash
# Run linter
ruff check .

# Auto-format code
black .

# Check types
mypy .
```

## Next Steps

- Explore the API documentation at `http://localhost:8000/docs`
- Check out the example notebooks in the `examples/` directory
- Read the [API Reference](./api.md) for detailed documentation
