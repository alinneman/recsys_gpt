# RecSys GPT

A Foundation Model for Recommender Systems, inspired by modern transformer-based architectures and [Netflix's new Foundation Model](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39).

## 🚀 Features

- End-to-end recommender system pipeline
- Preprocessing and feature engineering utilities
- Scalable training pipeline
- Model deployment and serving
- Comprehensive testing and evaluation

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alinneman/recsys_gpt.git
   cd recsys_gpt
   ```

2. Install dependencies using Poetry:
   ```bash
   pip install poetry
   poetry install
   ```

3. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

## 🛠️ Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting and Linting

```bash
# Run Ruff linter
poetry run ruff check .

# Run Black formatter
poetry run black .
```

## 🚀 End-to-End Demo

Run the complete pipeline including training, embedding export, and serving recommendations:

```bash
# Install additional requirements for the demo
pip install requests

# Run the end-to-end demo (takes a few minutes)
python scripts/e2e_demo.py

# Or run with verbose output
DO_E2E=1 python scripts/e2e_demo.py
```

The demo will:
1. Train the model with 1 epoch
2. Export embeddings to Qdrant
3. Start the recommendation API
4. Test the recommendations endpoint
5. Clean up resources

## 🐳 Docker Deployment

To run the system using Docker Compose:

```bash
# Build and start all services
docker compose up -d --build

# Check service status
docker compose ps

# View logs
docker compose logs -f

# Run tests in the container
docker compose exec recsys-api pytest

# Stop and clean up
docker compose down -v
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
