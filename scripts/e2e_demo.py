#!/usr/bin/env python3
"""End-to-end demo of the recommendation system.

This script demonstrates the full pipeline from training to serving recommendations.
Set the DO_E2E environment variable to run the full demo.
"""

import os
import sys
import json
import time
import subprocess
import requests
from pathlib import Path

import pytest

def run_command(cmd: str) -> None:
    """Run a shell command and raise an exception if it fails."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

def train_model() -> None:
    """Train the recommendation model."""
    print("\n=== Training Model ===")
    run_command("python scripts/train.py --epochs 1")

def export_embeddings() -> None:
    """Export embeddings to Qdrant."""
    print("\n=== Exporting Embeddings ===")
    run_command("python scripts/export_embeddings.py --push")

def start_api() -> None:
    """Start the recommendation API using Docker Compose."""
    print("\n=== Starting API ===")
    run_command("docker compose up -d recsys-api")
    
    # Wait for the API to be ready
    print("Waiting for API to be ready...")
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("API is ready!")
                return
        except requests.exceptions.RequestException:
            pass
        print(f"  Waiting... ({attempt + 1}/{max_attempts})")
        time.sleep(2)
    raise RuntimeError("API did not become ready in time")

def test_recommendations() -> None:
    """Test the recommendations endpoint."""
    print("\n=== Testing Recommendations ===")
    url = "http://localhost:8000/recommendations"
    payload = {"user_id": "user1", "k": 10}
    
    print(f"Sending request to {url} with payload: {payload}")
    response = requests.post(url, json=payload, timeout=10)
    
    print(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Response data: {json.dumps(data, indent=2)}")
        assert isinstance(data.get("item_ids"), list), "Response should contain 'item_ids' list"
        assert len(data["item_ids"]) == 10, "Should return 10 recommendations"
        assert all(isinstance(i, str) for i in data["item_ids"]), "All item IDs should be strings"
    else:
        print(f"Error response: {response.text}")
        response.raise_for_status()

def main() -> None:
    """Run the end-to-end demo."""
    # Skip if not in CI or DO_E2E is not set
    if not os.environ.get("DO_E2E") and not os.environ.get("CI"):
        print("Skipping end-to-end tests. Set DO_E2E=1 to run them.")
        pytest.skip("Skipping end-to-end tests. Set DO_E2E=1 to run them.")
        return
    
    try:
        # Change to the project root directory
        os.chdir(Path(__file__).parent.parent)
        
        # Run the full pipeline
        train_model()
        export_embeddings()
        start_api()
        test_recommendations()
        
        print("\n=== End-to-end demo completed successfully! ===")
        
    except Exception as e:
        print(f"\n=== Demo failed: {str(e)} ===", file=sys.stderr)
        sys.exit(1)
    finally:
        # Always try to clean up
        try:
            print("\n=== Cleaning up... ===")
            subprocess.run("docker compose down", shell=True, check=False)
        except Exception as e:
            print(f"Warning: Failed to clean up: {e}")

if __name__ == "__main__":
    main()
