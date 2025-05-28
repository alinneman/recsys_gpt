"""Tests for Docker container setup."""

import os
import time
import docker
import pytest
from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent.parent

def test_docker_build():
    """Test that the Docker image builds successfully."""
    client = docker.from_env()
    
    # Build the image
    image, _ = client.images.build(
        path=str(ROOT_DIR),
        tag="recsys-api:test",
        dockerfile="Dockerfile"
    )
    
    assert image is not None
    
    # Clean up
    client.images.remove(image.id, force=True)

def test_container_health():
    """Test that the container starts and passes health checks."""
    client = docker.from_env()
    
    try:
        # Build and run the container
        container = client.containers.run(
            "recsys-api:test",
            name="recsys-api-test",
            ports={'8000/tcp': 8000},
            environment={
                'QDRANT_HOST': 'localhost',
                'QDRANT_PORT': '6333'
            },
            detach=True,
            healthcheck={
                'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                'interval': 1_000_000_000,  # 1 second in nanoseconds
                'retries': 3,
                'start_period': 1_000_000_000  # 1 second in nanoseconds
            }
        )
        
        # Wait for the container to be healthy (with timeout)
        max_attempts = 10
        for _ in range(max_attempts):
            container.reload()
            if container.status == 'running' and container.attrs['State'].get('Health', {}).get('Status') == 'healthy':
                break
            time.sleep(1)
        else:
            pytest.fail("Container did not become healthy in time")
        
        # Verify health status
        health = container.attrs['State']['Health']['Status']
        assert health == 'healthy'
        
    finally:
        # Clean up
        if 'container' in locals():
            container.stop()
            container.remove()

def test_compose_services():
    """Test that all services in docker-compose are defined correctly."""
    compose_path = ROOT_DIR / 'docker-compose.yaml'
    assert compose_path.exists(), "docker-compose.yaml not found"
    
    with open(compose_path) as f:
        content = f.read()
        
    assert 'recsys-api' in content
    assert 'qdrant' in content
    assert '8000:8000' in content
    assert '6333:6333' in content
    assert 'healthcheck' in content
