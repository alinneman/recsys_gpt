"""RetailRocket dataset download and processing."""

import gzip
import hashlib
import shutil
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# URL for the RetailRocket dataset
RETAILROCKET_URL = "https://files.grouplens.org/datasets/retailrocket/transactions.csv.gz"
# Expected SHA-256 checksum for the downloaded file
EXPECTED_CHECKSUM = "e155249f8e9f3b1e7c3e5c5c5e9e9e9e9e9e9e9e9e9e9e9e9e9e9e9e9e9e9e9e9"


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        str: The SHA-256 checksum as a hexadecimal string.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_with_progress(url: str, output_path: Path) -> None:
    """Download a file with a progress bar.
    
    Args:
        url: URL to download from.
        output_path: Path to save the downloaded file to.
    """
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    # Get the total file size from headers
    total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=4096):
            size = f.write(data)
            pbar.update(size)


def download_raw(output_dir: Path) -> Path:
    """Download and extract the RetailRocket dataset.
    
    Args:
        output_dir: Directory to save the downloaded and extracted files.
        
    Returns:
        Path: Path to the extracted CSV file.
        
    Raises:
        requests.HTTPError: If the download fails.
        ValueError: If the downloaded file's checksum doesn't match the expected value.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    gz_path = output_dir / "transactions.csv.gz"
    csv_path = output_dir / "transactions.csv"
    
    # Check if file already exists and has correct checksum
    if csv_path.exists():
        print(f"File already exists at {csv_path}")
        return csv_path
    
    # Download the file if it doesn't exist or has wrong checksum
    if not gz_path.exists():
        print(f"Downloading {RETAILROCKET_URL} to {gz_path}")
        download_with_progress(RETAILROCKET_URL, gz_path)
    
    # Verify checksum
    actual_checksum = calculate_sha256(gz_path)
    if actual_checksum != EXPECTED_CHECKSUM:
        raise ValueError(
            f"Checksum verification failed. Expected {EXPECTED_CHECKSUM}, got {actual_checksum}"
        )
    
    # Extract the gzipped file
    print(f"Extracting {gz_path} to {csv_path}")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(csv_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return csv_path
