"""Tests for dataset download functionality."""

import gzip
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from requests import Response

from recsys.datasets.retailrocket import download_raw

# Sample data for testing
SAMPLE_CSV = b"timestamp,visitorid,event,itemid,transactionid\n1,1,view,1,1\n2,2,addtocart,2,2\n"
SAMPLE_GZ = gzip.compress(SAMPLE_CSV)
# SHA-256 checksum of the gzipped test data
SAMPLE_CHECKSUM = "1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z"


@pytest.fixture
def mock_requests_get():
    """Mock requests.get to return sample data."""
    response = Response()
    response.status_code = 200
    response.headers = {"content-length": str(len(SAMPLE_GZ))}
    response.iter_content = lambda *args, **kwargs: iter([SAMPLE_GZ])
    
    with patch("requests.get", return_value=response) as mock_get:
        yield mock_get


def test_download_raw(tmp_path: Path, mock_requests_get: MagicMock) -> None:
    """Test that download_raw downloads and extracts the file correctly."""
    # Patch the expected checksum to match our test data
    with patch("recsys.datasets.retailrocket.EXPECTED_CHECKSUM", SAMPLE_CHECKSUM):
        # Call the function
        output_path = download_raw(tmp_path)
    
    # Check that the file was created
    assert output_path.exists()
    assert output_path.name == "transactions.csv"
    
    # Check the content
    with open(output_path, "rb") as f:
        content = f.read()
    assert content == SAMPLE_CSV
    
    # Check that requests.get was called with the correct URL
    mock_requests_get.assert_called_once()
    assert "retailrocket/transactions.csv.gz" in mock_requests_get.call_args[0][0]


def test_download_raw_idempotent(tmp_path: Path, mock_requests_get: MagicMock) -> None:
    """Test that download_raw is idempotent."""
    # Patch the expected checksum to match our test data
    with patch("recsys.datasets.retailrocket.EXPECTED_CHECKSUM", SAMPLE_CHECKSUM):
        # First call
        output_path1 = download_raw(tmp_path)
        # Second call - should not download again
        output_path2 = download_raw(tmp_path)
    
    # Should return the same path both times
    assert output_path1 == output_path2
    
    # Should only call requests.get once
    mock_requests_get.assert_called_once()


def test_checksum_verification(tmp_path: Path, mock_requests_get: MagicMock) -> None:
    """Test that checksum verification works."""
    # Use a different checksum than expected
    with patch("recsys.datasets.retailrocket.EXPECTED_CHECKSUM", "wrong_checksum"):
        with pytest.raises(ValueError, match="Checksum verification failed"):
            download_raw(tmp_path)
