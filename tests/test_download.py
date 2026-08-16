from pathlib import Path

import pytest

from src.download import download_month


def test_download_rejects_invalid_month(tmp_path):
    with pytest.raises(ValueError, match="Invalid year or month"):
        download_month(2024, 13, tmp_path)


def test_download_reuses_existing_file(tmp_path):
    existing = tmp_path / "yellow_tripdata_2024-04.parquet"
    existing.write_bytes(b"existing")

    result = download_month(2024, 4, tmp_path)

    assert result == existing
    assert result.read_bytes() == b"existing"
