"""
test_asset_writer.py
====================

Unit tests for AssetExtractionWriter (Phase 2 atomic persistence).
"""

from pathlib import Path
import numpy as np
import pytest

from modules.asset_extraction_components.asset_writer import AssetExtractionWriter
from modules.asset_extraction_exceptions import AssetWriteError


def test_write_image_success_and_purge(tmp_path: Path):
    writer = AssetExtractionWriter()
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    file_path = tmp_path / "sub" / "test.png"

    ok = writer.write_image(image, file_path)
    assert ok is True
    assert file_path.exists()
    assert file_path.stat().st_size > 0

    # Test purge directory
    purged = writer.purge_directory(tmp_path / "sub")
    assert purged is True
    assert not file_path.exists()


def test_write_image_invalid_array(tmp_path: Path):
    writer = AssetExtractionWriter()
    empty_image = np.array([])
    file_path = tmp_path / "invalid.png"

    with pytest.raises(AssetWriteError):
        writer.write_image(empty_image, file_path)


def test_write_json_sidecar(tmp_path: Path):
    writer = AssetExtractionWriter()
    data = {"key": "value", "numbers": [1, 2, 3]}
    json_path = tmp_path / "sidecar" / "data.json"

    ok = writer.write_json_sidecar(data, json_path)
    assert ok is True
    assert json_path.exists()
    assert json_path.stat().st_size > 0
