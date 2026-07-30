"""
test_asset_registry.py
======================

Unit tests for AssetRegistry component in Module 10 Asset Composer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

from composition_components.asset_registry import AssetRegistry
from models import AssetMetadata, VisualReferenceManifest


@pytest.fixture
def temp_vre_files(tmp_path: Path):
    face_file = tmp_path / "creator_face.png"
    face_bytes = b"fake_face_image_bytes"
    face_file.write_bytes(face_bytes)
    face_checksum = hashlib.sha256(face_bytes).hexdigest()

    bg_file = tmp_path / "background.png"
    bg_bytes = b"fake_bg_image_bytes"
    bg_file.write_bytes(bg_bytes)
    bg_checksum = hashlib.sha256(bg_bytes).hexdigest()

    return {
        "face_path": str(face_file),
        "face_checksum": face_checksum,
        "bg_path": str(bg_file),
        "bg_checksum": bg_checksum,
    }


def test_asset_registry_index_and_resolve(temp_vre_files):
    manifest = VisualReferenceManifest(
        video_id="test_vid",
        source_image_path="source.jpg",
        source_hash="a" * 64,
        created_at="2026-07-30T00:00:00Z",
        assets={
            "creator_face": AssetMetadata(
                asset_type="creator_face",
                file_path=temp_vre_files["face_path"],
                checksum=temp_vre_files["face_checksum"],
                resolution=(200, 200),
            ),
            "background": AssetMetadata(
                asset_type="background",
                file_path=temp_vre_files["bg_path"],
                checksum=temp_vre_files["bg_checksum"],
                resolution=(1280, 720),
            ),
            "missing_asset": None,
        },
    )

    registry = AssetRegistry(manifest)
    assert registry.resolve("creator_face") is not None
    assert registry.resolve("background") is not None
    assert registry.resolve("missing_asset") is None
    assert registry.resolve("non_existent") is None

    # Verify integrity passes
    invalid = registry.verify_integrity()
    assert invalid == []


def test_asset_registry_integrity_failure(temp_vre_files, tmp_path: Path):
    # Corrupt checksum
    corrupt_meta = AssetMetadata(
        asset_type="creator_face",
        file_path=temp_vre_files["face_path"],
        checksum="f" * 64,  # wrong checksum
        resolution=(200, 200),
    )
    # Missing file
    missing_meta = AssetMetadata(
        asset_type="missing",
        file_path=str(tmp_path / "non_existent.png"),
        checksum="a" * 64,
        resolution=(100, 100),
    )

    manifest = VisualReferenceManifest(
        video_id="test_vid",
        source_image_path="source.jpg",
        source_hash="a" * 64,
        created_at="2026-07-30T00:00:00Z",
        assets={
            "creator_face": corrupt_meta,
            "missing": missing_meta,
        },
    )

    registry = AssetRegistry(manifest)
    invalid = registry.verify_integrity()
    assert "creator_face" in invalid
    assert "missing" in invalid
