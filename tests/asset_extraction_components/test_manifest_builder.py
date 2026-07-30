"""
test_manifest_builder.py
========================

Unit tests for ManifestBuilder (Phase 2 manifest construction & atomic serialization).
"""

from pathlib import Path
import pytest

from modules.asset_extraction_components.manifest_builder import ManifestBuilder
from modules.asset_extraction_exceptions import ManifestValidationError
from modules.models import AssetExtractionStatus, BoundingBox, TypographyAsset


def test_manifest_builder_build_and_serialize(tmp_path: Path):
    builder = ManifestBuilder()
    valid_hash = "a" * 64

    typo_asset = TypographyAsset(
        text_region_index=0,
        text="Sample",
        bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.6, y_max=0.3),
        source_text_region_index=0,
    )

    manifest = builder.build(
        video_id="v999",
        source_thumbnail_path="data/test.jpg",
        source_hash=valid_hash,
        intelligence_hash=valid_hash,
        typography=[typo_asset],
        status=AssetExtractionStatus.SUCCESS,
    )

    assert manifest.video_id == "v999"
    assert len(manifest.typography) == 1
    assert manifest.typography[0].text == "Sample"

    manifest_path = tmp_path / "asset_manifest.json"
    builder.serialize_to_disk(manifest, manifest_path)

    assert manifest_path.exists()
    assert manifest_path.stat().st_size > 0


def test_manifest_builder_validation_failure():
    builder = ManifestBuilder()

    with pytest.raises(ManifestValidationError):
        builder.build(
            video_id="",  # Blank video ID causes validation error
            source_thumbnail_path="path",
            source_hash="a" * 64,
            intelligence_hash="a" * 64,
        )
