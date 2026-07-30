"""
test_asset_extraction_engine.py
================================

Integration and end-to-end unit tests for AssetExtractionEngine (Module 8).
"""

from pathlib import Path
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest

from modules.asset_extraction_engine import (
    AssetExtractionEngine,
    extract_assets,
    load_asset_manifest,
    save_asset_manifest,
)
from modules.asset_extraction_exceptions import (
    IntelligenceReportInvalidError,
    ManifestNotFoundError,
    SourceImageNotFoundError,
)
from modules.models import (
    AssetExtractionStatus,
    BoundingBox,
    ColorProfile,
    CompositionAnalysis,
    DetectedObject,
    FaceAnalysis,
    FaceDetail,
    OCRResult,
    TextRegion,
    ThumbnailIntelligence,
)


@pytest.fixture
def dummy_intelligence() -> ThumbnailIntelligence:
    return ThumbnailIntelligence(
        video_id="v_test_888",
        thumbnail_path="data/test_thumb.jpg",
        ocr=OCRResult(
            text_regions=[
                TextRegion(
                    text="TITLE TEXT",
                    confidence=0.95,
                    bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.3),
                )
            ]
        ),
        faces=FaceAnalysis(
            face_count=1,
            faces=[
                FaceDetail(
                    bbox=BoundingBox(x_min=0.3, y_min=0.2, x_max=0.7, y_max=0.8),
                    detection_confidence=0.9,
                    is_largest=True,
                )
            ],
            has_face=True,
        ),
        objects=[
            DetectedObject(
                label="car",
                confidence=0.88,
                bbox=BoundingBox(x_min=0.4, y_min=0.4, x_max=0.9, y_max=0.9),
            )
        ],
        colors=ColorProfile(
            dominant_colors=["#ff0000", "#00ff00"],
            palette_hex=["#ff0000", "#00ff00"],
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
        ),
        composition=CompositionAnalysis(
            rule_of_thirds_score=0.8,
            negative_space_ratio=0.3,
            visual_clutter_score=0.2,
            symmetry_score=0.5,
            balance_score=0.7,
        ),
        status="success",
        analyzed_at="2026-07-30T00:00:00Z",
    )


@pytest.fixture
def dummy_image_file(tmp_path: Path) -> Path:
    img_path = tmp_path / "test_thumb.jpg"
    img = np.full((300, 400, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


def test_extract_assets_end_to_end(tmp_path: Path, dummy_image_file: Path, dummy_intelligence: ThumbnailIntelligence):
    storage_root = tmp_path / "asset_extraction"

    manifest = extract_assets(
        video_id="v_test_888",
        source_image_path=str(dummy_image_file),
        intelligence=dummy_intelligence,
        storage_root=storage_root,
    )

    assert manifest.video_id == "v_test_888"
    assert manifest.status == AssetExtractionStatus.SUCCESS
    assert len(manifest.completed_families) == 7
    assert len(manifest.typography) == 1
    assert manifest.typography[0].text == "TITLE TEXT"
    assert manifest.visual_properties is not None
    assert manifest.composition is not None
    assert len(manifest.objects) == 1
    assert len(manifest.people) == 1
    assert manifest.scene is not None
    assert manifest.effects is not None

    # Check disk persistence
    manifest_on_disk = storage_root / "v_test_888" / "asset_manifest.json"
    assert manifest_on_disk.exists()
    assert manifest_on_disk.stat().st_size > 0


def test_save_and_load_manifest_roundtrip(tmp_path: Path, dummy_image_file: Path, dummy_intelligence: ThumbnailIntelligence):
    storage_root = tmp_path / "asset_extraction"

    manifest = extract_assets(
        video_id="v_test_888",
        source_image_path=str(dummy_image_file),
        intelligence=dummy_intelligence,
        storage_root=storage_root,
    )

    saved_path = save_asset_manifest(manifest, storage_root=storage_root)
    assert saved_path.exists()

    loaded = load_asset_manifest("v_test_888", storage_root=storage_root)
    assert loaded.video_id == manifest.video_id
    assert loaded.source_hash == manifest.source_hash


def test_caching_hit_and_invalidation(tmp_path: Path, dummy_image_file: Path, dummy_intelligence: ThumbnailIntelligence):
    storage_root = tmp_path / "asset_extraction"
    engine = AssetExtractionEngine(storage_root=storage_root, cache_enabled=True)

    # First extraction
    m1 = engine.extract("v_test_888", str(dummy_image_file), dummy_intelligence)
    assert len(m1.completed_families) == 7

    # Second extraction should be cache hit
    m2 = engine.extract("v_test_888", str(dummy_image_file), dummy_intelligence)
    assert m2.source_hash == m1.source_hash


def test_missing_source_image_raises(tmp_path: Path, dummy_intelligence: ThumbnailIntelligence):
    with pytest.raises(SourceImageNotFoundError):
        extract_assets(
            video_id="v_missing",
            source_image_path=str(tmp_path / "non_existent.jpg"),
            intelligence=dummy_intelligence,
            storage_root=tmp_path,
        )


def test_invalid_intelligence_raises(tmp_path: Path, dummy_image_file: Path, dummy_intelligence: ThumbnailIntelligence):
    bad_intel = dummy_intelligence.model_copy(update={"status": "error"})
    with pytest.raises(IntelligenceReportInvalidError):
        extract_assets(
            video_id="v_bad_intel",
            source_image_path=str(dummy_image_file),
            intelligence=bad_intel,
            storage_root=tmp_path,
        )
