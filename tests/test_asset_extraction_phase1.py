"""
test_asset_extraction_phase1.py
================================

Unit tests for Module 8 Phase 1:
- Exception hierarchy
- Configuration constants
- Pydantic data models & validation
- Abstract interface contracts
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from modules.asset_extraction_exceptions import (
    AssetExtractionError,
    AssetFamilyDegradedWarning,
    AssetFamilyModelError,
    AssetWriteError,
    CacheCorruptError,
    IntelligenceReportInvalidError,
    ManifestNotFoundError,
    ManifestValidationError,
    SourceImageNotFoundError,
)
from modules.config import (
    ASSET_EXTRACTION_CACHE_ENABLED,
    ASSET_EXTRACTION_ENGINE_VERSION,
    ASSET_EXTRACTION_FAMILY_ORDER,
    ASSET_MANIFEST_FILENAME,
    DEFAULT_ASSET_EXTRACTION_DIR,
    MODULE8_LOG_PATH,
)
from modules.models import (
    AssetExtractionManifest,
    AssetExtractionStatus,
    AssetFileRef,
    BoundingBox,
    CompositionAnalysis,
    CompositionAsset,
    EffectsAsset,
    ObjectAsset,
    PersonAsset,
    SceneAsset,
    TypographyAsset,
    VisualPropertiesAsset,
)
from modules.asset_extraction_components.interfaces import (
    IAssetExtractionWriter,
    IAssetManifestBuilder,
    ICompositionAssetProcessor,
    IEffectsProcessor,
    IModelBridge,
    IObjectProcessor,
    IPersonProcessor,
    ISceneProcessor,
    ITypographyProcessor,
    IVisualPropertiesProcessor,
)


def test_exception_hierarchy():
    """Verify all exceptions inherit from AssetExtractionError."""
    assert issubclass(SourceImageNotFoundError, AssetExtractionError)
    assert issubclass(IntelligenceReportInvalidError, AssetExtractionError)
    assert issubclass(AssetFamilyModelError, AssetExtractionError)
    assert issubclass(AssetWriteError, AssetExtractionError)
    assert issubclass(ManifestValidationError, AssetExtractionError)
    assert issubclass(ManifestNotFoundError, AssetExtractionError)
    assert issubclass(CacheCorruptError, AssetExtractionError)

    # Degraded warning should inherit from Warning
    assert issubclass(AssetFamilyDegradedWarning, Warning)

    err = AssetFamilyModelError("Model failed", family_name="objects", model_name="sam2")
    assert err.family_name == "objects"
    assert err.model_name == "sam2"
    assert str(err) == "Model failed"


def test_config_constants():
    """Verify configuration constants are properly set."""
    assert isinstance(MODULE8_LOG_PATH, Path)
    assert isinstance(DEFAULT_ASSET_EXTRACTION_DIR, Path)
    assert ASSET_MANIFEST_FILENAME == "asset_manifest.json"
    assert ASSET_EXTRACTION_ENGINE_VERSION == "1.0.0"
    assert ASSET_EXTRACTION_CACHE_ENABLED is True
    assert isinstance(ASSET_EXTRACTION_FAMILY_ORDER, tuple)
    assert len(ASSET_EXTRACTION_FAMILY_ORDER) == 7
    assert "typography" in ASSET_EXTRACTION_FAMILY_ORDER
    assert "scene" in ASSET_EXTRACTION_FAMILY_ORDER


def test_asset_file_ref_validation():
    """Verify AssetFileRef validations."""
    ref = AssetFileRef(
        asset_type="face_crop",
        file_path="data/asset_extraction/v123/people/face_01.png",
        checksum="a" * 64,
        resolution=(256, 256),
        confidence_score=0.95,
        source="extracted",
    )
    assert ref.checksum == "a" * 64
    assert ref.resolution == (256, 256)

    # Test blank field rejection
    with pytest.raises(ValidationError):
        AssetFileRef(
            asset_type="  ",
            file_path="path",
            checksum="a" * 64,
            resolution=(10, 10),
            source="extracted",
        )

    # Test bad checksum
    with pytest.raises(ValidationError):
        AssetFileRef(
            asset_type="crop",
            file_path="path",
            checksum="invalid_hash",
            resolution=(10, 10),
            source="extracted",
        )

    # Test bad confidence score
    with pytest.raises(ValidationError):
        AssetFileRef(
            asset_type="crop",
            file_path="path",
            checksum="a" * 64,
            resolution=(10, 10),
            confidence_score=1.5,
            source="extracted",
        )


def test_manifest_validation():
    """Verify AssetExtractionManifest validation."""
    valid_hash = "f" * 64
    manifest = AssetExtractionManifest(
        video_id="v123",
        source_thumbnail_path="data/thumb.jpg",
        source_hash=valid_hash,
        intelligence_hash=valid_hash,
        engine_version="1.0.0",
        extracted_at="2026-07-30T00:00:00Z",
    )
    assert manifest.status == AssetExtractionStatus.SUCCESS
    assert manifest.video_id == "v123"

    with pytest.raises(ValidationError):
        AssetExtractionManifest(
            video_id="",
            source_thumbnail_path="data/thumb.jpg",
            source_hash=valid_hash,
            intelligence_hash=valid_hash,
            engine_version="1.0.0",
            extracted_at="2026-07-30T00:00:00Z",
        )


def test_interfaces_cannot_be_instantiated():
    """Verify ABCs raise TypeError if instantiated directly."""
    for abc_cls in (
        IPersonProcessor,
        ISceneProcessor,
        IObjectProcessor,
        ITypographyProcessor,
        IVisualPropertiesProcessor,
        ICompositionAssetProcessor,
        IEffectsProcessor,
        IModelBridge,
        IAssetExtractionWriter,
        IAssetManifestBuilder,
    ):
        with pytest.raises(TypeError):
            abc_cls()
