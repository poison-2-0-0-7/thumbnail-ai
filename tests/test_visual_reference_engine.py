"""Comprehensive tests for Module 6.5 Visual Reference Engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Optional
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import (  # noqa: E402
    VRE_CACHE_ENABLED,
    VRE_CANNY_HIGH_THRESHOLD,
    VRE_CANNY_LOW_THRESHOLD,
    VRE_FACE_DETECTION_CONFIDENCE,
    VRE_STORAGE_ROOT,
)
from models import AssetMetadata, VisualBoundingBox, VisualReferenceManifest  # noqa: E402
from visual_reference_engine import VisualReferenceEngine  # noqa: E402
from vre_components.asset_writer import AssetWriter  # noqa: E402
from vre_components.face_processor import FaceProcessor  # noqa: E402
from vre_components.interfaces import (  # noqa: E402
    IAssetWriter,
    IFaceProcessor,
    IManifestBuilder,
    ISegmentationProcessor,
    ITopologyProcessor,
)
from vre_components.manifest_builder import ManifestBuilder  # noqa: E402
from vre_components.segmentation_processor import SegmentationProcessor  # noqa: E402
from vre_components.topology_processor import TopologyProcessor  # noqa: E402
from vre_exceptions import (  # noqa: E402
    AssetWriteError,
    ManifestValidationError,
    SegmentationInferenceError,
    SourceImageNotFoundError,
    TopologyExtractionError,
    VREBaseError,
)


VIDEO_ID = "abcdEFGH123"


def _image(size: tuple[int, int] = (320, 280), color: tuple[int, int, int] = (120, 80, 40)) -> np.ndarray:
    width, height = size
    return np.full((height, width, 3), color, dtype=np.uint8)


def _write_image(path: Path, image: np.ndarray | None = None) -> Path:
    array = image if image is not None else _image()
    assert cv2.imwrite(str(path), array)
    return path


class FakeFaceProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def process(
        self, image: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict[str, Any]]:
        self.calls += 1
        return (
            image[0:64, 0:64].copy(),
            np.full(image.shape[:2], 255, dtype=np.uint8),
            {"confidence": 0.98, "face_detected": True},
        )


class FakeSegmentationProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.calls += 1
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[20:120, 30:130] = 255
        return image.copy(), np.zeros_like(image), image[20:120, 30:130].copy(), mask


class FakeTopologyProcessor:
    def __init__(self) -> None:
        self.depth_calls = 0
        self.canny_calls = 0

    def generate_depth_map(self, image: np.ndarray) -> np.ndarray:
        self.depth_calls += 1
        return np.full(image.shape[:2], 127, dtype=np.uint8)

    def generate_canny_map(self, image: np.ndarray) -> np.ndarray:
        self.canny_calls += 1
        return np.zeros(image.shape[:2], dtype=np.uint8)


def _engine(tmp_path: Path) -> tuple[VisualReferenceEngine, FakeFaceProcessor, FakeSegmentationProcessor, FakeTopologyProcessor]:
    face = FakeFaceProcessor()
    segmentation = FakeSegmentationProcessor()
    topology = FakeTopologyProcessor()
    return (
        VisualReferenceEngine(
            storage_root=tmp_path / "visual_references",
            face_processor=face,
            segmentation_processor=segmentation,
            topology_processor=topology,
        ),
        face,
        segmentation,
        topology,
    )


def test_config_constants_are_present_and_typed() -> None:
    assert isinstance(VRE_STORAGE_ROOT, Path)
    assert isinstance(VRE_CANNY_LOW_THRESHOLD, int)
    assert isinstance(VRE_CANNY_HIGH_THRESHOLD, int)
    assert isinstance(VRE_FACE_DETECTION_CONFIDENCE, float)
    assert isinstance(VRE_CACHE_ENABLED, bool)


def test_vre_models_validate_pixel_bbox_and_manifest_metadata() -> None:
    bbox = VisualBoundingBox(x=1, y=2, width=3, height=4)
    assert bbox.width == 3
    with pytest.raises(ValueError):
        VisualBoundingBox(x=0, y=0, width=0, height=1)
    checksum = "a" * 64
    metadata = AssetMetadata(
        asset_type="depth_map",
        file_path="/tmp/depth.png",
        checksum=checksum,
        resolution=(320, 240),
    )
    manifest = VisualReferenceManifest(
        video_id=VIDEO_ID,
        source_image_path="/tmp/source.jpg",
        source_hash="b" * 64,
        created_at="2026-01-01T00:00:00+00:00",
        assets={"depth_map": metadata},
    )
    assert manifest.assets["depth_map"] == metadata


def test_processor_interfaces_are_abstract_contracts() -> None:
    for interface in (IFaceProcessor, ISegmentationProcessor, ITopologyProcessor, IAssetWriter, IManifestBuilder):
        with pytest.raises(TypeError):
            interface()  # type: ignore[abstract]


def test_source_hash_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "source.jpg"
    path.write_bytes(b"stable bytes")
    engine = VisualReferenceEngine(storage_root=tmp_path / "out")
    expected = hashlib.sha256(b"stable bytes").hexdigest()
    assert engine._compute_asset_hash(str(path)) == expected
    assert engine._compute_asset_hash(str(path)) == expected


def test_prepare_assets_writes_complete_manifest_and_assets(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "source.jpg")
    engine, face, segmentation, topology = _engine(tmp_path)

    manifest = engine.prepare_assets(VIDEO_ID, str(source))

    assert manifest.video_id == VIDEO_ID
    assert manifest.source_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest.processing_metadata["cached_hit"] is False
    assert face.calls == 1
    assert segmentation.calls == 1
    assert topology.depth_calls == 1
    assert topology.canny_calls == 1
    expected_keys = set(ManifestBuilder.EXPECTED_ASSETS)
    assert set(manifest.assets) == expected_keys
    for asset in manifest.assets.values():
        assert asset is not None
        assert Path(asset.file_path).is_file()
        assert len(asset.checksum) == 64
    manifest_path = tmp_path / "visual_references" / VIDEO_ID / "reference_manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["video_id"] == VIDEO_ID


def test_cache_hit_bypasses_processors(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "source.jpg")
    engine, face, segmentation, topology = _engine(tmp_path)
    first = engine.prepare_assets(VIDEO_ID, str(source))
    second = engine.prepare_assets(VIDEO_ID, str(source))

    assert second.source_hash == first.source_hash
    assert second.processing_metadata["cached_hit"] is True
    assert face.calls == 1
    assert segmentation.calls == 1
    assert topology.depth_calls == 1


def test_cache_miss_when_asset_is_missing(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "source.jpg")
    engine, face, _, _ = _engine(tmp_path)
    first = engine.prepare_assets(VIDEO_ID, str(source))
    creator_face = first.assets["creator_face"]
    assert creator_face is not None
    Path(creator_face.file_path).unlink()

    second = engine.prepare_assets(VIDEO_ID, str(source))

    assert second.processing_metadata["cached_hit"] is False
    assert face.calls == 2


def test_prepare_assets_rejects_missing_or_too_small_sources(tmp_path: Path) -> None:
    engine, _, _, _ = _engine(tmp_path)
    with pytest.raises(SourceImageNotFoundError):
        engine.prepare_assets(VIDEO_ID, str(tmp_path / "missing.jpg"))
    small = _write_image(tmp_path / "small.jpg", _image(size=(64, 64)))
    with pytest.raises(SourceImageNotFoundError):
        engine.prepare_assets(VIDEO_ID, str(small))


def test_clean_assets_removes_video_shard(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "source.jpg")
    engine, _, _, _ = _engine(tmp_path)
    engine.prepare_assets(VIDEO_ID, str(source))
    shard = tmp_path / "visual_references" / VIDEO_ID
    assert shard.exists()
    assert engine.clean_assets(VIDEO_ID) is True
    assert not shard.exists()


def test_manifest_builder_rejects_missing_asset(tmp_path: Path) -> None:
    builder = ManifestBuilder()
    with pytest.raises(ManifestValidationError):
        builder.build(
            VIDEO_ID,
            str(tmp_path / "source.jpg"),
            {"depth_map": str(tmp_path / "missing.png")},
            {"source_hash": "a" * 64, "created_at": "2026-01-01T00:00:00+00:00"},
        )


def test_manifest_builder_serializes_atomically(tmp_path: Path) -> None:
    asset = _write_image(tmp_path / "depth.png")
    source = _write_image(tmp_path / "source.jpg")
    builder = ManifestBuilder()
    manifest = builder.build(
        VIDEO_ID,
        str(source),
        {"depth_map": str(asset)},
        {"source_hash": hashlib.sha256(source.read_bytes()).hexdigest(), "created_at": "2026-01-01T00:00:00+00:00"},
    )
    destination = tmp_path / "manifest.json"
    builder.serialize_to_disk(manifest, destination)
    assert destination.exists()
    assert not destination.with_suffix(".tmp").exists()


def test_asset_writer_wraps_encode_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)
    with pytest.raises(AssetWriteError):
        AssetWriter().write_image(_image(), tmp_path / "bad.png")
    assert not list(tmp_path.glob("*.tmp"))


def test_topology_processor_validates_shape_and_preserves_resolution() -> None:
    processor = TopologyProcessor()
    image = _image(size=(300, 260))
    assert processor.generate_depth_map(image).shape == image.shape[:2]
    assert processor.generate_canny_map(image).shape == image.shape[:2]
    with pytest.raises(TopologyExtractionError):
        processor.generate_depth_map(np.zeros((10, 10), dtype=np.uint8))


def test_segmentation_processor_returns_required_assets() -> None:
    image = _image(size=(300, 260))
    cv2.rectangle(image, (90, 80), (180, 170), (255, 255, 255), thickness=-1)
    foreground, background, object_crop, object_mask = SegmentationProcessor().process(image)
    assert foreground.shape == image.shape
    assert background.shape == image.shape
    assert object_crop.ndim == 3
    assert object_mask.shape == image.shape[:2]
    with pytest.raises(SegmentationInferenceError):
        SegmentationProcessor().process(np.zeros((10, 10), dtype=np.uint8))


def test_face_processor_degrades_without_detector_result() -> None:
    face, mask, metadata = FaceProcessor(cascade_path=Path("missing.xml")).process(_image())
    assert face is None
    assert mask is None
    assert metadata["face_detected"] is False


def test_exception_hierarchy() -> None:
    assert issubclass(SourceImageNotFoundError, VREBaseError)
    assert issubclass(SegmentationInferenceError, VREBaseError)
    assert issubclass(TopologyExtractionError, VREBaseError)
    assert issubclass(AssetWriteError, VREBaseError)
    assert issubclass(ManifestValidationError, VREBaseError)


def test_processor_failure_propagates_explicitly(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "source.jpg")
    engine, _, _, _ = _engine(tmp_path)
    failing = MagicMock()
    failing.process.side_effect = SegmentationInferenceError("segmentation failed")
    engine.segmentation_processor = failing
    with pytest.raises(SegmentationInferenceError):
        engine.prepare_assets(VIDEO_ID, str(source), options={"cache_enabled": False})
