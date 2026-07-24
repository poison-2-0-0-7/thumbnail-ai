"""Module 6.5 Visual Reference Engine orchestration.

VRE prepares deterministic visual conditioning assets from a source thumbnail.
It performs no image generation and writes all artifacts into a video-id shard.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger
from pydantic import ValidationError

_MODULES_DIR = Path(__file__).resolve().parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import (  # noqa: E402
    LOG_DIR,
    MODULE65_LOG_PATH,
    VRE_CACHE_ENABLED,
    VRE_ENGINE_VERSION,
    VRE_MANIFEST_FILENAME,
    VRE_MIN_IMAGE_DIMENSION_PX,
    VRE_STORAGE_ROOT,
)
from models import VisualReferenceManifest  # noqa: E402
from vre_components import (  # noqa: E402
    AssetWriter,
    FaceProcessor,
    ManifestBuilder,
    SegmentationProcessor,
    TopologyProcessor,
)
from vre_components.interfaces import (  # noqa: E402
    IAssetWriter,
    IFaceProcessor,
    IManifestBuilder,
    ISegmentationProcessor,
    ITopologyProcessor,
)
from vre_exceptions import (  # noqa: E402
    AssetWriteError,
    ManifestValidationError,
    SourceImageNotFoundError,
    VREBaseError,
)


_LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(MODULE65_LOG_PATH),
        rotation="10 MB",
        retention="30 days",
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,
    )


_configure_logger()


class VisualReferenceEngine:
    """Core VRE state machine coordinating cache, processors, and writes."""

    def __init__(
        self,
        storage_root: Path = VRE_STORAGE_ROOT,
        face_processor: IFaceProcessor | None = None,
        segmentation_processor: ISegmentationProcessor | None = None,
        topology_processor: ITopologyProcessor | None = None,
        asset_writer: IAssetWriter | None = None,
        manifest_builder: IManifestBuilder | None = None,
        cache_enabled: bool = VRE_CACHE_ENABLED,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.face_processor = face_processor or FaceProcessor()
        self.segmentation_processor = segmentation_processor or SegmentationProcessor()
        self.topology_processor = topology_processor or TopologyProcessor()
        self.asset_writer = asset_writer or AssetWriter()
        self.manifest_builder = manifest_builder or ManifestBuilder()
        self.cache_enabled = cache_enabled
        self._lock = threading.RLock()
        logger.info("Initialized VisualReferenceEngine storage_root={root}", root=self.storage_root)

    def prepare_assets(
        self,
        video_id: str,
        source_image_path: str,
        options: Optional[dict] = None,
    ) -> VisualReferenceManifest:
        """Prepare all VRE assets for one source image and return a manifest."""
        options = dict(options or {})
        source_path = Path(source_image_path)
        self._validate_video_id(video_id)
        self._validate_source_path(source_path)
        source_hash = self._compute_asset_hash(str(source_path))
        cache_enabled = bool(options.get("cache_enabled", self.cache_enabled))

        if cache_enabled:
            cached = self._verify_cache(video_id, source_hash)
            if cached is not None:
                logger.info("VRE cache hit for video_id={id}", id=video_id)
                return cached

        started_at = time.monotonic()
        target_dir = self._target_dir(video_id)
        image = self._load_source_image(source_path)

        with self._lock:
            asset_paths, processor_metadata = self._dispatch_processors(image, target_dir)
            duration_ms = int((time.monotonic() - started_at) * 1000)
            metadata = {
                "source_hash": source_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "confidence_scores": processor_metadata.get("confidence_scores", {}),
                "processing_metadata": {
                    "engine_version": VRE_ENGINE_VERSION,
                    "total_duration_ms": duration_ms,
                    "processors_executed": [
                        "FaceProcessor",
                        "SegmentationProcessor",
                        "TopologyProcessor",
                    ],
                    "cached_hit": False,
                    **processor_metadata,
                },
            }
            manifest = self.manifest_builder.build(
                video_id=video_id,
                source_path=str(source_path),
                asset_paths=asset_paths,
                metadata=metadata,
            )
            self.manifest_builder.serialize_to_disk(
                manifest,
                target_dir / VRE_MANIFEST_FILENAME,
            )

        logger.info(
            "VRE pipeline complete for video_id={id}: {duration}ms",
            id=video_id,
            duration=duration_ms,
        )
        return manifest

    def clean_assets(self, video_id: str) -> bool:
        """Remove the generated shard for one video ID."""
        self._validate_video_id(video_id)
        return self.asset_writer.purge_directory(self._target_dir(video_id))

    def _compute_asset_hash(self, file_path: str) -> str:
        path = Path(file_path)
        self._validate_source_path(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_cache(
        self, video_id: str, asset_hash: str
    ) -> Optional[VisualReferenceManifest]:
        manifest_path = self._target_dir(video_id) / VRE_MANIFEST_FILENAME
        if not manifest_path.is_file():
            logger.debug("VRE cache miss for video_id={id}", id=video_id)
            return None
        try:
            manifest = VisualReferenceManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            if manifest.source_hash != asset_hash:
                logger.debug("VRE cache hash mismatch for video_id={id}", id=video_id)
                return None
            for asset in manifest.assets.values():
                if asset is None:
                    continue
                path = Path(asset.file_path)
                if not path.is_file() or path.stat().st_size <= 0:
                    logger.warning("VRE cache asset missing or empty: {path}", path=path)
                    return None
            return manifest.model_copy(
                update={
                    "processing_metadata": {
                        **manifest.processing_metadata,
                        "cached_hit": True,
                    }
                }
            )
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.warning(
                "VRE cache for video_id={id} is unreadable ({reason}); recomputing",
                id=video_id,
                reason=exc,
            )
            return None

    def _dispatch_processors(
        self, source_image: np.ndarray, target_dir: Path
    ) -> tuple[dict[str, str], dict[str, Any]]:
        asset_paths: dict[str, str] = {}
        confidence_scores: dict[str, Optional[float]] = {}

        face_crop, face_mask, face_metadata = self.face_processor.process(source_image)
        if face_crop is not None:
            path = target_dir / "creator_face.png"
            self.asset_writer.write_image(face_crop, path)
            asset_paths["creator_face"] = str(path)
            confidence_scores["creator_face"] = face_metadata.get("confidence")
        if face_mask is not None:
            path = target_dir / "face_mask.png"
            self.asset_writer.write_image(face_mask, path)
            asset_paths["face_mask"] = str(path)
            confidence_scores["face_mask"] = face_metadata.get("confidence")

        foreground, background, object_crop, object_mask = self.segmentation_processor.process(source_image)
        segmentation_assets = {
            "foreground": foreground,
            "background": background,
            "object_crop": object_crop,
            "object_mask": object_mask,
        }
        for name, array in segmentation_assets.items():
            path = target_dir / f"{name}.png"
            self.asset_writer.write_image(array, path)
            asset_paths[name] = str(path)
            confidence_scores[name] = 0.95 if name in {"foreground", "background"} else 0.915

        depth_map = self.topology_processor.generate_depth_map(source_image)
        canny_map = self.topology_processor.generate_canny_map(source_image)
        for name, array in {"depth_map": depth_map, "canny_map": canny_map}.items():
            path = target_dir / f"{name}.png"
            self.asset_writer.write_image(array, path)
            asset_paths[name] = str(path)
            confidence_scores[name] = None

        return asset_paths, {
            "confidence_scores": confidence_scores,
            "face_metadata": face_metadata,
        }

    def _load_source_image(self, source_path: Path) -> np.ndarray:
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise SourceImageNotFoundError(f"Source image is unreadable: {source_path}")
        height, width = image.shape[:2]
        if min(width, height) < VRE_MIN_IMAGE_DIMENSION_PX:
            raise SourceImageNotFoundError(
                f"Source image must be at least {VRE_MIN_IMAGE_DIMENSION_PX}px on each side: "
                f"{width}x{height}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise SourceImageNotFoundError("Source image must decode to a 3-channel color image")
        return image

    def _target_dir(self, video_id: str) -> Path:
        return self.storage_root / video_id

    @staticmethod
    def _validate_video_id(video_id: str) -> None:
        if not video_id or not video_id.strip():
            raise VREBaseError("video_id must not be empty")
        if any(character in video_id for character in ("/", "\\", ":", "..")):
            raise VREBaseError(f"video_id contains unsafe path characters: {video_id!r}")

    @staticmethod
    def _validate_source_path(source_path: Path) -> None:
        if not source_path.is_file():
            raise SourceImageNotFoundError(f"Source image does not exist: {source_path}")


__all__ = [
    "VisualReferenceEngine",
    "VREBaseError",
    "SourceImageNotFoundError",
    "AssetWriteError",
    "ManifestValidationError",
]
