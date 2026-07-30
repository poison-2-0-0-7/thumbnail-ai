"""
manifest_builder.py
===================

Assembles and validates AssetExtractionManifest objects and persists them atomically.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import ValidationError

from modules.config import ASSET_EXTRACTION_ENGINE_VERSION
from modules.asset_extraction_components.interfaces import IAssetManifestBuilder
from modules.asset_extraction_exceptions import AssetWriteError, ManifestValidationError
from modules.models import (
    AssetExtractionManifest,
    AssetExtractionStatus,
    CompositionAsset,
    EffectsAsset,
    ObjectAsset,
    PersonAsset,
    SceneAsset,
    TypographyAsset,
    VisualPropertiesAsset,
)


class ManifestBuilder(IAssetManifestBuilder):
    """Builds and serializes AssetExtractionManifest objects."""

    def build(
        self,
        video_id: str,
        source_thumbnail_path: str,
        source_hash: str,
        intelligence_hash: str,
        *,
        engine_version: str = ASSET_EXTRACTION_ENGINE_VERSION,
        people: Optional[list[PersonAsset]] = None,
        scene: Optional[SceneAsset] = None,
        objects: Optional[list[ObjectAsset]] = None,
        typography: Optional[list[TypographyAsset]] = None,
        visual_properties: Optional[VisualPropertiesAsset] = None,
        composition: Optional[CompositionAsset] = None,
        effects: Optional[EffectsAsset] = None,
        status: AssetExtractionStatus = AssetExtractionStatus.SUCCESS,
        partial_failure_reasons: Optional[list[str]] = None,
        completed_families: Optional[list[str]] = None,
        total_duration_seconds: float = 0.0,
        extracted_at: Optional[str] = None,
    ) -> AssetExtractionManifest:
        """Assemble an AssetExtractionManifest from family assets."""
        if extracted_at is None:
            extracted_at = datetime.now(timezone.utc).isoformat()

        try:
            manifest = AssetExtractionManifest(
                video_id=video_id,
                source_thumbnail_path=source_thumbnail_path,
                source_hash=source_hash,
                intelligence_hash=intelligence_hash,
                engine_version=engine_version,
                people=people or [],
                scene=scene,
                objects=objects or [],
                typography=typography or [],
                visual_properties=visual_properties,
                composition=composition,
                effects=effects,
                status=status,
                partial_failure_reasons=partial_failure_reasons or [],
                completed_families=completed_families or [],
                total_duration_seconds=total_duration_seconds,
                extracted_at=extracted_at,
            )
            return manifest
        except ValidationError as exc:
            raise ManifestValidationError(
                f"Validation failed while building AssetExtractionManifest for video {video_id}: {exc}"
            ) from exc

    def serialize_to_disk(self, manifest: AssetExtractionManifest, path: Path) -> None:
        """Atomically persist manifest as JSON to disk."""
        target_path = Path(path)
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AssetWriteError(f"Could not create directory {target_path.parent}: {exc}") from exc

        json_bytes = manifest.model_dump_json(indent=2).encode("utf-8")
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target_path.parent, suffix=".tmp", mode="wb", delete=False
            ) as handle:
                tmp_name = handle.name
                handle.write(json_bytes)

            Path(tmp_name).replace(target_path)
            logger.debug("Serialized asset manifest -> {path}", path=target_path)
        except OSError as exc:
            raise AssetWriteError(f"Could not write asset manifest to {target_path}: {exc}") from exc
        finally:
            if tmp_name is not None:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except OSError:
                    pass
