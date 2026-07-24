"""Reference manifest construction and validation for Module 6.5 VRE."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import cv2
from pydantic import ValidationError

from models import AssetMetadata, VisualReferenceManifest
from vre_components.interfaces import IManifestBuilder
from vre_exceptions import ManifestValidationError


class ManifestBuilder(IManifestBuilder):
    """Build and atomically persist validated VRE manifests."""

    EXPECTED_ASSETS: tuple[str, ...] = (
        "creator_face",
        "face_mask",
        "object_crop",
        "object_mask",
        "foreground",
        "background",
        "depth_map",
        "canny_map",
    )

    def build(
        self,
        video_id: str,
        source_path: str,
        asset_paths: dict[str, str],
        metadata: dict[str, Any],
    ) -> VisualReferenceManifest:
        source_hash = str(metadata.get("source_hash", ""))
        created_at = str(metadata.get("created_at", ""))
        confidence = dict(metadata.get("confidence_scores", {}))
        assets: dict[str, Optional[AssetMetadata]] = {}

        for asset_type in self.EXPECTED_ASSETS:
            raw_path = asset_paths.get(asset_type)
            if raw_path is None:
                assets[asset_type] = None
                continue
            path = Path(raw_path)
            assets[asset_type] = self._asset_metadata(
                asset_type,
                path,
                confidence.get(asset_type),
            )

        manifest_dict = {
            "video_id": video_id,
            "source_image_path": str(Path(source_path).resolve()),
            "source_hash": source_hash,
            "created_at": created_at,
            "assets": assets,
            "processing_metadata": dict(metadata.get("processing_metadata", {})),
        }
        return self._validate_schema(manifest_dict)

    def serialize_to_disk(
        self, manifest: VisualReferenceManifest, destination_path: Path
    ) -> None:
        target = Path(destination_path)
        tmp = target.with_suffix(".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(target)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise ManifestValidationError(f"Could not write VRE manifest to {target}: {exc}") from exc

    def _asset_metadata(
        self,
        asset_type: str,
        path: Path,
        confidence_score: object,
    ) -> AssetMetadata:
        if not path.is_file():
            raise ManifestValidationError(f"VRE asset does not exist: {path}")
        if path.stat().st_size <= 0:
            raise ManifestValidationError(f"VRE asset is empty: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise ManifestValidationError(f"VRE asset is not decodable: {path}")
        height, width = image.shape[:2]
        score = None if confidence_score is None else float(confidence_score)
        return AssetMetadata(
            asset_type=asset_type,
            file_path=str(path.resolve()),
            checksum=self._file_sha256(path),
            resolution=(width, height),
            confidence_score=score,
        )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _validate_schema(manifest_dict: dict[str, Any]) -> VisualReferenceManifest:
        try:
            return VisualReferenceManifest.model_validate(manifest_dict)
        except ValidationError as exc:
            raise ManifestValidationError(f"Invalid VRE manifest schema: {exc}") from exc
