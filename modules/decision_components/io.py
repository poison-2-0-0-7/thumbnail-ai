"""
io.py
====

Ingestion, normalization, atomic persistence, and cache access for Module 9.
Reads upstream artifacts (M4, M5, M6, M8) and builds the in-memory DecisionInputBundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from modules.config import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_ASSET_EXTRACTION_DIR,
    DEFAULT_DECISION_DIR,
    DEFAULT_PROMPT_PACKAGE_DIR,
    DEFAULT_REDESIGN_SPEC_DIR,
    DECISION_MANIFEST_FILENAME,
)
from modules.decision_components.interfaces import IDecisionCache
from modules.decision_exceptions import (
    ArtifactValidationError,
    AssetExtractionManifestError,
    DecisionCacheError,
    ManifestPersistError,
    MissingArtifactError,
)
from modules.models import (
    AssetExtractionManifest,
    DecisionManifest,
    PromptPackage,
    RedesignSpecification,
    ThumbnailIntelligence,
)


@dataclass(frozen=True)
class DecisionInputBundle:
    """Immutable in-memory value object holding all loaded upstream artifacts."""

    video_id: str
    intelligence: ThumbnailIntelligence
    redesign_spec: RedesignSpecification
    prompt_package: PromptPackage
    asset_extraction: Optional[AssetExtractionManifest] = None
    cross_reference_index: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_input_bundle(
    video_id: str,
    *,
    analysis_dir: Optional[Path] = None,
    redesign_spec_dir: Optional[Path] = None,
    prompt_package_dir: Optional[Path] = None,
    asset_extraction_dir: Optional[Path] = None,
) -> DecisionInputBundle:
    """Load and cross-reference all 4 upstream artifacts for a video_id."""
    v_id = _validate_video_id(video_id)

    a_dir = Path(analysis_dir) if analysis_dir is not None else DEFAULT_ANALYSIS_DIR
    r_spec_dir = Path(redesign_spec_dir) if redesign_spec_dir is not None else DEFAULT_REDESIGN_SPEC_DIR
    p_pkg_dir = Path(prompt_package_dir) if prompt_package_dir is not None else DEFAULT_PROMPT_PACKAGE_DIR
    a_ext_dir = Path(asset_extraction_dir) if asset_extraction_dir is not None else DEFAULT_ASSET_EXTRACTION_DIR

    # 1. Load Module 4 ThumbnailIntelligence
    intel_path = a_dir / f"{v_id}.json"
    if not intel_path.exists():
        raise MissingArtifactError(f"Missing Module 4 intelligence report at {intel_path}")
    try:
        intel_raw = intel_path.read_text(encoding="utf-8")
        intelligence = ThumbnailIntelligence.model_validate_json(intel_raw)
    except Exception as exc:
        raise ArtifactValidationError(f"Invalid Module 4 intelligence report at {intel_path}: {exc}") from exc

    # 2. Load Module 5 RedesignSpecification
    spec_path = r_spec_dir / f"{v_id}.json"
    if not spec_path.exists():
        raise MissingArtifactError(f"Missing Module 5 redesign specification at {spec_path}")
    try:
        spec_raw = spec_path.read_text(encoding="utf-8")
        redesign_spec = RedesignSpecification.model_validate_json(spec_raw)
    except Exception as exc:
        raise ArtifactValidationError(f"Invalid Module 5 redesign specification at {spec_path}: {exc}") from exc

    # 3. Load Module 6 PromptPackage
    prompt_path = p_pkg_dir / f"{v_id}.json"
    if not prompt_path.exists():
        raise MissingArtifactError(f"Missing Module 6 prompt package at {prompt_path}")
    try:
        prompt_raw = prompt_path.read_text(encoding="utf-8")
        prompt_package = PromptPackage.model_validate_json(prompt_raw)
    except Exception as exc:
        raise ArtifactValidationError(f"Invalid Module 6 prompt package at {prompt_path}: {exc}") from exc

    # 4. Load Module 8 AssetExtractionManifest (Degrades gracefully if missing)
    asset_manifest_path = a_ext_dir / v_id / "asset_manifest.json"
    asset_extraction: Optional[AssetExtractionManifest] = None
    if asset_manifest_path.exists():
        try:
            asset_raw = asset_manifest_path.read_text(encoding="utf-8")
            asset_extraction = AssetExtractionManifest.model_validate_json(asset_raw)
        except Exception as exc:
            logger.warning(
                "Module 8 asset manifest for video_id={id} corrupted or invalid: {exc}. Degrading to partial bundle.",
                id=v_id,
                exc=str(exc),
            )
    else:
        logger.warning(
            "Module 8 asset manifest missing for video_id={id} at {path}. Degrading to partial bundle.",
            id=v_id,
            path=asset_manifest_path,
        )

    # 5. Build cross-reference index
    cross_index = _build_cross_reference_index(intelligence, redesign_spec, prompt_package, asset_extraction)

    return DecisionInputBundle(
        video_id=v_id,
        intelligence=intelligence,
        redesign_spec=redesign_spec,
        prompt_package=prompt_package,
        asset_extraction=asset_extraction,
        cross_reference_index=cross_index,
    )


def _build_cross_reference_index(
    intelligence: ThumbnailIntelligence,
    redesign_spec: RedesignSpecification,
    prompt_package: PromptPackage,
    asset_extraction: Optional[AssetExtractionManifest],
) -> dict[str, dict[str, Any]]:
    """Build unified element_id -> metadata cross-reference map."""
    index: dict[str, dict[str, Any]] = {}

    # Index Module 4 Detected Objects
    if intelligence.objects:
        for idx, obj in enumerate(intelligence.objects):
            elem_id = f"m4_obj_{idx}"
            index[elem_id] = {
                "element_id": elem_id,
                "element_type": "object",
                "label": obj.label,
                "bbox": obj.bbox,
                "source": "module4_detection",
                "confidence": obj.confidence,
            }

    # Index Module 4 OCR Text Regions
    if intelligence.ocr and intelligence.ocr.text_regions:
        for idx, text_reg in enumerate(intelligence.ocr.text_regions):
            elem_id = f"m4_text_{idx}"
            index[elem_id] = {
                "element_id": elem_id,
                "element_type": "text",
                "label": text_reg.text,
                "bbox": text_reg.bbox,
                "source": "module4_ocr",
                "confidence": text_reg.confidence,
            }

    # Index Module 4 Faces
    if intelligence.faces and intelligence.faces.faces:
        for idx, face in enumerate(intelligence.faces.faces):
            elem_id = f"m4_face_{idx}"
            index[elem_id] = {
                "element_id": elem_id,
                "element_type": "face",
                "label": "creator face" if face.is_largest else f"face {idx+1}",
                "bbox": face.bbox,
                "source": "module4_face",
                "confidence": face.detection_confidence,
            }

    # Index Module 8 Extracted Assets if present
    if asset_extraction:
        for obj in asset_extraction.objects:
            elem_id = f"m8_obj_{obj.object_index}"
            index[elem_id] = {
                "element_id": elem_id,
                "element_type": "object",
                "label": obj.label,
                "bbox": obj.bbox,
                "source": "module8_object",
                "confidence": obj.confidence,
            }
        for p in asset_extraction.people:
            elem_id = f"m8_person_{p.person_index}"
            index[elem_id] = {
                "element_id": elem_id,
                "element_type": "face",
                "label": f"person {p.person_index+1}",
                "bbox": None,
                "source": "module8_person",
                "confidence": 0.9,
            }
        for typo in asset_extraction.typography:
            elem_id = f"m8_typo_{typo.text_region_index}"
            index[elem_id] = {
                "element_id": elem_id,
                "element_type": "text",
                "label": typo.text,
                "bbox": typo.bbox,
                "source": "module8_typography",
                "confidence": 0.9,
            }

    return index


def save_decision_manifest(
    manifest: DecisionManifest,
    target_dir: Path = DEFAULT_DECISION_DIR,
) -> Path:
    """Atomically write decision manifest and per-action JSON projections."""
    try:
        video_dir = Path(target_dir) / manifest.video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = video_dir / DECISION_MANIFEST_FILENAME
        _atomic_write_json(manifest.model_dump(mode="json"), manifest_path)

        # Write five per-action JSON projections
        for action in ["keep", "remove", "replace", "enhance", "add"]:
            filtered = [
                d.model_dump(mode="json") for d in manifest.decisions if d.action == action
            ]
            _atomic_write_json(filtered, video_dir / f"{action}.json")

        return manifest_path
    except Exception as exc:
        raise ManifestPersistError(f"Failed to persist decision manifest for {manifest.video_id}: {exc}") from exc


def load_cached_decision_manifest(
    video_id: str,
    decision_dir: Path = DEFAULT_DECISION_DIR,
) -> Optional[DecisionManifest]:
    """Load cached decision manifest for video_id if present and valid."""
    v_id = _validate_video_id(video_id)
    manifest_path = Path(decision_dir) / v_id / DECISION_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None

    try:
        raw_text = manifest_path.read_text(encoding="utf-8")
        manifest = DecisionManifest.model_validate_json(raw_text)
        return manifest
    except Exception as exc:
        logger.warning("Failed to load cached decision manifest at {path}: {exc}", path=manifest_path, exc=str(exc))
        return None


class DecisionCache(IDecisionCache):
    """File-based cache implementation for Module 9."""

    def __init__(self, decision_dir: Path = DEFAULT_DECISION_DIR) -> None:
        self.decision_dir = Path(decision_dir)

    def load(self, video_id: str) -> Optional[DecisionManifest]:
        return load_cached_decision_manifest(video_id, decision_dir=self.decision_dir)

    def save(self, manifest: DecisionManifest) -> None:
        save_decision_manifest(manifest, target_dir=self.decision_dir)


def _atomic_write_json(data: Any, path: Path) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def _validate_video_id(video_id: str) -> str:
    if not video_id or not video_id.strip():
        raise ValueError("video_id must not be empty")
    return video_id.strip()
