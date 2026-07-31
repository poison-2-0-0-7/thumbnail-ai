"""
io.py
====

Ingestion, atomic persistence, and cache access for Module 10.5 (Thumbnail Planner).
Reads upstream artifacts (Modules 4, 5, 6, 8, 9, 10) and builds PlannerInputBundle.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from config import (
    COMPOSITION_MANIFEST_FILENAME,
    COMPOSITION_WORKSPACE_ROOT,
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_ASSET_EXTRACTION_DIR,
    DEFAULT_DECISION_DIR,
    DEFAULT_GENERATION_PLAN_DIR,
    DEFAULT_PROMPT_PACKAGE_DIR,
    DEFAULT_REDESIGN_SPEC_DIR,
    GENERATION_PLAN_FILENAME,
)
from models import (
    AssetExtractionManifest,
    CompositionWorkspace,
    DecisionManifest,
    GenerationPlan,
    PromptPackage,
    RedesignSpecification,
    ThumbnailIntelligence,
)
from planner_components.interfaces import IPlanCache
from thumbnail_planner_exceptions import (
    PlanCacheError,
    PlanPersistError,
    PlanValidationError,
    UpstreamArtifactMissingError,
)


@dataclass(frozen=True)
class PlannerInputBundle:
    """Immutable in-memory value object holding loaded upstream artifacts for the Planner."""

    video_id: str
    prompt_package: PromptPackage
    workspace: CompositionWorkspace
    intelligence: Optional[ThumbnailIntelligence] = None
    redesign_spec: Optional[RedesignSpecification] = None
    asset_extraction: Optional[AssetExtractionManifest] = None
    decision_manifest: Optional[DecisionManifest] = None


def load_planner_input_bundle(
    video_id: str,
    *,
    analysis_dir: Optional[Path] = None,
    redesign_spec_dir: Optional[Path] = None,
    prompt_package_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    asset_extraction_dir: Optional[Path] = None,
    decision_dir: Optional[Path] = None,
) -> PlannerInputBundle:
    """
    Load all available upstream artifacts for video_id.

    Requires PromptPackage and CompositionWorkspace.
    Degrades gracefully for missing or error-status optional artifacts (M4, M5, M8, M9).
    """
    v_id = _validate_video_id(video_id)

    a_dir = Path(analysis_dir) if analysis_dir is not None else DEFAULT_ANALYSIS_DIR
    r_spec_dir = Path(redesign_spec_dir) if redesign_spec_dir is not None else DEFAULT_REDESIGN_SPEC_DIR
    p_pkg_dir = Path(prompt_package_dir) if prompt_package_dir is not None else DEFAULT_PROMPT_PACKAGE_DIR
    w_dir = Path(workspace_dir) if workspace_dir is not None else COMPOSITION_WORKSPACE_ROOT
    a_ext_dir = Path(asset_extraction_dir) if asset_extraction_dir is not None else DEFAULT_ASSET_EXTRACTION_DIR
    d_dir = Path(decision_dir) if decision_dir is not None else DEFAULT_DECISION_DIR

    # 1. Load Module 6 PromptPackage (required)
    prompt_path = p_pkg_dir / f"{v_id}.json"
    if not prompt_path.exists():
        raise UpstreamArtifactMissingError(f"Missing required Module 6 prompt package at '{prompt_path}'")
    try:
        prompt_package = PromptPackage.model_validate_json(prompt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PlanValidationError(f"Invalid Module 6 prompt package at '{prompt_path}': {exc}") from exc

    # 2. Load Module 10 CompositionWorkspace (required)
    workspace_manifest_path = w_dir / v_id / COMPOSITION_MANIFEST_FILENAME
    if not workspace_manifest_path.exists():
        workspace_manifest_path = w_dir / f"{v_id}.json"

    if not workspace_manifest_path.exists():
        raise UpstreamArtifactMissingError(f"Missing required Module 10 workspace at '{workspace_manifest_path}'")
    try:
        workspace = CompositionWorkspace.model_validate_json(workspace_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PlanValidationError(f"Invalid Module 10 workspace at '{workspace_manifest_path}': {exc}") from exc

    # 3. Load Module 4 ThumbnailIntelligence (optional)
    intel_path = a_dir / f"{v_id}.json"
    intelligence: Optional[ThumbnailIntelligence] = None
    if intel_path.exists():
        try:
            intelligence = ThumbnailIntelligence.model_validate_json(intel_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Module 4 intelligence for video_id={id} invalid: {exc}", id=v_id, exc=str(exc))

    # 4. Load Module 5 RedesignSpecification (optional)
    spec_path = r_spec_dir / f"{v_id}.json"
    redesign_spec: Optional[RedesignSpecification] = None
    if spec_path.exists():
        try:
            redesign_spec = RedesignSpecification.model_validate_json(spec_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Module 5 spec for video_id={id} invalid: {exc}", id=v_id, exc=str(exc))

    # 5. Load Module 8 AssetExtractionManifest (optional)
    asset_manifest_path = a_ext_dir / v_id / "asset_manifest.json"
    asset_extraction: Optional[AssetExtractionManifest] = None
    if asset_manifest_path.exists():
        try:
            manifest_candidate = AssetExtractionManifest.model_validate_json(asset_manifest_path.read_text(encoding="utf-8"))
            if manifest_candidate.status != "error":
                asset_extraction = manifest_candidate
        except Exception as exc:
            logger.warning("Module 8 asset manifest for video_id={id} invalid: {exc}", id=v_id, exc=str(exc))

    # 6. Load Module 9 DecisionManifest (optional)
    decision_manifest_path = d_dir / v_id / "decision_manifest.json"
    decision_manifest: Optional[DecisionManifest] = None
    if decision_manifest_path.exists():
        try:
            manifest_candidate = DecisionManifest.model_validate_json(decision_manifest_path.read_text(encoding="utf-8"))
            if manifest_candidate.status != "error":
                decision_manifest = manifest_candidate
        except Exception as exc:
            logger.warning("Module 9 decision manifest for video_id={id} invalid: {exc}", id=v_id, exc=str(exc))

    return PlannerInputBundle(
        video_id=v_id,
        prompt_package=prompt_package,
        workspace=workspace,
        intelligence=intelligence,
        redesign_spec=redesign_spec,
        asset_extraction=asset_extraction,
        decision_manifest=decision_manifest,
    )


def save_generation_plan(
    plan: GenerationPlan,
    plan_dir: Path = DEFAULT_GENERATION_PLAN_DIR,
) -> Path:
    """Atomically persist GenerationPlan to disk."""
    try:
        target_dir = Path(plan_dir) / plan.video_id
        target_dir.mkdir(parents=True, exist_ok=True)

        plan_path = target_dir / GENERATION_PLAN_FILENAME
        _atomic_write_json(plan.model_dump(mode="json"), plan_path)
        return plan_path
    except Exception as exc:
        raise PlanPersistError(f"Failed to persist generation plan for {plan.video_id}: {exc}") from exc


def load_cached_generation_plan(
    video_id: str,
    plan_dir: Path = DEFAULT_GENERATION_PLAN_DIR,
) -> Optional[GenerationPlan]:
    """Load cached GenerationPlan for video_id if valid."""
    v_id = _validate_video_id(video_id)
    plan_path = Path(plan_dir) / v_id / GENERATION_PLAN_FILENAME
    if not plan_path.exists():
        return None

    try:
        return GenerationPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load cached generation plan at {path}: {exc}", path=plan_path, exc=str(exc))
        return None


class PlanCache(IPlanCache):
    """File-based cache implementation for Module 10.5 GenerationPlan artifacts."""

    def __init__(self, plan_dir: Path = DEFAULT_GENERATION_PLAN_DIR) -> None:
        self.plan_dir = Path(plan_dir)

    def load(self, video_id: str) -> Optional[GenerationPlan]:
        return load_cached_generation_plan(video_id, plan_dir=self.plan_dir)

    def save(self, plan: GenerationPlan) -> Path:
        return save_generation_plan(plan, plan_dir=self.plan_dir)


def _atomic_write_json(data: Any, path: Path) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def _validate_video_id(video_id: str) -> str:
    if not video_id or not video_id.strip():
        raise ValueError("video_id must not be empty")
    return video_id.strip()
