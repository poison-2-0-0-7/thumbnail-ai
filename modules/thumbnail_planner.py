"""
thumbnail_planner.py
====================

Module 10.5 — Thumbnail Planner Orchestrator.

Aggregates outputs from Modules 4, 5, 6, 8, 9, and 10 into a deterministic,
versioned GenerationPlan artifact.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from config import (
    DEFAULT_GENERATION_PLAN_DIR,
    MODULE10_5_LOG_PATH,
    PLANNER_CACHE_ENABLED,
    PLANNER_ENGINE_VERSION,
)
from generation_components.conditioning_asset_resolver import ConditioningAssetResolver
from models import (
    GenerationPlan,
)
from planner_components.conditioning_manifest_builder import ConditioningManifestBuilder
from planner_components.headline_planner import HeadlinePlanner
from planner_components.interfaces import (
    IConditioningManifestBuilder,
    IHeadlinePlanner,
    IPlanCache,
    IPrecedenceResolver,
    IStrategyDeriver,
)
from planner_components.io import (
    PlanCache,
    load_cached_generation_plan,
    load_planner_input_bundle,
    save_generation_plan,
)
from planner_components.precedence_resolver import PrecedenceResolver
from planner_components.strategy_deriver import StrategyDeriver
from thumbnail_planner_exceptions import (
    PlanValidationError,
    ThumbnailPlannerError,
)


def _configure_logger() -> None:
    """Configure Loguru logger sink for Module 10.5."""
    MODULE10_5_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.add(
            MODULE10_5_LOG_PATH,
            rotation="10 MB",
            retention="30 days",
            enqueue=True,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
            level="DEBUG",
        )
    except ValueError:
        pass


_configure_logger()


class ThumbnailPlanner:
    """Orchestrator for Module 10.5 Thumbnail Planner."""

    def __init__(
        self,
        plan_dir: Optional[Path] = None,
        precedence_resolver: Optional[IPrecedenceResolver] = None,
        headline_planner: Optional[IHeadlinePlanner] = None,
        strategy_deriver: Optional[IStrategyDeriver] = None,
        conditioning_manifest_builder: Optional[IConditioningManifestBuilder] = None,
        plan_cache: Optional[IPlanCache] = None,
    ) -> None:
        self._plan_dir = Path(plan_dir) if plan_dir is not None else DEFAULT_GENERATION_PLAN_DIR
        self._precedence_resolver = precedence_resolver or PrecedenceResolver()
        self._headline_planner = headline_planner or HeadlinePlanner()
        self._strategy_deriver = strategy_deriver or StrategyDeriver(self._precedence_resolver)
        self._conditioning_manifest_builder = (
            conditioning_manifest_builder or ConditioningManifestBuilder()
        )
        self._plan_cache = plan_cache or PlanCache(plan_dir=self._plan_dir)
        self._conditioning_resolver = ConditioningAssetResolver()

    def plan(
        self,
        video_id: str,
        *,
        force_recompute: bool = False,
        options: Optional[dict] = None,
    ) -> GenerationPlan:
        """
        Generate or load a cached GenerationPlan for video_id.

        Args:
            video_id: YouTube video identifier.
            force_recompute: If True, ignore cache and recompute.
            options: Optional execution parameters dictionary.

        Returns:
            GenerationPlan artifact.

        Raises:
            ThumbnailPlannerError: On failure to build or load plan.
        """
        options = options or {}
        use_cache = options.get("use_cache", PLANNER_CACHE_ENABLED) and not force_recompute

        if use_cache:
            cached_plan = self._plan_cache.load(video_id)
            if cached_plan is not None:
                logger.info("Cache hit for GenerationPlan video_id='{vid}'", vid=video_id)
                return cached_plan

        logger.info("Building GenerationPlan for video_id='{vid}'", vid=video_id)

        # 1. Ingest upstream bundle
        input_bundle = load_planner_input_bundle(video_id)

        # 2. Plan headline
        headline_text, headline_source = self._headline_planner.plan_headline(
            spec=input_bundle.redesign_spec
            or self._dummy_spec_fallback(input_bundle.video_id),
            intelligence=input_bundle.intelligence,
            extraction_manifest=input_bundle.asset_extraction,
        )

        # 3. Derive strategies
        strategies = self._strategy_deriver.derive_strategies(
            workspace=input_bundle.workspace,
            decision_manifest=input_bundle.decision_manifest,
            extraction_manifest=input_bundle.asset_extraction,
            prompt_package=input_bundle.prompt_package,
            intelligence=input_bundle.intelligence,
            spec=input_bundle.redesign_spec,
        )

        # 4. Resolve conditioning context and build manifest
        context = self._conditioning_resolver.resolve(
            bundle=None,
            workspace=input_bundle.workspace,
        )
        conditioning_assets = self._conditioning_manifest_builder.build_manifest(
            context, input_bundle.asset_extraction
        )

        # 5. Compute hashes and check partial degradation
        pkg_hash = _compute_model_hash(input_bundle.prompt_package)
        ws_hash = _compute_model_hash(input_bundle.workspace)
        decision_hash = (
            _compute_model_hash(input_bundle.decision_manifest)
            if input_bundle.decision_manifest
            else None
        )
        extraction_hash = (
            _compute_model_hash(input_bundle.asset_extraction)
            if input_bundle.asset_extraction
            else None
        )

        status: str = "success"
        partial_reasons: list[str] = []

        if input_bundle.asset_extraction is None:
            status = "partial"
            partial_reasons.append("Module 8 AssetExtractionManifest missing or disabled")
            logger.warning("Module 8 asset extraction missing for video_id='{vid}'", vid=video_id)

        if input_bundle.decision_manifest is None:
            status = "partial"
            partial_reasons.append("Module 9 DecisionManifest missing or disabled")
            logger.warning("Module 9 decision manifest missing for video_id='{vid}'", vid=video_id)

        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            plan_obj = GenerationPlan(
                video_id=video_id,
                headline=headline_text,
                headline_source=headline_source,
                face_strategy=strategies["face_strategy"],
                background_strategy=strategies["background_strategy"],
                preserve_objects=strategies["preserve_objects"],
                composition_strategy=strategies["composition_strategy"],
                camera_distance=strategies["camera_distance"],
                lighting=strategies["lighting"],
                color_palette=strategies["color_palette"],
                negative_constraints=strategies["negative_constraints"],
                conditioning_assets=conditioning_assets,
                decision_manifest_hash=decision_hash,
                asset_extraction_manifest_hash=extraction_hash,
                prompt_package_hash=pkg_hash,
                workspace_hash=ws_hash,
                status=status,
                partial_failure_reasons=partial_reasons,
                engine_version=PLANNER_ENGINE_VERSION,
                generated_at=now_iso,
            )
        except Exception as exc:
            raise PlanValidationError(f"Failed to validate GenerationPlan for '{video_id}': {exc}") from exc

        # 6. Save and return
        self._plan_cache.save(plan_obj)
        logger.info("Successfully generated GenerationPlan for video_id='{vid}' status={st}", vid=video_id, st=status)
        return plan_obj

    @staticmethod
    def _dummy_spec_fallback(video_id: str):
        from models import (
            ColorDirection,
            LayoutDirection,
            RedesignSpecification,
            SubjectTreatment,
            TextOverlaySpec,
        )

        return RedesignSpecification(
            video_id=video_id,
            source_thumbnail_path="thumb.jpg",
            color_direction=ColorDirection(),
            subject_treatment=SubjectTreatment(),
            text_overlay=TextOverlaySpec(include_text=False),
            layout_direction=LayoutDirection(),
            source_ctr_potential_score=0.5,
            source_curiosity_gap_score=0.5,
            source_content_mismatch_detected=False,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )


def build_generation_plan(
    video_id: str,
    *,
    force_recompute: bool = False,
    options: Optional[dict] = None,
    plan_dir: Optional[Path] = None,
) -> GenerationPlan:
    """Free-function wrapper around ThumbnailPlanner.plan()."""
    planner = ThumbnailPlanner(plan_dir=plan_dir)
    return planner.plan(video_id, force_recompute=force_recompute, options=options)


def _compute_model_hash(model: BaseModel) -> str:
    raw = model.model_dump_json(exclude_none=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
