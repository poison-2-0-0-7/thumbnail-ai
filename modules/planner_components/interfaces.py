"""
interfaces.py
=============

Interfaces for Module 10.5 Thumbnail Planner components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from generation_components.conditioning_asset_resolver import GenerationConditioningContext
from models import (
    AssetExtractionManifest,
    CompositionWorkspace,
    DecisionManifest,
    GenerationPlan,
    HeadlineSource,
    PlanConditioningAsset,
    PromptPackage,
    RedesignSpecification,
    ThumbnailIntelligence,
)


class IPrecedenceResolver(Protocol):
    """Resolves decision precedence across Module 8, 9, 10, and 5 artifacts."""

    def resolve_layer_decisions(
        self,
        workspace: CompositionWorkspace,
        decision_manifest: DecisionManifest | None = None,
    ) -> list[tuple[str, str, str, str]]:
        """Returns list of (element_key, role, decision_value, rationale)."""
        ...


class IHeadlinePlanner(Protocol):
    """Plans thumbnail headline text passthrough or placeholder."""

    def plan_headline(
        self,
        spec: RedesignSpecification,
        intelligence: ThumbnailIntelligence | None = None,
        extraction_manifest: AssetExtractionManifest | None = None,
    ) -> tuple[str, HeadlineSource]:
        """Derives (headline_text, headline_source)."""
        ...


class IStrategyDeriver(Protocol):
    """Derives high-level generation strategies from upstream artifacts."""

    def derive_strategies(
        self,
        workspace: CompositionWorkspace,
        decision_manifest: DecisionManifest | None = None,
        extraction_manifest: AssetExtractionManifest | None = None,
        prompt_package: PromptPackage | None = None,
        intelligence: ThumbnailIntelligence | None = None,
        spec: RedesignSpecification | None = None,
    ) -> dict[str, Any]:
        """
        Returns dict containing face_strategy, background_strategy, preserve_objects,
        composition_strategy, camera_distance, lighting, color_palette, negative_constraints.
        """
        ...


class IConditioningManifestBuilder(Protocol):
    """Converts resolved GenerationConditioningContext into PlanConditioningAsset list."""

    def build_manifest(
        self,
        context: GenerationConditioningContext,
        extraction_manifest: AssetExtractionManifest | None = None,
    ) -> list[PlanConditioningAsset]:
        """Builds list of PlanConditioningAsset objects."""
        ...


class IPlanCache(Protocol):
    """Cache interface for saving/loading GenerationPlan artifacts."""

    def load(self, video_id: str) -> GenerationPlan | None:
        ...

    def save(self, plan: GenerationPlan) -> Path:
        ...
