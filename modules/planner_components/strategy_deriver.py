"""
strategy_deriver.py
===================

Implements Section 6.5 strategy derivation logic.
Pure logic, zero I/O.
"""

from __future__ import annotations

from typing import Any, Optional

from models import (
    AssetExtractionManifest,
    BackgroundStrategy,
    CompositionWorkspace,
    DecisionManifest,
    FaceStrategy,
    LayerDecision,
    LayerRole,
    PromptPackage,
    RedesignSpecification,
    ThumbnailIntelligence,
)
from planner_components.interfaces import IStrategyDeriver
from planner_components.precedence_resolver import PrecedenceResolver


class StrategyDeriver(IStrategyDeriver):
    """Derives high-level generation strategies from upstream artifacts per §6.5."""

    def __init__(self, precedence_resolver: Optional[PrecedenceResolver] = None) -> None:
        self._precedence_resolver = precedence_resolver or PrecedenceResolver()

    def derive_strategies(
        self,
        workspace: CompositionWorkspace,
        decision_manifest: Optional[DecisionManifest] = None,
        extraction_manifest: Optional[AssetExtractionManifest] = None,
        prompt_package: Optional[PromptPackage] = None,
        intelligence: Optional[ThumbnailIntelligence] = None,
        spec: Optional[RedesignSpecification] = None,
    ) -> dict[str, Any]:
        """
        Derives all strategy fields per §6.5.
        """
        resolved_decisions = self._precedence_resolver.resolve_layer_decisions(
            workspace, decision_manifest
        )

        # 1. Face Strategy
        face_strat = self._derive_face_strategy(resolved_decisions, extraction_manifest)

        # 2. Background Strategy
        bg_strat = self._derive_background_strategy(resolved_decisions, workspace, extraction_manifest)

        # 3. Preserve Objects list
        preserve_objects: list[str] = []
        for key, role, decision, _ in resolved_decisions:
            if role == LayerRole.OBJECT.value and decision in (
                LayerDecision.KEEP.value,
                LayerDecision.ENHANCE.value,
                "keep",
                "enhance",
            ):
                preserve_objects.append(key)

        # 4. Composition Strategy
        composition_strategy = "rule_of_thirds"
        if spec and hasattr(spec, "layout_direction") and spec.layout_direction:
            composition_strategy = str(spec.layout_direction.focal_point or "rule_of_thirds")
        elif intelligence and hasattr(intelligence, "composition") and intelligence.composition:
            composition_strategy = str(getattr(intelligence.composition, "grid_alignment", "rule_of_thirds"))

        # 5. Camera Distance
        camera_distance = "medium"
        if spec and hasattr(spec, "subject_treatment") and spec.subject_treatment:
            camera_distance = "close_up" if spec.subject_treatment.crop_tighter else "medium"

        # 6. Lighting
        lighting = "neutral"
        if (
            extraction_manifest
            and extraction_manifest.visual_properties
            and extraction_manifest.visual_properties.lighting_direction
        ):
            lighting = extraction_manifest.visual_properties.lighting_direction
        elif spec and hasattr(spec, "color_direction") and spec.color_direction:
            lighting = spec.color_direction.warm_or_cool or "neutral"

        # 7. Color Palette
        color_palette: list[str] = []
        if (
            extraction_manifest
            and extraction_manifest.visual_properties
            and extraction_manifest.visual_properties.extended_palette
        ):
            color_palette = list(extraction_manifest.visual_properties.extended_palette)
        elif intelligence and hasattr(intelligence, "colors") and intelligence.colors:
            color_palette = list(intelligence.colors.dominant_colors)
        elif spec and hasattr(spec, "color_direction") and spec.color_direction:
            color_palette = list(getattr(spec.color_direction, "color_palette", []) or [])

        # 8. Negative Constraints
        negative_constraints: list[str] = []
        if prompt_package:
            raw_constraints = []
            if prompt_package.rendering_constraints:
                raw_constraints.extend(prompt_package.rendering_constraints)
            if prompt_package.safety_constraints:
                raw_constraints.extend(prompt_package.safety_constraints)

            for rc in raw_constraints:
                rc_str = str(rc).strip()
                lower_rc = rc_str.lower()
                if lower_rc.startswith("preserve") or "elements exactly" in lower_rc:
                    continue
                if rc_str and rc_str not in negative_constraints:
                    negative_constraints.append(rc_str)

        # Add explicit negative constraints for REMOVE decisions
        for key, _, decision, _ in resolved_decisions:
            if decision in (LayerDecision.REMOVE.value, "remove"):
                neg = f"no {key}"
                if neg not in negative_constraints:
                    negative_constraints.append(neg)

        return {
            "face_strategy": face_strat,
            "background_strategy": bg_strat,
            "preserve_objects": preserve_objects,
            "composition_strategy": composition_strategy,
            "camera_distance": camera_distance,
            "lighting": lighting,
            "color_palette": color_palette,
            "negative_constraints": negative_constraints,
        }

    def _derive_face_strategy(
        self,
        resolved_decisions: list[tuple[str, str, str, str]],
        extraction_manifest: Optional[AssetExtractionManifest],
    ) -> FaceStrategy:
        person_decisions = [
            (d, rat)
            for key, role, d, rat in resolved_decisions
            if role == LayerRole.PERSON.value or key.startswith("person")
        ]
        if not person_decisions:
            return FaceStrategy.NONE

        decision = person_decisions[0][0]
        if decision in (LayerDecision.REMOVE.value, "remove"):
            return FaceStrategy.NONE

        has_identity_lock = False
        if extraction_manifest and extraction_manifest.people:
            for p in extraction_manifest.people:
                if p.face_embedding is not None:
                    has_identity_lock = True
                    break

        if decision in (LayerDecision.ENHANCE.value, "enhance"):
            return (
                FaceStrategy.ENHANCE_EXISTING_IDENTITY_LOCKED
                if has_identity_lock
                else FaceStrategy.ENHANCE_EXISTING
            )
        else:
            return (
                FaceStrategy.PRESERVE_AS_IS_IDENTITY_LOCKED
                if has_identity_lock
                else FaceStrategy.PRESERVE_AS_IS
            )

    def _derive_background_strategy(
        self,
        resolved_decisions: list[tuple[str, str, str, str]],
        workspace: CompositionWorkspace,
        extraction_manifest: Optional[AssetExtractionManifest],
    ) -> BackgroundStrategy:
        bg_decisions = [
            d for key, role, d, _ in resolved_decisions if role == LayerRole.BACKGROUND.value
        ]
        if bg_decisions and bg_decisions[0] in (LayerDecision.KEEP.value, "keep"):
            return BackgroundStrategy.KEEP

        has_structure_guidance = False
        for layer in workspace.layers:
            if layer.depth_hint_path or layer.canny_hint_path:
                has_structure_guidance = True
                break

        if extraction_manifest and extraction_manifest.scene:
            if extraction_manifest.scene.depth_map or extraction_manifest.scene.segmentation_map:
                has_structure_guidance = True

        return (
            BackgroundStrategy.STRUCTURE_GUIDED_REPLACE
            if has_structure_guidance
            else BackgroundStrategy.UNGUIDED_REPLACE
        )
