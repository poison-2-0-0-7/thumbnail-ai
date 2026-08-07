"""Main deterministic Edit Planner for Renderer V2 Phase 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from loguru import logger

from ..phase1.schemas import Instance, SceneGraph
from .planner_types import (
    CompositionAnalysis,
    CompositionDirectives,
    EditAction,
    EditPlanOutput,
    ObjectEditChange,
    ScoreBreakdown,
    TargetCategory,
)
from .saliency import SaliencyEngine
from .composition import CompositionEngine
from .scoring import ScoringEngine
from .planner_rules import PlannerRuleEngine


class EditPlanner:
    """Deterministic intelligence layer for YouTube thumbnail redesign.

    Decides WHAT should be changed across all objects, background, lighting, and typography
    without generating prompts or passing locked subjects through diffusion models.
    """

    def __init__(self) -> None:
        self.saliency_engine = SaliencyEngine()
        self.composition_engine = CompositionEngine()
        self.scoring_engine = ScoringEngine()
        self.rule_engine = PlannerRuleEngine()

    def plan(
        self,
        scene_graph: SceneGraph,
        original_image: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        analysis_hints: Optional[Dict[str, Any]] = None,
    ) -> EditPlanOutput:
        """Generate deterministic, structured Edit Plan from decomposed scene graph.

        Args:
            scene_graph: SceneGraph emitted by Phase 1 decomposition.
            original_image: Optional HxWx3 uint8 RGB image (defaults to scene_graph.source_image).
            depth_map: Optional HxW float32 depth map (defaults to scene_graph.depth_map).
            metadata: Optional dictionary with video title, branding, archetype, etc.
            analysis_hints: Optional auxiliary analysis hints.

        Returns:
            EditPlanOutput containing deterministic structured JSON edit directives.
        """
        logger.info("=== Starting Edit Planner (Renderer V2 Phase 2) ===")
        
        # 1. Resolve inputs
        image = original_image if original_image is not None else scene_graph.source_image
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Valid HxWx3 RGB original image is required, got {type(image)}")

        depth = depth_map if depth_map is not None else scene_graph.depth_map
        h, w, _ = image.shape
        meta = metadata or {}

        # 2. Extract primary subject mask and bounding box
        locked_instances = scene_graph.get_locked_instances()
        primary_subject_mask: Optional[np.ndarray] = None
        primary_bbox: Optional[tuple[int, int, int, int]] = None

        if locked_instances:
            # Combine locked instances masks
            combined_mask = np.zeros((h, w), dtype=np.float32)
            for inst in locked_instances:
                if inst.alpha_matte is not None:
                    combined_mask = np.maximum(combined_mask, inst.alpha_matte)
                elif inst.mask is not None:
                    combined_mask = np.maximum(combined_mask, inst.mask.astype(np.float32))
            primary_subject_mask = np.clip(combined_mask, 0.0, 1.0)
            primary_bbox = locked_instances[0].bbox
        elif scene_graph.instances:
            first_inst = scene_graph.instances[0]
            primary_subject_mask = first_inst.alpha_matte if first_inst.alpha_matte is not None else first_inst.mask.astype(np.float32)
            primary_bbox = first_inst.bbox

        # 3. Compute Deterministic Saliency & Visual Clutter
        saliency_map = self.saliency_engine.compute_saliency_map(
            image=image,
            depth_map=depth,
            depth_weight=0.25,
        )

        visual_clutter = self.saliency_engine.compute_visual_clutter(
            image=image,
            subject_mask=primary_subject_mask,
        )

        depth_variance = float(np.std(depth)) if depth is not None else 0.5
        depth_variance = float(np.clip(depth_variance * 2.5, 0.2, 1.0))

        # 4. Spatial & Composition Analysis
        analysis: CompositionAnalysis = self.composition_engine.analyze_scene(
            image=image,
            subject_mask=primary_subject_mask,
            subject_bbox=primary_bbox,
            depth_map=depth,
            saliency_map=saliency_map,
        )

        # 5. Objective Scoring (Baseline 0-100)
        scores: ScoreBreakdown = self.scoring_engine.calculate_scores(
            analysis=analysis,
            visual_clutter_score=visual_clutter,
            depth_variance=depth_variance,
            locked_identity_intact=True,
        )

        # 6. Deterministic Rule Decisions for Objects & Background
        changes: List[ObjectEditChange] = self.rule_engine.evaluate_instance_decisions(
            scene_graph=scene_graph,
            analysis=analysis,
            scores=scores,
            metadata=meta,
        )

        # 7. Composition Directives
        directives: CompositionDirectives = self.rule_engine.derive_composition_directives(
            analysis=analysis,
            scores=scores,
            scene_graph=scene_graph,
            metadata=meta,
        )

        # 8. Projected Target Score Post-Edits
        # Proposed edits improve contrast, background quality, and rule of thirds
        target_score = float(round(min(98.5, scores.overall + max(12.0, (100.0 - scores.overall) * 0.65)), 2))

        # 9. Strategic Summary
        summary = self.rule_engine.generate_strategic_summary(
            changes=changes,
            analysis=analysis,
            scores=scores,
            target_score=target_score,
        )

        locked_ids = [inst.instance_id for inst in locked_instances]

        # Quality validation targets
        quality_targets = {
            "min_identity_similarity": 0.90,
            "min_composition_preservation": 0.85,
            "min_brand_preservation": 0.90,
            "min_readability_score": 0.80,
            "target_contrast_ratio": 4.50,
        }

        plan_output = EditPlanOutput(
            summary=summary,
            composition_score=scores.overall,
            target_composition_score=target_score,
            changes=changes,
            scoring_breakdown=scores,
            composition_analysis=analysis,
            composition_directives=directives,
            locked_instances=locked_ids,
            quality_targets=quality_targets,
            metadata={
                "image_dimensions": f"{w}x{h}",
                "instance_count": len(scene_graph.instances),
                "locked_count": len(locked_instances),
                "archetype": meta.get("archetype", "single_creator_face"),
                "channel_id": meta.get("channel_id", "default_channel"),
            },
        )

        logger.info("Generated EditPlan successfully: baseline_score={b:.1f}, changes_count={c}", b=scores.overall, c=len(changes))
        return plan_output

    def save_plan(self, plan: EditPlanOutput, output_path: Union[str, Path]) -> Path:
        """Save structured Edit Plan to a JSON file."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(plan.to_json(indent=2))
        return out_file
