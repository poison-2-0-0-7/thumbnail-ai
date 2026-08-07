"""Deterministic rule engine for object-level decisions and composition directives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..phase1.schemas import Instance, SceneGraph
from .planner_types import (
    CompositionAnalysis,
    CompositionDirectives,
    EditAction,
    ObjectEditChange,
    ScoreBreakdown,
    TargetCategory,
)


class PlannerRuleEngine:
    """Evaluates scene graphs and composition analysis to emit deterministic edit instructions."""

    @staticmethod
    def evaluate_instance_decisions(
        scene_graph: SceneGraph,
        analysis: CompositionAnalysis,
        scores: ScoreBreakdown,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ObjectEditChange]:
        """Determine deterministic edit actions for every detected instance in the scene."""
        changes: List[ObjectEditChange] = []
        w, h = scene_graph.width, scene_graph.height

        creator_instances = [inst for inst in scene_graph.instances if inst.cls in ("creator", "person") or inst.locked]
        logo_instances = [inst for inst in scene_graph.instances if inst.cls == "logo"]
        product_instances = [inst for inst in scene_graph.instances if inst.cls == "product"]
        other_instances = [inst for inst in scene_graph.instances if inst.cls == "other" and not inst.locked]

        # 1. Evaluate Creator / Face Instances
        for idx, inst in enumerate(creator_instances):
            xmin, ymin, xmax, ymax = inst.bbox
            bw, bh = max(1, xmax - xmin), max(1, ymax - ymin)
            area_fraction = float(bw * bh) / float(w * h)

            # Rule 1.1: Always KEEP locked creator identity
            changes.append(
                ObjectEditChange(
                    target=inst.instance_id,
                    action=EditAction.KEEP,
                    reason="Preserve creator identity and facial features as locked raster layer without generative diffusion drift",
                    target_category=TargetCategory.CREATOR_FACE if "face" in inst.instance_id.lower() else TargetCategory.CREATOR_BODY,
                    parameters={"locked": True, "instance_id": inst.instance_id, "bbox": inst.bbox},
                    confidence=1.0,
                    priority=1,
                )
            )

            # Rule 1.2: Check if Creator needs RESIZE / SCALE
            if area_fraction < 0.28:
                recommended_scale = float(round(min(1.45, 0.35 / max(0.05, area_fraction)), 2))
                changes.append(
                    ObjectEditChange(
                        target=inst.instance_id,
                        action=EditAction.RESIZE,
                        reason=f"Scale creator by {recommended_scale}x to achieve dominant thumbnail subject prominence (target 30-40% canvas)",
                        target_category=TargetCategory.CREATOR_BODY,
                        parameters={"scale_factor": recommended_scale, "target_area_pct": 0.35},
                        confidence=0.95,
                        priority=2,
                    )
                )

            # Rule 1.3: Check if Creator needs MOVE for Rule of Thirds alignment
            cx = float(xmin + xmax) / (2.0 * w)
            cy = float(ymin + ymax) / (2.0 * h)
            if analysis.rule_of_thirds_alignment < 0.65:
                # Suggest repositioning toward nearest rule-of-thirds power line
                target_cx = 0.67 if cx >= 0.5 else 0.33
                target_cy = 0.50
                changes.append(
                    ObjectEditChange(
                        target=inst.instance_id,
                        action=EditAction.MOVE,
                        reason=f"Reposition creator centroid from ({cx:.2f}, {cy:.2f}) to rule-of-thirds power point ({target_cx:.2f}, {target_cy:.2f})",
                        target_category=TargetCategory.CREATOR_BODY,
                        parameters={"target_centroid": (target_cx, target_cy), "anchor": "power_point"},
                        confidence=0.90,
                        priority=3,
                    )
                )

            # Rule 1.4: Check if Creator needs RELIGHT
            if analysis.contrast_ratio < 4.5 or scores.contrast < 75.0:
                changes.append(
                    ObjectEditChange(
                        target=inst.instance_id,
                        action=EditAction.RELIGHT,
                        reason="Relight foreground creator with directional key and rim lighting for high-contrast subject separation",
                        target_category=TargetCategory.CREATOR_BODY,
                        parameters={
                            "key_light_direction": "top_left",
                            "key_light_angle_deg": 135,
                            "rim_light_enabled": True,
                            "rim_light_strength": 0.75,
                            "color_temp_k": 5600,
                        },
                        confidence=0.95,
                        priority=2,
                    )
                )

            # Rule 1.5: Enhance facial features and contrast
            changes.append(
                ObjectEditChange(
                    target=inst.instance_id,
                    action=EditAction.ENHANCE,
                    reason="Apply micro-contrast and edge clarity enhancement to creator eyes and facial expression",
                    target_category=TargetCategory.CREATOR_FACE,
                    parameters={"clarity_boost": 0.20, "sharpen_radius": 1.5},
                    confidence=0.90,
                    priority=4,
                )
            )

        # 2. Evaluate Logo / Branding Instances
        for inst in logo_instances:
            xmin, ymin, xmax, ymax = inst.bbox
            # Check if logo is inside the dangerous bottom-right YouTube timestamp safe zone (x > 75%, y > 75%)
            in_yt_badge_zone = (xmax > int(w * 0.75)) and (ymax > int(h * 0.75))
            if in_yt_badge_zone:
                changes.append(
                    ObjectEditChange(
                        target=inst.instance_id,
                        action=EditAction.MOVE,
                        reason="Relocate brand logo from bottom-right to top-left to avoid occlusion by YouTube video duration badge",
                        target_category=TargetCategory.LOGO,
                        parameters={"target_box": (int(w * 0.06), int(h * 0.06), int(w * 0.20), int(h * 0.20))},
                        confidence=1.0,
                        priority=1,
                    )
                )
            else:
                changes.append(
                    ObjectEditChange(
                        target=inst.instance_id,
                        action=EditAction.KEEP,
                        reason="Preserve brand logo intact with clean alpha matte edges",
                        target_category=TargetCategory.LOGO,
                        parameters={"locked": True},
                        confidence=1.0,
                        priority=2,
                    )
                )

        # 3. Evaluate Product Instances
        for inst in product_instances:
            changes.append(
                ObjectEditChange(
                    target=inst.instance_id,
                    action=EditAction.KEEP,
                    reason="Preserve core product object geometry and surface details",
                    target_category=TargetCategory.PRODUCT,
                    parameters={"locked": True},
                    confidence=0.95,
                    priority=2,
                )
            )
            changes.append(
                ObjectEditChange(
                    target=inst.instance_id,
                    action=EditAction.ENHANCE,
                    reason="Enhance product specular highlights and edge contrast for crisp visual pop",
                    target_category=TargetCategory.PRODUCT,
                    parameters={"specular_boost": 0.25},
                    confidence=0.85,
                    priority=3,
                )
            )

        # 4. Evaluate Stray / Clutter Instances (cls == 'other')
        for inst in other_instances:
            changes.append(
                ObjectEditChange(
                    target=inst.instance_id,
                    action=EditAction.REMOVE,
                    reason="Remove distracting secondary clutter to clarify focus hierarchy and open text safe zones",
                    target_category=TargetCategory.CLUTTER,
                    parameters={"inpaint_fill": True},
                    confidence=0.90,
                    priority=2,
                )
            )

        # 5. Evaluate Background
        if scores.visual_clutter < 70.0 or scores.contrast < 65.0 or scores.background_quality < 65.0:
            changes.append(
                ObjectEditChange(
                    target="background",
                    action=EditAction.REPLACE,
                    reason="Replace low-contrast, cluttered background with a depth-layered stylized studio backdrop to boost subject contrast",
                    target_category=TargetCategory.BACKGROUND,
                    parameters={
                        "depth_style": "shallow_dof",
                        "palette_ref": "brand_contrast_palette",
                        "lighting_sync": "top_left",
                    },
                    confidence=0.95,
                    priority=1,
                )
            )
        else:
            # Clean background: Apply blur / desaturation to increase subject separation
            changes.append(
                ObjectEditChange(
                    target="background",
                    action=EditAction.BLUR,
                    reason="Apply subtle depth-of-field Gaussian blur to background to enhance focal separation",
                    target_category=TargetCategory.BACKGROUND,
                    parameters={"blur_sigma": 6.0, "preserve_depth": True},
                    confidence=0.90,
                    priority=3,
                )
            )
            changes.append(
                ObjectEditChange(
                    target="background",
                    action=EditAction.DESATURATE,
                    reason="Desaturate background by 15% to increase chromatic contrast against vibrant foreground subject",
                    target_category=TargetCategory.BACKGROUND,
                    parameters={"saturation_factor": 0.85},
                    confidence=0.85,
                    priority=4,
                )
            )

        return changes

    @staticmethod
    def derive_composition_directives(
        analysis: CompositionAnalysis,
        scores: ScoreBreakdown,
        scene_graph: SceneGraph,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CompositionDirectives:
        """Derive target geometric and rendering directives for downstream compositor and typography engines."""
        w, h = scene_graph.width, scene_graph.height

        # Target subject scale: optimal range for YouTube thumbnails is 0.35 (35% of canvas)
        target_scale = 0.35 if analysis.subject_scale < 0.28 else min(0.42, max(0.30, analysis.subject_scale))

        # Target subject position: Rule-of-thirds alignment
        cur_x, cur_y = analysis.subject_position
        target_cx = 0.67 if cur_x >= 0.50 else 0.33
        target_cy = 0.50

        # Primary text safe zone
        if analysis.text_safe_zones:
            rec_zone = analysis.text_safe_zones[0]
        else:
            # Default to top-left quadrant safe zone
            rec_zone = (int(w * 0.06), int(h * 0.06), int(w * 0.55), int(h * 0.45))

        # Ordered depth layering
        layer_order = ["background", "shadow_sync", "locked_instances", "typography", "graphic_overlays", "color_grade"]

        # Color palette target based on harmony
        if analysis.color_harmony in ("complementary", "split_complementary"):
            color_target = ["#FF2E63", "#08D9D6", "#0F172A", "#FFFFFF"]
        else:
            color_target = ["#F59E0B", "#3B82F6", "#1E293B", "#FFFFFF"]

        contrast_boost = 1.25 if scores.contrast < 70.0 else 1.10

        return CompositionDirectives(
            target_subject_scale=float(round(target_scale, 2)),
            target_subject_position=(float(round(target_cx, 2)), float(round(target_cy, 2)),),
            rule_of_thirds_target=(0.67, 0.50),
            recommended_text_zone=rec_zone,
            depth_layering_order=layer_order,
            lighting_direction="top_left",
            color_palette_target=color_target,
            contrast_boost_factor=contrast_boost,
        )

    @staticmethod
    def generate_strategic_summary(
        changes: List[ObjectEditChange],
        analysis: CompositionAnalysis,
        scores: ScoreBreakdown,
        target_score: float,
    ) -> str:
        """Synthesize a human-auditable strategic explanation of all proposed edits."""
        actions_summary = []
        for ch in changes:
            actions_summary.append(f"{ch.action.value.upper()} on '{ch.target}' ({ch.reason})")

        summary_text = (
            f"Thumbnail Optimization Plan (Baseline Score: {scores.overall:.1f}/100 -> Target: {target_score:.1f}/100). "
            f"Subject occupies {analysis.subject_scale * 100:.1f}% canvas with {analysis.contrast_ratio:.1f}:1 contrast ratio. "
            f"Strategic operations: {'; '.join(actions_summary[:4])}."
        )
        return summary_text
