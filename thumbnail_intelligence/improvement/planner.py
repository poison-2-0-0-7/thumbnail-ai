"""
planner.py
==========

ModificationPlanner Implementation for Phase 5.5.
Translates an ImprovementPlan into targeted layer modifications on a base RenderExecutionPackage.
Preserves unchanged layers, resolves parameter conflicts deterministically, and validates canvas geometries.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Set, Tuple

from thumbnail_intelligence.critique.models import ImprovementPlan, ImprovementSuggestion
from thumbnail_intelligence.improvement.models import (
    ImprovementExecutionPlan,
    ImprovementStrategyType,
    LayerModification,
    ModificationReport,
)
from thumbnail_intelligence.reasoning.renderer_adapter_models import (
    PixelBoundingBox,
    RenderBackgroundInstruction,
    RenderExecutionPackage,
    RenderLightingInstruction,
    RenderPlacementCoordinate,
    RenderTypographyInstruction,
)

logger = logging.getLogger(__name__)


class PlannerValidationError(RuntimeError):
    """Exception raised for modification planning or geometry validation errors."""
    pass


class ModificationPlanner:
    """Targeted modification planner updating RenderExecutionPackage while preserving unchanged layers."""

    def plan_modifications(
        self,
        base_package: RenderExecutionPackage,
        suggestions: List[ImprovementSuggestion],
        strategy: ImprovementStrategyType = ImprovementStrategyType.BALANCED,
    ) -> Tuple[RenderExecutionPackage, ImprovementExecutionPlan, ModificationReport]:
        """Plan and apply targeted modifications to a base RenderExecutionPackage.

        Args:
            base_package: Source RenderExecutionPackage.
            suggestions: Prioritized list of ImprovementSuggestion objects.
            strategy: ImprovementStrategyType governing suggestion filtering.

        Returns:
            Tuple of (Updated RenderExecutionPackage, ImprovementExecutionPlan, ModificationReport).
        """
        if not base_package:
            raise PlannerValidationError("Base RenderExecutionPackage cannot be None.")

        # Filter suggestions by strategy limit
        if strategy == ImprovementStrategyType.CONSERVATIVE:
            active_suggs = suggestions[:2]
        elif strategy == ImprovementStrategyType.BALANCED:
            active_suggs = suggestions[:4]
        else:  # AGGRESSIVE
            active_suggs = list(suggestions)

        canvas_w = base_package.scene_graph.canvas_width_px
        canvas_h = base_package.scene_graph.canvas_height_px

        modified_layer_ids: Set[str] = set()
        layer_modifications: List[LayerModification] = []
        expected_gain_pts = 0.0

        # Create mutable working copies of package components
        new_typo = list(base_package.typography_instructions)
        new_lighting = list(base_package.lighting_instructions)
        new_bg = base_package.background_instruction.model_copy(deep=True)
        new_placements = list(base_package.placement_coordinates)

        # Merge parameter changes to resolve conflicts deterministically
        merged_changes: Dict[str, Any] = {}
        for sug in active_suggs:
            expected_gain_pts += sug.expected_ctr_gain
            for k, v in sug.parameter_changes.items():
                if k in merged_changes:
                    # Conflict resolution: take maximum multiplier or non-None override
                    if isinstance(v, (int, float)) and isinstance(merged_changes[k], (int, float)):
                        merged_changes[k] = max(v, merged_changes[k])
                    else:
                        merged_changes[k] = v
                else:
                    merged_changes[k] = v

        # 1. Apply Typography Modifications
        typo_modified = False
        for idx, typo in enumerate(new_typo):
            orig_params = {
                "font_size_px": typo.font_size_px,
                "stroke_width_px": typo.stroke_width_px,
                "font_color_hex": typo.font_color_hex,
                "stroke_color_hex": typo.stroke_color_hex,
            }

            mult_size = merged_changes.get("typography_scale_multiplier", 1.0)
            mult_stroke = merged_changes.get("stroke_width_multiplier", 1.0)
            new_col = merged_changes.get("font_color_hex", typo.font_color_hex)
            new_stroke_col = merged_changes.get("stroke_color_hex", typo.stroke_color_hex)

            if mult_size != 1.0 or mult_stroke != 1.0 or new_col != typo.font_color_hex or new_stroke_col != typo.stroke_color_hex:
                new_size = int(round(typo.font_size_px * mult_size))
                new_stroke_w = int(round(typo.stroke_width_px * mult_stroke))

                new_typo[idx] = typo.model_copy(
                    update={
                        "font_size_px": new_size,
                        "stroke_width_px": new_stroke_w,
                        "font_color_hex": new_col,
                        "stroke_color_hex": new_stroke_col,
                    }
                )
                typo_modified = True
                layer_id = f"layer_{typo.text_id}"
                modified_layer_ids.add(layer_id)

                layer_modifications.append(
                    LayerModification(
                        modification_id=f"mod_typo_{uuid.uuid4().hex[:6]}",
                        layer_id=layer_id,
                        element_name="Headline Typography Overlay",
                        modification_type="typography_resizing_recoloring",
                        action_type="increase_font_size",
                        original_params=orig_params,
                        new_params={
                            "font_size_px": new_size,
                            "stroke_width_px": new_stroke_w,
                            "font_color_hex": new_col,
                            "stroke_color_hex": new_stroke_col,
                        },
                    )
                )

        # 2. Apply Lighting Modifications
        lighting_modified = False
        for idx, light in enumerate(new_lighting):
            orig_params = {
                "key_light_intensity": light.key_light_intensity,
                "rim_light_enabled": light.rim_light_enabled,
            }

            mult_light = merged_changes.get("key_light_intensity_multiplier", 1.0)
            rim_flag = merged_changes.get("rim_light_enabled_override", light.rim_light_enabled)

            if mult_light != 1.0 or rim_flag != light.rim_light_enabled:
                new_intensity = min(1.0, max(0.0, light.key_light_intensity * mult_light))
                new_lighting[idx] = light.model_copy(
                    update={
                        "key_light_intensity": new_intensity,
                        "rim_light_enabled": rim_flag,
                    }
                )
                lighting_modified = True
                layer_id = f"layer_lighting_{light.target_element_id}"
                modified_layer_ids.add(layer_id)

                layer_modifications.append(
                    LayerModification(
                        modification_id=f"mod_light_{uuid.uuid4().hex[:6]}",
                        layer_id=layer_id,
                        element_name=f"Relighting layer for {light.target_element_id}",
                        modification_type="lighting_adjustments",
                        action_type="boost_key_lighting",
                        original_params=orig_params,
                        new_params={
                            "key_light_intensity": new_intensity,
                            "rim_light_enabled": rim_flag,
                        },
                    )
                )

        # 3. Apply Background Modifications
        orig_bg_style = base_package.background_instruction.style_prompt_direction
        orig_bg_colors = base_package.background_instruction.dominant_colors
        new_bg_style = merged_changes.get("background_style_direction", orig_bg_style)
        new_bg_colors = merged_changes.get("dominant_colors_override", orig_bg_colors)

        if new_bg_style != orig_bg_style or new_bg_colors != orig_bg_colors:
            new_bg = base_package.background_instruction.model_copy(
                update={
                    "style_prompt_direction": new_bg_style,
                    "dominant_colors": new_bg_colors,
                }
            )
            layer_id = "layer_elem_01_background"
            modified_layer_ids.add(layer_id)

            layer_modifications.append(
                LayerModification(
                    modification_id=f"mod_bg_{uuid.uuid4().hex[:6]}",
                    layer_id=layer_id,
                    element_name="Background Generation Layer",
                    modification_type="background_regeneration",
                    action_type="reduce_clutter",
                    original_params={"style": orig_bg_style, "colors": orig_bg_colors},
                    new_params={"style": new_bg_style, "colors": new_bg_colors},
                )
            )

        # 4. Apply Placement Coordinates Modifications (Subject Scaling & Repositioning)
        scale_mult = merged_changes.get("subject_scale_multiplier", 1.0)
        dx_pct = merged_changes.get("subject_x_offset_pct", 0.0)
        dy_pct = merged_changes.get("subject_y_offset_pct", 0.0)

        for idx, p in enumerate(new_placements):
            is_subject = (
                "subject" in p.element_name.lower()
                or "hero" in p.element_id.lower()
                or "subject" in p.element_id.lower()
                or "person" in p.element_name.lower()
            )
            if is_subject and (scale_mult != 1.0 or dx_pct != 0.0 or dy_pct != 0.0):
                orig_params = {
                    "scale": p.scale,
                    "bbox_pixels": p.bbox_pixels.to_tuple(),
                }

                new_scale = p.scale * scale_mult
                dx = int(round(dx_pct * canvas_w))
                dy = int(round(dy_pct * canvas_h))

                old_box = p.bbox_pixels
                new_w = max(1, min(canvas_w, int(round(old_box.width_px * scale_mult))))
                new_h = max(1, min(canvas_h, int(round(old_box.height_px * scale_mult))))
                new_x = max(0, min(canvas_w - new_w, old_box.x_px + dx))
                new_y = max(0, min(canvas_h - new_h, old_box.y_px + dy))

                new_box = PixelBoundingBox(
                    x_px=new_x,
                    y_px=new_y,
                    width_px=new_w,
                    height_px=new_h,
                )

                new_placements[idx] = p.model_copy(update={"scale": new_scale, "bbox_pixels": new_box})
                layer_id = f"layer_{p.element_id}"
                modified_layer_ids.add(layer_id)

                layer_modifications.append(
                    LayerModification(
                        modification_id=f"mod_place_{uuid.uuid4().hex[:6]}",
                        layer_id=layer_id,
                        element_name=p.element_name,
                        modification_type="face_scaling_repositioning",
                        action_type="scale_subject",
                        original_params=orig_params,
                        new_params={
                            "scale": new_scale,
                            "bbox_pixels": new_box.to_tuple(),
                        },
                    )
                )

        # Assemble Updated Package
        new_pkg_id = f"pkg_render_improved_{uuid.uuid4().hex[:8]}"
        new_meta = base_package.metadata.model_copy(update={"package_id": new_pkg_id})

        updated_package = base_package.model_copy(
            update={
                "metadata": new_meta,
                "typography_instructions": new_typo,
                "lighting_instructions": new_lighting,
                "background_instruction": new_bg,
                "placement_coordinates": new_placements,
            }
        )

        # Calculate preserved vs modified layers
        all_layer_ids = [l.layer_id for l in base_package.layer_stack]
        preserved_layer_ids = [lid for lid in all_layer_ids if lid not in modified_layer_ids]

        total_cnt = len(all_layer_ids)
        mod_cnt = len(modified_layer_ids)
        pres_cnt = len(preserved_layer_ids)
        pres_ratio = pres_cnt / float(total_cnt) if total_cnt > 0 else 1.0

        exec_plan = ImprovementExecutionPlan(
            plan_id=f"plan_exec_{uuid.uuid4().hex[:8]}",
            base_package_id=base_package.metadata.package_id,
            strategy_used=strategy,
            modified_layer_ids=list(modified_layer_ids),
            preserved_layer_ids=preserved_layer_ids,
            layer_modifications=layer_modifications,
        )

        mod_names = [m.element_name for m in layer_modifications]
        pres_names = [l.layer_name for l in base_package.layer_stack if l.layer_id in preserved_layer_ids]

        # Render cost estimation: LOW if <= 2 layers modified, MEDIUM if <= 4, HIGH if > 4
        cost_est = "LOW" if mod_cnt <= 2 else ("MEDIUM" if mod_cnt <= 4 else "HIGH")

        report = ModificationReport(
            report_id=f"mod_report_{uuid.uuid4().hex[:8]}",
            base_package_id=base_package.metadata.package_id,
            updated_package_id=new_pkg_id,
            total_layers_count=total_cnt,
            modified_layers_count=mod_cnt,
            preserved_layers_count=pres_cnt,
            preservation_ratio=round(pres_ratio, 2),
            expected_ctr_gain_pts=round(min(25.0, expected_gain_pts * 0.6), 1),
            estimated_render_cost=cost_est,
            modified_layer_names=mod_names,
            preserved_layer_names=pres_names,
        )

        return updated_package, exec_plan, report
