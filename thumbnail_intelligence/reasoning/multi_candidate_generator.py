"""
multi_candidate_generator.py
============================

MultiCandidateGenerator Implementation (Phase 5.1).
Generates multiple strategically distinct thumbnail candidates (Candidate A, B, C, D, E)
from a base RenderExecutionPackage, DesignBrief, or SpatialComposition + ExecutionPlan.

Variations are driven by strategic VariationProfiles (Emotional, Curiosity, Typography, Color, Composition),
NOT random seeds alone. Reuses existing RendererV2Pipeline without duplicating rendering logic.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from thumbnail_intelligence.knowledge_base.models import _utc_now_iso
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.execution_plan_models import ExecutionPlan
from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner
from thumbnail_intelligence.reasoning.multi_candidate_models import (
    CandidateDescriptor,
    CandidateMetadata,
    CandidateResult,
    CandidateSet,
    VariationDimension,
    VariationProfile,
)
from thumbnail_intelligence.reasoning.renderer_adapter import RendererV2Adapter
from thumbnail_intelligence.reasoning.renderer_adapter_models import (
    PixelBoundingBox,
    RenderBackgroundInstruction,
    RenderExecutionPackage,
    RenderLightingInstruction,
    RenderPlacementCoordinate,
    RenderTypographyInstruction,
)
from thumbnail_intelligence.reasoning.spatial_composition_models import SpatialComposition
from thumbnail_intelligence.reasoning.spatial_composition_planner import SpatialCompositionPlanner

logger = logging.getLogger(__name__)


class MultiCandidateGeneratorError(RuntimeError):
    """Exception raised for candidate generation failures."""
    pass


class MultiCandidateGenerator:
    """Orchestrates generation of strategically distinct thumbnail candidates using RendererV2Pipeline."""

    def __init__(self, pipeline: Optional[Any] = None) -> None:
        if pipeline is None:
            from renderer_v2.pipeline import RendererV2Pipeline
            self.pipeline = RendererV2Pipeline()
        else:
            self.pipeline = pipeline
        self.adapter = RendererV2Adapter()

    @staticmethod
    def create_default_profiles(count: int = 5) -> List[VariationProfile]:
        """Create 5 default strategic variation profiles for Candidates A through E."""
        profiles = [
            # Candidate A: Emotional & High-Impact Hero
            VariationProfile(
                profile_id="profile_cand_a_emotional",
                profile_name="Candidate A (Emotional Emphasis)",
                primary_dimension=VariationDimension.EMOTIONAL_EMPHASIS,
                secondary_dimension=VariationDimension.FACE_EMPHASIS,
                subject_scale_multiplier=1.15,
                key_light_intensity_multiplier=1.2,
                rim_light_enabled_override=True,
                background_style_direction="vibrant cinematic studio background with warm dramatic lighting",
                dominant_colors_override=["#FF2E63", "#08D9D6", "#252A34"],
                deterministic_seed=101,
            ),
            # Candidate B: Curiosity & Mystery
            VariationProfile(
                profile_id="profile_cand_b_curiosity",
                profile_name="Candidate B (Curiosity Emphasis)",
                primary_dimension=VariationDimension.CURIOSITY_EMPHASIS,
                secondary_dimension=VariationDimension.COLOR_EMPHASIS,
                subject_scale_multiplier=0.9,
                font_color_hex="#FFD700",
                pill_fill_hex="#1A1A2E",
                background_style_direction="dark mysterious atmospheric background with deep shadow contrast",
                dominant_colors_override=["#1A1A2E", "#16213E", "#0F3460"],
                deterministic_seed=102,
            ),
            # Candidate C: Bold Typography Power
            VariationProfile(
                profile_id="profile_cand_c_typography",
                profile_name="Candidate C (Typography Emphasis)",
                primary_dimension=VariationDimension.TYPOGRAPHY_EMPHASIS,
                typography_scale_multiplier=1.3,
                stroke_width_multiplier=1.5,
                font_color_hex="#FFFFFF",
                pill_fill_hex="#FF0055",
                background_style_direction="minimalist clean contrast background for maximum text legibility",
                dominant_colors_override=["#FF0055", "#FFFFFF", "#000000"],
                deterministic_seed=103,
            ),
            # Candidate D: High-Saturation Color Contrast
            VariationProfile(
                profile_id="profile_cand_d_color",
                profile_name="Candidate D (Color Emphasis)",
                primary_dimension=VariationDimension.COLOR_EMPHASIS,
                font_color_hex="#00F5D4",
                stroke_color_hex="#7B2CBF",
                background_style_direction="electric hyper-saturated neon gradient background",
                dominant_colors_override=["#7952B3", "#FFC107", "#17A2B8"],
                deterministic_seed=104,
            ),
            # Candidate E: Dynamic Asymmetrical Layout
            VariationProfile(
                profile_id="profile_cand_e_composition",
                profile_name="Candidate E (Composition Emphasis)",
                primary_dimension=VariationDimension.COMPOSITION_EMPHASIS,
                subject_x_offset_pct=0.08,
                subject_scale_multiplier=1.05,
                background_style_direction="dynamic diagonal light rays studio backdrop",
                dominant_colors_override=["#00B4D8", "#90E0EF", "#03045E"],
                deterministic_seed=105,
            ),
        ]

        if count > len(profiles):
            cand_letters = ["F", "G", "H", "I", "J", "K", "L", "M", "N"]
            dims = list(VariationDimension)
            for i in range(len(profiles), count):
                letter = cand_letters[i - 5] if (i - 5) < len(cand_letters) else f"Extra_{i+1}"
                dim = dims[i % len(dims)]
                profiles.append(
                    VariationProfile(
                        profile_id=f"profile_cand_{letter.lower()}",
                        profile_name=f"Candidate {letter} ({dim.value.replace('_', ' ').title()})",
                        primary_dimension=dim,
                        subject_scale_multiplier=1.0 + (i * 0.03),
                        typography_scale_multiplier=1.0 + (i * 0.05),
                        deterministic_seed=100 + i + 1,
                    )
                )

        return profiles[:count]

    def apply_profile_to_package(
        self,
        base_package: RenderExecutionPackage,
        profile: VariationProfile,
        candidate_id: str,
    ) -> RenderExecutionPackage:
        """Apply a VariationProfile deterministically to transform a base RenderExecutionPackage."""
        canvas_w = base_package.scene_graph.canvas_width_px
        canvas_h = base_package.scene_graph.canvas_height_px

        # 1. Transform Typography Instructions
        new_typo: List[RenderTypographyInstruction] = []
        for typo in base_package.typography_instructions:
            new_size = int(round(typo.font_size_px * profile.typography_scale_multiplier))
            new_stroke_w = int(round(typo.stroke_width_px * profile.stroke_width_multiplier))
            font_col = profile.font_color_hex or typo.font_color_hex
            stroke_col = profile.stroke_color_hex or typo.stroke_color_hex

            new_typo.append(
                typo.model_copy(
                    update={
                        "font_size_px": new_size,
                        "stroke_width_px": new_stroke_w,
                        "font_color_hex": font_col,
                        "stroke_color_hex": stroke_col,
                    }
                )
            )

        # 2. Transform Lighting Instructions
        new_lighting: List[RenderLightingInstruction] = []
        for light in base_package.lighting_instructions:
            new_intensity = min(1.0, max(0.0, light.key_light_intensity * profile.key_light_intensity_multiplier))
            rim_flag = profile.rim_light_enabled_override if profile.rim_light_enabled_override is not None else light.rim_light_enabled
            new_lighting.append(
                light.model_copy(
                    update={
                        "key_light_intensity": new_intensity,
                        "rim_light_enabled": rim_flag,
                    }
                )
            )

        # 3. Transform Background Instruction
        bg_style = profile.background_style_direction or base_package.background_instruction.style_prompt_direction
        bg_colors = profile.dominant_colors_override or base_package.background_instruction.dominant_colors
        new_bg = base_package.background_instruction.model_copy(
            update={
                "style_prompt_direction": bg_style,
                "dominant_colors": bg_colors,
            }
        )

        # 4. Transform Placements
        new_placements: List[RenderPlacementCoordinate] = []
        for p in base_package.placement_coordinates:
            is_subject = (
                "subject" in p.element_name.lower()
                or "hero" in p.element_id.lower()
                or "subject" in p.element_id.lower()
                or "person" in p.element_name.lower()
            )
            if is_subject:
                new_scale = p.scale * profile.subject_scale_multiplier
                dx = int(round(profile.subject_x_offset_pct * canvas_w))
                dy = int(round(profile.subject_y_offset_pct * canvas_h))

                old_box = p.bbox_pixels
                new_w = max(1, min(canvas_w, int(round(old_box.width_px * profile.subject_scale_multiplier))))
                new_h = max(1, min(canvas_h, int(round(old_box.height_px * profile.subject_scale_multiplier))))
                new_x = max(0, min(canvas_w - new_w, old_box.x_px + dx))
                new_y = max(0, min(canvas_h - new_h, old_box.y_px + dy))

                new_box = PixelBoundingBox(
                    x_px=new_x,
                    y_px=new_y,
                    width_px=new_w,
                    height_px=new_h,
                )
                new_p = p.model_copy(update={"scale": new_scale, "bbox_pixels": new_box})
                new_placements.append(new_p)
            else:
                new_placements.append(p)

        return base_package.model_copy(
            update={
                "typography_instructions": new_typo,
                "lighting_instructions": new_lighting,
                "background_instruction": new_bg,
                "placement_coordinates": new_placements,
            }
        )

    def generate_candidates(
        self,
        base_package: RenderExecutionPackage,
        count: int = 5,
        custom_profiles: Optional[List[VariationProfile]] = None,
        output_directory: Optional[Union[str, Path]] = None,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> CandidateSet:
        """Generate multiple strategically distinct thumbnail candidates from a base RenderExecutionPackage.

        Args:
            base_package: The input RenderExecutionPackage.
            count: Number of candidates to generate (default 5 for Candidates A-E).
            custom_profiles: Optional list of custom VariationProfile objects.
            output_directory: Target directory to save rendered candidate thumbnail files.
            context_overrides: Additional runtime execution metadata overrides.

        Returns:
            CandidateSet containing CandidateResult objects for Candidates A through E.
        """
        if not base_package:
            raise MultiCandidateGeneratorError("Input base_package cannot be None.")

        profiles = custom_profiles or self.create_default_profiles(count=count)
        set_id = f"candset_{uuid.uuid4().hex[:8]}"

        # Setup output directory
        if output_directory is not None:
            out_dir = Path(output_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="cand_gen_"))

        cand_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        candidate_results: List[CandidateResult] = []
        strategy_summary: Dict[str, str] = {}
        latencies: Dict[str, float] = {}

        logger.info(f"=== Starting MultiCandidateGenerator for set '{set_id}' ({len(profiles)} candidates) ===")

        for i, profile in enumerate(profiles):
            letter = cand_letters[i] if i < len(cand_letters) else f"{i+1}"
            cand_id = f"candidate_{letter.lower()}"
            cand_label = f"Candidate {letter} ({profile.primary_dimension.value.replace('_', ' ').title()})"

            variant_pkg = self.apply_profile_to_package(base_package, profile, cand_id)
            out_file = str(out_dir / f"thumbnail_{cand_id}.png")

            t0 = time.time()
            report = self.pipeline.render_package(variant_pkg, output_path=out_file, context_overrides=context_overrides)
            latency = time.time() - t0

            cand_result = CandidateResult(
                candidate_id=cand_id,
                candidate_label=cand_label,
                profile=profile,
                report=report,
                image_path=report.output_image_path or out_file,
                package=variant_pkg,
            )
            candidate_results.append(cand_result)

            strategy_summary[cand_id] = f"{profile.profile_name} (Focus: {profile.primary_dimension.value})"
            latencies[cand_id] = latency

        metadata = CandidateMetadata(
            set_id=set_id,
            generated_at=_utc_now_iso(),
            total_requested=count,
            total_generated=len(candidate_results),
            variation_dimensions=[p.primary_dimension.value for p in profiles],
            strategy_summary=strategy_summary,
            execution_latencies_s=latencies,
        )

        cand_set = CandidateSet(
            set_id=set_id,
            candidates=candidate_results,
            metadata=metadata,
        )

        logger.info(f"=== Completed MultiCandidateGenerator for set '{set_id}' ({len(candidate_results)} candidates produced) ===")
        return cand_set

    def generate_from_brief(
        self,
        brief: DesignBrief,
        count: int = 5,
        output_directory: Optional[Union[str, Path]] = None,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> CandidateSet:
        """Convenience method: Generate candidate set directly from a DesignBrief."""
        plan = ExecutionPlanner().plan(brief)
        comp = SpatialCompositionPlanner().plan(plan, brief)
        package = self.adapter.translate(comp, plan)
        return self.generate_candidates(package, count=count, output_directory=output_directory, context_overrides=context_overrides)

    def generate_from_composition(
        self,
        composition: SpatialComposition,
        plan: ExecutionPlan,
        count: int = 5,
        output_directory: Optional[Union[str, Path]] = None,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> CandidateSet:
        """Convenience method: Generate candidate set from SpatialComposition + ExecutionPlan."""
        package = self.adapter.translate(composition, plan)
        return self.generate_candidates(package, count=count, output_directory=output_directory, context_overrides=context_overrides)
