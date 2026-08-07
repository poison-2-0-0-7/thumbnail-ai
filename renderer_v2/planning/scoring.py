"""Objective scoring engine for thumbnail visual dimensions (0-100)."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from .planner_types import CompositionAnalysis, ScoreBreakdown


class ScoringEngine:
    """Calculates objective, deterministic 0-100 scores across all thumbnail dimensions."""

    @staticmethod
    def calculate_scores(
        analysis: CompositionAnalysis,
        visual_clutter_score: float,
        depth_variance: float = 0.5,
        locked_identity_intact: bool = True,
    ) -> ScoreBreakdown:
        """Calculate exact 0-100 scores for every dimension.

        Args:
            analysis: Quantitative CompositionAnalysis metrics.
            visual_clutter_score: Visual clutter measure in [0.0, 1.0] (0 = clean, 1 = noisy).
            depth_variance: Depth map foreground/background separation in [0.0, 1.0].
            locked_identity_intact: True if creator face/identity is fully preserved.

        Returns:
            ScoreBreakdown containing all 10 dimension scores (0-100) and overall score.
        """
        # 1. Composition (0-100): Rule of thirds (40%) + visual balance (35%) + subject scale optimality (25%)
        # Optimal scale is 0.28 to 0.45
        if 0.28 <= analysis.subject_scale <= 0.45:
            scale_optimality = 1.0
        elif analysis.subject_scale < 0.28:
            scale_optimality = max(0.0, analysis.subject_scale / 0.28)
        else:
            scale_optimality = max(0.0, 1.0 - (analysis.subject_scale - 0.45) * 1.5)

        comp_raw = (
            0.40 * analysis.rule_of_thirds_alignment
            + 0.35 * analysis.visual_balance
            + 0.25 * scale_optimality
        )
        comp_score = float(round(np.clip(comp_raw * 100.0, 0.0, 100.0), 2))

        # 2. Contrast (0-100): Contrast ratio >= 4.5 is WCAG AA standard (100%), 1.0 is 0%
        # contrast_ratio is typically 1.0 to 10.0+
        contrast_norm = min(1.0, (analysis.contrast_ratio - 1.0) / 4.0)
        contrast_score = float(round(np.clip(contrast_norm * 100.0, 0.0, 100.0), 2))

        # 3. Subject Prominence (0-100): Subject scale (50%) + Focus score (50%)
        prominence_raw = 0.50 * scale_optimality + 0.50 * analysis.focus_score
        prominence_score = float(round(np.clip(prominence_raw * 100.0, 0.0, 100.0), 2))

        # 4. Readability (0-100): Text safe zone presence (60%) + negative space (40%)
        safe_zone_factor = 1.0 if analysis.text_safe_zone_available else 0.3
        readability_raw = 0.60 * safe_zone_factor + 0.40 * analysis.negative_space_ratio
        readability_score = float(round(np.clip(readability_raw * 100.0, 0.0, 100.0), 2))

        # 5. Visual Clutter (0-100): 100 = perfectly clean / minimal clutter
        cleanliness = 1.0 - visual_clutter_score
        clutter_score = float(round(np.clip(cleanliness * 100.0, 0.0, 100.0), 2))

        # 6. Background Quality (0-100): Low clutter in bg (60%) + depth separation (40%)
        bg_quality_raw = 0.60 * cleanliness + 0.40 * depth_variance
        bg_quality_score = float(round(np.clip(bg_quality_raw * 100.0, 0.0, 100.0), 2))

        # 7. Identity Preservation (0-100): 100 if locked raster layer is untouched
        identity_score = 100.0 if locked_identity_intact else 0.0

        # 8. Text Placement (0-100): Quality and size of available text safe zones
        if analysis.text_safe_zones:
            best_zone = analysis.text_safe_zones[0]
            zone_w = best_zone[2] - best_zone[0]
            zone_h = best_zone[3] - best_zone[1]
            zone_area_fraction = (zone_w * zone_h) / (1280.0 * 720.0)
            text_placement_norm = min(1.0, zone_area_fraction / 0.18)
        else:
            text_placement_norm = 0.25
        text_placement_score = float(round(np.clip(text_placement_norm * 100.0, 0.0, 100.0), 2))

        # 9. Depth Usage (0-100): Depth variance and foreground/background layering
        depth_score = float(round(np.clip(depth_variance * 100.0, 0.0, 100.0), 2))

        # 10. Focus Hierarchy (0-100): Distinction between primary focal point and background
        focus_score = float(round(np.clip(analysis.hierarchy_clarity * 100.0, 0.0, 100.0), 2))

        # Overall Weighted Score (0-100)
        overall_raw = (
            0.20 * comp_score
            + 0.15 * contrast_score
            + 0.15 * prominence_score
            + 0.10 * readability_score
            + 0.10 * clutter_score
            + 0.10 * bg_quality_score
            + 0.10 * identity_score
            + 0.05 * depth_score
            + 0.05 * focus_score
        )
        overall_score = float(round(np.clip(overall_raw, 0.0, 100.0), 2))

        return ScoreBreakdown(
            composition=comp_score,
            contrast=contrast_score,
            subject_prominence=prominence_score,
            readability=readability_score,
            visual_clutter=clutter_score,
            background_quality=bg_quality_score,
            identity_preservation=identity_score,
            text_placement=text_placement_score,
            depth_usage=depth_score,
            focus_hierarchy=focus_score,
            overall=overall_score,
        )
