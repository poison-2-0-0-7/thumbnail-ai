"""Unit tests for objective 0-100 scoring across all 10 visual dimensions."""

import pytest
from renderer_v2.planning.planner_types import CompositionAnalysis, ScoreBreakdown
from renderer_v2.planning.scoring import ScoringEngine


def test_scoring_all_dimensions_in_bounds():
    """Verify all 10 dimension scores and composite score are in [0, 100]."""
    analysis = CompositionAnalysis(
        subject_scale=0.35,
        subject_position=(0.67, 0.50),
        rule_of_thirds_alignment=0.90,
        negative_space_ratio=0.50,
        text_safe_zone_available=True,
        text_safe_zones=[(40, 40, 500, 300)],
        hierarchy_clarity=0.85,
        contrast_ratio=5.5,
        visual_balance=0.88,
        focus_score=0.92,
        attention_direction="left_to_right",
        color_harmony="complementary",
        ctr_improvement_potential=0.15,
    )

    scores = ScoringEngine.calculate_scores(
        analysis=analysis,
        visual_clutter_score=0.15,
        depth_variance=0.70,
        locked_identity_intact=True,
    )

    assert 0.0 <= scores.composition <= 100.0
    assert 0.0 <= scores.contrast <= 100.0
    assert 0.0 <= scores.subject_prominence <= 100.0
    assert 0.0 <= scores.readability <= 100.0
    assert 0.0 <= scores.visual_clutter <= 100.0
    assert 0.0 <= scores.background_quality <= 100.0
    assert scores.identity_preservation == 100.0
    assert 0.0 <= scores.text_placement <= 100.0
    assert 0.0 <= scores.depth_usage <= 100.0
    assert 0.0 <= scores.focus_hierarchy <= 100.0
    assert 0.0 <= scores.overall <= 100.0


def test_scoring_penalizes_low_contrast_and_high_clutter():
    """Verify scoring reflects penalties for degraded scenes."""
    bad_analysis = CompositionAnalysis(
        subject_scale=0.08,  # Too small
        subject_position=(0.05, 0.05),  # Bad position
        rule_of_thirds_alignment=0.20,
        negative_space_ratio=0.10,
        text_safe_zone_available=False,
        text_safe_zones=[],
        hierarchy_clarity=0.20,
        contrast_ratio=1.2,  # Low contrast
        visual_balance=0.30,
        focus_score=0.25,
        attention_direction="center_outward",
        color_harmony="monochromatic",
        ctr_improvement_potential=0.85,
    )

    scores = ScoringEngine.calculate_scores(
        analysis=bad_analysis,
        visual_clutter_score=0.90,  # Highly cluttered
        depth_variance=0.10,
        locked_identity_intact=False,
    )

    assert scores.contrast < 20.0
    assert scores.subject_prominence < 40.0
    assert scores.readability < 40.0
    assert scores.overall < 50.0
