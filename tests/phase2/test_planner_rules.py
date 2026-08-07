"""Unit tests for deterministic rule engine decisions."""

import numpy as np
import pytest
from renderer_v2.phase1.schemas import Instance, SceneGraph
from renderer_v2.planning.planner_types import (
    CompositionAnalysis,
    EditAction,
    ScoreBreakdown,
    TargetCategory,
)
from renderer_v2.planning.planner_rules import PlannerRuleEngine


def test_rule_engine_creator_decisions():
    """Verify creator instance generates KEEP, and triggers RESIZE/RELIGHT when appropriate."""
    h, w = 720, 1280
    mask = np.zeros((h, w), dtype=np.uint8)
    # Small creator in corner
    mask[50:150, 50:150] = 255
    alpha = mask.astype(np.float32) / 255.0
    inst = Instance(
        instance_id="creator_0",
        cls="creator",
        mask=mask,
        alpha_matte=alpha,
        bbox=(50, 50, 150, 150),
        depth_layer=0.1,
        locked=True,
    )
    sg = SceneGraph(
        source_image=np.zeros((h, w, 3), dtype=np.uint8),
        instances=[inst],
        depth_map=np.zeros((h, w), dtype=np.float32),
        width=w,
        height=h,
    )
    analysis = CompositionAnalysis(
        subject_scale=0.015,  # Too small
        subject_position=(0.1, 0.1),
        rule_of_thirds_alignment=0.20,
        negative_space_ratio=0.80,
        text_safe_zone_available=True,
        text_safe_zones=[(200, 200, 800, 500)],
        hierarchy_clarity=0.40,
        contrast_ratio=2.5,  # Low contrast
        visual_balance=0.30,
        focus_score=0.40,
        attention_direction="left_to_right",
        color_harmony="complementary",
        ctr_improvement_potential=0.75,
    )
    scores = ScoreBreakdown(
        composition=30.0,
        contrast=35.0,
        subject_prominence=20.0,
        readability=75.0,
        visual_clutter=50.0,
        background_quality=40.0,
        identity_preservation=100.0,
        text_placement=60.0,
        depth_usage=30.0,
        focus_hierarchy=35.0,
        overall=42.0,
    )

    changes = PlannerRuleEngine.evaluate_instance_decisions(sg, analysis, scores)
    actions = {c.action for c in changes if c.target == "creator_0"}

    # Must contain KEEP (identity), RESIZE (scale up), MOVE, and RELIGHT (due to low contrast)
    assert EditAction.KEEP in actions
    assert EditAction.RESIZE in actions
    assert EditAction.RELIGHT in actions
    assert EditAction.ENHANCE in actions

    # Every change must have an auditable reason
    for ch in changes:
        assert isinstance(ch.reason, str) and len(ch.reason) > 5


def test_rule_engine_logo_in_yt_badge_triggers_move():
    """Verify logo located in bottom-right triggers MOVE action."""
    h, w = 720, 1280
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[600:700, 1100:1250] = 255
    inst = Instance(
        instance_id="logo_0",
        cls="logo",
        mask=mask,
        alpha_matte=mask.astype(np.float32) / 255.0,
        bbox=(1100, 600, 1250, 700),  # In bottom-right YouTube badge zone
        depth_layer=0.1,
        locked=True,
    )
    sg = SceneGraph(
        source_image=np.zeros((h, w, 3), dtype=np.uint8),
        instances=[inst],
        depth_map=np.zeros((h, w), dtype=np.float32),
        width=w,
        height=h,
    )
    analysis = CompositionAnalysis(
        subject_scale=0.05,
        subject_position=(0.9, 0.9),
        rule_of_thirds_alignment=0.3,
        negative_space_ratio=0.9,
        text_safe_zone_available=True,
        text_safe_zones=[(50, 50, 500, 300)],
        hierarchy_clarity=0.5,
        contrast_ratio=6.0,
        visual_balance=0.5,
        focus_score=0.5,
        attention_direction="center_outward",
        color_harmony="analogous",
        ctr_improvement_potential=0.3,
    )
    scores = ScoreBreakdown(overall=65.0)

    changes = PlannerRuleEngine.evaluate_instance_decisions(sg, analysis, scores)
    logo_actions = [c.action for c in changes if c.target == "logo_0"]
    assert EditAction.MOVE in logo_actions
