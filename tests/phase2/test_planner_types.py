"""Unit tests for Phase 2 Edit Planner data types, schemas, and serialization."""

import json
import pytest
from renderer_v2.planning.planner_types import (
    EditAction,
    TargetCategory,
    ObjectEditChange,
    ScoreBreakdown,
    CompositionAnalysis,
    CompositionDirectives,
    EditPlanOutput,
)


def test_edit_action_enum_values():
    """Verify all required edit action enum values exist."""
    required_actions = {
        "keep", "remove", "move", "resize", "recolor", "relight",
        "replace", "regenerate", "blur", "desaturate", "enhance"
    }
    actual_actions = {a.value for a in EditAction}
    assert required_actions.issubset(actual_actions)


def test_object_edit_change_validation():
    """Verify ObjectEditChange model validation and immutability."""
    change = ObjectEditChange(
        target="creator_face",
        action=EditAction.KEEP,
        reason="identity preservation as locked raster layer",
        target_category=TargetCategory.CREATOR_FACE,
        parameters={"locked": True},
        confidence=1.0,
        priority=1,
    )
    assert change.target == "creator_face"
    assert change.action == EditAction.KEEP
    assert change.confidence == 1.0

    # Verify frozen model immutability
    with pytest.raises(Exception):
        change.target = "other"  # type: ignore


def test_score_breakdown_bounds():
    """Verify ScoreBreakdown validates 0-100 bounds."""
    scores = ScoreBreakdown(
        composition=85.5,
        contrast=78.0,
        subject_prominence=90.0,
        readability=82.5,
        visual_clutter=75.0,
        background_quality=80.0,
        identity_preservation=100.0,
        text_placement=88.0,
        depth_usage=70.0,
        focus_hierarchy=85.0,
        overall=83.5,
    )
    assert scores.composition == 85.5
    assert scores.identity_preservation == 100.0
    assert 0 <= scores.overall <= 100


def test_edit_plan_output_json_roundtrip():
    """Verify EditPlanOutput can be serialized to JSON and deserialized identically."""
    scores = ScoreBreakdown(
        composition=80.0,
        contrast=75.0,
        subject_prominence=85.0,
        readability=70.0,
        visual_clutter=65.0,
        background_quality=70.0,
        identity_preservation=100.0,
        text_placement=80.0,
        depth_usage=60.0,
        focus_hierarchy=80.0,
        overall=77.0,
    )
    analysis = CompositionAnalysis(
        subject_scale=0.35,
        subject_position=(0.67, 0.50),
        rule_of_thirds_alignment=0.88,
        negative_space_ratio=0.45,
        text_safe_zone_available=True,
        text_safe_zones=[(50, 50, 600, 300)],
        hierarchy_clarity=0.80,
        contrast_ratio=4.8,
        visual_balance=0.85,
        focus_score=0.90,
        attention_direction="left_to_right",
        color_harmony="complementary",
        ctr_improvement_potential=0.25,
    )
    directives = CompositionDirectives(
        target_subject_scale=0.35,
        target_subject_position=(0.67, 0.50),
        rule_of_thirds_target=(0.67, 0.50),
        recommended_text_zone=(50, 50, 600, 300),
        depth_layering_order=["background", "creator_0", "typography"],
        lighting_direction="top_left",
        color_palette_target=["#FF2E63", "#08D9D6", "#0F172A", "#FFFFFF"],
        contrast_boost_factor=1.15,
    )
    changes = [
        ObjectEditChange(
            target="creator_face",
            action=EditAction.KEEP,
            reason="identity preservation as locked raster layer",
            target_category=TargetCategory.CREATOR_FACE,
            parameters={"locked": True},
        ),
        ObjectEditChange(
            target="background",
            action=EditAction.REPLACE,
            reason="replace cluttered background with high-contrast studio backdrop",
            target_category=TargetCategory.BACKGROUND,
            parameters={"depth_style": "shallow_dof"},
        ),
    ]

    plan = EditPlanOutput(
        summary="Strategic optimization plan",
        composition_score=77.0,
        target_composition_score=92.5,
        changes=changes,
        scoring_breakdown=scores,
        composition_analysis=analysis,
        composition_directives=directives,
        locked_instances=["creator_0"],
        quality_targets={"min_identity_similarity": 0.90},
        metadata={"video_id": "test_vid_123"},
    )

    json_str = plan.to_json(indent=2)
    assert "creator_face" in json_str
    assert "replace" in json_str

    parsed = EditPlanOutput.from_json(json_str)
    assert parsed.composition_score == 77.0
    assert len(parsed.changes) == 2
    assert parsed.changes[0].action == EditAction.KEEP
    assert parsed.changes[1].action == EditAction.REPLACE
    assert parsed.to_dict() == plan.to_dict()
