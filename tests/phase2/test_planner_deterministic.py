"""Tests asserting strict determinism of the Edit Planner across identical inputs."""

import numpy as np
import pytest
from renderer_v2.phase1.schemas import Instance, SceneGraph
from renderer_v2.planning.planner import EditPlanner


def test_planner_strict_determinism():
    """Verify that given identical inputs, EditPlanner produces 100% identical outputs."""
    planner = EditPlanner()
    h, w = 720, 1280

    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[200:600, 700:1100, 0] = 220
    image[200:600, 700:1100, 1] = 180
    image[200:600, 700:1100, 2] = 160

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[200:600, 700:1100] = 255
    alpha = mask.astype(np.float32) / 255.0

    inst = Instance(
        instance_id="creator_0",
        cls="creator",
        mask=mask,
        alpha_matte=alpha,
        bbox=(700, 200, 1100, 600),
        depth_layer=0.15,
        locked=True,
    )
    depth = np.full((h, w), 0.8, dtype=np.float32)
    depth[200:600, 700:1100] = 0.15

    sg = SceneGraph(
        source_image=image,
        instances=[inst],
        depth_map=depth,
        width=w,
        height=h,
    )
    meta = {"video_id": "test_video_deterministic_001", "archetype": "single_creator_face"}

    # Run planning twice on identical input
    plan_1 = planner.plan(sg, metadata=meta)
    plan_2 = planner.plan(sg, metadata=meta)

    # Check exact score matching
    assert plan_1.composition_score == plan_2.composition_score
    assert plan_1.target_composition_score == plan_2.target_composition_score
    assert plan_1.scoring_breakdown.to_dict() == plan_2.scoring_breakdown.to_dict()

    # Check exact changes matching
    assert len(plan_1.changes) == len(plan_2.changes)
    for c1, c2 in zip(plan_1.changes, plan_2.changes):
        assert c1.target == c2.target
        assert c1.action == c2.action
        assert c1.reason == c2.reason
        assert c1.parameters == c2.parameters

    # Check exact JSON string matching
    json_1 = plan_1.to_json(indent=2)
    json_2 = plan_2.to_json(indent=2)
    assert json_1 == json_2
