"""Integration tests for Renderer V2 Phase 2 Edit Planner."""

import json
from pathlib import Path
import numpy as np
import pytest
from renderer_v2.phase1.schemas import Instance, SceneGraph
from renderer_v2.planning.planner import EditPlanner
from renderer_v2.planning.planner_types import EditAction, EditPlanOutput


def test_planner_integration_end_to_end(tmp_path: Path):
    """Verify end-to-end plan generation from SceneGraph and verify output JSON."""
    planner = EditPlanner()
    h, w = 720, 1280

    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :, :] = 40  # Dark background
    image[150:550, 750:1150, 0] = 240  # Subject
    image[150:550, 750:1150, 1] = 190
    image[150:550, 750:1150, 2] = 160

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[150:550, 750:1150] = 255
    alpha = mask.astype(np.float32) / 255.0

    inst_creator = Instance(
        instance_id="creator_0",
        cls="creator",
        mask=mask,
        alpha_matte=alpha,
        bbox=(750, 150, 1150, 550),
        depth_layer=0.10,
        locked=True,
    )
    inst_logo = Instance(
        instance_id="logo_0",
        cls="logo",
        mask=np.zeros((h, w), dtype=np.uint8),
        alpha_matte=np.zeros((h, w), dtype=np.float32),
        bbox=(50, 50, 200, 150),
        depth_layer=0.05,
        locked=True,
    )
    depth = np.full((h, w), 0.85, dtype=np.float32)
    depth[150:550, 750:1150] = 0.10

    sg = SceneGraph(
        source_image=image,
        instances=[inst_creator, inst_logo],
        depth_map=depth,
        width=w,
        height=h,
    )

    plan = planner.plan(
        scene_graph=sg,
        metadata={"video_id": "test_video_1001", "archetype": "single_creator_face"},
    )

    assert isinstance(plan, EditPlanOutput)
    assert plan.composition_score >= 0.0
    assert len(plan.changes) >= 2
    assert "creator_0" in plan.locked_instances

    # Save to file
    out_file = tmp_path / "test_edit_plan.json"
    saved_path = planner.save_plan(plan, out_file)
    assert saved_path.exists()

    # Read back and validate JSON structure
    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "summary" in data
    assert "composition_score" in data
    assert "changes" in data
    assert isinstance(data["changes"], list)
    for ch in data["changes"]:
        assert "target" in ch
        assert "action" in ch
        assert "reason" in ch
