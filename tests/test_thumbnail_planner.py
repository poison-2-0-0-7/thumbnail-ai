"""
test_thumbnail_planner.py
==========================

Integration and determinism tests for ThumbnailPlanner orchestrator (Module 10.5).
"""

from pathlib import Path
import pytest

from models import (
    AssetPlacement,
    CanvasTransform,
    CompositionLayer,
    CompositionWorkspace,
    GenerationParameters,
    GenerationPlan,
    LayerDecision,
    LayerRole,
    LayerTransform,
    LightingAdjustment,
    ModelSettings,
    PlacementConstraints,
    PromptPackage,
    QualityParameters,
    TextPlacement,
    WorkspaceMetadata,
    WorkspaceStatistics,
)
from thumbnail_planner import ThumbnailPlanner, build_generation_plan

VALID_HASH_1 = "a" * 64
VALID_HASH_2 = "b" * 64


@pytest.fixture
def test_workspace(tmp_path: Path):
    f_bg = tmp_path / "bg.png"
    f_bg.write_bytes(b"bg")
    f_person = tmp_path / "person.png"
    f_person.write_bytes(b"person")

    l_bg = CompositionLayer(
        layer_id="l_bg",
        placement=AssetPlacement(
            asset_id="bg",
            role=LayerRole.BACKGROUND,
            decision=LayerDecision.REPLACE,
            source_path=str(f_bg),
            transform=LayerTransform(),
            z_index=0,
        ),
    )
    l_person = CompositionLayer(
        layer_id="l_person",
        placement=AssetPlacement(
            asset_id="person",
            role=LayerRole.PERSON,
            decision=LayerDecision.KEEP,
            source_path=str(f_person),
            transform=LayerTransform(),
            z_index=10,
        ),
    )
    return CompositionWorkspace(
        video_id="vid_test_1",
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        layers=[l_bg, l_person],
        groups=[],
        text_placement=TextPlacement(),
        lighting=LightingAdjustment(
            target_brightness=0.5, target_contrast=0.5, target_saturation=0.5, warm_or_cool="neutral"
        ),
        constraints=PlacementConstraints(),
        statistics=WorkspaceStatistics(total_layers=2, replaced=1, kept=1),
        metadata=WorkspaceMetadata(
            video_id="vid_test_1",
            created_at="2026-08-01T00:00:00Z",
            vre_source_hash=VALID_HASH_1,
            redesign_spec_hash=VALID_HASH_1,
            prompt_package_hash=VALID_HASH_1,
            engine_version="1.0.0",
        ),
    )


@pytest.fixture
def test_prompt_package():
    return PromptPackage(
        video_id="vid_test_1",
        positive_prompt="High quality thumbnail",
        negative_prompt="blurry",
        subject_instructions="Focus on creator",
        background_instructions="Studio background",
        lighting_instructions="Warm",
        color_instructions="Vibrant",
        typography_instructions="Bold",
        composition_instructions="Centered",
        generation_parameters=GenerationParameters(
            width=1280,
            height=720,
            aspect_ratio="16:9",
            seed=42,
        ),
        quality_parameters=QualityParameters(),
        model_settings=ModelSettings(),
        compiled_at="2026-08-01T00:00:00Z",
        generated_at="2026-08-01T00:00:00Z",
    )


def test_thumbnail_planner_end_to_end(tmp_path: Path, test_workspace, test_prompt_package, monkeypatch):
    pkg_dir = tmp_path / "prompt_packages"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "vid_test_1.json").write_text(test_prompt_package.model_dump_json(), encoding="utf-8")

    ws_dir = tmp_path / "composition_workspaces" / "vid_test_1"
    ws_dir.mkdir(parents=True)
    (ws_dir / "workspace_manifest.json").write_text(test_workspace.model_dump_json(), encoding="utf-8")

    monkeypatch.setattr("planner_components.io.DEFAULT_PROMPT_PACKAGE_DIR", pkg_dir)
    monkeypatch.setattr("planner_components.io.COMPOSITION_WORKSPACE_ROOT", tmp_path / "composition_workspaces")

    planner = ThumbnailPlanner(plan_dir=tmp_path / "generation_plans")
    plan = planner.plan("vid_test_1")

    assert isinstance(plan, GenerationPlan)
    assert plan.video_id == "vid_test_1"
    assert plan.status == "partial"  # Module 8/9 absent -> partial degradation
    assert len(plan.partial_failure_reasons) == 2


def test_thumbnail_planner_determinism(tmp_path: Path, test_workspace, test_prompt_package, monkeypatch):
    pkg_dir = tmp_path / "prompt_packages"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "vid_test_1.json").write_text(test_prompt_package.model_dump_json(), encoding="utf-8")

    ws_dir = tmp_path / "composition_workspaces" / "vid_test_1"
    ws_dir.mkdir(parents=True)
    (ws_dir / "workspace_manifest.json").write_text(test_workspace.model_dump_json(), encoding="utf-8")

    monkeypatch.setattr("planner_components.io.DEFAULT_PROMPT_PACKAGE_DIR", pkg_dir)
    monkeypatch.setattr("planner_components.io.COMPOSITION_WORKSPACE_ROOT", tmp_path / "composition_workspaces")

    plan1 = build_generation_plan("vid_test_1", force_recompute=True, plan_dir=tmp_path / "plans")
    plan2 = build_generation_plan("vid_test_1", force_recompute=True, plan_dir=tmp_path / "plans")

    json1 = plan1.model_dump_json(exclude={"generated_at"})
    json2 = plan2.model_dump_json(exclude={"generated_at"})

    assert json1 == json2
