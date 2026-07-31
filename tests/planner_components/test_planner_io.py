"""
test_planner_io.py
==================

Unit tests for io.py in planner_components (Phase 4).
"""

from pathlib import Path
import pytest

from models import (
    AssetPlacement,
    CanvasTransform,
    CompositionLayer,
    CompositionWorkspace,
    FaceStrategy,
    BackgroundStrategy,
    GenerationParameters,
    GenerationPlan,
    HeadlineSource,
    LayerDecision,
    LayerRole,
    LayerTransform,
    LightingAdjustment,
    PlacementConstraints,
    PromptPackage,
    TextPlacement,
    WorkspaceMetadata,
    WorkspaceStatistics,
)
from planner_components.io import (
    PlanCache,
    load_cached_generation_plan,
    load_planner_input_bundle,
    save_generation_plan,
)
from thumbnail_planner_exceptions import UpstreamArtifactMissingError

VALID_HASH_1 = "a" * 64
VALID_HASH_2 = "b" * 64


@pytest.fixture
def sample_prompt_package():
    from models import ModelSettings, QualityParameters
    return PromptPackage(
        video_id="vid123",
        positive_prompt="A sharp thumbnail",
        negative_prompt="blurry",
        subject_instructions="Focus on creator",
        background_instructions="Clean studio background",
        lighting_instructions="Warm studio lighting",
        color_instructions="Vibrant colors",
        typography_instructions="Bold text",
        composition_instructions="Centered",
        generation_parameters=GenerationParameters(
            width=1280,
            height=720,
            aspect_ratio="16:9",
            seed=12345,
        ),
        quality_parameters=QualityParameters(),
        model_settings=ModelSettings(),
        compiled_at="2026-08-01T00:00:00Z",
        generated_at="2026-08-01T00:00:00Z",
    )


@pytest.fixture
def sample_workspace():
    l_bg = CompositionLayer(
        layer_id="l_bg",
        placement=AssetPlacement(
            asset_id="bg",
            role=LayerRole.BACKGROUND,
            decision=LayerDecision.REPLACE,
            source_path="bg.png",
            transform=LayerTransform(),
            z_index=0,
        ),
    )
    return CompositionWorkspace(
        video_id="vid123",
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        layers=[l_bg],
        groups=[],
        text_placement=TextPlacement(),
        lighting=LightingAdjustment(
            target_brightness=0.5, target_contrast=0.5, target_saturation=0.5, warm_or_cool="neutral"
        ),
        constraints=PlacementConstraints(),
        statistics=WorkspaceStatistics(total_layers=1, replaced=1),
        metadata=WorkspaceMetadata(
            video_id="vid123",
            created_at="2026-08-01T00:00:00Z",
            vre_source_hash=VALID_HASH_1,
            redesign_spec_hash=VALID_HASH_1,
            prompt_package_hash=VALID_HASH_1,
            engine_version="1.0.0",
        ),
    )


def test_load_planner_input_bundle_missing_raises(tmp_path: Path):
    with pytest.raises(UpstreamArtifactMissingError):
        load_planner_input_bundle("missing_vid", prompt_package_dir=tmp_path)


def test_load_planner_input_bundle_success(tmp_path: Path, sample_prompt_package, sample_workspace):
    pkg_dir = tmp_path / "prompt_packages"
    pkg_dir.mkdir(parents=True)
    pkg_file = pkg_dir / "vid123.json"
    pkg_file.write_text(sample_prompt_package.model_dump_json(), encoding="utf-8")

    ws_dir = tmp_path / "composition_workspaces" / "vid123"
    ws_dir.mkdir(parents=True)
    ws_file = ws_dir / "workspace_manifest.json"
    ws_file.write_text(sample_workspace.model_dump_json(), encoding="utf-8")

    bundle = load_planner_input_bundle(
        "vid123",
        prompt_package_dir=pkg_dir,
        workspace_dir=tmp_path / "composition_workspaces",
    )
    assert bundle.video_id == "vid123"
    assert bundle.prompt_package.video_id == "vid123"
    assert bundle.workspace.video_id == "vid123"
    assert bundle.intelligence is None


def test_save_and_load_generation_plan(tmp_path: Path):
    plan = GenerationPlan(
        video_id="vid123",
        headline="TOP SECRET",
        headline_source=HeadlineSource.PRESERVED_OCR,
        face_strategy=FaceStrategy.PRESERVE_AS_IS,
        background_strategy=BackgroundStrategy.STRUCTURE_GUIDED_REPLACE,
        composition_strategy="centered",
        camera_distance="medium",
        lighting="neutral",
        prompt_package_hash=VALID_HASH_1,
        workspace_hash=VALID_HASH_2,
        engine_version="1.0.0",
        generated_at="2026-08-01T00:00:00Z",
    )

    plan_dir = tmp_path / "generation_plans"
    saved_path = save_generation_plan(plan, plan_dir=plan_dir)
    assert saved_path.exists()

    loaded = load_cached_generation_plan("vid123", plan_dir=plan_dir)
    assert loaded is not None
    assert loaded.video_id == "vid123"
    assert loaded.headline == "TOP SECRET"


def test_plan_cache_class(tmp_path: Path):
    cache = PlanCache(plan_dir=tmp_path)
    assert cache.load("vid123") is None

    plan = GenerationPlan(
        video_id="vid123",
        headline="CACHED",
        headline_source=HeadlineSource.NONE,
        face_strategy=FaceStrategy.NONE,
        background_strategy=BackgroundStrategy.UNGUIDED_REPLACE,
        composition_strategy="rule_of_thirds",
        camera_distance="close",
        lighting="warm",
        prompt_package_hash=VALID_HASH_1,
        workspace_hash=VALID_HASH_2,
        engine_version="1.0.0",
        generated_at="2026-08-01T00:00:00Z",
    )
    cache.save(plan)
    loaded = cache.load("vid123")
    assert loaded is not None
    assert loaded.headline == "CACHED"
