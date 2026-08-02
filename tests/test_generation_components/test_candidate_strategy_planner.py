"""Tests for CandidateStrategyPlanner."""

from __future__ import annotations

import pytest
from models import (
    CandidateStrategy,
    DesignBlueprint,
    GenerationParameters,
    ModelSettings,
    PromptPackage,
    QualityParameters,
    TextPlacement,
)
from generation_components import CandidateStrategyPlanner, StrategyPackLibrary


@pytest.fixture
def base_prompt_package() -> PromptPackage:
    return PromptPackage(
        video_id="test_video_123",
        positive_prompt="Subject in center. Clear background. High contrast.",
        negative_prompt="blurry, low quality",
        subject_instructions="Place primary subject in center.",
        background_instructions="Keep background minimal.",
        typography_instructions="Render text overlay: 'BEST HACKS' on thumbnail.",
        composition_instructions="Anchor focal point in center.",
        lighting_instructions="Use bright balanced lighting.",
        color_instructions="Render vivid colors.",
        object_placement=["phone: include", "camera: preserve"],
        rendering_constraints=["Do not add watermark.", "Preserve key face."],
        safety_constraints=["Do not depict graphic content."],
        generation_parameters=GenerationParameters(seed=42),
        quality_parameters=QualityParameters(),
        model_settings=ModelSettings(),
        duration_seconds=0.1,
        generated_at="2026-08-01T00:00:00Z",
    )


@pytest.fixture
def sample_blueprint() -> DesignBlueprint:
    return DesignBlueprint(
        video_id="test_video_123",
        headline="BEST HACKS",
        headline_score=0.9,
        hook_type="curiosity",
        emotion="excited",
        face_strategy="smile",
        background_strategy="blur",
        text_position=TextPlacement(include_text=True),
        camera_distance="medium",
        lighting="bright",
        duration_seconds=0.1,
        generated_at="2026-08-01T00:00:00Z",
    )


def test_planner_faithful_strategy(base_prompt_package, sample_blueprint):
    planner = CandidateStrategyPlanner()
    strategy = CandidateStrategy.faithful_default()
    derived = planner.derive_package(base_prompt_package, sample_blueprint, strategy, candidate_index=0)

    assert derived.generation_parameters.seed == 42
    assert derived.subject_instructions == base_prompt_package.subject_instructions
    assert derived.background_instructions == base_prompt_package.background_instructions
    assert derived.typography_instructions == base_prompt_package.typography_instructions


def test_planner_candidate_index_seed_increment(base_prompt_package, sample_blueprint):
    planner = CandidateStrategyPlanner()
    strategy = CandidateStrategy.faithful_default()
    derived1 = planner.derive_package(base_prompt_package, sample_blueprint, strategy, candidate_index=1)
    assert derived1.generation_parameters.seed == 43


def test_planner_determinism(base_prompt_package, sample_blueprint):
    planner = CandidateStrategyPlanner()
    strategy = CandidateStrategy(
        name="aggressive_ctr",
        camera_distance_shift=0,
        object_emphasis_bias=0.2,
        color_grade_bias=0.2,
    )
    res1 = planner.derive_package(base_prompt_package, sample_blueprint, strategy, candidate_index=2)
    res2 = planner.derive_package(base_prompt_package, sample_blueprint, strategy, candidate_index=2)

    assert res1.model_dump_json() == res2.model_dump_json()


def test_planner_all_five_strategies(base_prompt_package, sample_blueprint):
    planner = CandidateStrategyPlanner()
    library = StrategyPackLibrary()
    pack = library.load("default_five")

    packages = [
        planner.derive_package(base_prompt_package, sample_blueprint, strat, idx)
        for idx, strat in enumerate(pack.strategies)
    ]

    assert len(packages) == 5
    # Confirm headline & video_id invariant across all candidates
    for pkg in packages:
        assert pkg.video_id == "test_video_123"
        assert "BEST HACKS" in pkg.typography_instructions
        assert pkg.negative_prompt == base_prompt_package.negative_prompt

    # Candidate 2 (cleaner composition) has wider framing instruction
    assert "wider" in packages[2].composition_instructions
    # Candidate 3 (higher contrast) has color perturbation
    assert "increased contrast" in packages[3].color_instructions


def test_planner_lighting_and_framing_bias(base_prompt_package, sample_blueprint):
    planner = CandidateStrategyPlanner()
    strat_pos = CandidateStrategy(
        name="lighting_framing_pos",
        lighting_bias=0.3,
        framing_bias=0.3,
    )
    derived_pos = planner.derive_package(base_prompt_package, sample_blueprint, strat_pos, candidate_index=1)
    assert "dramatic" in derived_pos.lighting_instructions
    assert "Tighten framing" in derived_pos.composition_instructions
    assert derived_pos.typography_instructions == base_prompt_package.typography_instructions

    strat_neg = CandidateStrategy(
        name="lighting_framing_neg",
        lighting_bias=-0.3,
        framing_bias=-0.3,
    )
    derived_neg = planner.derive_package(base_prompt_package, sample_blueprint, strat_neg, candidate_index=2)
    assert "soft, diffused" in derived_neg.lighting_instructions
    assert "Expand framing" in derived_neg.composition_instructions

