"""Tests for WorkflowGraphCache and WorkflowBuilder caching integration."""

from __future__ import annotations

import pytest
from pathlib import Path
from models import GenerationParameters, ModelSettings, PromptPackage, QualityParameters
from generation_components import WorkflowGraphCache
from image_generator import ProfileSelector, WorkflowBuilder
from workflow_library import WorkflowLibrary

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent.parent / "workflows"


@pytest.fixture
def workflow_library():
    return WorkflowLibrary(WORKFLOWS_DIR)


@pytest.fixture
def standard_profile():
    return ProfileSelector().select(available_vram_gb=16.0, requested_profile="PROFILE_STANDARD")


@pytest.fixture
def low_vram_profile():
    return ProfileSelector().select(available_vram_gb=4.0, requested_profile="PROFILE_LOW_VRAM")


@pytest.fixture
def base_prompt_package():
    return PromptPackage(
        video_id="cache_test_vid",
        positive_prompt="Thumbnail subject",
        negative_prompt="blurry",
        subject_instructions="Center subject",
        background_instructions="Clean background",
        typography_instructions="Render text",
        composition_instructions="Rule of thirds",
        lighting_instructions="Bright",
        color_instructions="Vivid",
        generation_parameters=GenerationParameters(seed=100),
        quality_parameters=QualityParameters(),
        model_settings=ModelSettings(),
        duration_seconds=0.1,
        generated_at="2026-08-01T00:00:00Z",
    )


def test_workflow_graph_cache_basic():
    cache = WorkflowGraphCache(enabled=True)
    assert len(cache) == 0

    key = cache.make_key("/path/to/wf.json", "v1", "PROFILE_STANDARD", "cond_hash_1")
    assert cache.get(key) is None

    base_dict = {"1": {"class_type": "KSampler", "inputs": {}}}
    cache.put(key, base_dict)
    assert cache.get(key) == base_dict
    assert len(cache) == 1


def test_workflow_builder_uses_cache(workflow_library, standard_profile, base_prompt_package):
    builder = WorkflowBuilder()
    wf_ref = workflow_library.resolve("general", standard_profile)
    cache = WorkflowGraphCache(enabled=True)

    # First build -> cache miss & put
    built1 = builder.build(
        base_prompt_package, standard_profile, wf_ref, library=workflow_library, cache=cache
    )
    assert len(cache) == 1

    # Second build with different prompt/seed -> cache hit
    cand_package = base_prompt_package.model_copy(
        update={
            "positive_prompt": "Different candidate prompt",
            "generation_parameters": base_prompt_package.generation_parameters.model_copy(update={"seed": 101}),
        }
    )
    built2 = builder.build(
        cand_package, standard_profile, wf_ref, library=workflow_library, cache=cache
    )
    assert len(cache) == 1

    # Workflow hashes differ because prompt and seed differ!
    assert built1.workflow_hash != built2.workflow_hash


def test_workflow_cache_invalidation_on_profile_change(workflow_library, standard_profile, low_vram_profile, base_prompt_package):
    builder = WorkflowBuilder()
    wf_ref1 = workflow_library.resolve("general", standard_profile)
    wf_ref2 = workflow_library.resolve("general", low_vram_profile)
    cache = WorkflowGraphCache(enabled=True)

    builder.build(base_prompt_package, standard_profile, wf_ref1, library=workflow_library, cache=cache)
    assert len(cache) == 1

    # Fallback to low VRAM profile produces a new key
    builder.build(base_prompt_package, low_vram_profile, wf_ref2, library=workflow_library, cache=cache)
    assert len(cache) == 2
