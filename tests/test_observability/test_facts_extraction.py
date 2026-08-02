"""
tests/test_observability/test_facts_extraction.py
===================================================

Comprehensive tests for Sprint 3A: Facts Extraction Layer (PORCE).
Verifies:
- Fact extraction from valid, partial, and corrupted traces
- Fact persistence & atomic saves
- Fact loading & deterministic reload
- Schema validation & versioning
- Strict objective observation rules (no inferences/opinions)
- Backward compatibility & regression safety
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from observability.config import OBS_FACTS_VERSION
from observability.exceptions import FactPersistenceError, FactValidationError
from observability.facts import (
    FactCollection,
    FactExtractor,
    FactLoader,
    FactModel,
    FactPersistence,
    FactRegistry,
    FactSerializer,
    FactValidation,
    TraceFacts,
)
from observability.models import (
    ArtifactIndex,
    ArtifactRef,
    FragmentAttachmentRecord,
    GenerationTraceRecord,
    LogLineRef,
    ModuleTraceEntry,
    PipelineTrace,
)


@pytest.fixture
def sample_generation_trace() -> GenerationTraceRecord:
    return GenerationTraceRecord(
        video_id="vid_test_123",
        attempt_index=0,
        generation_id="vid_test_123_cand_0",
        workflow_template="gaming.json",
        workflow_hash="sha256_hash_123",
        workflow_fragments=["fragment_controlnet", "fragment_ipadapter"],
        fragments_attached=[
            FragmentAttachmentRecord(
                fragment_name="fragment_controlnet",
                attach_point="PREVIOUS",
                strength_or_weight=0.8,
            )
        ],
        latent_source="noise",
        denoise=1.0,
        seed=42,
        scheduler="normal",
        sampler="euler",
        steps=20,
        cfg=7.0,
        controlnet_enabled=True,
        ipadapter_enabled=True,
        edit_mode="txt2img",
        generation_profile="gaming_v1",
        controlnet_config={"enabled": True, "strength": 0.8},
        ipadapter_config={"enabled": True, "weight": 0.5},
        conditioning_assets=["data/cond/depth.png"],
        asset_references=["data/assets/bg.jpg", "data/assets/subject.jpg"],
        mask_references=["data/masks/mask1.png"],
        source_thumbnail_path="data/thumbnails/vid_test_123.jpg",
        edit_mask_paths=["data/masks/edit1.png"],
        renderer_version="1.0.0",
        recorded_at="2026-08-02T12:00:00Z",
    )


@pytest.fixture
def sample_pipeline_trace(sample_generation_trace: GenerationTraceRecord) -> PipelineTrace:
    ref_m3 = ArtifactRef(module="module3", artifact_type="thumbnail", path="data/thumbnails/vid_test_123.jpg", exists=True)
    ref_m7 = ArtifactRef(module="module7", artifact_type="generated_image", path="data/generated_thumbnails/vid_test_123/vid_test_123.png", exists=True)
    ref_m10 = ArtifactRef(module="module10", artifact_type="workspace", path="data/composition_workspaces/vid_test_123", exists=True)

    index = ArtifactIndex(video_id="vid_test_123", refs=[ref_m3, ref_m7, ref_m10], built_at="2026-08-02T12:00:00Z")

    mod_entries = [
        ModuleTraceEntry(
            module="module1", stage_order=1, status="success", duration_seconds=0.5, outputs=[]
        ),
        ModuleTraceEntry(
            module="module3", stage_order=3, status="success", duration_seconds=1.2, outputs=[ref_m3]
        ),
        ModuleTraceEntry(
            module="module10", stage_order=10, status="success", duration_seconds=2.0, outputs=[ref_m10]
        ),
        ModuleTraceEntry(
            module="module7",
            stage_order=12,
            status="success",
            duration_seconds=15.5,
            config_snapshot={"ASSET_EXTRACTION_ENABLED": True, "DECISION_ENGINE_ENABLED": True},
            outputs=[ref_m7],
        ),
    ]

    return PipelineTrace(
        video_id="vid_test_123",
        modules=mod_entries,
        artifact_index=index,
        generation_trace=sample_generation_trace,
        overall_status="success",
        assembled_at="2026-08-02T12:00:00Z",
    )


def test_fact_extraction_valid_trace(sample_pipeline_trace: PipelineTrace) -> None:
    extractor = FactExtractor()
    collection = extractor.extract(sample_pipeline_trace)

    assert collection.video_id == "vid_test_123"
    assert collection.fact_version == OBS_FACTS_VERSION

    facts = collection.trace_facts
    assert facts.video_id == "vid_test_123"
    assert facts.workflow_selected == "gaming.json"
    assert facts.edit_mode == "txt2img"
    assert facts.generation_profile == "gaming_v1"
    assert facts.sampler == "euler"
    assert facts.scheduler == "normal"
    assert facts.seed == 42
    assert facts.cfg == 7.0
    assert facts.steps == 20
    assert facts.denoise == 1.0
    assert facts.latent_initialization_mode == "EmptyLatentImage"

    assert facts.controlnet_count == 1
    assert facts.controlnet_enabled is True
    assert facts.ipadapter_count == 1
    assert facts.ipadapter_enabled is True

    assert facts.source_thumbnail_exists is True
    assert facts.generated_thumbnail_exists is True
    assert facts.has_composition_workspace is True

    assert facts.asset_extraction_enabled is True
    assert facts.decision_engine_enabled is True
    assert facts.thumbnail_planner_enabled is False

    assert len(collection.atomic_facts) > 0


def test_fact_extraction_missing_fields_and_trace_none() -> None:
    index = ArtifactIndex(video_id="vid_partial", refs=[], built_at="2026-08-02T12:00:00Z")
    partial_trace = PipelineTrace(
        video_id="vid_partial",
        modules=[],
        artifact_index=index,
        generation_trace=None,
        overall_status="partial",
        assembled_at="2026-08-02T12:00:00Z",
    )

    extractor = FactExtractor()
    collection = extractor.extract(partial_trace)

    assert collection.video_id == "vid_partial"
    facts = collection.trace_facts
    assert facts.workflow_selected is None
    assert facts.edit_mode is None
    assert facts.seed is None
    assert facts.controlnet_count == 0
    assert facts.source_thumbnail_exists is False
    assert facts.generated_thumbnail_exists is False


def test_fact_persistence_and_loading(tmp_path: Path, sample_pipeline_trace: PipelineTrace) -> None:
    extractor = FactExtractor()
    collection = extractor.extract(sample_pipeline_trace)

    persistence = FactPersistence(output_dir=tmp_path)
    saved_path = persistence.save(collection)

    assert saved_path.exists()
    assert saved_path.name == "facts.json"
    assert "vid_test_123" in str(saved_path)

    reloaded = persistence.load("vid_test_123")
    assert reloaded is not None
    assert reloaded.video_id == collection.video_id
    assert reloaded.trace_facts.workflow_selected == collection.trace_facts.workflow_selected
    assert reloaded.trace_facts.seed == collection.trace_facts.seed


def test_fact_loader_helper(tmp_path: Path, sample_pipeline_trace: PipelineTrace) -> None:
    extractor = FactExtractor()
    collection = extractor.extract(sample_pipeline_trace)

    persistence = FactPersistence(output_dir=tmp_path)
    saved_path = persistence.save(collection)

    loader = FactLoader(persistence=persistence)
    by_id = loader.load_by_video_id("vid_test_123")
    assert by_id is not None
    assert by_id.video_id == "vid_test_123"

    by_path = loader.load_by_path(saved_path)
    assert by_path.video_id == "vid_test_123"


def test_fact_serializer_and_validation(sample_pipeline_trace: PipelineTrace) -> None:
    extractor = FactExtractor()
    collection = extractor.extract(sample_pipeline_trace)

    serializer = FactSerializer()
    json_str = serializer.serialize(collection)
    assert isinstance(json_str, str)
    assert "vid_test_123" in json_str

    deserialized = serializer.deserialize(json_str)
    assert deserialized.video_id == collection.video_id

    # Validation tests
    assert FactValidation.validate_collection_data(collection) is True
    assert FactValidation.validate_collection_data(json.loads(json_str)) is True
    assert FactValidation.validate_collection_data({"invalid": "data"}) is False

    assert FactValidation.validate_trace_facts_data(collection.trace_facts) is True
    assert FactValidation.check_version_compatibility(OBS_FACTS_VERSION) is True
    assert FactValidation.check_version_compatibility("2.0.0") is False


def test_corrupted_facts_file(tmp_path: Path) -> None:
    corrupted_dir = tmp_path / "vid_bad"
    corrupted_dir.mkdir(parents=True, exist_ok=True)
    bad_file = corrupted_dir / "facts.json"
    bad_file.write_text("{corrupted json", encoding="utf-8")

    persistence = FactPersistence(output_dir=tmp_path)
    assert persistence.load("vid_bad") is None

    with pytest.raises(FactPersistenceError):
        persistence.load_file(bad_file)


def test_fact_registry_custom_handlers(sample_pipeline_trace: PipelineTrace) -> None:
    registry = FactRegistry()
    registry.register("custom_perf", lambda trace: {"custom_metric": 42})

    assert "custom_perf" in registry.get_registered_categories()

    extractor = FactExtractor(registry=registry)
    collection = extractor.extract(sample_pipeline_trace)

    custom_fact_keys = [f.fact_key for f in collection.atomic_facts]
    assert "custom_metric" in custom_fact_keys


def test_strict_objective_facts_no_inferences(sample_pipeline_trace: PipelineTrace) -> None:
    extractor = FactExtractor()
    collection = extractor.extract(sample_pipeline_trace)

    raw_dict = collection.model_dump()
    json_dump = json.dumps(raw_dict)

    # Ensure facts do NOT contain opinion/diagnostic words
    bad_words = [
        "ignored edit plan",
        "editing failed",
        "quality is poor",
        "unreasonably slow",
        "unexpected failure",
    ]
    for word in bad_words:
        assert word not in json_dump.lower()
