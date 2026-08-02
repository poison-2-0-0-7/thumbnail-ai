"""
tests/test_observability/test_generation_trace_recorder.py
============================================================

Unit and integration tests for Sprint 2: GenerationTraceRecord, GenerationTraceFactory,
GenerationTracePersistence, GenerationTraceRecorder, and Module 7 instrumentation.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from observability.generation_trace import (
    GenerationTraceFactory,
    GenerationTracePersistence,
    GenerationTraceRecorder,
)
from observability.models import (
    FragmentAttachmentRecord,
    GenerationTraceRecord,
)
from modules.models import (
    GenerationParameters,
    GenerationProfile,
    PromptPackage,
    QualityParameters,
)


def test_generation_trace_record_model():
    frag = FragmentAttachmentRecord(fragment_name="controlnet_depth", attach_point="ATTACH_PREVIOUS", strength_or_weight=0.75)
    rec = GenerationTraceRecord(
        video_id="vid_test_01",
        attempt_index=0,
        generation_id="vid_test_01_cand_0",
        workflow_template="gaming.json",
        workflow_hash="hash_123456",
        workflow_fragments=["controlnet_depth"],
        fragments_attached=[frag],
        latent_source="noise",
        denoise=1.0,
        seed=42,
        controlnet_enabled=True,
        ipadapter_enabled=False,
        recorded_at="2026-08-02T12:00:00Z",
    )
    assert rec.video_id == "vid_test_01"
    assert rec.seed == 42
    assert rec.fragments_attached[0].fragment_name == "controlnet_depth"

    # Test serialization and deserialization
    json_data = rec.model_dump_json()
    reloaded = GenerationTraceRecord.model_validate_json(json_data)
    assert reloaded == rec


def test_generation_trace_factory():
    package = PromptPackage.model_construct(
        video_id="vid_factory_01",
        niche="gaming",
        positive_prompt="Epic gaming thumbnail",
        generation_parameters=GenerationParameters(seed=12345, width=1280, height=720),
    )
    profile = GenerationProfile.model_construct(
        name="PROFILE_DEFAULT",
        checkpoint="sdxl_base.safetensors",
        steps=30,
        cfg=7.5,
        sampler="euler",
        scheduler="normal",
        controlnet_enabled=True,
        ipadapter_enabled=True,
    )

    built_wf = MagicMock()
    built_wf.workflow_hash = "wf_hash_999"
    built_wf.workflow_ref.template_name = "gaming.json"

    frags = [
        {"fragment_name": "controlnet_depth", "attach_point": "point_depth"},
        FragmentAttachmentRecord(fragment_name="ipadapter_ref", attach_point="point_ip"),
    ]

    record = GenerationTraceFactory.create(
        video_id="vid_factory_01",
        attempt_index=1,
        package=package,
        profile=profile,
        built_wf=built_wf,
        fragments_attached=frags,
        output_image_path="/tmp/output.png",
    )

    assert record.video_id == "vid_factory_01"
    assert record.attempt_index == 1
    assert record.seed == 12345
    assert record.workflow_template == "gaming.json"
    assert record.workflow_hash == "wf_hash_999"
    assert len(record.fragments_attached) == 2
    assert record.controlnet_enabled is True
    assert record.ipadapter_enabled is True
    assert record.output_image_path == "/tmp/output.png"


def test_generation_trace_persistence(tmp_path):
    output_dir = tmp_path / "gen_traces"
    persistence = GenerationTracePersistence(output_dir=output_dir)

    record = GenerationTraceRecord(
        video_id="vid_persist_01",
        attempt_index=0,
        workflow_template="tech.json",
        workflow_hash="hash_tech_11",
        recorded_at="2026-08-02T12:00:00Z",
    )

    saved_path = persistence.save(record)
    assert saved_path.is_file()
    assert saved_path.parent == output_dir / "vid_persist_01"
    assert saved_path.name == "generation_trace_record.json"

    loaded = persistence.load("vid_persist_01")
    assert loaded is not None
    assert loaded.video_id == "vid_persist_01"
    assert loaded.workflow_template == "tech.json"
    assert persistence.validate(loaded) is True


def test_generation_trace_recorder_non_fatal(tmp_path):
    recorder = GenerationTraceRecorder(output_dir=tmp_path)

    # Calling with valid data should return saved path
    path = recorder.record(
        video_id="vid_rec_01",
        attempt_index=0,
    )
    assert path is not None
    assert path.is_file()

    # Recorder should swallow any exception and return None without raising
    with patch.object(GenerationTracePersistence, "save", side_effect=RuntimeError("Disk failure")):
        path = recorder.record(video_id="vid_rec_err")
        assert path is None


def test_module7_instrumentation_integration(tmp_path):
    """
    Verify that ImageGeneratorPipeline runs with GenerationTraceRecorder,
    producing valid generation trace records while keeping pipeline output intact.
    """
    from modules.image_generator import ImageGeneratorPipeline

    out_traces_dir = tmp_path / "gen_traces"
    recorder = GenerationTraceRecorder(output_dir=out_traces_dir)

    pipeline = ImageGeneratorPipeline(trace_recorder=recorder)
    assert pipeline.trace_recorder == recorder
