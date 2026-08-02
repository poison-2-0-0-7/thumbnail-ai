"""
tests/test_observability/test_pipeline_trace.py
=================================================

Tests for Part 3: TraceAssembler and PipelineTraceBuilder.
"""

import json
import pytest
from pathlib import Path
from observability.trace.artifact_index_builder import ArtifactIndexBuilder
from observability.trace.log_correlator import LogCorrelator
from observability.trace.trace_assembler import TraceAssembler
from observability.trace.pipeline_trace_builder import PipelineTraceBuilder


def test_trace_assembler_ordering_and_status(tmp_path, monkeypatch):
    video_id = "vid_trace_test"

    # Setup directories
    thumb_dir = tmp_path / "thumbnails"
    thumb_dir.mkdir()
    (thumb_dir / f"{video_id}.jpg").write_bytes(b"thumb")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / f"{video_id}.json").write_text(json.dumps({"status": "ok"}))

    metrics_path = tmp_path / "module7_metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"video_id": video_id, "total_duration_seconds": 12.34}) + "\n"
    )

    m1_log = tmp_path / "module1.log"
    m1_log.write_text(
        "2026-08-02 11:00:00 | INFO | module1 | Reading video vid_trace_test\n"
        "2026-08-02 11:00:02 | INFO | module1 | Done vid_trace_test\n"
    )

    m7_log = tmp_path / "module7.log"
    m7_log.write_text(
        "2026-08-02 11:05:00 | INFO | module7 | Generation start vid_trace_test\n"
    )

    builder = ArtifactIndexBuilder(
        thumbnail_dir=thumb_dir,
        analysis_dir=analysis_dir,
        redesign_spec_dir=tmp_path / "specs",
        design_blueprint_dir=tmp_path / "blueprints",
        prompt_package_dir=tmp_path / "packages",
        asset_extraction_dir=tmp_path / "assets",
        decision_dir=tmp_path / "decisions",
        composition_workspace_dir=tmp_path / "workspaces",
        generation_plan_dir=tmp_path / "plans",
        strategy_pack_dir=tmp_path / "strategy",
        generated_thumbnail_dir=tmp_path / "generated",
    )

    correlator = LogCorrelator(log_files={"module1": m1_log, "module7": m7_log})
    assembler = TraceAssembler(metrics_path=metrics_path)

    index = builder.collect(video_id)
    logs = correlator.correlate(video_id)

    trace = assembler.assemble_from_parts(video_id, index, logs)

    assert trace.video_id == video_id
    assert len(trace.modules) == 12

    # Verify stage order
    module_names = [m.module for m in trace.modules]
    expected_order = [
        "module1",
        "module2",
        "module3",
        "module4",
        "module8",
        "module5",
        "module5.5",
        "module6",
        "module9",
        "module10",
        "module10.5",
        "module7",
    ]
    assert module_names == expected_order

    # Verify Module 8/9/10.5 flag-gated default status ("not_run")
    m8_entry = next(m for m in trace.modules if m.module == "module8")
    assert m8_entry.status == "not_run"

    # Verify exact duration for Module 7 from metrics
    m7_entry = next(m for m in trace.modules if m.module == "module7")
    assert m7_entry.duration_seconds == 12.34
    assert m7_entry.duration_source == "exact"

    # Verify log-derived duration for Module 1
    m1_entry = next(m for m in trace.modules if m.module == "module1")
    assert m1_entry.duration_seconds == 2.0
    assert m1_entry.duration_source == "log_derived"


def test_pipeline_trace_builder_persistence(tmp_path):
    video_id = "vid_persist_test"
    out_dir = tmp_path / "obs_traces"

    builder = PipelineTraceBuilder(output_dir=out_dir)
    trace = builder.build_and_persist(video_id)

    trace_file = out_dir / video_id / "pipeline_trace.json"
    index_file = out_dir / video_id / "artifact_index.json"

    assert trace_file.is_file()
    assert index_file.is_file()

    saved_data = json.loads(trace_file.read_text())
    assert saved_data["video_id"] == video_id
