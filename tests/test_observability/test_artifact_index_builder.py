"""
tests/test_observability/test_artifact_index_builder.py
=========================================================

Tests for Part 1: ArtifactIndexBuilder.
"""

import hashlib
import json
import pytest
from pathlib import Path
from observability.trace.artifact_index_builder import ArtifactIndexBuilder, compute_sha256


def test_compute_sha256(tmp_path):
    file_path = tmp_path / "test.txt"
    content = b"hello thumbnail-ai observability"
    file_path.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    assert compute_sha256(file_path) == expected_hash

    missing_path = tmp_path / "missing.txt"
    assert compute_sha256(missing_path) is None


def test_artifact_index_builder_collect(tmp_path):
    video_id = "test_vid_123"

    # Setup mock directory structure with some existing and some missing artifacts
    thumb_dir = tmp_path / "thumbnails"
    thumb_dir.mkdir()
    (thumb_dir / f"{video_id}.jpg").write_bytes(b"jpg content")

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / f"{video_id}.json").write_text(json.dumps({"analysis": "ok"}))

    gen_dir = tmp_path / "generated"
    (gen_dir / video_id).mkdir(parents=True)
    (gen_dir / video_id / f"{video_id}.png").write_bytes(b"png content")

    builder = ArtifactIndexBuilder(
        thumbnail_dir=thumb_dir,
        analysis_dir=analysis_dir,
        generated_thumbnail_dir=gen_dir,
        redesign_spec_dir=tmp_path / "specs",
        design_blueprint_dir=tmp_path / "blueprints",
        prompt_package_dir=tmp_path / "packages",
        asset_extraction_dir=tmp_path / "assets",
        decision_dir=tmp_path / "decisions",
        composition_workspace_dir=tmp_path / "workspaces",
        generation_plan_dir=tmp_path / "plans",
        strategy_pack_dir=tmp_path / "strategy",
    )

    index = builder.collect(video_id)

    assert index.video_id == video_id
    assert len(index.refs) > 0

    # Verify thumbnail_image ref exists
    thumb_ref = next(r for r in index.refs if r.artifact_type == "thumbnail_image")
    assert thumb_ref.exists is True
    assert thumb_ref.sha256 is not None
    assert thumb_ref.size_bytes == len(b"jpg content")

    # Verify redesign_specification ref is marked as missing
    spec_ref = next(r for r in index.refs if r.artifact_type == "redesign_specification")
    assert spec_ref.exists is False
    assert spec_ref.sha256 is None
    assert spec_ref.size_bytes is None
