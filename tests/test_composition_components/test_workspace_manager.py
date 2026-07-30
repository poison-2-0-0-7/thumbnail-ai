"""
test_workspace_manager.py
==========================

Unit tests for WorkspaceManager component in Module 10 Asset Composer.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from composition_components.workspace_manager import WorkspaceManager
from models import (
    AssetPlacement,
    CanvasTransform,
    CompositionLayer,
    CompositionWorkspace,
    LayerDecision,
    LayerGroup,
    LayerRole,
    LayerTransform,
    LightingAdjustment,
    PlacementConstraints,
    TextPlacement,
    WorkspaceMetadata,
    WorkspaceStatistics,
)


@pytest.fixture
def dummy_workspace():
    valid_hash = "a" * 64
    layer = CompositionLayer(
        layer_id="l1",
        placement=AssetPlacement(
            asset_id="bg",
            role=LayerRole.BACKGROUND,
            decision=LayerDecision.REPLACE,
            transform=LayerTransform(),
            z_index=0,
        ),
    )
    return CompositionWorkspace(
        video_id="test_vid",
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        layers=[layer],
        groups=[LayerGroup(group_id="g1", role=LayerRole.BACKGROUND, layer_ids=["l1"])],
        text_placement=TextPlacement(),
        lighting=LightingAdjustment(
            target_brightness=0.5,
            target_contrast=0.5,
            target_saturation=0.5,
            warm_or_cool="neutral",
        ),
        constraints=PlacementConstraints(),
        statistics=WorkspaceStatistics(total_layers=1, replaced=1),
        metadata=WorkspaceMetadata(
            video_id="test_vid",
            created_at="2026-07-30T00:00:00Z",
            vre_source_hash=valid_hash,
            redesign_spec_hash=valid_hash,
            prompt_package_hash=valid_hash,
            engine_version="1.0.0",
        ),
    )


def test_workspace_manager_persist_load_purge(tmp_path: Path, dummy_workspace):
    mgr = WorkspaceManager(root_dir=tmp_path)
    target = mgr.persist(dummy_workspace)

    assert target.is_dir()
    assert (target / "composition.json").is_file()
    assert (target / "metadata.json").is_file()
    assert (target / "workspace_manifest.json").is_file()
    assert (target / "layers" / "l1.json").is_file()

    # Load
    loaded = mgr.load("test_vid")
    assert loaded.video_id == "test_vid"
    assert loaded.canvas.width == 1280
    assert len(loaded.layers) == 1

    # Resume hit
    expected_hashes = {
        "vre_source_hash": "a" * 64,
        "redesign_spec_hash": "a" * 64,
        "prompt_package_hash": "a" * 64,
    }
    resumed = mgr.resume("test_vid", expected_hashes)
    assert resumed is not None
    assert resumed.video_id == "test_vid"

    # Resume miss on hash mismatch
    bad_hashes = dict(expected_hashes, vre_source_hash="b" * 64)
    resumed_miss = mgr.resume("test_vid", bad_hashes)
    assert resumed_miss is None

    # Purge
    assert mgr.purge("test_vid") is True
    assert not target.exists()
    assert mgr.purge("test_vid") is False
