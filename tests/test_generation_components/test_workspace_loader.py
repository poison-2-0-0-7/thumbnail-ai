"""
Tests for CompositionWorkspaceLoader.
"""

from pathlib import Path
import pytest
from generation_components.workspace_loader import CompositionWorkspaceLoader
from module7_exceptions import WorkspaceNotFoundError
from models import (
    AssetPlacement,
    CanvasTransform,
    CompositionLayer,
    CompositionWorkspace,
    LayerDecision,
    LayerRole,
    LayerTransform,
    LightingAdjustment,
    PlacementConstraints,
    TextPlacement,
    WorkspaceMetadata,
    WorkspaceStatistics,
)


def _make_dummy_workspace(video_id: str = "vid123") -> CompositionWorkspace:
    layer = CompositionLayer(
        layer_id="layer_bg",
        placement=AssetPlacement(
            asset_id="asset_bg",
            role=LayerRole.BACKGROUND,
            decision=LayerDecision.KEEP,
            source_path="/path/to/bg.png",
            transform=LayerTransform(),
            z_index=0,
        ),
    )
    return CompositionWorkspace(
        video_id=video_id,
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        layers=[layer],
        groups=[],
        text_placement=TextPlacement(),
        lighting=LightingAdjustment(
            target_brightness=0.5,
            target_contrast=0.5,
            target_saturation=0.5,
            warm_or_cool="neutral",
        ),
        constraints=PlacementConstraints(),
        statistics=WorkspaceStatistics(total_layers=1, kept=1),
        metadata=WorkspaceMetadata(
            video_id=video_id,
            created_at="2026-01-01T00:00:00Z",
            vre_source_hash="a" * 64,
            redesign_spec_hash="b" * 64,
            prompt_package_hash="c" * 64,
            engine_version="1.0.0",
        ),
    )


def test_workspace_loader_success(tmp_path: Path):
    ws = _make_dummy_workspace("vid123")
    vid_dir = tmp_path / "vid123"
    vid_dir.mkdir()
    comp_file = vid_dir / "composition.json"
    comp_file.write_text(ws.model_dump_json(), encoding="utf-8")

    loader = CompositionWorkspaceLoader(root_dir=tmp_path)
    loaded = loader.load("vid123")

    assert loaded.video_id == "vid123"
    assert loaded.canvas.width == 1280
    assert len(loaded.layers) == 1


def test_workspace_loader_missing_raises(tmp_path: Path):
    loader = CompositionWorkspaceLoader(root_dir=tmp_path)
    with pytest.raises(WorkspaceNotFoundError, match="not found"):
        loader.load("nonexistent_vid")


def test_workspace_loader_mismatched_video_id(tmp_path: Path):
    ws = _make_dummy_workspace("other_vid")
    vid_dir = tmp_path / "vid123"
    vid_dir.mkdir()
    comp_file = vid_dir / "composition.json"
    comp_file.write_text(ws.model_dump_json(), encoding="utf-8")

    loader = CompositionWorkspaceLoader(root_dir=tmp_path)
    with pytest.raises(WorkspaceNotFoundError, match="mismatch"):
        loader.load("vid123")
