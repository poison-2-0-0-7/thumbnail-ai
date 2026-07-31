"""
test_strategy_deriver.py
========================

Unit tests for StrategyDeriver in planner_components.
"""

import pytest

from models import (
    AssetPlacement,
    CanvasTransform,
    CompositionLayer,
    CompositionWorkspace,
    FaceStrategy,
    BackgroundStrategy,
    LayerDecision,
    LayerRole,
    LayerTransform,
    LightingAdjustment,
    PlacementConstraints,
    TextPlacement,
    WorkspaceMetadata,
    WorkspaceStatistics,
)
from planner_components.strategy_deriver import StrategyDeriver


@pytest.fixture
def minimal_workspace():
    valid_hash = "a" * 64
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
        depth_hint_path="depth.png",
    )
    l_person = CompositionLayer(
        layer_id="l_person",
        placement=AssetPlacement(
            asset_id="person",
            role=LayerRole.PERSON,
            decision=LayerDecision.KEEP,
            source_path="person.png",
            transform=LayerTransform(),
            z_index=10,
        ),
    )
    l_obj = CompositionLayer(
        layer_id="l_obj",
        placement=AssetPlacement(
            asset_id="object_0_mic",
            role=LayerRole.OBJECT,
            decision=LayerDecision.KEEP,
            source_path="mic.png",
            transform=LayerTransform(),
            z_index=15,
        ),
    )
    l_rem = CompositionLayer(
        layer_id="l_rem",
        placement=AssetPlacement(
            asset_id="logo_old",
            role=LayerRole.OBJECT,
            decision=LayerDecision.REMOVE,
            source_path="old.png",
            transform=LayerTransform(),
            z_index=20,
        ),
    )
    return CompositionWorkspace(
        video_id="vid1",
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        layers=[l_bg, l_person, l_obj, l_rem],
        groups=[],
        text_placement=TextPlacement(),
        lighting=LightingAdjustment(
            target_brightness=0.5, target_contrast=0.5, target_saturation=0.5, warm_or_cool="neutral"
        ),
        constraints=PlacementConstraints(),
        statistics=WorkspaceStatistics(total_layers=4, replaced=1, kept=2, removed=1),
        metadata=WorkspaceMetadata(
            video_id="vid1",
            created_at="2026-08-01T00:00:00Z",
            vre_source_hash=valid_hash,
            redesign_spec_hash=valid_hash,
            prompt_package_hash=valid_hash,
            engine_version="1.0.0",
        ),
    )


def test_strategy_deriver_basic(minimal_workspace):
    deriver = StrategyDeriver()
    res = deriver.derive_strategies(minimal_workspace)

    assert res["face_strategy"] == FaceStrategy.PRESERVE_AS_IS
    assert res["background_strategy"] == BackgroundStrategy.STRUCTURE_GUIDED_REPLACE
    assert "object_0_mic" in res["preserve_objects"]
    assert "logo_old" not in res["preserve_objects"]
    assert "no logo_old" in res["negative_constraints"]
