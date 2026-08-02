from __future__ import annotations

from pathlib import Path
import sys

import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from generation_components.region_plan_validator import RegionPlanValidator
from models import (
    AssetPlacement,
    CanvasTransform,
    CompositionLayer,
    CompositionWorkspace,
    DecisionAction,
    DecisionManifest,
    DecisionSource,
    LayerDecision,
    LayerRole,
    LayerTransform,
    LightingAdjustment,
    PlacementConstraints,
    ResolvedDecision,
    TargetElement,
    TextPlacement,
    WorkspaceMetadata,
    WorkspaceStatistics,
)


VIDEO_ID = "test_vid_123"


def _decision(element_id: str, element_type: str, action: DecisionAction) -> ResolvedDecision:
    return ResolvedDecision(
        decision_id=f"dec_{element_id}",
        target=TargetElement(element_id=element_id, element_type=element_type, label=element_id),
        action=action,
        confidence=0.9,
        source=DecisionSource.RULE,
        rationale="test rationale",
        priority_rank=1,
    )


def _workspace(video_id: str, layers: list[CompositionLayer], role_mask_paths: dict[str, str] | None = None) -> CompositionWorkspace:
    ws = CompositionWorkspace(
        video_id=video_id,
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        layers=layers,
        groups=[],
        text_placement=TextPlacement(),
        lighting=LightingAdjustment(
            target_brightness=0.5,
            target_contrast=0.5,
            target_saturation=0.5,
            warm_or_cool="neutral",
        ),
        constraints=PlacementConstraints(),
        statistics=WorkspaceStatistics(total_layers=len(layers), kept=1),
        metadata=WorkspaceMetadata(
            video_id=video_id,
            created_at="2026-01-01T00:00:00Z",
            vre_source_hash="a" * 64,
            redesign_spec_hash="b" * 64,
            prompt_package_hash="c" * 64,
            engine_version="1.0.0",
        ),
    )
    if role_mask_paths is not None:
        object.__setattr__(ws, "role_mask_paths", role_mask_paths)
    return ws


def test_classify_none_when_no_manifest(monkeypatch) -> None:
    monkeypatch.setattr("generation_components.region_plan_validator.DECISION_ENGINE_ENABLED", True)
    validator = RegionPlanValidator()
    plan = validator.classify(VIDEO_ID, decision_manifest=None)
    assert plan.edit_scope == "none"
    assert len(plan.regions) == 0


def test_classify_background_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("generation_components.region_plan_validator.DECISION_ENGINE_ENABLED", True)
    bg_mask = tmp_path / "bg_mask.png"
    bg_mask.write_bytes(b"mask")

    manifest = DecisionManifest(
        video_id=VIDEO_ID,
        source_generated_image_path="src.png",
        source_generated_image_hash="a" * 64,
        decisions=[_decision("background", "background", DecisionAction.REPLACE)],
        decided_at="2026-08-01T00:00:00Z",
    )

    layer = CompositionLayer(
        layer_id="background",
        mask_path=str(bg_mask),
        placement=AssetPlacement(
            asset_id="asset_bg",
            role=LayerRole.BACKGROUND,
            decision=LayerDecision.REPLACE,
            source_path="/path/to/bg.png",
            transform=LayerTransform(),
            z_index=0,
        ),
    )
    ws = _workspace(VIDEO_ID, [layer], role_mask_paths={"background": str(bg_mask)})

    validator = RegionPlanValidator()
    plan = validator.classify(VIDEO_ID, manifest, ws)

    assert plan.edit_scope == "background_only"
    assert len(plan.regions) == 1
    assert plan.regions[0].stage == "background"
    assert plan.regions[0].decision_type == "replace"


def test_classify_object_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("generation_components.region_plan_validator.DECISION_ENGINE_ENABLED", True)
    obj_mask = tmp_path / "obj_mask.png"
    obj_mask.write_bytes(b"mask")

    manifest = DecisionManifest(
        video_id=VIDEO_ID,
        source_generated_image_path="src.png",
        source_generated_image_hash="a" * 64,
        decisions=[_decision("person_0", "person", DecisionAction.ENHANCE)],
        decided_at="2026-08-01T00:00:00Z",
    )
    layer = CompositionLayer(
        layer_id="person_0",
        mask_path=str(obj_mask),
        placement=AssetPlacement(
            asset_id="asset_person",
            role=LayerRole.PERSON,
            decision=LayerDecision.ENHANCE,
            source_path="/path/to/person.png",
            transform=LayerTransform(),
            z_index=1,
        ),
    )
    ws = _workspace(VIDEO_ID, [layer], role_mask_paths={"person_0": str(obj_mask)})

    validator = RegionPlanValidator()
    plan = validator.classify(VIDEO_ID, manifest, ws)

    assert plan.edit_scope == "object_only"
    assert len(plan.regions) == 1
    assert plan.regions[0].stage == "object"
    assert plan.regions[0].decision_type == "enhance"
    assert plan.regions[0].denoise_strength == 0.35


def test_classify_heavy_redesign(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("generation_components.region_plan_validator.DECISION_ENGINE_ENABLED", True)
    bg_mask = tmp_path / "bg_mask.png"
    bg_mask.write_bytes(b"mask")
    obj_mask = tmp_path / "obj_mask.png"
    obj_mask.write_bytes(b"mask")

    manifest = DecisionManifest(
        video_id=VIDEO_ID,
        source_generated_image_path="src.png",
        source_generated_image_hash="a" * 64,
        decisions=[
            _decision("background", "background", DecisionAction.REPLACE),
            _decision("product_x", "object", DecisionAction.ADD),
        ],
        decided_at="2026-08-01T00:00:00Z",
    )
    layer_bg = CompositionLayer(
        layer_id="background",
        mask_path=str(bg_mask),
        placement=AssetPlacement(
            asset_id="asset_bg",
            role=LayerRole.BACKGROUND,
            decision=LayerDecision.REPLACE,
            source_path="/path/to/bg.png",
            transform=LayerTransform(),
            z_index=0,
        ),
    )
    layer_obj = CompositionLayer(
        layer_id="product_x",
        mask_path=str(obj_mask),
        placement=AssetPlacement(
            asset_id="asset_prod",
            role=LayerRole.OBJECT,
            decision=LayerDecision.ADD,
            source_path="/path/to/prod.png",
            transform=LayerTransform(),
            z_index=2,
        ),
    )
    ws = _workspace(VIDEO_ID, [layer_bg, layer_obj], role_mask_paths={"background": str(bg_mask), "product_x": str(obj_mask)})

    validator = RegionPlanValidator()
    plan = validator.classify(VIDEO_ID, manifest, ws)

    assert plan.edit_scope == "heavy_redesign"
    assert len(plan.regions) == 2


def test_classify_fallback_for_unresolvable_mask(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("generation_components.region_plan_validator.DECISION_ENGINE_ENABLED", True)
    manifest = DecisionManifest(
        video_id=VIDEO_ID,
        source_generated_image_path="src.png",
        source_generated_image_hash="a" * 64,
        decisions=[_decision("missing_obj", "object", DecisionAction.REPLACE)],
        decided_at="2026-08-01T00:00:00Z",
    )
    ws = _workspace(VIDEO_ID, [])

    validator = RegionPlanValidator()
    plan = validator.classify(VIDEO_ID, manifest, ws)

    assert plan.edit_scope == "none"
    assert len(plan.regions) == 0
    assert len(plan.fallback_elements) == 1
    assert plan.fallback_elements[0]["element_id"] == "missing_obj"
    assert plan.fallback_elements[0]["reason"] == "unresolvable_mask"
