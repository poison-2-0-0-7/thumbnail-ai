"""
test_composition_models.py
==========================

Unit tests for Module 10 Asset Composer data models and exception hierarchy.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from composition_exceptions import (
    AssetRegistryError,
    CompositionBaseError,
    CompositionInputInvalidError,
    GenerationBundleError,
    LayerPlacementError,
    MaskResolutionError,
    WorkspacePersistenceError,
    WorkspaceValidationError,
)
from models import (
    AssetPlacement,
    CanvasTransform,
    CompositionLayer,
    CompositionWorkspace,
    GenerationBundle,
    LayerDecision,
    LayerGroup,
    LayerRole,
    LayerTransform,
    LightingAdjustment,
    MaskReference,
    PlacementConstraints,
    TextPlacement,
    WorkspaceMetadata,
    WorkspaceStatistics,
)


def test_exception_hierarchy():
    """Verify all custom exceptions derive from CompositionBaseError."""
    exceptions = [
        CompositionInputInvalidError,
        AssetRegistryError,
        LayerPlacementError,
        MaskResolutionError,
        WorkspaceValidationError,
        WorkspacePersistenceError,
        GenerationBundleError,
    ]
    for exc in exceptions:
        assert issubclass(exc, CompositionBaseError)
        assert issubclass(exc, Exception)


def test_layer_decision_and_role_enums():
    """Verify enum string values."""
    assert LayerDecision.KEEP == "keep"
    assert LayerDecision.REMOVE == "remove"
    assert LayerDecision.REPLACE == "replace"
    assert LayerDecision.ENHANCE == "enhance"
    assert LayerDecision.ADD == "add"

    assert LayerRole.BACKGROUND == "background"
    assert LayerRole.FOREGROUND == "foreground"
    assert LayerRole.PERSON == "person"
    assert LayerRole.OBJECT == "object"
    assert LayerRole.TEXT == "text"
    assert LayerRole.EFFECT == "effect"


def test_canvas_transform_validation():
    """Verify CanvasTransform positive dimensions."""
    canvas = CanvasTransform(width=1280, height=720, aspect_ratio="16:9")
    assert canvas.width == 1280
    assert canvas.height == 720

    with pytest.raises(ValidationError):
        CanvasTransform(width=0, height=720, aspect_ratio="16:9")

    with pytest.raises(ValidationError):
        CanvasTransform(width=1280, height=-10, aspect_ratio="16:9")


def test_mask_reference_validation():
    """Verify MaskReference SHA-256 validation and empty path rejection."""
    valid_hash = "a" * 64
    mask = MaskReference(mask_path="path/to/mask.png", mask_checksum=valid_hash)
    assert mask.source == "vre"
    assert mask.mask_checksum == valid_hash

    with pytest.raises(ValidationError):
        MaskReference(mask_path="", mask_checksum=valid_hash)

    with pytest.raises(ValidationError):
        MaskReference(mask_path="path/to/mask.png", mask_checksum="not_sha256")


def test_workspace_metadata_validation():
    """Verify WorkspaceMetadata validation of hashes and text fields."""
    valid_hash = "b" * 64
    meta = WorkspaceMetadata(
        video_id="vid123",
        created_at="2026-07-30T00:00:00Z",
        vre_source_hash=valid_hash,
        redesign_spec_hash=valid_hash,
        prompt_package_hash=valid_hash,
        engine_version="1.0.0",
    )
    assert meta.video_id == "vid123"

    with pytest.raises(ValidationError):
        WorkspaceMetadata(
            video_id="",
            created_at="2026-07-30T00:00:00Z",
            vre_source_hash=valid_hash,
            redesign_spec_hash=valid_hash,
            prompt_package_hash=valid_hash,
            engine_version="1.0.0",
        )

    with pytest.raises(ValidationError):
        WorkspaceMetadata(
            video_id="vid123",
            created_at="2026-07-30T00:00:00Z",
            vre_source_hash="invalid_hash",
            redesign_spec_hash=valid_hash,
            prompt_package_hash=valid_hash,
            engine_version="1.0.0",
        )


def test_composition_workspace_construction():
    """Verify construction of a complete CompositionWorkspace."""
    valid_hash = "c" * 64
    canvas = CanvasTransform(width=1280, height=720, aspect_ratio="16:9")
    placement = AssetPlacement(
        asset_id="bg_1",
        role=LayerRole.BACKGROUND,
        decision=LayerDecision.REPLACE,
        transform=LayerTransform(),
        z_index=0,
    )
    layer = CompositionLayer(layer_id="layer_bg", placement=placement)
    group = LayerGroup(group_id="grp_bg", role=LayerRole.BACKGROUND, layer_ids=["layer_bg"])
    text_placement = TextPlacement(include_text=False)
    lighting = LightingAdjustment(
        target_brightness=0.5,
        target_contrast=0.5,
        target_saturation=0.5,
        warm_or_cool="neutral",
    )
    constraints = PlacementConstraints(safe_margin_px=24)
    stats = WorkspaceStatistics(total_layers=1, replaced=1)
    metadata = WorkspaceMetadata(
        video_id="test_vid",
        created_at="2026-07-30T00:00:00Z",
        vre_source_hash=valid_hash,
        redesign_spec_hash=valid_hash,
        prompt_package_hash=valid_hash,
        engine_version="1.0.0",
    )

    workspace = CompositionWorkspace(
        video_id="test_vid",
        canvas=canvas,
        layers=[layer],
        groups=[group],
        text_placement=text_placement,
        lighting=lighting,
        constraints=constraints,
        statistics=stats,
        metadata=metadata,
    )

    assert workspace.video_id == "test_vid"
    assert workspace.status == "success"
    assert len(workspace.layers) == 1


def test_generation_bundle_construction():
    """Verify GenerationBundle model construction and validation."""
    valid_hash = "d" * 64
    canvas = CanvasTransform(width=1280, height=720, aspect_ratio="16:9")
    bundle = GenerationBundle(
        video_id="test_vid",
        canvas=canvas,
        reference_image_paths={"person": "data/visual_references/test_vid/creator_face.png"},
        mask_paths={"person": "data/visual_references/test_vid/face_mask.png"},
        layer_order=["layer_person"],
        workspace_hash=valid_hash,
        prompt_package_hash=valid_hash,
        generated_at="2026-07-30T00:00:00Z",
    )

    assert bundle.video_id == "test_vid"
    assert bundle.reference_image_paths["person"].endswith("creator_face.png")

    with pytest.raises(ValidationError):
        GenerationBundle(
            video_id="",
            canvas=canvas,
            workspace_hash=valid_hash,
            prompt_package_hash=valid_hash,
            generated_at="2026-07-30T00:00:00Z",
        )
