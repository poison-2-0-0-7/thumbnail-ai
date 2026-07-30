"""
test_generation_bundle_builder.py
==================================

Unit tests for GenerationBundleBuilder component in Module 10 Asset Composer.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from composition_components.generation_bundle_builder import GenerationBundleBuilder
from composition_exceptions import GenerationBundleError
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
    MaskReference,
    PlacementConstraints,
    TextPlacement,
    WorkspaceMetadata,
    WorkspaceStatistics,
)


@pytest.fixture
def sample_workspace(tmp_path: Path):
    valid_hash = "a" * 64
    f_bg = tmp_path / "bg.png"
    f_bg.write_bytes(b"bg")
    f_face = tmp_path / "face.png"
    f_face.write_bytes(b"face")
    f_mask = tmp_path / "mask.png"
    f_mask.write_bytes(b"mask")
    f_depth = tmp_path / "depth.png"
    f_depth.write_bytes(b"depth")
    mask_checksum = "m" * 64
    f_mask_bytes = b"mask"
    import hashlib
    mask_checksum = hashlib.sha256(f_mask_bytes).hexdigest()

    l_bg = CompositionLayer(
        layer_id="l_bg",
        placement=AssetPlacement(
            asset_id="bg",
            role=LayerRole.BACKGROUND,
            decision=LayerDecision.REPLACE,
            source_path=str(f_bg),
            transform=LayerTransform(),
            z_index=0,
        ),
        depth_hint_path=str(f_depth),
    )

    l_person = CompositionLayer(
        layer_id="l_person",
        placement=AssetPlacement(
            asset_id="person",
            role=LayerRole.PERSON,
            decision=LayerDecision.KEEP,
            source_path=str(f_face),
            mask=MaskReference(mask_path=str(f_mask), mask_checksum=mask_checksum),
            transform=LayerTransform(),
            z_index=10,
        ),
    )

    l_remove = CompositionLayer(
        layer_id="l_remove",
        placement=AssetPlacement(
            asset_id="removed_obj",
            role=LayerRole.OBJECT,
            decision=LayerDecision.REMOVE,
            source_path=str(tmp_path / "removed.png"),
            transform=LayerTransform(),
            z_index=20,
        ),
    )

    return CompositionWorkspace(
        video_id="test_vid",
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        layers=[l_bg, l_person, l_remove],
        groups=[
            LayerGroup(group_id="g_bg", role=LayerRole.BACKGROUND, layer_ids=["l_bg"]),
            LayerGroup(group_id="g_person", role=LayerRole.PERSON, layer_ids=["l_person"]),
        ],
        text_placement=TextPlacement(),
        lighting=LightingAdjustment(
            target_brightness=0.5,
            target_contrast=0.5,
            target_saturation=0.5,
            warm_or_cool="neutral",
        ),
        constraints=PlacementConstraints(),
        statistics=WorkspaceStatistics(total_layers=3, replaced=1, kept=1, removed=1),
        metadata=WorkspaceMetadata(
            video_id="test_vid",
            created_at="2026-07-30T00:00:00Z",
            vre_source_hash=valid_hash,
            redesign_spec_hash=valid_hash,
            prompt_package_hash=valid_hash,
            engine_version="1.0.0",
        ),
    )


def test_generation_bundle_builder_success(sample_workspace):
    builder = GenerationBundleBuilder()
    bundle = builder.build_generation_bundle(sample_workspace)

    assert bundle.video_id == "test_vid"
    assert "background" in bundle.reference_image_paths
    assert "person" in bundle.reference_image_paths
    # Removed object should NOT be in reference_image_paths
    assert "object" not in bundle.reference_image_paths
    assert "person" in bundle.mask_paths
    assert bundle.depth_path is not None
    assert bundle.layer_order == ["l_bg", "l_person", "l_remove"]
    assert len(bundle.workspace_hash) == 64


def test_generation_bundle_builder_error_workspace(sample_workspace):
    err_workspace = sample_workspace.model_copy(
        update={"status": "error", "error_message": "Invalid upstream data"}
    )
    builder = GenerationBundleBuilder()
    with pytest.raises(GenerationBundleError):
        builder.build_generation_bundle(err_workspace)
