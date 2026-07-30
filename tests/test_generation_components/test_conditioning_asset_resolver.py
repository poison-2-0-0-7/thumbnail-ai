"""
Tests for ConditioningAssetResolver.
"""

from pathlib import Path
import pytest

from generation_components.conditioning_asset_resolver import ConditioningAssetResolver
from module7_exceptions import ConditioningResolutionError
from models import (
    AssetPlacement,
    CanvasTransform,
    CompositionLayer,
    CompositionWorkspace,
    GenerationBundle,
    LayerDecision,
    LayerRole,
    LayerTransform,
    LightingAdjustment,
    MaskReference,
    PlacementConstraints,
    TextPlacement,
    WorkspaceMetadata,
    WorkspaceStatistics,
)
from image_generator import ReferenceAssets


def test_resolver_empty_inputs():
    resolver = ConditioningAssetResolver()
    ctx = resolver.resolve()

    assert ctx.source_thumbnail_path is None
    assert ctx.canvas_width == 1280
    assert ctx.canvas_height == 720
    assert ctx.role_image_paths == {}
    assert ctx.role_mask_paths == {}
    assert ctx.depth_path is None
    assert ctx.canny_path is None
    assert ctx.segmentation_path is None
    assert ctx.ip_adapter_reference_paths == {}
    assert ctx.text_exclusion_mask_path is None
    assert ctx.layer_order == ()
    assert ctx.per_layer is None


def test_resolver_with_existing_files(tmp_path: Path):
    bg_file = tmp_path / "bg.png"
    bg_file.write_bytes(b"fake png")
    mask_file = tmp_path / "mask.png"
    mask_file.write_bytes(b"fake mask")
    depth_file = tmp_path / "depth.png"
    depth_file.write_bytes(b"fake depth")
    thumb_file = tmp_path / "thumb.jpg"
    thumb_file.write_bytes(b"fake thumb")

    bundle = GenerationBundle(
        video_id="vid123",
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        reference_image_paths={"background": str(bg_file)},
        mask_paths={"person": str(mask_file)},
        depth_path=str(depth_file),
        canny_path=None,
        layer_order=["background", "person"],
        workspace_hash="a" * 64,
        prompt_package_hash="b" * 64,
        generated_at="2026-01-01T00:00:00Z",
    )
    ref_assets = ReferenceAssets(source_thumbnail_path=thumb_file)

    resolver = ConditioningAssetResolver()
    ctx = resolver.resolve(bundle=bundle, reference_assets=ref_assets)

    assert ctx.source_thumbnail_path == thumb_file
    assert ctx.role_image_paths["background"] == bg_file
    assert ctx.role_mask_paths["person"] == mask_file
    assert ctx.depth_path == depth_file
    assert ctx.canny_path is None
    assert ctx.layer_order == ("background", "person")


def test_resolver_defensive_getattr(tmp_path: Path):
    # Simulates GenerationBundle without new-schema attributes
    bundle = GenerationBundle(
        video_id="vid123",
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        reference_image_paths={},
        mask_paths={},
        depth_path=None,
        canny_path=None,
        layer_order=[],
        workspace_hash="a" * 64,
        prompt_package_hash="b" * 64,
        generated_at="2026-01-01T00:00:00Z",
    )
    resolver = ConditioningAssetResolver()
    ctx = resolver.resolve(bundle=bundle)

    assert ctx.segmentation_path is None
    assert ctx.ip_adapter_reference_paths == {}
    assert ctx.text_exclusion_mask_path is None


def test_resolver_workspace_only_derives_bundle(tmp_path: Path):
    bg_file = tmp_path / "bg.png"
    bg_file.write_bytes(b"fake png")
    mask_file = tmp_path / "mask.png"
    mask_file.write_bytes(b"fake mask")

    layer = CompositionLayer(
        layer_id="layer_bg",
        placement=AssetPlacement(
            asset_id="asset_bg",
            role=LayerRole.BACKGROUND,
            decision=LayerDecision.KEEP,
            source_path=str(bg_file),
            mask=MaskReference(
                mask_path=str(mask_file),
                mask_checksum="a" * 64,
                feather_px=4,
            ),
            transform=LayerTransform(),
            z_index=1,
        ),
    )
    ws = CompositionWorkspace(
        video_id="vid123",
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
            video_id="vid123",
            created_at="2026-01-01T00:00:00Z",
            vre_source_hash="a" * 64,
            redesign_spec_hash="b" * 64,
            prompt_package_hash="c" * 64,
            engine_version="1.0.0",
        ),
    )

    resolver = ConditioningAssetResolver()
    ctx = resolver.resolve(workspace=ws)

    assert ctx.role_image_paths["background"] == bg_file
    assert ctx.role_mask_paths["background"] == mask_file
    assert ctx.per_layer is not None
    assert "layer_bg" in ctx.per_layer
    assert ctx.per_layer["layer_bg"].feather_px == 4


def test_resolver_missing_referenced_file_raises(tmp_path: Path):
    nonexistent = tmp_path / "nonexistent.png"
    bundle = GenerationBundle(
        video_id="vid123",
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        reference_image_paths={"background": str(nonexistent)},
        mask_paths={},
        depth_path=None,
        canny_path=None,
        layer_order=[],
        workspace_hash="a" * 64,
        prompt_package_hash="b" * 64,
        generated_at="2026-01-01T00:00:00Z",
    )
    resolver = ConditioningAssetResolver()
    with pytest.raises(ConditioningResolutionError, match="does not exist on disk"):
        resolver.resolve(bundle=bundle)
