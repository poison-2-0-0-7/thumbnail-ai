"""
test_conditioning_manifest_builder.py
=====================================

Unit tests for ConditioningManifestBuilder in planner_components.
"""

from pathlib import Path

from generation_components.conditioning_asset_resolver import GenerationConditioningContext
from planner_components.conditioning_manifest_builder import ConditioningManifestBuilder


def test_conditioning_manifest_builder_basic(tmp_path: Path):
    f_bg = tmp_path / "data" / "visual_references" / "vid1" / "bg.png"
    f_bg.parent.mkdir(parents=True, exist_ok=True)
    f_bg.write_bytes(b"bg")

    f_depth = tmp_path / "data" / "visual_references" / "vid1" / "depth.png"
    f_depth.write_bytes(b"depth")

    ctx = GenerationConditioningContext(
        role_image_paths={"background": f_bg},
        depth_path=f_depth,
    )

    builder = ConditioningManifestBuilder()
    assets = builder.build_manifest(ctx)

    assert len(assets) == 2
    bg_asset = next(a for a in assets if a.role == "background")
    assert bg_asset.kind == "reference_image"
    assert bg_asset.source_module == "vre"

    depth_asset = next(a for a in assets if a.role == "depth_map")
    assert depth_asset.kind == "depth"
    assert depth_asset.source_module == "vre"
