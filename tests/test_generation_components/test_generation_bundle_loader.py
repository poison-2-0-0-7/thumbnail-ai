"""
Tests for GenerationBundleLoader.
"""

from pathlib import Path
import pytest
from generation_components.generation_bundle_loader import GenerationBundleLoader
from module7_exceptions import GenerationBundleInvalidError
from models import CanvasTransform, GenerationBundle


def test_generation_bundle_loader_success(tmp_path: Path):
    bundle = GenerationBundle(
        video_id="vid123",
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        reference_image_paths={"background": "/path/to/bg.png"},
        mask_paths={},
        depth_path=None,
        canny_path=None,
        layer_order=["bg"],
        workspace_hash="a" * 64,
        prompt_package_hash="b" * 64,
        generated_at="2026-01-01T00:00:00Z",
    )
    vid_dir = tmp_path / "vid123"
    vid_dir.mkdir()
    bundle_file = vid_dir / "generation_bundle.json"
    bundle_file.write_text(bundle.model_dump_json(), encoding="utf-8")

    loader = GenerationBundleLoader(root_dir=tmp_path)
    loaded = loader.load("vid123")

    assert loaded.video_id == "vid123"
    assert loaded.canvas.width == 1280
    assert loaded.reference_image_paths["background"] == "/path/to/bg.png"


def test_generation_bundle_loader_missing_file_raises(tmp_path: Path):
    loader = GenerationBundleLoader(root_dir=tmp_path)
    with pytest.raises(GenerationBundleInvalidError, match="not found"):
        loader.load("nonexistent_video")


def test_generation_bundle_loader_mismatched_video_id(tmp_path: Path):
    bundle = GenerationBundle(
        video_id="other_video",
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
    vid_dir = tmp_path / "vid123"
    vid_dir.mkdir()
    bundle_file = vid_dir / "generation_bundle.json"
    bundle_file.write_text(bundle.model_dump_json(), encoding="utf-8")

    loader = GenerationBundleLoader(root_dir=tmp_path)
    with pytest.raises(GenerationBundleInvalidError, match="mismatch"):
        loader.load("vid123")
