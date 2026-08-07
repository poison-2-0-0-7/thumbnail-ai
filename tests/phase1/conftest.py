"""Pytest fixtures for Phase 1 tests."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from renderer_v2.phase1.config import Phase1Config
from renderer_v2.phase1.schemas import Instance, SceneGraph
from renderer_v2.phase1.model_registry import ModelRegistry


@pytest.fixture
def test_config(tmp_path: Path) -> Phase1Config:
    """Fixture providing isolated Phase1Config."""
    return Phase1Config(
        max_vram_gb=8.0,
        debug_dir=tmp_path / "debug",
        models_cache_dir=tmp_path / "models_cache",
    )


@pytest.fixture
def model_registry(test_config: Phase1Config) -> ModelRegistry:
    """Fixture providing clean ModelRegistry."""
    return ModelRegistry(test_config)


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Fixture providing 640x360 RGB test image with background gradient and central box."""
    h, w = 360, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Background gradient
    y, x = np.ogrid[:h, :w]
    img[:, :, 0] = (x / w * 255).astype(np.uint8)
    img[:, :, 1] = (y / h * 255).astype(np.uint8)
    img[:, :, 2] = 200

    # Central talking head person box
    cy, cx = h // 2, w // 2
    rh, rw = int(h * 0.35), int(w * 0.25)
    img[cy - rh : cy + rh, cx - rw : cx + rw] = [240, 180, 140]

    # Logo box in top-right
    img[20:80, w - 80 : w - 20] = [255, 0, 0]

    return img


@pytest.fixture
def sample_scene_graph(sample_rgb_image: np.ndarray) -> SceneGraph:
    """Fixture providing valid SceneGraph with creator and logo instances."""
    h, w, _ = sample_rgb_image.shape
    
    # Creator mask
    creator_mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    rh, rw = int(h * 0.35), int(w * 0.25)
    creator_mask[cy - rh : cy + rh, cx - rw : cx + rw] = True
    creator_alpha = creator_mask.astype(np.float32)

    # Logo mask
    logo_mask = np.zeros((h, w), dtype=bool)
    logo_mask[20:80, w - 80 : w - 20] = True
    logo_alpha = logo_mask.astype(np.float32)

    inst_creator = Instance(
        instance_id="creator_0",
        cls="creator",
        mask=creator_mask,
        alpha_matte=creator_alpha,
        bbox=(cx - rw, cy - rh, cx + rw, cy + rh),
        depth_layer=0.3,
        locked=True,
    )

    inst_logo = Instance(
        instance_id="logo_0",
        cls="logo",
        mask=logo_mask,
        alpha_matte=logo_alpha,
        bbox=(w - 80, 20, w - 20, 80),
        depth_layer=0.4,
        locked=True,
    )

    depth_map = np.full((h, w), 0.8, dtype=np.float32)
    depth_map[creator_mask] = 0.3
    depth_map[logo_mask] = 0.4

    return SceneGraph(
        source_image=sample_rgb_image,
        instances=[inst_creator, inst_logo],
        depth_map=depth_map,
        width=w,
        height=h,
    )


@pytest.fixture(autouse=True)
def setup_fixtures_dir(tmp_path: Path):
    """Ensure tests/phase1/fixtures directory exists with sample image."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    sample_path = fixtures_dir / "sample_talking_head.png"
    if not sample_path.exists():
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[90:270, 240:400] = [240, 180, 140]
        Image.fromarray(img).save(sample_path)
