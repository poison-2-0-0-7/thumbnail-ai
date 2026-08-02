from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from generation_components.staged_edit_stages import BaseLatentStage, MaskedCompositeStage


def _create_img(path: Path, color: str = "blue", width: int = 100, height: int = 100) -> Path:
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)
    return path


def test_base_latent_stage_prepares_anchor(tmp_path: Path) -> None:
    src_path = _create_img(tmp_path / "src.jpg", color="blue", width=200, height=100)
    stage = BaseLatentStage()
    anchor = stage.prepare(src_path)

    assert anchor.width == 200
    assert anchor.height == 100
    assert anchor.source_array.shape == (100, 200, 3)


def test_masked_composite_stage_byte_identical_when_zero_masks(tmp_path: Path) -> None:
    src_path = _create_img(tmp_path / "src.png", color="blue", width=100, height=100)
    gen_path = _create_img(tmp_path / "gen.png", color="red", width=100, height=100)
    out_path = tmp_path / "out.png"

    stage = MaskedCompositeStage()
    res = stage.composite(src_path, gen_path, sampled_mask_paths=[], output_path=out_path)

    assert res.is_file()
    with Image.open(res) as img:
        arr = np.array(img.convert("RGB"))
        # Must be blue (source color), zero red
        assert np.all(arr[:, :, 2] == 255)
        assert np.all(arr[:, :, 0] == 0)


def test_masked_composite_stage_composites_mask_region(tmp_path: Path) -> None:
    src_path = _create_img(tmp_path / "src.png", color="blue", width=100, height=100)
    gen_path = _create_img(tmp_path / "gen.png", color="red", width=100, height=100)
    mask_path = tmp_path / "mask.png"

    # Mask: left half is 255 (sampled/generated), right half is 0 (source)
    m_img = Image.new("L", (100, 100), color=0)
    draw = ImageDraw.Draw(m_img)
    draw.rectangle([0, 0, 49, 99], fill=255)
    m_img.save(mask_path)

    out_path = tmp_path / "out.png"
    stage = MaskedCompositeStage()
    res = stage.composite(src_path, gen_path, sampled_mask_paths=[mask_path], output_path=out_path)

    with Image.open(res) as img:
        arr = np.array(img.convert("RGB"))
        # Left half (x=0..49) must be red (generated)
        assert np.all(arr[:, 0:50, 0] == 255)
        assert np.all(arr[:, 0:50, 2] == 0)
        # Right half (x=50..99) must be blue (source)
        assert np.all(arr[:, 50:100, 2] == 255)
        assert np.all(arr[:, 50:100, 0] == 0)
