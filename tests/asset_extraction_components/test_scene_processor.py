"""
test_scene_processor.py
=======================

Unit tests for SceneProcessor (Phase 5 Scene family).
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from modules.asset_extraction_components.scene_processor import SceneProcessor


def test_scene_processor_empty():
    processor = SceneProcessor()
    assert processor.process(None) == {}


def test_scene_processor_synthetic():
    mock_bridge = MagicMock()
    mock_bridge.run.side_effect = [
        # BiRefNet fg/bg matting
        (np.full((120, 160, 3), 200, dtype=np.uint8), np.full((120, 160, 3), 50, dtype=np.uint8)),
        # DepthAnything depth map
        np.full((120, 160), 128, dtype=np.uint8),
    ]

    processor = SceneProcessor(model_bridge=mock_bridge)
    image = np.full((120, 160, 3), 100, dtype=np.uint8)

    results = processor.process(image)

    assert "foreground" in results
    assert "background" in results
    assert "depth_map" in results
    assert "segmentation_map" in results
    assert "sky_mask" in results
    assert "ground_mask" in results

    assert results["foreground"].shape == (120, 160, 3)
    assert results["background"].shape == (120, 160, 3)
    assert results["depth_map"].shape == (120, 160)
    assert results["segmentation_map"].shape == (120, 160)
    assert results["sky_mask"].shape == (120, 160)
    assert results["ground_mask"].shape == (120, 160)
