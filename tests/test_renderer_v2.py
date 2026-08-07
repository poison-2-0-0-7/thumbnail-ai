"""
Unit Tests for Rendering Engine V2.1 Pipeline
"""

import numpy as np
import pytest

from renderer.core.schema import (
    EditPlan,
    LayerSpec,
    LayerType,
    LayerAction,
    RelightingSpec,
    TypographySpec,
)
from renderer.core.canvas import Canvas, Layer
from renderer.engine import RenderingEngineV2


def test_canvas_compositing():
    canvas = Canvas(width=1280, height=720)
    
    # Layer 1: Red Background
    red_rgba = np.zeros((720, 1280, 4), dtype=np.uint8)
    red_rgba[:, :, 0] = 255
    red_rgba[:, :, 3] = 255
    red_mask = np.full((720, 1280), 255, dtype=np.uint8)
    layer1 = Layer("bg", LayerType.BACKGROUND, red_rgba, red_mask, z_index=0, bounding_box=(0,0,1280,720))
    
    canvas.add_layer(layer1)
    composite = canvas.composite_rgba()
    
    assert composite.shape == (720, 1280, 3)
    assert np.all(composite[:, :, 0] == 255)
    assert np.all(composite[:, :, 1] == 0)


def test_rendering_engine_v2_pipeline():
    engine = RenderingEngineV2()
    orig_thumb = np.zeros((720, 1280, 3), dtype=np.uint8)
    orig_thumb[:, :, 0] = 50  # Dark red base

    edit_plan = EditPlan(
        plan_id="test_plan_001",
        timestamp="2026-08-07T11:00:00Z",
        layers=[
            LayerSpec(
                layer_id="layer_bg",
                layer_type=LayerType.BACKGROUND,
                z_index=0,
                action=LayerAction.GENERATIVE_REPLACE,
            ),
            LayerSpec(
                layer_id="layer_text_primary",
                layer_type=LayerType.TYPOGRAPHY,
                z_index=10,
                action=LayerAction.RENDER_VECTOR_TEXT,
                typography_spec=TypographySpec(text_content="TEST REDESIGN"),
            ),
        ],
    )

    rendered_img, report = engine.render(orig_thumb, edit_plan)
    assert rendered_img.shape == (720, 1280, 3)
    assert isinstance(report.passed, bool)
