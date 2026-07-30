"""
test_object_processor.py
========================

Unit tests for ObjectProcessor (Phase 3 Object family).
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from modules.asset_extraction_components.object_processor import ObjectProcessor
from modules.models import BoundingBox, DetectedObject


def test_object_processor_empty():
    processor = ObjectProcessor()
    assert processor.process(None, []) == []


def test_object_processor_synthetic_with_hierarchy():
    # Mock bridge that simulates SAM2 returning a mask
    mock_bridge = MagicMock()
    mask_result = np.full((200, 200), 255, dtype=np.uint8)
    mock_bridge.run.return_value = (mask_result, 0.92)

    processor = ObjectProcessor(model_bridge=mock_bridge)
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Parent object (large) and child object (small, inside parent)
    parent_obj = DetectedObject(
        label="laptop",
        confidence=0.9,
        bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9),
    )
    child_obj = DetectedObject(
        label="sticker",
        confidence=0.85,
        bbox=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.4, y_max=0.4),
    )

    results = processor.process(image, [parent_obj, child_obj])
    assert len(results) == 2

    # Verify hierarchy
    parent_res = results[0]
    child_res = results[1]

    assert child_res["parent_object_index"] == 0
    assert 1 in parent_res["child_object_indices"]
    assert parent_res["crop"].shape[2] == 3
    assert parent_res["mask"].shape == (200, 200)
