"""
test_person_processor.py
========================

Unit tests for PersonProcessor (Phase 4 Person family).
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from modules.asset_extraction_components.person_processor import PersonProcessor
from modules.models import BoundingBox, FaceAnalysis, FaceDetail


def test_person_processor_empty():
    processor = PersonProcessor()
    assert processor.process(None, FaceAnalysis()) == []


def test_person_processor_synthetic():
    mock_bridge = MagicMock()
    mock_bridge.run.side_effect = [
        # BiSeNet masks
        {
            "body_mask": np.full((200, 200), 255, dtype=np.uint8),
            "hair_mask": np.zeros((200, 200), dtype=np.uint8),
            "clothing_mask": np.full((200, 200), 255, dtype=np.uint8),
            "accessories_mask": np.zeros((200, 200), dtype=np.uint8),
        },
        # InsightFace multi-face features
        [
            {
                "face_index": 0,
                "crop": np.zeros((50, 50, 3), dtype=np.uint8),
                "embedding": [0.1] * 512,
                "landmarks": [(25.0, 25.0)],
            }
        ],
    ]

    processor = PersonProcessor(model_bridge=mock_bridge)
    image = np.full((200, 200, 3), 150, dtype=np.uint8)

    face_detail = FaceDetail(
        bbox=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6),
        detection_confidence=0.95,
        is_largest=True,
    )
    faces = FaceAnalysis(face_count=1, faces=[face_detail], has_face=True)

    results = processor.process(image, faces)
    assert len(results) == 1

    person = results[0]
    assert person["person_index"] == 0
    assert person["face"].shape == (80, 80, 3)
    assert person["face_mask"].shape == (200, 200)
    assert person["face_embedding"] == [0.1] * 512
    assert person["facial_landmarks"] == [(25.0, 25.0)]
    assert person["body_mask"].shape == (200, 200)
    assert len(person["pose_keypoints"]) > 0
