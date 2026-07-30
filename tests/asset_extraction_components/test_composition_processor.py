"""
test_composition_processor.py
==============================

Unit tests for CompositionProcessor (Phase 2 zero-model family).
"""

import numpy as np
import pytest

from modules.asset_extraction_components.composition_processor import CompositionProcessor
from modules.models import BoundingBox, CompositionAnalysis


def test_composition_processor_synthetic():
    processor = CompositionProcessor()
    image = np.full((120, 160, 3), 128, dtype=np.uint8)

    comp = CompositionAnalysis(
        rule_of_thirds_score=0.8,
        negative_space_ratio=0.3,
        visual_clutter_score=0.2,
        symmetry_score=0.5,
        balance_score=0.7,
        primary_subject_bbox=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6),
    )

    res = processor.process(image, comp)
    assert res["eye_flow_map"].shape == (120, 160, 3)
    assert res["negative_space_mask"].shape == (120, 160)
    assert res["visual_hierarchy_overlay"].shape == (120, 160, 3)
    assert res["source_composition_analysis"] == comp
