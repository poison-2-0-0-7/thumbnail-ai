"""Tests for Module 10 StyleDriftDetector component."""

from __future__ import annotations

import pytest
from modules.creator_style.drift_detector import StyleDriftDetector
from modules.models import CreatorStyleEmbedding


def test_drift_detector_single_outlier_does_not_trigger_drift():
    detector = StyleDriftDetector()
    profile_emb = CreatorStyleEmbedding(
        channel_id="UC123",
        embedding=[1.0, 0.0, 0.0, 0.0],
        sample_count=5,
    )

    # 3 inputs: 2 matching old profile, 1 outlier
    recent_inputs = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],  # 1 outlier
    ]

    res = detector.assess_drift(
        channel_id="UC123",
        recent_image_inputs=recent_inputs,
        profile_embedding=profile_emb,
        drift_window=3,
        similarity_threshold=0.75,
    )

    assert res.drift_detected is False
    assert res.recommended_action == "monitor"


def test_drift_detector_sustained_shift_triggers_drift():
    detector = StyleDriftDetector()
    profile_emb = CreatorStyleEmbedding(
        channel_id="UC123",
        embedding=[1.0, 0.0, 0.0, 0.0],
        sample_count=5,
    )

    # 3 consecutive inputs that diverged from old profile [1.0, 0.0, 0.0, 0.0]
    # but are mutually similar to each other [0.0, 1.0, 0.0, 0.0]
    recent_inputs = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.98, 0.02, 0.0],
        [0.0, 0.99, 0.0, 0.01],
    ]

    res = detector.assess_drift(
        channel_id="UC123",
        recent_image_inputs=recent_inputs,
        profile_embedding=profile_emb,
        drift_window=3,
        similarity_threshold=0.75,
    )

    assert res.drift_detected is True
    assert res.recommended_action == "update_centroid"
    assert res.drift_confidence == 0.85
