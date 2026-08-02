"""Tests for Module 10 StyleAwareRankingEngine component."""

from __future__ import annotations

import pytest
from modules.creator_style.style_aware_ranking import StyleAwareRankingEngine
from modules.models import CreatorStyleEmbedding


def test_style_aware_ranking_bonus_calculation():
    engine = StyleAwareRankingEngine()
    profile_emb = CreatorStyleEmbedding(
        channel_id="UC123",
        embedding=[1.0, 0.0, 0.0],
        sample_count=3,
    )

    candidates = [
        (0, [1.0, 0.0, 0.0]),  # similarity 1.0 -> bonus (1.0 - 0.75) * 0.15 = 0.0375
        (1, [0.0, 1.0, 0.0]),  # similarity 0.0 -> bonus 0.0
    ]

    scores = engine.evaluate_candidates(
        video_id="v1",
        channel_id="UC123",
        candidate_images=candidates,
        profile_embedding=profile_emb,
        similarity_threshold=0.75,
        style_weight=0.15,
    )

    assert len(scores) == 2
    assert scores[0].candidate_index == 0
    assert pytest.approx(scores[0].style_similarity, 0.01) == 1.0
    assert pytest.approx(scores[0].style_bonus, 0.001) == 0.0375

    assert scores[1].candidate_index == 1
    assert pytest.approx(scores[1].style_similarity, 0.01) == 0.0
    assert scores[1].style_bonus == 0.0
