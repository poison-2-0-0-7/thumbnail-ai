"""Tests for Module 10 StyleSimilarityEngine component."""

from __future__ import annotations

import pytest
from modules.creator_style.style_similarity import StyleSimilarityEngine
from modules.models import CreatorStyleEmbedding


def test_style_similarity_unestablished_profile_cold_start():
    engine = StyleSimilarityEngine()
    res = engine.evaluate_similarity(
        video_id="v1",
        channel_id="UC123",
        candidate_image=[1.0, 0.0, 0.0],
        profile_embedding=None,
        min_samples=3,
    )

    assert res.profile_established is False
    assert res.belongs_to_identity is True
    assert res.similarity_score == 1.0


def test_style_similarity_established_profile_evaluation():
    engine = StyleSimilarityEngine()
    profile_emb = CreatorStyleEmbedding(
        channel_id="UC123",
        embedding=[1.0, 0.0, 0.0],
        sample_count=3,
    )

    # Candidate 1: Identical vector -> high similarity
    res1 = engine.evaluate_similarity(
        video_id="v1",
        channel_id="UC123",
        candidate_image=[1.0, 0.0, 0.0],
        profile_embedding=profile_emb,
        min_samples=3,
        similarity_threshold=0.75,
    )
    assert res1.profile_established is True
    assert res1.belongs_to_identity is True
    assert pytest.approx(res1.similarity_score, 0.01) == 1.0

    # Candidate 2: Orthogonal vector -> low similarity
    res2 = engine.evaluate_similarity(
        video_id="v2",
        channel_id="UC123",
        candidate_image=[0.0, 1.0, 0.0],
        profile_embedding=profile_emb,
        min_samples=3,
        similarity_threshold=0.75,
    )
    assert res2.profile_established is True
    assert res2.belongs_to_identity is False
    assert pytest.approx(res2.similarity_score, 0.01) == 0.0
