"""Tests for Module 10 StyleProfileStore component."""

from __future__ import annotations

from pathlib import Path
import pytest
from modules.creator_style.profile_store import StyleProfileStore
from modules.models import ThumbnailStyleSignature


@pytest.fixture
def temp_store(tmp_path: Path) -> StyleProfileStore:
    return StyleProfileStore(base_dir=tmp_path)


def test_profile_store_incremental_centroid_update(temp_store: StyleProfileStore):
    channel_id = "UC_TEST_CHANNEL"
    sig1 = ThumbnailStyleSignature(
        video_id="v1",
        channel_id=channel_id,
        dominant_colors=["#123456"],
    )

    vec1 = [1.0, 0.0, 0.0, 0.0]
    manifest1, emb1 = temp_store.update_profile("v1", channel_id, sig1, vec1, min_samples=2)

    assert manifest1.sample_count == 1
    assert manifest1.profile_established is False
    assert emb1.embedding == [1.0, 0.0, 0.0, 0.0]

    # Add second vector: [0.0, 1.0, 0.0, 0.0]
    # Running mean = [0.5, 0.5, 0.0, 0.0]
    sig2 = ThumbnailStyleSignature(
        video_id="v2",
        channel_id=channel_id,
        dominant_colors=["#654321"],
    )
    vec2 = [0.0, 1.0, 0.0, 0.0]
    manifest2, emb2 = temp_store.update_profile("v2", channel_id, sig2, vec2, min_samples=2)

    assert manifest2.sample_count == 2
    assert manifest2.profile_established is True
    assert pytest.approx(emb2.embedding[0], 0.01) == 0.5
    assert pytest.approx(emb2.embedding[1], 0.01) == 0.5


def test_profile_store_centroid_reset(temp_store: StyleProfileStore):
    channel_id = "UC_RESET_CHANNEL"
    seeds = ["v1", "v2"]
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    new_emb = temp_store.reset_centroid(channel_id, seeds, vectors)
    assert new_emb.sample_count == 2
    assert pytest.approx(new_emb.embedding[0], 0.01) == 0.5
    assert pytest.approx(new_emb.embedding[1], 0.01) == 0.5
