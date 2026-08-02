"""Tests for CandidateClusteringEngine."""

from __future__ import annotations

import pytest
from pathlib import Path
from PIL import Image
from generation_components import CandidateClusteringEngine, compute_dhash, hamming_distance
from modules.models import CandidateStrategy


@pytest.fixture
def test_images(tmp_path: Path) -> tuple[Path, Path, Path]:
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"
    img3_path = tmp_path / "img3.png"

    # Image 1: horizontal gradient 0 to 255
    img1 = Image.new("L", (100, 100))
    for y in range(100):
        for x in range(100):
            img1.putpixel((x, y), int(x * 2.55))
    img1.save(img1_path)

    # Image 2: identical to Image 1
    img2 = Image.new("L", (100, 100))
    for y in range(100):
        for x in range(100):
            img2.putpixel((x, y), int(x * 2.55))
    img2.save(img2_path)

    # Image 3: inverted horizontal gradient 255 to 0
    img3 = Image.new("L", (100, 100))
    for y in range(100):
        for x in range(100):
            img3.putpixel((x, y), int((99 - x) * 2.55))
    img3.save(img3_path)

    return img1_path, img2_path, img3_path



class MockQAReport:
    def __init__(self, score: float = 0.8, passed: bool = True):
        self.overall_score = score
        self.hard_gate_passed = passed


def test_dhash_computation(test_images):
    img1, img2, img3 = test_images
    hash1 = compute_dhash(img1)
    hash2 = compute_dhash(img2)
    hash3 = compute_dhash(img3)

    assert hash1 == hash2
    assert hamming_distance(hash1, hash2) == 0
    assert hamming_distance(hash1, hash3) > 0


def test_clustering_engine_duplicate_detection(test_images):
    img1, img2, img3 = test_images
    engine = CandidateClusteringEngine(threshold=5)

    strat = CandidateStrategy.faithful_default()
    cands = [
        (0, img1, MockQAReport(score=0.85), None, strat, None, "hash1", {}),
        (1, img2, MockQAReport(score=0.90), None, strat, None, "hash2", {}),
        (2, img3, MockQAReport(score=0.75), None, strat, None, "hash3", {}),
    ]

    res = engine.cluster_candidates(cands)

    assert len(res.clusters) == 2
    # Candidate 0 and Candidate 1 form a cluster, Candidate 1 has higher QA score (0.90 vs 0.85) -> Candidate 1 is survivor
    assert 1 in res.survivor_indices
    assert 0 in res.excluded_duplicates
    assert res.excluded_duplicates[0].startswith("duplicate_cluster_")
