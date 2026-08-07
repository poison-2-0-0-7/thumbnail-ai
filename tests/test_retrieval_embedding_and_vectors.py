"""
Unit tests for VectorMath, InMemoryVectorIndex, and EmbeddingProvider interface.
Tests cosine similarity mathematics, normalization, vector dimension enforcement,
and candidate-constrained similarity lookups.
"""

from __future__ import annotations

import math
import pytest

from thumbnail_intelligence.retrieval.embedding import (
    InMemoryVectorIndex,
    MockEmbeddingProvider,
    VectorMath,
)
from thumbnail_intelligence.retrieval.exceptions import EmbeddingDimensionError


def test_vector_math_operations() -> None:
    # 1. L2 Norm
    vec = [3.0, 4.0]
    assert VectorMath.l2_norm(vec) == 5.0
    assert VectorMath.l2_norm([]) == 0.0

    # 2. Normalization
    norm_vec = VectorMath.normalize(vec)
    assert math.isclose(norm_vec[0], 0.6)
    assert math.isclose(norm_vec[1], 0.8)
    assert math.isclose(VectorMath.l2_norm(norm_vec), 1.0)

    # 3. Dot Product & Cosine Similarity
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    v3 = [1.0, 0.0]
    v4 = [-1.0, 0.0]

    # Orthogonal vectors -> cosine sim = 0.0
    assert math.isclose(VectorMath.cosine_similarity(v1, v2), 0.0)

    # Identical vectors -> cosine sim = 1.0
    assert math.isclose(VectorMath.cosine_similarity(v1, v3), 1.0)

    # Opposite vectors -> cosine sim = -1.0
    assert math.isclose(VectorMath.cosine_similarity(v1, v4), -1.0)

    # Dimension mismatch raises EmbeddingDimensionError
    with pytest.raises(EmbeddingDimensionError):
        VectorMath.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_in_memory_vector_index_search_and_candidates() -> None:
    index = InMemoryVectorIndex(expected_dim=4)

    # Vector 1: aligned with [1, 0, 0, 0]
    index.add("doc_1", [1.0, 0.0, 0.0, 0.0], metadata={"category": "A"})
    # Vector 2: aligned with [0, 1, 0, 0]
    index.add("doc_2", [0.0, 1.0, 0.0, 0.0], metadata={"category": "B"})
    # Vector 3: aligned with [0.8, 0.6, 0.0, 0.0]
    index.add("doc_3", [0.8, 0.6, 0.0, 0.0], metadata={"category": "A"})

    assert index.count() == 3

    # Query vector close to doc_1
    query = [1.0, 0.0, 0.0, 0.0]
    results = index.search(query_vector=query, top_k=2)
    assert len(results) == 2
    # doc_1 should be first with similarity 1.0
    assert results[0][0] == "doc_1"
    assert math.isclose(results[0][1], 1.0)
    # doc_3 should be second with similarity 0.8
    assert results[1][0] == "doc_3"
    assert math.isclose(results[1][1], 0.8)

    # Search with candidate_ids restriction (e.g. only doc_2 and doc_3)
    constrained_results = index.search(
        query_vector=query,
        top_k=2,
        candidate_ids={"doc_2", "doc_3"},
    )
    assert len(constrained_results) == 2
    assert constrained_results[0][0] == "doc_3"
    assert constrained_results[1][0] == "doc_2"
    assert "doc_1" not in [r[0] for r in constrained_results]


def test_in_memory_vector_index_dimension_guards_and_removal() -> None:
    index = InMemoryVectorIndex(expected_dim=512)

    # Wrong dimension add raises EmbeddingDimensionError
    with pytest.raises(EmbeddingDimensionError):
        index.add("bad_doc", [0.1] * 256)

    # Wrong dimension search raises EmbeddingDimensionError
    with pytest.raises(EmbeddingDimensionError):
        index.search([0.1] * 128)

    # Valid add and remove
    index.add("valid_doc", [0.1] * 512)
    assert index.count() == 1
    assert index.get("valid_doc") is not None

    removed = index.remove("valid_doc")
    assert removed is True
    assert index.count() == 0
    assert index.get("valid_doc") is None


def test_mock_embedding_provider() -> None:
    provider = MockEmbeddingProvider(dim=512, name="Test-Mock-ViT")
    assert provider.dimension == 512
    assert provider.provider_name == "Test-Mock-ViT"

    text_vec = provider.encode_text("Amazing viral thumbnail title")
    assert len(text_vec) == 512
    assert math.isclose(VectorMath.l2_norm(text_vec), 1.0)

    img_vec = provider.encode_image("dummy_image_input")
    assert len(img_vec) == 512
    assert math.isclose(VectorMath.l2_norm(img_vec), 1.0)
