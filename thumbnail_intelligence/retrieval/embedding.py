"""
embedding.py
============

Abstract embedding search interface and in-memory vector indexing engine.
Defines:
- The EmbeddingProvider protocol (pluggable for future BGE, E5, Jina, NV-Embed, OpenCLIP)
- Pure vector similarity mathematics (cosine similarity, normalization, dot product)
- Fast in-memory linear cosine similarity index over candidate knowledge entries
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

from thumbnail_intelligence.retrieval.exceptions import (
    EmbeddingDimensionError,
    ProviderNotFoundError,
)


class VectorMath:
    """Mathematical utilities for vector normalization, dot product, and cosine similarity."""

    @staticmethod
    def l2_norm(vec: List[float]) -> float:
        """Compute the Euclidean L2 norm of a vector."""
        if not vec:
            return 0.0
        return math.sqrt(sum(x * x for x in vec))

    @staticmethod
    def normalize(vec: List[float]) -> List[float]:
        """Normalize vector to unit length (L2 norm = 1.0)."""
        norm = VectorMath.l2_norm(vec)
        if norm == 0.0:
            return [0.0] * len(vec)
        return [x / norm for x in vec]

    @staticmethod
    def dot_product(u: List[float], v: List[float]) -> float:
        """Compute dot product between two equal-dimension vectors."""
        if len(u) != len(v):
            raise EmbeddingDimensionError(
                message=f"Vector length mismatch: {len(u)} != {len(v)}",
                context={"len_u": len(u), "len_v": len(v)},
            )
        return sum(a * b for a, b in zip(u, v))

    @staticmethod
    def cosine_similarity(u: List[float], v: List[float]) -> float:
        """
        Compute cosine similarity between two float vectors.
        Returns value in [-1.0, 1.0], bounded to [0.0, 1.0] for non-negative feature spaces.
        """
        if not u or not v:
            return 0.0
        if len(u) != len(v):
            raise EmbeddingDimensionError(
                message=f"Cosine similarity dimension mismatch: {len(u)} != {len(v)}",
                context={"dim_u": len(u), "dim_v": len(v)},
            )
        norm_u = VectorMath.l2_norm(u)
        norm_v = VectorMath.l2_norm(v)
        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0
        dot = sum(a * b for a, b in zip(u, v))
        sim = dot / (norm_u * norm_v)
        # Numerical guard for float precision
        return max(-1.0, min(1.0, sim))


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Abstract embedding provider protocol.
    Future models (BGE, E5, Jina, NV-Embed, OpenCLIP) implement this interface
    without modifying retrieval engine code.
    """

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension (e.g. 512, 768, 1024, 4096)."""
        ...

    @property
    def provider_name(self) -> str:
        """Return human-readable identifier of the embedding model."""
        ...

    def encode_text(self, text: str) -> List[float]:
        """Encode arbitrary text into a normalized embedding vector."""
        ...

    def encode_image(self, image_input: Any) -> List[float]:
        """Encode image input into a normalized visual embedding vector."""
        ...


class MockEmbeddingProvider:
    """
    In-memory deterministic test provider generating synthetic embedding vectors.
    """

    def __init__(self, dim: int = 512, name: str = "Mock-Embedding-ViT-512") -> None:
        self._dim = dim
        self._name = name

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return self._name

    def encode_text(self, text: str) -> List[float]:
        """Generate deterministic pseudo-embedding from text hash."""
        val = sum(ord(c) for c in text) % 1000 / 1000.0
        vec = [val + (i * 0.001) for i in range(self._dim)]
        return VectorMath.normalize(vec)

    def encode_image(self, image_input: Any) -> List[float]:
        """Generate deterministic pseudo-embedding for image input."""
        vec = [0.1 + (i * 0.001) for i in range(self._dim)]
        return VectorMath.normalize(vec)


class InMemoryVectorIndex:
    """
    Fast in-memory linear cosine similarity vector index.
    Maintains normalized vectors and supports candidate filtering prior to vector similarity scoring.
    """

    def __init__(self, expected_dim: int = 512) -> None:
        self.expected_dim = expected_dim
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add(self, entry_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add or replace a vector in the index.
        Validates dimensions and normalizes the vector.
        """
        if not entry_id or not str(entry_id).strip():
            return
        if not vector or len(vector) != self.expected_dim:
            raise EmbeddingDimensionError(
                message=f"Vector dimension {len(vector)} does not match index dimension {self.expected_dim}",
                context={"entry_id": entry_id, "actual_dim": len(vector), "expected_dim": self.expected_dim},
            )
        safe_id = str(entry_id).strip()
        normalized = VectorMath.normalize(vector)
        self._vectors[safe_id] = normalized
        self._metadata[safe_id] = metadata or {}

    def remove(self, entry_id: str) -> bool:
        """Remove a vector from the index."""
        safe_id = str(entry_id).strip()
        removed = False
        if safe_id in self._vectors:
            del self._vectors[safe_id]
            removed = True
        if safe_id in self._metadata:
            del self._metadata[safe_id]
        return removed

    def get(self, entry_id: str) -> Optional[List[float]]:
        """Retrieve stored normalized vector for entry_id."""
        return self._vectors.get(str(entry_id).strip())

    def search(
        self,
        query_vector: List[float],
        top_k: int = 8,
        min_score: float = 0.0,
        candidate_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Perform cosine similarity search over stored vectors.
        If candidate_ids is provided, restricts vector comparison strictly to candidates.
        Returns sorted list of (entry_id, similarity_score) descending.
        """
        if not query_vector:
            return []
        if len(query_vector) != self.expected_dim:
            raise EmbeddingDimensionError(
                message=f"Query vector dimension {len(query_vector)} != {self.expected_dim}",
                context={"query_dim": len(query_vector), "expected_dim": self.expected_dim},
            )

        norm_query = VectorMath.normalize(query_vector)
        scores: List[Tuple[str, float]] = []

        # Determine evaluation target pool
        target_ids = candidate_ids if candidate_ids is not None else self._vectors.keys()

        for eid in target_ids:
            vec = self._vectors.get(eid)
            if vec is None:
                continue
            # Since both vectors are unit normalized, dot product == cosine similarity
            sim = VectorMath.dot_product(norm_query, vec)
            if sim >= min_score:
                scores.append((eid, sim))

        # Sort descending by similarity
        scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None and top_k > 0:
            scores = scores[:top_k]

        return scores

    def count(self) -> int:
        """Return total indexed vector count."""
        return len(self._vectors)

    def clear(self) -> None:
        """Clear all indexed vectors and metadata."""
        self._vectors.clear()
        self._metadata.clear()
