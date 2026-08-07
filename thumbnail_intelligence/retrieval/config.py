"""
config.py
=========

Configuration models and ranking weight definitions for the Hybrid Retrieval Engine.
Controls top-k bounds, cosine similarity thresholds, recency time-decay half-life,
cache policies, and explainable multi-signal scoring weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class RankingWeights:
    """
    Relative importance weights for scoring and ranking candidate knowledge entries.
    All weights must be non-negative and sum to 1.0 for calibrated explainable scores.
    """

    visual_similarity: float = 0.35
    creator_channel_affinity: float = 0.15
    archetype_match: float = 0.15
    niche_match: float = 0.10
    recency: float = 0.10
    confidence: float = 0.10
    metadata_quality: float = 0.05

    def __post_init__(self) -> None:
        total = (
            self.visual_similarity
            + self.creator_channel_affinity
            + self.archetype_match
            + self.niche_match
            + self.recency
            + self.confidence
            + self.metadata_quality
        )
        if not (0.99 <= total <= 1.01):
            # Normalize automatically if minor floating point drift occurs
            norm = total if total > 0 else 1.0
            object.__setattr__(self, "visual_similarity", self.visual_similarity / norm)
            object.__setattr__(self, "creator_channel_affinity", self.creator_channel_affinity / norm)
            object.__setattr__(self, "archetype_match", self.archetype_match / norm)
            object.__setattr__(self, "niche_match", self.niche_match / norm)
            object.__setattr__(self, "recency", self.recency / norm)
            object.__setattr__(self, "confidence", self.confidence / norm)
            object.__setattr__(self, "metadata_quality", self.metadata_quality / norm)

    def to_dict(self) -> Dict[str, float]:
        return {
            "visual_similarity": self.visual_similarity,
            "creator_channel_affinity": self.creator_channel_affinity,
            "archetype_match": self.archetype_match,
            "niche_match": self.niche_match,
            "recency": self.recency,
            "confidence": self.confidence,
            "metadata_quality": self.metadata_quality,
        }


@dataclass
class RetrievalConfig:
    """
    Master configuration for the Hybrid Retrieval Engine.
    Configures query limits, thresholds, caching policies, and ranking behavior.
    """

    default_top_k: int = 8
    min_similarity_threshold: float = 0.0
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_size: int = 1000
    embedding_dim: int = 512
    deduplication_threshold: float = 0.95
    recency_half_life_days: float = 90.0
    weights: RankingWeights = field(default_factory=RankingWeights)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_top_k": self.default_top_k,
            "min_similarity_threshold": self.min_similarity_threshold,
            "cache_enabled": self.cache_enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_cache_size": self.max_cache_size,
            "embedding_dim": self.embedding_dim,
            "deduplication_threshold": self.deduplication_threshold,
            "recency_half_life_days": self.recency_half_life_days,
            "weights": self.weights.to_dict(),
        }
