"""
ranking.py
==========

Ranking and deduplication engine for the Hybrid Retrieval Engine.
Applies:
- Explainable sorting by composite RetrievalScore
- Semantic and ID deduplication with score priority
- Top-K bounding and ranking metadata generation
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from thumbnail_intelligence.knowledge_base.models import BaseKBModel
from thumbnail_intelligence.retrieval.embedding import VectorMath
from thumbnail_intelligence.retrieval.exceptions import DeduplicationError, RankingError
from thumbnail_intelligence.retrieval.query import RetrievalQuery
from thumbnail_intelligence.retrieval.scoring import RetrievalScore


class RankingMetadata(BaseKBModel):
    """
    Metadata recording an entry's assigned rank, score breakdown, and retrieval pathway.
    """

    rank: int = Field(ge=1, description="1-based assigned position in final retrieved list")
    score: RetrievalScore = Field(description="Full explainable score breakdown")
    retrieval_stage: str = Field(default="hybrid", description="Pipeline stage that retrieved the entry")
    matched_terms: List[str] = Field(default_factory=list, description="Keywords or facets matched")


class EvidenceDeduplicator:
    """
    Removes duplicate and near-duplicate evidence items while preserving the highest-scoring candidate.
    Combines exact identifier matching and high cosine similarity vector thresholding.
    """

    @staticmethod
    def deduplicate(
        candidates: List[Tuple[Any, RetrievalScore, str]],
        threshold: float = 0.95,
    ) -> List[Tuple[Any, RetrievalScore, str]]:
        """
        Deduplicate a scored list of (entry, score, stage) tuples.
        If two candidates have the same ID or vector similarity >= threshold, keeps the higher scoring one.
        """
        if not candidates:
            return []

        # Sort first by overall score descending to prioritize best candidates
        sorted_candidates = sorted(candidates, key=lambda c: c[1].overall_score, reverse=True)

        unique_items: List[Tuple[Any, RetrievalScore, str]] = []
        seen_ids: Set[str] = set()
        retained_vectors: List[List[float]] = []

        for entry, score, stage in sorted_candidates:
            entry_id = getattr(
                entry,
                "entry_id",
                getattr(
                    entry,
                    "archetype_id",
                    getattr(
                        entry,
                        "pattern_id",
                        getattr(entry, "creator_id", getattr(entry, "channel_id", getattr(entry, "competitor_id", None))),
                    ),
                ),
            )

            # Check 1: Exact ID match
            if entry_id and entry_id in seen_ids:
                continue

            # Check 2: Vector semantic similarity threshold
            vec = getattr(entry, "embedding", getattr(entry, "centroid_embedding", getattr(entry, "style_embedding", None)))
            is_semantic_duplicate = False

            if vec and len(vec) > 0 and threshold < 1.0:
                for existing_vec in retained_vectors:
                    if len(existing_vec) == len(vec):
                        try:
                            sim = VectorMath.cosine_similarity(vec, existing_vec)
                            if sim >= threshold:
                                is_semantic_duplicate = True
                                break
                        except Exception:
                            pass

            if is_semantic_duplicate:
                continue

            # Retain this unique candidate
            if entry_id:
                seen_ids.add(entry_id)
            if vec and len(vec) > 0:
                retained_vectors.append(vec)
            unique_items.append((entry, score, stage))

        return unique_items


class HybridRanker:
    """
    Ranks, deduplicates, and caps candidate entries according to RetrievalQuery constraints.
    """

    def __init__(self, deduplicator: Optional[EvidenceDeduplicator] = None) -> None:
        self.deduplicator = deduplicator or EvidenceDeduplicator()

    def rank(
        self,
        candidates: List[Tuple[Any, RetrievalScore, str]],
        query: RetrievalQuery,
        deduplication_threshold: float = 0.95,
    ) -> List[Tuple[Any, RetrievalScore, RankingMetadata]]:
        """
        Rank scored candidates, apply deduplication if enabled, and enforce query.top_k limit.
        Returns ordered list of (entry, score, ranking_metadata).
        """
        if not candidates:
            return []

        # 1. Deduplicate if enabled in query
        items = candidates
        if query.deduplicate:
            items = self.deduplicator.deduplicate(candidates, threshold=deduplication_threshold)

        # 2. Filter out entries below min_similarity
        if query.min_similarity > 0.0:
            items = [c for c in items if c[1].visual_similarity >= query.min_similarity or c[1].overall_score >= query.min_similarity]

        # 3. Sort by overall_score descending
        items.sort(key=lambda c: c[1].overall_score, reverse=True)

        # 4. Top-K cutoff
        top_items = items[: query.top_k]

        # 5. Build ranking metadata with 1-based ranks
        ranked_results: List[Tuple[Any, RetrievalScore, RankingMetadata]] = []
        for idx, (entry, score, stage) in enumerate(top_items, start=1):
            meta = RankingMetadata(
                rank=idx,
                score=score,
                retrieval_stage=stage,
                matched_terms=[],
            )
            ranked_results.append((entry, score, meta))

        return ranked_results
