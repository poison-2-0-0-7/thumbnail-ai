"""
hybrid_search.py
================

Multi-stage Hybrid Search engine for the Thumbnail Intelligence Engine.
Orchestrates:
1. Retrieval Cache Lookup
2. Stage 1: Hard Metadata Filter Stage
3. Stage 2: Lexical / Keyword Matching
4. Stage 3: Vector Embedding Similarity Search
5. Stage 4: Multi-Signal Composite Scoring
6. Stage 5: Explainable Hybrid Ranking & Top-K Cutoff
7. Stage 6: Semantic & ID Deduplication
8. Stage 7: Evidence Bundle Assembly & Provenance Packaging
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from loguru import logger

from thumbnail_intelligence.knowledge_base.models import BaseKBModel
from thumbnail_intelligence.retrieval.cache import RetrievalCache
from thumbnail_intelligence.retrieval.config import RetrievalConfig
from thumbnail_intelligence.retrieval.embedding import InMemoryVectorIndex
from thumbnail_intelligence.retrieval.evidence_bundle import (
    EvidenceBundle,
    RetrievalResult,
    RetrievedEvidence,
    SearchStatistics,
)
from thumbnail_intelligence.retrieval.filters import MetadataFilterEngine
from thumbnail_intelligence.retrieval.metadata_search import MetadataSearchEngine
from thumbnail_intelligence.retrieval.query import RetrievalQuery
from thumbnail_intelligence.retrieval.ranking import HybridRanker
from thumbnail_intelligence.retrieval.scoring import ScoringEngine


class HybridSearchEngine:
    """
    Primary multi-stage retrieval engine implementing the hybrid search pipeline.
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        vector_index: Optional[InMemoryVectorIndex] = None,
        cache: Optional[RetrievalCache] = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.vector_index = vector_index or InMemoryVectorIndex(expected_dim=self.config.embedding_dim)
        self.cache = cache or (RetrievalCache(max_size=self.config.max_cache_size, default_ttl_seconds=self.config.cache_ttl_seconds) if self.config.cache_enabled else None)
        self.filter_engine = MetadataFilterEngine()
        self.metadata_search = MetadataSearchEngine(filter_engine=self.filter_engine)
        self.scoring_engine = ScoringEngine()
        self.ranker = HybridRanker()

    def search(
        self,
        candidates: List[Any],
        query: RetrievalQuery,
    ) -> RetrievalResult:
        """
        Execute multi-stage hybrid search across candidate corpus.
        Returns complete, structured RetrievalResult containing the EvidenceBundle.
        """
        start_time = time.perf_counter()

        # Step 1: Check Retrieval Cache
        if self.cache and self.config.cache_enabled:
            cached_res = self.cache.get(query)
            if cached_res is not None:
                return cached_res

        total_examined = len(candidates)
        stages_executed = ["metadata_filtering"]

        # Step 2: Stage 1 Hard Metadata Filter
        t0 = time.perf_counter()
        passed_filters: List[Any] = []
        candidate_ids: Set[str] = set()

        for c in candidates:
            if self.filter_engine.matches(c, query.filters):
                passed_filters.append(c)
                cid = getattr(
                    c,
                    "entry_id",
                    getattr(
                        c,
                        "archetype_id",
                        getattr(
                            c,
                            "pattern_id",
                            getattr(c, "creator_id", getattr(c, "channel_id", getattr(c, "competitor_id", None))),
                        ),
                    ),
                )
                if cid:
                    candidate_ids.add(str(cid))

        filter_latency = (time.perf_counter() - t0) * 1000.0

        if not passed_filters:
            # Explicit empty result
            stats = SearchStatistics(
                filter_latency_ms=round(filter_latency, 2),
                total_latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
                stages_executed=stages_executed,
            )
            bundle = EvidenceBundle(
                query_id=query.query_id,
                items=[],
                total_candidates_examined=total_examined,
                total_candidates_passed_filters=0,
                statistics=stats,
            )
            return RetrievalResult(
                query=query,
                bundle=bundle,
                status="empty",
                message="No candidate records satisfied the stage-1 metadata filters.",
            )

        # Step 3: Stage 2 Vector Similarity Scoring
        t1 = time.perf_counter()
        vector_sim_map: Dict[str, float] = {}

        if query.query_embedding and len(query.query_embedding) > 0:
            stages_executed.append("vector_similarity")
            try:
                matches = self.vector_index.search(
                    query_vector=query.query_embedding,
                    top_k=len(passed_filters),
                    min_score=0.0,
                    candidate_ids=candidate_ids,
                )
                for eid, sim in matches:
                    vector_sim_map[eid] = sim
            except Exception as e:
                logger.warning(f"Vector search failed, continuing with metadata search: {e}")

        embedding_latency = (time.perf_counter() - t1) * 1000.0

        # Step 4: Stage 3 & 4 Lexical Keyword + Multi-Signal Composite Scoring
        t2 = time.perf_counter()
        stages_executed.append("multi_signal_scoring")
        scored_candidates: List[Tuple[Any, Any, str]] = []

        for candidate in passed_filters:
            cid = getattr(
                candidate,
                "entry_id",
                getattr(
                    candidate,
                    "archetype_id",
                    getattr(
                        candidate,
                        "pattern_id",
                        getattr(
                            candidate,
                            "creator_id",
                            getattr(candidate, "channel_id", getattr(candidate, "competitor_id", "")),
                        ),
                    ),
                ),
            )
            v_sim = vector_sim_map.get(str(cid), 0.0)

            # Compute composite explainable score
            score = self.scoring_engine.score_candidate(
                entry=candidate,
                query=query,
                visual_similarity=v_sim,
                weights=self.config.weights,
                half_life_days=self.config.recency_half_life_days,
            )
            stage_name = "hybrid" if v_sim > 0.0 else "metadata_filtered"
            scored_candidates.append((candidate, score, stage_name))

        scoring_latency = (time.perf_counter() - t2) * 1000.0

        # Step 5: Stage 5 Hybrid Ranking & Deduplication
        stages_executed.append("hybrid_ranking_and_deduplication")
        ranked_items = self.ranker.rank(
            candidates=scored_candidates,
            query=query,
            deduplication_threshold=self.config.deduplication_threshold,
        )

        # Step 6: Package into RetrievedEvidence and EvidenceBundle
        evidence_items: List[RetrievedEvidence] = []
        for entry, score, meta in ranked_items:
            reason = f"Ranked #{meta.rank} with overall score {score.overall_score:.3f} via {meta.retrieval_stage}."
            ev = RetrievedEvidence.from_entry(
                entry=entry,
                score=score,
                ranking=meta,
                reason=reason,
            )
            evidence_items.append(ev)

        total_latency = (time.perf_counter() - start_time) * 1000.0

        stats = SearchStatistics(
            filter_latency_ms=round(filter_latency, 2),
            embedding_latency_ms=round(embedding_latency, 2),
            scoring_latency_ms=round(scoring_latency, 2),
            total_latency_ms=round(total_latency, 2),
            cache_hit=False,
            stages_executed=stages_executed,
        )

        bundle = EvidenceBundle(
            query_id=query.query_id,
            items=evidence_items,
            total_candidates_examined=total_examined,
            total_candidates_passed_filters=len(passed_filters),
            deduplicated_count=len(scored_candidates) - len(evidence_items),
            statistics=stats,
        )
        bundle.partition_by_domain()

        result = RetrievalResult(
            query=query,
            bundle=bundle,
            status="success" if len(evidence_items) > 0 else "empty",
            message=f"Retrieved {len(evidence_items)} evidence records out of {total_examined} candidates in {total_latency:.2f}ms.",
        )

        # Step 7: Store in Cache if enabled
        if self.cache and self.config.cache_enabled:
            self.cache.set(query, result)

        return result
