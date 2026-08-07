"""
scoring.py
==========

Explainable multi-signal scoring engine for the Hybrid Retrieval Engine.
Evaluates:
- Visual & embedding cosine similarity
- Creator & channel affinity
- Archetype alignment
- Niche relevance
- Exponential recency time-decay
- Confidence and metadata richness
- Source priority and empirical evidence quality
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from thumbnail_intelligence.knowledge_base.models import (
    BaseKBModel,
    EvidenceGrade,
    KnowledgeEntry,
    KnowledgeEntryType,
)
from thumbnail_intelligence.retrieval.config import RankingWeights
from thumbnail_intelligence.retrieval.query import RetrievalQuery

_SOURCE_PRIORITY: Dict[KnowledgeEntryType, float] = {
    KnowledgeEntryType.HISTORICAL_THUMBNAIL: 1.0,
    KnowledgeEntryType.ARCHETYPE_EXAMPLE: 0.90,
    KnowledgeEntryType.COMPETITOR_THUMBNAIL: 0.85,
    KnowledgeEntryType.DESIGN_PATTERN: 0.80,
    KnowledgeEntryType.VISUAL_PATTERN: 0.80,
    KnowledgeEntryType.THUMBNAIL_PATTERN: 0.85,
    KnowledgeEntryType.CREATOR_PROFILE_ENTRY: 0.95,
}

_GRADE_SCORES: Dict[EvidenceGrade, float] = {
    EvidenceGrade.STRONG: 1.0,
    EvidenceGrade.MODERATE: 0.75,
    EvidenceGrade.WEAK: 0.50,
    EvidenceGrade.PATTERN_ONLY: 0.35,
    EvidenceGrade.NONE: 0.10,
}


class RetrievalScore(BaseKBModel):
    """
    Explainable breakdown of the relevance and quality score for a retrieved knowledge entry.
    Every sub-component is transparently disclosed.
    """

    overall_score: float = 0.0
    visual_similarity: float = 0.0
    creator_channel_similarity: float = 0.0
    archetype_similarity: float = 0.0
    niche_match_score: float = 0.0
    recency_score: float = 1.0
    confidence_score: float = 1.0
    metadata_quality_score: float = 1.0
    source_priority_score: float = 1.0
    evidence_grade_score: float = 1.0
    component_breakdown: Dict[str, float] = {}
    explanation: str = ""


class ScoringEngine:
    """
    Evaluates multi-signal mathematical scores combining visual, categorical, and recency dimensions.
    """

    @staticmethod
    def compute_recency_score(created_at_str: Optional[str], half_life_days: float = 90.0) -> float:
        """
        Compute exponential time-decay score in [0.0, 1.0] from creation timestamp.
        score = 2 ** (-age_in_days / half_life_days)
        """
        if not created_at_str:
            return 0.80  # Default neutral recency for un-timestamped legacy entries
        try:
            created_dt = datetime.fromisoformat(created_at_str)
            now = datetime.now(timezone.utc)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            age_seconds = max(0.0, (now - created_dt).total_seconds())
            age_days = age_seconds / 86400.0
            decay = math.pow(2.0, -age_days / max(1.0, half_life_days))
            return max(0.05, min(1.0, decay))
        except Exception:
            return 0.80

    @staticmethod
    def compute_metadata_quality(entry: Any) -> float:
        """
        Evaluate metadata completeness and richness (facets, tags, descriptors).
        """
        score = 0.5
        facets = getattr(entry, "facets", getattr(entry, "metadata", {}))
        if facets and len(facets) > 0:
            score += min(0.3, len(facets) * 0.05)
        if getattr(entry, "description", None):
            score += 0.1
        if getattr(entry, "outcome_ref", None):
            score += 0.1
        return min(1.0, score)

    @staticmethod
    def compute_source_priority(entry: Any) -> float:
        """Evaluate base priority of entry origin."""
        entry_type = getattr(entry, "entry_type", None)
        if entry_type and entry_type in _SOURCE_PRIORITY:
            return _SOURCE_PRIORITY[entry_type]
        return 0.80

    @staticmethod
    def compute_evidence_grade_score(entry: Any) -> float:
        """Evaluate quality of empirical grounding."""
        grade = getattr(entry, "evidence_grade", None)
        if grade and grade in _GRADE_SCORES:
            return _GRADE_SCORES[grade]
        return 0.70

    @classmethod
    def score_candidate(
        cls,
        entry: Any,
        query: RetrievalQuery,
        visual_similarity: float = 0.0,
        weights: Optional[RankingWeights] = None,
        half_life_days: float = 90.0,
    ) -> RetrievalScore:
        """
        Calculate composite, fully-explainable RetrievalScore for an individual candidate.
        """
        w = weights or RankingWeights()
        if query.weights_override:
            # Construct customized weights if query specifies overrides
            custom_w = w.to_dict()
            custom_w.update(query.weights_override)
            w = RankingWeights(**custom_w)

        # 1. Visual / embedding similarity
        v_sim = max(0.0, min(1.0, visual_similarity))

        # 2. Creator & Channel affinity
        creator_sim = 0.0
        q_chan = query.context.channel_id or query.filters.channel_id
        entry_chan = getattr(entry, "source_channel_id", getattr(entry, "channel_id", None))
        q_creator = query.context.creator_id or query.filters.creator_id
        entry_creator = getattr(entry, "creator_id", None)

        if q_chan and entry_chan and q_chan == entry_chan:
            creator_sim = 1.0
        elif q_creator and entry_creator and q_creator == entry_creator:
            creator_sim = 0.85
        else:
            creator_sim = 0.20

        # 3. Archetype match
        archetype_sim = 0.0
        q_arch = query.context.archetype_id or query.filters.archetype_id
        entry_arch = getattr(entry, "archetype_id", None)
        if q_arch and entry_arch and q_arch == entry_arch:
            archetype_sim = 1.0
        elif entry_arch:
            archetype_sim = 0.50
        else:
            archetype_sim = 0.30

        # 4. Niche relevance
        niche_score = 0.0
        q_niche = query.context.niche or query.filters.niche or "general"
        entry_niche = getattr(entry, "niche", getattr(entry, "primary_niche", "general"))
        if entry_niche == q_niche and q_niche != "general":
            niche_score = 1.0
        elif entry_niche == "general" or q_niche == "general":
            niche_score = 0.60
        else:
            niche_score = 0.20

        # 5. Recency
        created_at_str = getattr(entry, "created_at", None)
        recency = cls.compute_recency_score(created_at_str, half_life_days=half_life_days)

        # 6. Model / Entry confidence
        confidence = getattr(entry, "confidence", getattr(entry, "match_confidence", 1.0))
        confidence = max(0.0, min(1.0, confidence))

        # 7. Metadata Quality
        meta_qual = cls.compute_metadata_quality(entry)

        # 8. Source Priority & Evidence Grade
        src_prio = cls.compute_source_priority(entry)
        ev_grade_score = cls.compute_evidence_grade_score(entry)

        # Compute weighted sum
        overall = (
            w.visual_similarity * v_sim
            + w.creator_channel_affinity * creator_sim
            + w.archetype_match * archetype_sim
            + w.niche_match * niche_score
            + w.recency * recency
            + w.confidence * confidence
            + w.metadata_quality * meta_qual
        )
        # Scale slightly with source priority and evidence grade
        overall = overall * (0.8 + 0.1 * src_prio + 0.1 * ev_grade_score)
        overall = max(0.0, min(1.0, overall))

        breakdown = {
            "visual_similarity": v_sim,
            "creator_channel_affinity": creator_sim,
            "archetype_match": archetype_sim,
            "niche_match": niche_score,
            "recency": recency,
            "confidence": confidence,
            "metadata_quality": meta_qual,
            "source_priority": src_prio,
            "evidence_grade": ev_grade_score,
        }

        explanation = (
            f"Overall {overall:.3f} | Visual: {v_sim:.2f} (w={w.visual_similarity:.2f}), "
            f"Channel: {creator_sim:.2f}, Arch: {archetype_sim:.2f}, Niche: {niche_score:.2f}, "
            f"Recency: {recency:.2f}, Conf: {confidence:.2f}"
        )

        return RetrievalScore(
            overall_score=round(overall, 4),
            visual_similarity=round(v_sim, 4),
            creator_channel_similarity=round(creator_sim, 4),
            archetype_similarity=round(archetype_sim, 4),
            niche_match_score=round(niche_score, 4),
            recency_score=round(recency, 4),
            confidence_score=round(confidence, 4),
            metadata_quality_score=round(meta_qual, 4),
            source_priority_score=round(src_prio, 4),
            evidence_grade_score=round(ev_grade_score, 4),
            component_breakdown=breakdown,
            explanation=explanation,
        )
