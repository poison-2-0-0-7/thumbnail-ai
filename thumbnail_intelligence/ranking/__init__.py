"""
Ranking Engine Package (Phase 5.3 — Candidate Ranking Engine).
Determines the objectively best thumbnail candidate from an EvaluationSet using configurable policies,
deterministic tie-breaking, score separation confidence estimation, and pairwise explainability.
"""

from thumbnail_intelligence.ranking.engine import (
    CandidateRankingEngine,
    RankingEngineError,
)
from thumbnail_intelligence.ranking.models import (
    MetricComparison,
    RankedCandidate,
    RankingExplanation,
    RankingPolicy,
    RankingProfile,
    RankingReport,
    RankingResult,
)

__all__ = [
    "CandidateRankingEngine",
    "RankingEngineError",
    "RankingPolicy",
    "RankingProfile",
    "MetricComparison",
    "RankingExplanation",
    "RankedCandidate",
    "RankingReport",
    "RankingResult",
]
