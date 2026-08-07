"""
evidence_bundle.py
==================

Evidence representation, packaging, and retrieval outcome containers.
Guarantees that every retrieved artifact carries origin, confidence, reason retrieved,
explainable score, and grounding evidence references. No anonymous evidence.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from thumbnail_intelligence.knowledge_base.models import (
    BaseKBModel,
    EvidenceReference,
    EvidenceSourceType,
    KnowledgeEntryType,
    _utc_now_iso,
)
from thumbnail_intelligence.retrieval.query import RetrievalQuery
from thumbnail_intelligence.retrieval.ranking import RankingMetadata
from thumbnail_intelligence.retrieval.scoring import RetrievalScore


class SearchStatistics(BaseKBModel):
    """Execution timing and telemetry metrics across retrieval stages."""

    filter_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    scoring_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    cache_hit: bool = False
    stages_executed: List[str] = Field(default_factory=list)


class RetrievedEvidence(BaseKBModel):
    """
    An individual grounded, scored, and ranked evidence item.
    Guarantees full provenance: origin, confidence, reason retrieved, and metadata.
    """

    evidence_id: str = Field(description="Deterministic hash identifying this evidence record")
    entry_id: str = Field(description="Underlying knowledge base record identifier")
    entry_type: KnowledgeEntryType = Field(description="Classification of retrieved knowledge")
    origin: str = Field(description="Structured origin descriptor (e.g. 'historical:video_123')")
    source_id: Optional[str] = Field(default=None, description="Direct entity identifier if available")
    source_type: Optional[EvidenceSourceType] = Field(default=None, description="Grounding source classification")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Model or empirical confidence")
    reason_retrieved: str = Field(description="Explainable natural language reason why this evidence was selected")
    score: RetrievalScore = Field(description="Full explainable multi-signal score breakdown")
    ranking: RankingMetadata = Field(description="Assigned position and rank metadata")
    data_payload: Dict[str, Any] = Field(default_factory=dict, description="Raw underlying entity attributes")
    evidence_refs: List[EvidenceReference] = Field(
        default_factory=list, description="Associated grounding references"
    )

    @classmethod
    def from_entry(
        cls,
        entry: Any,
        score: RetrievalScore,
        ranking: RankingMetadata,
        reason: str,
    ) -> RetrievedEvidence:
        """Construct a RetrievedEvidence instance with automated provenance hashing."""
        entry_id = str(
            getattr(
                entry,
                "entry_id",
                getattr(
                    entry,
                    "archetype_id",
                    getattr(
                        entry,
                        "pattern_id",
                        getattr(
                            entry,
                            "creator_id",
                            getattr(entry, "channel_id", getattr(entry, "competitor_id", "unknown_entry")),
                        ),
                    ),
                ),
            )
        )
        entry_type = getattr(entry, "entry_type", KnowledgeEntryType.HISTORICAL_THUMBNAIL)
        origin = f"{entry_type.value if hasattr(entry_type, 'value') else entry_type}:{entry_id}"

        hasher = hashlib.sha256()
        hasher.update(f"{entry_id}:{ranking.rank}:{score.overall_score}".encode("utf-8"))
        ev_id = f"ev_{hasher.hexdigest()[:12]}"

        # Extract data payload
        payload = entry.to_dict() if hasattr(entry, "to_dict") else {}
        evidence_refs = getattr(entry, "evidence_refs", getattr(entry, "evidence", []))

        return cls(
            evidence_id=ev_id,
            entry_id=entry_id,
            entry_type=entry_type,
            origin=origin,
            source_id=getattr(entry, "source_video_id", getattr(entry, "source_id", entry_id)),
            source_type=getattr(entry, "source_type", EvidenceSourceType.KNOWLEDGE_ENTRY),
            confidence=getattr(entry, "confidence", 1.0),
            reason_retrieved=reason,
            score=score,
            ranking=ranking,
            data_payload=payload,
            evidence_refs=evidence_refs if isinstance(evidence_refs, list) else [],
        )


class EvidenceBundle(BaseKBModel):
    """
    Consolidated, bounded bundle of retrieved evidence partitioned by domain.
    Serves as the primary evidence artifact passed to downstream reasoning engines.
    """

    query_id: str = Field(description="Target query identifier")
    items: List[RetrievedEvidence] = Field(default_factory=list, description="Unified ordered list of top-K evidence")
    archetype_evidence: List[RetrievedEvidence] = Field(
        default_factory=list, description="Retrieved archetype examples and templates"
    )
    historical_evidence: List[RetrievedEvidence] = Field(
        default_factory=list, description="Retrieved creator and channel historical records"
    )
    competitor_evidence: List[RetrievedEvidence] = Field(
        default_factory=list, description="Retrieved competitor benchmarks and signatures"
    )
    pattern_evidence: List[RetrievedEvidence] = Field(
        default_factory=list, description="Retrieved design and visual patterns"
    )
    creator_evidence: List[RetrievedEvidence] = Field(
        default_factory=list, description="Creator profiles and brand constraint evidence"
    )
    total_candidates_examined: int = Field(default=0, ge=0)
    total_candidates_passed_filters: int = Field(default=0, ge=0)
    deduplicated_count: int = Field(default=0, ge=0)
    statistics: SearchStatistics = Field(default_factory=SearchStatistics)
    retrieved_at: str = Field(default_factory=_utc_now_iso)

    def partition_by_domain(self) -> None:
        """Helper to partition items into domain-specific lists."""
        arch, hist, comp, pat, creat = [], [], [], [], []
        for it in self.items:
            t = it.entry_type
            if t == KnowledgeEntryType.ARCHETYPE_EXAMPLE:
                arch.append(it)
            elif t == KnowledgeEntryType.HISTORICAL_THUMBNAIL:
                hist.append(it)
            elif t == KnowledgeEntryType.COMPETITOR_THUMBNAIL:
                comp.append(it)
            elif t in (KnowledgeEntryType.DESIGN_PATTERN, KnowledgeEntryType.VISUAL_PATTERN, KnowledgeEntryType.THUMBNAIL_PATTERN):
                pat.append(it)
            elif t == KnowledgeEntryType.CREATOR_PROFILE_ENTRY:
                creat.append(it)
            else:
                hist.append(it)

        object.__setattr__(self, "archetype_evidence", arch)
        object.__setattr__(self, "historical_evidence", hist)
        object.__setattr__(self, "competitor_evidence", comp)
        object.__setattr__(self, "pattern_evidence", pat)
        object.__setattr__(self, "creator_evidence", creat)


class RetrievalResult(BaseKBModel):
    """
    Top-level output wrapper from the Hybrid Retrieval Engine.
    """

    query: RetrievalQuery = Field(description="Origin query specification")
    bundle: EvidenceBundle = Field(description="Consolidated retrieved evidence bundle")
    status: Literal["success", "partial", "empty", "error"] = Field(
        default="success", description="Outcome status of retrieval execution"
    )
    message: str = Field(default="", description="Optional diagnostic message")
