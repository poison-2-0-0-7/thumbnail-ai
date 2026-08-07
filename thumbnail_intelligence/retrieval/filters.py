"""
filters.py
==========

Deterministic stage 1 metadata filter engine for the Hybrid Retrieval Engine.
Evaluates strict predicates (entry types, niche, channel, archetype, dates, custom facets)
to narrow the candidate corpus prior to vector and keyword scoring.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TypeVar

from pydantic import BaseModel

from thumbnail_intelligence.knowledge_base.models import (
    BaseKBModel,
    EvidenceGrade,
    KnowledgeEntry,
)
from thumbnail_intelligence.retrieval.exceptions import FilterError
from thumbnail_intelligence.retrieval.query import SearchFilters

T = TypeVar("T", bound=BaseKBModel)

_EVIDENCE_GRADE_HIERARCHY: Dict[EvidenceGrade, int] = {
    EvidenceGrade.NONE: 0,
    EvidenceGrade.PATTERN_ONLY: 1,
    EvidenceGrade.WEAK: 2,
    EvidenceGrade.MODERATE: 3,
    EvidenceGrade.STRONG: 4,
}


class MetadataFilterEngine:
    """
    Evaluates hard filter predicates on knowledge entries.
    Guarantees deterministic, explainable inclusion and rejection of candidate records.
    """

    @staticmethod
    def matches(entry: Any, filters: SearchFilters) -> bool:
        """
        Evaluate whether an entity satisfies all active search filter constraints.
        Returns True if compliant, False otherwise.
        """
        if not isinstance(entry, BaseModel):
            return False

        # 1. Excluded IDs check
        entry_id = getattr(entry, "entry_id", getattr(entry, "archetype_id", getattr(entry, "pattern_id", getattr(entry, "creator_id", getattr(entry, "channel_id", getattr(entry, "competitor_id", None))))))
        if entry_id and entry_id in filters.exclude_ids:
            return False

        # 2. Entry Types check (for KnowledgeEntry models)
        if filters.entry_types:
            entry_type = getattr(entry, "entry_type", None)
            if entry_type is not None and entry_type not in filters.entry_types:
                return False

        # 3. Niche constraint check
        if filters.niche and filters.niche != "general":
            entry_niche = getattr(entry, "niche", getattr(entry, "primary_niche", None))
            if entry_niche is not None and entry_niche not in (filters.niche, "general"):
                return False

        # 4. Channel ID check (own-creator history queries)
        if filters.channel_id:
            source_chan = getattr(entry, "source_channel_id", getattr(entry, "channel_id", None))
            channel_ids_list = getattr(entry, "channel_ids", None)
            if source_chan:
                if source_chan != filters.channel_id:
                    return False
            elif channel_ids_list:
                if filters.channel_id not in channel_ids_list:
                    return False

        # 5. Creator ID check
        if filters.creator_id:
            entry_creator_id = getattr(entry, "creator_id", None)
            if entry_creator_id and entry_creator_id != filters.creator_id:
                return False

        # 6. Archetype ID check
        if filters.archetype_id:
            entry_arch = getattr(entry, "archetype_id", None)
            if entry_arch and entry_arch != filters.archetype_id:
                return False

        # 7. Confidence floor check
        if filters.min_confidence > 0.0:
            entry_conf = getattr(entry, "confidence", getattr(entry, "match_confidence", 1.0))
            if entry_conf < filters.min_confidence:
                return False

        # 8. Evidence Grade floor check
        if filters.min_evidence_grade is not None:
            entry_grade = getattr(entry, "evidence_grade", None)
            if entry_grade is not None:
                req_level = _EVIDENCE_GRADE_HIERARCHY.get(filters.min_evidence_grade, 0)
                actual_level = _EVIDENCE_GRADE_HIERARCHY.get(entry_grade, 0)
                if actual_level < req_level:
                    return False

        # 9. Date range bounds
        if filters.date_from or filters.date_to:
            created_at_str = getattr(entry, "created_at", None)
            if created_at_str:
                try:
                    entry_dt = datetime.fromisoformat(created_at_str)
                    if filters.date_from:
                        from_dt = datetime.fromisoformat(filters.date_from)
                        if entry_dt < from_dt:
                            return False
                    if filters.date_to:
                        to_dt = datetime.fromisoformat(filters.date_to)
                        if entry_dt > to_dt:
                            return False
                except Exception:
                    pass

        # 10. Custom facets matching
        if filters.custom_facets:
            entry_facets = getattr(entry, "facets", getattr(entry, "metadata", {}))
            for key, expected_val in filters.custom_facets.items():
                if key not in entry_facets:
                    return False
                if entry_facets[key] != expected_val:
                    return False

        return True

    @classmethod
    def filter_entries(cls, entries: List[T], filters: SearchFilters) -> List[T]:
        """Filter a list of entity models, returning only those satisfying search filters."""
        return [e for e in entries if cls.matches(e, filters)]

    @classmethod
    def explain_filter_rejection(cls, entry: Any, filters: SearchFilters) -> List[str]:
        """Explain why an entry was rejected by the active filters for audit and debugging."""
        reasons: List[str] = []
        entry_id = getattr(entry, "entry_id", None)
        if entry_id and entry_id in filters.exclude_ids:
            reasons.append(f"entry_id '{entry_id}' is in exclude_ids")

        if filters.entry_types:
            entry_type = getattr(entry, "entry_type", None)
            if entry_type and entry_type not in filters.entry_types:
                reasons.append(f"entry_type '{entry_type}' not in requested {filters.entry_types}")

        if filters.niche and filters.niche != "general":
            entry_niche = getattr(entry, "niche", None)
            if entry_niche and entry_niche not in (filters.niche, "general"):
                reasons.append(f"niche '{entry_niche}' != requested '{filters.niche}'")

        if filters.min_confidence > 0.0:
            entry_conf = getattr(entry, "confidence", 1.0)
            if entry_conf < filters.min_confidence:
                reasons.append(f"confidence {entry_conf} < min_confidence {filters.min_confidence}")

        return reasons
