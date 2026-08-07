"""
provenance.py
=============

Provenance tracking and lineage recording for evidence nodes.
Ensures full auditability, non-repudiation, and origin verification across
the normalization pipeline.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from thumbnail_intelligence.evidence.models import ProvenanceRecord
from thumbnail_intelligence.knowledge_base.models import EvidenceSourceType, _utc_now_iso
from thumbnail_intelligence.retrieval.evidence_bundle import RetrievedEvidence


class ProvenanceTracker:
    """
    Creates and manages cryptographic-like trace provenance records for evidence nodes.
    """

    @staticmethod
    def create_record(
        evidence: RetrievedEvidence,
        query_id: str,
        parent_origins: Optional[List[str]] = None,
    ) -> ProvenanceRecord:
        """
        Construct an immutable ProvenanceRecord from raw RetrievedEvidence.
        """
        source_id = evidence.source_id or evidence.entry_id
        source_type = evidence.source_type or EvidenceSourceType.KNOWLEDGE_ENTRY
        created_at_val = (
            evidence.data_payload.get("created_at")
            if isinstance(evidence.data_payload, dict)
            else None
        ) or _utc_now_iso()

        trace_id = f"tr_{uuid.uuid4().hex[:12]}"

        return ProvenanceRecord(
            origin=evidence.origin,
            source_id=str(source_id),
            source_type=source_type,
            retrieval_query_id=query_id,
            retrieval_reason=evidence.reason_retrieved,
            retrieved_at=_utc_now_iso(),
            created_at=created_at_val,
            parent_origins=parent_origins or [],
            trace_id=trace_id,
        )

    @staticmethod
    def derive_record(
        parent: ProvenanceRecord,
        derivation_reason: str,
    ) -> ProvenanceRecord:
        """
        Create a child ProvenanceRecord inheriting parent lineage.
        """
        child_parents = list(parent.parent_origins)
        if parent.origin not in child_parents:
            child_parents.append(parent.origin)

        return ProvenanceRecord(
            origin=f"derived:{parent.origin}",
            source_id=parent.source_id,
            source_type=parent.source_type,
            retrieval_query_id=parent.retrieval_query_id,
            retrieval_reason=f"Derived from {parent.origin}: {derivation_reason}",
            retrieved_at=parent.retrieved_at,
            created_at=_utc_now_iso(),
            parent_origins=child_parents,
            trace_id=f"tr_{uuid.uuid4().hex[:12]}",
        )
