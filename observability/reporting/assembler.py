"""
observability/reporting/assembler.py
====================================

RootCauseAssembler builds the canonical RootCauseReport for a video_id.
Calculates trace SHA-256 hash, aggregates evidence, and computes top_root_causes.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

from observability.diagnostics.models import FindingCollection
from observability.facts.models import FactCollection
from observability.models import PipelineTrace
from observability.reporting.evidence_aggregator import EvidenceAggregator, FindingAggregator
from observability.reporting.interfaces import IRootCauseAssembler
from observability.reporting.models import RootCauseReport
from observability.reporting.ranking import RootCauseRanking


def _compute_trace_hash(trace: PipelineTrace) -> str:
    """Compute SHA-256 hash of PipelineTrace json content."""
    try:
        content_json = trace.model_dump_json()
        return hashlib.sha256(content_json.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(trace.video_id.encode("utf-8")).hexdigest()


class RootCauseAssembler(IRootCauseAssembler):
    """
    Assembles a canonical RootCauseReport from PipelineTrace, FindingCollection, and TraceFacts.
    """

    def assemble(
        self,
        video_id: str,
        pipeline_trace: PipelineTrace,
        finding_collection: FindingCollection,
        fact_collection: Optional[FactCollection] = None,
    ) -> RootCauseReport:
        """
        Assemble canonical RootCauseReport.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        trace_hash = _compute_trace_hash(pipeline_trace)

        findings = finding_collection.findings
        aggregator = FindingAggregator(findings)
        counts = aggregator.compute_counts()

        # Compute top_root_causes
        top_causes = RootCauseRanking.extract_top_root_causes(findings, limit=3)

        # Aggregate evidence summary
        evidence_sum = EvidenceAggregator.aggregate_evidence(findings)

        # Determine report overall status
        if counts["FAIL"] > 0 or pipeline_trace.overall_status == "error":
            status = "error"
        elif counts["WARNING"] > 0 or pipeline_trace.overall_status == "partial":
            status = "partial"
        else:
            status = "success"

        return RootCauseReport(
            video_id=video_id,
            findings=findings,
            fail_count=counts["FAIL"],
            warning_count=counts["WARNING"],
            info_count=counts["INFO"],
            pass_count=counts["PASS"],
            top_root_causes=top_causes,
            generated_from_trace_hash=trace_hash,
            engine_version="1.0.0",
            status=status,
            generated_at=now_str,
            evidence_summary=evidence_sum,
        )
