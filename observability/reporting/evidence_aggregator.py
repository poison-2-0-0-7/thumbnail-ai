"""
observability/reporting/evidence_aggregator.py
===============================================

FindingAggregator and EvidenceAggregator for consolidating finding evidence and statistics in PORCE.
"""

from __future__ import annotations

from typing import Any

from observability.diagnostics.models import Finding, FindingCollection


class FindingAggregator:
    """
    Aggregates statistics and provides filtering methods over a collection of Findings.
    """

    def __init__(self, findings: list[Finding]) -> None:
        self.findings = findings

    def get_by_severity(self, severity: str) -> list[Finding]:
        """Return findings matching specified severity."""
        return [f for f in self.findings if f.severity == severity]

    def get_by_module(self, module_name: str) -> list[Finding]:
        """Return findings matching specified affected_module."""
        return [f for f in self.findings if f.affected_module == module_name]

    def compute_counts(self) -> dict[str, int]:
        """Compute counts by severity."""
        counts = {"FAIL": 0, "WARNING": 0, "INFO": 0, "PASS": 0}
        for f in self.findings:
            if f.severity in counts:
                counts[f.severity] += 1
        return counts


class EvidenceAggregator:
    """
    Aggregates supporting evidence, facts, and artifacts across all findings in a report.
    """

    @staticmethod
    def aggregate_evidence(findings: list[Finding]) -> dict[str, Any]:
        """
        Consolidate supporting evidence across all findings.
        Returns a structured evidence summary dictionary.
        """
        all_facts: set[str] = set()
        all_artifacts: set[str] = set()
        affected_modules: set[str] = set()
        evidence_item_count = 0

        module_evidence_map: dict[str, list[str]] = {}

        for f in findings:
            affected_modules.add(f.affected_module)

            for fact_str in f.supporting_facts:
                all_facts.add(fact_str)

            for art_str in f.related_artifacts:
                all_artifacts.add(art_str)

            evidence_item_count += len(f.supporting_evidence)

            mod_list = module_evidence_map.setdefault(f.affected_module, [])
            mod_list.append(f.finding_id)

        return {
            "total_findings": len(findings),
            "total_evidence_items": evidence_item_count,
            "affected_modules": sorted(list(affected_modules)),
            "supporting_facts_count": len(all_facts),
            "related_artifacts_count": len(all_artifacts),
            "module_evidence_map": module_evidence_map,
            "supporting_facts_sample": sorted(list(all_facts))[:10],
        }
