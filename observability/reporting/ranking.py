"""
observability/reporting/ranking.py
==================================

RootCauseRanking implements deterministic root cause ranking logic.
Ranks findings by severity and confidence, deduplicating by affected_module to extract top 3 root causes.
"""

from __future__ import annotations

from observability.diagnostics.models import Finding

SEVERITY_RANK_ORDER = {
    "FAIL": 0,
    "WARNING": 1,
    "INFO": 2,
    "PASS": 3,
}


class RootCauseRanking:
    """
    Deterministic ranking engine for sorting findings and deriving top_root_causes summary.
    """

    @staticmethod
    def rank_findings(findings: list[Finding]) -> list[Finding]:
        """
        Sort findings by severity ("FAIL" > "WARNING" > "INFO" > "PASS") then confidence (descending).
        """
        return sorted(
            findings,
            key=lambda f: (SEVERITY_RANK_ORDER.get(f.severity, 4), -f.confidence),
        )

    @classmethod
    def extract_top_root_causes(
        cls, findings: list[Finding], limit: int = 3
    ) -> list[str]:
        """
        Derive top_root_causes list:
        1. Rank findings by severity then confidence.
        2. Deduplicate by affected_module.
        3. Extract root_cause sentence from top N findings.
        """
        sorted_findings = cls.rank_findings(findings)

        top_causes: list[str] = []
        seen_modules: set[str] = set()

        for f in sorted_findings:
            # Only consider actionable findings (FAIL, WARNING, INFO)
            if f.severity in ("FAIL", "WARNING", "INFO"):
                if f.affected_module not in seen_modules:
                    seen_modules.add(f.affected_module)
                    top_causes.append(f.root_cause)
                    if len(top_causes) >= limit:
                        break

        return top_causes
