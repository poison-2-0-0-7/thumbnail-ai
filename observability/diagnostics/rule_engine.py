"""
observability/diagnostics/rule_engine.py
=========================================

RuleEngine & RuleExecutionEngine for running diagnostic rules over TraceFacts.
Outputs deterministic FindingCollection objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import (
    Finding,
    FindingCollection,
    RuleContext,
    RuleResult,
)
from observability.diagnostics.registry import RuleRegistry
from observability.facts.models import TraceFacts
from observability.models import PipelineTrace


class RuleExecutionEngine:
    """
    Core executor that evaluates a single IDiagnosticRule against RuleContext.
    Catches exceptions non-fatally to ensure resilient diagnosis.
    """

    @staticmethod
    def execute_rule(rule: IDiagnosticRule, context: RuleContext) -> RuleResult:
        """
        Execute a single rule defensively against RuleContext.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        try:
            finding = rule.check(context.facts, context)
            if finding is None:
                return RuleResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    passed=True,
                    finding=None,
                )
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                passed=(finding.severity == "PASS"),
                finding=finding,
            )
        except Exception as exc:
            logger.warning(
                "Rule {r_id} execution raised an exception: {exc}",
                r_id=rule.rule_id,
                exc=exc,
            )
            error_finding = Finding(
                finding_id=rule.rule_id,
                rule_name=rule.rule_name,
                category=rule.category,
                severity="INFO",
                confidence=1.0,
                affected_module="observability",
                root_cause=f"Diagnostic rule execution encountered a non-fatal error: {exc}",
                recommended_action="Inspect rule check implementation for edge case handling.",
                supporting_facts=[f"exception={type(exc).__name__}"],
                evaluation_timestamp=now_str,
            )
            return RuleResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                passed=False,
                finding=error_finding,
            )


class RuleEngine:
    """
    Main entry point for running all registered diagnostic rules over TraceFacts.
    """

    def __init__(self, registry: Optional[RuleRegistry] = None) -> None:
        self.registry = registry or RuleRegistry()
        self.executor = RuleExecutionEngine()

    def evaluate(
        self,
        facts: TraceFacts,
        pipeline_trace: Optional[PipelineTrace] = None,
    ) -> FindingCollection:
        """
        Evaluate all enabled diagnostic rules against facts and return FindingCollection.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        context = RuleContext(
            facts=facts,
            pipeline_trace=pipeline_trace,
            generation_trace=pipeline_trace.generation_trace if (pipeline_trace and hasattr(pipeline_trace, "generation_trace")) else None,
        )

        findings: list[Finding] = []
        fail_cnt = 0
        warn_cnt = 0
        info_cnt = 0
        pass_cnt = 0

        rules_to_run = self.registry.get_enabled_rules()

        for rule in rules_to_run:
            res = self.executor.execute_rule(rule, context)
            if res.finding is not None:
                findings.append(res.finding)
                if res.finding.severity == "FAIL":
                    fail_cnt += 1
                elif res.finding.severity == "WARNING":
                    warn_cnt += 1
                elif res.finding.severity == "INFO":
                    info_cnt += 1
                elif res.finding.severity == "PASS":
                    pass_cnt += 1
            else:
                pass_cnt += 1

        return FindingCollection(
            video_id=facts.video_id,
            findings=findings,
            fail_count=fail_cnt,
            warning_count=warn_cnt,
            info_count=info_cnt,
            pass_count=pass_cnt,
            evaluated_at=now_str,
        )
