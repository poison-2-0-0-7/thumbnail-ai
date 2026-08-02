"""
observability/diagnostics package
==================================

Rule Engine & Diagnostic Engine for the Pipeline Observability & Root Cause Engine (PORCE).
Evaluates TraceFacts deterministically to produce FindingCollection objects.
"""

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import (
    Finding,
    FindingCategory,
    FindingCollection,
    FindingSeverity,
    RuleContext,
    RuleResult,
)
from observability.diagnostics.registry import RuleRegistry
from observability.diagnostics.rule_engine import RuleEngine, RuleExecutionEngine
from observability.diagnostics.validation import RuleValidation

__all__ = [
    "IDiagnosticRule",
    "FindingSeverity",
    "FindingCategory",
    "Finding",
    "FindingCollection",
    "RuleContext",
    "RuleResult",
    "RuleRegistry",
    "RuleExecutionEngine",
    "RuleEngine",
    "RuleValidation",
]
