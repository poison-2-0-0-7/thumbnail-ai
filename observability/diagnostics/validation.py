"""
observability/diagnostics/validation.py
========================================

Validation utilities for Diagnostic Engine rules, findings, and rule collections.
"""

from __future__ import annotations

from typing import Any
from pydantic import ValidationError

from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.models import Finding, FindingCollection


class RuleValidation:
    """
    Validator for IDiagnosticRule implementations and Finding objects.
    """

    @staticmethod
    def validate_rule_instance(rule: Any) -> bool:
        """
        Check if object implements IDiagnosticRule interface correctly.
        """
        if not isinstance(rule, IDiagnosticRule):
            return False
        return bool(rule.rule_id and rule.rule_name and rule.category)

    @staticmethod
    def validate_finding_data(data: Any) -> bool:
        """
        Validate dictionary or model against Finding schema.
        """
        if isinstance(data, Finding):
            return True
        if isinstance(data, dict):
            try:
                Finding.model_validate(data)
                return True
            except ValidationError:
                return False
        return False

    @staticmethod
    def validate_finding_collection(data: Any) -> bool:
        """
        Validate dictionary or model against FindingCollection schema.
        """
        if isinstance(data, FindingCollection):
            return True
        if isinstance(data, dict):
            try:
                FindingCollection.model_validate(data)
                return True
            except ValidationError:
                return False
        return False
