"""
observability/diagnostics/registry.py
======================================

RuleRegistry for registering, looking up, and configuring IDiagnosticRules in PORCE.
Matches the project's static, explicit composition design discipline.
"""

from __future__ import annotations

from typing import Optional

from observability.config import OBS_RULE_REGISTRY_ENABLED
from observability.diagnostics.interfaces import IDiagnosticRule
from observability.diagnostics.rules import DEFAULT_RULE_CLASSES


class RuleRegistry:
    """
    Registry managing registered IDiagnosticRule instances.
    """

    def __init__(self, load_defaults: bool = True) -> None:
        self._rules: dict[str, IDiagnosticRule] = {}
        if load_defaults:
            self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default rule instances."""
        for rule_cls in DEFAULT_RULE_CLASSES:
            instance = rule_cls()
            self.register_rule(instance)

    def register_rule(self, rule: IDiagnosticRule) -> None:
        """Register a diagnostic rule instance."""
        self._rules[rule.rule_id] = rule

    def unregister_rule(self, rule_id: str) -> None:
        """Unregister a rule by rule_id."""
        self._rules.pop(rule_id, None)

    def get_rule(self, rule_id: str) -> Optional[IDiagnosticRule]:
        """Get a registered rule by rule_id."""
        return self._rules.get(rule_id)

    def get_all_rules(self) -> list[IDiagnosticRule]:
        """Return list of all registered rule instances."""
        return list(self._rules.values())

    def get_enabled_rules(self) -> list[IDiagnosticRule]:
        """
        Return list of enabled rule instances based on OBS_RULE_REGISTRY_ENABLED config.
        """
        enabled: list[IDiagnosticRule] = []
        for r_id, rule in self._rules.items():
            is_enabled = OBS_RULE_REGISTRY_ENABLED.get(r_id, True)
            if is_enabled:
                enabled.append(rule)
        return enabled

    def get_rules_by_category(self, category: str) -> list[IDiagnosticRule]:
        """Return list of registered rules for a specific category."""
        return [r for r in self._rules.values() if r.category == category]
