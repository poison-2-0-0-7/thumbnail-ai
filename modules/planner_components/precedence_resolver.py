"""
precedence_resolver.py
======================

Implements Section 6.3 documented precedence rules across upstream artifacts.
Pure logic, zero I/O.
"""

from __future__ import annotations

from typing import Optional

from models import (
    CompositionWorkspace,
    DecisionManifest,
)
from planner_components.interfaces import IPrecedenceResolver


class PrecedenceResolver(IPrecedenceResolver):
    """Resolves precedence between Module 9 DecisionManifest and Module 10 fallback decisions."""

    def resolve_layer_decisions(
        self,
        workspace: CompositionWorkspace,
        decision_manifest: Optional[DecisionManifest] = None,
    ) -> list[tuple[str, str, str, str]]:
        """
        Returns list of (element_key, role_str, decision_str, rationale).

        Precedence (§6.3):
        Uses DecisionManifest when present and status != 'error', else falls back
        to CompositionWorkspace layer placement decisions.
        """
        if decision_manifest is not None and decision_manifest.status != "error" and decision_manifest.decisions:
            result: list[tuple[str, str, str, str]] = []
            for d in decision_manifest.decisions:
                if hasattr(d, "target") and d.target:
                    element_key = d.target.element_id
                    role_str = d.target.element_type
                else:
                    element_key = getattr(d, "target_element_key", "element")
                    role_str = str(getattr(d, "role", "object"))

                action_str = d.action.value if hasattr(d.action, "value") else str(d.action)
                result.append((element_key, role_str, action_str, d.rationale))
            return result

        result: list[tuple[str, str, str, str]] = []
        for layer in workspace.layers:
            element_key = layer.placement.asset_id or layer.layer_id
            role_str = layer.placement.role.value
            decision_str = layer.placement.decision.value
            rationale = layer.placement.rationale
            result.append((element_key, role_str, decision_str, rationale))
        return result
