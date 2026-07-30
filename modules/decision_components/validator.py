"""
validator.py
============

Structural and business-rule validator for resolved decision lists.
Implements IDecisionValidator.
"""

from collections import defaultdict
from typing import Any

from modules.decision_components.interfaces import IDecisionValidator
from modules.models import DecisionAction, ResolvedDecision


class DecisionValidator(IDecisionValidator):
    """Validates structural constraints and business rules for resolved decisions."""

    def validate(
        self, decisions: list[ResolvedDecision], bundle: Any = None
    ) -> dict[str, Any]:
        """Validate resolved decisions against structural and business rules."""
        hard_failures: list[str] = []
        soft_warnings: list[str] = []

        if not decisions:
            soft_warnings.append("Resolved decision list is empty")
            return {
                "valid": True,
                "hard_failures": hard_failures,
                "soft_warnings": soft_warnings,
            }

        # 1. Structural Checks
        element_actions: dict[str, set[DecisionAction]] = defaultdict(set)

        for dec in decisions:
            elem_id = dec.target.element_id
            action = dec.action

            # Record action per element_id
            element_actions[elem_id].add(action)

            # Check bbox bounds if present
            if dec.target.bbox is not None:
                b = dec.target.bbox
                if not (0.0 <= b.x_min <= 1.0 and 0.0 <= b.y_min <= 1.0 and 0.0 <= b.x_max <= 1.0 and 0.0 <= b.y_max <= 1.0):
                    hard_failures.append(
                        f"Decision {dec.decision_id} target bbox out of [0, 1] normalized bounds: {b}"
                    )
                if b.x_min >= b.x_max or b.y_min >= b.y_max:
                    hard_failures.append(
                        f"Decision {dec.decision_id} target bbox min coordinates >= max coordinates"
                    )

            # Check ADD decision requirements
            if action == DecisionAction.ADD and not dec.target.label.strip():
                hard_failures.append(f"ADD decision {dec.decision_id} has empty target label")

        # 2. Business-rule Checks (Mutual Exclusion)
        for elem_id, actions in element_actions.items():
            if len(actions) > 1:
                # Same element having mutually exclusive actions post resolution is a hard failure
                if DecisionAction.KEEP in actions and (
                    DecisionAction.REMOVE in actions or DecisionAction.REPLACE in actions
                ):
                    hard_failures.append(
                        f"Element {elem_id} has mutually exclusive actions: {[a.value for a in actions]}"
                    )
                if DecisionAction.REMOVE in actions and DecisionAction.ENHANCE in actions:
                    hard_failures.append(
                        f"Element {elem_id} has mutually exclusive REMOVE and ENHANCE actions"
                    )

        # 3. Coverage Check
        if bundle is not None and hasattr(bundle, "cross_reference_index"):
            index = bundle.cross_reference_index
            decided_ids = set(element_actions.keys())
            for elem_id, meta in index.items():
                if meta.get("element_type") == "face" and elem_id not in decided_ids:
                    soft_warnings.append(
                        f"Detected face element '{elem_id}' ({meta.get('label')}) has no explicit decision"
                    )

        is_valid = len(hard_failures) == 0

        return {
            "valid": is_valid,
            "hard_failures": hard_failures,
            "soft_warnings": soft_warnings,
        }
