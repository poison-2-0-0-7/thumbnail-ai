"""
rule_engine.py
==============

RuleEngine orchestrator evaluating deterministic decision rules across all action families.
Implements IRuleEngine.
"""

from typing import Any

from modules.decision_components.interfaces import IRuleEngine
from modules.decision_components.io import DecisionInputBundle
from modules.decision_components.rules.add_rules import evaluate_add_rules
from modules.decision_components.rules.enhance_rules import evaluate_enhance_rules
from modules.decision_components.rules.keep_rules import evaluate_keep_rules
from modules.decision_components.rules.remove_rules import evaluate_remove_rules
from modules.decision_components.rules.replace_rules import evaluate_replace_rules
from modules.decision_exceptions import RuleEvaluationError
from modules.models import CandidateDecision


class RuleEngine(IRuleEngine):
    """Evaluates all pure rule families against a DecisionInputBundle."""

    def evaluate(self, bundle: Any) -> list[CandidateDecision]:
        """Collect candidate decisions from every rule family."""
        if not isinstance(bundle, DecisionInputBundle):
            raise RuleEvaluationError(f"Expected DecisionInputBundle, got {type(bundle)}")

        candidates: list[CandidateDecision] = []

        try:
            candidates.extend(evaluate_keep_rules(bundle))
            candidates.extend(evaluate_remove_rules(bundle))
            candidates.extend(evaluate_replace_rules(bundle))
            candidates.extend(evaluate_enhance_rules(bundle))
            candidates.extend(evaluate_add_rules(bundle))
            return candidates
        except Exception as exc:
            raise RuleEvaluationError(f"RuleEngine evaluation failed: {exc}") from exc
