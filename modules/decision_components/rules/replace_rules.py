"""
replace_rules.py
================

Rule family for generating REPLACE candidate decisions.
"""

from modules.decision_components.io import DecisionInputBundle
from modules.models import CandidateDecision, DecisionAction, DecisionSource, TargetElement


def evaluate_replace_rules(bundle: DecisionInputBundle) -> list[CandidateDecision]:
    """Evaluate REPLACE rules against input bundle."""
    candidates: list[CandidateDecision] = []
    index = bundle.cross_reference_index

    # Rule REP1: Background replacement triggered by color temperature flip or low contrast
    color_dir = bundle.redesign_spec.color_direction
    if color_dir and color_dir.warm_or_cool:
        bg_elem_id = "global_background"
        target = TargetElement(
            element_id=bg_elem_id,
            element_type="background",
            label="background environment",
            bbox=None,
        )
        candidates.append(
            CandidateDecision(
                candidate_id="cand_replace_bg_color",
                target=target,
                action=DecisionAction.REPLACE,
                confidence=0.75,
                source=DecisionSource.RULE,
                rationale=f"Color temperature shift required ({color_dir.warm_or_cool})",
                rule_ids=["rule_replace_background_color_flip"],
            )
        )

    return candidates
