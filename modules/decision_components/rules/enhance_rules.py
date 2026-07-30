"""
enhance_rules.py
================

Rule family for generating ENHANCE candidate decisions.
"""

from modules.decision_components.io import DecisionInputBundle
from modules.models import CandidateDecision, DecisionAction, DecisionSource, TargetElement


def evaluate_enhance_rules(bundle: DecisionInputBundle) -> list[CandidateDecision]:
    """Evaluate ENHANCE rules against input bundle."""
    candidates: list[CandidateDecision] = []

    # Rule E1: Global contrast and saturation enhancement
    colors = bundle.intelligence.colors
    if colors:
        if colors.contrast < 0.4:
            target = TargetElement(
                element_id="global_contrast",
                element_type="lighting",
                label="global image contrast",
                bbox=None,
            )
            candidates.append(
                CandidateDecision(
                    candidate_id="cand_enhance_contrast",
                    target=target,
                    action=DecisionAction.ENHANCE,
                    confidence=0.82,
                    source=DecisionSource.RULE,
                    rationale=f"Image contrast ({colors.contrast:.2f}) below threshold 0.40",
                    rule_ids=["rule_enhance_global_contrast"],
                )
            )

        if colors.saturation < 0.35:
            target = TargetElement(
                element_id="global_saturation",
                element_type="color",
                label="global color saturation",
                bbox=None,
            )
            candidates.append(
                CandidateDecision(
                    candidate_id="cand_enhance_saturation",
                    target=target,
                    action=DecisionAction.ENHANCE,
                    confidence=0.78,
                    source=DecisionSource.RULE,
                    rationale=f"Image saturation ({colors.saturation:.2f}) below threshold 0.35",
                    rule_ids=["rule_enhance_global_saturation"],
                )
            )

    return candidates
