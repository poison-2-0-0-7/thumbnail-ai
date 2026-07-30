"""
add_rules.py
============

Rule family for generating ADD candidate decisions.
"""

from modules.decision_components.io import DecisionInputBundle
from modules.models import CandidateDecision, DecisionAction, DecisionSource, TargetElement


def evaluate_add_rules(bundle: DecisionInputBundle) -> list[CandidateDecision]:
    """Evaluate ADD rules against input bundle."""
    candidates: list[CandidateDecision] = []

    # Rule A1: Text overlay addition
    text_spec = bundle.redesign_spec.text_overlay
    if text_spec and text_spec.include_text:
        target = TargetElement(
            element_id="add_text_overlay_0",
            element_type="text",
            label="text overlay",
            bbox=text_spec.placement_zone,  # Use placement zone if present
        )
        candidates.append(
            CandidateDecision(
                candidate_id="cand_add_text_overlay",
                target=target,
                action=DecisionAction.ADD,
                confidence=0.85,
                source=DecisionSource.RULE,
                rationale="Explicit text overlay requested in redesign specification",
                rule_ids=["rule_add_text_overlay"],
            )
        )

    # Rule A2: Visual emphasis element (arrow / glow) when CTR score is low
    if bundle.redesign_spec.source_ctr_potential_score < 0.6:
        target = TargetElement(
            element_id="add_emphasis_arrow_0",
            element_type="effect",
            label="directional focal arrow",
            bbox=None,  # Unplaced -- to be routed to LLM/layout
        )
        candidates.append(
            CandidateDecision(
                candidate_id="cand_add_focal_arrow",
                target=target,
                action=DecisionAction.ADD,
                confidence=0.55,  # Low confidence -> triggers LLM ambiguity router
                source=DecisionSource.RULE,
                rationale="Low CTR potential score; proposing focal arrow for eye flow guidance",
                rule_ids=["rule_add_focal_arrow_recommendation"],
            )
        )

    return candidates
