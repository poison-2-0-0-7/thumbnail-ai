"""
keep_rules.py
=============

Rule family for generating KEEP candidate decisions.
"""

from typing import Any

from modules.decision_components.io import DecisionInputBundle
from modules.models import CandidateDecision, DecisionAction, DecisionSource, TargetElement


def evaluate_keep_rules(bundle: DecisionInputBundle) -> list[CandidateDecision]:
    """Evaluate KEEP rules against input bundle."""
    candidates: list[CandidateDecision] = []
    index = bundle.cross_reference_index

    # Rule K1: Explicit elements_to_preserve from RedesignSpecification
    for elem_id, meta in index.items():
        label = meta.get("label", "").lower()
        for preserve_target in bundle.redesign_spec.elements_to_preserve:
            if preserve_target.lower() in label or label in preserve_target.lower():
                target = TargetElement(
                    element_id=elem_id,
                    element_type=meta.get("element_type", "object"),
                    label=meta.get("label", preserve_target),
                    bbox=meta.get("bbox"),
                )
                candidates.append(
                    CandidateDecision(
                        candidate_id=f"cand_keep_preserve_{elem_id}",
                        target=target,
                        action=DecisionAction.KEEP,
                        confidence=0.95,
                        source=DecisionSource.RULE,
                        rationale=f"Matches explicit elements_to_preserve entry '{preserve_target}'",
                        rule_ids=["rule_keep_explicit_preserve"],
                    )
                )

    # Rule K2: Primary creator face preservation
    for elem_id, meta in index.items():
        if meta.get("element_type") == "face" and "creator" in meta.get("label", "").lower():
            # Avoid duplicate if already added by K1
            if not any(c.target.element_id == elem_id for c in candidates):
                target = TargetElement(
                    element_id=elem_id,
                    element_type="face",
                    label=meta.get("label", "creator face"),
                    bbox=meta.get("bbox"),
                )
                candidates.append(
                    CandidateDecision(
                        candidate_id=f"cand_keep_face_{elem_id}",
                        target=target,
                        action=DecisionAction.KEEP,
                        confidence=0.90,
                        source=DecisionSource.RULE,
                        rationale="Primary creator face detected in ThumbnailIntelligence",
                        rule_ids=["rule_keep_primary_face"],
                    )
                )

    return candidates
