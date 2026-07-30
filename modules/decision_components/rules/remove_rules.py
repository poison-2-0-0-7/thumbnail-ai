"""
remove_rules.py
===============

Rule family for generating REMOVE candidate decisions.
"""

from modules.decision_components.io import DecisionInputBundle
from modules.models import CandidateDecision, DecisionAction, DecisionSource, TargetElement


def evaluate_remove_rules(bundle: DecisionInputBundle) -> list[CandidateDecision]:
    """Evaluate REMOVE rules against input bundle."""
    candidates: list[CandidateDecision] = []
    index = bundle.cross_reference_index

    # Rule R1: Low-value clutter text regions flagged by clutter score or M4 OCR
    comp = bundle.intelligence.composition
    is_cluttered = comp and comp.clutter_score > 0.5

    if is_cluttered and bundle.intelligence.ocr and bundle.intelligence.ocr.text_regions:
        for idx, text_reg in enumerate(bundle.intelligence.ocr.text_regions):
            elem_id = f"m4_text_{idx}"
            # Check if text is explicitly requested to be included
            if bundle.redesign_spec.text_overlay and bundle.redesign_spec.text_overlay.include_text:
                continue  # Do not remove text if text overlay is requested

            target = TargetElement(
                element_id=elem_id,
                element_type="text",
                label=text_reg.text,
                bbox=text_reg.bbox,
            )
            candidates.append(
                CandidateDecision(
                    candidate_id=f"cand_remove_text_{elem_id}",
                    target=target,
                    action=DecisionAction.REMOVE,
                    confidence=0.80,
                    source=DecisionSource.RULE,
                    rationale="High visual clutter score; text region not preserved",
                    rule_ids=["rule_remove_cluttered_text"],
                )
            )

    return candidates
