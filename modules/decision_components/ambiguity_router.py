"""
ambiguity_router.py
===================

Routes candidate decisions needing LLM adjudication based on confidence thresholds,
same-element action conflicts, and unplaced ADD recommendations.
Implements IAmbiguityRouter.
"""

from collections import defaultdict

from modules.config import AMBIGUITY_CONFIDENCE_THRESHOLD
from modules.decision_components.interfaces import IAmbiguityRouter
from modules.models import CandidateDecision, DecisionAction


class AmbiguityRouter(IAmbiguityRouter):
    """Selects ambiguous candidate decisions that require LLM adjudication."""

    def __init__(self, threshold: float = AMBIGUITY_CONFIDENCE_THRESHOLD) -> None:
        self.threshold = threshold

    def select(
        self, candidates: list[CandidateDecision]
    ) -> tuple[list[CandidateDecision], list[CandidateDecision]]:
        """Split candidates into (confident, needs_llm_review)."""
        if not candidates:
            return [], []

        # Map element_id -> list of candidate decisions
        element_candidates: dict[str, list[CandidateDecision]] = defaultdict(list)
        for cand in candidates:
            element_candidates[cand.target.element_id].append(cand)

        # Identify element_ids with conflicting actions
        conflicting_element_ids: set[str] = set()
        for elem_id, cand_list in element_candidates.items():
            distinct_actions = {c.action for c in cand_list}
            if len(distinct_actions) > 1:
                conflicting_element_ids.add(elem_id)

        confident: list[CandidateDecision] = []
        needs_llm: list[CandidateDecision] = []

        for cand in candidates:
            elem_id = cand.target.element_id
            is_low_confidence = cand.confidence < self.threshold
            has_conflict = elem_id in conflicting_element_ids
            is_unplaced_add = cand.action == DecisionAction.ADD and cand.target.bbox is None

            if is_low_confidence or has_conflict or is_unplaced_add:
                needs_llm.append(cand)
            else:
                confident.append(cand)

        return confident, needs_llm
