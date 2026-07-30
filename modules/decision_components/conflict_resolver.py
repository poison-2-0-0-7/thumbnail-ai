"""
conflict_resolver.py
====================

Resolves same-element decision conflicts using fixed priority hierarchy and mutual exclusion rules.
Implements IConflictResolver.
"""

from collections import defaultdict
from typing import Any

from modules.config import ADD_DEDUP_IOU_THRESHOLD, DECISION_PRIORITY_ORDER
from modules.decision_components.interfaces import IConflictResolver
from modules.decision_exceptions import ConflictResolutionError
from modules.models import BoundingBox, CandidateDecision, DecisionAction, DecisionSource, ResolvedDecision, TargetElement


class ConflictResolver(IConflictResolver):
    """Resolves conflicts among candidate decisions into final ResolvedDecisions."""

    def __init__(self, priority_order: tuple[str, ...] = DECISION_PRIORITY_ORDER) -> None:
        self.priority_order = priority_order
        self._priority_map = {action_str: rank for rank, action_str in enumerate(self.priority_order)}

    def resolve(self, candidates: list[CandidateDecision]) -> list[ResolvedDecision]:
        """Resolve candidate decisions per element_id and apply global deduplication."""
        if not candidates:
            return []

        # Group candidates by target element_id
        grouped: dict[str, list[CandidateDecision]] = defaultdict(list)
        for cand in candidates:
            grouped[cand.target.element_id].append(cand)

        resolved_list: list[ResolvedDecision] = []

        # Step 1: Resolve conflicts per element_id
        for elem_id, cand_group in grouped.items():
            winner, superseded_ids = self._resolve_single_element_group(cand_group)
            priority_rank = self._priority_map.get(winner.action.value, 99)

            resolved = ResolvedDecision(
                decision_id=f"dec_{winner.candidate_id}",
                target=winner.target,
                action=winner.action,
                confidence=winner.confidence,
                source=winner.source,
                rationale=winner.rationale,
                priority_rank=priority_rank,
                superseded_candidate_ids=superseded_ids,
                machine_reasoning={
                    "rule_ids": winner.rule_ids,
                    "priority_rank": priority_rank,
                    "superseded_count": len(superseded_ids),
                },
            )
            resolved_list.append(resolved)

        # Step 2: Global deduplication for ADD candidates based on IoU overlap
        final_resolved = self._deduplicate_add_decisions(resolved_list)
        return final_resolved

    def _resolve_single_element_group(
        self, cand_group: list[CandidateDecision]
    ) -> tuple[CandidateDecision, list[str]]:
        """Select winner candidate within an element group based on priority rank and confidence."""
        if len(cand_group) == 1:
            return cand_group[0], []

        # Sort key: (priority_rank, -confidence, -is_llm)
        def sort_key(c: CandidateDecision) -> tuple[int, float, int]:
            rank = self._priority_map.get(c.action.value, 99)
            is_llm = 1 if c.source == DecisionSource.LLM else 0
            return (rank, -c.confidence, -is_llm)

        sorted_candidates = sorted(cand_group, key=sort_key)
        winner = sorted_candidates[0]
        superseded_ids = [c.candidate_id for c in sorted_candidates[1:]]
        return winner, superseded_ids

    def _deduplicate_add_decisions(
        self, resolved_list: list[ResolvedDecision]
    ) -> list[ResolvedDecision]:
        """Deduplicate ADD decisions proposing near-identical bboxes (IoU > threshold)."""
        add_decisions = [r for r in resolved_list if r.action == DecisionAction.ADD]
        non_add_decisions = [r for r in resolved_list if r.action != DecisionAction.ADD]

        if len(add_decisions) <= 1:
            return resolved_list

        kept_adds: list[ResolvedDecision] = []
        for curr in add_decisions:
            is_duplicate = False
            for existing in kept_adds:
                if (
                    curr.target.label == existing.target.label
                    and curr.target.bbox is not None
                    and existing.target.bbox is not None
                ):
                    iou = self._calculate_iou(curr.target.bbox, existing.target.bbox)
                    if iou > ADD_DEDUP_IOU_THRESHOLD:
                        is_duplicate = True
                        break
            if not is_duplicate:
                kept_adds.append(curr)

        return non_add_decisions + kept_adds

    @staticmethod
    def _calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x_min_inter = max(box1.x_min, box2.x_min)
        y_min_inter = max(box1.y_min, box2.y_min)
        x_max_inter = min(box1.x_max, box2.x_max)
        y_max_inter = min(box1.y_max, box2.y_max)

        inter_width = max(0.0, x_max_inter - x_min_inter)
        inter_height = max(0.0, y_max_inter - y_min_inter)
        inter_area = inter_width * inter_height

        area1 = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)
        area2 = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)
        union_area = area1 + area2 - inter_area

        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)
