"""CandidateRankingEngine: Multi-dimensional weighted ranking engine for candidate thumbnails."""

from __future__ import annotations

import sys
from pathlib import Path

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from typing import Any, Sequence
from loguru import logger
from models import CandidateScore, CandidateStrategy
from module7_exceptions import NoEligibleCandidateError


DEFAULT_RANKING_WEIGHTS: dict[str, float] = {
    "ctr_score": 0.30,
    "readability_score": 0.25,
    "branding_consistency": 0.20,
    "originality_score": 0.15,
    "diversity_bonus": 0.10,
}


class CandidateRankingEngine:
    """Multi-dimensional candidate ranking engine preserving hard-gate guarantees."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or DEFAULT_RANKING_WEIGHTS)
        # Normalize weights so they sum to 1.0
        total_w = sum(self.weights.values())
        if total_w > 0 and abs(total_w - 1.0) > 1e-4:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def compute_dimension_scores(
        self,
        candidate_item: tuple[int, Path, Any, Any, CandidateStrategy, Any, str, dict[str, float]],
        cluster_id: str,
        cluster_survivors: set[int],
        total_candidates: int,
        hash_distances: dict[int, float],
    ) -> dict[str, float]:
        """Compute the 5 ranking dimensions for an eligible candidate."""
        cand_idx = candidate_item[0]
        qa_report = candidate_item[2]
        face_match = candidate_item[3]
        strategy = candidate_item[4]

        # 1. CTR Score: derived from overall QA score and layout visual hierarchy
        overall = getattr(qa_report, "overall_score", 0.0) if qa_report else 0.0
        hier_score = getattr(qa_report, "visual_hierarchy_score", overall) if qa_report else overall
        ctr_score = min(1.0, max(0.0, 0.7 * overall + 0.3 * hier_score))

        # 2. Readability Score: derived from text safe zone score
        text_score = getattr(qa_report, "text_safe_zone_score", 1.0) if qa_report else 1.0
        readability_score = min(1.0, max(0.0, float(text_score)))

        # 3. Branding Consistency: derived from creator face similarity & subject match
        similarity = getattr(face_match, "similarity", 1.0) if face_match else 1.0
        face_passed = getattr(face_match, "passed", True) if face_match else True
        sim_val = float(similarity) if face_passed else 0.0
        branding_consistency = min(1.0, max(0.0, 0.8 * sim_val + 0.2 * overall))

        # 4. Originality Score: derived from hash uniqueness relative to other candidates
        avg_hash_dist = hash_distances.get(cand_idx, 32.0)
        originality_score = min(1.0, max(0.0, avg_hash_dist / 64.0))

        # 5. Diversity Bonus: awarded if candidate strategy is non-faithful and survivor in cluster
        is_survivor = cand_idx in cluster_survivors
        is_novel_strategy = strategy.name != "faithful" if strategy else False
        diversity_bonus = 1.0 if (is_survivor and is_novel_strategy) else (0.5 if is_survivor else 0.0)

        return {
            "ctr_score": round(ctr_score, 4),
            "readability_score": round(readability_score, 4),
            "branding_consistency": round(branding_consistency, 4),
            "originality_score": round(originality_score, 4),
            "diversity_bonus": round(diversity_bonus, 4),
        }

    def rank_candidates(
        self,
        candidates: Sequence[tuple[int, Path, Any, Any, CandidateStrategy, Any, str, dict[str, float]]],
        candidate_cluster_map: dict[int, str] | None = None,
        survivor_indices: list[int] | None = None,
        perceptual_hashes: dict[int, str] | None = None,
    ) -> tuple[tuple[int, Path, Any, Any, CandidateStrategy, Any, str, dict[str, float]], list[CandidateScore]]:
        """
        Rank candidates using multi-dimensional weighted scoring while enforcing QA hard gates.

        Args:
            candidates: List of candidate tuples.
            candidate_cluster_map: Map of candidate_idx -> cluster_id.
            survivor_indices: List of cluster survivor candidate indices.
            perceptual_hashes: Map of candidate_idx -> hex perceptual hash string.

        Returns:
            Tuple of (winning_candidate_tuple, list[CandidateScore]).
        """
        if not candidates:
            raise NoEligibleCandidateError("No candidates provided for ranking")

        cluster_map = candidate_cluster_map or {c[0]: f"cluster_{i+1}" for i, c in enumerate(candidates)}
        survivors = set(survivor_indices or [c[0] for c in candidates])
        hashes = perceptual_hashes or {}

        # Compute pairwise hash distances for originality
        hash_distances: dict[int, float] = {}
        for cand in candidates:
            idx = cand[0]
            cand_hash = hashes.get(idx, "")
            if cand_hash and len(candidates) > 1:
                from modules.generation_components.clustering_engine import hamming_distance
                dists = [
                    hamming_distance(cand_hash, hashes.get(o[0], ""))
                    for o in candidates
                    if o[0] != idx and hashes.get(o[0])
                ]
                hash_distances[idx] = sum(dists) / len(dists) if dists else 32.0
            else:
                hash_distances[idx] = 32.0

        # Hard Gate Filtering - FAILURES NEVER REACH RANKING
        eligible = [c for c in candidates if c[2].hard_gate_passed]

        if not eligible:
            scores = [
                CandidateScore(
                    candidate_index=c[0],
                    overall_score=c[2].overall_score,
                    identity_similarity=c[3].similarity if c[3] else 0.0,
                    hard_gate_passed=False,
                    rank=None,
                    selected=False,
                )
                for c in candidates
            ]
            raise NoEligibleCandidateError("No candidate passed quality assurance hard gates")

        # Evaluate multi-dimensional scores for eligible candidates
        candidate_evals: list[dict[str, Any]] = []

        for cand in eligible:
            cand_idx = cand[0]
            qa_report = cand[2]
            face_match = cand[3]
            strategy = cand[4]
            cluster_id = cluster_map.get(cand_idx, f"cluster_{cand_idx}")

            dim_scores = self.compute_dimension_scores(
                candidate_item=cand,
                cluster_id=cluster_id,
                cluster_survivors=survivors,
                total_candidates=len(candidates),
                hash_distances=hash_distances,
            )

            # Weighted composite score
            composite_score = sum(dim_scores[dim] * self.weights.get(dim, 0.0) for dim in dim_scores)
            composite_score = round(composite_score, 4)

            candidate_evals.append(
                {
                    "candidate_tuple": cand,
                    "candidate_index": cand_idx,
                    "composite_score": composite_score,
                    "dimension_scores": dim_scores,
                    "overall_score": qa_report.overall_score,
                    "identity_similarity": face_match.similarity if face_match else 0.0,
                    "strategy_name": strategy.name if strategy else "faithful",
                    "cluster_id": cluster_id,
                }
            )

        # Sort eligible candidates by (-composite_score, -overall_score, -identity_similarity, candidate_index)
        sorted_evals = sorted(
            candidate_evals,
            key=lambda e: (-e["composite_score"], -e["overall_score"], -e["identity_similarity"], e["candidate_index"]),
        )

        winner_eval = sorted_evals[0]
        winner_tuple = winner_eval["candidate_tuple"]

        rank_map = {eval_item["candidate_index"]: idx + 1 for idx, eval_item in enumerate(sorted_evals)}
        eval_by_idx = {eval_item["candidate_index"]: eval_item for eval_item in candidate_evals}

        candidate_scores: list[CandidateScore] = []
        for cand in candidates:
            cand_idx = cand[0]
            is_eligible = cand[2].hard_gate_passed
            eval_data = eval_by_idx.get(cand_idx)

            if is_eligible and eval_data:
                score_obj = CandidateScore(
                    candidate_index=cand_idx,
                    overall_score=eval_data["composite_score"],
                    identity_similarity=eval_data["identity_similarity"],
                    hard_gate_passed=True,
                    rank=rank_map.get(cand_idx),
                    selected=(cand_idx == winner_tuple[0]),
                )
            else:
                score_obj = CandidateScore(
                    candidate_index=cand_idx,
                    overall_score=cand[2].overall_score,
                    identity_similarity=cand[3].similarity if cand[3] else 0.0,
                    hard_gate_passed=False,
                    rank=None,
                    selected=False,
                )
            candidate_scores.append(score_obj)

        logger.info(
            "CandidateRankingEngine selected winner index={idx} ({strat}) composite_score={score:.4f}",
            idx=winner_tuple[0],
            strat=winner_eval["strategy_name"],
            score=winner_eval["composite_score"],
        )

        return winner_tuple, candidate_scores
