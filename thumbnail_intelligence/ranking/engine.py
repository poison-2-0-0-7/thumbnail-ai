"""
engine.py
=========

CandidateRankingEngine Implementation for Phase 5.3.
Consumes an EvaluationSet and determines the objectively best thumbnail candidate.
Does NOT generate thumbnails. Does NOT evaluate thumbnails. ONLY ranks them based on EvaluationSet.

Provides:
- Configurable RankingProfile & RankingPolicy (OVERALL_SCORE, WEIGHTED_METRIC, COMPOSITE_CTR)
- Deterministic tie-breaking across configurable priority metrics (NO random selection)
- 3-component Ranking Confidence estimation (Score margin, Metric consistency, Propagated confidence)
- Explainable pairwise comparisons detailing WHY Winner beat Runner-Up
- Structured RankingReport with metric comparison table and tie-break logs
- Strict input validation (empty sets, duplicates, NaN values, missing metrics)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from thumbnail_intelligence.evaluation.models import EvaluationResult, EvaluationSet
from thumbnail_intelligence.ranking.models import (
    MetricComparison,
    RankedCandidate,
    RankingExplanation,
    RankingPolicy,
    RankingProfile,
    RankingReport,
    RankingResult,
)

logger = logging.getLogger(__name__)


class RankingEngineError(RuntimeError):
    """Exception raised for ranking engine errors or invalid inputs."""
    pass


class CandidateRankingEngine:
    """Deterministic, explainable ranking engine for thumbnail evaluation sets."""

    def __init__(self, profile: Optional[RankingProfile] = None) -> None:
        self.profile = profile or RankingProfile()

    def rank_evaluation_set(
        self,
        eval_set: EvaluationSet,
        profile: Optional[RankingProfile] = None,
        top_n: Optional[int] = None,
    ) -> RankingResult:
        """Rank all candidates in an EvaluationSet and return a RankingResult.

        Args:
            eval_set: Input EvaluationSet containing EvaluationResult objects.
            profile: Optional RankingProfile override.
            top_n: Optional Top-N selection count override.

        Returns:
            Master RankingResult containing ordered candidates, winner, runner-up, ranking confidence,
            explanation, and structured report.
        """
        prof = profile or self.profile
        top_count = top_n or prof.top_n_default

        # 1. Pre-flight Validation
        self.validate_evaluation_set(eval_set)

        results = list(eval_set.results)
        logger.info(f"=== Starting CandidateRankingEngine for set '{eval_set.set_id}' ({len(results)} candidates, policy='{prof.policy.value}') ===")

        # 2. Sort candidates deterministically with tie-breaking
        ranked_entries, tie_logs = self._sort_candidates(results, prof)

        # 3. Extract Winner & Runner-up
        winner = ranked_entries[0]
        runner_up = ranked_entries[1] if len(ranked_entries) > 1 else None
        top_n_list = ranked_entries[: min(top_count, len(ranked_entries))]

        # 4. Compute Ranking Confidence
        confidence = self.compute_confidence(winner.evaluation_result, runner_up.evaluation_result if runner_up else None, prof)

        # 5. Generate Pairwise Explanation (Winner vs Runner-up)
        if runner_up:
            explanation = self.explain_ranking(winner.evaluation_result, runner_up.evaluation_result)
        else:
            explanation = RankingExplanation(
                winner_candidate_id=winner.candidate_id,
                runner_up_candidate_id="N/A",
                score_delta=0.0,
                strengths=["Only candidate in EvaluationSet; default winner."],
                weaknesses=[],
                metric_comparisons=[],
                summary_reasoning=f"{winner.candidate_label} ranked 1st as the single candidate evaluated.",
            )

        # 6. Generate Metric Comparison Matrix & Ranking Report
        matrix: Dict[str, Dict[str, float]] = {}
        for entry in ranked_entries:
            matrix[entry.candidate_id] = {
                "overall_score": entry.final_score,
                "ctr_score": entry.evaluation_result.metrics["estimated_ctr_score"].score,
                "readability": entry.evaluation_result.metrics["text_readability"].score,
                "saliency": entry.evaluation_result.metrics["subject_saliency"].score,
                "face_size": entry.evaluation_result.metrics["face_size"].score,
                "clarity": entry.evaluation_result.metrics["thumbnail_clarity"].score,
            }

        winner_summary = {
            "candidate_id": winner.candidate_id,
            "label": winner.candidate_label,
            "overall_score": winner.final_score,
            "top_metric": max(winner.evaluation_result.metrics.values(), key=lambda m: m.score).metric_name,
        }

        runner_up_summary = None
        if runner_up:
            runner_up_summary = {
                "candidate_id": runner_up.candidate_id,
                "label": runner_up.candidate_label,
                "overall_score": runner_up.final_score,
                "score_gap": round(winner.final_score - runner_up.final_score, 2),
            }

        report = RankingReport(
            set_id=eval_set.set_id,
            ranking_policy_used=prof.policy.value,
            winner_summary=winner_summary,
            runner_up_summary=runner_up_summary,
            tie_break_log=tie_logs,
            metric_comparison_matrix=matrix,
        )

        res = RankingResult(
            set_id=eval_set.set_id,
            schema_version="1.0.0",
            ranked_candidates=ranked_entries,
            winner=winner,
            runner_up=runner_up,
            top_n=top_n_list,
            ranking_confidence=round(confidence, 3),
            explanation=explanation,
            report=report,
        )

        logger.info(f"=== Completed CandidateRankingEngine for set '{eval_set.set_id}' (Winner: '{winner.candidate_id}' score={winner.final_score:.2f}, conf={confidence:.2f}) ===")
        return res

    def validate_evaluation_set(self, eval_set: EvaluationSet) -> None:
        """Validate input EvaluationSet integrity before ranking."""
        if not eval_set:
            raise RankingEngineError("EvaluationSet cannot be None.")

        if not eval_set.results or len(eval_set.results) == 0:
            raise RankingEngineError("EvaluationSet contains no candidate results to rank.")

        seen_ids = set()
        for res in eval_set.results:
            if not res.candidate_id:
                raise RankingEngineError("EvaluationResult has empty candidate_id.")

            if res.candidate_id in seen_ids:
                raise RankingEngineError(f"Duplicate candidate_id '{res.candidate_id}' found in EvaluationSet.")
            seen_ids.add(res.candidate_id)

            if math.isnan(res.overall_score) or math.isinf(res.overall_score):
                raise RankingEngineError(f"Candidate '{res.candidate_id}' has invalid NaN or Inf overall score: {res.overall_score}")

            if res.overall_score < 0.0 or res.overall_score > 100.0:
                raise RankingEngineError(f"Candidate '{res.candidate_id}' overall score {res.overall_score} is out of valid range [0.0, 100.0].")

            if not res.metrics or len(res.metrics) < 22:
                raise RankingEngineError(f"Candidate '{res.candidate_id}' is missing required 22 evaluation metrics (found {len(res.metrics) if res.metrics else 0}).")

    def _sort_candidates(self, results: List[EvaluationResult], prof: RankingProfile) -> Tuple[List[RankedCandidate], List[str]]:
        """Sort EvaluationResult list deterministically using primary policy and tie-breaking priority."""
        tie_logs: List[str] = []

        def score_key(res: EvaluationResult) -> float:
            if prof.policy == RankingPolicy.COMPOSITE_CTR:
                return res.metrics["estimated_ctr_score"].score
            elif prof.policy == RankingPolicy.WEIGHTED_METRIC:
                return res.weighted_score
            else:
                return res.overall_score

        # Sort initially by score descending
        initial_sorted = sorted(results, key=score_key, reverse=True)

        # Bubble-sort pass with deterministic tie-breaking for equal scores
        n = len(initial_sorted)
        sorted_res = list(initial_sorted)

        for i in range(n):
            for j in range(0, n - i - 1):
                res_a = sorted_res[j]
                res_b = sorted_res[j + 1]
                s_a = score_key(res_a)
                s_b = score_key(res_b)

                if abs(s_a - s_b) <= prof.tie_threshold_pts:
                    # Score tie! Perform tie-break resolution
                    winner_res, log_msg = self._resolve_tie(res_a, res_b, prof)
                    if log_msg:
                        tie_logs.append(log_msg)

                    if winner_res.candidate_id == res_b.candidate_id:
                        # Swap if res_b beat res_a in tie-break
                        sorted_res[j], sorted_res[j + 1] = sorted_res[j + 1], sorted_res[j]

        # Build RankedCandidate entries
        ranked_entries: List[RankedCandidate] = []
        for rank_idx, res in enumerate(sorted_res, start=1):
            f_score = score_key(res)

            # Check if this candidate position was tie-broken
            is_tie_broken = False
            tb_reason = None

            if rank_idx > 1:
                prev_res = sorted_res[rank_idx - 2]
                if abs(score_key(prev_res) - f_score) <= prof.tie_threshold_pts:
                    is_tie_broken = True
                    tb_reason = f"Position resolved via tie-break metrics against {prev_res.candidate_id}"

            ranked_entries.append(
                RankedCandidate(
                    rank=rank_idx,
                    candidate_id=res.candidate_id,
                    candidate_label=res.candidate_label,
                    final_score=round(f_score, 2),
                    tie_broken=is_tie_broken,
                    tie_break_reason=tb_reason,
                    evaluation_result=res,
                )
            )

        return ranked_entries, tie_logs

    def _resolve_tie(self, res_a: EvaluationResult, res_b: EvaluationResult, prof: RankingProfile) -> Tuple[EvaluationResult, Optional[str]]:
        """Deterministically break tie between candidate A and B using priority metrics."""
        for metric_name in prof.tie_break_priority:
            score_a = res_a.metrics[metric_name].score
            score_b = res_b.metrics[metric_name].score

            if abs(score_a - score_b) > 1e-4:
                winner = res_a if score_a > score_b else res_b
                loser = res_b if score_a > score_b else res_a
                diff = abs(score_a - score_b)
                log_msg = f"Tie between '{res_a.candidate_id}' and '{res_b.candidate_id}' broken by '{metric_name}' ({winner.candidate_id}: {max(score_a, score_b):.1f} vs {loser.candidate_id}: {min(score_a, score_b):.1f}, delta={diff:.1f} pts)."
                return winner, log_msg

        # Lexicographical fallback tie-break
        winner = res_a if res_a.candidate_id < res_b.candidate_id else res_b
        log_msg = f"Tie between '{res_a.candidate_id}' and '{res_b.candidate_id}' broken by candidate_id lexicographical order ('{winner.candidate_id}' wins)."
        return winner, log_msg

    def explain_ranking(self, winner: EvaluationResult, runner_up: EvaluationResult) -> RankingExplanation:
        """Generate human-readable pairwise explanation of why Winner beat Runner-Up."""
        strengths: List[str] = []
        weaknesses: List[str] = []
        comparisons: List[MetricComparison] = []

        delta_overall = winner.overall_score - runner_up.overall_score

        for m_name, m_win in winner.metrics.items():
            if m_name in runner_up.metrics:
                m_run = runner_up.metrics[m_name]
                d = m_win.score - m_run.score

                if d > 0.01:
                    adv = "winner"
                elif d < -0.01:
                    adv = "runner_up"
                else:
                    adv = "neutral"

                comparisons.append(
                    MetricComparison(
                        metric_name=m_name,
                        candidate_a_score=m_win.score,
                        candidate_b_score=m_run.score,
                        delta=round(d, 2),
                        advantage=adv,
                    )
                )

                if d >= 2.0:
                    strengths.append(f"+ Higher {m_name.replace('_', ' ').title()} (+{d:.1f} pts)")
                elif d <= -2.0:
                    weaknesses.append(f"- Slightly lower {m_name.replace('_', ' ').title()} ({d:.1f} pts)")

        # Build summary plain-text explanation
        top_str = strengths[:3] if strengths else ["Equal metric performance across major categories"]
        top_str_text = ", ".join(top_str)
        summary = (
            f"{winner.candidate_label} ranked 1st over {runner_up.candidate_label} "
            f"with a score lead of +{delta_overall:.2f} pts ({winner.overall_score:.1f} vs {runner_up.overall_score:.1f}). "
            f"Key advantages: {top_str_text}."
        )

        return RankingExplanation(
            winner_candidate_id=winner.candidate_id,
            runner_up_candidate_id=runner_up.candidate_id,
            score_delta=round(delta_overall, 2),
            strengths=strengths,
            weaknesses=weaknesses,
            metric_comparisons=comparisons,
            summary_reasoning=summary,
        )

    def compute_confidence(
        self,
        winner: EvaluationResult,
        runner_up: Optional[EvaluationResult],
        profile: Optional[RankingProfile] = None,
    ) -> float:
        """Compute ranking confidence score based on score separation, metric consistency, and propagated confidence."""
        if not runner_up:
            return 1.0

        prof = profile or self.profile
        delta = max(0.0, winner.overall_score - runner_up.overall_score)

        # 1. Score separation confidence component (0.0 to 1.0)
        c_sep = min(1.0, delta / prof.max_confidence_margin_pts)

        # 2. Metric consistency component (fraction of individual metrics won)
        win_count = 0
        total_count = 0
        for m_name, m_win in winner.metrics.items():
            if m_name in runner_up.metrics:
                total_count += 1
                if m_win.score >= runner_up.metrics[m_name].score:
                    win_count += 1

        c_cons = win_count / float(total_count) if total_count > 0 else 0.5

        # 3. Propagated confidence component
        c_prop = (winner.confidence + runner_up.confidence) / 2.0

        # Weighted combination
        confidence = 0.50 * c_sep + 0.35 * c_cons + 0.15 * c_prop
        return max(0.0, min(1.0, float(confidence)))
