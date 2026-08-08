"""
engine.py
=========

IterativeOptimizationEngine Implementation for Phase 5.6.
Orchestrates closed-loop iterative optimization of thumbnail assets until quality converges.

Pipeline loop per iteration:
  Generate ➔ Evaluate ➔ Rank ➔ Critique ➔ Improve ➔ Render ➔ Evaluate Again ➔ Repeat

Does NOT replace any existing module. Reuses RendererV2Pipeline, MultiCandidateGenerator,
ThumbnailEvaluationEngine, CandidateRankingEngine, IntelligentCritiqueEngine, and AutomaticImprovementEngine.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from thumbnail_intelligence.critique.engine import IntelligentCritiqueEngine
from thumbnail_intelligence.evaluation.engine import ThumbnailEvaluationEngine
from thumbnail_intelligence.improvement.engine import AutomaticImprovementEngine
from thumbnail_intelligence.optimization.convergence import ConvergenceDetector
from thumbnail_intelligence.optimization.models import (
    IterationResult,
    OptimizationHistory,
    OptimizationReport,
    OptimizationSession,
    StoppingPolicy,
    StoppingReason,
)
from thumbnail_intelligence.ranking.engine import CandidateRankingEngine
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.execution_plan_models import ExecutionPlan
from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner
from thumbnail_intelligence.reasoning.multi_candidate_generator import MultiCandidateGenerator
from thumbnail_intelligence.reasoning.renderer_adapter import RendererV2Adapter
from thumbnail_intelligence.reasoning.renderer_adapter_models import RenderExecutionPackage
from thumbnail_intelligence.reasoning.spatial_composition_models import SpatialComposition
from thumbnail_intelligence.reasoning.spatial_composition_planner import SpatialCompositionPlanner
from renderer_v2.pipeline import RendererV2Pipeline

logger = logging.getLogger(__name__)


class OptimizationEngineError(RuntimeError):
    """Exception raised for iterative optimization engine failures."""
    pass


class IterativeOptimizationEngine:
    """Orchestrates closed-loop iterative optimization until thumbnail quality converges."""

    def __init__(
        self,
        detector: Optional[ConvergenceDetector] = None,
        generator: Optional[MultiCandidateGenerator] = None,
        eval_engine: Optional[ThumbnailEvaluationEngine] = None,
        rank_engine: Optional[CandidateRankingEngine] = None,
        critique_engine: Optional[IntelligentCritiqueEngine] = None,
        improvement_engine: Optional[AutomaticImprovementEngine] = None,
        pipeline: Optional[RendererV2Pipeline] = None,
    ) -> None:
        self.detector = detector or ConvergenceDetector()
        self.pipeline = pipeline or RendererV2Pipeline()
        self.generator = generator or MultiCandidateGenerator(pipeline=self.pipeline)
        self.eval_engine = eval_engine or ThumbnailEvaluationEngine()
        self.rank_engine = rank_engine or CandidateRankingEngine()
        self.critique_engine = critique_engine or IntelligentCritiqueEngine()
        self.improvement_engine = improvement_engine or AutomaticImprovementEngine()

    def optimize_package(
        self,
        base_package: RenderExecutionPackage,
        policy: Optional[StoppingPolicy] = None,
        output_directory: Optional[Union[str, Path]] = None,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> OptimizationSession:
        """Execute closed-loop iterative optimization starting from a base RenderExecutionPackage.

        Args:
            base_package: Input RenderExecutionPackage.
            policy: Optional StoppingPolicy governing early stopping thresholds.
            output_directory: Directory to save rendered thumbnail rasters.
            context_overrides: Additional runtime execution metadata overrides.

        Returns:
            OptimizationSession containing complete history, best package, and summary report.
        """
        if not base_package:
            raise OptimizationEngineError("Input base_package cannot be None.")

        stop_policy = policy or StoppingPolicy()
        session_id = f"session_{uuid.uuid4().hex[:8]}"

        # Setup output directory
        if output_directory is not None:
            session_out_dir = Path(output_directory) / session_id
        else:
            session_out_dir = Path(tempfile.mkdtemp(prefix=f"opt_{session_id}_"))
        session_out_dir.mkdir(parents=True, exist_ok=True)

        history = OptimizationHistory(session_id=session_id, iterations=[])
        current_package = base_package
        start_time = time.time()

        final_stop_reason = StoppingReason.MAX_ITERATIONS_REACHED
        final_stop_desc = f"Maximum allowed iterations ({stop_policy.max_iterations}) executed."

        logger.info(f"=== Starting IterativeOptimizationEngine Session '{session_id}' (target_score={stop_policy.target_overall_score}, max_iter={stop_policy.max_iterations}) ===")

        for k in range(1, stop_policy.max_iterations + 1):
            t_iter_start = time.time()
            it_dir = session_out_dir / f"iteration_{k:02d}"
            it_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"--- Iteration {k}/{stop_policy.max_iterations} (Package: '{current_package.metadata.package_id}') ---")

            # 1. Generate Candidates
            cand_set = self.generator.generate_candidates(
                base_package=current_package,
                count=5,
                output_directory=it_dir,
                context_overrides=context_overrides,
            )

            # 2. Evaluate Candidates
            eval_set = self.eval_engine.evaluate_candidate_set(cand_set)

            # 3. Rank Candidates
            rank_res = self.rank_engine.rank_evaluation_set(eval_set)

            # 4. Critique Winner
            critique = self.critique_engine.critique_ranking_result(rank_res)

            # Retrieve Winner Details
            winner_cand = cand_set.get_candidate(rank_res.winner.candidate_id)
            win_img_path = winner_cand.image_path if winner_cand else str(it_dir / "thumbnail_candidate_a.png")
            win_score = rank_res.winner.final_score

            iter_latency = time.time() - t_iter_start
            elapsed_total = time.time() - start_time

            # Record IterationResult
            it_record = IterationResult(
                iteration_index=k,
                package_id=current_package.metadata.package_id,
                candidate_id=rank_res.winner.candidate_id,
                overall_score=win_score,
                ranking_confidence=rank_res.ranking_confidence,
                image_path=win_img_path,
                latency_s=round(iter_latency, 2),
                evaluation_set=eval_set,
                ranking_result=rank_res,
                critique_report=critique,
                updated_package=None,
            )
            history.iterations.append(it_record)

            # Check Convergence & Stopping Conditions
            should_stop, stop_reason, stop_desc = self.detector.check_convergence(history, stop_policy, elapsed_seconds=elapsed_total)

            if should_stop or k == stop_policy.max_iterations:
                final_stop_reason = stop_reason or StoppingReason.MAX_ITERATIONS_REACHED
                final_stop_desc = stop_desc
                logger.info(f"Optimization loop stopping after iteration {k}: {final_stop_desc}")
                break

            # 5. Improve Package for next iteration
            updated_pkg_res = self.improvement_engine.improve_package(current_package, critique.improvement_plan)
            history.iterations[-1] = history.iterations[-1].model_copy(update={"updated_package": updated_pkg_res})
            current_package = updated_pkg_res.package

        # Assemble Session & Final Summary Report
        best_it = history.get_best_iteration()
        if not best_it:
            raise OptimizationEngineError("Optimization session failed to produce any valid iteration results.")

        initial_score = history.iterations[0].overall_score
        final_score = best_it.overall_score
        total_gain = final_score - initial_score
        curve = [it.overall_score for it in history.iterations]

        best_pkg = base_package
        for it in history.iterations:
            if it.iteration_index == best_it.iteration_index and it.updated_package:
                best_pkg = it.updated_package.package

        render_cost = "LOW" if len(history.iterations) <= 2 else ("MEDIUM" if len(history.iterations) <= 4 else "HIGH")

        report = OptimizationReport(
            session_id=session_id,
            initial_score=round(initial_score, 2),
            final_score=round(final_score, 2),
            total_gain_pts=round(total_gain, 2),
            total_iterations=len(history.iterations),
            stopping_reason=final_stop_reason,
            stopping_description=final_stop_desc,
            improvement_curve=curve,
            estimated_render_cost=render_cost,
            best_candidate_id=best_it.candidate_id,
            best_image_path=best_it.image_path,
        )

        session = OptimizationSession(
            session_id=session_id,
            schema_version="1.0.0",
            base_package=base_package,
            best_package=best_pkg,
            stopping_policy=stop_policy,
            history=history,
            best_iteration=best_it,
            report=report,
        )

        logger.info(
            f"=== Completed IterativeOptimizationEngine Session '{session_id}' "
            f"(Initial: {initial_score:.1f} ➔ Final: {final_score:.1f}, gain=+{total_gain:.1f} pts in {len(history.iterations)} iterations) ==="
        )
        return session

    def optimize_brief(
        self,
        brief: DesignBrief,
        policy: Optional[StoppingPolicy] = None,
        output_directory: Optional[Union[str, Path]] = None,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> OptimizationSession:
        """Convenience method: Optimize iteratively starting from a DesignBrief."""
        plan = ExecutionPlanner().plan(brief)
        comp = SpatialCompositionPlanner().plan(plan, brief)
        package = RendererV2Adapter().translate(comp, plan)
        return self.optimize_package(package, policy=policy, output_directory=output_directory, context_overrides=context_overrides)

    def optimize_composition(
        self,
        composition: SpatialComposition,
        plan: ExecutionPlan,
        policy: Optional[StoppingPolicy] = None,
        output_directory: Optional[Union[str, Path]] = None,
        context_overrides: Optional[Dict[str, Any]] = None,
    ) -> OptimizationSession:
        """Convenience method: Optimize iteratively starting from SpatialComposition + ExecutionPlan."""
        package = RendererV2Adapter().translate(composition, plan)
        return self.optimize_package(package, policy=policy, output_directory=output_directory, context_overrides=context_overrides)
