"""
models.py
=========

Data Models and Contracts for Phase 5.6 Iterative Optimization Engine.
Defines:
- StoppingReason (Enum: TARGET_SCORE_REACHED, SCORE_PLATEAU, MAX_ITERATIONS_REACHED, CONFIDENCE_PLATEAU, REPEATED_SUGGESTIONS, TIMED_OUT)
- StoppingPolicy (Configurable thresholds for early stopping)
- IterationResult (Record of a single optimization loop iteration)
- OptimizationHistory (Chronological sequence of IterationResult objects)
- OptimizationReport (Summary report with initial/final scores, improvement curve, render cost)
- OptimizationSession (Master session container)
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import Field

from thumbnail_intelligence.knowledge_base.models import BaseKBModel, _utc_now_iso
from thumbnail_intelligence.evaluation.models import EvaluationSet
from thumbnail_intelligence.ranking.models import RankingResult
from thumbnail_intelligence.critique.models import CritiqueReport
from thumbnail_intelligence.improvement.models import UpdatedRenderExecutionPackage
from thumbnail_intelligence.reasoning.renderer_adapter_models import RenderExecutionPackage


class StoppingReason(str, Enum):
    """Classified reasons for terminating the iterative optimization loop."""

    TARGET_SCORE_REACHED = "target_score_reached"
    SCORE_PLATEAU = "score_plateau"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    CONFIDENCE_PLATEAU = "confidence_plateau"
    REPEATED_SUGGESTIONS = "repeated_suggestions"
    TIMED_OUT = "timed_out"


class StoppingPolicy(BaseKBModel):
    """Configurable thresholds governing early stopping in the iterative optimization loop."""

    target_overall_score: float = Field(90.0, ge=0.0, le=100.0, description="Target score threshold to stop optimization")
    max_iterations: int = Field(3, ge=1, le=10, description="Maximum allowed optimization iterations")
    min_gain_threshold_pts: float = Field(1.0, ge=0.0, description="Minimum overall score gain required over consecutive iterations")
    min_confidence_threshold: float = Field(0.30, ge=0.0, le=1.0, description="Minimum ranking confidence threshold")
    max_elapsed_seconds: float = Field(300.0, gt=0.0, description="Maximum elapsed wall-clock seconds limit")


class IterationResult(BaseKBModel):
    """Record of a single optimization iteration in the closed-loop optimization cycle."""

    iteration_index: int = Field(..., ge=1, description="Iteration number (1-indexed)")
    package_id: str = Field(..., description="RenderExecutionPackage package_id for this iteration")
    candidate_id: str = Field(..., description="Winning candidate_id for this iteration")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Winning candidate overall score")
    ranking_confidence: float = Field(..., ge=0.0, le=1.0, description="Ranking confidence score")
    image_path: str = Field(..., description="Path to rendered thumbnail raster image")
    latency_s: float = Field(..., ge=0.0, description="Wall-clock latency in seconds for this iteration")

    evaluation_set: EvaluationSet = Field(..., description="EvaluationSet for this iteration")
    ranking_result: RankingResult = Field(..., description="RankingResult for this iteration")
    critique_report: CritiqueReport = Field(..., description="CritiqueReport for this iteration")
    updated_package: Optional[UpdatedRenderExecutionPackage] = Field(None, description="Next revised package (None if final iteration)")
    timestamp: str = Field(default_factory=_utc_now_iso, description="ISO timestamp of iteration completion")


class OptimizationHistory(BaseKBModel):
    """Chronological history tracking all IterationResult objects in an optimization session."""

    session_id: str = Field(..., description="Unique optimization session identifier")
    iterations: List[IterationResult] = Field(default_factory=list, description="Sequence of IterationResult records")

    def get_iteration(self, index: int) -> Optional[IterationResult]:
        """Retrieve IterationResult by 1-indexed iteration number."""
        for it in self.iterations:
            if it.iteration_index == index:
                return it
        return None

    def get_best_iteration(self) -> Optional[IterationResult]:
        """Return the IterationResult with the highest overall score."""
        if not self.iterations:
            return None
        return max(self.iterations, key=lambda it: it.overall_score)


class OptimizationReport(BaseKBModel):
    """Summary report detailing iterative optimization outcomes, score curve, and render cost."""

    session_id: str = Field(..., description="Optimization session identifier")
    initial_score: float = Field(..., ge=0.0, le=100.0, description="Initial overall score (Iteration 1)")
    final_score: float = Field(..., ge=0.0, le=100.0, description="Final overall score (Highest score achieved)")
    total_gain_pts: float = Field(..., description="Final score minus initial score")
    total_iterations: int = Field(..., ge=1, description="Total number of iterations executed")
    stopping_reason: StoppingReason = Field(..., description="Reason for stopping optimization")
    stopping_description: str = Field(..., description="Plain text explanation of stopping condition")
    improvement_curve: List[float] = Field(default_factory=list, description="Sequence of overall scores per iteration")
    estimated_render_cost: str = Field("MEDIUM", description="Cumulative render cost estimate (LOW, MEDIUM, HIGH)")
    best_candidate_id: str = Field(..., description="ID of winning candidate from best iteration")
    best_image_path: str = Field(..., description="File path to best thumbnail raster image")


class OptimizationSession(BaseKBModel):
    """Master optimization session container wrapping history, best iteration, and summary report."""

    session_id: str = Field(..., description="Unique optimization session identifier")
    schema_version: str = Field("1.0.0", description="Optimization schema version")
    base_package: RenderExecutionPackage = Field(..., description="Original input RenderExecutionPackage")
    best_package: RenderExecutionPackage = Field(..., description="Optimized best RenderExecutionPackage")
    stopping_policy: StoppingPolicy = Field(..., description="StoppingPolicy used for session")
    history: OptimizationHistory = Field(..., description="Chronological iteration history")
    best_iteration: IterationResult = Field(..., description="IterationResult corresponding to highest score")
    report: OptimizationReport = Field(..., description="Summary report")
    created_at: str = Field(default_factory=_utc_now_iso, description="ISO timestamp of session completion")

    def to_json(self, indent: int = 2) -> str:
        """Serialize OptimizationSession to formatted JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> OptimizationSession:
        """Deserialize OptimizationSession from JSON string."""
        return cls.model_validate(json.loads(json_str))
