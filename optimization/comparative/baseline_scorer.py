"""
optimization/comparative/baseline_scorer.py
============================================

Scores the original source thumbnail using PVQEF's Aggregator to establish a quality baseline.
"""

from __future__ import annotations

from pathlib import Path
from loguru import logger
from pydantic import BaseModel, ConfigDict

from evaluation.quality.aggregator import Aggregator
from evaluation.quality.scoring_context import QualityScoringContext
from optimization.exceptions import BaselineScoringError


class BaselineScore(BaseModel):
    """Quality baseline for an original source thumbnail."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    overall_score: float
    dimension_scores: dict[str, float]
    source_path: str


class BaselineScorer:
    """Evaluates and caches the baseline quality score for an original thumbnail."""

    def __init__(self, aggregator: Aggregator | None = None) -> None:
        self.aggregator = aggregator if aggregator is not None else Aggregator()
        self._cache: dict[str, BaselineScore] = {}

    def score(self, video_id: str, source_thumbnail_path: str | Path) -> BaselineScore:
        """Compute or retrieve cached baseline score for source thumbnail."""
        source_path = Path(source_thumbnail_path)
        cache_key = f"{video_id}_{source_path.name}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        if not source_path.exists():
            # If source file doesn't exist, provide neutral default baseline
            logger.warning("Source thumbnail {path} not found; using default baseline 0.50", path=source_path)
            default_score = BaselineScore(
                video_id=video_id,
                overall_score=0.50,
                dimension_scores={},
                source_path=str(source_path),
            )
            self._cache[cache_key] = default_score
            return default_score

        try:
            context = QualityScoringContext(
                video_id=video_id,
                generated_asset_path=source_path,
                source_thumbnail_path=source_path,
            )
            report = self.aggregator.evaluate(context)
            dim_scores = {ds.dimension: ds.score for ds in report.dimension_scores}

            baseline = BaselineScore(
                video_id=video_id,
                overall_score=report.weighted_overall_score,
                dimension_scores=dim_scores,
                source_path=str(source_path),
            )
            self._cache[cache_key] = baseline
            return baseline
        except Exception as exc:
            logger.error("Failed to compute baseline score for {vid}: {exc}", vid=video_id, exc=exc)
            raise BaselineScoringError(f"Baseline scoring failed for {video_id}: {exc}") from exc
