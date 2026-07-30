"""
golden_sample_manager.py
========================

Loads fixed golden creator set and manages golden regression suite execution.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from modules.config import EVAL_GOLDEN_DIR
from modules.models import BenchmarkRecord, PipelineRunReport, QualityEvaluationReport
from evaluation.evaluation_exceptions import GoldenSampleInvalidError
from evaluation.pipeline_runner import PipelineRunner
from .historical_store import HistoricalStore
from .regression_detector import detect_regressions, RegressionDetector


class GoldenSampleManager:
    """Manages golden sample manifest, baseline loading, and regression verification."""

    def __init__(self, golden_dir: Path | None = None) -> None:
        self.golden_dir = Path(golden_dir or EVAL_GOLDEN_DIR)

    @property
    def manifest_path(self) -> Path:
        return self.golden_dir / "golden_manifest.json"

    def load_manifest(self) -> dict:
        """Load pinned golden manifest."""
        if not self.manifest_path.exists():
            raise GoldenSampleInvalidError(f"Golden manifest missing at {self.manifest_path}")
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise GoldenSampleInvalidError(f"Failed parsing golden manifest: {exc}") from exc

    def load_baseline(self, video_id: str) -> QualityEvaluationReport | None:
        """Load baseline QualityEvaluationReport for a golden creator."""
        path = self.golden_dir / "baselines" / f"{video_id}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return QualityEvaluationReport.model_validate(raw)
        except Exception as exc:
            logger.warning("Failed loading baseline for {vid}: {exc}", vid=video_id, exc=exc)
            return None

    def run_golden_suite(self, runner: PipelineRunner | None = None) -> PipelineRunReport:
        """Run validation pipeline across golden creators and detect regressions against baseline."""
        manifest = self.load_manifest()
        csv_path = self.golden_dir / manifest.get("csv_rel_path", "golden_creators.csv")

        p_runner = runner or PipelineRunner()
        report = p_runner.run(csv_path=csv_path, golden_only=True)

        baseline_rec = BenchmarkRecord(
            run_id="golden_baseline",
            recorded_at=manifest.get("updated_at", "2026-07-30T00:00:00Z"),
            total_creators=manifest.get("total_creators", report.total_creators),
            succeeded=manifest.get("total_creators", report.total_creators),
            skipped=0,
            mean_weighted_overall_score=manifest.get("expected_mean_score", 0.85),
            per_dimension_mean_scores=manifest.get("expected_dimension_scores", {}),
        )

        regressions = detect_regressions(report, baseline_rec)
        updated_report = report.model_copy(update={"regressions": regressions})
        return updated_report


def run_golden_regression_suite() -> PipelineRunReport:
    """Public helper function to execute the golden regression suite."""
    return GoldenSampleManager().run_golden_suite()
