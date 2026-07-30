"""
report_builder.py
=================

Assembles and persists canonical PipelineRunReport manifests.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from loguru import logger

from modules.config import EVAL_RUNS_DIR
from modules.models import PipelineRunReport, QualityEvaluationReport
from evaluation.evaluation_exceptions import ReportPersistError


class ReportBuilder:
    """Assembles and atomically persists PipelineRunReport objects."""

    def __init__(self, runs_dir: Path | None = None) -> None:
        self.runs_dir = Path(runs_dir or EVAL_RUNS_DIR)

    def persist_run_report(self, report: PipelineRunReport) -> Path:
        """Atomically persist PipelineRunReport to data/evaluation/runs/{run_id}/run_manifest.json."""
        target_dir = self.runs_dir / report.run_id
        manifest_path = target_dir / "run_manifest.json"

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            content = report.model_dump_json(indent=2)

            tmp_fd, tmp_path_str = tempfile.mkstemp(
                prefix=".run_manifest_tmp_",
                suffix=".json",
                dir=str(target_dir),
            )
            tmp_path = Path(tmp_path_str)
            try:
                with open(tmp_fd, "w", encoding="utf-8") as f:
                    f.write(content)
                tmp_path.replace(manifest_path)
            finally:
                tmp_path.unlink(missing_ok=True)

            logger.info("Persisted PipelineRunReport to {path}", path=str(manifest_path))
            return manifest_path

        except Exception as exc:
            logger.error("Failed persisting run report {run_id}: {exc}", run_id=report.run_id, exc=exc)
            raise ReportPersistError(f"Failed persisting run report: {exc}") from exc
