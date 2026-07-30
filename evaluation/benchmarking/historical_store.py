"""
historical_store.py
===================

Append-only JSONL historical benchmark storage.
Mirrors module7_metrics.jsonl pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from modules.config import EVAL_HISTORY_PATH
from modules.models import BenchmarkRecord, PipelineRunReport
from .interfaces import IHistoricalStore


class HistoricalStore(IHistoricalStore):
    """Persists append-only BenchmarkRecord lines in data/evaluation/history/benchmark_history.jsonl."""

    def __init__(self, history_path: Path | None = None) -> None:
        self.history_path = Path(history_path or EVAL_HISTORY_PATH)

    def append(self, record: BenchmarkRecord) -> None:
        """Append one BenchmarkRecord line to history JSONL file."""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            line = record.model_dump_json() + "\n"
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(line)
            logger.info("Appended BenchmarkRecord run_id={run_id} to history", run_id=record.run_id)
        except Exception as exc:
            logger.error("Failed writing to history {path}: {exc}", path=str(self.history_path), exc=exc)

    def load_recent(self, n: int = 5) -> list[BenchmarkRecord]:
        """Load up to N most recent BenchmarkRecord records."""
        if not self.history_path.exists():
            return []

        records: list[BenchmarkRecord] = []
        try:
            lines = self.history_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines[-n:]:
                if line.strip():
                    raw = json.loads(line)
                    records.append(BenchmarkRecord.model_validate(raw))
            return records
        except Exception as exc:
            logger.error("Failed loading history from {path}: {exc}", path=str(self.history_path), exc=exc)
            return []

    def create_record_from_run(self, report: PipelineRunReport) -> BenchmarkRecord:
        """Derive a BenchmarkRecord from a completed PipelineRunReport."""
        weighted_scores = [q.weighted_overall_score for q in report.quality_reports.values()]
        mean_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0

        per_dim: dict[str, list[float]] = {}
        for q in report.quality_reports.values():
            for d in q.dimension_scores:
                per_dim.setdefault(d.dimension, []).append(d.score)

        per_dim_mean = {dim: sum(vals) / len(vals) for dim, vals in per_dim.items() if vals}

        return BenchmarkRecord(
            run_id=report.run_id,
            recorded_at=report.completed_at,
            total_creators=report.total_creators,
            succeeded=report.succeeded,
            skipped=report.skipped,
            mean_weighted_overall_score=mean_score,
            per_dimension_mean_scores=per_dim_mean,
            mean_stage_durations_seconds=report.aggregate_performance,
        )
