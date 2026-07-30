"""
test_benchmarking.py
====================

Unit tests for HistoricalStore and BenchmarkRecord persistence.
"""

from pathlib import Path
import pytest

from evaluation.benchmarking import HistoricalStore
from modules.models import BenchmarkRecord


def test_historical_store_append_and_load(tmp_path):
    hist_file = tmp_path / "history.jsonl"
    store = HistoricalStore(history_path=hist_file)

    rec1 = BenchmarkRecord(
        run_id="run_1",
        recorded_at="2026-07-30T00:00:00Z",
        total_creators=5,
        succeeded=5,
        skipped=0,
        mean_weighted_overall_score=0.85,
    )

    rec2 = BenchmarkRecord(
        run_id="run_2",
        recorded_at="2026-07-30T01:00:00Z",
        total_creators=5,
        succeeded=4,
        skipped=1,
        mean_weighted_overall_score=0.80,
    )

    store.append(rec1)
    store.append(rec2)

    recent = store.load_recent(n=5)
    assert len(recent) == 2
    assert recent[0].run_id == "run_1"
    assert recent[1].run_id == "run_2"
