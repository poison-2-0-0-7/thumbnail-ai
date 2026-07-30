"""
test_batch_executor.py
======================

Unit tests for BatchExecutor.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from evaluation.batch import BatchExecutor
from evaluation.pipeline_runner import PipelineRunner
from modules.models import PipelineRunReport


def test_batch_executor_run_batch(tmp_path):
    mock_runner = MagicMock(spec=PipelineRunner)
    mock_runner.run.return_value = PipelineRunReport(
        run_id="batch_1",
        csv_path=str(tmp_path / "creators.csv"),
        total_creators=2,
        succeeded=2,
        skipped=0,
        started_at="2026-07-30T00:00:00Z",
        completed_at="2026-07-30T00:00:01Z",
    )

    executor = BatchExecutor(runner=mock_runner)
    report = executor.run_batch(csv_path=tmp_path / "creators.csv")

    assert report.run_id == "batch_1"
    assert mock_runner.run.called
