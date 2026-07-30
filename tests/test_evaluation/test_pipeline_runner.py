"""
test_pipeline_runner.py
========================

Unit tests for PipelineRunner in evaluation.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.module_validators import IModuleValidator
from evaluation.pipeline_runner import PipelineRunner, run_full_evaluation
from modules.models import ModuleValidationResult, PipelineRunReport


def test_pipeline_runner_init():
    runner = PipelineRunner()
    assert len(runner.validators) == 8


def test_pipeline_runner_mock_csv(tmp_path):
    csv_file = tmp_path / "creators.csv"
    csv_file.write_text("email,video_url\ntest@example.com,https://www.youtube.com/watch?v=dQw4w9WgXcQ\n")

    # Mock validator returning canned success
    mock_val = MagicMock(spec=IModuleValidator)
    mock_val.module_name = "module1_csv_reader"
    mock_val.validate.return_value = ModuleValidationResult(
        video_id="dQw4w9WgXcQ",
        module_name="module1_csv_reader",
        artifact_path=str(csv_file),
        schema_valid=True,
        invariants_checked=["file_exists"],
        invariants_failed=[],
        status="success",
        validated_at="2026-07-30T00:00:00Z"
    )

    runner = PipelineRunner(validators=[mock_val])
    report = runner.run(csv_path=csv_file, stages=("module1_csv_reader",))

    assert isinstance(report, PipelineRunReport)
    assert report.total_creators == 1
    assert report.succeeded == 1
    assert len(report.module_results) == 1
    v_id = next(iter(report.module_results.keys()))
    assert report.module_results[v_id][0].module_name == "module1_csv_reader"
