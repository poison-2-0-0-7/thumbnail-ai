"""
test_e2e_golden.py
==================

Golden sample suite tests for PVQEF.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from evaluation.benchmarking import GoldenSampleManager
from evaluation.module_validators import IModuleValidator
from evaluation.pipeline_runner import PipelineRunner
from modules.models import ModuleValidationResult, PipelineRunReport


def test_golden_sample_manager_load_manifest(tmp_path):
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = golden_dir / "golden_manifest.json"
    manifest_p.write_text('{"version": "1.0.0", "total_creators": 1}')

    mgr = GoldenSampleManager(golden_dir=golden_dir)
    data = mgr.load_manifest()
    assert data["version"] == "1.0.0"


def test_golden_suite_run_mock(tmp_path):
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = golden_dir / "golden_manifest.json"
    manifest_p.write_text('{"version": "1.0.0", "total_creators": 1, "csv_rel_path": "creators.csv", "expected_mean_score": 0.85}')

    csv_p = golden_dir / "creators.csv"
    csv_p.write_text("email,video_url\ntest@example.com,https://www.youtube.com/watch?v=dQw4w9WgXcQ\n")

    mock_val = MagicMock(spec=IModuleValidator)
    mock_val.module_name = "module1_csv_reader"
    mock_val.validate.return_value = ModuleValidationResult(
        video_id="dQw4w9WgXcQ",
        module_name="module1_csv_reader",
        artifact_path=str(csv_p),
        schema_valid=True,
        invariants_checked=["file_exists"],
        invariants_failed=[],
        status="success",
        validated_at="2026-07-30T00:00:00Z"
    )

    runner = PipelineRunner(validators=[mock_val])
    mgr = GoldenSampleManager(golden_dir=golden_dir)
    report = mgr.run_golden_suite(runner=runner)

    assert isinstance(report, PipelineRunReport)
    assert report.golden_only is True
    assert report.total_creators == 1
