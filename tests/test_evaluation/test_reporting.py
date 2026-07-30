"""
test_reporting.py
=================

Unit tests for ReportBuilder and ReportRenderers.
"""

from pathlib import Path
import json
import pytest

from evaluation.reporting import (
    HTMLReportRenderer,
    JSONReportRenderer,
    MarkdownReportRenderer,
    ReportBuilder,
    render_report,
)
from modules.models import ModuleValidationResult, PipelineRunReport


def _make_sample_report():
    return PipelineRunReport(
        run_id="run_test_123",
        csv_path="data/creators.csv",
        total_creators=2,
        succeeded=2,
        skipped=0,
        module_results={
            "vid1": [
                ModuleValidationResult(
                    video_id="vid1",
                    module_name="module1_csv_reader",
                    schema_valid=True,
                    validated_at="2026-07-30T00:00:00Z"
                )
            ]
        },
        started_at="2026-07-30T00:00:00Z",
        completed_at="2026-07-30T00:00:01Z",
        total_duration_seconds=1.0,
    )


def test_report_builder_persist(tmp_path):
    builder = ReportBuilder(runs_dir=tmp_path)
    report = _make_sample_report()
    path = builder.persist_run_report(report)

    assert path.exists()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["run_id"] == "run_test_123"


def test_renderers():
    report = _make_sample_report()

    json_str = render_report(report, fmt="json")
    assert "run_test_123" in json_str

    md_str = render_report(report, fmt="markdown")
    assert "# Pipeline Evaluation Report" in md_str

    html_str = render_report(report, fmt="html")
    assert "<html>" in html_str
