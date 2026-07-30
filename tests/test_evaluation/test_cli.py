"""
test_cli.py
===========

Unit tests for evaluation CLI entry point.
"""

from unittest.mock import patch
import pytest

from evaluation.cli import main
from modules.models import PipelineRunReport


def test_cli_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])


@patch("evaluation.cli.run_full_evaluation")
@patch("evaluation.cli.ReportBuilder")
@patch("evaluation.cli.HistoricalStore")
def test_cli_run_cmd(mock_store, mock_builder, mock_run, capsys):
    mock_run.return_value = PipelineRunReport(
        run_id="run_cli_1",
        csv_path="creators.csv",
        total_creators=1,
        succeeded=1,
        skipped=0,
        started_at="2026-07-30T00:00:00Z",
        completed_at="2026-07-30T00:00:01Z",
    )

    ret = main(["run", "--csv", "data/creators.csv"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Pipeline Evaluation Report" in captured.out
