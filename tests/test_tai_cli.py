"""
test_tai_cli.py
===============

Unit tests for thumbnail-ai CLI (`tai`).
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MODULES_DIR = _PROJECT_ROOT / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

import cli


def test_cli_help_output(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Thumbnail AI" in captured.out
    assert "run" in captured.out
    assert "doctor" in captured.out
    assert "status" in captured.out


def test_cli_version_output(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["--version"])
    assert code == 0
    captured = capsys.readouterr()
    assert "thumbnail-ai v1.0.0" in captured.out


def _assert_pipeline_kwargs_equal(kwargs1: dict, kwargs2: dict) -> None:
    assert kwargs1.keys() == kwargs2.keys()
    for key in kwargs1:
        if key == "comfyui_manager":
            assert type(kwargs1[key]) is type(kwargs2[key])
        else:
            assert kwargs1[key] == kwargs2[key]


def test_cli_no_subcommand_identical_to_run(capsys: pytest.CaptureFixture[str]) -> None:
    # Test tai (no args)
    with patch("comfyui_manager.ComfyUIProcessManager.is_healthy", return_value=True), \
         patch("cli.run_pipeline") as mock_pipeline_no_args:
        code_no_args = cli.main([])
        captured_no_args = capsys.readouterr()

    # Test tai run
    with patch("comfyui_manager.ComfyUIProcessManager.is_healthy", return_value=True), \
         patch("cli.run_pipeline") as mock_pipeline_run:
        code_run = cli.main(["run"])
        captured_run = capsys.readouterr()

    assert code_no_args == code_run == 0
    assert captured_no_args.out == captured_run.out
    assert captured_no_args.err == captured_run.err
    assert mock_pipeline_no_args.call_count == mock_pipeline_run.call_count == 1
    _assert_pipeline_kwargs_equal(mock_pipeline_no_args.call_args.kwargs, mock_pipeline_run.call_args.kwargs)


def test_cli_no_subcommand_with_options(capsys: pytest.CaptureFixture[str]) -> None:
    # Test tai --csv custom.csv
    with patch("comfyui_manager.ComfyUIProcessManager.is_healthy", return_value=True), \
         patch("cli.run_pipeline") as mock_pipeline_implicit:
        code_implicit = cli.main(["--csv", "custom.csv"])
        captured_implicit = capsys.readouterr()

    # Test tai run --csv custom.csv
    with patch("comfyui_manager.ComfyUIProcessManager.is_healthy", return_value=True), \
         patch("cli.run_pipeline") as mock_pipeline_explicit:
        code_explicit = cli.main(["run", "--csv", "custom.csv"])
        captured_explicit = capsys.readouterr()

    assert code_implicit == code_explicit == 0
    assert captured_implicit.out == captured_explicit.out
    _assert_pipeline_kwargs_equal(mock_pipeline_implicit.call_args.kwargs, mock_pipeline_explicit.call_args.kwargs)


def test_cli_status(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("comfyui_manager.ComfyUIProcessManager.is_healthy", return_value=True):
        code = cli.main(["status"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Project ............ READY" in captured.out
        assert "ComfyUI ............ RUNNING" in captured.out
        assert "Pipeline ........... IDLE" in captured.out
        assert "Models ............. READY" in captured.out


def test_cli_doctor(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("comfyui_manager.ComfyUIProcessManager.is_healthy", return_value=True), \
         patch("config.validate_controlnet_capability_availability", return_value=None):
        code = cli.main(["doctor"])
        assert code == 0
        captured = capsys.readouterr()
        assert "THUMBNAIL AI HEALTH CHECK" in captured.out
        assert "Python version" in captured.out
        assert "Doctor report:" in captured.out


def test_cli_run_success(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("comfyui_manager.ComfyUIProcessManager.is_healthy", return_value=True), \
         patch("cli.run_pipeline") as mock_pipeline:
        code = cli.main(["run"])
        assert code == 0
        mock_pipeline.assert_called_once()
        captured = capsys.readouterr()
        assert "Starting Thumbnail AI..." in captured.out
        assert "Checking ComfyUI..." in captured.out
        assert "Running pipeline..." in captured.out
        assert "Pipeline completed successfully." in captured.out


def test_cli_argument_parsing() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["run", "--csv", "custom.csv", "--thumbnail-dir", "custom_thumbs"])
    assert args.command == "run"
    assert args.csv == "custom.csv"
    assert args.thumbnail_dir == "custom_thumbs"


def test_cli_subcommands_placeholders(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["test"]) == 0
    assert cli.main(["comfy", "status"]) == 0
    assert cli.main(["ollama", "status"]) == 0
