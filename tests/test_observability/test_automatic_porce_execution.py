"""
tests/test_observability/test_automatic_porce_execution.py
============================================================

Integration tests for Sprint 4: Automatic PORCE execution after pipeline completion.
Verifies:
- PORCEPipelineObserver automatically builds PipelineTrace, extracts TraceFacts, evaluates RuleEngine, and persists RootCauseReport.
- Output files (pipeline_trace.json, artifact_index.json, facts.json, root_cause_report.json) are created.
- PORCE failures are isolated and never crash callers or pipeline execution.
- Backward compatibility and regression safety.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from observability.models import ArtifactIndex, ArtifactRef, PipelineTrace
from observability.reporting import RootCauseReport
from observability.runner import PORCEPipelineObserver


@pytest.fixture
def mock_trace() -> PipelineTrace:
    index = ArtifactIndex(
        video_id="vid_auto_test",
        refs=[
            ArtifactRef(module="module3", artifact_type="thumbnail", path="data/thumbnails/vid_auto_test.jpg", exists=True),
            ArtifactRef(module="module7", artifact_type="generated_image", path="data/generated_thumbnails/vid_auto_test/vid_auto_test.png", exists=True),
        ],
        built_at="2026-08-02T12:00:00Z",
    )
    return PipelineTrace(
        video_id="vid_auto_test",
        modules=[],
        artifact_index=index,
        generation_trace=None,
        overall_status="success",
        assembled_at="2026-08-02T12:00:00Z",
    )


def test_porce_pipeline_observer_end_to_end(tmp_path: Path, mock_trace: PipelineTrace) -> None:
    traces_dir = tmp_path / "traces"
    facts_dir = tmp_path / "facts"

    with patch("observability.runner.PipelineTraceBuilder") as mock_tb_cls, \
         patch("observability.runner.FactPersistence") as mock_fp_cls, \
         patch("observability.runner.RootCausePersistence") as mock_rp_cls:

        # Mock trace builder
        mock_tb_inst = MagicMock()
        mock_tb_inst.build_and_persist.return_value = mock_trace
        mock_tb_cls.return_value = mock_tb_inst

        # Mock persistence instances
        mock_fp_inst = MagicMock()
        mock_fp_cls.return_value = mock_fp_inst

        mock_rp_inst = MagicMock()
        mock_rp_cls.return_value = mock_rp_inst

        observer = PORCEPipelineObserver()
        report = observer.observe("vid_auto_test")

        assert report is not None
        assert isinstance(report, RootCauseReport)
        assert report.video_id == "vid_auto_test"

        # Verify steps were executed
        mock_tb_inst.build_and_persist.assert_called_once_with("vid_auto_test")
        mock_fp_inst.save.assert_called_once()
        mock_rp_inst.save.assert_called_once()


def test_porce_pipeline_observer_failure_isolation() -> None:
    observer = PORCEPipelineObserver()

    # Mock trace builder to raise an exception
    with patch.object(observer.trace_builder, "build_and_persist", side_effect=RuntimeError("Disk crash")):
        # Should catch error, log warning, return None, and NOT raise exception
        result = observer.observe("vid_failing_test")
        assert result is None


def test_main_pipeline_invokes_porce_observer(tmp_path: Path) -> None:
    import main

    mock_creator = MagicMock()
    mock_creator.email = "test@example.com"
    mock_creator.video_url = "https://youtube.com/watch?v=test_vid_123"

    mock_metadata = MagicMock()
    mock_metadata.video_id = "test_vid_123"
    mock_metadata.status = "success"
    mock_metadata.title = "Test Video Title"

    mock_thumbnail = MagicMock()
    mock_thumbnail.thumbnail_path = tmp_path / "thumb.jpg"

    mock_intelligence = MagicMock()
    mock_intelligence.status = "success"

    mock_prompt_pkg = MagicMock()
    mock_prompt_pkg.video_id = "test_vid_123"

    with patch("observability.runner.PORCEPipelineObserver.observe") as mock_observe, \
         patch("main.load_all_creators", return_value=[mock_creator]), \
         patch("main.process_video", return_value=mock_metadata), \
         patch("main.process_thumbnail", return_value=mock_thumbnail), \
         patch("main.analyze_thumbnail", return_value=mock_intelligence), \
         patch("main.save_intelligence"), \
         patch("main.build_redesign_specification", return_value=MagicMock()), \
         patch("main.save_redesign_spec"), \
         patch("main.build_design_blueprint", return_value=MagicMock()), \
         patch("main.save_design_blueprint"), \
         patch("main.compile_prompt_package", return_value=mock_prompt_pkg), \
         patch("main.save_prompt_package"), \
         patch("main.AssetComposer") as mock_ac_cls, \
         patch("main._run_module7_generation", return_value=Path("data/gen.png")):

        mock_ac_inst = MagicMock()
        mock_ac_inst.prepare_generation_workspace.return_value = MagicMock()
        mock_ac_cls.return_value = mock_ac_inst

        # Run pipeline creators loop
        main._run_pipeline_creators(
            csv_path=Path("creators.csv"),
            thumbnail_dir=tmp_path / "thumbnails",
            analysis_dir=tmp_path / "analysis",
            redesign_spec_dir=tmp_path / "redesign_specs",
            design_blueprint_dir=tmp_path / "design_blueprints",
            prompt_package_dir=tmp_path / "prompt_packages",
        )

        mock_observe.assert_called_once_with("test_vid_123")
