"""
tests/test_observability/test_foundation.py
============================================

Tests for Part 0: Foundation (config, exceptions, models, interfaces, directory creation).
"""

import pytest
from pathlib import Path
from observability.config import (
    OBS_TRACES_DIR,
    OBS_REPORTS_DIR,
    OBS_GENERATION_TRACES_DIR,
    OBS_LOG_PATH,
    OBS_RULE_REGISTRY_ENABLED,
)
from observability.exceptions import (
    PORCEError,
    ArtifactIndexError,
    LogCorrelationError,
    TraceAssemblyError,
    RuleEngineError,
    ReportRenderingError,
)
from observability.models import (
    ArtifactRef,
    ArtifactIndex,
    LogLineRef,
    ModuleTraceEntry,
    PipelineTrace,
)
from observability import ensure_observability_directories, setup_observability_logging


def test_observability_config_paths():
    assert isinstance(OBS_TRACES_DIR, Path)
    assert isinstance(OBS_REPORTS_DIR, Path)
    assert isinstance(OBS_GENERATION_TRACES_DIR, Path)
    assert isinstance(OBS_LOG_PATH, Path)
    assert isinstance(OBS_RULE_REGISTRY_ENABLED, dict)


def test_exception_hierarchy():
    assert issubclass(ArtifactIndexError, PORCEError)
    assert issubclass(LogCorrelationError, PORCEError)
    assert issubclass(TraceAssemblyError, PORCEError)
    assert issubclass(RuleEngineError, PORCEError)
    assert issubclass(ReportRenderingError, PORCEError)

    err = ArtifactIndexError("test error")
    assert str(err) == "test error"


def test_models_immutability():
    ref = ArtifactRef(
        module="module3",
        artifact_type="thumbnail_image",
        path="/tmp/test.jpg",
        exists=True,
        sha256="abc123hash",
        size_bytes=1024,
    )
    assert ref.module == "module3"
    assert ref.exists is True

    # Pydantic frozen model should disallow mutations
    with pytest.raises(Exception):
        ref.exists = False  # type: ignore


def test_ensure_observability_directories(tmp_path, monkeypatch):
    test_traces = tmp_path / "traces"
    test_reports = tmp_path / "reports"
    test_gen_traces = tmp_path / "gen_traces"
    test_log = tmp_path / "logs" / "obs.log"

    monkeypatch.setattr("observability.config.OBS_TRACES_DIR", test_traces)
    monkeypatch.setattr("observability.config.OBS_REPORTS_DIR", test_reports)
    monkeypatch.setattr("observability.config.OBS_GENERATION_TRACES_DIR", test_gen_traces)
    monkeypatch.setattr("observability.config.OBS_LOG_PATH", test_log)

    ensure_observability_directories()

    assert test_traces.exists()
    assert test_reports.exists()
    assert test_gen_traces.exists()
    assert test_log.parent.exists()
