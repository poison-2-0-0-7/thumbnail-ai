"""
tests/test_observability/test_log_correlator.py
=================================================

Tests for Part 2: LogCorrelator.
"""

import pytest
from pathlib import Path
from observability.trace.log_correlator import LogCorrelator, LOGURU_LINE_PATTERN, _parse_timestamp


def test_parse_timestamp():
    dt = _parse_timestamp("2026-08-02 12:30:00.123456")
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 8
    assert dt.day == 2

    dt_no_ms = _parse_timestamp("2026-08-02 12:30:00")
    assert dt_no_ms is not None
    assert dt_no_ms.second == 0


def test_loguru_line_pattern():
    line = "2026-08-02 12:30:00 | INFO     | module1.csv_reader | Processing video_id=vid_99"
    match = LOGURU_LINE_PATTERN.match(line)
    assert match is not None
    assert match.group("timestamp") == "2026-08-02 12:30:00"
    assert match.group("level") == "INFO"
    assert match.group("module") == "module1.csv_reader"
    assert "vid_99" in match.group("message")


def test_log_correlator_correlate(tmp_path):
    video_id = "target_vid_456"

    m1_log = tmp_path / "module1.log"
    m1_log.write_text(
        "2026-08-02 10:00:00 | INFO     | module1 | Starting run for target_vid_456\n"
        "2026-08-02 10:00:05 | INFO     | module1 | Finished CSV read for target_vid_456\n"
    )

    m7_log = tmp_path / "module7.log"
    m7_log.write_text(
        "2026-08-02 10:05:00 | INFO     | module7 | ComfyUI queue for target_vid_456\n"
        "2026-08-02 10:05:30 | ERROR    | module7 | Generation retry for target_vid_456\n"
    )

    other_log = tmp_path / "other.log"
    other_log.write_text("2026-08-02 10:02:00 | INFO | module2 | Ignored video video_999\n")

    log_files = {
        "module1": m1_log,
        "module7": m7_log,
        "module2": other_log,
    }

    correlator = LogCorrelator(log_files=log_files)
    refs = correlator.correlate(video_id)

    assert len(refs) == 4
    # Verify chronological ordering
    assert refs[0].timestamp == "2026-08-02 10:00:00"
    assert refs[1].timestamp == "2026-08-02 10:00:05"
    assert refs[2].timestamp == "2026-08-02 10:05:00"
    assert refs[3].timestamp == "2026-08-02 10:05:30"
    assert refs[3].level == "ERROR"
