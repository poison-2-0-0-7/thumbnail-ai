"""
config.py
=========

Centralized configuration for Module 1 (CSV Reader).

This module holds constants that are shared across the CSV reader
implementation: schema definitions, validation patterns, and logging
configuration. Keeping these values in one place avoids magic strings
scattered throughout the codebase and gives future modules a single,
predictable place to look for shared configuration.

This module has zero dependencies on other project modules, in keeping
with the project's requirement that every module be independently
testable and loosely coupled.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

#: Root of the project. config.py lives in <root>/src/, so the parent of
#: the parent directory is the project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Default location of the creators CSV file.
DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "data" / "creators.csv"

#: Directory where log files are written.
LOG_DIR: Path = PROJECT_ROOT / "logs"

#: Log file used by Module 1.
MODULE1_LOG_PATH: Path = LOG_DIR / "module1.log"

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

#: Canonical column order for the creators CSV. This is the single source
#: of truth for the schema; any CSV that does not match this header is
#: considered malformed.
CSV_COLUMNS: tuple[str, ...] = ("id", "email", "video_url")

#: Encoding used for all CSV reads/writes.
CSV_ENCODING: str = "utf-8"

# ---------------------------------------------------------------------------
# Validation patterns
# ---------------------------------------------------------------------------

#: A pragmatic (not fully RFC 5322 compliant, but production-sane) email
#: validation pattern. Rejects obviously malformed addresses while
#: avoiding the complexity/false-negative tradeoffs of a fully compliant
#: regex.
EMAIL_PATTERN: str = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

#: Accepted YouTube URL formats:
#:   - https://www.youtube.com/watch?v=VIDEO_ID
#:   - https://youtube.com/watch?v=VIDEO_ID
#:   - https://youtu.be/VIDEO_ID
#: VIDEO_ID is exactly 11 characters of [A-Za-z0-9_-], which matches
#: YouTube's actual video ID format.
YOUTUBE_URL_PATTERN: str = (
    r"^https://(?:www\.)?(?:"
    r"youtube\.com/watch\?v=(?P<id_long>[A-Za-z0-9_-]{11})(?:&\S*)?"
    r"|"
    r"youtu\.be/(?P<id_short>[A-Za-z0-9_-]{11})(?:\?\S*)?"
    r")$"
)

# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------

#: Maximum number of seconds to wait for a file lock before giving up.
LOCK_TIMEOUT_SECONDS: float = 10.0

#: Polling interval (seconds) while waiting to acquire a lock.
LOCK_CHECK_INTERVAL_SECONDS: float = 0.1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
