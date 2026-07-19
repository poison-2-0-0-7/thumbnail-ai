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
CSV_COLUMNS: tuple[str, ...] = ("email", "video_url")

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

# ---------------------------------------------------------------------------
# Module 2 — YouTube Metadata Extractor (log path only)
# ---------------------------------------------------------------------------

#: Log file used by Module 2.
MODULE2_LOG_PATH: Path = LOG_DIR / "module2.log"

# ---------------------------------------------------------------------------
# Module 3 — Thumbnail Downloader
# ---------------------------------------------------------------------------

#: Log file used by Module 3.
MODULE3_LOG_PATH: Path = LOG_DIR / "module3.log"

#: Directory where downloaded thumbnails are stored.
DEFAULT_THUMBNAIL_DIR: Path = PROJECT_ROOT / "data" / "thumbnails"

#: Filename template for a saved thumbnail; formatted with ``video_id``.
THUMBNAIL_FILENAME_TEMPLATE: str = "{video_id}.jpg"

#: Total seconds to wait for a thumbnail HTTP response before giving up.
THUMBNAIL_REQUEST_TIMEOUT_SECONDS: float = 30.0

#: Maximum retry attempts for transient download failures.
THUMBNAIL_MAX_RETRY_ATTEMPTS: int = 3

#: Minimum seconds to wait between retry attempts (exponential back-off base).
THUMBNAIL_RETRY_WAIT_MIN_SECONDS: float = 1.0

#: Maximum seconds to wait between retry attempts.
THUMBNAIL_RETRY_WAIT_MAX_SECONDS: float = 8.0

#: Minimum acceptable file size in bytes.  Files smaller than this are
#: rejected as empty or truncated even if Pillow can open them.
THUMBNAIL_MIN_FILE_SIZE_BYTES: int = 1_024  # 1 KB

#: Image format strings (as returned by ``PIL.Image.format``) that are
#: accepted as valid thumbnails.  YouTube serves JPEG for maxresdefault
#: but may also serve WEBP or PNG depending on the CDN edge node.
THUMBNAIL_ACCEPTED_IMAGE_FORMATS: frozenset[str] = frozenset(
    {"JPEG", "PNG", "WEBP", "GIF"}
)

#: HTTP status codes that indicate a permanent failure and should NOT be
#: retried.  All other non-2xx codes are considered transient.
THUMBNAIL_PERMANENT_HTTP_ERRORS: frozenset[int] = frozenset({403, 404, 410})

# ---------------------------------------------------------------------------
# Module 4 — Thumbnail Intelligence Engine
# ---------------------------------------------------------------------------

#: Log file used by Module 4.
MODULE4_LOG_PATH: Path = LOG_DIR / "module4.log"

#: Directory where structured intelligence reports are stored as JSON.
DEFAULT_ANALYSIS_DIR: Path = PROJECT_ROOT / "data" / "analysis"

#: Filename template for a saved intelligence report; formatted with
#: ``video_id``.
ANALYSIS_FILENAME_TEMPLATE: str = "{video_id}.json"

#: Device string passed to CV/ML models. Resolved once per process by
#: ``thumbnail_intelligence`` via ``torch.cuda.is_available()`` — this
#: constant is the fallback used when that resolution is unavailable.
DEFAULT_DEVICE: str = "cpu"

#: EasyOCR language list. English is sufficient for the current
#: creator base; additional languages can be appended without any
#: other code changes.
OCR_LANGUAGES: list[str] = ["en"]

#: Minimum per-detection confidence for an OCR text region to be kept.
#: Regions below this threshold are dropped as noise but still counted
#: toward ``average_confidence`` bookkeeping in the raw engine output.
OCR_MIN_CONFIDENCE: float = 0.35

#: InsightFace model pack name.
FACE_MODEL_NAME: str = "buffalo_l"

#: Minimum detector confidence for a face to be kept.
FACE_MIN_CONFIDENCE: float = 0.5

#: YOLO model checkpoint. Ultralytics resolves this name to a cached
#: weights file (downloading it once on first use).
YOLO_MODEL_NAME: str = "yolo11n.pt"

#: Minimum per-detection confidence for a YOLO object to be kept.
YOLO_MIN_CONFIDENCE: float = 0.4

#: Maximum number of dominant colors to extract per thumbnail.
COLOR_PALETTE_SIZE: int = 5

#: Gemini model used for the reasoning stage.
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

#: Maximum seconds to wait for a Gemini response before giving up.
GEMINI_REQUEST_TIMEOUT_SECONDS: float = 60.0

#: Maximum retry attempts for transient Gemini failures.
GEMINI_MAX_RETRY_ATTEMPTS: int = 3

#: Minimum seconds to wait between Gemini retry attempts.
GEMINI_RETRY_WAIT_MIN_SECONDS: float = 2.0

#: Maximum seconds to wait between Gemini retry attempts.
GEMINI_RETRY_WAIT_MAX_SECONDS: float = 20.0

#: Name of the environment variable holding the Gemini API key.
GEMINI_API_KEY_ENV_VAR: str = "GEMINI_API_KEY"

#: Maximum number of transcript characters forwarded to Gemini. Long
#: transcripts are truncated (keeping the head, where creators usually
#: state the video's premise) to keep prompt size and cost bounded.
GEMINI_TRANSCRIPT_CHAR_LIMIT: int = 6_000
