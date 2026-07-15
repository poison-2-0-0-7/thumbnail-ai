"""
youtube_metadata.py
====================

Module 2 — YouTube Metadata Extractor for the AI Thumbnail Outreach
Automation system.

Responsibility
--------------
Given a YouTube URL (typically supplied via a :class:`Creator` record
from Module 1), extract *every* piece of metadata a later AI thumbnail
pipeline will need:

* Video ID, title, description, channel name & ID, upload date,
  duration, view/like counts, thumbnail URL, categories, tags, language.
* Full transcript text, assembled from the highest-priority caption
  track available (manual English → auto-generated English → any manual
  → any auto-generated).

Out of scope
------------
This module **never** downloads the video binary, analyses pixel
content, generates AI prompts, talks to Gemini or Stable Diffusion,
or sends emails.  Those are the exclusive responsibilities of later
modules.

Caching
-------
Every successful extraction is persisted to
``data/metadata/{video_id}.json``.  Subsequent calls for the same video
load from the cache without hitting YouTube, making the system resilient
to transient network failures and respectful of YouTube's rate limits.

Authentication
--------------
YouTube increasingly demands a bot-check ("Sign in to confirm you're
not a bot") for anonymous requests.  To avoid this without requiring
the user to manually export ``cookies.txt``, this module automatically
tries to source cookies from an installed browser, in order: Chrome,
then Edge, then Firefox (see :func:`_resolve_cookie_browser`).  If none
are available, extraction proceeds unauthenticated and a clean warning
is logged; requests that specifically require sign-in raise
:class:`AuthenticationError`.

Public API
----------
- :func:`extract_video_id` — parse a YouTube URL into its 11-char ID.
- :func:`extract_metadata` — fetch full metadata via yt-dlp.
- :func:`extract_transcript` — fetch transcript via youtube-transcript-api.
- :func:`save_metadata` — persist a :class:`VideoMetadata` to JSON.
- :func:`load_cached_metadata` — load a previously-cached record.
- :func:`process_video` — orchestrate the complete pipeline.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional, Union

import requests
import yt_dlp
from loguru import logger
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)
from youtube_transcript_api import VideoUnavailable as TranscriptVideoUnavailable

# Try to import Creator from Module 1; fall back to a structural protocol
# so that Module 2 can be developed and tested independently.
try:
    _MODULE1_SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_MODULE1_SRC) not in sys.path:
        sys.path.insert(0, str(_MODULE1_SRC))
    from csv_reader import Creator  # type: ignore[import-not-found]
except ImportError:  # Module 1 not present in this layout
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Creator:  # type: ignore[no-redef]
        """Minimal stand-in used when Module 1 is not importable."""

        id: str
        email: str
        video_url: str


from models import VideoMetadata  # noqa: E402 — must come after Creator import

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: Directory where log files are written.
_LOG_DIR: Path = Path(__file__).resolve().parent.parent / "logs"

#: Log file for this module.
_LOG_FILE: Path = _LOG_DIR / "module2.log"

#: Default directory for cached metadata JSON files.
DEFAULT_CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "metadata"

#: Maximum retries for transient yt-dlp network failures.
_MAX_RETRY_ATTEMPTS: int = 3

#: Minimum wait (seconds) between retry attempts.
_RETRY_WAIT_MIN_SECONDS: float = 2.0

#: Maximum wait (seconds) between retry attempts.
_RETRY_WAIT_MAX_SECONDS: float = 10.0

#: Accepted YouTube video ID characters (YouTube's actual charset).
_VIDEO_ID_CHARSET: str = r"[A-Za-z0-9_-]{11}"

#: Regex for all supported YouTube URL formats.
_YOUTUBE_URL_RE: re.Pattern[str] = re.compile(
    r"^https://(?:www\.)?(?:"
    r"youtube\.com/watch\?(?:[^&]*&)*v=(?P<id_long>" + _VIDEO_ID_CHARSET + r")(?:[&\s].*)?$"
    r"|"
    r"youtu\.be/(?P<id_short>" + _VIDEO_ID_CHARSET + r")(?:[?#].*)?$"
    r")"
)

#: yt-dlp options that guarantee metadata-only extraction.
_YDL_OPTIONS: dict = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "skip_download": True,
    "writeinfojson": False,
    "writethumbnail": False,
    "extract_flat": False,
}

#: YouTube error substrings that indicate a permanent failure (no retry).
_PERMANENT_ERROR_SIGNATURES: tuple[str, ...] = (
    "private video",
    "video unavailable",
    "has been removed",
    "no longer available",
    "age-restricted",
    "sign in to confirm your age",
    "this video is not available",
    "video is age restricted",
    "members-only",
    "not available in your country",
    "not made this video available in your country",
    "blocked it in your country",
)

#: Substrings specific to YouTube's region-blocking messages, used to give
#: region restriction its own human-readable reason (a subset of the
#: permanent-error signatures above).
_REGION_ERROR_SIGNATURES: tuple[str, ...] = (
    "not available in your country",
    "not made this video available in your country",
    "blocked it in your country",
)

#: YouTube error substrings indicating yt-dlp needs an authenticated
#: session (the "Sign in to confirm you're not a bot" bot-check). These
#: are checked *before* the permanent-error signatures above: retrying
#: without valid cookies will only reproduce the same failure, so this
#: is treated as its own category rather than a generic transient error.
_AUTHENTICATION_ERROR_SIGNATURES: tuple[str, ...] = (
    "sign in to confirm you're not a bot",
    "sign in to confirm you are not a bot",
    "use --cookies-from-browser or --cookies",
)

#: Substrings that identify a transient *network* failure, used only to
#: produce a clearer, differentiated log/error message. Does not change
#: which exception type is raised (still :class:`MetadataExtractionError`,
#: so Tenacity continues to retry it).
_NETWORK_TIMEOUT_SIGNATURES: tuple[str, ...] = (
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "temporary failure",
    "read timed out",
    "name or service not known",
    "failed to establish a new connection",
)

#: Browsers to try, in order, for automatic cookie sourcing. Chrome first,
#: then Edge, then Firefox, matching the project's stated preference.
_COOKIE_BROWSER_PRIORITY: tuple[str, ...] = ("chrome", "edge", "firefox")

#: YouTube's public oEmbed endpoint. Used as a lightweight fallback source
#: of real (not fabricated) title/uploader/thumbnail data when the primary
#: yt-dlp extraction fails due to a bot-check or transient error. This is a
#: separate, unauthenticated, purpose-built API that does not go through
#: yt-dlp's extractor pipeline, so it is not subject to the same bot-check.
_OEMBED_URL: str = "https://www.youtube.com/oembed"

#: Total seconds to wait for the oEmbed fallback request before giving up.
_OEMBED_TIMEOUT_SECONDS: float = 10.0

#: YouTube's predictable static-CDN thumbnail path. Requires no API call at
#: all, so it works even when both yt-dlp and the oEmbed fallback fail.
_THUMBNAIL_URL_TEMPLATE: str = "https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOG_FORMAT: str = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"
)


def _configure_logger() -> None:
    """
    Attach a rotating file handler to the Loguru logger for this module.

    Idempotent: checks for an existing sink pointing at the same file
    before adding a duplicate.  Rotation happens at 10 MB; 30 days of
    logs are retained.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Remove the default stderr sink (index 0) only once at module load.
    # We re-add it so that stderr still works; caller can suppress it
    # by removing sink 0 after importing this module.
    logger.add(
        str(_LOG_FILE),
        rotation="10 MB",
        retention="30 days",
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,  # process-safe async writes
    )


_configure_logger()

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class YouTubeMetadataError(Exception):
    """Base exception for all failures raised by this module."""


class InvalidURLError(YouTubeMetadataError):
    """Raised when a URL is not a recognised YouTube video URL."""


class VideoUnavailableError(YouTubeMetadataError):
    """Raised for videos that are private, deleted, or age-restricted."""

    def __init__(self, reason: str, video_url: str) -> None:
        self.reason = reason
        self.video_url = video_url
        super().__init__(f"{reason} — {video_url}")


class MetadataExtractionError(YouTubeMetadataError):
    """
    Raised when yt-dlp fails for a transient reason (network timeout,
    rate limit, etc.) that is worth retrying.
    """


class AuthenticationError(YouTubeMetadataError):
    """
    Raised when YouTube requires an authenticated session to proceed
    (yt-dlp's "Sign in to confirm you're not a bot" bot-check) and no
    working browser cookie source could be found.

    This is distinct from :class:`VideoUnavailableError`: the video
    itself may be perfectly accessible — it's the *request* that lacks
    authentication, not the video that is gone. It is also distinct
    from :class:`MetadataExtractionError` because retrying with the
    same (missing) cookies will not help, so it is never retried by
    Tenacity.
    """


class TranscriptExtractionError(YouTubeMetadataError):
    """Raised when transcript extraction encounters an unexpected failure."""


class CacheError(YouTubeMetadataError):
    """Raised when reading or writing the metadata cache fails."""


# ---------------------------------------------------------------------------
# URL / ID utilities
# ---------------------------------------------------------------------------


def extract_video_id(url: str) -> str:
    """
    Parse a YouTube URL and return its 11-character video ID.

    Accepts:
    * ``https://www.youtube.com/watch?v=VIDEO_ID``
    * ``https://youtube.com/watch?v=VIDEO_ID``
    * ``https://youtu.be/VIDEO_ID``

    Additional query parameters (playlists, timestamps, etc.) are
    tolerated and ignored.

    Args:
        url: Raw YouTube URL to parse.

    Returns:
        The 11-character video ID string.

    Raises:
        InvalidURLError: If the URL does not match any accepted format.

    Examples:
        >>> extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    if not url or not url.strip():
        raise InvalidURLError("URL must not be empty")

    clean_url = url.strip()
    match = _YOUTUBE_URL_RE.match(clean_url)
    if not match:
        logger.warning("Invalid YouTube URL rejected: {url}", url=clean_url)
        raise InvalidURLError(f"Not a valid YouTube video URL: {clean_url!r}")

    video_id = match.group("id_long") or match.group("id_short")
    logger.debug("Extracted video_id={id} from {url}", id=video_id, url=clean_url)
    return video_id


# ---------------------------------------------------------------------------
# yt-dlp metadata extraction
# ---------------------------------------------------------------------------


def _is_permanent_ydl_error(exc: Exception) -> bool:
    """
    Determine whether a yt-dlp error is permanent (private/deleted/
    age-restricted) and therefore should not be retried.

    Args:
        exc: Exception raised by yt-dlp.

    Returns:
        True if the error is permanent; False if it may be transient.
    """
    message = str(exc).lower()
    return any(sig in message for sig in _PERMANENT_ERROR_SIGNATURES)


def _is_authentication_error(message: str) -> bool:
    """
    Determine whether a yt-dlp error is YouTube's bot-check / sign-in
    demand ("Sign in to confirm you're not a bot").

    Args:
        message: Raw yt-dlp error message.

    Returns:
        True if the message matches a known authentication-required
        signature.
    """
    msg = message.lower()
    return any(sig in msg for sig in _AUTHENTICATION_ERROR_SIGNATURES)


def _classify_transient_error(message: str) -> str:
    """
    Give a transient (retryable) yt-dlp failure a short, differentiated
    category label for logs and error messages. Purely cosmetic — it
    never changes which exception type gets raised.

    Args:
        message: Raw yt-dlp error message.

    Returns:
        ``"Network timeout"`` for connection/timeout-style failures,
        ``"Rate limited"`` for HTTP 429 / rate-limit style failures, or
        ``"Metadata extraction failed"`` as a generic fallback.
    """
    msg = message.lower()
    if any(sig in msg for sig in _NETWORK_TIMEOUT_SIGNATURES):
        return "Network timeout"
    if "429" in msg or "too many requests" in msg or "rate-limit" in msg or "rate limit" in msg:
        return "Rate limited"
    return "Metadata extraction failed"


def _before_sleep_log(retry_state: RetryCallState) -> None:
    """
    Loguru-compatible Tenacity sleep callback.

    Args:
        retry_state: Tenacity state object for the current retry.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    # NOTE: log only the string representation of the exception, never the
    # exception object itself. Loguru copies every keyword argument used for
    # message formatting into record["extra"], and with enqueue=True that
    # record is pickled to hand off to the background writer thread.
    # Exception objects (especially yt-dlp's DownloadError) carry a live
    # traceback, and traceback objects cannot be pickled — passing `exc=exc`
    # here previously caused `TypeError: cannot pickle 'traceback' object`.
    logger.warning(
        "Retrying {fn} (attempt {n}/{max}): {exc}",
        fn=retry_state.fn.__name__ if retry_state.fn else "unknown",
        n=retry_state.attempt_number,
        max=_MAX_RETRY_ATTEMPTS,
        exc=str(exc) if exc is not None else "unknown error",
    )


@retry(
    stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1,
        min=_RETRY_WAIT_MIN_SECONDS,
        max=_RETRY_WAIT_MAX_SECONDS,
    ),
    retry=retry_if_exception_type(MetadataExtractionError),
    before_sleep=_before_sleep_log,
    reraise=True,
)
def _fetch_yt_dlp_info(url: str) -> dict:
    """
    Call yt-dlp to fetch raw video information (no download).

    This function is the **only** place that calls yt-dlp; all retry
    and error-classification logic lives here.  Permanent errors are
    converted to :class:`VideoUnavailableError` immediately (no retry).
    A bot-check / sign-in demand becomes :class:`AuthenticationError`
    (also not retried — retrying without new cookies changes nothing).
    Transient errors become :class:`MetadataExtractionError` so that
    Tenacity retries them.

    Args:
        url: A validated YouTube video URL.

    Returns:
        Raw yt-dlp info dict.

    Raises:
        VideoUnavailableError: For private/deleted/age/region-restricted videos.
        AuthenticationError: If YouTube demands a bot-check sign-in and no
            working browser cookie source could be found.
        MetadataExtractionError: For transient network/API failures.
    """
    try:
        with yt_dlp.YoutubeDL(_build_ydl_options()) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            raise MetadataExtractionError(
                f"yt-dlp returned empty info for {url!r}"
            )
        return info

    except yt_dlp.utils.DownloadError as exc:
        message = str(exc)

        if _is_authentication_error(message):
            logger.error(
                "Authentication required for {url}: YouTube's bot-check "
                "rejected the request and no browser cookies were usable.",
                url=url,
            )
            raise AuthenticationError(
                "YouTube requires sign-in to confirm you're not a bot, and "
                "automatic cookie sourcing (Chrome, Edge, Firefox) did not "
                "resolve it. Log in to YouTube in one of those browsers on "
                "this machine, or supply cookies manually as a last resort."
            ) from exc

        if _is_permanent_ydl_error(exc):
            reason = _classify_video_error(message)
            logger.error("Permanent video error for {url}: {reason}", url=url, reason=reason)
            raise VideoUnavailableError(reason=reason, video_url=url) from exc

        category = _classify_transient_error(message)
        logger.warning(
            "Transient yt-dlp error for {url}: {category}: {exc}",
            url=url,
            category=category,
            exc=message,
        )
        raise MetadataExtractionError(f"{category}: {message}") from exc

    except Exception as exc:
        logger.warning("Unexpected yt-dlp error for {url}: {exc}", url=url, exc=str(exc))
        raise MetadataExtractionError(str(exc)) from exc


def _classify_video_error(message: str) -> str:
    """
    Return a human-readable reason for a permanent video error based
    on the yt-dlp error message text.

    Args:
        message: Raw yt-dlp error message (lowercased for comparison).

    Returns:
        A short, readable reason string.
    """
    msg = message.lower()
    if "private video" in msg:
        return "Private video"
    if "age-restricted" in msg or "sign in to confirm your age" in msg or "age restricted" in msg:
        return "Age-restricted video"
    if "members-only" in msg:
        return "Members-only video"
    if any(sig in msg for sig in _REGION_ERROR_SIGNATURES):
        return "Region-restricted video"
    if "removed" in msg or "no longer available" in msg or "deleted" in msg:
        return "Video deleted or removed"
    return "Video unavailable"


# ---------------------------------------------------------------------------
# Automatic browser cookie sourcing
# ---------------------------------------------------------------------------

#: Process-wide cache so the local filesystem is probed for browser cookies
#: at most once, no matter how many videos are processed in a batch.
_cookie_resolution_cache: dict[str, object] = {"checked": False, "browser": None}


def _resolve_cookie_browser() -> Optional[str]:
    """
    Determine which installed browser's cookies yt-dlp should use to
    authenticate outgoing requests, so that "Sign in to confirm you're
    not a bot" challenges can be avoided the same way a logged-in
    browser session would avoid them.

    Tries Chrome first, then Edge, then Firefox (:data:`_COOKIE_BROWSER_PRIORITY`);
    the first browser whose local cookie store can actually be read wins.
    The result is cached for the lifetime of the process.

    Returns:
        The browser name to pass to yt-dlp's ``cookiesfrombrowser``
        option, or ``None`` if no supported browser's cookies could be
        read — in which case yt-dlp proceeds without authentication and
        the caller logs a clean warning.
    """
    if _cookie_resolution_cache["checked"]:
        return _cookie_resolution_cache["browser"]  # type: ignore[return-value]

    for browser in _COOKIE_BROWSER_PRIORITY:
        try:
            yt_dlp.cookies.extract_cookies_from_browser(browser)
        except Exception as exc:
            logger.debug(
                "Browser cookie source unavailable: browser={b} reason={r}",
                b=browser,
                r=str(exc),
            )
            continue

        logger.info("Auto-detected {b} cookies for yt-dlp authentication", b=browser)
        _cookie_resolution_cache["checked"] = True
        _cookie_resolution_cache["browser"] = browser
        return browser

    logger.warning(
        "No browser cookies available (tried chrome, edge, firefox). "
        "Continuing without authentication — videos that trigger "
        "YouTube's bot-check will fail with an AuthenticationError."
    )
    _cookie_resolution_cache["checked"] = True
    _cookie_resolution_cache["browser"] = None
    return None


def _build_ydl_options() -> dict:
    """
    Build the yt-dlp options dict for a single extraction call, adding
    an automatically-detected browser cookie source when one is
    available.

    Returns:
        A copy of :data:`_YDL_OPTIONS`, with ``cookiesfrombrowser`` set
        when a usable browser cookie store was found.
    """
    options = dict(_YDL_OPTIONS)
    browser = _resolve_cookie_browser()
    if browser:
        options["cookiesfrombrowser"] = (browser,)
    return options


def _parse_yt_dlp_info(info: dict, url: str) -> VideoMetadata:
    """
    Convert a raw yt-dlp info dict into a :class:`VideoMetadata`.

    Only populated from info dict keys; transcript is added by the
    caller after a separate API call.

    Args:
        info: Raw dict returned by ``yt_dlp.YoutubeDL.extract_info``.
        url:  Original URL (used for error context only).

    Returns:
        A :class:`VideoMetadata` with ``transcript=None`` and
        ``status="success"``.

    Raises:
        MetadataExtractionError: If essential fields (id, title,
            uploader) are missing from the info dict.
    """
    video_id: Optional[str] = info.get("id")
    title: Optional[str] = info.get("title")
    uploader: Optional[str] = info.get("uploader") or info.get("channel")

    if not video_id:
        raise MetadataExtractionError(f"yt-dlp returned no video id for {url!r}")
    if not title:
        raise MetadataExtractionError(f"yt-dlp returned no title for {url!r}")
    if not uploader:
        raise MetadataExtractionError(f"yt-dlp returned no uploader for {url!r}")

    raw_date: Optional[str] = info.get("upload_date")

    return VideoMetadata(
        video_id=video_id,
        title=title,
        description=info.get("description"),
        uploader=uploader,
        uploader_id=info.get("uploader_id") or "",
        channel_id=info.get("channel_id") or info.get("uploader_id") or "",
        upload_date=raw_date,           # model validator converts YYYYMMDD → YYYY-MM-DD
        duration=info.get("duration"),
        view_count=info.get("view_count"),
        like_count=info.get("like_count"),
        thumbnail_url=info.get("thumbnail"),
        categories=info.get("categories") or [],
        tags=info.get("tags") or [],
        language=info.get("language"),
        transcript=None,
        status="success",
    )


def extract_metadata(url: str) -> VideoMetadata:
    """
    Fetch video metadata from YouTube using yt-dlp.

    Retries up to :data:`_MAX_RETRY_ATTEMPTS` times on transient
    failures (network errors, rate limits).  Never retries permanent
    failures (private, deleted, age-restricted videos) or
    authentication failures (bot-check).

    Note:
        This function does **not** fetch the transcript.  Use
        :func:`extract_transcript` separately and merge the result
        with :func:`process_video`.

    Args:
        url: A YouTube video URL (any supported format).

    Returns:
        A :class:`VideoMetadata` with ``transcript=None``.

    Raises:
        InvalidURLError: If ``url`` is not a valid YouTube URL.
        VideoUnavailableError: If the video is private/deleted/restricted.
        AuthenticationError: If YouTube demands a bot-check sign-in.
        MetadataExtractionError: If yt-dlp fails after all retries.
    """
    video_id = extract_video_id(url)
    logger.info("Fetching metadata for video_id={id}", id=video_id)

    info = _fetch_yt_dlp_info(url)
    metadata = _parse_yt_dlp_info(info, url)

    logger.info(
        "Metadata extracted: video_id={id} title={title!r}",
        id=metadata.video_id,
        title=metadata.title,
    )
    return metadata


# ---------------------------------------------------------------------------
# Transcript extraction
# ---------------------------------------------------------------------------


def _format_transcript_entries(entries: object) -> str:
    """
    Convert an iterable of transcript entries (either
    :class:`FetchedTranscriptSnippet` dataclass instances or legacy
    dicts) into a single whitespace-joined string.

    Args:
        entries: Iterable of transcript entries.

    Returns:
        Full transcript as a single string.
    """
    parts: list[str] = []
    for entry in entries:  # type: ignore[union-attr]
        if isinstance(entry, dict):
            text = entry.get("text", "")
        else:
            text = getattr(entry, "text", "")
        if text:
            parts.append(str(text).strip())
    return " ".join(parts)


def _get_best_transcript(transcript_list: object) -> Optional[str]:
    """
    Retrieve the highest-priority transcript from a
    :class:`TranscriptList` following the project priority order:

    1. Manually-created English captions.
    2. Auto-generated English captions.
    3. Any manually-created captions (first available language).
    4. Any auto-generated captions (first available language).

    Args:
        transcript_list: A ``TranscriptList`` returned by
            ``YouTubeTranscriptApi().list(video_id)``.

    Returns:
        Full transcript text, or ``None`` if nothing is available.
    """
    english_codes = ["en", "en-US", "en-GB", "en-CA", "en-AU"]

    # 1. Manual English
    try:
        t = transcript_list.find_manually_created_transcript(english_codes)  # type: ignore[attr-defined]
        return _format_transcript_entries(t.fetch())
    except Exception:
        pass

    # 2. Auto-generated English
    try:
        t = transcript_list.find_generated_transcript(english_codes)  # type: ignore[attr-defined]
        return _format_transcript_entries(t.fetch())
    except Exception:
        pass

    # 3. Any manual transcript
    try:
        manual: dict = transcript_list._manually_created_transcripts  # type: ignore[attr-defined]
        if manual:
            t = next(iter(manual.values()))
            return _format_transcript_entries(t.fetch())
    except Exception:
        pass

    # 4. Any auto-generated transcript
    try:
        generated: dict = transcript_list._generated_transcripts  # type: ignore[attr-defined]
        if generated:
            t = next(iter(generated.values()))
            return _format_transcript_entries(t.fetch())
    except Exception:
        pass

    return None


def extract_transcript(video_id: str) -> Optional[str]:
    """
    Retrieve the transcript for a YouTube video.

    Selects captions in this priority order:

    1. Manually-created English captions.
    2. Auto-generated English captions.
    3. Any manually-created captions (any language).
    4. Any auto-generated captions (any language).

    This function **never raises**.  If no transcript is available for
    any reason (disabled captions, private video, network failure), it
    logs the event and returns ``None``.

    Args:
        video_id: The 11-character YouTube video ID.

    Returns:
        The full transcript text, or ``None`` if unavailable.
    """
    api = YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)
    except (TranscriptsDisabled, TranscriptVideoUnavailable) as exc:
        logger.warning(
            "Transcript unavailable for video_id={id}: {reason}",
            id=video_id,
            reason=str(exc),
        )
        return None
    except CouldNotRetrieveTranscript as exc:
        logger.warning(
            "Could not retrieve transcript for video_id={id}: {reason}",
            id=video_id,
            reason=str(exc),
        )
        return None
    except Exception as exc:
        logger.error(
            "Unexpected error fetching transcript list for video_id={id}: {exc}",
            id=video_id,
            exc=str(exc),
        )
        return None

    text = _get_best_transcript(transcript_list)

    if text:
        word_count = len(text.split())
        logger.info(
            "Transcript extracted for video_id={id} ({words} words)",
            id=video_id,
            words=word_count,
        )
    else:
        logger.warning(
            "No usable transcript found for video_id={id}",
            id=video_id,
        )

    return text


# ---------------------------------------------------------------------------
# Cache: save / load
# ---------------------------------------------------------------------------


def _cache_path(video_id: str, cache_dir: Path) -> Path:
    """
    Return the canonical JSON cache path for a given video ID.

    Args:
        video_id:  The 11-character video ID.
        cache_dir: Root directory for cached metadata files.

    Returns:
        ``cache_dir / f"{video_id}.json"``
    """
    return cache_dir / f"{video_id}.json"


def save_metadata(metadata: VideoMetadata, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """
    Persist a :class:`VideoMetadata` instance to a JSON file.

    The file is written atomically: the JSON is first written to a
    temporary sibling file and then moved into place, so a concurrent
    reader never sees a partial write.

    Args:
        metadata:  The record to persist.
        cache_dir: Root directory for cached metadata files.  Created
                   automatically if it does not exist.

    Raises:
        CacheError: If the file cannot be written due to a permissions
            error or other OS failure.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _cache_path(metadata.video_id, cache_dir)
    tmp = target.with_suffix(".tmp")

    try:
        tmp.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(target)
        logger.debug(
            "Cached metadata for video_id={id} → {path}",
            id=metadata.video_id,
            path=target,
        )
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        logger.error(
            "Failed to cache metadata for video_id={id}: {exc}",
            id=metadata.video_id,
            exc=str(exc),
        )
        raise CacheError(
            f"Could not write cache file {target}: {exc}"
        ) from exc


def load_cached_metadata(
    video_id: str, cache_dir: Path = DEFAULT_CACHE_DIR
) -> Optional[VideoMetadata]:
    """
    Load a previously-cached :class:`VideoMetadata` from disk.

    Args:
        video_id:  The 11-character video ID to look up.
        cache_dir: Root directory for cached metadata files.

    Returns:
        The deserialized :class:`VideoMetadata`, or ``None`` if no
        cache file exists for this video.

    Raises:
        CacheError: If the cache file exists but cannot be read or
            fails Pydantic validation.
    """
    target = _cache_path(video_id, cache_dir)

    if not target.exists():
        logger.debug("Cache miss for video_id={id}", id=video_id)
        return None

    logger.info("Cache hit for video_id={id}", id=video_id)

    try:
        raw = target.read_text(encoding="utf-8")
        return VideoMetadata.model_validate_json(raw)
    except Exception as exc:
        logger.error(
            "Failed to load cache for video_id={id}: {exc}",
            id=video_id,
            exc=str(exc),
        )
        raise CacheError(
            f"Could not read or validate cache file {target}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Fallback metadata source (used only when yt-dlp itself cannot be used)
# ---------------------------------------------------------------------------


def _guess_thumbnail_url(video_id: str) -> str:
    """
    Return YouTube's predictable static-CDN thumbnail URL for a video ID.

    This requires no API call at all — YouTube serves a thumbnail at a
    fixed path for every uploaded video — so it works even when both
    yt-dlp and the oEmbed fallback have failed.

    Args:
        video_id: The 11-character YouTube video ID.

    Returns:
        A best-effort thumbnail URL. Not guaranteed to resolve (e.g. for
        truly private videos), but safe to attempt.
    """
    return _THUMBNAIL_URL_TEMPLATE.format(video_id=video_id)


def _fetch_oembed_metadata(video_id: str) -> Optional[VideoMetadata]:
    """
    Best-effort fallback metadata source used when the primary yt-dlp
    extraction cannot be completed (bot-check, transient network error).

    YouTube's public oEmbed endpoint exposes a small but *genuine*
    subset of a video's metadata (title, channel name, thumbnail)
    without going through yt-dlp's extractor pipeline, so it is not
    subject to the same bot-check. This function never fabricates
    data: if the oEmbed request fails or returns incomplete data, it
    returns ``None`` and the caller falls back to a hard failure.

    Args:
        video_id: The 11-character YouTube video ID.

    Returns:
        A minimal :class:`VideoMetadata` (``status="success"``) built
        from real oEmbed data, or ``None`` if the fallback also failed.
    """
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        response = requests.get(
            _OEMBED_URL,
            params={"url": watch_url, "format": "json"},
            timeout=_OEMBED_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.warning(
            "oEmbed fallback failed for video_id={id}: {exc}",
            id=video_id,
            exc=str(exc),
        )
        return None

    title = data.get("title")
    uploader = data.get("author_name")
    if not title or not uploader:
        logger.warning(
            "oEmbed fallback returned incomplete data for video_id={id}",
            id=video_id,
        )
        return None

    thumbnail_url = data.get("thumbnail_url") or _guess_thumbnail_url(video_id)

    logger.info(
        "oEmbed fallback succeeded for video_id={id}: title={title!r}",
        id=video_id,
        title=title,
    )
    return VideoMetadata(
        video_id=video_id,
        title=title,
        uploader=uploader,
        uploader_id="",
        channel_id="",
        thumbnail_url=thumbnail_url,
        status="success",
    )


def _try_oembed_fallback(video_id: str, enabled: bool) -> Optional[VideoMetadata]:
    """
    Attempt the oEmbed fallback if (and only if) it is enabled.

    Args:
        video_id: The 11-character YouTube video ID.
        enabled:  Whether the caller opted in to the fallback.

    Returns:
        The fallback :class:`VideoMetadata`, or ``None`` if disabled or
        unsuccessful.
    """
    if not enabled:
        return None
    logger.info(
        "Attempting oEmbed fallback for video_id={id} after yt-dlp failure",
        id=video_id,
    )
    return _fetch_oembed_metadata(video_id)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def process_video(
    creator: Creator,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    enable_oembed_fallback: bool = False,
) -> VideoMetadata:
    """
    Run the full metadata extraction pipeline for a single creator lead.

    Pipeline steps:

    1. Validate the URL and extract the video ID.
    2. Check the on-disk cache; if a hit is found, return it immediately.
    3. Fetch metadata via yt-dlp (with automatic retries on transient errors,
       automatic browser-cookie authentication, and — when
       ``enable_oembed_fallback`` is set — a best-effort fallback to
       YouTube's public oEmbed endpoint if yt-dlp itself cannot be used).
    4. Fetch the transcript via youtube-transcript-api (independent of
       step 3, so a transcript can still be captured even when step 3
       ultimately fails).
    5. Merge transcript into the metadata record.
    6. Save the merged record to the cache.
    7. Return the record.

    On any failure (invalid URL, unavailable video, unrecoverable yt-dlp
    error) this function does **not** raise.  It instead returns a
    :class:`VideoMetadata` with ``status="error"`` and a descriptive
    ``error_message``, so that callers always receive a consistent type.

    Args:
        creator:   A :class:`Creator` record from Module 1.
        cache_dir: Directory used for JSON caching.
        enable_oembed_fallback: When ``True``, a failed yt-dlp extraction
            (bot-check or transient network failure) falls back to
            YouTube's public oEmbed endpoint for a smaller but genuine
            set of fields (title, uploader, thumbnail) rather than
            failing the video outright. Defaults to ``False`` so that
            this function's default behavior — and every existing
            caller/test — stays fully offline and deterministic;
            production callers (see ``main.py``) opt in explicitly.

    Returns:
        A :class:`VideoMetadata` with ``status="success"`` on success
        or ``status="error"`` on any failure.
    """
    url: str = creator.video_url
    creator_id: str = creator.id
    logger.info(
        "Processing creator_id={cid} url={url}",
        cid=creator_id,
        url=url,
    )

    # --- Step 1: validate URL and extract video ID ---
    try:
        video_id = extract_video_id(url)
    except InvalidURLError as exc:
        logger.error(
            "Invalid URL for creator_id={cid}: {exc}",
            cid=creator_id,
            exc=str(exc),
        )
        return _error_metadata(
            video_id="",
            reason=str(exc),
            url=url,
        )

    # --- Step 2: cache lookup ---
    try:
        cached = load_cached_metadata(video_id, cache_dir)
        if cached is not None:
            logger.info(
                "Returning cached metadata for creator_id={cid} video_id={id}",
                cid=creator_id,
                id=video_id,
            )
            return cached
    except CacheError as exc:
        # Corrupted cache — log and continue to re-fetch from YouTube.
        logger.warning(
            "Cache read failed for video_id={id}, re-fetching: {exc}",
            id=video_id,
            exc=str(exc),
        )

    # --- Step 3: fetch metadata ---
    try:
        metadata = extract_metadata(url)
    except InvalidURLError as exc:
        return _error_metadata(video_id=video_id, reason=str(exc), url=url)
    except VideoUnavailableError as exc:
        # Video is private/deleted/age- or region-restricted: this is a
        # property of the video itself, not the request, so no fallback
        # is attempted — a different metadata source will not make a
        # private video public.
        logger.error(
            "Video unavailable for creator_id={cid} video_id={id}: {reason}",
            cid=creator_id,
            id=video_id,
            reason=exc.reason,
        )
        return _error_metadata(video_id=video_id, reason=exc.reason, url=url)
    except AuthenticationError as exc:
        logger.warning(
            "Authentication required for creator_id={cid} video_id={id}: {exc}",
            cid=creator_id,
            id=video_id,
            exc=str(exc),
        )
        metadata = _try_oembed_fallback(video_id, enable_oembed_fallback)
        if metadata is None:
            return _error_metadata(video_id=video_id, reason=str(exc), url=url)
    except MetadataExtractionError as exc:
        logger.error(
            "Metadata extraction failed for creator_id={cid} video_id={id}: {exc}",
            cid=creator_id,
            id=video_id,
            exc=str(exc),
        )
        metadata = _try_oembed_fallback(video_id, enable_oembed_fallback)
        if metadata is None:
            return _error_metadata(video_id=video_id, reason=str(exc), url=url)

    # --- Step 4: fetch transcript ---
    transcript_text = extract_transcript(video_id)

    # --- Step 5: merge transcript into metadata ---
    metadata = VideoMetadata(**{
        **metadata.model_dump(),
        "transcript": transcript_text,
    })

    # --- Step 6: save to cache ---
    try:
        save_metadata(metadata, cache_dir)
    except CacheError as exc:
        # Non-fatal: log and continue without caching.
        logger.warning(
            "Cache write failed for video_id={id}: {exc}",
            id=video_id,
            exc=str(exc),
        )

    logger.info(
        "process_video complete: creator_id={cid} video_id={id} status={s}",
        cid=creator_id,
        id=video_id,
        s=metadata.status,
    )
    return metadata


def _error_metadata(
    video_id: str,
    reason: str,
    url: str,
) -> VideoMetadata:
    """
    Build a minimal :class:`VideoMetadata` record representing a failed
    extraction.

    Args:
        video_id: The video ID if it could be parsed, or ``""`` otherwise.
        reason:   Human-readable error description.
        url:      Original URL (used to populate a placeholder title).

    Returns:
        A frozen :class:`VideoMetadata` with ``status="error"``.
    """
    return VideoMetadata(
        video_id=video_id or "unknown",
        title=url,
        uploader="",
        uploader_id="",
        channel_id="",
        status="error",
        error_message=reason,
    )
