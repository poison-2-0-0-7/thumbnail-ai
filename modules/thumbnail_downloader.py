"""
thumbnail_downloader.py
========================

Module 3 — YouTube Thumbnail Downloader for the AI Thumbnail Outreach
Automation system.

Responsibility
--------------
Given an immutable :class:`VideoMetadata` object produced by Module 2,
download the video's static thumbnail image from YouTube's CDN and
persist it to ``data/thumbnails/{video_id}.jpg``.

Out of scope
------------
This module **never** analyses pixel content, detects faces, removes
backgrounds, generates AI prompts, or calls any AI API. Those tasks
belong exclusively to later modules.

Caching
-------
If ``data/thumbnails/{video_id}.jpg`` already exists, it is validated
with Pillow and returned without a network request.  A cached file that
fails validation is silently deleted and re-downloaded, so the caller
always receives a verified image.

Public API
----------
- :func:`download_thumbnail`      — HTTP download with automatic retries.
- :func:`validate_image`          — Pillow-based integrity check.
- :func:`save_thumbnail`          — Atomic bytes-to-disk write.
- :func:`load_cached_thumbnail`   — Check whether a local copy exists.
- :func:`process_thumbnail`       — Orchestrate the full pipeline.

Design contract with Module 4
------------------------------
Module 4 (AI thumbnail analyser) receives a :class:`ThumbnailData`
object.  It reads ``thumbnail_data.thumbnail_path`` to open the local
JPEG and ``thumbnail_data.metadata`` for video context.  Module 4 never
downloads anything — that contract belongs entirely to this module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import requests
from loguru import logger
from PIL import Image, UnidentifiedImageError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Project-level imports
# ---------------------------------------------------------------------------

_MODULES_DIR: Path = Path(__file__).resolve().parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import (  # noqa: E402
    DEFAULT_THUMBNAIL_DIR,
    LOG_DIR,
    MODULE3_LOG_PATH,
    THUMBNAIL_ACCEPTED_IMAGE_FORMATS,
    THUMBNAIL_FILENAME_TEMPLATE,
    THUMBNAIL_MAX_RETRY_ATTEMPTS,
    THUMBNAIL_MIN_FILE_SIZE_BYTES,
    THUMBNAIL_PERMANENT_HTTP_ERRORS,
    THUMBNAIL_REQUEST_TIMEOUT_SECONDS,
    THUMBNAIL_RETRY_WAIT_MAX_SECONDS,
    THUMBNAIL_RETRY_WAIT_MIN_SECONDS,
)
from models import ThumbnailData, VideoMetadata  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    """
    Attach a rotating file sink for Module 3 to the Loguru logger.

    Idempotent across repeated imports.  Rotation at 10 MB, 30-day
    retention, async-safe enqueue mode.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(MODULE3_LOG_PATH),
        rotation="10 MB",
        retention="30 days",
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,
    )


_configure_logger()

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ThumbnailDownloaderError(Exception):
    """Base exception for all failures raised by Module 3."""


class ThumbnailDownloadError(ThumbnailDownloaderError):
    """
    Raised when an HTTP download fails for a *transient* reason such as
    a connection error or a 5xx server error.

    Tenacity will retry on this exception type.
    """


class NetworkTimeoutError(ThumbnailDownloaderError):
    """
    Raised when the HTTP request times out.

    Tenacity will retry on this exception type.
    """


class InvalidThumbnailError(ThumbnailDownloaderError):
    """
    Raised when the server returns a *permanent* HTTP error (404, 403,
    410) or when the URL is missing entirely.

    Tenacity will **not** retry on this exception.
    """


class ImageValidationError(ThumbnailDownloaderError):
    """
    Raised when Pillow cannot identify the downloaded bytes as a valid
    image, or when the file is empty / below the minimum size threshold.

    Tenacity will **not** retry on this exception.
    """


class ThumbnailCacheError(ThumbnailDownloaderError):
    """Raised when a filesystem error prevents reading or writing the cache."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _thumbnail_path(video_id: str, thumbnail_dir: Path) -> Path:
    """
    Return the canonical local path for a thumbnail.

    Args:
        video_id:      The 11-character YouTube video ID.
        thumbnail_dir: Root directory for downloaded thumbnails.

    Returns:
        ``thumbnail_dir / "{video_id}.jpg"``
    """
    filename = THUMBNAIL_FILENAME_TEMPLATE.format(video_id=video_id)
    return thumbnail_dir / filename


def _before_sleep_log(retry_state: RetryCallState) -> None:
    """
    Loguru-compatible Tenacity sleep callback logged at WARNING level.

    Args:
        retry_state: Current Tenacity retry state.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    fn_name = retry_state.fn.__name__ if retry_state.fn else "unknown"
    logger.warning(
        "Retrying {fn} (attempt {n}/{max}): {exc}",
        fn=fn_name,
        n=retry_state.attempt_number,
        max=THUMBNAIL_MAX_RETRY_ATTEMPTS,
        exc=exc,
    )


# ---------------------------------------------------------------------------
# Download layer (with retry)
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(THUMBNAIL_MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1,
        min=THUMBNAIL_RETRY_WAIT_MIN_SECONDS,
        max=THUMBNAIL_RETRY_WAIT_MAX_SECONDS,
    ),
    retry=retry_if_exception_type((ThumbnailDownloadError, NetworkTimeoutError)),
    before_sleep=_before_sleep_log,
    reraise=True,
)
def _fetch_thumbnail_bytes(
    url: str,
    timeout: float = THUMBNAIL_REQUEST_TIMEOUT_SECONDS,
) -> bytes:
    """
    Perform a single HTTP GET for ``url`` and return the raw response body.

    This is the **only** place that makes network calls in Module 3.
    Error classification happens here so that the retry decorator only
    fires on transient failures:

    * :class:`NetworkTimeoutError` — ``requests.Timeout`` → retried.
    * :class:`ThumbnailDownloadError` — connection error or 5xx → retried.
    * :class:`InvalidThumbnailError` — 403 / 404 / 410 → not retried.

    Args:
        url:     Thumbnail URL to fetch.
        timeout: Maximum seconds to wait for a complete response.

    Returns:
        Raw image bytes.

    Raises:
        NetworkTimeoutError:   On request timeout.
        ThumbnailDownloadError: On connection failure or server 5xx.
        InvalidThumbnailError:  On permanent HTTP errors (403, 404, 410).
    """
    try:
        response = requests.get(url, timeout=timeout, stream=False)
        response.raise_for_status()
        return response.content

    except requests.Timeout as exc:
        logger.warning("Request timed out for {url}", url=url)
        raise NetworkTimeoutError(f"Request timed out: {url!r}") from exc

    except requests.HTTPError as exc:
        status_code: int = exc.response.status_code if exc.response is not None else 0
        if status_code in THUMBNAIL_PERMANENT_HTTP_ERRORS:
            logger.error(
                "Permanent HTTP {code} for {url}",
                code=status_code,
                url=url,
            )
            raise InvalidThumbnailError(
                f"HTTP {status_code} — thumbnail not available: {url!r}"
            ) from exc
        logger.warning(
            "Transient HTTP {code} for {url}",
            code=status_code,
            url=url,
        )
        raise ThumbnailDownloadError(
            f"HTTP {status_code} error downloading thumbnail: {url!r}"
        ) from exc

    except requests.ConnectionError as exc:
        logger.warning("Connection error for {url}: {exc}", url=url, exc=exc)
        raise ThumbnailDownloadError(
            f"Connection error downloading thumbnail: {url!r}"
        ) from exc

    except requests.RequestException as exc:
        logger.warning("Unexpected request error for {url}: {exc}", url=url, exc=exc)
        raise ThumbnailDownloadError(
            f"Unexpected error downloading thumbnail: {url!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def download_thumbnail(
    thumbnail_url: str,
    timeout: float = THUMBNAIL_REQUEST_TIMEOUT_SECONDS,
) -> bytes:
    """
    Download a thumbnail from ``thumbnail_url`` and return its bytes.

    Retries up to :data:`~config.THUMBNAIL_MAX_RETRY_ATTEMPTS` times on
    transient network failures.  Never retries permanent HTTP errors.

    Args:
        thumbnail_url: Fully-qualified URL to the thumbnail image.
        timeout:       HTTP response timeout in seconds.

    Returns:
        Raw image bytes suitable for passing to :func:`save_thumbnail`.

    Raises:
        ThumbnailDownloadError: On transient network failure after all retries.
        NetworkTimeoutError:    On request timeout after all retries.
        InvalidThumbnailError:  On permanent HTTP error (403/404/410).
    """
    if not thumbnail_url or not thumbnail_url.strip():
        raise InvalidThumbnailError("thumbnail_url is empty or None")

    logger.info("Downloading thumbnail: {url}", url=thumbnail_url)
    data = _fetch_thumbnail_bytes(thumbnail_url, timeout=timeout)
    logger.info(
        "Download complete: {size} bytes from {url}",
        size=len(data),
        url=thumbnail_url,
    )
    return data


def validate_image(image_path: Path) -> None:
    """
    Verify that ``image_path`` contains a valid, non-empty image.

    Validation steps, in order:

    1. File size must be at least
       :data:`~config.THUMBNAIL_MIN_FILE_SIZE_BYTES` bytes.
    2. Pillow must be able to open **and** fully decode the file
       (``img.load()`` forces pixel data to be read, catching truncation
       and most forms of corruption that ``verify()`` can miss).
    3. The reported image format must be in
       :data:`~config.THUMBNAIL_ACCEPTED_IMAGE_FORMATS`.

    Args:
        image_path: Path to the image file to validate.

    Raises:
        ImageValidationError: If any of the above checks fail.
    """
    # --- size check ---
    try:
        size = image_path.stat().st_size
    except OSError as exc:
        raise ImageValidationError(
            f"Cannot stat image file {image_path}: {exc}"
        ) from exc

    if size == 0:
        raise ImageValidationError(f"Image file is empty: {image_path}")
    if size < THUMBNAIL_MIN_FILE_SIZE_BYTES:
        raise ImageValidationError(
            f"Image file is too small ({size} bytes < "
            f"{THUMBNAIL_MIN_FILE_SIZE_BYTES} bytes): {image_path}"
        )

    # --- Pillow decode check ---
    try:
        with Image.open(image_path) as img:
            img.load()  # Forces full pixel-data decode; catches corruption.
            image_format: Optional[str] = img.format
    except UnidentifiedImageError as exc:
        raise ImageValidationError(
            f"Pillow cannot identify image format: {image_path}"
        ) from exc
    except (OSError, SyntaxError) as exc:
        raise ImageValidationError(
            f"Image file is corrupted or truncated: {image_path} — {exc}"
        ) from exc

    # --- format allowlist ---
    if image_format not in THUMBNAIL_ACCEPTED_IMAGE_FORMATS:
        raise ImageValidationError(
            f"Unsupported image format {image_format!r} for {image_path}. "
            f"Accepted: {sorted(THUMBNAIL_ACCEPTED_IMAGE_FORMATS)}"
        )

    logger.debug(
        "Image validated: {path} ({fmt}, {size} bytes)",
        path=image_path,
        fmt=image_format,
        size=size,
    )


def save_thumbnail(image_data: bytes, thumbnail_path: Path) -> None:
    """
    Write ``image_data`` to ``thumbnail_path`` atomically.

    The bytes are first written to a ``.tmp`` sibling file in the same
    directory, then moved into place with :func:`os.replace` (via
    :meth:`Path.replace`).  A concurrent reader of ``thumbnail_path``
    will therefore always see either the previous complete file or the
    newly-written complete file — never a partial write.

    Args:
        image_data:     Raw image bytes to persist.
        thumbnail_path: Destination file path.

    Raises:
        ThumbnailCacheError: If the directory cannot be created or the
            file cannot be written.
    """
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = thumbnail_path.with_suffix(".tmp")

    try:
        tmp_path.write_bytes(image_data)
        tmp_path.replace(thumbnail_path)
        logger.debug(
            "Saved thumbnail: {path} ({size} bytes)",
            path=thumbnail_path,
            size=len(image_data),
        )
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        logger.error(
            "Failed to save thumbnail to {path}: {exc}",
            path=thumbnail_path,
            exc=exc,
        )
        raise ThumbnailCacheError(
            f"Could not write thumbnail to {thumbnail_path}: {exc}"
        ) from exc


def load_cached_thumbnail(
    video_id: str,
    thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
) -> Optional[Path]:
    """
    Return the local path if a cached thumbnail exists for ``video_id``.

    This function performs **only** an existence check; it does not
    validate the image.  Callers that need a guaranteed-valid file
    should follow up with :func:`validate_image`.

    Args:
        video_id:      The 11-character YouTube video ID.
        thumbnail_dir: Root directory for downloaded thumbnails.

    Returns:
        The :class:`Path` to the cached file, or ``None`` if absent.
    """
    path = _thumbnail_path(video_id, thumbnail_dir)
    if path.exists():
        logger.debug("Cache hit: {path}", path=path)
        return path
    logger.debug("Cache miss for video_id={id}", id=video_id)
    return None


def process_thumbnail(
    metadata: VideoMetadata,
    thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
) -> ThumbnailData:
    """
    Run the complete thumbnail pipeline for a single video.

    Pipeline steps:

    1. Check the local cache.  If a valid cached thumbnail exists,
       return it immediately without a network request.
    2. If the cached file exists but fails validation (corrupted / empty),
       delete it and proceed to step 3.
    3. Validate that ``metadata.thumbnail_url`` is non-empty.
    4. Download the thumbnail bytes via :func:`download_thumbnail` (with
       automatic retries on transient failures).
    5. Persist the bytes atomically via :func:`save_thumbnail`.
    6. Validate the saved file via :func:`validate_image`.
    7. Return a :class:`ThumbnailData` combining the original
       ``metadata`` with the local file path.

    Unlike Module 2's ``process_video``, this function **raises**
    exceptions on failure rather than returning an error record.  That
    is intentional: the thumbnail is a hard dependency of Module 4, so
    the caller must decide whether to skip, retry, or abort.

    Args:
        metadata:      Immutable :class:`VideoMetadata` from Module 2.
                       Must have ``video_id`` and ``thumbnail_url`` set.
        thumbnail_dir: Root directory for downloaded thumbnails.  Created
                       automatically if it does not exist.

    Returns:
        A frozen :class:`ThumbnailData` with the original ``metadata``
        and the absolute local path of the verified thumbnail.

    Raises:
        InvalidThumbnailError:   If ``thumbnail_url`` is missing or the
                                 server returns a permanent HTTP error.
        ThumbnailDownloadError:  If the download fails after all retries.
        NetworkTimeoutError:     If every attempt times out.
        ImageValidationError:    If the saved image fails Pillow validation.
        ThumbnailCacheError:     If the thumbnail cannot be written to disk.
    """
    video_id: str = metadata.video_id
    logger.info("Processing thumbnail for video_id={id}", id=video_id)

    # --- Step 1 & 2: Cache check with validation ---
    cached_path = load_cached_thumbnail(video_id, thumbnail_dir)
    if cached_path is not None:
        try:
            validate_image(cached_path)
            logger.info(
                "Returning cached thumbnail for video_id={id}: {path}",
                id=video_id,
                path=cached_path,
            )
            return ThumbnailData(
                metadata=metadata,
                thumbnail_path=str(cached_path),
            )
        except ImageValidationError as exc:
            logger.warning(
                "Cached thumbnail for video_id={id} failed validation "
                "({reason}) — re-downloading",
                id=video_id,
                reason=exc,
            )
            cached_path.unlink(missing_ok=True)

    # --- Step 3: Validate URL ---
    thumbnail_url: Optional[str] = metadata.thumbnail_url
    if not thumbnail_url or not thumbnail_url.strip():
        logger.error(
            "No thumbnail_url in VideoMetadata for video_id={id}", id=video_id
        )
        raise InvalidThumbnailError(
            f"VideoMetadata for video_id={video_id!r} has no thumbnail_url"
        )

    # --- Step 4: Download ---
    logger.info(
        "Downloading thumbnail for video_id={id} from {url}",
        id=video_id,
        url=thumbnail_url,
    )
    image_data = download_thumbnail(thumbnail_url)

    # --- Step 5: Save ---
    dest_path = _thumbnail_path(video_id, thumbnail_dir)
    save_thumbnail(image_data, dest_path)

    # --- Step 6: Validate ---
    validate_image(dest_path)

    logger.info(
        "Thumbnail pipeline complete for video_id={id}: {path}",
        id=video_id,
        path=dest_path,
    )

    # --- Step 7: Return ---
    return ThumbnailData(
        metadata=metadata,
        thumbnail_path=str(dest_path),
    )
