"""
csv_reader.py
==============

Module 1 — CSV Reader for the YouTube Thumbnail Outreach Automation system.

This module owns all reading and writing of the ``creators.csv`` lead
file. The CSV is the single source of truth for three fields only:
``id``, ``email``, and ``video_url``. Every other attribute of a creator
(channel name, thumbnail URL, video title, etc.) is intentionally left
to be discovered by later modules from the YouTube video URL — storing
that information here would create duplicate sources of truth and
tightly couple this module to YouTube's API surface.

Public API
----------
- :class:`Creator` — immutable record representing one lead.
- :func:`load_all_creators` — read every valid creator from the CSV.
- :func:`add_creator` — append a new creator, rejecting duplicates.
- :func:`remove_creator` — delete a creator by id.
- :func:`get_creator` — fetch a single creator by id.

Design principles
------------------
- **Never crash the caller.** Every public function catches and logs
  its own failures and returns a safe default (``[]``, ``False``, or
  ``None``) rather than propagating an exception. Custom exceptions
  are still raised internally by the validation/IO layer so that the
  failure mode is precise and testable in isolation.
- **Atomic writes.** Every write goes to a temporary file in the same
  directory as the target CSV and is then moved into place with
  ``os.replace``, which is atomic on POSIX and Windows. A reader can
  never observe a half-written file.
- **Process-safe locking.** All reads and writes are wrapped in an
  exclusive ``portalocker`` file lock so that concurrent modules (or
  multiple instances of this module) cannot race on the same CSV.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import pandas as pd
import portalocker

from config import (
    CSV_COLUMNS,
    CSV_ENCODING,
    LOCK_CHECK_INTERVAL_SECONDS,
    LOCK_TIMEOUT_SECONDS,
    LOG_DATE_FORMAT,
    LOG_DIR,
    LOG_FORMAT,
    MODULE1_LOG_PATH,
    EMAIL_PATTERN,
    YOUTUBE_URL_PATTERN,
)

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class CSVReaderError(Exception):
    """Base class for all exceptions raised by this module."""


class CSVValidationError(CSVReaderError):
    """Raised when a row or schema fails validation."""


class InvalidEmailError(CSVValidationError):
    """Raised when an email address fails validation."""


class InvalidYouTubeURLError(CSVValidationError):
    """Raised when a YouTube URL fails validation."""


class DuplicateCreatorError(CSVReaderError):
    """Raised when an id, email, or video URL already exists in the CSV."""


class CSVAccessError(CSVReaderError):
    """Raised for filesystem-level failures: permissions, lock timeouts."""


class CSVCorruptionError(CSVReaderError):
    """Raised when the CSV file cannot be parsed or its schema is wrong."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Creator:
    """
    An immutable record representing a single outreach lead.

    Attributes:
        id: Unique identifier for the creator/lead. Free-form but must
            be non-empty after stripping whitespace.
        email: Contact email address for outreach.
        video_url: A YouTube video URL belonging to the creator's
            channel. This is the single source of truth from which all
            other channel metadata is derived by later modules.
    """

    id: str
    email: str
    video_url: str


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logger() -> logging.Logger:
    """
    Configure and return the module-level logger.

    Idempotent: calling this more than once (e.g. across repeated test
    imports) will not attach duplicate handlers to the logger.

    Returns:
        A :class:`logging.Logger` writing to ``logs/module1.log``.
    """
    logger = logging.getLogger("module1.csv_reader")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(MODULE1_LOG_PATH, encoding=CSV_ENCODING)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


_logger = _setup_logger()


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


@contextmanager
def _file_lock(csv_path: Path) -> Iterator[None]:
    """
    Acquire an exclusive, process-safe lock on ``csv_path``.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")

    try:
        with portalocker.Lock(
            str(lock_path),
            mode="a+",
            timeout=LOCK_TIMEOUT_SECONDS,
            check_interval=LOCK_CHECK_INTERVAL_SECONDS,
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        ):
            yield

    except portalocker.exceptions.LockException as exc:
        _logger.error("Timed out waiting for file lock on %s: %s", csv_path, exc)
        raise CSVAccessError(f"Could not acquire lock on {csv_path}") from exc

    except PermissionError as exc:
        _logger.error("Permission denied accessing %s: %s", csv_path, exc)
        raise CSVAccessError(f"Permission denied accessing {csv_path}") from exc


# ---------------------------------------------------------------------------
# CSV initialization
# ---------------------------------------------------------------------------


def _initialize_csv(csv_path: Path) -> None:
    """
    Create ``csv_path`` with the correct header row if it does not
    already exist, or is empty.

    Must be called while holding the lock for ``csv_path`` to avoid a
    race between two processes both attempting to create the file.

    Args:
        csv_path: Path to the CSV file.
    """
    needs_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    if needs_header:
        empty_df = pd.DataFrame(columns=CSV_COLUMNS)
        _atomic_write(csv_path, empty_df)
        _logger.info("Initialized new CSV with headers at %s", csv_path)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_email(email: str) -> bool:
    """
    Check whether ``email`` is a syntactically valid email address.

    Args:
        email: The email address to validate.

    Returns:
        True if valid, False otherwise.
    """
    if not email:
        return False
    return re.match(EMAIL_PATTERN, email.strip()) is not None


def _validate_youtube_url(url: str) -> bool:
    """
    Check whether ``url`` matches one of the accepted YouTube video URL
    formats (``youtube.com/watch?v=``, ``www.youtube.com/watch?v=``, or
    ``youtu.be/``).

    This performs syntactic validation only. It never makes a network
    request and never extracts the video ID for use elsewhere; that is
    the responsibility of a later module.

    Args:
        url: The URL to validate.

    Returns:
        True if valid, False otherwise.
    """
    if not url:
        return False
    return re.match(YOUTUBE_URL_PATTERN, url.strip()) is not None


def _validate_row(row: dict) -> bool:
    """
    Validate a single raw CSV row (as a dict of strings).

    Checks, in order: required fields present and non-empty, valid
    email, valid YouTube URL. Each failure is logged at WARNING level
    with the offending row's id (if available) so that data-quality
    issues are visible without halting the load.

    Args:
        row: Mapping of column name to raw string value.

    Returns:
        True if the row is valid and safe to convert into a
        :class:`Creator`, False otherwise.
    """
    row_id = str(row.get("id", "")).strip()
    email = str(row.get("email", "")).strip()
    video_url = str(row.get("video_url", "")).strip()

    if not row_id:
        _logger.warning("Skipping row: missing id (email=%r)", email)
        return False

    if not email:
        _logger.warning("Skipping row id=%s: missing email", row_id)
        return False

    if not video_url:
        _logger.warning("Skipping row id=%s: missing video_url", row_id)
        return False

    if not _validate_email(email):
        _logger.warning("Skipping row id=%s: invalid email %r", row_id, email)
        return False

    if not _validate_youtube_url(video_url):
        _logger.warning(
            "Skipping row id=%s: invalid YouTube URL %r", row_id, video_url
        )
        return False

    return True


def _validate_schema(df: pd.DataFrame) -> None:
    """
    Verify that ``df`` has exactly the expected columns.

    Args:
        df: The DataFrame loaded from the CSV.

    Raises:
        CSVCorruptionError: If the columns do not match
            :data:`config.CSV_COLUMNS`.
    """
    actual_columns = tuple(df.columns)
    if actual_columns != CSV_COLUMNS:
        raise CSVCorruptionError(
            f"CSV schema mismatch. Expected {CSV_COLUMNS}, got {actual_columns}"
        )


# ---------------------------------------------------------------------------
# DataFrame <-> Creator conversion
# ---------------------------------------------------------------------------


def _dataframe_to_creator(row: dict) -> Optional[Creator]:
    """
    Convert a single validated raw row into a :class:`Creator`.

    Args:
        row: Mapping of column name to raw string value. Assumed to
            have already passed :func:`_validate_row`.

    Returns:
        A :class:`Creator` instance, or None if the row is invalid.
    """
    if not _validate_row(row):
        return None
    return Creator(
        id=str(row["id"]).strip(),
        email=str(row["email"]).strip(),
        video_url=str(row["video_url"]).strip(),
    )


def _dataframe_to_creators(df: pd.DataFrame) -> list[Creator]:
    """
    Convert every row of ``df`` into :class:`Creator` instances,
    skipping (and logging) any row that fails validation.

    Args:
        df: DataFrame already confirmed to match the expected schema.

    Returns:
        List of valid :class:`Creator` objects, in file order.
    """
    creators: list[Creator] = []
    skipped = 0

    for record in df.to_dict(orient="records"):
        creator = _dataframe_to_creator(record)
        if creator is not None:
            creators.append(creator)
        else:
            skipped += 1

    _logger.info(
        "Converted %d valid creators, skipped %d invalid rows",
        len(creators),
        skipped,
    )
    return creators


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------


def _read_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    Read the raw CSV at ``csv_path`` into a DataFrame of strings.

    Must be called while holding the lock for ``csv_path``.

    Args:
        csv_path: Path to the CSV file. Assumed to already exist.

    Returns:
        DataFrame with all values read as strings (no type inference,
        no NaN coercion), preserving values like leading zeros in ids.

    Raises:
        CSVCorruptionError: If the file cannot be parsed as CSV, the
            schema does not match, or the bytes are not valid UTF-8.
    """
    try:
        df = pd.read_csv(
            csv_path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding=CSV_ENCODING,
        )
    except pd.errors.EmptyDataError:
        # File exists but has no header at all (e.g. truncated to 0
        # bytes between our existence check and the read). Treat as an
        # empty, schema-correct table.
        return pd.DataFrame(columns=CSV_COLUMNS)
    except pd.errors.ParserError as exc:
        raise CSVCorruptionError(f"Could not parse CSV at {csv_path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise CSVCorruptionError(
            f"CSV at {csv_path} is not valid UTF-8: {exc}"
        ) from exc

    _validate_schema(df)
    return df


def _atomic_write(csv_path: Path, df: pd.DataFrame) -> None:
    """
    Write ``df`` to ``csv_path`` atomically.

    The DataFrame is first written to a temporary file in the same
    directory as ``csv_path`` (so that the subsequent move is on the
    same filesystem and therefore atomic), then moved into place with
    ``os.replace``. Any reader of ``csv_path`` will see either the old
    file or the fully-written new file, never a partial write.

    Must be called while holding the lock for ``csv_path``.

    Args:
        csv_path: Destination path for the CSV.
        df: DataFrame to write, expected to match :data:`CSV_COLUMNS`.

    Raises:
        CSVAccessError: If the write fails due to a filesystem error.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path_str = tempfile.mkstemp(
        prefix=".creators_tmp_", suffix=".csv", dir=str(csv_path.parent)
    )
    tmp_path = Path(tmp_path_str)
    try:
        os.close(tmp_fd)
        df.to_csv(tmp_path, columns=CSV_COLUMNS, index=False, encoding=CSV_ENCODING)
        os.replace(tmp_path, csv_path)
    except OSError as exc:
        _logger.error("Atomic write to %s failed: %s", csv_path, exc)
        raise CSVAccessError(f"Failed to write CSV at {csv_path}: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_dataframe(csv_path: Path, df: pd.DataFrame) -> None:
    """
    Validate and persist ``df`` as the new contents of ``csv_path``.

    Must be called while holding the lock for ``csv_path``.

    Args:
        csv_path: Destination path for the CSV.
        df: DataFrame to write.
    """
    _validate_schema(df)
    _atomic_write(csv_path, df)


# ---------------------------------------------------------------------------
# Duplicate checking
# ---------------------------------------------------------------------------


def _check_duplicates(df: pd.DataFrame, creator: Creator) -> None:
    """
    Verify that ``creator`` does not collide with any existing row in
    ``df`` on id, email, or video_url.

    Args:
        df: Current contents of the CSV.
        creator: The candidate creator to check.

    Raises:
        DuplicateCreatorError: If id, email, or video_url already
            exists in ``df``.
    """
    if (df["id"] == creator.id).any():
        _logger.warning("Duplicate id rejected: %s", creator.id)
        raise DuplicateCreatorError(f"Duplicate id: {creator.id}")

    if (df["email"].str.lower() == creator.email.lower()).any():
        _logger.warning("Duplicate email rejected: %s", creator.email)
        raise DuplicateCreatorError(f"Duplicate email: {creator.email}")

    if (df["video_url"] == creator.video_url).any():
        _logger.warning("Duplicate video_url rejected: %s", creator.video_url)
        raise DuplicateCreatorError(f"Duplicate video_url: {creator.video_url}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_all_creators(csv_path: PathLike) -> list[Creator]:
    """
    Load every valid creator from the CSV at ``csv_path``.

    Creates the CSV (with headers) if it does not exist. Malformed or
    invalid rows are skipped and logged rather than raised; this
    function never raises an exception to the caller.

    Args:
        csv_path: Path to the creators CSV file.

    Returns:
        A list of valid :class:`Creator` objects, in file order. Empty
        list if the file has no valid rows, is newly created, or any
        error occurs while reading it.
    """
    path = Path(csv_path)
    _logger.info("Loading creators from %s", path)

    try:
        with _file_lock(path):
            _initialize_csv(path)
            df = _read_dataframe(path)
            creators = _dataframe_to_creators(df)
        _logger.info("Load complete: %d creators loaded from %s", len(creators), path)
        return creators

    except CSVCorruptionError as exc:
        _logger.error("CSV corruption while loading %s: %s", path, exc)
        return []
    except CSVAccessError as exc:
        _logger.error("Access error while loading %s: %s", path, exc)
        return []
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        _logger.exception("Unexpected error while loading %s: %s", path, exc)
        return []


def add_creator(csv_path: PathLike, creator: Creator) -> bool:
    """
    Append ``creator`` to the CSV at ``csv_path``.

    Rejects the creator (returns False) if its id, email, or
    video_url is already present, or if the creator itself fails
    field-level validation. Existing data is always preserved; the
    write is atomic.

    Args:
        csv_path: Path to the creators CSV file.
        creator: The creator to add.

    Returns:
        True if the creator was added successfully, False otherwise
        (validation failure, duplicate, or any I/O error).
    """
    path = Path(csv_path)

    candidate_row = {
        "id": creator.id,
        "email": creator.email,
        "video_url": creator.video_url,
    }
    if not _validate_row(candidate_row):
        _logger.warning("add_creator rejected invalid creator: %r", creator)
        return False

    try:
        with _file_lock(path):
            _initialize_csv(path)
            df = _read_dataframe(path)
            _check_duplicates(df, creator)

            new_row = pd.DataFrame([candidate_row], columns=CSV_COLUMNS)
            updated_df = pd.concat([df, new_row], ignore_index=True)
            _write_dataframe(path, updated_df)

        _logger.info("Added creator id=%s", creator.id)
        return True

    except DuplicateCreatorError as exc:
        _logger.warning("add_creator failed for id=%s: %s", creator.id, exc)
        return False
    except (CSVCorruptionError, CSVAccessError) as exc:
        _logger.error("add_creator failed for id=%s: %s", creator.id, exc)
        return False
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        _logger.exception("Unexpected error adding creator id=%s: %s", creator.id, exc)
        return False


def remove_creator(csv_path: PathLike, creator_id: str) -> bool:
    """
    Remove the creator with id ``creator_id`` from the CSV.

    Args:
        csv_path: Path to the creators CSV file.
        creator_id: The id of the creator to remove.

    Returns:
        True if a matching creator was found and removed, False if no
        match was found or any error occurred.
    """
    path = Path(csv_path)
    target_id = str(creator_id).strip()

    if not target_id:
        _logger.warning("remove_creator rejected empty creator_id")
        return False

    try:
        with _file_lock(path):
            _initialize_csv(path)
            df = _read_dataframe(path)

            mask = df["id"] == target_id
            if not mask.any():
                _logger.info("remove_creator: no match for id=%s", target_id)
                return False

            updated_df = df.loc[~mask].reset_index(drop=True)
            _write_dataframe(path, updated_df)

        _logger.info("Removed creator id=%s", target_id)
        return True

    except (CSVCorruptionError, CSVAccessError) as exc:
        _logger.error("remove_creator failed for id=%s: %s", target_id, exc)
        return False
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        _logger.exception(
            "Unexpected error removing creator id=%s: %s", target_id, exc
        )
        return False


def get_creator(csv_path: PathLike, creator_id: str) -> Optional[Creator]:
    """
    Fetch a single creator by id.

    Args:
        csv_path: Path to the creators CSV file.
        creator_id: The id of the creator to fetch.

    Returns:
        The matching :class:`Creator`, or None if not found or any
        error occurred while reading the CSV.
    """
    path = Path(csv_path)
    target_id = str(creator_id).strip()

    if not target_id:
        _logger.warning("get_creator rejected empty creator_id")
        return None

    try:
        with _file_lock(path):
            _initialize_csv(path)
            df = _read_dataframe(path)
            matches = df.loc[df["id"] == target_id]

        if matches.empty:
            _logger.info("get_creator: no match for id=%s", target_id)
            return None

        creator = _dataframe_to_creator(matches.iloc[0].to_dict())
        if creator is None:
            _logger.warning(
                "get_creator: row for id=%s exists but failed validation", target_id
            )
        return creator

    except (CSVCorruptionError, CSVAccessError) as exc:
        _logger.error("get_creator failed for id=%s: %s", target_id, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        _logger.exception("Unexpected error getting creator id=%s: %s", target_id, exc)
        return None
