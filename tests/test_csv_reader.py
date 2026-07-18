"""
test_csv_reader.py
===================

Pytest suite for Module 1 (CSV Reader).

Each test uses a fresh, isolated CSV path under pytest's ``tmp_path``
fixture so that tests never touch the real ``data/creators.csv`` and
never interfere with each other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import csv_reader as cr  # noqa: E402


VALID_URL_1 = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VALID_URL_2 = "https://youtu.be/aBcDeFgHiJk"
VALID_URL_3 = "https://youtube.com/watch?v=ZzYyXxWwVvU"


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    """A CSV path inside an isolated temp directory, not yet created."""
    return tmp_path / "creators.csv"


@pytest.fixture
def populated_csv(csv_path: Path) -> Path:
    """A CSV pre-populated with two valid creators."""
    cr.add_creator(
        csv_path, cr.Creator(email="a@example.com", video_url=VALID_URL_1)
    )
    cr.add_creator(
        csv_path, cr.Creator(email="b@example.com", video_url=VALID_URL_2)
    )
    return csv_path


# ---------------------------------------------------------------------------
# CSV creation / missing file
# ---------------------------------------------------------------------------


def test_missing_csv_is_created_on_load(csv_path: Path):
    assert not csv_path.exists()
    creators = cr.load_all_creators(csv_path)
    assert creators == []
    assert csv_path.exists()


def test_created_csv_has_correct_header(csv_path: Path):
    cr.load_all_creators(csv_path)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    assert tuple(df.columns) == cr.CSV_COLUMNS


def test_missing_csv_is_created_on_add(csv_path: Path):
    assert not csv_path.exists()
    result = cr.add_creator(
        csv_path, cr.Creator(email="a@example.com", video_url=VALID_URL_1)
    )
    assert result is True
    assert csv_path.exists()


# ---------------------------------------------------------------------------
# Valid loading
# ---------------------------------------------------------------------------


def test_load_all_creators_returns_valid_rows(populated_csv: Path):
    creators = cr.load_all_creators(populated_csv)
    assert len(creators) == 2
    emails = {c.email for c in creators}
    assert emails == {"a@example.com", "b@example.com"}
    assert all(isinstance(c, cr.Creator) for c in creators)


def test_load_preserves_field_values(populated_csv: Path):
    creators = cr.load_all_creators(populated_csv)
    first = next(c for c in creators if c.email == "a@example.com")
    assert first.email == "a@example.com"
    assert first.video_url == VALID_URL_1


def test_load_ignores_blank_rows(csv_path: Path):
    csv_path.write_text(
        "email,video_url\n"
        "a@example.com," + VALID_URL_1 + "\n"
        "\n"
        "b@example.com," + VALID_URL_2 + "\n",
        encoding="utf-8",
    )
    creators = cr.load_all_creators(csv_path)
    assert len(creators) == 2


# ---------------------------------------------------------------------------
# Invalid email / URL handling
# ---------------------------------------------------------------------------


def test_load_skips_invalid_email(csv_path: Path):
    csv_path.write_text(
        "email,video_url\n"
        f"not-an-email,{VALID_URL_1}\n"
        f"valid@example.com,{VALID_URL_2}\n",
        encoding="utf-8",
    )
    creators = cr.load_all_creators(csv_path)
    assert len(creators) == 1
    assert creators[0].email == "valid@example.com"


def test_load_skips_invalid_youtube_url(csv_path: Path):
    csv_path.write_text(
        "email,video_url\n"
        "a@example.com,https://vimeo.com/12345\n"
        f"b@example.com,{VALID_URL_2}\n",
        encoding="utf-8",
    )
    creators = cr.load_all_creators(csv_path)
    assert len(creators) == 1
    assert creators[0].email == "b@example.com"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
    ],
)
def test_validate_youtube_url_accepts_known_formats(url: str):
    assert cr._validate_youtube_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://vimeo.com/12345",
        "https://youtube.com/watch?v=short",
        "ftp://youtu.be/dQw4w9WgXcQ",
        "youtube.com/watch?v=dQw4w9WgXcQwithtrailing",
    ],
)
def test_validate_youtube_url_rejects_bad_formats(url: str):
    assert cr._validate_youtube_url(url) is False


@pytest.mark.parametrize(
    "email",
    ["a@example.com", "first.last+tag@sub.example.co.uk"],
)
def test_validate_email_accepts_valid(email: str):
    assert cr._validate_email(email) is True


@pytest.mark.parametrize(
    "email",
    ["", "not-an-email", "missing@domain", "@example.com", "spaces in@example.com"],
)
def test_validate_email_rejects_invalid(email: str):
    assert cr._validate_email(email) is False


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


def test_add_creator_rejects_duplicate_email(populated_csv: Path):
    result = cr.add_creator(
        populated_csv,
        cr.Creator(email="a@example.com", video_url=VALID_URL_3),
    )
    assert result is False
    assert len(cr.load_all_creators(populated_csv)) == 2


def test_add_creator_rejects_duplicate_video_url(populated_csv: Path):
    result = cr.add_creator(
        populated_csv,
        cr.Creator(email="new@example.com", video_url=VALID_URL_1),
    )
    assert result is False
    assert len(cr.load_all_creators(populated_csv)) == 2


# ---------------------------------------------------------------------------
# add_creator / remove_creator / get_creator
# ---------------------------------------------------------------------------


def test_add_creator_success(csv_path: Path):
    result = cr.add_creator(
        csv_path, cr.Creator(email="a@example.com", video_url=VALID_URL_1)
    )
    assert result is True
    creators = cr.load_all_creators(csv_path)
    assert len(creators) == 1
    assert creators[0].email == "a@example.com"


def test_add_creator_preserves_existing_data(populated_csv: Path):
    cr.add_creator(
        populated_csv,
        cr.Creator(email="c@example.com", video_url=VALID_URL_3),
    )
    creators = cr.load_all_creators(populated_csv)
    emails = {c.email for c in creators}
    assert emails == {"a@example.com", "b@example.com", "c@example.com"}


def test_add_creator_rejects_invalid_data(csv_path: Path):
    result = cr.add_creator(
        csv_path, cr.Creator(email="not-an-email", video_url=VALID_URL_1)
    )
    assert result is False
    assert cr.load_all_creators(csv_path) == []


def test_remove_creator_success(populated_csv: Path):
    result = cr.remove_creator(populated_csv, "a@example.com")
    assert result is True
    creators = cr.load_all_creators(populated_csv)
    emails = {c.email for c in creators}
    assert emails == {"b@example.com"}


def test_remove_creator_not_found(populated_csv: Path):
    result = cr.remove_creator(populated_csv, "does-not-exist@example.com")
    assert result is False
    assert len(cr.load_all_creators(populated_csv)) == 2


def test_remove_creator_on_missing_csv(csv_path: Path):
    result = cr.remove_creator(csv_path, "a@example.com")
    assert result is False


def test_get_creator_found(populated_csv: Path):
    creator = cr.get_creator(populated_csv, "a@example.com")
    assert creator is not None
    assert creator.email == "a@example.com"
    assert creator.video_url == VALID_URL_1


def test_get_creator_not_found(populated_csv: Path):
    creator = cr.get_creator(populated_csv, "does-not-exist@example.com")
    assert creator is None


def test_get_creator_on_missing_csv(csv_path: Path):
    creator = cr.get_creator(csv_path, "a@example.com")
    assert creator is None


# ---------------------------------------------------------------------------
# Malformed / corrupted CSV
# ---------------------------------------------------------------------------


def test_load_malformed_schema_returns_empty_list(csv_path: Path):
    csv_path.write_text(
        "email,channel_name,video_url\na@example.com,SomeChannel," + VALID_URL_1 + "\n",
        encoding="utf-8",
    )
    creators = cr.load_all_creators(csv_path)
    assert creators == []


def test_load_corrupted_csv_does_not_raise(csv_path: Path):
    csv_path.write_text(
        'email,video_url\n"unterminated quote,' + VALID_URL_1,
        encoding="utf-8",
    )
    # Must never raise, regardless of how malformed the content is.
    creators = cr.load_all_creators(csv_path)
    assert isinstance(creators, list)


def test_load_empty_file_returns_empty_list(csv_path: Path):
    csv_path.write_text("", encoding="utf-8")
    creators = cr.load_all_creators(csv_path)
    assert creators == []


# ---------------------------------------------------------------------------
# Dataclass sanity
# ---------------------------------------------------------------------------


def test_creator_is_frozen():
    creator = cr.Creator(email="a@example.com", video_url=VALID_URL_1)
    with pytest.raises(Exception):
        creator.email = "b@example.com"  # type: ignore[misc]


def test_validate_row_rejects_missing_fields():
    assert cr._validate_row({"email": "", "video_url": VALID_URL_1}) is False
    assert cr._validate_row({"email": "a@example.com", "video_url": ""}) is False


# ---------------------------------------------------------------------------
# Locking / access errors / unexpected failures (error-handling paths)
# ---------------------------------------------------------------------------


class _RaisingCtx:
    """Context manager test double that raises ``func()`` on enter."""

    def __init__(self, func):
        self._func = func

    def __enter__(self):
        self._func()

    def __exit__(self, *exc_info):
        return False


def test_file_lock_timeout_raises_csv_access_error(csv_path: Path, monkeypatch):
    import portalocker

    class _FakeLock:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise portalocker.exceptions.LockException("simulated timeout")

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(cr.portalocker, "Lock", _FakeLock)
    with pytest.raises(cr.CSVAccessError):
        with cr._file_lock(csv_path):
            pass  # pragma: no cover - never reached


def test_file_lock_permission_error_raises_csv_access_error(csv_path: Path, monkeypatch):
    class _FakeLock:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise PermissionError("simulated permission denied")

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(cr.portalocker, "Lock", _FakeLock)
    with pytest.raises(cr.CSVAccessError):
        with cr._file_lock(csv_path):
            pass  # pragma: no cover - never reached


def test_load_all_creators_handles_access_error(csv_path: Path, monkeypatch):
    def _raise_access_error(*args, **kwargs):
        raise cr.CSVAccessError("simulated")

    monkeypatch.setattr(cr, "_file_lock", lambda path: _RaisingCtx(_raise_access_error))
    assert cr.load_all_creators(csv_path) == []


def test_load_all_creators_handles_unexpected_error(csv_path: Path, monkeypatch):
    def _raise_unexpected(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cr, "_file_lock", lambda path: _RaisingCtx(_raise_unexpected))
    assert cr.load_all_creators(csv_path) == []


def test_add_creator_handles_access_error(csv_path: Path, monkeypatch):
    def _raise_access_error(*args, **kwargs):
        raise cr.CSVAccessError("simulated")

    monkeypatch.setattr(cr, "_file_lock", lambda path: _RaisingCtx(_raise_access_error))
    result = cr.add_creator(
        csv_path, cr.Creator(email="a@example.com", video_url=VALID_URL_1)
    )
    assert result is False


def test_add_creator_handles_unexpected_error(csv_path: Path, monkeypatch):
    def _raise_unexpected(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cr, "_file_lock", lambda path: _RaisingCtx(_raise_unexpected))
    result = cr.add_creator(
        csv_path, cr.Creator(email="a@example.com", video_url=VALID_URL_1)
    )
    assert result is False


def test_remove_creator_rejects_empty_email(populated_csv: Path):
    assert cr.remove_creator(populated_csv, "") is False


def test_remove_creator_handles_unexpected_error(populated_csv: Path, monkeypatch):
    def _raise_unexpected(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cr, "_file_lock", lambda path: _RaisingCtx(_raise_unexpected))
    assert cr.remove_creator(populated_csv, "a@example.com") is False


def test_get_creator_rejects_empty_email(populated_csv: Path):
    assert cr.get_creator(populated_csv, "") is None


def test_get_creator_handles_unexpected_error(populated_csv: Path, monkeypatch):
    def _raise_unexpected(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cr, "_file_lock", lambda path: _RaisingCtx(_raise_unexpected))
    assert cr.get_creator(populated_csv, "a@example.com") is None
