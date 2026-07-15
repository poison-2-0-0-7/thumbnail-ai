"""
test_thumbnail_downloader.py
==============================

Pytest suite for Module 3 (YouTube Thumbnail Downloader).

All HTTP calls are mocked with ``unittest.mock.patch`` so the tests run
fully offline.  File-system operations use pytest's ``tmp_path`` fixture
for complete test isolation.

Coverage targets:
    - :func:`download_thumbnail`     URL validation, HTTP success,
                                     timeout, 404, 500, connection error
    - :func:`validate_image`         empty file, too-small file, corrupt
                                     image, unrecognised format, valid JPEG
    - :func:`save_thumbnail`         atomic write, directory creation,
                                     OS error propagation
    - :func:`load_cached_thumbnail`  cache hit, cache miss
    - :func:`process_thumbnail`      full pipeline, cache hit, cache miss,
                                     corrupted cache, missing URL, retry
    - Exception hierarchy            all Module 3 exceptions share base
    - Config constants               presence and correct types
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Path bootstrap — identical pattern used by test_youtube_metadata.py
# ---------------------------------------------------------------------------

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import (  # noqa: E402
    THUMBNAIL_ACCEPTED_IMAGE_FORMATS,
    THUMBNAIL_MAX_RETRY_ATTEMPTS,
    THUMBNAIL_MIN_FILE_SIZE_BYTES,
    THUMBNAIL_PERMANENT_HTTP_ERRORS,
    THUMBNAIL_REQUEST_TIMEOUT_SECONDS,
)
from models import ThumbnailData, VideoMetadata  # noqa: E402
from thumbnail_downloader import (  # noqa: E402
    ImageValidationError,
    InvalidThumbnailError,
    NetworkTimeoutError,
    ThumbnailCacheError,
    ThumbnailDownloadError,
    ThumbnailDownloaderError,
    _before_sleep_log,
    _fetch_thumbnail_bytes,
    _thumbnail_path,
    download_thumbnail,
    load_cached_thumbnail,
    process_thumbnail,
    save_thumbnail,
    validate_image,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

VALID_VIDEO_ID = "dQw4w9WgXcQ"
VALID_THUMBNAIL_URL = "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg"


def _make_valid_jpeg_bytes() -> bytes:
    """Return a valid JPEG that exceeds THUMBNAIL_MIN_FILE_SIZE_BYTES."""
    buf = io.BytesIO()
    # 200×200 solid-color JPEG is ~1.3 KB, safely above the 1 KB minimum.
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_valid_png_bytes() -> bytes:
    """Return a valid PNG that exceeds THUMBNAIL_MIN_FILE_SIZE_BYTES.

    Solid-color PNGs compress very aggressively; we use random pixel data
    so the compressed output stays above the 1 KB minimum.
    """
    import random

    buf = io.BytesIO()
    img = Image.new("RGB", (30, 30))
    img.putdata(
        [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(30 * 30)
        ]
    )
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_bmp_bytes_above_min() -> bytes:
    """Return a valid BMP image that exceeds THUMBNAIL_MIN_FILE_SIZE_BYTES.

    BMP is not in THUMBNAIL_ACCEPTED_IMAGE_FORMATS, making it useful for
    testing the format allowlist rejection.  A 30×30 BMP is ~2.8 KB.
    """
    buf = io.BytesIO()
    img = Image.new("RGB", (30, 30), color=(0, 0, 255))
    img.save(buf, format="BMP")
    return buf.getvalue()


def _make_metadata(
    thumbnail_url: str | None = VALID_THUMBNAIL_URL,
    video_id: str = VALID_VIDEO_ID,
) -> VideoMetadata:
    """Build a minimal VideoMetadata for testing."""
    return VideoMetadata(
        video_id=video_id,
        title="Test Video",
        uploader="Test Channel",
        uploader_id="@test",
        channel_id="UCtest",
        thumbnail_url=thumbnail_url,
        status="success",
    )


def _make_http_error(status_code: int) -> "requests.HTTPError":
    """Build a mock requests.HTTPError for the given status code."""
    import requests

    response = MagicMock()
    response.status_code = status_code
    exc = requests.HTTPError(response=response)
    return exc


# ---------------------------------------------------------------------------
# Config constant sanity checks
# ---------------------------------------------------------------------------


class TestConfigConstants:
    def test_thumbnail_accepted_formats_is_frozenset(self) -> None:
        assert isinstance(THUMBNAIL_ACCEPTED_IMAGE_FORMATS, frozenset)
        assert "JPEG" in THUMBNAIL_ACCEPTED_IMAGE_FORMATS

    def test_permanent_http_errors_contains_404(self) -> None:
        assert 404 in THUMBNAIL_PERMANENT_HTTP_ERRORS
        assert 403 in THUMBNAIL_PERMANENT_HTTP_ERRORS
        assert 410 in THUMBNAIL_PERMANENT_HTTP_ERRORS

    def test_min_file_size_is_positive(self) -> None:
        assert THUMBNAIL_MIN_FILE_SIZE_BYTES > 0

    def test_max_retry_attempts_is_at_least_two(self) -> None:
        assert THUMBNAIL_MAX_RETRY_ATTEMPTS >= 2

    def test_request_timeout_is_positive(self) -> None:
        assert THUMBNAIL_REQUEST_TIMEOUT_SECONDS > 0


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_all_exceptions_inherit_from_base(self) -> None:
        for cls in (
            ThumbnailDownloadError,
            NetworkTimeoutError,
            InvalidThumbnailError,
            ImageValidationError,
            ThumbnailCacheError,
        ):
            assert issubclass(cls, ThumbnailDownloaderError), (
                f"{cls.__name__} must subclass ThumbnailDownloaderError"
            )

    def test_base_inherits_from_exception(self) -> None:
        assert issubclass(ThumbnailDownloaderError, Exception)

    def test_exceptions_can_be_instantiated_with_message(self) -> None:
        for cls in (
            ThumbnailDownloadError,
            NetworkTimeoutError,
            InvalidThumbnailError,
            ImageValidationError,
            ThumbnailCacheError,
        ):
            exc = cls("test message")
            assert "test message" in str(exc)


# ---------------------------------------------------------------------------
# _thumbnail_path helper
# ---------------------------------------------------------------------------


class TestThumbnailPath:
    def test_returns_path_with_video_id_and_jpg_extension(
        self, tmp_path: Path
    ) -> None:
        result = _thumbnail_path(VALID_VIDEO_ID, tmp_path)
        assert result.name == f"{VALID_VIDEO_ID}.jpg"
        assert result.parent == tmp_path

    def test_different_video_ids_produce_different_paths(
        self, tmp_path: Path
    ) -> None:
        p1 = _thumbnail_path("aaaaaaaaaaa", tmp_path)
        p2 = _thumbnail_path("bbbbbbbbbbb", tmp_path)
        assert p1 != p2


# ---------------------------------------------------------------------------
# _fetch_thumbnail_bytes (internal, retried)
# ---------------------------------------------------------------------------


class TestFetchThumbnailBytes:
    @patch("thumbnail_downloader.requests.get")
    def test_success_returns_content(self, mock_get: MagicMock) -> None:
        jpeg = _make_valid_jpeg_bytes()
        mock_get.return_value.content = jpeg
        mock_get.return_value.raise_for_status = MagicMock()

        result = _fetch_thumbnail_bytes.__wrapped__(VALID_THUMBNAIL_URL)
        assert result == jpeg

    @patch("thumbnail_downloader.requests.get")
    def test_timeout_raises_network_timeout_error(
        self, mock_get: MagicMock
    ) -> None:
        import requests

        mock_get.side_effect = requests.Timeout("timed out")
        with pytest.raises(NetworkTimeoutError):
            _fetch_thumbnail_bytes.__wrapped__(VALID_THUMBNAIL_URL)

    @patch("thumbnail_downloader.requests.get")
    def test_http_404_raises_invalid_thumbnail_error(
        self, mock_get: MagicMock
    ) -> None:
        mock_get.return_value.raise_for_status.side_effect = _make_http_error(404)
        with pytest.raises(InvalidThumbnailError):
            _fetch_thumbnail_bytes.__wrapped__(VALID_THUMBNAIL_URL)

    @patch("thumbnail_downloader.requests.get")
    def test_http_403_raises_invalid_thumbnail_error(
        self, mock_get: MagicMock
    ) -> None:
        mock_get.return_value.raise_for_status.side_effect = _make_http_error(403)
        with pytest.raises(InvalidThumbnailError):
            _fetch_thumbnail_bytes.__wrapped__(VALID_THUMBNAIL_URL)

    @patch("thumbnail_downloader.requests.get")
    def test_http_410_raises_invalid_thumbnail_error(
        self, mock_get: MagicMock
    ) -> None:
        mock_get.return_value.raise_for_status.side_effect = _make_http_error(410)
        with pytest.raises(InvalidThumbnailError):
            _fetch_thumbnail_bytes.__wrapped__(VALID_THUMBNAIL_URL)

    @patch("thumbnail_downloader.requests.get")
    def test_http_500_raises_thumbnail_download_error(
        self, mock_get: MagicMock
    ) -> None:
        mock_get.return_value.raise_for_status.side_effect = _make_http_error(500)
        with pytest.raises(ThumbnailDownloadError):
            _fetch_thumbnail_bytes.__wrapped__(VALID_THUMBNAIL_URL)

    @patch("thumbnail_downloader.requests.get")
    def test_connection_error_raises_thumbnail_download_error(
        self, mock_get: MagicMock
    ) -> None:
        import requests

        mock_get.side_effect = requests.ConnectionError("refused")
        with pytest.raises(ThumbnailDownloadError):
            _fetch_thumbnail_bytes.__wrapped__(VALID_THUMBNAIL_URL)

    @patch("thumbnail_downloader.requests.get")
    def test_generic_request_exception_raises_thumbnail_download_error(
        self, mock_get: MagicMock
    ) -> None:
        import requests

        mock_get.side_effect = requests.RequestException("unknown")
        with pytest.raises(ThumbnailDownloadError):
            _fetch_thumbnail_bytes.__wrapped__(VALID_THUMBNAIL_URL)


# ---------------------------------------------------------------------------
# download_thumbnail (public wrapper)
# ---------------------------------------------------------------------------


class TestDownloadThumbnail:
    @patch("thumbnail_downloader._fetch_thumbnail_bytes")
    def test_success_returns_bytes(self, mock_fetch: MagicMock) -> None:
        jpeg = _make_valid_jpeg_bytes()
        mock_fetch.return_value = jpeg

        result = download_thumbnail(VALID_THUMBNAIL_URL)
        assert result == jpeg

    def test_empty_url_raises_invalid_thumbnail_error(self) -> None:
        with pytest.raises(InvalidThumbnailError):
            download_thumbnail("")

    def test_whitespace_only_url_raises_invalid_thumbnail_error(self) -> None:
        with pytest.raises(InvalidThumbnailError):
            download_thumbnail("   ")

    @patch("thumbnail_downloader._fetch_thumbnail_bytes")
    def test_propagates_network_timeout_error(
        self, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.side_effect = NetworkTimeoutError("timeout")
        with pytest.raises(NetworkTimeoutError):
            download_thumbnail(VALID_THUMBNAIL_URL)

    @patch("thumbnail_downloader._fetch_thumbnail_bytes")
    def test_propagates_invalid_thumbnail_error(
        self, mock_fetch: MagicMock
    ) -> None:
        mock_fetch.side_effect = InvalidThumbnailError("404")
        with pytest.raises(InvalidThumbnailError):
            download_thumbnail(VALID_THUMBNAIL_URL)


# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------


class TestValidateImage:
    def test_valid_jpeg_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "thumb.jpg"
        path.write_bytes(_make_valid_jpeg_bytes())
        validate_image(path)  # Must not raise

    def test_valid_png_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "thumb.png"
        path.write_bytes(_make_valid_png_bytes())
        validate_image(path)  # Must not raise

    def test_empty_file_raises_image_validation_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
        with pytest.raises(ImageValidationError, match="empty"):
            validate_image(path)

    def test_file_below_min_size_raises_image_validation_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "tiny.jpg"
        path.write_bytes(b"\xff" * (THUMBNAIL_MIN_FILE_SIZE_BYTES - 1))
        with pytest.raises(ImageValidationError, match="too small"):
            validate_image(path)

    def test_corrupt_bytes_raise_image_validation_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "corrupt.jpg"
        # Enough bytes to pass size check, but not valid JPEG
        path.write_bytes(b"NOTANIMAGE" * 200)
        with pytest.raises(ImageValidationError):
            validate_image(path)

    def test_non_existent_file_raises_image_validation_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "missing.jpg"
        with pytest.raises(ImageValidationError):
            validate_image(path)

    def test_unsupported_format_raises_image_validation_error(
        self, tmp_path: Path
    ) -> None:
        # BMP is a real Pillow-readable format not in ACCEPTED_IMAGE_FORMATS.
        # A 30×30 BMP is ~2.8 KB so it passes the size check but fails the
        # format allowlist check.
        path = tmp_path / "thumb.bmp"
        path.write_bytes(_make_bmp_bytes_above_min())
        with pytest.raises(ImageValidationError, match="Unsupported image format"):
            validate_image(path)


# ---------------------------------------------------------------------------
# save_thumbnail
# ---------------------------------------------------------------------------


class TestSaveThumbnail:
    def test_creates_file_with_correct_content(self, tmp_path: Path) -> None:
        data = _make_valid_jpeg_bytes()
        dest = tmp_path / "thumb.jpg"
        save_thumbnail(data, dest)
        assert dest.exists()
        assert dest.read_bytes() == data

    def test_creates_parent_directories_if_missing(
        self, tmp_path: Path
    ) -> None:
        data = _make_valid_jpeg_bytes()
        dest = tmp_path / "a" / "b" / "c" / "thumb.jpg"
        save_thumbnail(data, dest)
        assert dest.exists()

    def test_write_is_atomic_no_tmp_file_left_on_success(
        self, tmp_path: Path
    ) -> None:
        data = _make_valid_jpeg_bytes()
        dest = tmp_path / "thumb.jpg"
        save_thumbnail(data, dest)
        assert not (tmp_path / "thumb.tmp").exists()

    def test_overwriting_existing_file_succeeds(self, tmp_path: Path) -> None:
        dest = tmp_path / "thumb.jpg"
        dest.write_bytes(b"old")
        new_data = _make_valid_jpeg_bytes()
        save_thumbnail(new_data, dest)
        assert dest.read_bytes() == new_data

    def test_os_error_raises_thumbnail_cache_error(
        self, tmp_path: Path
    ) -> None:
        data = _make_valid_jpeg_bytes()
        dest = tmp_path / "thumb.jpg"
        with patch("thumbnail_downloader.Path.replace", side_effect=OSError("full")):
            with pytest.raises(ThumbnailCacheError):
                save_thumbnail(data, dest)


# ---------------------------------------------------------------------------
# load_cached_thumbnail
# ---------------------------------------------------------------------------


class TestLoadCachedThumbnail:
    def test_returns_none_when_no_cached_file(self, tmp_path: Path) -> None:
        result = load_cached_thumbnail(VALID_VIDEO_ID, tmp_path)
        assert result is None

    def test_returns_path_when_cached_file_exists(
        self, tmp_path: Path
    ) -> None:
        expected = tmp_path / f"{VALID_VIDEO_ID}.jpg"
        expected.write_bytes(_make_valid_jpeg_bytes())
        result = load_cached_thumbnail(VALID_VIDEO_ID, tmp_path)
        assert result == expected

    def test_returns_none_for_different_video_id(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "aaaaaaaaaaa.jpg").write_bytes(b"x" * 2000)
        result = load_cached_thumbnail("bbbbbbbbbbb", tmp_path)
        assert result is None

    def test_does_not_validate_image_content(self, tmp_path: Path) -> None:
        """load_cached_thumbnail returns path even for corrupt files."""
        path = tmp_path / f"{VALID_VIDEO_ID}.jpg"
        path.write_bytes(b"garbage" * 300)
        result = load_cached_thumbnail(VALID_VIDEO_ID, tmp_path)
        assert result == path


# ---------------------------------------------------------------------------
# process_thumbnail — full pipeline
# ---------------------------------------------------------------------------


class TestProcessThumbnail:
    @patch("thumbnail_downloader.download_thumbnail")
    def test_success_returns_thumbnail_data(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        jpeg = _make_valid_jpeg_bytes()
        mock_dl.return_value = jpeg
        metadata = _make_metadata()

        result = process_thumbnail(metadata, thumbnail_dir=tmp_path)

        assert isinstance(result, ThumbnailData)
        assert result.metadata is metadata
        assert Path(result.thumbnail_path).exists()
        assert Path(result.thumbnail_path).read_bytes() == jpeg

    @patch("thumbnail_downloader.download_thumbnail")
    def test_thumbnail_is_saved_to_correct_path(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        mock_dl.return_value = _make_valid_jpeg_bytes()
        metadata = _make_metadata()

        result = process_thumbnail(metadata, thumbnail_dir=tmp_path)

        expected_path = tmp_path / f"{VALID_VIDEO_ID}.jpg"
        assert result.thumbnail_path == str(expected_path)

    @patch("thumbnail_downloader.download_thumbnail")
    def test_cache_hit_skips_download(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        # Pre-populate cache with a valid image
        cached = tmp_path / f"{VALID_VIDEO_ID}.jpg"
        cached.write_bytes(_make_valid_jpeg_bytes())
        metadata = _make_metadata()

        result = process_thumbnail(metadata, thumbnail_dir=tmp_path)

        mock_dl.assert_not_called()
        assert Path(result.thumbnail_path) == cached

    @patch("thumbnail_downloader.download_thumbnail")
    def test_corrupted_cache_triggers_redownload(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        # Write corrupt bytes to cache
        cached = tmp_path / f"{VALID_VIDEO_ID}.jpg"
        cached.write_bytes(b"NOT_AN_IMAGE" * 200)
        fresh_jpeg = _make_valid_jpeg_bytes()
        mock_dl.return_value = fresh_jpeg
        metadata = _make_metadata()

        result = process_thumbnail(metadata, thumbnail_dir=tmp_path)

        mock_dl.assert_called_once()
        assert Path(result.thumbnail_path).read_bytes() == fresh_jpeg

    @patch("thumbnail_downloader.download_thumbnail")
    def test_empty_cached_file_triggers_redownload(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        cached = tmp_path / f"{VALID_VIDEO_ID}.jpg"
        cached.write_bytes(b"")
        mock_dl.return_value = _make_valid_jpeg_bytes()

        process_thumbnail(_make_metadata(), thumbnail_dir=tmp_path)
        mock_dl.assert_called_once()

    def test_missing_thumbnail_url_raises_invalid_thumbnail_error(
        self, tmp_path: Path
    ) -> None:
        metadata = _make_metadata(thumbnail_url=None)
        with pytest.raises(InvalidThumbnailError):
            process_thumbnail(metadata, thumbnail_dir=tmp_path)

    def test_empty_thumbnail_url_raises_invalid_thumbnail_error(
        self, tmp_path: Path
    ) -> None:
        metadata = _make_metadata(thumbnail_url="")
        with pytest.raises(InvalidThumbnailError):
            process_thumbnail(metadata, thumbnail_dir=tmp_path)

    @patch("thumbnail_downloader.download_thumbnail")
    def test_download_error_propagates(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        mock_dl.side_effect = ThumbnailDownloadError("server error")
        with pytest.raises(ThumbnailDownloadError):
            process_thumbnail(_make_metadata(), thumbnail_dir=tmp_path)

    @patch("thumbnail_downloader.download_thumbnail")
    def test_network_timeout_propagates(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        mock_dl.side_effect = NetworkTimeoutError("timeout")
        with pytest.raises(NetworkTimeoutError):
            process_thumbnail(_make_metadata(), thumbnail_dir=tmp_path)

    @patch("thumbnail_downloader.download_thumbnail")
    def test_invalid_thumbnail_error_propagates(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        mock_dl.side_effect = InvalidThumbnailError("HTTP 404")
        with pytest.raises(InvalidThumbnailError):
            process_thumbnail(_make_metadata(), thumbnail_dir=tmp_path)

    @patch("thumbnail_downloader.download_thumbnail")
    def test_downloaded_corrupt_image_raises_image_validation_error(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        # Server returns something that looks like bytes but isn't a real image
        mock_dl.return_value = b"FAKEIMAGE" * 200
        with pytest.raises(ImageValidationError):
            process_thumbnail(_make_metadata(), thumbnail_dir=tmp_path)

    @patch("thumbnail_downloader.download_thumbnail")
    def test_returned_thumbnail_data_is_frozen(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        mock_dl.return_value = _make_valid_jpeg_bytes()
        result = process_thumbnail(_make_metadata(), thumbnail_dir=tmp_path)
        with pytest.raises(Exception):
            result.thumbnail_path = "/changed"  # type: ignore[misc]

    @patch("thumbnail_downloader.download_thumbnail")
    def test_returned_metadata_is_original_object(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        mock_dl.return_value = _make_valid_jpeg_bytes()
        metadata = _make_metadata()
        result = process_thumbnail(metadata, thumbnail_dir=tmp_path)
        assert result.metadata is metadata

    @patch("thumbnail_downloader.download_thumbnail")
    def test_thumbnail_directory_is_created_automatically(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        nested_dir = tmp_path / "x" / "y" / "z"
        assert not nested_dir.exists()
        mock_dl.return_value = _make_valid_jpeg_bytes()
        process_thumbnail(_make_metadata(), thumbnail_dir=nested_dir)
        assert nested_dir.is_dir()

    @patch("thumbnail_downloader.download_thumbnail")
    def test_second_call_for_same_video_uses_cache(
        self, mock_dl: MagicMock, tmp_path: Path
    ) -> None:
        mock_dl.return_value = _make_valid_jpeg_bytes()
        meta = _make_metadata()
        process_thumbnail(meta, thumbnail_dir=tmp_path)
        process_thumbnail(meta, thumbnail_dir=tmp_path)
        assert mock_dl.call_count == 1


# ---------------------------------------------------------------------------
# Retry integration
# ---------------------------------------------------------------------------


class TestRetryBehaviour:
    @patch("thumbnail_downloader.requests.get")
    def test_retries_on_connection_error(self, mock_get: MagicMock) -> None:
        import requests as req_lib

        jpeg = _make_valid_jpeg_bytes()
        success_response = MagicMock()
        success_response.content = jpeg
        success_response.raise_for_status = MagicMock()

        # Fail twice, succeed on third attempt
        mock_get.side_effect = [
            req_lib.ConnectionError("refused"),
            req_lib.ConnectionError("refused"),
            success_response,
        ]

        result = _fetch_thumbnail_bytes(VALID_THUMBNAIL_URL)
        assert result == jpeg
        assert mock_get.call_count == 3

    @patch("thumbnail_downloader.requests.get")
    def test_does_not_retry_on_404(self, mock_get: MagicMock) -> None:
        mock_get.return_value.raise_for_status.side_effect = _make_http_error(404)

        with pytest.raises(InvalidThumbnailError):
            _fetch_thumbnail_bytes(VALID_THUMBNAIL_URL)

        # Should have been called only once — no retries for permanent errors
        assert mock_get.call_count == 1

    @patch("thumbnail_downloader.requests.get")
    def test_raises_after_exhausting_all_retries(
        self, mock_get: MagicMock
    ) -> None:
        import requests as req_lib

        mock_get.side_effect = req_lib.ConnectionError("refused")

        with pytest.raises(ThumbnailDownloadError):
            _fetch_thumbnail_bytes(VALID_THUMBNAIL_URL)

        assert mock_get.call_count == THUMBNAIL_MAX_RETRY_ATTEMPTS

    @patch("thumbnail_downloader.requests.get")
    def test_retries_on_timeout(self, mock_get: MagicMock) -> None:
        import requests as req_lib

        jpeg = _make_valid_jpeg_bytes()
        success_response = MagicMock()
        success_response.content = jpeg
        success_response.raise_for_status = MagicMock()

        mock_get.side_effect = [
            req_lib.Timeout("slow"),
            success_response,
        ]

        result = _fetch_thumbnail_bytes(VALID_THUMBNAIL_URL)
        assert result == jpeg
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# _before_sleep_log (Tenacity hook)
# ---------------------------------------------------------------------------


class TestBeforeSleepLog:
    def test_logs_without_raising_when_outcome_is_none(self) -> None:
        state = MagicMock()
        state.fn = None
        state.attempt_number = 1
        state.outcome = None
        _before_sleep_log(state)  # Must not raise

    def test_logs_without_raising_with_exception(self) -> None:
        state = MagicMock()
        state.fn.__name__ = "_fetch_thumbnail_bytes"
        state.attempt_number = 2
        state.outcome.exception.return_value = ThumbnailDownloadError("boom")
        _before_sleep_log(state)  # Must not raise


# ---------------------------------------------------------------------------
# ThumbnailData model
# ---------------------------------------------------------------------------


class TestThumbnailDataModel:
    def test_model_is_frozen(self, tmp_path: Path) -> None:
        meta = _make_metadata()
        td = ThumbnailData(metadata=meta, thumbnail_path="/tmp/test.jpg")
        with pytest.raises(Exception):
            td.thumbnail_path = "/changed"  # type: ignore[misc]

    def test_thumbnail_path_is_string(self, tmp_path: Path) -> None:
        meta = _make_metadata()
        td = ThumbnailData(metadata=meta, thumbnail_path="/tmp/test.jpg")
        assert isinstance(td.thumbnail_path, str)

    def test_metadata_field_holds_original_object(self) -> None:
        meta = _make_metadata()
        td = ThumbnailData(metadata=meta, thumbnail_path="/tmp/test.jpg")
        assert td.metadata is meta


# ---------------------------------------------------------------------------
# validate_image — UnidentifiedImageError branch (coverage completeness)
# ---------------------------------------------------------------------------


class TestValidateImageUnidentifiedError:
    def test_unidentified_image_error_raises_image_validation_error(
        self, tmp_path: Path
    ) -> None:
        """
        If PIL raises UnidentifiedImageError (vs OSError/SyntaxError),
        validate_image must still raise ImageValidationError.
        """
        from unittest.mock import patch as _patch
        from PIL import UnidentifiedImageError

        path = tmp_path / "thumb.jpg"
        path.write_bytes(_make_valid_jpeg_bytes())

        with _patch("thumbnail_downloader.Image.open") as mock_open:
            mock_open.side_effect = UnidentifiedImageError("cannot identify")
            with pytest.raises(ImageValidationError, match="cannot identify"):
                validate_image(path)


class TestValidateImageOsErrorDuringLoad:
    def test_os_error_during_load_raises_image_validation_error(
        self, tmp_path: Path
    ) -> None:
        """
        If an OSError is raised during img.load() (e.g. truncated JPEG that
        Pillow can identify but not fully decode), validate_image must raise
        ImageValidationError with the 'corrupted or truncated' message.
        """
        path = tmp_path / "thumb.jpg"
        path.write_bytes(_make_valid_jpeg_bytes())

        mock_img = MagicMock()
        mock_img.load.side_effect = OSError("IO error during decode")
        mock_img.format = "JPEG"

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_img)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("thumbnail_downloader.Image.open", return_value=mock_ctx):
            with pytest.raises(ImageValidationError, match="corrupted or truncated"):
                validate_image(path)
