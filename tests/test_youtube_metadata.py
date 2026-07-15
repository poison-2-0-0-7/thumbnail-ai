"""
test_youtube_metadata.py
=========================

Pytest suite for Module 2 (YouTube Metadata Extractor).

All YouTube API calls (yt-dlp, youtube-transcript-api) are mocked so
that the tests run fully offline, deterministically, and quickly.
File-system operations use pytest's ``tmp_path`` fixture to guarantee
complete test isolation.

Coverage targets:
    - :func:`extract_video_id`           URL parsing (valid / invalid)
    - :func:`extract_metadata`           yt-dlp integration (success,
                                         private, deleted, age-gated,
                                         transient failures, retries)
    - :func:`extract_transcript`         transcript priority logic,
                                         disabled captions, network errors
    - :func:`save_metadata`             JSON persistence, atomic write
    - :func:`load_cached_metadata`      cache hit / miss / corruption
    - :func:`process_video`             full pipeline with and without cache
    - :class:`VideoMetadata`            Pydantic validation edge cases
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow importing from modules/ without installing the package
# ---------------------------------------------------------------------------

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from models import VideoMetadata  # noqa: E402
from youtube_metadata import (  # noqa: E402
    CacheError,
    Creator,
    InvalidURLError,
    MetadataExtractionError,
    TranscriptExtractionError,
    VideoUnavailableError,
    YouTubeMetadataError,
    _before_sleep_log,
    _classify_video_error,
    _error_metadata,
    _fetch_yt_dlp_info,
    _format_transcript_entries,
    _get_best_transcript,
    _is_permanent_ydl_error,
    _parse_yt_dlp_info,
    extract_metadata,
    extract_transcript,
    extract_video_id,
    load_cached_metadata,
    process_video,
    save_metadata,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

VALID_VIDEO_ID = "dQw4w9WgXcQ"
VALID_URL_FULL = f"https://www.youtube.com/watch?v={VALID_VIDEO_ID}"
VALID_URL_SHORT = f"https://youtu.be/{VALID_VIDEO_ID}"
VALID_URL_NOWWW = f"https://youtube.com/watch?v={VALID_VIDEO_ID}"


def _make_ydl_info(**overrides: Any) -> dict:
    """Return a minimal but realistic yt-dlp info dict."""
    base = {
        "id": VALID_VIDEO_ID,
        "title": "Rick Astley - Never Gonna Give You Up",
        "description": "The official video for 'Never Gonna Give You Up'",
        "uploader": "Rick Astley",
        "uploader_id": "@RickAstleyYT",
        "channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw",
        "upload_date": "20091025",
        "duration": 213,
        "view_count": 1_400_000_000,
        "like_count": 15_000_000,
        "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
        "categories": ["Music"],
        "tags": ["rick", "astley"],
        "language": "en",
    }
    base.update(overrides)
    return base


def _make_creator(url: str = VALID_URL_FULL) -> Creator:
    """Return a minimal Creator for testing."""
    return Creator(id="001", email="test@example.com", video_url=url)


def _make_valid_metadata(**overrides: Any) -> VideoMetadata:
    """Build a complete VideoMetadata suitable for cache tests."""
    base: dict[str, Any] = dict(
        video_id=VALID_VIDEO_ID,
        title="Rick Astley - Never Gonna Give You Up",
        description="Classic track",
        uploader="Rick Astley",
        uploader_id="@RickAstleyYT",
        channel_id="UCuAXFkgsw1L7xaCfnd5JJOw",
        upload_date="2009-10-25",
        duration=213,
        view_count=1_400_000_000,
        like_count=15_000_000,
        thumbnail_url="https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
        categories=["Music"],
        tags=["rick", "astley"],
        transcript="Never gonna give you up",
        language="en",
        status="success",
    )
    base.update(overrides)
    return VideoMetadata(**base)


def _make_transcript_snippet(text: str, start: float = 0.0) -> MagicMock:
    """Return a mock FetchedTranscriptSnippet with a .text attribute."""
    snippet = MagicMock()
    snippet.text = text
    snippet.start = start
    snippet.duration = 1.5
    return snippet


# ---------------------------------------------------------------------------
# extract_video_id — URL parsing
# ---------------------------------------------------------------------------


class TestExtractVideoId:
    """Tests for :func:`extract_video_id`."""

    @pytest.mark.parametrize(
        "url",
        [
            VALID_URL_FULL,
            VALID_URL_SHORT,
            VALID_URL_NOWWW,
        ],
    )
    def test_accepts_all_supported_url_formats(self, url: str) -> None:
        assert extract_video_id(url) == VALID_VIDEO_ID

    def test_url_with_extra_query_params_is_accepted(self) -> None:
        url = f"{VALID_URL_FULL}&list=PLSomething&index=1"
        assert extract_video_id(url) == VALID_VIDEO_ID

    def test_short_url_with_query_string_is_accepted(self) -> None:
        url = f"https://youtu.be/{VALID_VIDEO_ID}?t=42"
        assert extract_video_id(url) == VALID_VIDEO_ID

    @pytest.mark.parametrize(
        "bad_url",
        [
            "",
            "   ",
            "https://vimeo.com/123456",
            "https://youtube.com/channel/UCxxx",
            "https://youtube.com/watch?v=short",          # < 11 chars
            "ftp://youtu.be/dQw4w9WgXcQ",
            "youtube.com/watch?v=dQw4w9WgXcQ",            # missing schema
            "https://www.youtube.com/watch?v=TOOLONGIDXXX", # 12 chars — > 11
        ],
    )
    def test_invalid_urls_raise_invalid_url_error(self, bad_url: str) -> None:
        with pytest.raises(InvalidURLError):
            extract_video_id(bad_url)

    def test_returns_exact_11_char_id(self) -> None:
        result = extract_video_id(VALID_URL_FULL)
        assert len(result) == 11

    def test_strips_leading_trailing_whitespace(self) -> None:
        url = f"  {VALID_URL_FULL}  "
        assert extract_video_id(url) == VALID_VIDEO_ID


# ---------------------------------------------------------------------------
# _is_permanent_ydl_error
# ---------------------------------------------------------------------------


class TestIsPermanentYdlError:
    def test_detects_private_video(self) -> None:
        assert _is_permanent_ydl_error(Exception("ERROR: Private video"))

    def test_detects_video_unavailable(self) -> None:
        assert _is_permanent_ydl_error(Exception("ERROR: Video unavailable"))

    def test_detects_removed_video(self) -> None:
        assert _is_permanent_ydl_error(Exception("This video has been removed"))

    def test_detects_age_restriction(self) -> None:
        assert _is_permanent_ydl_error(Exception("age-restricted"))

    def test_transient_network_error_is_not_permanent(self) -> None:
        assert not _is_permanent_ydl_error(Exception("Connection timed out"))

    def test_empty_error_is_not_permanent(self) -> None:
        assert not _is_permanent_ydl_error(Exception(""))


# ---------------------------------------------------------------------------
# _classify_video_error
# ---------------------------------------------------------------------------


class TestClassifyVideoError:
    def test_private(self) -> None:
        assert _classify_video_error("ERROR: Private video") == "Private video"

    def test_age_restricted(self) -> None:
        assert "Age-restricted" in _classify_video_error("age-restricted video")

    def test_members_only(self) -> None:
        assert "Members-only" in _classify_video_error("members-only content")

    def test_generic_fallback(self) -> None:
        assert _classify_video_error("something else entirely") == "Video unavailable"


# ---------------------------------------------------------------------------
# _parse_yt_dlp_info
# ---------------------------------------------------------------------------


class TestParseYtDlpInfo:
    def test_parses_all_standard_fields(self) -> None:
        info = _make_ydl_info()
        meta = _parse_yt_dlp_info(info, VALID_URL_FULL)

        assert meta.video_id == VALID_VIDEO_ID
        assert meta.title == info["title"]
        assert meta.uploader == "Rick Astley"
        assert meta.channel_id == "UCuAXFkgsw1L7xaCfnd5JJOw"
        assert meta.upload_date == "2009-10-25"  # converted from YYYYMMDD
        assert meta.duration == 213
        assert meta.view_count == 1_400_000_000
        assert meta.like_count == 15_000_000
        assert meta.thumbnail_url == info["thumbnail"]
        assert meta.categories == ["Music"]
        assert meta.tags == ["rick", "astley"]
        assert meta.language == "en"
        assert meta.transcript is None
        assert meta.status == "success"

    def test_missing_video_id_raises(self) -> None:
        info = _make_ydl_info(id=None)
        with pytest.raises(MetadataExtractionError):
            _parse_yt_dlp_info(info, VALID_URL_FULL)

    def test_missing_title_raises(self) -> None:
        info = _make_ydl_info(title=None)
        with pytest.raises(MetadataExtractionError):
            _parse_yt_dlp_info(info, VALID_URL_FULL)

    def test_missing_uploader_raises(self) -> None:
        info = _make_ydl_info(uploader=None, channel=None)
        with pytest.raises(MetadataExtractionError):
            _parse_yt_dlp_info(info, VALID_URL_FULL)

    def test_none_like_count_is_accepted(self) -> None:
        info = _make_ydl_info(like_count=None)
        meta = _parse_yt_dlp_info(info, VALID_URL_FULL)
        assert meta.like_count is None

    def test_none_categories_becomes_empty_list(self) -> None:
        info = _make_ydl_info(categories=None)
        meta = _parse_yt_dlp_info(info, VALID_URL_FULL)
        assert meta.categories == []

    def test_none_tags_becomes_empty_list(self) -> None:
        info = _make_ydl_info(tags=None)
        meta = _parse_yt_dlp_info(info, VALID_URL_FULL)
        assert meta.tags == []

    def test_channel_fallback_to_uploader_id(self) -> None:
        info = _make_ydl_info(channel_id=None)
        meta = _parse_yt_dlp_info(info, VALID_URL_FULL)
        # Falls back to uploader_id
        assert meta.channel_id == info["uploader_id"]


# ---------------------------------------------------------------------------
# _fetch_yt_dlp_info — mocked yt-dlp calls
# ---------------------------------------------------------------------------


class TestFetchYtDlpInfo:
    @patch("youtube_metadata.yt_dlp.YoutubeDL")
    def test_success_returns_info_dict(self, mock_ydl: MagicMock) -> None:
        info = _make_ydl_info()
        mock_ydl.return_value.__enter__.return_value.extract_info.return_value = info

        result = _fetch_yt_dlp_info(VALID_URL_FULL)
        assert result["id"] == VALID_VIDEO_ID

    @patch("youtube_metadata.yt_dlp.YoutubeDL")
    def test_private_video_raises_video_unavailable(self, mock_ydl: MagicMock) -> None:
        import yt_dlp as ydl_pkg

        mock_ydl.return_value.__enter__.return_value.extract_info.side_effect = (
            ydl_pkg.utils.DownloadError("ERROR: Private video. Sign in if you've been granted access to this video.")
        )

        with pytest.raises(VideoUnavailableError) as exc_info:
            _fetch_yt_dlp_info(VALID_URL_FULL)

        assert "Private video" in exc_info.value.reason

    @patch("youtube_metadata.yt_dlp.YoutubeDL")
    def test_deleted_video_raises_video_unavailable(self, mock_ydl: MagicMock) -> None:
        import yt_dlp as ydl_pkg

        mock_ydl.return_value.__enter__.return_value.extract_info.side_effect = (
            ydl_pkg.utils.DownloadError("ERROR: Video unavailable")
        )

        with pytest.raises(VideoUnavailableError):
            _fetch_yt_dlp_info(VALID_URL_FULL)

    @patch("youtube_metadata.yt_dlp.YoutubeDL")
    def test_age_restricted_video_raises_video_unavailable(self, mock_ydl: MagicMock) -> None:
        import yt_dlp as ydl_pkg

        mock_ydl.return_value.__enter__.return_value.extract_info.side_effect = (
            ydl_pkg.utils.DownloadError("ERROR: Sign in to confirm your age")
        )

        with pytest.raises(VideoUnavailableError) as exc_info:
            _fetch_yt_dlp_info(VALID_URL_FULL)

        assert "Age-restricted" in exc_info.value.reason

    @patch("youtube_metadata.yt_dlp.YoutubeDL")
    def test_transient_error_raises_metadata_extraction_error(self, mock_ydl: MagicMock) -> None:
        import yt_dlp as ydl_pkg

        mock_ydl.return_value.__enter__.return_value.extract_info.side_effect = (
            ydl_pkg.utils.DownloadError("Connection reset by peer")
        )

        with pytest.raises(MetadataExtractionError):
            _fetch_yt_dlp_info.retry.statistics  # access stats
            _fetch_yt_dlp_info.__wrapped__(VALID_URL_FULL)  # bypass retry

    @patch("youtube_metadata.yt_dlp.YoutubeDL")
    def test_empty_info_raises_metadata_extraction_error(self, mock_ydl: MagicMock) -> None:
        mock_ydl.return_value.__enter__.return_value.extract_info.return_value = None

        with pytest.raises(MetadataExtractionError):
            _fetch_yt_dlp_info.__wrapped__(VALID_URL_FULL)


# ---------------------------------------------------------------------------
# extract_metadata (public, wraps _fetch_yt_dlp_info)
# ---------------------------------------------------------------------------


class TestExtractMetadata:
    @patch("youtube_metadata._fetch_yt_dlp_info")
    def test_success_returns_video_metadata(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = _make_ydl_info()

        meta = extract_metadata(VALID_URL_FULL)
        assert isinstance(meta, VideoMetadata)
        assert meta.video_id == VALID_VIDEO_ID
        assert meta.status == "success"

    def test_invalid_url_raises_before_calling_yt_dlp(self) -> None:
        with pytest.raises(InvalidURLError):
            extract_metadata("https://not-youtube.com/watch?v=xxx")

    @patch("youtube_metadata._fetch_yt_dlp_info")
    def test_propagates_video_unavailable_error(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = VideoUnavailableError(
            reason="Private video", video_url=VALID_URL_FULL
        )
        with pytest.raises(VideoUnavailableError):
            extract_metadata(VALID_URL_FULL)

    @patch("youtube_metadata._fetch_yt_dlp_info")
    def test_propagates_metadata_extraction_error(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = MetadataExtractionError("Network timeout")
        with pytest.raises(MetadataExtractionError):
            extract_metadata(VALID_URL_FULL)


# ---------------------------------------------------------------------------
# _format_transcript_entries
# ---------------------------------------------------------------------------


class TestFormatTranscriptEntries:
    def test_formats_dataclass_entries(self) -> None:
        entries = [
            _make_transcript_snippet("Hello"),
            _make_transcript_snippet("World"),
        ]
        assert _format_transcript_entries(entries) == "Hello World"

    def test_formats_dict_entries(self) -> None:
        entries = [
            {"text": "Hello", "start": 0.0, "duration": 1.0},
            {"text": "World", "start": 1.0, "duration": 1.0},
        ]
        assert _format_transcript_entries(entries) == "Hello World"

    def test_skips_empty_text(self) -> None:
        entries = [
            _make_transcript_snippet("Hello"),
            _make_transcript_snippet(""),
            _make_transcript_snippet("World"),
        ]
        assert _format_transcript_entries(entries) == "Hello World"

    def test_empty_entries_returns_empty_string(self) -> None:
        assert _format_transcript_entries([]) == ""


# ---------------------------------------------------------------------------
# _get_best_transcript — priority logic
# ---------------------------------------------------------------------------


class TestGetBestTranscript:
    def _make_transcript_list(
        self,
        manual: dict | None = None,
        generated: dict | None = None,
    ) -> MagicMock:
        """Build a mock TranscriptList with the given caption tracks."""
        tl = MagicMock()
        tl._manually_created_transcripts = manual or {}
        tl._generated_transcripts = generated or {}
        return tl

    def test_prefers_manual_english_over_auto(self) -> None:
        manual_en = MagicMock()
        manual_en.is_generated = False
        manual_en.fetch.return_value = [_make_transcript_snippet("Manual EN")]

        auto_en = MagicMock()
        auto_en.is_generated = True
        auto_en.fetch.return_value = [_make_transcript_snippet("Auto EN")]

        tl = self._make_transcript_list(
            manual={"en": manual_en},
            generated={"en": auto_en},
        )
        tl.find_manually_created_transcript.return_value = manual_en
        tl.find_generated_transcript.return_value = auto_en

        result = _get_best_transcript(tl)
        assert result == "Manual EN"

    def test_falls_back_to_auto_english_when_no_manual(self) -> None:
        auto_en = MagicMock()
        auto_en.fetch.return_value = [_make_transcript_snippet("Auto EN")]

        from youtube_transcript_api import NoTranscriptFound

        tl = self._make_transcript_list(generated={"en": auto_en})
        tl.find_manually_created_transcript.side_effect = NoTranscriptFound(
            "x", ["en"], {}
        )
        tl.find_generated_transcript.return_value = auto_en

        result = _get_best_transcript(tl)
        assert result == "Auto EN"

    def test_falls_back_to_any_manual_when_no_english(self) -> None:
        manual_de = MagicMock()
        manual_de.language_code = "de"
        manual_de.fetch.return_value = [_make_transcript_snippet("Manual DE")]

        from youtube_transcript_api import NoTranscriptFound

        tl = self._make_transcript_list(manual={"de": manual_de})
        tl.find_manually_created_transcript.side_effect = NoTranscriptFound(
            "x", ["en"], {}
        )
        tl.find_generated_transcript.side_effect = NoTranscriptFound("x", ["en"], {})

        result = _get_best_transcript(tl)
        assert result == "Manual DE"

    def test_returns_none_when_no_transcripts_at_all(self) -> None:
        from youtube_transcript_api import NoTranscriptFound

        tl = self._make_transcript_list()
        tl.find_manually_created_transcript.side_effect = NoTranscriptFound(
            "x", ["en"], {}
        )
        tl.find_generated_transcript.side_effect = NoTranscriptFound("x", ["en"], {})

        result = _get_best_transcript(tl)
        assert result is None


# ---------------------------------------------------------------------------
# extract_transcript (public)
# ---------------------------------------------------------------------------


class TestExtractTranscript:
    @patch("youtube_metadata.YouTubeTranscriptApi")
    def test_returns_text_when_transcript_available(
        self, mock_api_cls: MagicMock
    ) -> None:
        api_instance = MagicMock()
        mock_api_cls.return_value = api_instance

        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = [
            _make_transcript_snippet("Hello"),
            _make_transcript_snippet("World"),
        ]

        tl = MagicMock()
        tl.find_manually_created_transcript.return_value = mock_transcript
        api_instance.list.return_value = tl

        result = extract_transcript(VALID_VIDEO_ID)
        assert result == "Hello World"

    @patch("youtube_metadata.YouTubeTranscriptApi")
    def test_returns_none_when_transcripts_disabled(
        self, mock_api_cls: MagicMock
    ) -> None:
        from youtube_transcript_api import TranscriptsDisabled

        api_instance = MagicMock()
        mock_api_cls.return_value = api_instance
        api_instance.list.side_effect = TranscriptsDisabled(VALID_VIDEO_ID)

        result = extract_transcript(VALID_VIDEO_ID)
        assert result is None

    @patch("youtube_metadata.YouTubeTranscriptApi")
    def test_returns_none_when_video_unavailable_for_transcripts(
        self, mock_api_cls: MagicMock
    ) -> None:
        from youtube_transcript_api import VideoUnavailable

        api_instance = MagicMock()
        mock_api_cls.return_value = api_instance
        api_instance.list.side_effect = VideoUnavailable(VALID_VIDEO_ID)

        result = extract_transcript(VALID_VIDEO_ID)
        assert result is None

    @patch("youtube_metadata.YouTubeTranscriptApi")
    def test_returns_none_on_unexpected_exception(
        self, mock_api_cls: MagicMock
    ) -> None:
        api_instance = MagicMock()
        mock_api_cls.return_value = api_instance
        api_instance.list.side_effect = RuntimeError("Network error")

        result = extract_transcript(VALID_VIDEO_ID)
        assert result is None

    @patch("youtube_metadata.YouTubeTranscriptApi")
    def test_returns_none_when_get_best_returns_none(
        self, mock_api_cls: MagicMock
    ) -> None:
        api_instance = MagicMock()
        mock_api_cls.return_value = api_instance
        tl = MagicMock()
        tl._manually_created_transcripts = {}
        tl._generated_transcripts = {}

        from youtube_transcript_api import NoTranscriptFound

        tl.find_manually_created_transcript.side_effect = NoTranscriptFound(
            "x", ["en"], {}
        )
        tl.find_generated_transcript.side_effect = NoTranscriptFound("x", ["en"], {})
        api_instance.list.return_value = tl

        result = extract_transcript(VALID_VIDEO_ID)
        assert result is None


# ---------------------------------------------------------------------------
# save_metadata / load_cached_metadata
# ---------------------------------------------------------------------------


class TestCacheReadWrite:
    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        meta = _make_valid_metadata()
        save_metadata(meta, cache_dir=tmp_path)

        expected = tmp_path / f"{VALID_VIDEO_ID}.json"
        assert expected.exists()

    def test_saved_json_is_valid(self, tmp_path: Path) -> None:
        meta = _make_valid_metadata()
        save_metadata(meta, cache_dir=tmp_path)

        raw = (tmp_path / f"{VALID_VIDEO_ID}.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        assert data["video_id"] == VALID_VIDEO_ID

    def test_load_returns_none_on_cache_miss(self, tmp_path: Path) -> None:
        result = load_cached_metadata("nonexistent11", cache_dir=tmp_path)
        assert result is None

    def test_roundtrip_preserves_all_fields(self, tmp_path: Path) -> None:
        meta = _make_valid_metadata()
        save_metadata(meta, cache_dir=tmp_path)
        loaded = load_cached_metadata(VALID_VIDEO_ID, cache_dir=tmp_path)

        assert loaded is not None
        assert loaded.video_id == meta.video_id
        assert loaded.title == meta.title
        assert loaded.uploader == meta.uploader
        assert loaded.transcript == meta.transcript
        assert loaded.upload_date == meta.upload_date

    def test_cache_dir_is_created_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        meta = _make_valid_metadata()
        save_metadata(meta, cache_dir=nested)
        assert nested.is_dir()

    def test_load_raises_cache_error_on_corrupted_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / f"{VALID_VIDEO_ID}.json"
        bad_file.write_text("NOT VALID JSON {{{{", encoding="utf-8")

        with pytest.raises(CacheError):
            load_cached_metadata(VALID_VIDEO_ID, cache_dir=tmp_path)

    def test_load_raises_cache_error_on_schema_mismatch(self, tmp_path: Path) -> None:
        bad_file = tmp_path / f"{VALID_VIDEO_ID}.json"
        # Valid JSON but missing required fields
        bad_file.write_text(json.dumps({"video_id": VALID_VIDEO_ID}), encoding="utf-8")

        with pytest.raises(CacheError):
            load_cached_metadata(VALID_VIDEO_ID, cache_dir=tmp_path)

    def test_save_overwrites_existing_cache(self, tmp_path: Path) -> None:
        meta_v1 = _make_valid_metadata(title="Old Title")
        save_metadata(meta_v1, cache_dir=tmp_path)

        meta_v2 = _make_valid_metadata(title="New Title")
        save_metadata(meta_v2, cache_dir=tmp_path)

        loaded = load_cached_metadata(VALID_VIDEO_ID, cache_dir=tmp_path)
        assert loaded is not None
        assert loaded.title == "New Title"


# ---------------------------------------------------------------------------
# process_video — full pipeline
# ---------------------------------------------------------------------------


class TestProcessVideo:
    @patch("youtube_metadata.extract_transcript")
    @patch("youtube_metadata._fetch_yt_dlp_info")
    def test_success_full_pipeline(
        self,
        mock_fetch: MagicMock,
        mock_transcript: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_fetch.return_value = _make_ydl_info()
        mock_transcript.return_value = "Never gonna give you up"

        result = process_video(_make_creator(), cache_dir=tmp_path)

        assert result.status == "success"
        assert result.video_id == VALID_VIDEO_ID
        assert result.transcript == "Never gonna give you up"

    @patch("youtube_metadata.extract_transcript")
    @patch("youtube_metadata._fetch_yt_dlp_info")
    def test_result_is_cached_after_first_fetch(
        self,
        mock_fetch: MagicMock,
        mock_transcript: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_fetch.return_value = _make_ydl_info()
        mock_transcript.return_value = "transcript text"

        process_video(_make_creator(), cache_dir=tmp_path)
        process_video(_make_creator(), cache_dir=tmp_path)

        # yt-dlp should only be called once; second call hits cache
        assert mock_fetch.call_count == 1

    @patch("youtube_metadata.extract_transcript")
    @patch("youtube_metadata._fetch_yt_dlp_info")
    def test_cache_hit_skips_network_calls(
        self,
        mock_fetch: MagicMock,
        mock_transcript: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Pre-populate cache
        save_metadata(_make_valid_metadata(), cache_dir=tmp_path)

        result = process_video(_make_creator(), cache_dir=tmp_path)

        assert result.status == "success"
        mock_fetch.assert_not_called()
        mock_transcript.assert_not_called()

    def test_invalid_url_returns_error_status(self, tmp_path: Path) -> None:
        creator = _make_creator(url="https://not-youtube.com")
        result = process_video(creator, cache_dir=tmp_path)

        assert result.status == "error"
        assert result.error_message is not None

    @patch("youtube_metadata._fetch_yt_dlp_info")
    def test_private_video_returns_error_status(
        self, mock_fetch: MagicMock, tmp_path: Path
    ) -> None:
        mock_fetch.side_effect = VideoUnavailableError(
            reason="Private video", video_url=VALID_URL_FULL
        )
        result = process_video(_make_creator(), cache_dir=tmp_path)

        assert result.status == "error"
        assert "Private video" in (result.error_message or "")

    @patch("youtube_metadata._fetch_yt_dlp_info")
    def test_deleted_video_returns_error_status(
        self, mock_fetch: MagicMock, tmp_path: Path
    ) -> None:
        mock_fetch.side_effect = VideoUnavailableError(
            reason="Video unavailable", video_url=VALID_URL_FULL
        )
        result = process_video(_make_creator(), cache_dir=tmp_path)

        assert result.status == "error"

    @patch("youtube_metadata.extract_transcript")
    @patch("youtube_metadata._fetch_yt_dlp_info")
    def test_unavailable_transcript_still_succeeds(
        self,
        mock_fetch: MagicMock,
        mock_transcript: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_fetch.return_value = _make_ydl_info()
        mock_transcript.return_value = None  # No transcript available

        result = process_video(_make_creator(), cache_dir=tmp_path)

        assert result.status == "success"
        assert result.transcript is None

    @patch("youtube_metadata._fetch_yt_dlp_info")
    def test_metadata_extraction_failure_returns_error(
        self, mock_fetch: MagicMock, tmp_path: Path
    ) -> None:
        mock_fetch.side_effect = MetadataExtractionError("yt-dlp timed out")
        result = process_video(_make_creator(), cache_dir=tmp_path)

        assert result.status == "error"
        assert "yt-dlp timed out" in (result.error_message or "")

    @patch("youtube_metadata.extract_transcript")
    @patch("youtube_metadata._fetch_yt_dlp_info")
    def test_cache_write_failure_does_not_crash_pipeline(
        self,
        mock_fetch: MagicMock,
        mock_transcript: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_fetch.return_value = _make_ydl_info()
        mock_transcript.return_value = "text"

        with patch("youtube_metadata.save_metadata", side_effect=CacheError("disk full")):
            result = process_video(_make_creator(), cache_dir=tmp_path)

        # Pipeline should still succeed even if caching fails
        assert result.status == "success"


# ---------------------------------------------------------------------------
# VideoMetadata Pydantic model edge cases
# ---------------------------------------------------------------------------


class TestVideoMetadata:
    def test_upload_date_is_converted_from_yyyymmdd(self) -> None:
        m = _make_valid_metadata(upload_date="20091025")
        assert m.upload_date == "2009-10-25"

    def test_upload_date_already_iso_is_unchanged(self) -> None:
        m = _make_valid_metadata(upload_date="2009-10-25")
        assert m.upload_date == "2009-10-25"

    def test_none_upload_date_is_accepted(self) -> None:
        m = _make_valid_metadata(upload_date=None)
        assert m.upload_date is None

    def test_model_is_frozen(self) -> None:
        m = _make_valid_metadata()
        with pytest.raises(Exception):
            m.title = "Mutated"  # type: ignore[misc]

    def test_empty_video_id_raises_validation_error(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoMetadata(
                video_id="",
                title="Test",
                uploader="x",
                uploader_id="y",
                channel_id="z",
            )

    def test_none_categories_coerced_to_empty_list(self) -> None:
        m = VideoMetadata(
            video_id=VALID_VIDEO_ID,
            title="T",
            uploader="U",
            uploader_id="uid",
            channel_id="cid",
            categories=None,  # type: ignore[arg-type]
        )
        assert m.categories == []

    def test_error_status_metadata_has_message(self) -> None:
        m = _error_metadata(
            video_id=VALID_VIDEO_ID,
            reason="Private video",
            url=VALID_URL_FULL,
        )
        assert m.status == "error"
        assert "Private video" in (m.error_message or "")

    def test_json_serialization_roundtrip(self) -> None:
        original = _make_valid_metadata()
        json_str = original.model_dump_json()
        restored = VideoMetadata.model_validate_json(json_str)
        assert restored == original


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_invalid_url_is_youtube_metadata_error(self) -> None:
        assert issubclass(InvalidURLError, YouTubeMetadataError)

    def test_video_unavailable_is_youtube_metadata_error(self) -> None:
        assert issubclass(VideoUnavailableError, YouTubeMetadataError)

    def test_metadata_extraction_error_is_youtube_metadata_error(self) -> None:
        assert issubclass(MetadataExtractionError, YouTubeMetadataError)

    def test_transcript_extraction_error_is_youtube_metadata_error(self) -> None:
        assert issubclass(TranscriptExtractionError, YouTubeMetadataError)

    def test_cache_error_is_youtube_metadata_error(self) -> None:
        assert issubclass(CacheError, YouTubeMetadataError)

    def test_video_unavailable_stores_reason_and_url(self) -> None:
        exc = VideoUnavailableError(reason="Private video", video_url=VALID_URL_FULL)
        assert exc.reason == "Private video"
        assert exc.video_url == VALID_URL_FULL
        assert "Private video" in str(exc)


# ---------------------------------------------------------------------------
# _before_sleep_log — Tenacity retry logger (uncovered path)
# ---------------------------------------------------------------------------


class TestBeforeSleepLog:
    def test_logs_without_raising_when_outcome_is_none(self) -> None:
        state = MagicMock()
        state.fn = None
        state.attempt_number = 1
        state.outcome = None
        # Must not raise
        _before_sleep_log(state)

    def test_logs_without_raising_with_exception(self) -> None:
        state = MagicMock()
        state.fn.__name__ = "_fetch_yt_dlp_info"
        state.attempt_number = 2
        state.outcome.exception.return_value = RuntimeError("boom")
        _before_sleep_log(state)


# ---------------------------------------------------------------------------
# _get_best_transcript — auto-generated fallback (path 4)
# ---------------------------------------------------------------------------


class TestGetBestTranscriptAutoFallback:
    def test_falls_back_to_any_generated_when_nothing_else(self) -> None:
        from youtube_transcript_api import NoTranscriptFound

        auto_de = MagicMock()
        auto_de.language_code = "de"
        auto_de.is_generated = True
        auto_de.fetch.return_value = [_make_transcript_snippet("Auto DE")]

        tl = MagicMock()
        tl._manually_created_transcripts = {}
        tl._generated_transcripts = {"de": auto_de}
        tl.find_manually_created_transcript.side_effect = NoTranscriptFound(
            "x", ["en"], {}
        )
        tl.find_generated_transcript.side_effect = NoTranscriptFound("x", ["en"], {})

        result = _get_best_transcript(tl)
        assert result == "Auto DE"


# ---------------------------------------------------------------------------
# extract_transcript — CouldNotRetrieveTranscript path
# ---------------------------------------------------------------------------


class TestExtractTranscriptCouldNotRetrieve:
    @patch("youtube_metadata.YouTubeTranscriptApi")
    def test_returns_none_on_could_not_retrieve(self, mock_api_cls: MagicMock) -> None:
        from youtube_transcript_api import CouldNotRetrieveTranscript

        api_instance = MagicMock()
        mock_api_cls.return_value = api_instance
        api_instance.list.side_effect = CouldNotRetrieveTranscript(VALID_VIDEO_ID)

        result = extract_transcript(VALID_VIDEO_ID)
        assert result is None


# ---------------------------------------------------------------------------
# save_metadata — OS error path
# ---------------------------------------------------------------------------


class TestSaveMetadataOsError:
    def test_raises_cache_error_on_os_error(self, tmp_path: Path) -> None:
        meta = _make_valid_metadata()

        with patch("youtube_metadata.Path.replace", side_effect=OSError("disk full")):
            with pytest.raises(CacheError):
                save_metadata(meta, cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# process_video — corrupted cache triggers re-fetch
# ---------------------------------------------------------------------------


class TestProcessVideoCacheRecovery:
    @patch("youtube_metadata.extract_transcript")
    @patch("youtube_metadata._fetch_yt_dlp_info")
    def test_corrupted_cache_triggers_refetch(
        self,
        mock_fetch: MagicMock,
        mock_transcript: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Write a corrupt cache file
        bad_file = tmp_path / f"{VALID_VIDEO_ID}.json"
        bad_file.write_text("GARBAGE{{", encoding="utf-8")

        mock_fetch.return_value = _make_ydl_info()
        mock_transcript.return_value = "text"

        result = process_video(_make_creator(), cache_dir=tmp_path)

        # Should have recovered via fresh fetch
        assert result.status == "success"
        mock_fetch.assert_called_once()
