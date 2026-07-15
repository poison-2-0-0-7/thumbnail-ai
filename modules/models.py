"""
models.py
=========

Pydantic data models for Module 2 (YouTube Metadata Extractor).

All fields mirror exactly what yt-dlp and youtube-transcript-api expose,
mapped to explicit Python types so that later modules (thumbnail fetcher,
AI prompt generator, email sender) can rely on strict type contracts with
no runtime surprises.

This module has zero project-internal dependencies; it may be imported
safely by any other module in the system.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VideoStatus(str, Enum):
    """Terminal outcome of a ``process_video`` call."""

    SUCCESS = "success"
    ERROR = "error"


# ---------------------------------------------------------------------------
# VideoMetadata
# ---------------------------------------------------------------------------


class VideoMetadata(BaseModel):
    """
    Strongly-typed record holding every piece of information extracted
    from a single YouTube video.

    All fields that YouTube may legitimately omit (like counts, language,
    transcript) are typed as ``Optional`` with a ``None`` default so that
    the model is constructible even from partial data.  The ``status``
    field signals the overall outcome of the extraction pipeline.

    Attributes:
        video_id:      YouTube video identifier (11-char alphanumeric).
        title:         Video title as shown on YouTube.
        description:   Full video description.  May be very long.
        uploader:      Human-readable channel name (e.g. "MrBeast").
        uploader_id:   Channel handle or legacy user-name (e.g. "@MrBeast").
        channel_id:    Opaque channel identifier (e.g. "UCX6OQ3DkcsbYNE6H8uQQuVA").
        upload_date:   ISO-8601 date string ``YYYY-MM-DD`` derived from
                       yt-dlp's raw ``YYYYMMDD`` field.
        duration:      Video length in whole seconds.
        view_count:    Total view count at time of extraction.
        like_count:    Like count; ``None`` when YouTube has hidden it.
        thumbnail_url: URL of the video's highest-quality static thumbnail.
        categories:    YouTube category list, e.g. ``["Education"]``.
        tags:          Creator-supplied tag list.
        transcript:    Full transcript text, whitespace-joined from all
                       caption entries.  ``None`` when unavailable.
        language:      BCP-47 language code reported by yt-dlp, e.g. ``"en"``.
        status:        ``"success"`` on clean extraction; ``"error"`` on any
                       failure.
        error_message: Human-readable reason when ``status == "error"``.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    # --- required fields (always populated on success) ---
    video_id: str
    title: str
    uploader: str
    uploader_id: str
    channel_id: str
    status: Literal["success", "error"] = "success"

    # --- optional fields ---
    description: Optional[str] = None
    upload_date: Optional[str] = None
    duration: Optional[int] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    thumbnail_url: Optional[str] = None
    categories: list[str] = []
    tags: list[str] = []
    transcript: Optional[str] = None
    language: Optional[str] = None
    error_message: Optional[str] = None

    # --- validators ---

    @field_validator("video_id")
    @classmethod
    def video_id_must_not_be_empty(cls, v: str) -> str:
        """Reject blank video IDs."""
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()

    @field_validator("upload_date")
    @classmethod
    def upload_date_format(cls, v: Optional[str]) -> Optional[str]:
        """Accept ``None``, ``YYYY-MM-DD``, or raw yt-dlp ``YYYYMMDD``."""
        if v is None:
            return v
        v = v.strip()
        # Already ISO-8601
        if len(v) == 10 and v[4] == "-" and v[7] == "-":
            return v
        # yt-dlp raw format → ISO-8601
        if len(v) == 8 and v.isdigit():
            return f"{v[:4]}-{v[4:6]}-{v[6:8]}"
        # Unknown format — pass through unchanged
        return v

    @field_validator("categories", "tags", mode="before")
    @classmethod
    def coerce_none_to_empty_list(cls, v: object) -> list:
        """Turn ``None`` (which yt-dlp sometimes emits) into ``[]``."""
        if v is None:
            return []
        return list(v)

# ---------------------------------------------------------------------------
# Downloaded Thumbnail
# ---------------------------------------------------------------------------

class ThumbnailData(BaseModel):
    """
    Output of Module 3.

    Combines the immutable VideoMetadata object with the
    local path of the downloaded thumbnail.
    """

    model_config = ConfigDict(frozen=True)

    metadata: VideoMetadata
    thumbnail_path: str