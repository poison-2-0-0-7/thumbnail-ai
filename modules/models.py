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


# ---------------------------------------------------------------------------
# Module 4 — Thumbnail Intelligence Engine
# ---------------------------------------------------------------------------


class IntelligenceStatus(str, Enum):
    """Terminal outcome of an ``analyze_thumbnail`` call."""

    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"


class BoundingBox(BaseModel):
    """
    Normalized bounding box, expressed as fractions of image width/height
    in ``[0.0, 1.0]`` so that boxes remain meaningful regardless of the
    thumbnail's actual pixel dimensions.

    Attributes:
        x_min: Left edge, as a fraction of image width.
        y_min: Top edge, as a fraction of image height.
        x_max: Right edge, as a fraction of image width.
        y_max: Bottom edge, as a fraction of image height.
    """

    model_config = ConfigDict(frozen=True)

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @field_validator("x_min", "y_min", "x_max", "y_max")
    @classmethod
    def coordinate_in_unit_range(cls, v: float) -> float:
        """Clamp coordinates into ``[0.0, 1.0]`` to absorb detector rounding."""
        return max(0.0, min(1.0, v))


class TextRegion(BaseModel):
    """
    A single OCR-detected text region.

    Attributes:
        text:       Recognized text content for this region.
        confidence: EasyOCR confidence score in ``[0.0, 1.0]``.
        bbox:       Approximate location of the text within the image.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: float
    bbox: BoundingBox


class OCRResult(BaseModel):
    """
    Structured output of Stage 2 (OCR).

    Attributes:
        visible_text:        All recognized text, joined with single spaces
                              in reading order (top-to-bottom, left-to-right).
        text_regions:         Individual detected text regions with their
                              own confidence and location.
        word_count:           Total number of words across all regions.
        text_coverage_ratio:  Fraction of total image area covered by text
                              bounding boxes, in ``[0.0, 1.0]``.
        average_confidence:   Mean confidence across all kept regions.
                              ``0.0`` when no text was detected.
        engine_available:     Whether the OCR engine ran successfully.
                              ``False`` when OCR failed and this result is
                              a safe empty default rather than a genuine
                              "no text" reading.
        duration_seconds:     Wall-clock time spent in this stage.
    """

    model_config = ConfigDict(frozen=True)

    visible_text: str = ""
    text_regions: list[TextRegion] = []
    word_count: int = 0
    text_coverage_ratio: float = 0.0
    average_confidence: float = 0.0
    engine_available: bool = True
    duration_seconds: float = 0.0


class FaceDetail(BaseModel):
    """
    Attributes of a single detected face.

    Attributes:
        bbox:            Location of the face within the image.
        detection_confidence: InsightFace detector confidence in ``[0.0, 1.0]``.
        is_largest:      Whether this is the largest face by bounding-box area.
        emotion:         Best-effort emotion label (e.g. "happy", "neutral",
                         "surprised"). ``None`` when it could not be estimated.
        emotion_confidence: Confidence for ``emotion``. ``None`` when
                         ``emotion`` is ``None``.
        smile_detected:  Whether a smile was detected. ``None`` when
                         indeterminate.
        eye_direction:   Best-effort gaze/eye-direction label (e.g.
                         "camera", "left", "right", "down"). ``None`` when
                         it could not be estimated.
        head_pose:       Best-effort head pose label (e.g. "frontal",
                         "profile", "tilted"). ``None`` when it could not
                         be estimated.
        position_label:  Coarse position of the face within the frame
                         (e.g. "left-third", "center", "right-third").
    """

    model_config = ConfigDict(frozen=True)

    bbox: BoundingBox
    detection_confidence: float
    is_largest: bool = False
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    smile_detected: Optional[bool] = None
    eye_direction: Optional[str] = None
    head_pose: Optional[str] = None
    position_label: str = "unknown"


class FaceAnalysis(BaseModel):
    """
    Structured output of Stage 3 (face analysis).

    Attributes:
        face_count:       Number of faces kept after confidence filtering.
        faces:            Per-face details, ordered largest-first.
        has_face:         Convenience flag, equivalent to ``face_count > 0``.
        engine_available: Whether the face-analysis engine ran successfully.
        duration_seconds: Wall-clock time spent in this stage.
    """

    model_config = ConfigDict(frozen=True)

    face_count: int = 0
    faces: list[FaceDetail] = []
    has_face: bool = False
    engine_available: bool = True
    duration_seconds: float = 0.0


class DetectedObject(BaseModel):
    """
    A single YOLO-detected object relevant to thumbnail analysis.

    Attributes:
        label:      Object class label (e.g. "person", "car", "phone").
        confidence: Detector confidence in ``[0.0, 1.0]``.
        bbox:       Location of the object within the image.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    confidence: float
    bbox: BoundingBox


class ColorProfile(BaseModel):
    """
    Structured output of Stage 5 (color analysis).

    Attributes:
        dominant_colors:  Dominant colors as ``#rrggbb`` hex strings,
                          ordered by prevalence (most prevalent first).
        brightness:       Mean perceptual brightness in ``[0.0, 1.0]``.
        contrast:         Normalized standard deviation of luminance in
                          ``[0.0, 1.0]``.
        saturation:       Mean HSV saturation in ``[0.0, 1.0]``.
        warm_or_cool:     Overall color temperature classification.
        harmony_score:     Heuristic color-harmony score in ``[0.0, 1.0]``;
                          higher means the dominant palette sits closer
                          together on the color wheel (more cohesive).
        duration_seconds: Wall-clock time spent in this stage.
    """

    model_config = ConfigDict(frozen=True)

    dominant_colors: list[str] = []
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    warm_or_cool: Literal["warm", "cool", "neutral"] = "neutral"
    harmony_score: float = 0.0
    duration_seconds: float = 0.0


class CompositionAnalysis(BaseModel):
    """
    Structured output of Stage 6 (composition analysis).

    Attributes:
        rule_of_thirds_score:   How closely the primary subject (largest
                                face, else largest detected object) sits
                                near a rule-of-thirds intersection, in
                                ``[0.0, 1.0]`` (higher is closer).
        subject_placement:      Coarse label for where the primary subject
                                sits (e.g. "center", "left-third",
                                "right-third", "none-detected").
        negative_space_ratio:   Fraction of the frame with no detected
                                face, object, or text, in ``[0.0, 1.0]``.
        clutter_score:          Heuristic visual-clutter score in
                                ``[0.0, 1.0]`` (higher is more cluttered),
                                derived from face/object/text density.
        visual_hierarchy_score: Heuristic score in ``[0.0, 1.0]`` for how
                                clearly a single dominant focal point
                                stands out from the rest of the frame.
        text_overlaps_subject:  Whether any OCR text bounding box
                                overlaps the primary subject's bounding box.
        balance_score:          Heuristic left/right visual-weight balance
                                in ``[0.0, 1.0]`` (higher is more balanced).
        symmetry_score:         Heuristic horizontal-symmetry score in
                                ``[0.0, 1.0]``.
        duration_seconds:       Wall-clock time spent in this stage.
    """

    model_config = ConfigDict(frozen=True)

    rule_of_thirds_score: float = 0.0
    subject_placement: str = "none-detected"
    negative_space_ratio: float = 0.0
    clutter_score: float = 0.0
    visual_hierarchy_score: float = 0.0
    text_overlaps_subject: bool = False
    balance_score: float = 0.0
    symmetry_score: float = 0.0
    duration_seconds: float = 0.0


class GeminiReasoning(BaseModel):
    """
    Structured output of the AI reasoning stage.

    Attributes:
        ctr_potential_score:         Estimated click-through-rate potential
                                     in ``[0.0, 1.0]``.
        curiosity_gap_score:         How strong a curiosity gap the
                                     thumbnail creates, in ``[0.0, 1.0]``.
        emotional_impact:            Short label/phrase for the dominant
                                     emotion the thumbnail conveys.
        visual_storytelling_notes:   Free-text notes on how the visual
                                     elements tell a story on their own.
        content_mismatch_detected:   Whether the thumbnail's implied
                                     content diverges meaningfully from the
                                     title/description/transcript.
        mismatch_explanation:        Explanation when a mismatch was
                                     detected. ``None`` otherwise.
        strengths:                   What the thumbnail does well.
        weaknesses:                  What is holding the thumbnail back.
        redesign_recommendations:    Concrete, actionable suggestions for
                                     a redesign.
        elements_to_preserve:        Specific elements that work and
                                     should survive any redesign.
        duration_seconds:            Wall-clock time spent in this stage.
    """

    model_config = ConfigDict(frozen=True)

    ctr_potential_score: float
    curiosity_gap_score: float
    emotional_impact: str
    visual_storytelling_notes: str
    content_mismatch_detected: bool
    mismatch_explanation: Optional[str] = None
    strengths: list[str] = []
    weaknesses: list[str] = []
    redesign_recommendations: list[str] = []
    elements_to_preserve: list[str] = []
    duration_seconds: float = 0.0


class ThumbnailIntelligence(BaseModel):
    """
    Output of Module 4.

    The complete structured intelligence report for a single creator's
    thumbnail: every computer-vision stage's findings, the merged
    video/transcript context those findings were reasoned over, the AI
    reasoning result, and bookkeeping about what succeeded or failed.

    Attributes:
        video_id:                YouTube video identifier this report
                                 describes.
        thumbnail_path:          Local path of the analyzed thumbnail.
        ocr:                     Stage 2 output.
        faces:                   Stage 3 output.
        objects:                 Stage 4 output.
        colors:                  Stage 5 output.
        composition:             Stage 6 output.
        reasoning:               AI reasoning output. ``None`` when the
                                 Gemini call failed.
        status:                  Overall outcome. ``"success"`` when every
                                 stage (including Gemini) completed;
                                 ``"partial"`` when at least one stage
                                 degraded to a safe default but the report
                                 is still usable; ``"error"`` when the
                                 report could not be produced at all.
        partial_failure_reasons: Human-readable reasons for each degraded
                                 stage. Empty when ``status == "success"``.
        error_message:           Populated when ``status == "error"``.
        total_duration_seconds:  Wall-clock time for the entire pipeline.
        analyzed_at:             ISO-8601 UTC timestamp of when this report
                                 was generated.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str
    thumbnail_path: str
    ocr: OCRResult
    faces: FaceAnalysis
    objects: list[DetectedObject] = []
    colors: ColorProfile
    composition: CompositionAnalysis
    reasoning: Optional[GeminiReasoning] = None
    status: Literal["success", "partial", "error"] = "success"
    partial_failure_reasons: list[str] = []
    error_message: Optional[str] = None
    total_duration_seconds: float = 0.0
    analyzed_at: str

    @field_validator("video_id")
    @classmethod
    def video_id_must_not_be_empty(cls, v: str) -> str:
        """Reject blank video IDs."""
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()