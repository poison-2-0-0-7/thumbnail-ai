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
from typing import Any, Literal, Optional

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator


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


# ---------------------------------------------------------------------------
# Module 5 — Redesign Specification Engine
# ---------------------------------------------------------------------------


class ColorDirection(BaseModel):
    """Deterministic target color adjustment for a redesigned thumbnail."""

    model_config = ConfigDict(frozen=True)

    target_brightness: float = 0.5
    target_contrast: float = 0.5
    target_saturation: float = 0.5
    warm_or_cool: Literal["warm", "cool", "neutral"] = "neutral"
    rationale: str = ""


class SubjectTreatment(BaseModel):
    """Deterministic placement and crop guidance for the primary subject."""

    model_config = ConfigDict(frozen=True)

    has_subject: bool = False
    target_bbox: Optional[BoundingBox] = None
    target_position_label: str = "center"
    crop_tighter: bool = False
    rationale: str = ""


class TextOverlaySpec(BaseModel):
    """Placement-only text guidance; this model never contains new copy."""

    model_config = ConfigDict(frozen=True)

    include_text: bool = False
    placement_zone: Optional[BoundingBox] = None
    avoid_zones: list[BoundingBox] = []
    rationale: str = ""


class ObjectDirective(BaseModel):
    """Deterministic action for one detected object."""

    model_config = ConfigDict(frozen=True)

    label: str
    action: Literal["include", "remove", "preserve"]
    rationale: str = ""


class LayoutDirection(BaseModel):
    """Deterministic composition targets for a redesigned thumbnail."""

    model_config = ConfigDict(frozen=True)

    target_negative_space_ratio: float = 0.0
    target_clutter_score: float = 0.0
    focal_zone: Optional[BoundingBox] = None
    rationale: str = ""


class RedesignSpecification(BaseModel):
    """Fully structured deterministic redesign guidance derived from Module 4."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    source_thumbnail_path: str
    color_direction: ColorDirection
    subject_treatment: SubjectTreatment
    text_overlay: TextOverlaySpec
    layout_direction: LayoutDirection
    object_directives: list[ObjectDirective] = []
    elements_to_preserve: list[str] = []
    overall_rationale: str = ""
    source_ctr_potential_score: float
    source_curiosity_gap_score: float
    source_content_mismatch_detected: bool
    status: Literal["success", "error"] = "success"
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    generated_at: str

    @field_validator("video_id")
    @classmethod
    def video_id_must_not_be_empty(cls, v: str) -> str:
        """Reject blank video IDs."""
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Module 6 — Prompt Compiler
# ---------------------------------------------------------------------------


class GenerationParameters(BaseModel):
    """Deterministic image-generation parameters for one prompt package."""

    model_config = ConfigDict(frozen=True)

    width: int = 1280
    height: int = 720
    aspect_ratio: str = "16:9"
    seed: int = 0
    guidance_scale: float = 7.5
    inference_steps: int = 30
    sampler: str = "deterministic"
    num_candidates: int = 1
    strategy_pack: Optional[str] = None

    @field_validator("num_candidates")
    @classmethod
    def num_candidates_must_be_at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_candidates must be at least 1")
        return v



class QualityParameters(BaseModel):
    """Deterministic output-quality targets for one prompt package."""

    model_config = ConfigDict(frozen=True)

    quality_tags: list[str] = []
    min_resolution_px: int = 1280
    upscale_requested: bool = False


class ModelSettings(BaseModel):
    """Deterministic model/style configuration for one prompt package."""

    model_config = ConfigDict(frozen=True)

    model_name: str = "thumbnail-diffusion-v1"
    style_preset: str = "photographic"
    negative_prompt_weight: float = 1.0


class PromptPackage(BaseModel):
    """Structured, deterministic Module 7 input compiled from Module 5.

    Every prompt field is assembled from fixed templates and Module 5 values;
    it never contains newly reasoned or invented creative content.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str
    positive_prompt: str
    negative_prompt: str
    subject_instructions: str
    background_instructions: str
    typography_instructions: str
    composition_instructions: str
    lighting_instructions: str
    color_instructions: str
    object_placement: list[str] = []
    rendering_constraints: list[str] = []
    safety_constraints: list[str] = []
    generation_parameters: GenerationParameters
    quality_parameters: QualityParameters
    model_settings: ModelSettings
    status: Literal["success", "error"] = "success"
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    generated_at: str

    @field_validator("video_id")
    @classmethod
    def video_id_must_not_be_empty(cls, v: str) -> str:
        """Reject blank video IDs."""
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Module 5.5 — Thumbnail Copywriter & Layout Planner
# ---------------------------------------------------------------------------


class HeadlineCandidate(BaseModel):
    """Authored headline candidate scored deterministically across multiple metrics."""

    model_config = ConfigDict(frozen=True)

    text: str
    template_id: str
    curiosity_score: float
    emotional_impact_score: float
    readability_score: float
    ctr_potential_score: float
    character_count: int
    mobile_readability_score: float
    brand_consistency_score: float
    composite_score: float


class ObjectLayoutDirective(BaseModel):
    """Layout directive for one object wrapping base action with scale and rank."""

    model_config = ConfigDict(frozen=True)

    label: str
    action: Literal["include", "remove", "preserve"]
    scale_factor: float = 1.0
    emphasis_rank: int = 1
    rationale: str = ""


class DesignBlueprint(BaseModel):
    """Top-level frozen artifact produced by Module 5.5 (Copywriter & Layout Planner)."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    headline: str
    headline_variants: list[HeadlineCandidate] = Field(default_factory=list)
    headline_score: float
    hook_type: Literal[
        "curiosity", "shock", "controversy", "benefit", "authority", "fomo", "question", "how_to"
    ]
    emotion: str
    face_strategy: Literal["smile", "neutral", "shock", "exaggerate", "remove", "preserve"]
    object_strategy: list[ObjectLayoutDirective] = Field(default_factory=list)
    background_strategy: Literal["keep", "replace", "blur", "darken", "simplify"]
    text_position: TextPlacement
    subject_position: Optional[BoundingBox] = None
    camera_distance: Literal["close_up", "medium", "wide"]
    lighting: str
    color_palette: list[str] = Field(default_factory=list)
    visual_priority: list[str] = Field(default_factory=list)
    branding_constraints: list[str] = Field(default_factory=list)
    conflicts_resolved: int = 0
    status: Literal["success", "partial", "error"] = "success"
    partial_failure_reasons: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    generated_at: str

    @field_validator("video_id")
    @classmethod
    def video_id_must_not_be_empty(cls, v: str) -> str:
        """Reject blank video IDs."""
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Module 6.5 - Visual Reference Engine
# ---------------------------------------------------------------------------


class VisualBoundingBox(BaseModel):
    """Absolute pixel bounding box used by VRE crop and mask processors."""

    model_config = ConfigDict(frozen=True)

    x: int
    y: int
    width: int
    height: int

    @field_validator("x", "y")
    @classmethod
    def coordinate_must_not_be_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("bounding-box coordinates must be non-negative")
        return v

    @field_validator("width", "height")
    @classmethod
    def span_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("bounding-box width and height must be positive")
        return v


class AssetMetadata(BaseModel):
    """Traceable metadata for one VRE-generated conditioning asset."""

    model_config = ConfigDict(frozen=True)

    asset_type: str
    file_path: str
    checksum: str
    resolution: tuple[int, int]
    confidence_score: Optional[float] = None

    @field_validator("asset_type", "file_path", "checksum")
    @classmethod
    def text_fields_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("asset metadata text fields must not be empty")
        return v.strip()

    @field_validator("checksum")
    @classmethod
    def checksum_must_be_sha256(cls, v: str) -> str:
        if len(v) != 64 or any(char not in "0123456789abcdef" for char in v.lower()):
            raise ValueError("checksum must be a SHA-256 hex digest")
        return v.lower()

    @field_validator("resolution")
    @classmethod
    def resolution_must_be_positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError("resolution must be a positive (width, height) tuple")
        return v

    @field_validator("confidence_score")
    @classmethod
    def confidence_must_be_unit_interval(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("confidence_score must be in [0.0, 1.0]")
        return v


class VisualReferenceManifest(BaseModel):
    """Immutable VRE contract consumed by downstream ComfyUI workflow builders."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    source_image_path: str
    source_hash: str
    created_at: str
    assets: dict[str, Optional[AssetMetadata]]
    processing_metadata: dict[str, Any] = {}

    @field_validator("video_id", "source_image_path", "created_at")
    @classmethod
    def manifest_text_fields_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("manifest text fields must not be empty")
        return v.strip()

    @field_validator("source_hash")
    @classmethod
    def source_hash_must_be_sha256(cls, v: str) -> str:
        if len(v) != 64 or any(char not in "0123456789abcdef" for char in v.lower()):
            raise ValueError("source_hash must be a SHA-256 hex digest")
        return v.lower()


# ---------------------------------------------------------------------------
# AI Vision Stack V2.1 - Configuration and Model Lifecycle
# ---------------------------------------------------------------------------

from vision_stack.models import (  # noqa: E402
    RegisteredVisionModel,
    VisionModelBackend,
    VisionModelConfig,
    VisionModelFallback,
    VisionModelLifecycleState,
    VisionModelPrecision,
    VisionStackConfig,
)


# ---------------------------------------------------------------------------
# Module 7 — Local Image Generation Engine
# ---------------------------------------------------------------------------


class GenerationProfile(BaseModel):
    """Complete, named hardware and quality contract for one generation run."""

    model_config = ConfigDict(frozen=True)

    name: str
    checkpoint: str
    checkpoint_family: Literal["sdxl", "flux"]
    sampler: str
    scheduler: str
    steps: int
    cfg: float
    controlnet_enabled: bool
    ipadapter_enabled: bool
    restoration: Literal["codeformer", "gfpgan", "both", "none"]
    restoration_fidelity: float
    upscaler: Literal["real_esrgan_x4", "lanczos_only"]
    expected_vram_gb: float
    expected_generation_seconds: float
    edit_mode_default: Literal["legacy_txt2img", "staged_edit"] | None = None


class WorkflowTemplateRef(BaseModel):
    """Resolved, versioned workflow-template reference for a niche/profile pair."""

    model_config = ConfigDict(frozen=True)

    niche: str
    profile_name: str
    template_path: str
    workflow_version: str
    template_name: str


class ComfyUIWorkflowRef(BaseModel):
    """Resolved concrete ComfyUI graph provenance retained for compatibility."""

    model_config = ConfigDict(frozen=True)

    template_name: str
    workflow_version: str
    workflow_hash: str


class GeneratedAsset(BaseModel):
    """Metadata for one generated image asset; image bytes are never embedded."""

    model_config = ConfigDict(frozen=True)

    path: str
    width: int
    height: int
    sha256: str
    candidate_index: int = 0


class FaceMatchResult(BaseModel):
    """Outcome of a deterministic identity-similarity comparison."""

    model_config = ConfigDict(frozen=True)

    similarity: float = 0.0
    threshold: float
    passed: bool
    face_detected: bool = False
    skipped: bool = False


class QualityAssuranceReport(BaseModel):
    """Per-candidate quality gates and weighted, auditable quality signals."""

    model_config = ConfigDict(frozen=True)

    resolution_passed: bool
    file_integrity_passed: bool
    safety_passed: bool
    identity_score: float = 0.0
    face_quality_score: float = 0.0
    composition_score: float = 0.0
    text_safe_zone_score: float = 0.0
    object_preservation_score: float = 0.0
    color_compliance_score: float = 0.0
    overall_score: float = 0.0
    hard_gate_passed: bool


class CandidateScore(BaseModel):
    """Audit record for a candidate, including its deterministic rank when eligible."""

    model_config = ConfigDict(frozen=True)

    candidate_index: int
    overall_score: float
    identity_similarity: float = 0.0
    hard_gate_passed: bool
    rank: Optional[int] = None
    selected: bool = False


class ImageGenerationResult(BaseModel):
    """Versioned Module 7 manifest written beside a generated thumbnail."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    status: Literal["success", "error"] = "success"
    error_message: Optional[str] = None
    generated_asset: Optional[GeneratedAsset] = None
    workflow_version: str
    workflow_hash: Optional[str] = None
    prompt_package_hash: str
    generation_hash: Optional[str] = None
    profile_name: Optional[str] = None
    checkpoint_hash: Optional[str] = None
    lora_hashes: list[str] = []
    controlnet_hashes: list[str] = []
    ipadapter_hash: Optional[str] = None
    restoration_model_hashes: list[str] = []
    upscaler_hash: Optional[str] = None
    seed: Optional[int] = None
    candidate_scores: list[CandidateScore] = []
    selected_candidate_index: Optional[int] = None
    retry_count: int = 0
    stage_durations_seconds: dict[str, float] = {}
    duration_seconds: float = 0.0
    generated_at: str


class GenerationMetrics(BaseModel):
    """One append-only, local monitoring record for a Module 7 attempt."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    niche: str
    profile_name: Optional[str] = None
    workflow_version: str
    workflow_hash: Optional[str] = None
    generation_hash: Optional[str] = None
    num_candidates_requested: int = 1
    queue_time_seconds: float = 0.0
    generation_time_seconds: list[float] = []
    total_duration_seconds: float = 0.0
    identity_retry_count: int = 0
    generation_retry_count: int = 0
    failure_reason: Optional[str] = None
    identity_failures_count: int = 0
    qa_failures_count: int = 0
    winning_overall_score: Optional[float] = None
    winning_signal_scores: dict[str, float] = {}
    peak_vram_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    recorded_at: str


# ---------------------------------------------------------------------------
# Module 7 — Phase 4 Multi-Candidate Models
# ---------------------------------------------------------------------------


class CandidateStrategy(BaseModel):
    """Named, bounded transformation parameters applied per candidate."""

    model_config = ConfigDict(frozen=True)

    name: str
    camera_distance_shift: int = 0
    object_emphasis_bias: float = 0.0
    background_intensity_bias: float = 0.0
    color_grade_bias: float = 0.0
    typography_weight_bias: float = 0.0
    emotion_bias: float = 0.0
    lighting_bias: float = 0.0
    framing_bias: float = 0.0
    description: str = ""

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("CandidateStrategy name must not be empty")
        return v.strip()

    @field_validator("camera_distance_shift")
    @classmethod
    def validate_camera_distance_shift(cls, v: int) -> int:
        if v not in {-1, 0, 1}:
            raise ValueError("camera_distance_shift must be -1, 0, or 1")
        return v

    @field_validator(
        "object_emphasis_bias",
        "background_intensity_bias",
        "color_grade_bias",
        "typography_weight_bias",
        "emotion_bias",
        "lighting_bias",
        "framing_bias",
    )
    @classmethod
    def validate_bias_range(cls, v: float) -> float:
        if not -0.5 <= v <= 0.5:
            raise ValueError("Strategy bias fields must be within [-0.5, 0.5]")
        return float(v)

    @classmethod
    def faithful_default(cls) -> CandidateStrategy:
        return cls(
            name="faithful",
            camera_distance_shift=0,
            object_emphasis_bias=0.0,
            background_intensity_bias=0.0,
            color_grade_bias=0.0,
            typography_weight_bias=0.0,
            emotion_bias=0.0,
            lighting_bias=0.0,
            framing_bias=0.0,
            description="Variant A — Faithful to base blueprint without perturbation.",
        )


class StrategyPack(BaseModel):
    """Ordered set of declarative candidate strategies."""

    model_config = ConfigDict(frozen=True)

    name: str
    strategies: list[CandidateStrategy] = Field(default_factory=list)
    pack_version: str = "1.0.0"

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("StrategyPack name must not be empty")
        return v.strip()

    @field_validator("strategies")
    @classmethod
    def strategies_must_not_be_empty(cls, v: list[CandidateStrategy]) -> list[CandidateStrategy]:
        if not v:
            raise ValueError("StrategyPack must contain at least one strategy")
        return v


class CandidateManifestEntry(BaseModel):
    """Detailed audit entry for one candidate generated in a run."""

    model_config = ConfigDict(frozen=True)

    candidate_index: int
    strategy_name: str
    seed: int
    workflow_hash: str
    generation_parameters: GenerationParameters
    qa_report: QualityAssuranceReport
    face_match: FaceMatchResult
    candidate_score: CandidateScore
    stage_durations_seconds: dict[str, float] = Field(default_factory=dict)
    output_path: str


class CandidateManifest(BaseModel):
    """Multi-candidate generation audit trail serialized to candidate_manifest.json."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    entries: list[CandidateManifestEntry] = Field(default_factory=list)
    winning_candidate_index: int
    strategy_pack_name: Optional[str] = None
    generated_at: str

    @field_validator("video_id")
    @classmethod
    def video_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()


class GenerationRunMetadata(BaseModel):
    """Run-level execution provenance serialized to generation_metadata.json."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    profile_name: str
    workflow_version: str
    workflow_hash: str
    conditioning_asset_hashes: dict[str, str] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
    num_candidates_requested: int
    num_candidates_completed: int
    total_duration_seconds: float
    parallel_generation_used: bool = False
    retry_summary: dict[str, int] = Field(default_factory=dict)

    @field_validator("video_id", "profile_name", "workflow_version")
    @classmethod
    def text_fields_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("GenerationRunMetadata text fields must not be empty")
        return v.strip()



# ---------------------------------------------------------------------------
# Module 8 — Asset Extraction Engine
# ---------------------------------------------------------------------------


class AssetFileRef(BaseModel):
    """Refers to a persisted pixel or JSON asset on disk with checksum integrity."""

    model_config = ConfigDict(frozen=True)

    asset_type: str
    file_path: str
    checksum: str
    resolution: tuple[int, int]
    confidence_score: Optional[float] = None
    source: Literal["module4_reuse", "extracted", "derived"]

    @field_validator("asset_type", "file_path", "checksum")
    @classmethod
    def text_fields_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("asset reference text fields must not be empty")
        return v.strip()

    @field_validator("checksum")
    @classmethod
    def checksum_must_be_sha256(cls, v: str) -> str:
        if len(v) != 64 or any(char not in "0123456789abcdef" for char in v.lower()):
            raise ValueError("checksum must be a SHA-256 hex digest")
        return v.lower()

    @field_validator("resolution")
    @classmethod
    def resolution_must_be_positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError("resolution must be a positive (width, height) tuple")
        return v

    @field_validator("confidence_score")
    @classmethod
    def confidence_must_be_unit_interval(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("confidence_score must be in [0.0, 1.0]")
        return v


class PersonAsset(BaseModel):
    """Extracted person-specific visual elements, masks, embeddings, and landmarks."""

    model_config = ConfigDict(frozen=True)

    person_index: int
    face: Optional[AssetFileRef] = None
    face_mask: Optional[AssetFileRef] = None
    face_embedding: Optional[list[float]] = None
    facial_landmarks: Optional[list[tuple[float, float]]] = None
    body_mask: Optional[AssetFileRef] = None
    pose_keypoints: Optional[list[tuple[float, float, float]]] = None
    clothing_mask: Optional[AssetFileRef] = None
    hair_mask: Optional[AssetFileRef] = None
    accessories_masks: list[AssetFileRef] = Field(default_factory=list)
    source_face_detail_index: int
    extraction_status: Literal["success", "partial", "skipped"]
    extraction_notes: list[str] = Field(default_factory=list)


class SceneAsset(BaseModel):
    """Extracted scene structure including depth, segmentation, background, and foreground."""

    model_config = ConfigDict(frozen=True)

    background: Optional[AssetFileRef] = None
    foreground: Optional[AssetFileRef] = None
    depth_map: Optional[AssetFileRef] = None
    segmentation_map: Optional[AssetFileRef] = None
    sky_mask: Optional[AssetFileRef] = None
    ground_mask: Optional[AssetFileRef] = None
    extraction_status: Literal["success", "partial", "skipped"]
    extraction_notes: list[str] = Field(default_factory=list)


class ObjectAsset(BaseModel):
    """Extracted discrete foreground or subject object with mask and hierarchy."""

    model_config = ConfigDict(frozen=True)

    object_index: int
    label: str
    crop: Optional[AssetFileRef] = None
    mask: Optional[AssetFileRef] = None
    bbox: BoundingBox
    confidence: float
    parent_object_index: Optional[int] = None
    child_object_indices: list[int] = Field(default_factory=list)
    source_detected_object_index: int
    depth_layer: Optional[float] = None
    priority: Optional[int] = None
    scene_element_ref: Optional[str] = None

    @field_validator("label")
    @classmethod
    def label_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("object label must not be empty")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")
        return v


class TypographyAsset(BaseModel):
    """Extracted text region crop and classical-CV typography property estimates."""

    model_config = ConfigDict(frozen=True)

    text_region_index: int
    crop: Optional[AssetFileRef] = None
    text: str
    bbox: BoundingBox
    estimated_font_family_guess: Optional[str] = None
    estimated_font_size_px: Optional[float] = None
    alignment: Literal["left", "center", "right", "unknown"] = "unknown"
    dominant_text_color: Optional[str] = None
    has_stroke_or_outline: bool = False
    source_text_region_index: int


class VisualPropertiesAsset(BaseModel):
    """Analytical lighting, extended palette, gradient, blur, and focus properties."""

    model_config = ConfigDict(frozen=True)

    dominant_colors: list[str]
    palette_extended: list[str]
    gradients_detected: list[str] = Field(default_factory=list)
    lighting_direction: Optional[str] = None
    shadow_regions: list[BoundingBox] = Field(default_factory=list)
    highlight_regions: list[BoundingBox] = Field(default_factory=list)
    blur_map_summary: Literal["sharp", "mixed", "soft"]
    focus_bbox: Optional[BoundingBox] = None


class CompositionAsset(BaseModel):
    """Rendered composition visual overlays derived from Module 4 scoring."""

    model_config = ConfigDict(frozen=True)

    eye_flow_map: Optional[AssetFileRef] = None
    negative_space_mask: Optional[AssetFileRef] = None
    visual_hierarchy_overlay: Optional[AssetFileRef] = None
    source_composition_analysis: CompositionAnalysis


class EffectsAsset(BaseModel):
    """Classical-CV heuristic flags for visual effects (glow, outline, shadow, etc.)."""

    model_config = ConfigDict(frozen=True)

    glow_detected: bool = False
    outline_detected: bool = False
    drop_shadow_detected: bool = False
    motion_blur_detected: bool = False
    particles_detected: bool = False
    confidence: float = 0.0
    notes: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")
        return v


class AssetExtractionStatus(str, Enum):
    """Overall outcome of an asset extraction engine run."""

    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"


class AssetExtractionManifest(BaseModel):
    """Immutable, disk-persisted contract for all extracted thumbnail assets."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    source_thumbnail_path: str
    source_hash: str
    intelligence_hash: str
    engine_version: str
    people: list[PersonAsset] = Field(default_factory=list)
    scene: Optional[SceneAsset] = None
    objects: list[ObjectAsset] = Field(default_factory=list)
    typography: list[TypographyAsset] = Field(default_factory=list)
    visual_properties: Optional[VisualPropertiesAsset] = None
    composition: Optional[CompositionAsset] = None
    effects: Optional[EffectsAsset] = None
    status: AssetExtractionStatus = AssetExtractionStatus.SUCCESS
    partial_failure_reasons: list[str] = Field(default_factory=list)
    completed_families: list[str] = Field(default_factory=list)
    total_duration_seconds: float = 0.0
    extracted_at: str

    @field_validator("video_id", "source_thumbnail_path", "engine_version", "extracted_at")
    @classmethod
    def text_fields_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("manifest text fields must not be empty")
        return v.strip()

    @field_validator("source_hash", "intelligence_hash")
    @classmethod
    def hashes_must_be_sha256(cls, v: str) -> str:
        if len(v) != 64 or any(char not in "0123456789abcdef" for char in v.lower()):
            raise ValueError("hash must be a SHA-256 hex digest")
        return v.lower()


# ==============================================================================
# Module 9 - AI Decision Engine Models
# ==============================================================================


class DecisionAction(str, Enum):
    """Supported visual decision action types for Module 9."""

    KEEP = "keep"
    REMOVE = "remove"
    REPLACE = "replace"
    ENHANCE = "enhance"
    ADD = "add"


class DecisionSource(str, Enum):
    """Origin of a decision recommendation."""

    RULE = "rule"
    LLM = "llm"
    RULE_LLM_AGREEMENT = "rule_llm_agreement"
    CONFLICT_RESOLUTION = "conflict_resolution"


class TargetElement(BaseModel):
    """Visual or spatial target element for a decision."""

    model_config = ConfigDict(frozen=True)

    element_id: str
    element_type: str
    label: str
    bbox: Optional[BoundingBox] = None


class CandidateDecision(BaseModel):
    """Candidate decision prior to conflict resolution and validation."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    target: TargetElement
    action: DecisionAction
    confidence: float
    source: DecisionSource
    rationale: str
    rule_ids: list[str] = Field(default_factory=list)
    llm_raw_response_ref: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_unit_interval(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be in range [0.0, 1.0]")
        return float(v)


class ResolvedDecision(BaseModel):
    """Finalized decision post conflict resolution and validation."""

    model_config = ConfigDict(frozen=True)

    decision_id: str
    target: TargetElement
    action: DecisionAction
    confidence: float
    source: DecisionSource
    rationale: str
    priority_rank: int
    superseded_candidate_ids: list[str] = Field(default_factory=list)
    machine_reasoning: dict[str, Any] = Field(default_factory=dict)
    expected_ctr_gain: Optional[float] = None
    risk: Literal["low", "medium", "high"] = "low"
    depends_on_decision_ids: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_unit_interval(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be in range [0.0, 1.0]")
        return float(v)


class DecisionManifestStatus(str, Enum):
    """Execution status for a Decision Manifest."""

    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"


class DecisionManifest(BaseModel):
    """Umbrella manifest containing all resolved decisions for a video."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    source_generated_image_path: str
    source_generated_image_hash: str
    decisions: list[ResolvedDecision] = Field(default_factory=list)
    keep_count: int = 0
    remove_count: int = 0
    replace_count: int = 0
    enhance_count: int = 0
    add_count: int = 0
    overall_confidence: float = 0.0
    conflicts_resolved: int = 0
    llm_adjudications: int = 0
    status: DecisionManifestStatus = DecisionManifestStatus.SUCCESS
    partial_failure_reasons: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    total_duration_seconds: float = 0.0
    decided_at: str

    @field_validator("video_id")
    @classmethod
    def video_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()


class ReasoningTraceEntry(BaseModel):
    """Machine-readable step snapshot in reasoning_trace.json audit log."""

    model_config = ConfigDict(frozen=True)

    decision_id: str
    stage: Literal[
        "rule_engine",
        "ambiguity_router",
        "llm_reasoner",
        "conflict_resolver",
        "validator",
    ]
    input_snapshot: dict[str, Any] = Field(default_factory=dict)
    output_snapshot: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


# ---------------------------------------------------------------------------
# Module 10 — Asset Composer
# ---------------------------------------------------------------------------


class LayerDecision(str, Enum):
    """Action bucket governing how a composition layer is treated."""

    KEEP = "keep"
    REMOVE = "remove"
    REPLACE = "replace"
    ENHANCE = "enhance"
    ADD = "add"


class LayerRole(str, Enum):
    """Semantic role of a composition layer."""

    BACKGROUND = "background"
    FOREGROUND = "foreground"
    PERSON = "person"
    OBJECT = "object"
    TEXT = "text"
    EFFECT = "effect"


class CanvasTransform(BaseModel):
    """Pixel resolution and aspect ratio of the composition canvas."""

    model_config = ConfigDict(frozen=True)

    width: int
    height: int
    aspect_ratio: str

    @field_validator("width", "height")
    @classmethod
    def dimension_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("canvas dimensions must be positive")
        return v


class LayerTransform(BaseModel):
    """Pixel placement translation, scale factor, and optional crop bounding box."""

    model_config = ConfigDict(frozen=True)

    translate_x: int = 0
    translate_y: int = 0
    scale_x: float = 1.0
    scale_y: float = 1.0
    crop_box: Optional[VisualBoundingBox] = None


class MaskReference(BaseModel):
    """Reference to a VRE mask asset and optional feathering parameters."""

    model_config = ConfigDict(frozen=True)

    mask_path: str
    mask_checksum: str
    feather_px: int = 0
    source: Literal["vre"] = "vre"

    @field_validator("mask_path")
    @classmethod
    def mask_path_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("mask_path must not be empty")
        return v.strip()

    @field_validator("mask_checksum")
    @classmethod
    def mask_checksum_must_be_sha256(cls, v: str) -> str:
        if len(v) != 64 or any(char not in "0123456789abcdef" for char in v.lower()):
            raise ValueError("mask_checksum must be a SHA-256 hex digest")
        return v.lower()


class PlacementConstraints(BaseModel):
    """Safe margins, focal zone, and avoid zones in pixel space."""

    model_config = ConfigDict(frozen=True)

    safe_margin_px: int = 0
    avoid_zones_px: list[VisualBoundingBox] = Field(default_factory=list)
    focal_zone_px: Optional[VisualBoundingBox] = None


class TextPlacement(BaseModel):
    """Placement geometry and avoid zones for text overlay in pixel space."""

    model_config = ConfigDict(frozen=True)

    include_text: bool = False
    placement_zone_px: Optional[VisualBoundingBox] = None
    avoid_zones_px: list[VisualBoundingBox] = Field(default_factory=list)


class LightingAdjustment(BaseModel):
    """Target lighting parameters for color direction pass-through."""

    model_config = ConfigDict(frozen=True)

    target_brightness: float
    target_contrast: float
    target_saturation: float
    warm_or_cool: Literal["warm", "cool", "neutral"]


class AssetPlacement(BaseModel):
    """Resolved placement, decision, transform, and mask reference for an asset."""

    model_config = ConfigDict(frozen=True)

    asset_id: str
    role: LayerRole
    decision: LayerDecision
    source_path: Optional[str] = None
    mask: Optional[MaskReference] = None
    transform: LayerTransform
    z_index: int
    rationale: str = ""

    @field_validator("asset_id")
    @classmethod
    def asset_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("asset_id must not be empty")
        return v.strip()


class CompositionLayer(BaseModel):
    """One z-ordered layer in the composition stack."""

    model_config = ConfigDict(frozen=True)

    layer_id: str
    placement: AssetPlacement
    depth_hint_path: Optional[str] = None
    canny_hint_path: Optional[str] = None

    @field_validator("layer_id")
    @classmethod
    def layer_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("layer_id must not be empty")
        return v.strip()


class LayerGroup(BaseModel):
    """Logical grouping of layer IDs sharing a role."""

    model_config = ConfigDict(frozen=True)

    group_id: str
    role: LayerRole
    layer_ids: list[str]

    @field_validator("group_id")
    @classmethod
    def group_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("group_id must not be empty")
        return v.strip()


class WorkspaceStatistics(BaseModel):
    """Summary counts of layer decisions in a workspace."""

    model_config = ConfigDict(frozen=True)

    total_layers: int
    kept: int = 0
    removed: int = 0
    replaced: int = 0
    enhanced: int = 0
    added: int = 0


class WorkspaceMetadata(BaseModel):
    """Traceable provenance and hash metadata for a composition workspace."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    created_at: str
    vre_source_hash: str
    redesign_spec_hash: str
    prompt_package_hash: str
    engine_version: str

    @field_validator("video_id", "created_at", "engine_version")
    @classmethod
    def text_fields_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("metadata text fields must not be empty")
        return v.strip()

    @field_validator("vre_source_hash", "redesign_spec_hash", "prompt_package_hash")
    @classmethod
    def hashes_must_be_sha256(cls, v: str) -> str:
        if len(v) != 64 or any(char not in "0123456789abcdef" for char in v.lower()):
            raise ValueError("hashes must be SHA-256 hex digests")
        return v.lower()


class CompositionWorkspace(BaseModel):
    """Complete versioned Composition Workspace schema."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    canvas: CanvasTransform
    layers: list[CompositionLayer]
    groups: list[LayerGroup]
    text_placement: TextPlacement
    lighting: LightingAdjustment
    constraints: PlacementConstraints
    statistics: WorkspaceStatistics
    metadata: WorkspaceMetadata
    status: Literal["success", "partial", "error"] = "success"
    error_message: Optional[str] = None
    duration_seconds: float = 0.0

    @field_validator("video_id")
    @classmethod
    def video_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("video_id must not be empty")
        return v.strip()


class GenerationBundle(BaseModel):
    """Flat, ComfyUI-consumable summary artifact exported from a workspace."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    canvas: CanvasTransform
    reference_image_paths: dict[str, str] = Field(default_factory=dict)
    mask_paths: dict[str, str] = Field(default_factory=dict)
    depth_path: Optional[str] = None
    canny_path: Optional[str] = None
    layer_order: list[str] = Field(default_factory=list)
    workspace_hash: str
    prompt_package_hash: str
    generated_at: str

    @field_validator("video_id", "generated_at")
    @classmethod
    def text_fields_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("bundle text fields must not be empty")
        return v.strip()

    @field_validator("workspace_hash", "prompt_package_hash")
    @classmethod
    def hashes_must_be_sha256(cls, v: str) -> str:
        if len(v) != 64 or any(char not in "0123456789abcdef" for char in v.lower()):
            raise ValueError("hashes must be SHA-256 hex digests")
        return v.lower()


# ---------------------------------------------------------------------------
# Evaluation Framework (PVQEF) Models
# ---------------------------------------------------------------------------


class ModuleValidationResult(BaseModel):
    """Schema & invariant validation result for one pipeline stage artifact."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    module_name: str
    artifact_path: Optional[str] = None
    schema_valid: bool
    invariants_checked: list[str] = Field(default_factory=list)
    invariants_failed: list[str] = Field(default_factory=list)
    status: Literal["success", "partial", "error", "skipped"] = "success"
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    validated_at: str


class DimensionScore(BaseModel):
    """Score evaluation result for one quality evaluation dimension."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    score: float
    passed: bool
    threshold: float
    detail: dict[str, Any] = Field(default_factory=dict)
    scorer_version: str
    duration_seconds: float = 0.0
    status: Literal["success", "partial", "error", "skipped"] = "success"
    error_message: Optional[str] = None


class QualityEvaluationReport(BaseModel):
    """Aggregated fourteen-dimension quality evaluation report for one thumbnail."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    generated_asset_sha256: str
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    inline_scores: dict[str, float] = Field(default_factory=dict)
    weighted_overall_score: float = 0.0
    hard_gate_passed: bool
    status: Literal["success", "partial", "error"] = "success"
    partial_failure_reasons: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    total_duration_seconds: float = 0.0
    evaluated_at: str


class RegressionFinding(BaseModel):
    """Audit record for a detected statistical or threshold regression."""

    model_config = ConfigDict(frozen=True)

    rule_name: str
    severity: Literal["info", "warning", "critical"]
    dimension_or_stage: Optional[str] = None
    current_value: float
    baseline_value: float
    delta: float
    message: str


class PipelineRunReport(BaseModel):
    """Canonical run manifest for a PVQEF validation and evaluation run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    csv_path: str
    golden_only: bool = False
    total_creators: int
    succeeded: int
    skipped: int
    module_results: dict[str, list[ModuleValidationResult]] = Field(default_factory=dict)
    quality_reports: dict[str, QualityEvaluationReport] = Field(default_factory=dict)
    regressions: list[RegressionFinding] = Field(default_factory=list)
    stage_failure_counts: dict[str, int] = Field(default_factory=dict)
    aggregate_performance: dict[str, float] = Field(default_factory=dict)
    status: Literal["success", "partial", "error"] = "success"
    started_at: str
    completed_at: str
    total_duration_seconds: float = 0.0


class BenchmarkRecord(BaseModel):
    """Append-only benchmark history record summarizing one pipeline run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    recorded_at: str
    total_creators: int
    succeeded: int
    skipped: int
    mean_weighted_overall_score: float
    per_dimension_mean_scores: dict[str, float] = Field(default_factory=dict)
    mean_stage_durations_seconds: dict[str, float] = Field(default_factory=dict)
    peak_vram_mb: Optional[float] = None
    profile_name: Optional[str] = None
    workflow_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Module 10.5 — Thumbnail Planner Models
# ---------------------------------------------------------------------------


class HeadlineSource(str, Enum):
    """Source provenance for thumbnail headline text."""

    PRESERVED_OCR = "preserved_ocr"
    NONE = "none"
    GENERATED = "generated"  # Reserved for future LLM copy-gen extension


class FaceStrategy(str, Enum):
    """Strategy for creator face treatment in generated thumbnail."""

    NONE = "none"
    PRESERVE_AS_IS = "preserve_as_is"
    ENHANCE_EXISTING = "enhance_existing"
    PRESERVE_AS_IS_IDENTITY_LOCKED = "preserve_as_is_identity_locked"
    ENHANCE_EXISTING_IDENTITY_LOCKED = "enhance_existing_identity_locked"


class BackgroundStrategy(str, Enum):
    """Strategy for background generation and controlnet guidance."""

    STRUCTURE_GUIDED_REPLACE = "structure_guided_replace"
    UNGUIDED_REPLACE = "unguided_replace"
    KEEP = "keep"


class PlanConditioningAsset(BaseModel):
    """A single conditioning asset entry in a GenerationPlan."""

    model_config = ConfigDict(frozen=True)

    role: str
    asset_id: str
    path: str
    kind: Literal[
        "reference_image",
        "mask",
        "depth",
        "canny",
        "segmentation",
        "ip_adapter_reference",
        "text_exclusion_mask",
    ]
    source_module: Literal["module8", "vre", "module10"]

    @field_validator("role", "asset_id", "path")
    @classmethod
    def asset_text_fields_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("PlanConditioningAsset text fields must not be empty")
        return v.strip()


class GenerationPlan(BaseModel):
    """Deterministic, versioned generation plan artifact (Module 10.5)."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    headline: str
    headline_source: HeadlineSource
    headline_placement_zone: Optional[BoundingBox] = None
    face_strategy: FaceStrategy
    background_strategy: BackgroundStrategy
    preserve_objects: list[str] = Field(default_factory=list)
    composition_strategy: str
    camera_distance: str
    lighting: str
    color_palette: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    conditioning_assets: list[PlanConditioningAsset] = Field(default_factory=list)
    decision_manifest_hash: Optional[str] = None
    asset_extraction_manifest_hash: Optional[str] = None
    scene_graph_reference: Optional[str] = None
    prompt_package_hash: str
    workspace_hash: str
    status: Literal["success", "partial", "error"] = "success"
    partial_failure_reasons: list[str] = Field(default_factory=list)
    engine_version: str
    generated_at: str

    @field_validator("video_id", "engine_version", "generated_at")
    @classmethod
    def text_fields_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("GenerationPlan text fields must not be empty")
        return v.strip()

    @field_validator("prompt_package_hash", "workspace_hash")
    @classmethod
    def hashes_must_be_sha256(cls, v: str) -> str:
        if len(v) != 64 or any(char not in "0123456789abcdef" for char in v.lower()):
            raise ValueError("hashes must be SHA-256 hex digests")
        return v.lower()

    @field_validator("decision_manifest_hash", "asset_extraction_manifest_hash")
    @classmethod
    def optional_hashes_must_be_sha256(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if len(v) != 64 or any(char not in "0123456789abcdef" for char in v.lower()):
            raise ValueError("optional hashes must be SHA-256 hex digests")
        return v.lower()


# ---------------------------------------------------------------------------
# Module 7 V2 — Editing Engine Models
# ---------------------------------------------------------------------------


class EditRegion(BaseModel):
    """Specific element region targeted for localized staged editing in Module 7 V2."""

    model_config = ConfigDict(frozen=True)

    element_id: str
    decision_type: Literal["keep", "remove", "replace", "enhance", "add"]
    mask_path: Optional[Path] = None
    denoise_strength: float = Field(default=0.85, ge=0.0, le=1.0)
    steps: int = Field(default=25, ge=0)
    stage: Literal["background", "object"]


class EditPlan(BaseModel):
    """Concrete, frozen execution plan mapping element decisions to localized edit passes."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    edit_scope: Literal["none", "background_only", "object_only", "heavy_redesign"]
    regions: list[EditRegion] = Field(default_factory=list)
    fallback_elements: list[dict[str, str]] = Field(default_factory=list)
    created_at: str


EditRegion.model_rebuild()
EditPlan.model_rebuild()


# ---------------------------------------------------------------------------
# Module 10 — Creator Style Learning Models
# ---------------------------------------------------------------------------


class ThumbnailStyleSignature(BaseModel):
    """Auditable structured signature extracted from a single video thumbnail's intelligence."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    channel_id: str
    dominant_colors: list[str] = Field(default_factory=list)
    brightness: float = 0.5
    contrast: float = 0.5
    saturation: float = 0.5
    warm_or_cool: Literal["warm", "cool", "neutral"] = "neutral"
    color_harmony_score: float = 0.5
    subject_placement: str = "center"
    negative_space_ratio: float = 0.3
    balance_score: float = 0.5
    symmetry_score: float = 0.5
    face_scale_ratio: Optional[float] = None
    text_coverage_ratio: float = 0.0
    text_region_count: int = 0
    object_classes_present: list[str] = Field(default_factory=list)
    extracted_at: str = ""


class CreatorStyleEmbedding(BaseModel):
    """Running centroid embedding vector accumulated across a creator's thumbnails."""

    model_config = ConfigDict(frozen=True)

    channel_id: str
    embedding: list[float] = Field(default_factory=list)
    embedding_model: str = "OpenCLIP-ViT-B-32"
    source_video_ids: list[str] = Field(default_factory=list)
    sample_count: int = 0
    computed_at: str = ""


class StyleProfileManifest(BaseModel):
    """Manifest summary of a creator's persistent style profile store."""

    model_config = ConfigDict(frozen=True)

    channel_id: str
    sample_count: int = 0
    profile_established: bool = False
    first_seen_at: str = ""
    last_updated_at: str = ""
    video_ids: list[str] = Field(default_factory=list)
    schema_version: str = "1.0.0"


class StyleSimilarityResult(BaseModel):
    """Evaluation result of candidate/thumbnail visual similarity against creator style centroid."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    channel_id: str
    similarity_score: float
    belongs_to_identity: bool
    profile_established: bool


class StylePromptGuidance(BaseModel):
    """Deterministic prompt guidance block generated to align generation with creator style."""

    model_config = ConfigDict(frozen=True)

    channel_id: str
    color_guidance: str = ""
    composition_guidance: str = ""
    face_scale_guidance: Optional[str] = None
    applied: bool = False


class StyleAwareScore(BaseModel):
    """Additive style similarity score and bonus term for multi-dimensional candidate ranking."""

    model_config = ConfigDict(frozen=True)

    candidate_index: int
    style_similarity: float
    style_bonus: float


class StyleDriftAssessment(BaseModel):
    """Assessment of whether a creator has intentionally shifted their visual style."""

    model_config = ConfigDict(frozen=True)

    channel_id: str
    recent_similarity_scores: list[float] = Field(default_factory=list)
    drift_detected: bool = False
    drift_confidence: float = 0.0
    recommended_action: Literal["none", "monitor", "update_centroid"] = "none"


ThumbnailStyleSignature.model_rebuild()
CreatorStyleEmbedding.model_rebuild()
StyleProfileManifest.model_rebuild()
StyleSimilarityResult.model_rebuild()
StylePromptGuidance.model_rebuild()
StyleAwareScore.model_rebuild()
StyleDriftAssessment.model_rebuild()








