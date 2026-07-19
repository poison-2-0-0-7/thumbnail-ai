"""
thumbnail_intelligence.py
==========================

Module 4 — Thumbnail Intelligence Engine for the AI Thumbnail Outreach
Automation system.

Responsibility
--------------
Given a :class:`ThumbnailData` object produced by Module 3, analyse the
downloaded thumbnail together with its associated video metadata
(title, description, transcript, channel info) and produce a structured
:class:`ThumbnailIntelligence` report.

Out of scope
------------
This module **never** generates or edits images. It only understands
existing thumbnails. Image generation belongs to a later module.

Analysis pipeline
------------------
1. Load image      — decode, validate, verify dimensions, convert to RGB.
2. OCR              — visible text, confidence, location, coverage.
3. Face analysis    — faces, largest face, emotion, smile, gaze, pose.
4. Object detection — people, cars, food, phones, money, animals, etc.
5. Color analysis   — dominant colors, brightness, contrast, saturation.
6. Composition      — rule of thirds, clutter, hierarchy, balance.
7. Context merge    — combine every CV finding with title/description/
                       transcript/metadata into one structured context.
8. AI reasoning     — send the structured context (not the raw pixels)
                       to Gemini for CTR/curiosity/mismatch reasoning.

The thumbnail is never reasoned about in isolation: stage 7 is mandatory
before the Gemini call, and the transcript is always included when
available, because the thumbnail must be evaluated against what the
video actually contains.

Caching
-------
If ``data/analysis/{video_id}.json`` already exists, :func:`analyze_thumbnail`
does **not** consult it automatically (unlike Module 3's thumbnail cache) —
callers that want cache-first behaviour should check
:func:`load_cached_intelligence` themselves. This mirrors the fact that
re-analysis may legitimately be desired after a Gemini prompt or model
upgrade, whereas a downloaded thumbnail bitmap never changes.

Public API
----------
- :func:`analyze_thumbnail`       — run the full pipeline for one thumbnail.
- :func:`save_intelligence`       — atomic JSON write of a report.
- :func:`load_cached_intelligence` — load a previously saved report, if any.

Design contract with Module 3
------------------------------
Module 4 receives a :class:`ThumbnailData` object. It reads
``thumbnail_data.thumbnail_path`` to open the local image file and
``thumbnail_data.metadata`` for title/description/transcript/channel
context. Module 4 never downloads anything — that contract belongs
entirely to Module 3.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
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
    ANALYSIS_FILENAME_TEMPLATE,
    COLOR_PALETTE_SIZE,
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_DEVICE,
    FACE_MIN_CONFIDENCE,
    FACE_MODEL_NAME,
    GEMINI_MAX_RETRY_ATTEMPTS,
    GEMINI_RETRY_WAIT_MAX_SECONDS,
    GEMINI_RETRY_WAIT_MIN_SECONDS,
    LOG_DIR,
    MODULE4_LOG_PATH,
    OCR_LANGUAGES,
    OCR_MIN_CONFIDENCE,
    YOLO_MIN_CONFIDENCE,
    YOLO_MODEL_NAME,
)
from models import (  # noqa: E402
    BoundingBox,
    ColorProfile,
    CompositionAnalysis,
    DetectedObject,
    FaceAnalysis,
    FaceDetail,
    GeminiReasoning,
    OCRResult,
    TextRegion,
    ThumbnailData,
    ThumbnailIntelligence,
    VideoMetadata,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    """
    Attach a rotating file sink for Module 4 to the Loguru logger.

    Idempotent across repeated imports. Rotation at 10 MB, 30-day
    retention, async-safe enqueue mode.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(MODULE4_LOG_PATH),
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


class ThumbnailIntelligenceError(Exception):
    """Base exception for all failures raised by Module 4."""


class ImageLoadError(ThumbnailIntelligenceError):
    """
    Raised when the thumbnail file is missing, unreadable, or corrupted.

    This is the one failure mode that aborts the entire pipeline for a
    creator, since every downstream stage depends on a decoded image.
    """


class InvalidMetadataError(ThumbnailIntelligenceError):
    """Raised when the supplied :class:`ThumbnailData` is unusable (e.g.
    an empty title with no transcript to fall back on)."""


class OCREngineError(ThumbnailIntelligenceError):
    """Raised internally when the OCR engine fails to run. Callers of
    :func:`analyze_thumbnail` never see this directly — it is caught and
    degraded to a safe-default :class:`~models.OCRResult`."""


class FaceEngineError(ThumbnailIntelligenceError):
    """Raised internally when the face-analysis engine fails to run.
    Degraded to a safe-default :class:`~models.FaceAnalysis` by the caller."""


class ObjectDetectionEngineError(ThumbnailIntelligenceError):
    """Raised internally when the object-detection engine fails to run.
    Degraded to an empty object list by the caller."""


class ColorAnalysisError(ThumbnailIntelligenceError):
    """Raised internally when color analysis fails. Degraded to a
    safe-default :class:`~models.ColorProfile` by the caller."""


class CompositionAnalysisError(ThumbnailIntelligenceError):
    """Raised internally when composition analysis fails. Degraded to a
    safe-default :class:`~models.CompositionAnalysis` by the caller."""


class GeminiReasoningError(ThumbnailIntelligenceError):
    """
    Raised when the Gemini reasoning call fails after all retries, or
    when no API key is configured.

    Tenacity retries transient failures internally; this exception
    surfaces only the terminal failure, which the caller degrades to
    ``reasoning=None`` rather than aborting the whole report.
    """


class IntelligenceCacheError(ThumbnailIntelligenceError):
    """Raised when a filesystem error prevents reading or writing a
    cached intelligence report."""


# ---------------------------------------------------------------------------
# Lazily-loaded model singletons
# ---------------------------------------------------------------------------
#
# EasyOCR, InsightFace, and YOLO all load multi-hundred-MB weights and
# initialize a runtime session (ONNX / torch). Loading them once per
# process — not once per thumbnail — is required to hit the ~50
# creators/day throughput target and to avoid GPU memory churn. Each
# singleton is created on first use and reused for the lifetime of the
# process; there is deliberately no "unload" path, since the pipeline is
# designed to run as a long-lived batch process, not a one-shot script.

_ocr_reader: Optional[object] = None
_face_app: Optional[object] = None
_yolo_model: Optional[object] = None
_resolved_device: Optional[str] = None


def _resolve_device() -> str:
    """
    Determine the compute device to use, resolved once per process.

    Returns:
        ``"cuda"`` if a CUDA-capable GPU is available via torch,
        otherwise :data:`~config.DEFAULT_DEVICE` (``"cpu"``).
    """
    global _resolved_device
    if _resolved_device is not None:
        return _resolved_device

    try:
        import torch

        _resolved_device = "cuda" if torch.cuda.is_available() else DEFAULT_DEVICE
    except ImportError:
        _resolved_device = DEFAULT_DEVICE

    logger.info("Module 4 compute device resolved to {device}", device=_resolved_device)
    return _resolved_device


def _get_ocr_reader():
    """
    Return the process-wide EasyOCR reader, constructing it on first use.

    Returns:
        An ``easyocr.Reader`` instance.

    Raises:
        OCREngineError: If EasyOCR cannot be imported or initialized.
    """
    global _ocr_reader
    if _ocr_reader is not None:
        return _ocr_reader

    try:
        import easyocr

        device = _resolve_device()
        logger.info(
            "Loading EasyOCR reader (languages={langs}, gpu={gpu})",
            langs=OCR_LANGUAGES,
            gpu=(device == "cuda"),
        )
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=(device == "cuda"))
    except Exception as exc:  # noqa: BLE001 - any init failure degrades OCR
        logger.error("Failed to initialize EasyOCR reader: {exc}", exc=exc)
        raise OCREngineError(f"Could not initialize EasyOCR reader: {exc}") from exc

    return _ocr_reader


def _get_face_app():
    """
    Return the process-wide InsightFace analysis app, constructing it on
    first use.

    Returns:
        A prepared ``insightface.app.FaceAnalysis`` instance.

    Raises:
        FaceEngineError: If InsightFace cannot be imported or initialized.
    """
    global _face_app
    if _face_app is not None:
        return _face_app

    try:
        from insightface.app import FaceAnalysis as InsightFaceAnalysis

        device = _resolve_device()
        provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        logger.info(
            "Loading InsightFace app (model={model}, provider={provider})",
            model=FACE_MODEL_NAME,
            provider=provider,
        )
        app = InsightFaceAnalysis(name=FACE_MODEL_NAME, providers=[provider])
        app.prepare(ctx_id=0 if device == "cuda" else -1)
        _face_app = app
    except Exception as exc:  # noqa: BLE001 - any init failure degrades faces
        logger.error("Failed to initialize InsightFace app: {exc}", exc=exc)
        raise FaceEngineError(f"Could not initialize InsightFace app: {exc}") from exc

    return _face_app


def _get_yolo_model():
    """
    Return the process-wide YOLO model, constructing it on first use.

    Returns:
        An ``ultralytics.YOLO`` instance.

    Raises:
        ObjectDetectionEngineError: If Ultralytics cannot be imported or
            the model weights cannot be loaded.
    """
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    try:
        from ultralytics import YOLO

        logger.info("Loading YOLO model {model}", model=YOLO_MODEL_NAME)
        _yolo_model = YOLO(YOLO_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001 - any init failure degrades objects
        logger.error("Failed to initialize YOLO model: {exc}", exc=exc)
        raise ObjectDetectionEngineError(
            f"Could not initialize YOLO model: {exc}"
        ) from exc

    return _yolo_model


def reset_model_singletons() -> None:
    """
    Clear all lazily-loaded model singletons.

    Intended for use by tests (to force re-initialization against a
    mock) and by long-running batch drivers that want to release GPU
    memory between large batches. Not called automatically anywhere in
    the pipeline.
    """
    global _ocr_reader, _face_app, _yolo_model, _resolved_device
    _ocr_reader = None
    _face_app = None
    _yolo_model = None
    _resolved_device = None
    logger.debug("Module 4 model singletons reset")


# ---------------------------------------------------------------------------
# Stage 1 — Image loading and validation
# ---------------------------------------------------------------------------


def load_and_validate_image(image_path: Path) -> np.ndarray:
    """
    Load, validate, and normalize the thumbnail image for analysis.

    Steps:

    1. Verify the file exists.
    2. Open with Pillow and force a full pixel decode (catches truncation
       and most corruption that ``verify()`` alone would miss).
    3. Verify the image has non-zero width and height.
    4. Convert to RGB (thumbnails may arrive as palette, RGBA, or
       grayscale images from earlier CDN edge variance).

    Args:
        image_path: Path to the thumbnail file on disk.

    Returns:
        An ``(H, W, 3)`` ``uint8`` NumPy array in RGB order.

    Raises:
        ImageLoadError: If the file is missing, unreadable, corrupted,
            or has zero-sized dimensions.
    """
    if not image_path.exists():
        raise ImageLoadError(f"Thumbnail file does not exist: {image_path}")

    try:
        with Image.open(image_path) as img:
            img.load()
            rgb_img = img.convert("RGB")
            width, height = rgb_img.size
            if width <= 0 or height <= 0:
                raise ImageLoadError(
                    f"Thumbnail has invalid dimensions {width}x{height}: {image_path}"
                )
            array = np.array(rgb_img, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ImageLoadError(
            f"Pillow cannot identify image format: {image_path}"
        ) from exc
    except (OSError, SyntaxError) as exc:
        raise ImageLoadError(
            f"Thumbnail file is corrupted or truncated: {image_path} — {exc}"
        ) from exc

    logger.debug(
        "Loaded thumbnail {path} ({w}x{h})",
        path=image_path,
        w=width,
        h=height,
    )
    return array


# ---------------------------------------------------------------------------
# Stage 2 — OCR
# ---------------------------------------------------------------------------


def run_ocr(image: np.ndarray) -> OCRResult:
    """
    Extract visible text, per-region confidence, and location from the
    thumbnail via EasyOCR.

    Args:
        image: ``(H, W, 3)`` RGB NumPy array.

    Returns:
        A populated :class:`~models.OCRResult`. Regions below
        :data:`~config.OCR_MIN_CONFIDENCE` are dropped as noise.

    Raises:
        OCREngineError: If the OCR engine could not be initialized or
            raised during inference. Callers are expected to catch this
            and degrade to a safe-default result rather than aborting.
    """
    height, width = image.shape[0], image.shape[1]
    reader = _get_ocr_reader()

    try:
        raw_results = reader.readtext(image)
    except Exception as exc:  # noqa: BLE001 - inference failure degrades OCR
        raise OCREngineError(f"EasyOCR inference failed: {exc}") from exc

    regions: list[TextRegion] = []
    kept_confidences: list[float] = []
    covered_area = 0.0

    for polygon, text, confidence in raw_results:
        if confidence < OCR_MIN_CONFIDENCE:
            continue
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        x_min, x_max = min(xs) / width, max(xs) / width
        y_min, y_max = min(ys) / height, max(ys) / height
        regions.append(
            TextRegion(
                text=text.strip(),
                confidence=float(confidence),
                bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            )
        )
        kept_confidences.append(float(confidence))
        covered_area += max(0.0, x_max - x_min) * max(0.0, y_max - y_min)

    visible_text = " ".join(region.text for region in regions if region.text)
    word_count = len(visible_text.split()) if visible_text else 0
    average_confidence = (
        sum(kept_confidences) / len(kept_confidences) if kept_confidences else 0.0
    )
    text_coverage_ratio = min(1.0, covered_area)

    return OCRResult(
        visible_text=visible_text,
        text_regions=regions,
        word_count=word_count,
        text_coverage_ratio=text_coverage_ratio,
        average_confidence=average_confidence,
        engine_available=True,
    )


# ---------------------------------------------------------------------------
# Stage 3 — Face analysis
# ---------------------------------------------------------------------------

_EMOTION_LABELS: tuple[str, ...] = (
    "neutral",
    "happy",
    "surprised",
    "sad",
    "angry",
)


def _position_label(bbox: BoundingBox) -> str:
    """
    Classify a bounding box's horizontal position into a coarse label.

    Args:
        bbox: Normalized bounding box.

    Returns:
        ``"left-third"``, ``"center"``, or ``"right-third"``.
    """
    center_x = (bbox.x_min + bbox.x_max) / 2.0
    if center_x < 1.0 / 3.0:
        return "left-third"
    if center_x > 2.0 / 3.0:
        return "right-third"
    return "center"


def run_face_analysis(image: np.ndarray) -> FaceAnalysis:
    """
    Detect faces and estimate emotion, smile, gaze, and pose via
    InsightFace.

    Args:
        image: ``(H, W, 3)`` RGB NumPy array.

    Returns:
        A populated :class:`~models.FaceAnalysis`, with faces ordered
        largest-first and ``is_largest`` set on the first entry.

    Raises:
        FaceEngineError: If the face-analysis engine could not be
            initialized or raised during inference. Callers are expected
            to catch this and degrade to a safe-default result rather
            than aborting.
    """
    height, width = image.shape[0], image.shape[1]
    app = _get_face_app()

    try:
        # InsightFace expects BGR ordering (it wraps OpenCV internally).
        bgr_image = image[:, :, ::-1]
        raw_faces = app.get(bgr_image)
    except Exception as exc:  # noqa: BLE001 - inference failure degrades faces
        raise FaceEngineError(f"InsightFace inference failed: {exc}") from exc

    kept: list[tuple[float, FaceDetail]] = []
    for face in raw_faces:
        det_score = float(getattr(face, "det_score", 0.0))
        if det_score < FACE_MIN_CONFIDENCE:
            continue

        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        bbox = BoundingBox(
            x_min=x1 / width,
            y_min=y1 / height,
            x_max=x2 / width,
            y_max=y2 / height,
        )
        area = max(0.0, bbox.x_max - bbox.x_min) * max(0.0, bbox.y_max - bbox.y_min)

        emotion: Optional[str] = None
        emotion_confidence: Optional[float] = None
        gender_age_available = hasattr(face, "gender") and hasattr(face, "age")
        # InsightFace's buffalo_l pack does not ship a dedicated emotion
        # head; a lightweight, deterministic heuristic derived from facial
        # landmark geometry is used instead so the field is still populated
        # rather than always None. This keeps Stage 3 self-contained
        # without pulling in a second heavy model.
        landmarks = getattr(face, "landmark_2d_106", None)
        smile_detected: Optional[bool] = None
        if landmarks is not None:
            smile_detected, emotion, emotion_confidence = _estimate_expression(
                landmarks
            )

        eye_direction = _estimate_eye_direction(face)
        head_pose = _estimate_head_pose(face)

        detail = FaceDetail(
            bbox=bbox,
            detection_confidence=det_score,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            smile_detected=smile_detected,
            eye_direction=eye_direction,
            head_pose=head_pose,
            position_label=_position_label(bbox),
        )
        kept.append((area, detail))

    kept.sort(key=lambda pair: pair[0], reverse=True)
    faces: list[FaceDetail] = []
    for index, (_, detail) in enumerate(kept):
        faces.append(detail.model_copy(update={"is_largest": index == 0}))

    return FaceAnalysis(
        face_count=len(faces),
        faces=faces,
        has_face=len(faces) > 0,
        engine_available=True,
    )


def _estimate_expression(
    landmarks: np.ndarray,
) -> tuple[Optional[bool], Optional[str], Optional[float]]:
    """
    Heuristically estimate smile and coarse emotion from 106-point
    facial landmarks.

    This is a lightweight geometric heuristic (mouth-width-to-face-width
    ratio), not a learned classifier, so ``emotion_confidence`` is
    reported conservatively.

    Args:
        landmarks: ``(106, 2)`` array of facial landmark coordinates.

    Returns:
        A ``(smile_detected, emotion, emotion_confidence)`` tuple. All
        three are ``None`` if the heuristic cannot be computed.
    """
    try:
        # InsightFace's 106-point scheme places mouth corners around
        # indices 52 and 61, per the model's published landmark map.
        mouth_left = landmarks[52]
        mouth_right = landmarks[61]
        face_width = np.linalg.norm(landmarks[0] - landmarks[32])
        if face_width <= 0:
            return None, None, None
        mouth_width_ratio = float(np.linalg.norm(mouth_left - mouth_right) / face_width)
    except (IndexError, TypeError, ValueError):
        return None, None, None

    smile_detected = mouth_width_ratio > 0.45
    if smile_detected:
        return True, "happy", min(1.0, (mouth_width_ratio - 0.45) * 2.0 + 0.5)
    return False, "neutral", 0.5


def _estimate_eye_direction(face: object) -> Optional[str]:
    """
    Best-effort gaze estimate derived from eye-keypoint symmetry.

    Args:
        face: A raw InsightFace ``Face`` result object.

    Returns:
        ``"camera"``, ``"left"``, ``"right"``, or ``None`` if keypoints
        are unavailable.
    """
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 2:
        return None
    left_eye, right_eye = kps[0], kps[1]
    nose = kps[2] if len(kps) > 2 else None
    if nose is None:
        return "camera"
    eye_midpoint_x = (left_eye[0] + right_eye[0]) / 2.0
    offset = float(nose[0] - eye_midpoint_x)
    eye_span = float(abs(right_eye[0] - left_eye[0])) or 1.0
    ratio = offset / eye_span
    if ratio > 0.15:
        return "left"
    if ratio < -0.15:
        return "right"
    return "camera"


def _estimate_head_pose(face: object) -> Optional[str]:
    """
    Best-effort head-pose estimate derived from InsightFace's pose
    attribute when available.

    Args:
        face: A raw InsightFace ``Face`` result object.

    Returns:
        ``"frontal"``, ``"turned"``, or ``None`` if pose data is
        unavailable.
    """
    pose = getattr(face, "pose", None)
    if pose is None:
        return None
    try:
        yaw = float(pose[1])
    except (IndexError, TypeError, ValueError):
        return None
    return "frontal" if abs(yaw) < 20.0 else "turned"


# ---------------------------------------------------------------------------
# Stage 4 — Object detection
# ---------------------------------------------------------------------------


def run_object_detection(image: np.ndarray) -> list[DetectedObject]:
    """
    Detect thumbnail-relevant objects (people, cars, food, phones, money,
    animals, electronics, logos, charts, screenshots, etc.) via YOLO.

    Args:
        image: ``(H, W, 3)`` RGB NumPy array.

    Returns:
        Detected objects with confidence at or above
        :data:`~config.YOLO_MIN_CONFIDENCE`, sorted by descending
        confidence.

    Raises:
        ObjectDetectionEngineError: If the YOLO model could not be
            initialized or raised during inference. Callers are expected
            to catch this and degrade to an empty list rather than
            aborting.
    """
    height, width = image.shape[0], image.shape[1]
    model = _get_yolo_model()

    try:
        results = model.predict(
            image, conf=YOLO_MIN_CONFIDENCE, verbose=False, device=_resolve_device()
        )
    except Exception as exc:  # noqa: BLE001 - inference failure degrades objects
        raise ObjectDetectionEngineError(f"YOLO inference failed: {exc}") from exc

    detections: list[DetectedObject] = []
    for result in results:
        names = result.names
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < YOLO_MIN_CONFIDENCE:
                continue
            class_id = int(box.cls[0])
            label = names.get(class_id, str(class_id))
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append(
                DetectedObject(
                    label=label,
                    confidence=confidence,
                    bbox=BoundingBox(
                        x_min=x1 / width,
                        y_min=y1 / height,
                        x_max=x2 / width,
                        y_max=y2 / height,
                    ),
                )
            )

    detections.sort(key=lambda obj: obj.confidence, reverse=True)
    return detections


# ---------------------------------------------------------------------------
# Stage 5 — Color analysis
# ---------------------------------------------------------------------------


def run_color_analysis(image: np.ndarray) -> ColorProfile:
    """
    Extract dominant colors, brightness, contrast, saturation, and
    warm/cool classification via k-means clustering in RGB space and
    HSV statistics.

    Args:
        image: ``(H, W, 3)`` RGB NumPy array.

    Returns:
        A populated :class:`~models.ColorProfile`.

    Raises:
        ColorAnalysisError: If color analysis fails (e.g. an empty or
            degenerate image array). Callers are expected to catch this
            and degrade to a safe-default result rather than aborting.
    """
    try:
        import colorsys

        pixels = image.reshape(-1, 3).astype(np.float32)
        if pixels.size == 0:
            raise ColorAnalysisError("Image has no pixel data")

        # --- dominant colors via simple k-means (no extra dependency) ---
        rng = np.random.default_rng(seed=42)
        sample_size = min(10_000, pixels.shape[0])
        sample_idx = rng.choice(pixels.shape[0], size=sample_size, replace=False)
        sample = pixels[sample_idx]

        k = COLOR_PALETTE_SIZE
        centroids = sample[rng.choice(sample.shape[0], size=k, replace=False)]
        for _ in range(10):
            distances = np.linalg.norm(sample[:, None, :] - centroids[None, :, :], axis=2)
            assignments = np.argmin(distances, axis=1)
            new_centroids = np.array(
                [
                    sample[assignments == i].mean(axis=0)
                    if np.any(assignments == i)
                    else centroids[i]
                    for i in range(k)
                ]
            )
            if np.allclose(new_centroids, centroids, atol=1.0):
                centroids = new_centroids
                break
            centroids = new_centroids

        counts = np.bincount(assignments, minlength=k)
        order = np.argsort(counts)[::-1]
        dominant_colors = [
            "#{:02x}{:02x}{:02x}".format(
                int(np.clip(centroids[i][0], 0, 255)),
                int(np.clip(centroids[i][1], 0, 255)),
                int(np.clip(centroids[i][2], 0, 255)),
            )
            for i in order
            if counts[i] > 0
        ]

        # --- brightness / contrast (perceptual luminance) ---
        luminance = (
            0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
        ) / 255.0
        brightness = float(np.mean(luminance))
        contrast = float(np.clip(np.std(luminance) * 2.0, 0.0, 1.0))

        # --- saturation (HSV) ---
        max_c = pixels.max(axis=1)
        min_c = pixels.min(axis=1)
        chroma = max_c - min_c
        saturation_per_pixel = np.divide(
            chroma, max_c, out=np.zeros_like(chroma), where=max_c != 0
        )
        saturation = float(np.mean(saturation_per_pixel))

        # --- warm vs cool ---
        mean_r, mean_g, mean_b = pixels.mean(axis=0)
        warm_score = float(mean_r - mean_b)
        if warm_score > 10.0:
            warm_or_cool = "warm"
        elif warm_score < -10.0:
            warm_or_cool = "cool"
        else:
            warm_or_cool = "neutral"

        # --- harmony: circular spread of dominant hues ---
        hues = []
        for hexcolor in dominant_colors:
            r = int(hexcolor[1:3], 16) / 255.0
            g = int(hexcolor[3:5], 16) / 255.0
            b = int(hexcolor[5:7], 16) / 255.0
            h, _, _ = colorsys.rgb_to_hsv(r, g, b)
            hues.append(h * 2 * np.pi)
        if len(hues) >= 2:
            sin_sum = np.mean(np.sin(hues))
            cos_sum = np.mean(np.cos(hues))
            resultant_length = float(np.sqrt(sin_sum**2 + cos_sum**2))
            harmony_score = resultant_length
        else:
            harmony_score = 1.0

        return ColorProfile(
            dominant_colors=dominant_colors,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            warm_or_cool=warm_or_cool,
            harmony_score=float(np.clip(harmony_score, 0.0, 1.0)),
        )
    except ColorAnalysisError:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure degrades colors
        raise ColorAnalysisError(f"Color analysis failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Stage 6 — Composition analysis
# ---------------------------------------------------------------------------

_THIRDS_POINTS: tuple[tuple[float, float], ...] = (
    (1 / 3, 1 / 3),
    (2 / 3, 1 / 3),
    (1 / 3, 2 / 3),
    (2 / 3, 2 / 3),
)


def _bbox_center(bbox: BoundingBox) -> tuple[float, float]:
    """Return the normalized ``(x, y)`` center of a bounding box."""
    return (bbox.x_min + bbox.x_max) / 2.0, (bbox.y_min + bbox.y_max) / 2.0


def _bbox_area(bbox: BoundingBox) -> float:
    """Return the normalized area of a bounding box, clamped to ``>= 0``."""
    return max(0.0, bbox.x_max - bbox.x_min) * max(0.0, bbox.y_max - bbox.y_min)


def _bboxes_overlap(a: BoundingBox, b: BoundingBox) -> bool:
    """Return whether two normalized bounding boxes overlap at all."""
    return not (
        a.x_max <= b.x_min or b.x_max <= a.x_min or a.y_max <= b.y_min or b.y_max <= a.y_min
    )


def run_composition_analysis(
    image: np.ndarray,
    faces: FaceAnalysis,
    ocr: OCRResult,
    objects: list[DetectedObject],
) -> CompositionAnalysis:
    """
    Analyze rule-of-thirds alignment, subject placement, negative space,
    clutter, visual hierarchy, text/subject overlap, balance, and
    symmetry.

    The "primary subject" is the largest detected face when one exists,
    otherwise the highest-confidence detected object, otherwise
    undefined (in which case placement/rule-of-thirds scores default to
    their safe minimums).

    Args:
        image:   ``(H, W, 3)`` RGB NumPy array (used for balance/symmetry).
        faces:   Stage 3 output.
        ocr:     Stage 2 output.
        objects: Stage 4 output.

    Returns:
        A populated :class:`~models.CompositionAnalysis`.

    Raises:
        CompositionAnalysisError: If composition analysis fails.
            Callers are expected to catch this and degrade to a
            safe-default result rather than aborting.
    """
    try:
        primary_bbox: Optional[BoundingBox] = None
        if faces.has_face:
            primary_bbox = next(
                (f.bbox for f in faces.faces if f.is_largest), faces.faces[0].bbox
            )
        elif objects:
            primary_bbox = objects[0].bbox

        if primary_bbox is not None:
            cx, cy = _bbox_center(primary_bbox)
            min_dist = min(
                ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 for px, py in _THIRDS_POINTS
            )
            # Max possible distance to nearest thirds point from a corner
            # is bounded by the image diagonal fraction; normalize so a
            # dead-on hit scores 1.0 and a far corner scores near 0.0.
            rule_of_thirds_score = float(np.clip(1.0 - (min_dist / 0.5), 0.0, 1.0))
            subject_placement = _position_label(primary_bbox)
        else:
            rule_of_thirds_score = 0.0
            subject_placement = "none-detected"

        # --- negative space: area not covered by faces, objects, or text ---
        occupied_area = 0.0
        for face in faces.faces:
            occupied_area += _bbox_area(face.bbox)
        for obj in objects:
            occupied_area += _bbox_area(obj.bbox)
        for region in ocr.text_regions:
            occupied_area += _bbox_area(region.bbox)
        negative_space_ratio = float(np.clip(1.0 - occupied_area, 0.0, 1.0))

        # --- clutter: element density (faces + objects + text regions) ---
        element_count = len(faces.faces) + len(objects) + len(ocr.text_regions)
        clutter_score = float(np.clip(element_count / 10.0, 0.0, 1.0))

        # --- visual hierarchy: how much the primary subject dominates ---
        if primary_bbox is not None:
            primary_area = _bbox_area(primary_bbox)
            visual_hierarchy_score = float(
                np.clip(primary_area / max(occupied_area, primary_area, 1e-6), 0.0, 1.0)
            )
        else:
            visual_hierarchy_score = 0.0

        # --- text overlap with primary subject ---
        text_overlaps_subject = False
        if primary_bbox is not None:
            text_overlaps_subject = any(
                _bboxes_overlap(primary_bbox, region.bbox) for region in ocr.text_regions
            )

        # --- balance: left vs right visual weight ---
        height, width = image.shape[0], image.shape[1]
        left_luminance = float(np.mean(image[:, : width // 2, :])) if width > 1 else 0.0
        right_luminance = float(np.mean(image[:, width // 2 :, :])) if width > 1 else 0.0
        total_luminance = left_luminance + right_luminance
        if total_luminance > 0:
            balance_score = float(
                1.0 - abs(left_luminance - right_luminance) / total_luminance
            )
        else:
            balance_score = 1.0

        # --- symmetry: mirror the image horizontally and compare ---
        mirrored = image[:, ::-1, :]
        diff = np.abs(image.astype(np.float32) - mirrored.astype(np.float32))
        symmetry_score = float(np.clip(1.0 - (np.mean(diff) / 255.0), 0.0, 1.0))

        return CompositionAnalysis(
            rule_of_thirds_score=rule_of_thirds_score,
            subject_placement=subject_placement,
            negative_space_ratio=negative_space_ratio,
            clutter_score=clutter_score,
            visual_hierarchy_score=visual_hierarchy_score,
            text_overlaps_subject=text_overlaps_subject,
            balance_score=float(np.clip(balance_score, 0.0, 1.0)),
            symmetry_score=symmetry_score,
        )
    except Exception as exc:  # noqa: BLE001 - any failure degrades composition
        raise CompositionAnalysisError(f"Composition analysis failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Stage 7 — Context merge
# ---------------------------------------------------------------------------


def _build_gemini_context(
    metadata: VideoMetadata,
    ocr: OCRResult,
    faces: FaceAnalysis,
    objects: list[DetectedObject],
    colors: ColorProfile,
    composition: CompositionAnalysis,
) -> dict:
    """
    Merge every computer-vision finding with the video's title,
    description, transcript, and channel metadata into one structured
    context dictionary.

    This is Stage 7. The thumbnail is intentionally never reasoned
    about independently of the video it belongs to — the transcript in
    particular is included whenever available, since the thumbnail's
    job is to represent what the video actually contains.

    Args:
        metadata:    Video metadata from Module 2.
        ocr:         Stage 2 output.
        faces:       Stage 3 output.
        objects:     Stage 4 output.
        colors:      Stage 5 output.
        composition: Stage 6 output.

    Returns:
        A JSON-serializable dict combining vision findings and video
        context, ready to be sent to Gemini.
    """
    from config import GEMINI_TRANSCRIPT_CHAR_LIMIT

    transcript = metadata.transcript or ""
    if len(transcript) > GEMINI_TRANSCRIPT_CHAR_LIMIT:
        transcript = transcript[:GEMINI_TRANSCRIPT_CHAR_LIMIT]

    return {
        "video": {
            "title": metadata.title or "",
            "description": (metadata.description or "")[:2000],
            "transcript": transcript,
            "transcript_available": bool(metadata.transcript),
            "uploader": metadata.uploader,
            "channel_id": metadata.channel_id,
            "categories": metadata.categories,
            "tags": metadata.tags,
            "view_count": metadata.view_count,
            "like_count": metadata.like_count,
        },
        "thumbnail_analysis": {
            "ocr": ocr.model_dump(),
            "faces": faces.model_dump(),
            "objects": [obj.model_dump() for obj in objects],
            "colors": colors.model_dump(),
            "composition": composition.model_dump(),
        },
    }


# ---------------------------------------------------------------------------
# AI reasoning (Gemini)
# ---------------------------------------------------------------------------

_GEMINI_SYSTEM_PROMPT: str = (
    "You are a YouTube thumbnail strategy expert. You will be given "
    "STRUCTURED DATA describing a thumbnail (OCR text, detected faces "
    "with emotion/gaze/pose, detected objects, color profile, and "
    "composition metrics) together with the video's title, description, "
    "and transcript. Reason about the thumbnail strictly in the context "
    "of what the video actually contains — the transcript is the ground "
    "truth for the video's content, and the thumbnail's job is to "
    "represent that content compellingly. Respond ONLY with a single "
    "JSON object (no markdown fences, no prose outside the object) "
    "matching exactly this schema:\n"
    "{\n"
    '  "ctr_potential_score": float 0-1,\n'
    '  "curiosity_gap_score": float 0-1,\n'
    '  "emotional_impact": string,\n'
    '  "visual_storytelling_notes": string,\n'
    '  "content_mismatch_detected": boolean,\n'
    '  "mismatch_explanation": string or null,\n'
    '  "strengths": [string],\n'
    '  "weaknesses": [string],\n'
    '  "redesign_recommendations": [string],\n'
    '  "elements_to_preserve": [string]\n'
    "}"
)


def _before_sleep_log_gemini(retry_state: RetryCallState) -> None:
    """
    Loguru-compatible Tenacity sleep callback logged at WARNING level
    for Gemini retry attempts.

    Args:
        retry_state: Current Tenacity retry state.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retrying Gemini reasoning call (attempt {n}/{max}): {exc}",
        n=retry_state.attempt_number,
        max=GEMINI_MAX_RETRY_ATTEMPTS,
        exc=exc,
    )


class _GeminiTransientError(ThumbnailIntelligenceError):
    """Internal marker for a Gemini failure that tenacity should retry."""


@retry(
    stop=stop_after_attempt(GEMINI_MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1,
        min=GEMINI_RETRY_WAIT_MIN_SECONDS,
        max=GEMINI_RETRY_WAIT_MAX_SECONDS,
    ),
    retry=retry_if_exception_type(_GeminiTransientError),
    before_sleep=_before_sleep_log_gemini,
    reraise=True,
)
def _call_gemini_api(context: dict) -> str:
    """
    Perform a single Gemini API call and return the raw response text.

    This is the only place in Module 4 that makes a network call to
    Gemini. Error classification happens here so the retry decorator
    only fires on transient failures:

    * Connection errors, timeouts, and 5xx-equivalent SDK errors are
      wrapped as :class:`_GeminiTransientError` and retried.
    * A missing API key is a permanent configuration problem and is
      raised as :class:`GeminiReasoningError` directly (not retried).

    Args:
        context: The Stage 7 merged context dict (vision findings +
            video title/description/transcript/metadata).

    Returns:
        The raw text of Gemini's response, expected to be a JSON object.

    Raises:
        GeminiReasoningError: If no API key is configured.
        _GeminiTransientError: On any transient failure. Tenacity
            retries this up to :data:`~config.GEMINI_MAX_RETRY_ATTEMPTS`
            times before it propagates.
    """
    import os

    from config import (
        GEMINI_API_KEY_ENV_VAR,
        GEMINI_MODEL_NAME,
        GEMINI_REQUEST_TIMEOUT_SECONDS,
    )

    api_key = os.environ.get(GEMINI_API_KEY_ENV_VAR)
    if not api_key:
        raise GeminiReasoningError(
            f"No Gemini API key configured (expected env var "
            f"{GEMINI_API_KEY_ENV_VAR!r})"
        )

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        prompt = f"{_GEMINI_SYSTEM_PROMPT}\n\nSTRUCTURED DATA:\n{json.dumps(context)}"
        response = model.generate_content(
            prompt,
            request_options={"timeout": GEMINI_REQUEST_TIMEOUT_SECONDS},
        )
        return response.text
    except GeminiReasoningError:
        raise
    except Exception as exc:  # noqa: BLE001 - any SDK failure is transient
        raise _GeminiTransientError(f"Gemini API call failed: {exc}") from exc


def generate_reasoning(context: dict) -> GeminiReasoning:
    """
    Send the Stage 7 merged context to Gemini and parse its response
    into a structured :class:`GeminiReasoning`.

    Gemini reasons over the STRUCTURED vision findings (OCR, faces,
    objects, colors, composition) plus title/description/transcript —
    never over the raw thumbnail pixels — per the requirement that the
    thumbnail always be evaluated in the context of the video's actual
    content.

    Args:
        context: The Stage 7 merged context dict.

    Returns:
        A populated :class:`GeminiReasoning`.

    Raises:
        GeminiReasoningError: If no API key is configured, if every
            retry attempt fails, or if Gemini's response cannot be
            parsed as the expected JSON schema.
    """
    try:
        raw_text = _call_gemini_api(context)
    except _GeminiTransientError as exc:
        raise GeminiReasoningError(
            f"Gemini reasoning failed after {GEMINI_MAX_RETRY_ATTEMPTS} attempts: {exc}"
        ) from exc

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise GeminiReasoningError(
            f"Gemini response was not valid JSON: {exc}"
        ) from exc

    try:
        return GeminiReasoning(
            ctr_potential_score=float(parsed["ctr_potential_score"]),
            curiosity_gap_score=float(parsed["curiosity_gap_score"]),
            emotional_impact=str(parsed["emotional_impact"]),
            visual_storytelling_notes=str(parsed["visual_storytelling_notes"]),
            content_mismatch_detected=bool(parsed["content_mismatch_detected"]),
            mismatch_explanation=parsed.get("mismatch_explanation"),
            strengths=list(parsed.get("strengths", [])),
            weaknesses=list(parsed.get("weaknesses", [])),
            redesign_recommendations=list(parsed.get("redesign_recommendations", [])),
            elements_to_preserve=list(parsed.get("elements_to_preserve", [])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GeminiReasoningError(
            f"Gemini response did not match the expected schema: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run_stage(stage_name: str, func, *args, default):
    """
    Run a single CV stage, timing it and degrading to ``default`` on
    any :class:`ThumbnailIntelligenceError` subclass rather than
    propagating the failure.

    Args:
        stage_name: Human-readable name for logging.
        func:       The stage function to call.
        *args:      Positional arguments forwarded to ``func``.
        default:    Value returned (with ``duration_seconds`` and
                    ``engine_available=False`` applied where supported)
                    if the stage raises.

    Returns:
        A ``(result, duration_seconds, failure_reason)`` tuple. Fields
        that carry ``duration_seconds`` on the model are populated with
        the measured elapsed time; ``failure_reason`` is ``None`` on
        success.
    """
    start = time.monotonic()
    try:
        result = func(*args)
        duration = time.monotonic() - start
        if hasattr(result, "model_copy"):
            result = result.model_copy(update={"duration_seconds": duration})
        return result, duration, None
    except ThumbnailIntelligenceError as exc:
        duration = time.monotonic() - start
        logger.warning(
            "Stage '{stage}' degraded to default after {dur:.2f}s: {exc}",
            stage=stage_name,
            dur=duration,
            exc=exc,
        )
        degraded = default
        if hasattr(degraded, "model_copy"):
            update = {"duration_seconds": duration}
            if "engine_available" in type(degraded).model_fields:
                update["engine_available"] = False
            degraded = degraded.model_copy(update=update)
        return degraded, duration, str(exc)


def analyze_thumbnail(
    thumbnail_data: ThumbnailData,
    generate_reasoning_fn=None,
) -> ThumbnailIntelligence:
    """
    Run the complete Module 4 pipeline for a single creator's thumbnail.

    Pipeline:

    1. Load and validate the thumbnail image (hard failure — aborts the
       report if the image cannot be decoded at all).
    2. Run OCR, face analysis, object detection, and color analysis.
       Each stage independently degrades to a safe default on failure.
    3. Run composition analysis, which depends on the outputs of 2.
    4. Merge every finding with title/description/transcript/metadata
       (Stage 7).
    5. Send the merged context to Gemini for CTR/curiosity/mismatch
       reasoning. Degrades to ``reasoning=None`` on failure.
    6. Assemble and return the final :class:`ThumbnailIntelligence`.

    Args:
        thumbnail_data: Output of Module 3 — the downloaded thumbnail
            path plus its :class:`VideoMetadata`.
        generate_reasoning_fn: Optional override for the Gemini call,
            used by tests to inject a mock. Defaults to
            :func:`generate_reasoning` when ``None``.

    Returns:
        A :class:`ThumbnailIntelligence` report. ``status`` is
        ``"success"`` when every stage completed cleanly, ``"partial"``
        when one or more stages degraded but a report was still
        produced, or ``"error"`` when the image itself could not be
        loaded at all (in which case CV fields carry their safe
        defaults and ``error_message`` is populated).

    Raises:
        InvalidMetadataError: If ``thumbnail_data`` has neither a title
            nor a transcript to reason about — there is nothing
            meaningful for Gemini to evaluate the thumbnail against.
    """
    metadata = thumbnail_data.metadata
    video_id = metadata.video_id
    pipeline_start = time.monotonic()
    failure_reasons: list[str] = []

    logger.info("Starting thumbnail intelligence analysis for video_id={id}", id=video_id)

    if not (metadata.title and metadata.title.strip()) and not (
        metadata.transcript and metadata.transcript.strip()
    ):
        raise InvalidMetadataError(
            f"VideoMetadata for video_id={video_id!r} has neither a title "
            "nor a transcript — nothing to evaluate the thumbnail against"
        )

    # --- Stage 1: load image (hard failure) ---
    try:
        image = load_and_validate_image(Path(thumbnail_data.thumbnail_path))
    except ImageLoadError as exc:
        logger.error(
            "Thumbnail intelligence aborted for video_id={id}: {exc}",
            id=video_id,
            exc=exc,
        )
        now = datetime.now(timezone.utc).isoformat()
        return ThumbnailIntelligence(
            video_id=video_id,
            thumbnail_path=thumbnail_data.thumbnail_path,
            ocr=OCRResult(engine_available=False),
            faces=FaceAnalysis(engine_available=False),
            objects=[],
            colors=ColorProfile(),
            composition=CompositionAnalysis(),
            reasoning=None,
            status="error",
            partial_failure_reasons=[str(exc)],
            error_message=str(exc),
            total_duration_seconds=time.monotonic() - pipeline_start,
            analyzed_at=now,
        )

    # --- Stages 2-5: independent CV stages, each degrades on failure ---
    ocr, _, ocr_failure = _run_stage(
        "ocr", run_ocr, image, default=OCRResult(engine_available=False)
    )
    if ocr_failure:
        failure_reasons.append(f"ocr: {ocr_failure}")

    faces, _, face_failure = _run_stage(
        "face_analysis",
        run_face_analysis,
        image,
        default=FaceAnalysis(engine_available=False),
    )
    if face_failure:
        failure_reasons.append(f"face_analysis: {face_failure}")

    objects, _, object_failure = _run_stage(
        "object_detection", run_object_detection, image, default=[]
    )
    if object_failure:
        failure_reasons.append(f"object_detection: {object_failure}")

    colors, _, color_failure = _run_stage(
        "color_analysis", run_color_analysis, image, default=ColorProfile()
    )
    if color_failure:
        failure_reasons.append(f"color_analysis: {color_failure}")

    # --- Stage 6: composition (depends on 2-4) ---
    composition, _, composition_failure = _run_stage(
        "composition_analysis",
        run_composition_analysis,
        image,
        faces,
        ocr,
        objects,
        default=CompositionAnalysis(),
    )
    if composition_failure:
        failure_reasons.append(f"composition_analysis: {composition_failure}")

    # --- Stage 7: context merge ---
    context = _build_gemini_context(metadata, ocr, faces, objects, colors, composition)

    # --- AI reasoning ---
    if generate_reasoning_fn is None:
        generate_reasoning_fn = generate_reasoning

    reasoning: Optional[GeminiReasoning] = None
    reasoning_start = time.monotonic()
    try:
        reasoning = generate_reasoning_fn(context)
        reasoning_duration = time.monotonic() - reasoning_start
        reasoning = reasoning.model_copy(update={"duration_seconds": reasoning_duration})
    except ThumbnailIntelligenceError as exc:
        logger.warning(
            "Gemini reasoning degraded to None for video_id={id}: {exc}",
            id=video_id,
            exc=exc,
        )
        failure_reasons.append(f"gemini_reasoning: {exc}")

    total_duration = time.monotonic() - pipeline_start
    status = "success" if not failure_reasons else "partial"

    logger.info(
        "Thumbnail intelligence analysis complete for video_id={id}: "
        "status={status} ({dur:.2f}s)",
        id=video_id,
        status=status,
        dur=total_duration,
    )

    return ThumbnailIntelligence(
        video_id=video_id,
        thumbnail_path=thumbnail_data.thumbnail_path,
        ocr=ocr,
        faces=faces,
        objects=objects,
        colors=colors,
        composition=composition,
        reasoning=reasoning,
        status=status,
        partial_failure_reasons=failure_reasons,
        error_message=None,
        total_duration_seconds=total_duration,
        analyzed_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def _intelligence_path(video_id: str, analysis_dir: Path) -> Path:
    """
    Return the canonical local path for a saved intelligence report.

    Args:
        video_id:     The 11-character YouTube video ID.
        analysis_dir: Root directory for saved intelligence reports.

    Returns:
        ``analysis_dir / "{video_id}.json"``
    """
    filename = ANALYSIS_FILENAME_TEMPLATE.format(video_id=video_id)
    return analysis_dir / filename


def save_intelligence(
    intelligence: ThumbnailIntelligence,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
) -> None:
    """
    Persist a :class:`ThumbnailIntelligence` report to disk as JSON,
    atomically.

    The JSON is first written to a ``.tmp`` sibling file in the same
    directory, then moved into place with :meth:`Path.replace`, so a
    concurrent reader never sees a partial write.

    Args:
        intelligence: The report to persist.
        analysis_dir: Root directory for saved reports. Created
            automatically if it does not exist.

    Raises:
        IntelligenceCacheError: If the file cannot be written due to a
            permissions error or other OS failure.
    """
    target = _intelligence_path(intelligence.video_id, analysis_dir)
    tmp = target.with_suffix(".tmp")

    try:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(intelligence.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(target)
        logger.debug(
            "Saved thumbnail intelligence for video_id={id} -> {path}",
            id=intelligence.video_id,
            path=target,
        )
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        logger.error(
            "Failed to save thumbnail intelligence for video_id={id}: {exc}",
            id=intelligence.video_id,
            exc=exc,
        )
        raise IntelligenceCacheError(
            f"Could not write intelligence report to {target}: {exc}"
        ) from exc


def load_cached_intelligence(
    video_id: str,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
) -> Optional[ThumbnailIntelligence]:
    """
    Load a previously saved :class:`ThumbnailIntelligence` report, if one
    exists and can be parsed.

    Args:
        video_id:     The 11-character YouTube video ID.
        analysis_dir: Root directory for saved reports.

    Returns:
        The cached :class:`ThumbnailIntelligence`, or ``None`` if no
        cache file exists or the cached file is unreadable/corrupted
        (in which case the caller should simply re-run the analysis).
    """
    path = _intelligence_path(video_id, analysis_dir)
    if not path.exists():
        logger.debug("Intelligence cache miss for video_id={id}", id=video_id)
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cached = ThumbnailIntelligence.model_validate(raw)
        logger.debug("Intelligence cache hit for video_id={id}: {path}", id=video_id, path=path)
        return cached
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Cached intelligence report for video_id={id} is unreadable "
            "({reason}) — treating as cache miss",
            id=video_id,
            reason=exc,
        )
        return None
