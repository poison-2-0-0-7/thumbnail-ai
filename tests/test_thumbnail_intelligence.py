"""
test_thumbnail_intelligence.py
================================

Pytest suite for Module 4 (Thumbnail Intelligence Engine).

All heavy ML engines (EasyOCR, InsightFace, Ultralytics YOLO) and the
Gemini API are mocked with ``unittest.mock.patch`` so the entire suite
runs fully offline and deterministically, without downloading any model
weights or making any network call. File-system operations use
pytest's ``tmp_path`` fixture for complete test isolation.

Coverage targets:
    - :func:`load_and_validate_image`   missing, corrupted, blank, valid
    - :func:`run_ocr`                   text found, no text, low
                                         confidence filtering, engine
                                         failure
    - :func:`run_face_analysis`         no face, single face, multiple
                                         faces, low confidence filtering,
                                         engine failure
    - :func:`run_object_detection`      detections, empty, confidence
                                         filtering, engine failure
    - :func:`run_color_analysis`        warm/cool classification,
                                         dominant colors, failure
    - :func:`run_composition_analysis`  face subject, object subject, no
                                         subject, text overlap, failure
    - :func:`generate_reasoning`        valid JSON, markdown-fenced JSON,
                                         invalid JSON, missing schema key,
                                         transient failure exhausting
                                         retries
    - :func:`analyze_thumbnail`         full success, partial failure
                                         (each stage independently),
                                         missing image, invalid metadata,
                                         heavy text, multiple faces
    - :func:`save_intelligence` /
      :func:`load_cached_intelligence`  atomic write, cache hit, cache
                                         miss, corrupted cache
    - Exception hierarchy               all Module 4 exceptions share a
                                         common base
    - Config constants                  presence and correct types
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tenacity
from PIL import Image

# ---------------------------------------------------------------------------
# Path bootstrap — identical pattern used by test_thumbnail_downloader.py
# ---------------------------------------------------------------------------

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import (  # noqa: E402
    COLOR_PALETTE_SIZE,
    DEFAULT_ANALYSIS_DIR,
    FACE_MIN_CONFIDENCE,
    GEMINI_MAX_RETRY_ATTEMPTS,
    OCR_MIN_CONFIDENCE,
    YOLO_MIN_CONFIDENCE,
)
from models import (  # noqa: E402
    BoundingBox,
    ColorProfile,
    CompositionAnalysis,
    DetectedObject,
    FaceAnalysis,
    GeminiReasoning,
    OCRResult,
    ThumbnailData,
    ThumbnailIntelligence,
    VideoMetadata,
)
import thumbnail_intelligence as ti  # noqa: E402
from thumbnail_intelligence import (  # noqa: E402
    ColorAnalysisError,
    CompositionAnalysisError,
    FaceEngineError,
    GeminiReasoningError,
    ImageLoadError,
    IntelligenceCacheError,
    InvalidMetadataError,
    ObjectDetectionEngineError,
    OCREngineError,
    ThumbnailIntelligenceError,
    analyze_thumbnail,
    generate_reasoning,
    load_and_validate_image,
    load_cached_intelligence,
    run_color_analysis,
    run_composition_analysis,
    run_face_analysis,
    run_object_detection,
    run_ocr,
    save_intelligence,
)

VALID_VIDEO_ID = "abcdEFGH123"


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_solid_image_path(
    tmp_path: Path,
    color: tuple[int, int, int] = (120, 130, 140),
    size: tuple[int, int] = (200, 100),
    name: str = "thumb.jpg",
) -> Path:
    """Write a solid-color JPEG to ``tmp_path`` and return its path."""
    path = tmp_path / name
    Image.new("RGB", size, color=color).save(path, format="JPEG")
    return path


def _make_array(
    color: tuple[int, int, int] = (120, 130, 140), size: tuple[int, int] = (200, 100)
) -> np.ndarray:
    """Build an ``(H, W, 3)`` uint8 RGB array of a solid color."""
    width, height = size
    return np.full((height, width, 3), color, dtype=np.uint8)


def _make_metadata(
    video_id: str = VALID_VIDEO_ID,
    title: str = "I Tried This For 30 Days",
    transcript: str | None = "In this video I try a 30 day challenge and show results.",
    description: str | None = "A description",
) -> VideoMetadata:
    """Build a minimal VideoMetadata for testing."""
    return VideoMetadata(
        video_id=video_id,
        title=title,
        uploader="TestCreator",
        uploader_id="@testcreator",
        channel_id="UCxxxxxxxx",
        description=description,
        transcript=transcript,
    )


def _make_thumbnail_data(
    tmp_path: Path,
    metadata: VideoMetadata | None = None,
    thumbnail_path: Path | None = None,
) -> ThumbnailData:
    """Build a ThumbnailData with a real on-disk image and matching metadata."""
    metadata = metadata or _make_metadata()
    path = thumbnail_path or _make_solid_image_path(tmp_path)
    return ThumbnailData(metadata=metadata, thumbnail_path=str(path))


def _make_face_result(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    det_score: float = 0.9,
) -> MagicMock:
    """Build a fake InsightFace ``Face`` result object."""
    face = MagicMock()
    face.det_score = det_score
    face.bbox = np.array([x1, y1, x2, y2])
    face.kps = np.array([[x1 + 5, y1 + 5], [x2 - 5, y1 + 5], [(x1 + x2) / 2, y1 + 15]])
    face.pose = np.array([0.0, 5.0, 0.0])
    landmarks = np.zeros((106, 2))
    landmarks[0] = [x1, (y1 + y2) / 2]
    landmarks[32] = [x2, (y1 + y2) / 2]
    landmarks[52] = [(x1 + x2) / 2 - 10, y2 - 10]
    landmarks[61] = [(x1 + x2) / 2 + 10, y2 - 10]
    face.landmark_2d_106 = landmarks
    face.gender = 1
    face.age = 30
    return face


def _make_yolo_result(detections: list[tuple[str, float, tuple[float, float, float, float]]]):
    """Build a fake Ultralytics ``Results`` object with the given detections."""
    result = MagicMock()
    names = {i: label for i, (label, _, _) in enumerate(detections)}
    result.names = names

    boxes = MagicMock()
    box_items = []
    for class_id, (_, confidence, bbox) in enumerate(detections):
        box = MagicMock()
        box.conf = [confidence]
        box.cls = [class_id]
        box.xyxy = [list(bbox)]
        box_items.append(box)
    boxes.__iter__ = lambda self: iter(box_items)
    result.boxes = boxes if detections else None
    return result


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------


class TestConfigConstants:
    def test_ocr_min_confidence_is_float_in_unit_range(self) -> None:
        assert isinstance(OCR_MIN_CONFIDENCE, float)
        assert 0.0 <= OCR_MIN_CONFIDENCE <= 1.0

    def test_face_min_confidence_is_float_in_unit_range(self) -> None:
        assert isinstance(FACE_MIN_CONFIDENCE, float)
        assert 0.0 <= FACE_MIN_CONFIDENCE <= 1.0

    def test_yolo_min_confidence_is_float_in_unit_range(self) -> None:
        assert isinstance(YOLO_MIN_CONFIDENCE, float)
        assert 0.0 <= YOLO_MIN_CONFIDENCE <= 1.0

    def test_color_palette_size_is_positive_int(self) -> None:
        assert isinstance(COLOR_PALETTE_SIZE, int)
        assert COLOR_PALETTE_SIZE > 0

    def test_gemini_max_retry_attempts_is_positive_int(self) -> None:
        assert isinstance(GEMINI_MAX_RETRY_ATTEMPTS, int)
        assert GEMINI_MAX_RETRY_ATTEMPTS > 0

    def test_default_analysis_dir_is_path(self) -> None:
        assert isinstance(DEFAULT_ANALYSIS_DIR, Path)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exc_cls",
        [
            ImageLoadError,
            InvalidMetadataError,
            OCREngineError,
            FaceEngineError,
            ObjectDetectionEngineError,
            ColorAnalysisError,
            CompositionAnalysisError,
            GeminiReasoningError,
            IntelligenceCacheError,
        ],
    )
    def test_all_exceptions_share_base(self, exc_cls: type) -> None:
        assert issubclass(exc_cls, ThumbnailIntelligenceError)

    def test_base_exception_is_exception_subclass(self) -> None:
        assert issubclass(ThumbnailIntelligenceError, Exception)


# ---------------------------------------------------------------------------
# Lazily-loaded model singletons
# ---------------------------------------------------------------------------


class TestModelSingletons:
    def setup_method(self) -> None:
        ti.reset_model_singletons()

    def teardown_method(self) -> None:
        ti.reset_model_singletons()

    def test_resolve_device_defaults_to_cpu_without_torch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError on `import torch`
        device = ti._resolve_device()
        assert device == "cpu"

    def test_resolve_device_uses_cuda_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        device = ti._resolve_device()
        assert device == "cuda"

    def test_resolve_device_is_cached_across_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        first = ti._resolve_device()
        # Change the mock; cached result should not change on second call.
        fake_torch.cuda.is_available.return_value = False
        second = ti._resolve_device()

        assert first == second == "cuda"

    def test_get_ocr_reader_constructs_once_and_caches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_easyocr = MagicMock()
        fake_reader_instance = MagicMock()
        fake_easyocr.Reader.return_value = fake_reader_instance
        monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)
        monkeypatch.setitem(sys.modules, "torch", None)

        first = ti._get_ocr_reader()
        second = ti._get_ocr_reader()

        assert first is fake_reader_instance
        assert second is first
        assert fake_easyocr.Reader.call_count == 1

    def test_get_ocr_reader_raises_ocr_engine_error_on_init_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_easyocr = MagicMock()
        fake_easyocr.Reader.side_effect = RuntimeError("weights download failed")
        monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)
        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(OCREngineError):
            ti._get_ocr_reader()

    def test_get_face_app_constructs_once_and_caches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_insightface_app_module = MagicMock()
        fake_app_instance = MagicMock()
        fake_insightface_app_module.FaceAnalysis.return_value = fake_app_instance
        fake_insightface_pkg = MagicMock()
        monkeypatch.setitem(sys.modules, "insightface", fake_insightface_pkg)
        monkeypatch.setitem(sys.modules, "insightface.app", fake_insightface_app_module)
        monkeypatch.setitem(sys.modules, "torch", None)

        first = ti._get_face_app()
        second = ti._get_face_app()

        assert first is fake_app_instance
        assert second is first
        fake_app_instance.prepare.assert_called_once()
        assert fake_insightface_app_module.FaceAnalysis.call_count == 1

    def test_get_face_app_raises_face_engine_error_on_init_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_insightface_app_module = MagicMock()
        fake_insightface_app_module.FaceAnalysis.side_effect = RuntimeError("no weights")
        fake_insightface_pkg = MagicMock()
        monkeypatch.setitem(sys.modules, "insightface", fake_insightface_pkg)
        monkeypatch.setitem(sys.modules, "insightface.app", fake_insightface_app_module)
        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(FaceEngineError):
            ti._get_face_app()

    def test_get_yolo_model_constructs_once_and_caches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_ultralytics = MagicMock()
        fake_model_instance = MagicMock()
        fake_ultralytics.YOLO.return_value = fake_model_instance
        monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

        first = ti._get_yolo_model()
        second = ti._get_yolo_model()

        assert first is fake_model_instance
        assert second is first
        assert fake_ultralytics.YOLO.call_count == 1

    def test_get_yolo_model_raises_object_detection_error_on_init_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_ultralytics = MagicMock()
        fake_ultralytics.YOLO.side_effect = RuntimeError("bad checkpoint")
        monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

        with pytest.raises(ObjectDetectionEngineError):
            ti._get_yolo_model()

    def test_reset_model_singletons_clears_all_caches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_easyocr = MagicMock()
        monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)
        monkeypatch.setitem(sys.modules, "torch", None)

        ti._get_ocr_reader()
        assert fake_easyocr.Reader.call_count == 1

        ti.reset_model_singletons()
        ti._get_ocr_reader()

        assert fake_easyocr.Reader.call_count == 2


# ---------------------------------------------------------------------------
# Stage 1 — load_and_validate_image
# ---------------------------------------------------------------------------


class TestLoadAndValidateImage:
    def test_valid_image_loads_as_rgb_array(self, tmp_path: Path) -> None:
        path = _make_solid_image_path(tmp_path, size=(64, 32))
        array = load_and_validate_image(path)
        assert isinstance(array, np.ndarray)
        assert array.shape == (32, 64, 3)
        assert array.dtype == np.uint8

    def test_blank_white_image_loads_successfully(self, tmp_path: Path) -> None:
        path = _make_solid_image_path(tmp_path, color=(255, 255, 255))
        array = load_and_validate_image(path)
        assert array.shape[2] == 3
        assert np.all(array >= 250)

    def test_grayscale_image_converts_to_rgb(self, tmp_path: Path) -> None:
        path = tmp_path / "gray.jpg"
        Image.new("L", (50, 50), color=128).save(path, format="JPEG")
        array = load_and_validate_image(path)
        assert array.shape == (50, 50, 3)

    def test_missing_file_raises_image_load_error(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.jpg"
        with pytest.raises(ImageLoadError, match="does not exist"):
            load_and_validate_image(path)

    def test_corrupted_file_raises_image_load_error(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.jpg"
        path.write_bytes(b"NOT AN IMAGE" * 100)
        with pytest.raises(ImageLoadError):
            load_and_validate_image(path)

    def test_empty_file_raises_image_load_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
        with pytest.raises(ImageLoadError):
            load_and_validate_image(path)


# ---------------------------------------------------------------------------
# Stage 2 — run_ocr
# ---------------------------------------------------------------------------


class TestRunOCR:
    @patch("thumbnail_intelligence._get_ocr_reader")
    def test_detects_visible_text(self, mock_get_reader: MagicMock) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[10, 10], [50, 10], [50, 30], [10, 30]], "DAY 30", 0.95),
        ]
        mock_get_reader.return_value = mock_reader

        result = run_ocr(_make_array(size=(100, 50)))

        assert result.visible_text == "DAY 30"
        assert result.word_count == 2
        assert result.average_confidence == pytest.approx(0.95)
        assert result.engine_available is True
        assert len(result.text_regions) == 1

    @patch("thumbnail_intelligence._get_ocr_reader")
    def test_no_text_detected_returns_empty_result(self, mock_get_reader: MagicMock) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mock_get_reader.return_value = mock_reader

        result = run_ocr(_make_array())

        assert result.visible_text == ""
        assert result.word_count == 0
        assert result.text_regions == []
        assert result.average_confidence == 0.0

    @patch("thumbnail_intelligence._get_ocr_reader")
    def test_low_confidence_regions_are_filtered_out(self, mock_get_reader: MagicMock) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "noise", OCR_MIN_CONFIDENCE - 0.05),
            ([[20, 20], [60, 20], [60, 40], [20, 40]], "REAL TEXT", 0.9),
        ]
        mock_get_reader.return_value = mock_reader

        result = run_ocr(_make_array(size=(100, 50)))

        assert result.visible_text == "REAL TEXT"
        assert len(result.text_regions) == 1

    @patch("thumbnail_intelligence._get_ocr_reader")
    def test_heavy_text_thumbnail_reports_high_coverage(
        self, mock_get_reader: MagicMock
    ) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 50], [0, 50]], "HUGE HEADLINE TEXT", 0.9),
        ]
        mock_get_reader.return_value = mock_reader

        result = run_ocr(_make_array(size=(100, 50)))

        assert result.text_coverage_ratio == pytest.approx(1.0, rel=0.05)

    @patch("thumbnail_intelligence._get_ocr_reader")
    def test_engine_init_failure_raises_ocr_engine_error(
        self, mock_get_reader: MagicMock
    ) -> None:
        mock_get_reader.side_effect = OCREngineError("could not init EasyOCR")
        with pytest.raises(OCREngineError):
            run_ocr(_make_array())

    @patch("thumbnail_intelligence._get_ocr_reader")
    def test_inference_failure_raises_ocr_engine_error(
        self, mock_get_reader: MagicMock
    ) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = RuntimeError("CUDA out of memory")
        mock_get_reader.return_value = mock_reader
        with pytest.raises(OCREngineError, match="EasyOCR inference failed"):
            run_ocr(_make_array())


# ---------------------------------------------------------------------------
# Stage 3 — run_face_analysis
# ---------------------------------------------------------------------------


class TestRunFaceAnalysis:
    @patch("thumbnail_intelligence._get_face_app")
    def test_no_faces_detected(self, mock_get_app: MagicMock) -> None:
        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_get_app.return_value = mock_app

        result = run_face_analysis(_make_array())

        assert result.face_count == 0
        assert result.has_face is False
        assert result.faces == []

    @patch("thumbnail_intelligence._get_face_app")
    def test_single_face_marked_as_largest(self, mock_get_app: MagicMock) -> None:
        mock_app = MagicMock()
        mock_app.get.return_value = [_make_face_result(10, 10, 60, 80, det_score=0.9)]
        mock_get_app.return_value = mock_app

        result = run_face_analysis(_make_array(size=(100, 100)))

        assert result.face_count == 1
        assert result.has_face is True
        assert result.faces[0].is_largest is True
        assert result.faces[0].detection_confidence == pytest.approx(0.9)

    @patch("thumbnail_intelligence._get_face_app")
    def test_multiple_faces_sorted_largest_first(self, mock_get_app: MagicMock) -> None:
        small_face = _make_face_result(5, 5, 20, 20, det_score=0.8)
        large_face = _make_face_result(10, 10, 90, 90, det_score=0.85)
        mock_app = MagicMock()
        mock_app.get.return_value = [small_face, large_face]
        mock_get_app.return_value = mock_app

        result = run_face_analysis(_make_array(size=(100, 100)))

        assert result.face_count == 2
        assert result.faces[0].is_largest is True
        assert result.faces[1].is_largest is False
        # Largest face's bbox area should exceed the second's.
        first_area = (result.faces[0].bbox.x_max - result.faces[0].bbox.x_min) * (
            result.faces[0].bbox.y_max - result.faces[0].bbox.y_min
        )
        second_area = (result.faces[1].bbox.x_max - result.faces[1].bbox.x_min) * (
            result.faces[1].bbox.y_max - result.faces[1].bbox.y_min
        )
        assert first_area >= second_area

    @patch("thumbnail_intelligence._get_face_app")
    def test_low_confidence_faces_filtered_out(self, mock_get_app: MagicMock) -> None:
        mock_app = MagicMock()
        mock_app.get.return_value = [
            _make_face_result(10, 10, 60, 80, det_score=FACE_MIN_CONFIDENCE - 0.1)
        ]
        mock_get_app.return_value = mock_app

        result = run_face_analysis(_make_array(size=(100, 100)))

        assert result.face_count == 0
        assert result.has_face is False

    @patch("thumbnail_intelligence._get_face_app")
    def test_engine_init_failure_raises_face_engine_error(
        self, mock_get_app: MagicMock
    ) -> None:
        mock_get_app.side_effect = FaceEngineError("could not init InsightFace")
        with pytest.raises(FaceEngineError):
            run_face_analysis(_make_array())

    @patch("thumbnail_intelligence._get_face_app")
    def test_inference_failure_raises_face_engine_error(
        self, mock_get_app: MagicMock
    ) -> None:
        mock_app = MagicMock()
        mock_app.get.side_effect = RuntimeError("model crashed")
        mock_get_app.return_value = mock_app
        with pytest.raises(FaceEngineError, match="InsightFace inference failed"):
            run_face_analysis(_make_array())


# ---------------------------------------------------------------------------
# Stage 4 — run_object_detection
# ---------------------------------------------------------------------------


class TestRunObjectDetection:
    @patch("thumbnail_intelligence._get_yolo_model")
    def test_detects_objects_above_threshold(self, mock_get_model: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            _make_yolo_result([("phone", 0.8, (10, 10, 50, 50))])
        ]
        mock_get_model.return_value = mock_model

        objects = run_object_detection(_make_array(size=(100, 100)))

        assert len(objects) == 1
        assert objects[0].label == "phone"
        assert objects[0].confidence == pytest.approx(0.8)

    @patch("thumbnail_intelligence._get_yolo_model")
    def test_no_detections_returns_empty_list(self, mock_get_model: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [_make_yolo_result([])]
        mock_get_model.return_value = mock_model

        objects = run_object_detection(_make_array())

        assert objects == []

    @patch("thumbnail_intelligence._get_yolo_model")
    def test_low_confidence_detections_filtered_out(self, mock_get_model: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            _make_yolo_result([("car", YOLO_MIN_CONFIDENCE - 0.1, (0, 0, 10, 10))])
        ]
        mock_get_model.return_value = mock_model

        objects = run_object_detection(_make_array())

        assert objects == []

    @patch("thumbnail_intelligence._get_yolo_model")
    def test_multiple_detections_sorted_by_confidence(
        self, mock_get_model: MagicMock
    ) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            _make_yolo_result(
                [
                    ("money", 0.55, (0, 0, 10, 10)),
                    ("car", 0.9, (20, 20, 40, 40)),
                ]
            )
        ]
        mock_get_model.return_value = mock_model

        objects = run_object_detection(_make_array())

        assert [obj.label for obj in objects] == ["car", "money"]

    @patch("thumbnail_intelligence._get_yolo_model")
    def test_engine_init_failure_raises_object_detection_error(
        self, mock_get_model: MagicMock
    ) -> None:
        mock_get_model.side_effect = ObjectDetectionEngineError("could not init YOLO")
        with pytest.raises(ObjectDetectionEngineError):
            run_object_detection(_make_array())

    @patch("thumbnail_intelligence._get_yolo_model")
    def test_inference_failure_raises_object_detection_error(
        self, mock_get_model: MagicMock
    ) -> None:
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("inference crashed")
        mock_get_model.return_value = mock_model
        with pytest.raises(ObjectDetectionEngineError, match="YOLO inference failed"):
            run_object_detection(_make_array())


# ---------------------------------------------------------------------------
# Stage 5 — run_color_analysis
# ---------------------------------------------------------------------------


class TestRunColorAnalysis:
    def test_warm_color_classified_as_warm(self) -> None:
        image = _make_array(color=(220, 80, 30))
        profile = run_color_analysis(image)
        assert profile.warm_or_cool == "warm"
        assert len(profile.dominant_colors) >= 1
        assert profile.dominant_colors[0].startswith("#")

    def test_cool_color_classified_as_cool(self) -> None:
        image = _make_array(color=(30, 80, 220))
        profile = run_color_analysis(image)
        assert profile.warm_or_cool == "cool"

    def test_gray_color_classified_as_neutral(self) -> None:
        image = _make_array(color=(128, 128, 128))
        profile = run_color_analysis(image)
        assert profile.warm_or_cool == "neutral"

    def test_solid_color_has_low_contrast(self) -> None:
        image = _make_array(color=(100, 100, 100))
        profile = run_color_analysis(image)
        assert profile.contrast == pytest.approx(0.0, abs=0.01)

    def test_brightness_in_unit_range(self) -> None:
        image = _make_array(color=(255, 255, 255))
        profile = run_color_analysis(image)
        assert 0.9 <= profile.brightness <= 1.0

    def test_dominant_colors_do_not_exceed_palette_size(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(50, 50, 3), dtype=np.uint8)
        profile = run_color_analysis(image)
        assert len(profile.dominant_colors) <= COLOR_PALETTE_SIZE

    def test_empty_array_raises_color_analysis_error(self) -> None:
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ColorAnalysisError):
            run_color_analysis(empty)


# ---------------------------------------------------------------------------
# Stage 6 — run_composition_analysis
# ---------------------------------------------------------------------------


class TestRunCompositionAnalysis:
    def test_no_subject_reports_none_detected(self) -> None:
        image = _make_array()
        result = run_composition_analysis(image, FaceAnalysis(), OCRResult(), [])
        assert result.subject_placement == "none-detected"
        assert result.rule_of_thirds_score == 0.0

    def test_centered_face_drives_subject_placement(self) -> None:
        image = _make_array(size=(100, 100))
        face_bbox = BoundingBox(x_min=0.4, y_min=0.3, x_max=0.6, y_max=0.7)
        faces = FaceAnalysis(
            face_count=1,
            has_face=True,
            faces=[
                ti.FaceDetail(
                    bbox=face_bbox,
                    detection_confidence=0.9,
                    is_largest=True,
                    position_label="center",
                )
            ],
        )
        result = run_composition_analysis(image, faces, OCRResult(), [])
        assert result.subject_placement == "center"

    def test_object_used_as_subject_when_no_face(self) -> None:
        image = _make_array(size=(100, 100))
        objects = [
            DetectedObject(
                label="car",
                confidence=0.8,
                bbox=BoundingBox(x_min=0.05, y_min=0.05, x_max=0.3, y_max=0.3),
            )
        ]
        result = run_composition_analysis(image, FaceAnalysis(), OCRResult(), objects)
        assert result.subject_placement == "left-third"

    def test_text_overlapping_subject_detected(self) -> None:
        image = _make_array(size=(100, 100))
        face_bbox = BoundingBox(x_min=0.2, y_min=0.2, x_max=0.5, y_max=0.6)
        faces = FaceAnalysis(
            face_count=1,
            has_face=True,
            faces=[ti.FaceDetail(bbox=face_bbox, detection_confidence=0.9, is_largest=True)],
        )
        overlapping_text = OCRResult(
            visible_text="TEXT",
            word_count=1,
            text_regions=[
                ti.TextRegion(
                    text="TEXT",
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.25, y_min=0.25, x_max=0.45, y_max=0.35),
                )
            ],
        )
        result = run_composition_analysis(image, faces, overlapping_text, [])
        assert result.text_overlaps_subject is True

    def test_non_overlapping_text_not_flagged(self) -> None:
        image = _make_array(size=(100, 100))
        face_bbox = BoundingBox(x_min=0.6, y_min=0.6, x_max=0.9, y_max=0.9)
        faces = FaceAnalysis(
            face_count=1,
            has_face=True,
            faces=[ti.FaceDetail(bbox=face_bbox, detection_confidence=0.9, is_largest=True)],
        )
        far_text = OCRResult(
            visible_text="TEXT",
            word_count=1,
            text_regions=[
                ti.TextRegion(
                    text="TEXT",
                    confidence=0.9,
                    bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=0.1, y_max=0.1),
                )
            ],
        )
        result = run_composition_analysis(image, faces, far_text, [])
        assert result.text_overlaps_subject is False

    def test_symmetric_image_scores_high_symmetry(self) -> None:
        # A solid-color image is perfectly symmetric under horizontal mirroring.
        image = _make_array(color=(50, 60, 70), size=(100, 100))
        result = run_composition_analysis(image, FaceAnalysis(), OCRResult(), [])
        assert result.symmetry_score == pytest.approx(1.0, abs=0.01)

    def test_failure_raises_composition_analysis_error(self) -> None:
        with pytest.raises(CompositionAnalysisError):
            run_composition_analysis(None, FaceAnalysis(), OCRResult(), [])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# generate_reasoning / _call_gemini_api
# ---------------------------------------------------------------------------


_VALID_GEMINI_JSON = json.dumps(
    {
        "ctr_potential_score": 0.72,
        "curiosity_gap_score": 0.65,
        "emotional_impact": "curiosity",
        "visual_storytelling_notes": "Bold color and bright text suggest energy.",
        "content_mismatch_detected": False,
        "mismatch_explanation": None,
        "strengths": ["Bright color palette"],
        "weaknesses": ["No human face"],
        "redesign_recommendations": ["Add a reacting face"],
        "elements_to_preserve": ["Color palette"],
    }
)


class TestGenerateReasoning:
    @patch("thumbnail_intelligence._call_gemini_api")
    def test_parses_valid_json_response(self, mock_call: MagicMock) -> None:
        mock_call.return_value = _VALID_GEMINI_JSON

        result = generate_reasoning({"video": {}, "thumbnail_analysis": {}})

        assert isinstance(result, GeminiReasoning)
        assert result.ctr_potential_score == pytest.approx(0.72)
        assert result.content_mismatch_detected is False
        assert result.strengths == ["Bright color palette"]

    @patch("thumbnail_intelligence._call_gemini_api")
    def test_strips_markdown_fences(self, mock_call: MagicMock) -> None:
        mock_call.return_value = f"```json\n{_VALID_GEMINI_JSON}\n```"

        result = generate_reasoning({"video": {}, "thumbnail_analysis": {}})

        assert result.ctr_potential_score == pytest.approx(0.72)

    @patch("thumbnail_intelligence._call_gemini_api")
    def test_invalid_json_raises_gemini_reasoning_error(self, mock_call: MagicMock) -> None:
        mock_call.return_value = "this is not json at all"
        with pytest.raises(GeminiReasoningError, match="not valid JSON"):
            generate_reasoning({"video": {}, "thumbnail_analysis": {}})

    @patch("thumbnail_intelligence._call_gemini_api")
    def test_missing_required_key_raises_gemini_reasoning_error(
        self, mock_call: MagicMock
    ) -> None:
        incomplete = json.dumps({"ctr_potential_score": 0.5})
        mock_call.return_value = incomplete
        with pytest.raises(GeminiReasoningError, match="expected schema"):
            generate_reasoning({"video": {}, "thumbnail_analysis": {}})

    @patch("thumbnail_intelligence._call_gemini_api")
    def test_transient_failure_propagates_as_gemini_reasoning_error(
        self, mock_call: MagicMock
    ) -> None:
        mock_call.side_effect = ti._GeminiTransientError("network blip")
        with pytest.raises(GeminiReasoningError, match="failed after"):
            generate_reasoning({"video": {}, "thumbnail_analysis": {}})


class TestCallGeminiApi:
    def test_missing_api_key_raises_gemini_reasoning_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(GeminiReasoningError, match="No Gemini API key"):
            ti._call_gemini_api.__wrapped__({"video": {}})

    def test_retries_transient_failure_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

        fake_genai = MagicMock()
        fake_response = MagicMock()
        fake_response.text = _VALID_GEMINI_JSON
        fake_model = MagicMock()
        fake_model.generate_content.side_effect = [
            RuntimeError("temporary hiccup"),
            fake_response,
        ]
        fake_genai.GenerativeModel.return_value = fake_model

        # Speed up the real tenacity retry loop for this test only.
        original_wait = ti._call_gemini_api.retry.wait
        ti._call_gemini_api.retry.wait = tenacity.wait_none()
        try:
            with patch.dict(sys.modules, {"google.generativeai": fake_genai}):
                result_text = ti._call_gemini_api({"video": {}})
        finally:
            ti._call_gemini_api.retry.wait = original_wait

        assert result_text == _VALID_GEMINI_JSON
        assert fake_model.generate_content.call_count == 2

    def test_exhausts_retries_and_raises_transient_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

        fake_genai = MagicMock()
        fake_model = MagicMock()
        fake_model.generate_content.side_effect = RuntimeError("always fails")
        fake_genai.GenerativeModel.return_value = fake_model

        original_wait = ti._call_gemini_api.retry.wait
        ti._call_gemini_api.retry.wait = tenacity.wait_none()
        try:
            with patch.dict(sys.modules, {"google.generativeai": fake_genai}):
                with pytest.raises(ti._GeminiTransientError):
                    ti._call_gemini_api({"video": {}})
        finally:
            ti._call_gemini_api.retry.wait = original_wait

        assert fake_model.generate_content.call_count == GEMINI_MAX_RETRY_ATTEMPTS


# ---------------------------------------------------------------------------
# analyze_thumbnail (full orchestration)
# ---------------------------------------------------------------------------


def _patch_all_cv_stages(
    ocr: OCRResult | None = None,
    faces: FaceAnalysis | None = None,
    objects: list[DetectedObject] | None = None,
    colors: ColorProfile | None = None,
    composition: CompositionAnalysis | None = None,
    ocr_side_effect=None,
    faces_side_effect=None,
    objects_side_effect=None,
    colors_side_effect=None,
    composition_side_effect=None,
):
    """Return a dict of patch targets → kwargs for patching every CV stage at once."""
    return dict(
        run_ocr=MagicMock(
            return_value=ocr if ocr_side_effect is None else None,
            side_effect=ocr_side_effect,
        )
        if ocr_side_effect is not None
        else MagicMock(return_value=ocr or OCRResult()),
        run_face_analysis=MagicMock(side_effect=faces_side_effect)
        if faces_side_effect is not None
        else MagicMock(return_value=faces or FaceAnalysis()),
        run_object_detection=MagicMock(side_effect=objects_side_effect)
        if objects_side_effect is not None
        else MagicMock(return_value=objects if objects is not None else []),
        run_color_analysis=MagicMock(side_effect=colors_side_effect)
        if colors_side_effect is not None
        else MagicMock(return_value=colors or ColorProfile()),
        run_composition_analysis=MagicMock(side_effect=composition_side_effect)
        if composition_side_effect is not None
        else MagicMock(return_value=composition or CompositionAnalysis()),
    )


class TestAnalyzeThumbnail:
    def test_full_success(self, tmp_path: Path) -> None:
        td = _make_thumbnail_data(tmp_path)

        def fake_reasoning(context: dict) -> GeminiReasoning:
            assert context["video"]["transcript_available"] is True
            return GeminiReasoning(
                ctr_potential_score=0.7,
                curiosity_gap_score=0.6,
                emotional_impact="curiosity",
                visual_storytelling_notes="notes",
                content_mismatch_detected=False,
            )

        patches = _patch_all_cv_stages(ocr=OCRResult(visible_text="DAY 30", word_count=2))
        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(td, generate_reasoning_fn=fake_reasoning)

        assert result.status == "success"
        assert result.partial_failure_reasons == []
        assert result.reasoning is not None
        assert result.reasoning.ctr_potential_score == pytest.approx(0.7)
        assert result.ocr.visible_text == "DAY 30"

    def test_missing_thumbnail_file_returns_error_status(self, tmp_path: Path) -> None:
        metadata = _make_metadata()
        td = ThumbnailData(metadata=metadata, thumbnail_path=str(tmp_path / "missing.jpg"))

        result = analyze_thumbnail(td)

        assert result.status == "error"
        assert result.error_message is not None
        assert result.reasoning is None

    def test_invalid_metadata_raises(self, tmp_path: Path) -> None:
        metadata = _make_metadata(title="", transcript=None)
        td = _make_thumbnail_data(tmp_path, metadata=metadata)

        with pytest.raises(InvalidMetadataError):
            analyze_thumbnail(td)

    def test_ocr_failure_degrades_to_partial(self, tmp_path: Path) -> None:
        td = _make_thumbnail_data(tmp_path)
        patches = _patch_all_cv_stages(ocr_side_effect=OCREngineError("ocr boom"))

        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(
                td, generate_reasoning_fn=lambda ctx: GeminiReasoning(
                    ctr_potential_score=0.5,
                    curiosity_gap_score=0.5,
                    emotional_impact="neutral",
                    visual_storytelling_notes="n/a",
                    content_mismatch_detected=False,
                )
            )

        assert result.status == "partial"
        assert result.ocr.engine_available is False
        assert any("ocr" in reason for reason in result.partial_failure_reasons)

    def test_face_failure_degrades_to_partial(self, tmp_path: Path) -> None:
        td = _make_thumbnail_data(tmp_path)
        patches = _patch_all_cv_stages(faces_side_effect=FaceEngineError("face boom"))

        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(
                td, generate_reasoning_fn=lambda ctx: GeminiReasoning(
                    ctr_potential_score=0.5,
                    curiosity_gap_score=0.5,
                    emotional_impact="neutral",
                    visual_storytelling_notes="n/a",
                    content_mismatch_detected=False,
                )
            )

        assert result.status == "partial"
        assert result.faces.engine_available is False

    def test_object_detection_failure_degrades_to_partial(self, tmp_path: Path) -> None:
        td = _make_thumbnail_data(tmp_path)
        patches = _patch_all_cv_stages(
            objects_side_effect=ObjectDetectionEngineError("yolo boom")
        )

        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(
                td, generate_reasoning_fn=lambda ctx: GeminiReasoning(
                    ctr_potential_score=0.5,
                    curiosity_gap_score=0.5,
                    emotional_impact="neutral",
                    visual_storytelling_notes="n/a",
                    content_mismatch_detected=False,
                )
            )

        assert result.status == "partial"
        assert result.objects == []

    def test_color_failure_degrades_to_partial(self, tmp_path: Path) -> None:
        td = _make_thumbnail_data(tmp_path)
        patches = _patch_all_cv_stages(colors_side_effect=ColorAnalysisError("color boom"))

        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(
                td, generate_reasoning_fn=lambda ctx: GeminiReasoning(
                    ctr_potential_score=0.5,
                    curiosity_gap_score=0.5,
                    emotional_impact="neutral",
                    visual_storytelling_notes="n/a",
                    content_mismatch_detected=False,
                )
            )

        assert result.status == "partial"

    def test_composition_failure_degrades_to_partial(self, tmp_path: Path) -> None:
        td = _make_thumbnail_data(tmp_path)
        patches = _patch_all_cv_stages(
            composition_side_effect=CompositionAnalysisError("composition boom")
        )

        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(
                td, generate_reasoning_fn=lambda ctx: GeminiReasoning(
                    ctr_potential_score=0.5,
                    curiosity_gap_score=0.5,
                    emotional_impact="neutral",
                    visual_storytelling_notes="n/a",
                    content_mismatch_detected=False,
                )
            )

        assert result.status == "partial"

    def test_gemini_failure_degrades_reasoning_to_none(self, tmp_path: Path) -> None:
        td = _make_thumbnail_data(tmp_path)
        patches = _patch_all_cv_stages()

        def failing_reasoning(context: dict) -> GeminiReasoning:
            raise GeminiReasoningError("no key configured")

        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(td, generate_reasoning_fn=failing_reasoning)

        assert result.status == "partial"
        assert result.reasoning is None
        assert any("gemini_reasoning" in reason for reason in result.partial_failure_reasons)

    def test_thumbnail_without_face_reports_zero_faces(self, tmp_path: Path) -> None:
        td = _make_thumbnail_data(tmp_path)
        patches = _patch_all_cv_stages(faces=FaceAnalysis(face_count=0, has_face=False))

        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(
                td, generate_reasoning_fn=lambda ctx: GeminiReasoning(
                    ctr_potential_score=0.5,
                    curiosity_gap_score=0.5,
                    emotional_impact="neutral",
                    visual_storytelling_notes="n/a",
                    content_mismatch_detected=False,
                )
            )

        assert result.faces.has_face is False
        assert result.status == "success"

    def test_thumbnail_with_multiple_faces_preserved_in_report(self, tmp_path: Path) -> None:
        td = _make_thumbnail_data(tmp_path)
        two_faces = FaceAnalysis(
            face_count=2,
            has_face=True,
            faces=[
                ti.FaceDetail(
                    bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.4),
                    detection_confidence=0.9,
                    is_largest=True,
                ),
                ti.FaceDetail(
                    bbox=BoundingBox(x_min=0.6, y_min=0.1, x_max=0.8, y_max=0.35),
                    detection_confidence=0.85,
                    is_largest=False,
                ),
            ],
        )
        patches = _patch_all_cv_stages(faces=two_faces)

        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(
                td, generate_reasoning_fn=lambda ctx: GeminiReasoning(
                    ctr_potential_score=0.5,
                    curiosity_gap_score=0.5,
                    emotional_impact="neutral",
                    visual_storytelling_notes="n/a",
                    content_mismatch_detected=False,
                )
            )

        assert result.faces.face_count == 2
        assert result.status == "success"

    def test_thumbnail_without_text_reports_empty_ocr(self, tmp_path: Path) -> None:
        td = _make_thumbnail_data(tmp_path)
        patches = _patch_all_cv_stages(ocr=OCRResult())

        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(
                td, generate_reasoning_fn=lambda ctx: GeminiReasoning(
                    ctr_potential_score=0.5,
                    curiosity_gap_score=0.5,
                    emotional_impact="neutral",
                    visual_storytelling_notes="n/a",
                    content_mismatch_detected=False,
                )
            )

        assert result.ocr.visible_text == ""
        assert result.status == "success"

    def test_heavy_text_thumbnail_preserved_in_report(self, tmp_path: Path) -> None:
        td = _make_thumbnail_data(tmp_path)
        heavy_text = OCRResult(
            visible_text="THIS IS A HUGE AMOUNT OF HEADLINE TEXT ON THE THUMBNAIL",
            word_count=11,
            text_coverage_ratio=0.85,
            average_confidence=0.8,
        )
        patches = _patch_all_cv_stages(ocr=heavy_text)

        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(
                td, generate_reasoning_fn=lambda ctx: GeminiReasoning(
                    ctr_potential_score=0.5,
                    curiosity_gap_score=0.5,
                    emotional_impact="neutral",
                    visual_storytelling_notes="n/a",
                    content_mismatch_detected=False,
                )
            )

        assert result.ocr.text_coverage_ratio == pytest.approx(0.85)
        assert result.status == "success"

    def test_invalid_metadata_no_title_but_has_transcript_is_allowed(
        self, tmp_path: Path
    ) -> None:
        metadata = _make_metadata(title="", transcript="Some transcript content exists.")
        td = _make_thumbnail_data(tmp_path, metadata=metadata)
        patches = _patch_all_cv_stages()

        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(
                td, generate_reasoning_fn=lambda ctx: GeminiReasoning(
                    ctr_potential_score=0.5,
                    curiosity_gap_score=0.5,
                    emotional_impact="neutral",
                    visual_storytelling_notes="n/a",
                    content_mismatch_detected=False,
                )
            )

        assert result.status == "success"

    def test_total_duration_is_recorded(self, tmp_path: Path) -> None:
        td = _make_thumbnail_data(tmp_path)
        patches = _patch_all_cv_stages()

        with patch.multiple("thumbnail_intelligence", **patches):
            result = analyze_thumbnail(
                td, generate_reasoning_fn=lambda ctx: GeminiReasoning(
                    ctr_potential_score=0.5,
                    curiosity_gap_score=0.5,
                    emotional_impact="neutral",
                    visual_storytelling_notes="n/a",
                    content_mismatch_detected=False,
                )
            )

        assert result.total_duration_seconds >= 0.0
        assert result.reasoning.duration_seconds >= 0.0
        assert result.ocr.duration_seconds >= 0.0
        assert result.faces.duration_seconds >= 0.0
        assert result.colors.duration_seconds >= 0.0
        assert result.composition.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# save_intelligence / load_cached_intelligence
# ---------------------------------------------------------------------------


def _make_intelligence(video_id: str = VALID_VIDEO_ID) -> ThumbnailIntelligence:
    """Build a minimal, valid ThumbnailIntelligence for persistence tests."""
    return ThumbnailIntelligence(
        video_id=video_id,
        thumbnail_path="/tmp/does_not_matter.jpg",
        ocr=OCRResult(),
        faces=FaceAnalysis(),
        objects=[],
        colors=ColorProfile(),
        composition=CompositionAnalysis(),
        reasoning=None,
        status="success",
        analyzed_at="2026-07-19T00:00:00+00:00",
    )


class TestSaveIntelligence:
    def test_creates_json_file_with_correct_content(self, tmp_path: Path) -> None:
        intelligence = _make_intelligence()
        save_intelligence(intelligence, analysis_dir=tmp_path)

        target = tmp_path / f"{intelligence.video_id}.json"
        assert target.exists()
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded["video_id"] == intelligence.video_id

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "nested" / "analysis"
        intelligence = _make_intelligence()
        save_intelligence(intelligence, analysis_dir=nested_dir)
        assert (nested_dir / f"{intelligence.video_id}.json").exists()

    def test_overwrites_existing_report(self, tmp_path: Path) -> None:
        intelligence = _make_intelligence()
        save_intelligence(intelligence, analysis_dir=tmp_path)

        updated = intelligence.model_copy(update={"status": "partial"})
        save_intelligence(updated, analysis_dir=tmp_path)

        loaded = json.loads(
            (tmp_path / f"{intelligence.video_id}.json").read_text(encoding="utf-8")
        )
        assert loaded["status"] == "partial"

    def test_os_error_raises_intelligence_cache_error(self, tmp_path: Path) -> None:
        intelligence = _make_intelligence()
        blocking_file = tmp_path / "blocked"
        blocking_file.write_text("not a directory")

        with pytest.raises(IntelligenceCacheError):
            save_intelligence(intelligence, analysis_dir=blocking_file / "analysis")


class TestLoadCachedIntelligence:
    def test_cache_hit_returns_matching_report(self, tmp_path: Path) -> None:
        intelligence = _make_intelligence()
        save_intelligence(intelligence, analysis_dir=tmp_path)

        loaded = load_cached_intelligence(intelligence.video_id, analysis_dir=tmp_path)

        assert loaded == intelligence

    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        result = load_cached_intelligence("doesNotExist1", analysis_dir=tmp_path)
        assert result is None

    def test_corrupted_cache_file_returns_none(self, tmp_path: Path) -> None:
        video_id = "corruptID12"
        (tmp_path / f"{video_id}.json").write_text("{not valid json", encoding="utf-8")

        result = load_cached_intelligence(video_id, analysis_dir=tmp_path)

        assert result is None

    def test_cache_file_missing_required_field_returns_none(self, tmp_path: Path) -> None:
        video_id = "missingFld1"
        (tmp_path / f"{video_id}.json").write_text(
            json.dumps({"video_id": video_id}), encoding="utf-8"
        )

        result = load_cached_intelligence(video_id, analysis_dir=tmp_path)

        assert result is None


# ---------------------------------------------------------------------------
# Integration tests (real Gemini API) — separated and skipped by default
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGeminiLiveIntegration:
    @pytest.mark.skipif(
        "GEMINI_API_KEY" not in __import__("os").environ,
        reason="requires a real GEMINI_API_KEY in the environment",
    )
    def test_generate_reasoning_against_live_api(self) -> None:
        context = {
            "video": {
                "title": "I Tried This For 30 Days",
                "description": "",
                "transcript": "In this video I try a 30 day challenge.",
                "transcript_available": True,
                "uploader": "TestCreator",
                "channel_id": "UCxxxx",
                "categories": [],
                "tags": [],
                "view_count": None,
                "like_count": None,
            },
            "thumbnail_analysis": {
                "ocr": OCRResult().model_dump(),
                "faces": FaceAnalysis().model_dump(),
                "objects": [],
                "colors": ColorProfile().model_dump(),
                "composition": CompositionAnalysis().model_dump(),
            },
        }
        result = generate_reasoning(context)
        assert isinstance(result, GeminiReasoning)
