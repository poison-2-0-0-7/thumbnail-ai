"""Tests for the Vision Stack GroundingDINO wrapper."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import load_vision_stack_config  # noqa: E402
from vision_stack.exceptions import VisionStackResourceError  # noqa: E402
from vision_stack.grounding_dino import (  # noqa: E402
    GroundingDINOWrapper,
    _GroundingDINOOutputParser,
)
from vision_stack.grounding_dino_exceptions import (  # noqa: E402
    GroundingDINOOutOfMemoryError,
    GroundingDINOLoadError,
)
from vision_stack.loader import build_registry  # noqa: E402
from vision_stack.models import (  # noqa: E402
    GroundingDINODetection,
    RegisteredVisionModel,
    VisionModelLifecycleState,
    VisionModelPrecision,
)
from vision_stack.resources import GPUResourceManager  # noqa: E402


class _FakeInferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


class _FakeCuda:
    class OutOfMemoryError(RuntimeError):
        """Fake torch CUDA OOM type."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.empty_cache = MagicMock()

    def is_available(self) -> bool:
        return self.available


class _FakeTorch:
    def __init__(self, *, cuda_available: bool = True) -> None:
        self.cuda = _FakeCuda(available=cuda_available)
        self.load = MagicMock(return_value={"model": {}})

    def inference_mode(self) -> _FakeInferenceMode:
        return _FakeInferenceMode()

    def no_grad(self) -> _FakeInferenceMode:
        return _FakeInferenceMode()


class _FakeModel:
    def __init__(self, output: tuple[object, object, object] | None = None) -> None:
        self.output = output or (
            [[0.5, 0.5, 0.25, 0.25]],
            [0.9],
            ["Person"],
        )
        self.eval = MagicMock(return_value=self)
        self.half = MagicMock(return_value=self)
        self.to = MagicMock(return_value=self)
        self.predict = MagicMock(return_value=self.output)


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> _FakeTorch:
    torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.fixture
def checkpoint_root(tmp_path: Path) -> Path:
    (tmp_path / "groundingdino_swint_ogc.pth").write_bytes(b"checkpoint fixture")
    (tmp_path / "GroundingDINO_SwinT_OGC.py").write_text("# fixture", encoding="utf-8")
    return tmp_path


def _registered_grounding_dino() -> RegisteredVisionModel:
    registry = build_registry(load_vision_stack_config())
    return registry.register("grounding_dino_test", load_vision_stack_config().grounding_dino)


def test_parser_converts_normalized_cxcywh_to_pixel_box() -> None:
    detections = _GroundingDINOOutputParser().to_detections(
        boxes_cxcywh=[[0.5, 0.5, 0.25, 0.5]],
        logits=[0.8],
        phrases=[" Person "],
        image_width=200,
        image_height=100,
        confidence_floor=0.35,
    )

    assert len(detections) == 1
    assert detections[0].label == "person"
    assert detections[0].bounding_box.x0 == 75.0
    assert detections[0].bounding_box.y0 == 25.0
    assert detections[0].bounding_box.x1 == 125.0
    assert detections[0].bounding_box.y1 == 75.0


def test_parser_excludes_boxes_below_confidence_floor() -> None:
    detections = _GroundingDINOOutputParser().to_detections(
        boxes_cxcywh=[[0.5, 0.5, 0.25, 0.25], [0.5, 0.5, 0.5, 0.5]],
        logits=[0.34, 0.35],
        phrases=["person", "logo"],
        image_width=100,
        image_height=100,
        confidence_floor=0.35,
    )

    assert [detection.label for detection in detections] == ["logo"]


def test_parser_clamps_boxes_to_image_bounds() -> None:
    detection = _GroundingDINOOutputParser().to_detections(
        boxes_cxcywh=[[0.05, 0.95, 0.3, 0.3]],
        logits=[0.9],
        phrases=["arrow"],
        image_width=100,
        image_height=100,
        confidence_floor=0.35,
    )[0]

    assert detection.bounding_box.x0 == 0.0
    assert detection.bounding_box.y1 == 100.0


def test_parser_empty_inputs_return_empty_list() -> None:
    assert (
        _GroundingDINOOutputParser().to_detections([], [], [], 100, 100, 0.35)
        == []
    )


def test_wrapper_construction_is_lazy(checkpoint_root: Path) -> None:
    wrapper = GroundingDINOWrapper(checkpoint_root)

    assert wrapper.is_loaded() is False


@pytest.mark.parametrize(
    ("image", "prompt", "box_threshold", "text_threshold", "match"),
    [
        ("not-array", "person", None, None, "image must be a numpy.ndarray"),
        (np.zeros((10, 10), dtype=np.uint8), "person", None, None, "image must be RGB"),
        (np.zeros((10, 10, 3), dtype=np.float32), "person", None, None, "image must be uint8"),
        (np.zeros((0, 10, 3), dtype=np.uint8), "person", None, None, "non-zero height"),
        (np.zeros((10, 10, 3), dtype=np.uint8), " ", None, None, "text_prompt must not be empty"),
        (np.zeros((10, 10, 3), dtype=np.uint8), "person", 0.0, None, "must be in"),
        (np.zeros((10, 10, 3), dtype=np.uint8), "person", None, 1.1, "must be in"),
    ],
)
def test_detect_rejects_invalid_inputs_before_model_interaction(
    checkpoint_root: Path,
    image: object,
    prompt: str,
    box_threshold: float | None,
    text_threshold: float | None,
    match: str,
) -> None:
    wrapper = GroundingDINOWrapper(checkpoint_root)
    active = _registered_grounding_dino().model_copy(
        update={"lifecycle_state": VisionModelLifecycleState.GPU_ACTIVE}
    )

    with patch.object(wrapper, "ensure_loaded") as ensure_loaded:
        with pytest.raises(ValueError, match=match):
            wrapper.detect(
                image,  # type: ignore[arg-type]
                prompt,
                active,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

    ensure_loaded.assert_not_called()


def test_detect_requires_active_gpu_reservation(checkpoint_root: Path) -> None:
    wrapper = GroundingDINOWrapper(checkpoint_root)

    with pytest.raises(VisionStackResourceError):
        wrapper.detect(
            np.zeros((10, 10, 3), dtype=np.uint8),
            "person",
            _registered_grounding_dino(),
        )


def test_integration_full_flow_with_mocked_model(
    checkpoint_root: Path,
    fake_torch: _FakeTorch,
) -> None:
    registry = build_registry(load_vision_stack_config())
    manager = GPUResourceManager(registry)
    wrapper = GroundingDINOWrapper(checkpoint_root)
    fake_model = _FakeModel()

    with patch.object(wrapper, "_build_model", return_value=fake_model):
        with manager.reserve("grounding_dino") as active:
            wrapper.ensure_loaded(active)
            detections = wrapper.detect(
                np.zeros((100, 200, 3), dtype=np.uint8),
                "person",
                active,
            )
            assert registry.get("grounding_dino").lifecycle_state == VisionModelLifecycleState.GPU_ACTIVE

    assert isinstance(detections[0], GroundingDINODetection)
    assert registry.get("grounding_dino").lifecycle_state == VisionModelLifecycleState.CPU_CACHED


def test_ensure_loaded_is_idempotent(
    checkpoint_root: Path,
    fake_torch: _FakeTorch,
) -> None:
    registry = build_registry(load_vision_stack_config())
    manager = GPUResourceManager(registry)
    wrapper = GroundingDINOWrapper(checkpoint_root)

    with patch.object(wrapper, "_build_model", return_value=_FakeModel()) as build_model:
        with manager.reserve("grounding_dino") as active:
            wrapper.ensure_loaded(active)
            wrapper.ensure_loaded(active)

    assert build_model.call_count == 1


def test_ensure_loaded_reloads_on_config_mismatch(
    checkpoint_root: Path,
    fake_torch: _FakeTorch,
) -> None:
    registry = build_registry(load_vision_stack_config())
    manager = GPUResourceManager(registry)
    wrapper = GroundingDINOWrapper(checkpoint_root)

    with patch.object(wrapper, "_build_model", return_value=_FakeModel()) as build_model:
        with patch.object(wrapper, "unload", wraps=wrapper.unload) as unload:
            with manager.reserve("grounding_dino") as active:
                wrapper.ensure_loaded(active)
                changed = active.model_copy(
                    update={
                        "config": active.config.model_copy(
                            update={"precision": VisionModelPrecision.FP32}
                        )
                    }
                )
                wrapper.ensure_loaded(changed)

    assert build_model.call_count == 2
    assert unload.call_count == 1


def test_cuda_oom_translates_and_reservation_releases(
    checkpoint_root: Path,
    fake_torch: _FakeTorch,
) -> None:
    registry = build_registry(load_vision_stack_config())
    manager = GPUResourceManager(registry)
    wrapper = GroundingDINOWrapper(checkpoint_root)
    fake_model = _FakeModel()
    fake_model.predict.side_effect = fake_torch.cuda.OutOfMemoryError("cuda out of memory")

    with patch.object(wrapper, "_build_model", return_value=fake_model):
        with pytest.raises(GroundingDINOOutOfMemoryError):
            with manager.reserve("grounding_dino") as active:
                wrapper.ensure_loaded(active)
                wrapper.detect(np.zeros((10, 10, 3), dtype=np.uint8), "person", active)

    fake_torch.cuda.empty_cache.assert_called()
    assert registry.get("grounding_dino").lifecycle_state == VisionModelLifecycleState.CPU_CACHED


def test_malformed_checkpoint_translates_to_load_error(
    checkpoint_root: Path,
    fake_torch: _FakeTorch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch.load.side_effect = RuntimeError("bad checkpoint")
    fake_model = _FakeModel()
    monkeypatch.setitem(
        sys.modules,
        "groundingdino.util.slconfig",
        SimpleNamespace(SLConfig=SimpleNamespace(fromfile=MagicMock(return_value=object()))),
    )
    monkeypatch.setitem(
        sys.modules,
        "groundingdino.models",
        SimpleNamespace(build_model=MagicMock(return_value=fake_model)),
    )
    monkeypatch.setitem(
        sys.modules,
        "groundingdino.util.utils",
        SimpleNamespace(clean_state_dict=MagicMock(return_value={})),
    )
    wrapper = GroundingDINOWrapper(checkpoint_root)
    active = _registered_grounding_dino().model_copy(
        update={"lifecycle_state": VisionModelLifecycleState.GPU_ACTIVE}
    )

    with pytest.raises(GroundingDINOLoadError) as exc_info:
        wrapper.ensure_loaded(active)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_unload_releases_loaded_model(
    checkpoint_root: Path,
    fake_torch: _FakeTorch,
) -> None:
    wrapper = GroundingDINOWrapper(checkpoint_root)
    active = _registered_grounding_dino().model_copy(
        update={"lifecycle_state": VisionModelLifecycleState.GPU_ACTIVE}
    )
    fake_model = _FakeModel()

    with patch.object(wrapper, "_build_model", return_value=fake_model):
        wrapper.ensure_loaded(active)

    wrapper.unload()

    assert wrapper.is_loaded() is False
    fake_torch.cuda.empty_cache.assert_called()


def test_detect_empty_detections_is_not_an_error(
    checkpoint_root: Path,
    fake_torch: _FakeTorch,
) -> None:
    wrapper = GroundingDINOWrapper(checkpoint_root)
    active = _registered_grounding_dino().model_copy(
        update={"lifecycle_state": VisionModelLifecycleState.GPU_ACTIVE}
    )

    with patch.object(wrapper, "_build_model", return_value=_FakeModel(output=([], [], []))):
        detections = wrapper.detect(
            np.zeros((10, 10, 3), dtype=np.uint8),
            "person",
            active,
        )

    assert detections == []
