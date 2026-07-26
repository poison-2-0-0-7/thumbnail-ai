"""Tests for AI Vision Stack V2.1 runtime bootstrap infrastructure."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time

import pytest
import yaml

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import VISION_STACK_CONFIG_PATH, load_vision_stack_config  # noqa: E402
from vision_stack.exceptions import (  # noqa: E402
    VisionStackCheckpointError,
    VisionStackConfigError,
    VisionStackRuntimeError,
)
from vision_stack.loader import ModelLoader, build_registry  # noqa: E402
from vision_stack.models import VisionModelLifecycleState, VisionStackConfig  # noqa: E402
from vision_stack.resources import GPUResourceManager  # noqa: E402
from vision_stack.runtime import RuntimeManager  # noqa: E402


def _create_checkpoint_artifacts(root: Path, config: VisionStackConfig) -> None:
    for name, model_config in config.model_items():
        checkpoint_path = root / Path(model_config.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.suffix:
            checkpoint_path.write_bytes(b"checkpoint metadata fixture")
        else:
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        for filename in {
            "grounding_dino": ("GroundingDINO_SwinT_OGC.py",),
            "sam2": ("sam2.1_hiera_l.yaml",),
            "insightface": ("det_10g.onnx", "w600k_r50.onnx"),
            "paddleocr": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
        }.get(name, ()):
            required_path = checkpoint_path.parent / filename
            if Path(filename).suffix:
                required_path.write_bytes(b"required sidecar fixture")
            else:
                required_path.mkdir(parents=True, exist_ok=True)


def test_model_loader_resolves_and_validates_checkpoint_metadata(tmp_path: Path) -> None:
    config = load_vision_stack_config()
    _create_checkpoint_artifacts(tmp_path, config)

    metadata = ModelLoader(tmp_path).bootstrap_metadata(config)

    assert len(metadata.models) == 10
    assert metadata.sequential_execution is True
    assert metadata.gpu_lock_enforced is True
    assert metadata.weights_loaded is False
    grounding_dino = next(model for model in metadata.models if model.model_name == "grounding_dino")
    assert grounding_dino.checkpoint_path == tmp_path / "groundingdino_swint_ogc.pth"
    assert tmp_path / "GroundingDINO_SwinT_OGC.py" in grounding_dino.required_paths
    assert grounding_dino.missing_paths == ()
    assert grounding_dino.weights_loaded is False


def test_model_loader_reports_missing_checkpoint_artifacts(tmp_path: Path) -> None:
    config = load_vision_stack_config()

    with pytest.raises(VisionStackCheckpointError, match="Missing checkpoint artifacts"):
        ModelLoader(tmp_path).bootstrap_metadata(config)


def test_model_loader_validates_required_sidecar_files(tmp_path: Path) -> None:
    config = load_vision_stack_config()
    _create_checkpoint_artifacts(tmp_path, config)
    (tmp_path / "GroundingDINO_SwinT_OGC.py").unlink()

    with pytest.raises(VisionStackCheckpointError) as exc_info:
        ModelLoader(tmp_path).bootstrap_metadata(config)

    assert "grounding_dino" in str(exc_info.value)
    assert "GroundingDINO_SwinT_OGC.py" in str(exc_info.value)


def test_runtime_manager_bootstrap_registers_loader_metadata(tmp_path: Path) -> None:
    config = load_vision_stack_config()
    _create_checkpoint_artifacts(tmp_path, config)
    manager = RuntimeManager(checkpoint_root=tmp_path)

    metadata = manager.bootstrap()

    assert len(metadata.models) == 10
    registered = manager.registry.get("sam2")
    assert registered.lifecycle_state == VisionModelLifecycleState.REGISTERED
    assert registered.metadata["checkpoint_identifier"] == "sam2.1_hiera_large.pt"
    assert registered.metadata["weights_loaded"] is False


def test_runtime_manager_rejects_invalid_configuration(tmp_path: Path) -> None:
    raw_config = yaml.safe_load(VISION_STACK_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_config["vision_stack"]["sam2"]["batch_size"] = 2
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    with pytest.raises(VisionStackConfigError):
        RuntimeManager(config_path=config_path, checkpoint_root=tmp_path).bootstrap()


def test_gpu_resource_manager_reserves_one_active_model_and_restores_cpu_state() -> None:
    registry = build_registry(load_vision_stack_config())
    manager = GPUResourceManager(registry)

    with manager.reserve("grounding_dino"):
        assert manager.active_model_name == "grounding_dino"
        assert (
            registry.get("grounding_dino").lifecycle_state
            == VisionModelLifecycleState.GPU_ACTIVE
        )

    assert manager.active_model_name is None
    assert registry.get("grounding_dino").lifecycle_state == VisionModelLifecycleState.CPU_CACHED
    assert registry.get("grounding_dino").runtime_state["cuda_executed"] is False


def test_runtime_manager_schedules_models_sequentially_across_threads(tmp_path: Path) -> None:
    config = load_vision_stack_config()
    _create_checkpoint_artifacts(tmp_path, config)
    manager = RuntimeManager(checkpoint_root=tmp_path)
    manager.bootstrap()
    events: list[str] = []

    def operation(model_name: str) -> None:
        with manager.reserve_model(model_name):
            events.append(f"start:{model_name}:{manager.gpu_resources.active_model_name}")
            time.sleep(0.02)
            events.append(f"end:{model_name}:{manager.gpu_resources.active_model_name}")

    first = threading.Thread(target=operation, args=("grounding_dino",))
    second = threading.Thread(target=operation, args=("sam2",))

    first.start()
    time.sleep(0.005)
    second.start()
    first.join()
    second.join()

    assert events == [
        "start:grounding_dino:grounding_dino",
        "end:grounding_dino:grounding_dino",
        "start:sam2:sam2",
        "end:sam2:sam2",
    ]


def test_runtime_manager_lifecycle_drain_and_shutdown(tmp_path: Path) -> None:
    config = load_vision_stack_config()
    _create_checkpoint_artifacts(tmp_path, config)
    manager = RuntimeManager(checkpoint_root=tmp_path, worker_restart_threshold=1)
    manager.bootstrap()

    results = manager.run_sequential(
        ("grounding_dino", "sam2"),
        lambda model: model.name,
    )
    count = manager.mark_thumbnail_processed()
    manager.begin_graceful_drain()

    assert results == ("grounding_dino", "sam2")
    assert count == 1
    assert manager.runtime.restart_required is True
    with pytest.raises(VisionStackRuntimeError, match="not accepting new work"):
        manager.run_sequential(("florence2",), lambda model: model.name)

    manager.shutdown()
    assert (
        manager.registry.get("grounding_dino").lifecycle_state
        == VisionModelLifecycleState.EVICTED
    )
    assert manager.registry.get("sam2").lifecycle_state == VisionModelLifecycleState.EVICTED
