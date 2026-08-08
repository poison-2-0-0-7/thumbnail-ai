"""
test_model_runtime_manager.py
==============================

Comprehensive test suite for Phase 4.4 Model Runtime Manager.
Tests cover:
- Model registration & descriptor retrieval (ModelRegistry)
- Lazy loading and model state transitions (ModelState)
- Reference counting and RAII handle context manager (ModelHandle)
- Device selection and resolution (DeviceManager)
- Memory tracking and VRAM statistics (MemoryTracker)
- Health checks and warmup validation (HealthMonitor)
- Cache management, pinning, and LRU eviction (ModelCache)
- Thread safety and concurrent handle acquisitions
- Integration with ExecutionDispatcher and Renderer Stage Adapters
"""

import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional
import pytest

from renderer_v2.runtime import (
    BaseModelAdapter,
    DeviceManager,
    GenericModelAdapter,
    HealthMonitor,
    MemoryTracker,
    ModelCache,
    ModelDescriptor,
    ModelHandle,
    ModelRegistry,
    ModelRuntimeManager,
    ModelRuntimeManagerError,
    ModelState,
)
from renderer_v2.execution.dispatcher import ExecutionDispatcher
from renderer_v2.execution.engine import ExecutionEngine
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner
from thumbnail_intelligence.reasoning.renderer_adapter import RendererV2Adapter
from thumbnail_intelligence.reasoning.spatial_composition_planner import SpatialCompositionPlanner


class DummyModel:
    """Mock neural network model for testing."""

    def __init__(self, name: str = "dummy") -> None:
        self.name = name
        self.is_warmed_up = False

    def predict(self, x: Any, **kwargs: Any) -> Any:
        return f"pred_{self.name}_{x}"


class CustomDummyAdapter(BaseModelAdapter):
    """Custom model adapter for testing."""

    def load(self) -> None:
        self.state = ModelState.LOADING
        self._model_instance = DummyModel(self.model_name)
        self.state = ModelState.READY

    def unload(self) -> None:
        self.state = ModelState.UNLOADING
        self._model_instance = None
        self.state = ModelState.OFFLOADED

    def warmup(self) -> bool:
        if self._model_instance is not None:
            self._model_instance.is_warmed_up = True
        return True

    def health_check(self) -> bool:
        return self.state in {ModelState.READY, ModelState.IN_USE}

    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        return self._model_instance.predict(inputs, **kwargs)

    def cleanup(self) -> None:
        pass


class TestModelRuntimeManager:

    def test_model_registry_pre_registered_models(self):
        """Verify pre-registration of all 8 core models in ModelRegistry."""
        registry = ModelRegistry()
        expected = ["GroundingDINO", "SAM2", "BiRefNet", "SDXL", "BrushNet", "CodeFormer", "GFPGAN", "DepthAnything"]

        for model_name in expected:
            assert registry.is_registered(model_name)
            desc = registry.get_descriptor(model_name)
            assert desc.model_name == model_name
            assert desc.estimated_vram_gb > 0.0

    def test_custom_model_registration_and_lazy_loading(self):
        """Test custom model descriptor registration and lazy loading."""
        manager = ModelRuntimeManager()

        desc = ModelDescriptor(
            model_name="CustomTestNet",
            version="1.1.0",
            framework="pytorch",
            estimated_vram_gb=2.5,
            supported_tasks=["custom_test"],
        )
        manager.register_model(desc, loader_fn=lambda: DummyModel("CustomTestNet"), adapter_cls=CustomDummyAdapter)

        assert manager.get_model_state("CustomTestNet") == ModelState.REGISTERED

        # Acquire model handle via context manager
        with manager.acquire_model("CustomTestNet") as handle:
            assert handle.model_name == "CustomTestNet"
            assert handle.state == ModelState.IN_USE
            res = handle.predict("input_data")
            assert res == "pred_CustomTestNet_input_data"

        # After exiting context manager, state transitions back to READY
        assert manager.get_model_state("CustomTestNet") == ModelState.READY

    def test_reference_counting_and_manual_release(self):
        """Test reference counting across multiple concurrent handle acquisitions."""
        manager = ModelRuntimeManager()
        manager.register_model(
            ModelDescriptor(model_name="RefCountNet", estimated_vram_gb=1.0),
            loader_fn=lambda: DummyModel("RefCountNet"),
        )

        h1 = manager.acquire_model("RefCountNet")
        assert manager.cache.get_ref_count("RefCountNet") == 1
        assert manager.get_model_state("RefCountNet") == ModelState.IN_USE

        h2 = manager.acquire_model("RefCountNet")
        assert manager.cache.get_ref_count("RefCountNet") == 2
        assert manager.get_model_state("RefCountNet") == ModelState.IN_USE

        h1.release()
        assert manager.cache.get_ref_count("RefCountNet") == 1
        assert manager.get_model_state("RefCountNet") == ModelState.IN_USE

        h2.release()
        assert manager.cache.get_ref_count("RefCountNet") == 0
        assert manager.get_model_state("RefCountNet") == ModelState.READY

    def test_lru_cache_eviction_and_pinning(self):
        """Test LRU cache eviction and protection of pinned models."""
        manager = ModelRuntimeManager(max_loaded_models=2)

        # Register 3 dummy models
        for i in range(1, 4):
            m_name = f"Net_{i}"
            desc = ModelDescriptor(model_name=m_name, is_pinned=(i == 1))
            manager.register_model(desc, loader_fn=lambda name=m_name: DummyModel(name))

        # Acquire Net_1 (pinned) and Net_2
        h1 = manager.acquire_model("Net_1")
        h1.release()

        time.sleep(0.01)

        h2 = manager.acquire_model("Net_2")
        h2.release()

        assert manager.get_model_state("Net_1") == ModelState.READY
        assert manager.get_model_state("Net_2") == ModelState.READY

        # Acquiring Net_3 should trigger LRU eviction of Net_2 (since Net_1 is pinned)
        h3 = manager.acquire_model("Net_3")
        h3.release()

        assert manager.get_model_state("Net_1") == ModelState.READY  # Protected by pinning
        assert manager.get_model_state("Net_2") == ModelState.REGISTERED  # Evicted (unloaded)
        assert manager.get_model_state("Net_3") == ModelState.READY

    def test_device_manager_resolution(self):
        """Test DeviceManager device resolution and fallback."""
        dm = DeviceManager()
        assert dm.resolve_device("cpu") == "cpu"

        # Resolving CUDA on CPU-only system falls back gracefully to CPU
        res_cuda = dm.resolve_device("cuda")
        assert res_cuda in {"cuda", "cpu"}

    def test_memory_tracker_stats(self):
        """Test MemoryTracker VRAM recording and memory status summary."""
        mem = MemoryTracker(max_vram_gb=16.0)
        mem.record_model_allocation("SDXL", 6.5)
        mem.record_model_allocation("SAM2", 2.0)

        status = mem.get_memory_status()
        assert status["max_budget_vram_gb"] == 16.0
        assert status["active_model_allocations"]["SDXL"] == 6.5
        assert status["active_model_allocations"]["SAM2"] == 2.0

        mem.record_model_deallocation("SDXL")
        status2 = mem.get_memory_status()
        assert "SDXL" not in status2["active_model_allocations"]

    def test_health_monitor_checks(self):
        """Test HealthMonitor validation of descriptors and loaded models."""
        monitor = HealthMonitor()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(b"dummy_weights")
            ckpt_path = tmp.name

        try:
            desc_valid = ModelDescriptor(model_name="ValidNet", checkpoint_path=ckpt_path)
            ok, msg = monitor.validate_checkpoint(desc_valid)
            assert ok is True

            desc_missing = ModelDescriptor(model_name="MissingNet", checkpoint_path="/invalid/path.pt")
            ok_m, msg_m = monitor.validate_checkpoint(desc_missing)
            assert ok_m is False

            adapter = CustomDummyAdapter(desc_valid)
            adapter.load()
            res = monitor.check_adapter_health(adapter)
            assert res.is_healthy is True
        finally:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    def test_thread_safe_concurrent_acquisitions(self):
        """Test thread safety under concurrent handle acquisitions."""
        manager = ModelRuntimeManager()
        manager.register_model(
            ModelDescriptor(model_name="SharedNet"),
            loader_fn=lambda: DummyModel("SharedNet"),
        )

        errors = []

        def worker():
            try:
                for _ in range(10):
                    with manager.acquire_model("SharedNet") as h:
                        res = h.predict("thread_input")
                        assert res == "pred_SharedNet_thread_input"
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_execution_dispatcher_integration(self):
        """Test integration of ModelRuntimeManager into ExecutionDispatcher and ExecutionEngine."""
        runtime_manager = ModelRuntimeManager()
        dispatcher = ExecutionDispatcher(use_placeholders=False, runtime_manager=runtime_manager)
        engine = ExecutionEngine(dispatcher=dispatcher)

        brief = DesignBrief()
        plan = ExecutionPlanner().plan(brief)
        comp = SpatialCompositionPlanner().plan(plan, brief)
        package = RendererV2Adapter().translate(comp, plan)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "runtime_out.jpg")
            report = engine.execute(package, context_overrides={"output_path": out_file})

            assert report.status.value in {"SUCCESS", "SUCCESS_WITH_DEGRADATION"}
            assert os.path.exists(out_file)
