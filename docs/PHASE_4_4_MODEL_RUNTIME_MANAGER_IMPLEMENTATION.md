# Phase 4.4 — Model Runtime Manager Implementation

**Status:** Completed  
**Subsystem:** Renderer V2 — Model Runtime & Memory Lifecycle Management  
**Consumes:** Model loading requests from `ExecutionDispatcher` & Stage Adapters  
**Produces:** Managed `ModelHandle` instances, VRAM memory tracking, LRU cache eviction, and health check diagnostics  

---

## 1. Architecture Overview

Phase 4.4 establishes the **Model Runtime Manager** (`renderer_v2/runtime/`), a centralized runtime system owning the full lifecycle of AI models across the Thumbnail AI ecosystem.

No renderer stage is permitted to directly instantiate AI models. All model acquisitions flow through `ModelRuntimeManager.acquire_model()`, returning reference-counted RAII handles (`ModelHandle`).

```mermaid
flowchart TD
    SA["Stage Adapters\n(BackgroundGeneratorAdapter, SubjectExtractorAdapter)"] -->|acquire_model(name)| MRM["ModelRuntimeManager"]
    
    subgraph ModelRuntimeManager ["Model Runtime System (renderer_v2/runtime/)"]
        MRM --> REG["ModelRegistry\n(Descriptors & Factories)"]
        MRM --> CACHE["ModelCache\n(LRU Eviction & Ref Counts)"]
        MRM --> DEV["DeviceManager\n(CPU / CUDA Placement)"]
        MRM --> MEM["MemoryTracker\n(VRAM Budgets & Allocation)"]
        MRM --> HEALTH["HealthMonitor\n(Warmup & Checkpoint Integrity)"]
    end

    CACHE -->|Returns| HANDLE["ModelHandle (RAII Context Manager)"]
    HANDLE -->|Release on __exit__| CACHE
```

---

## 2. Pre-Registered Models in Registry

The `ModelRegistry` pre-registers descriptors for all core vision and diffusion models:

| Model Name | Framework | Required Device | Estimated VRAM | Supported Tasks |
|---|---|---|---|---|
| `GroundingDINO` | PyTorch | `cuda` | 1.5 GB | Zero-shot object detection |
| `SAM2` | PyTorch | `cuda` | 2.0 GB | Instance segmentation & masking |
| `BiRefNet` | PyTorch | `cuda` | 1.2 GB | High-resolution alpha matting |
| `SDXL` | Diffusers | `cuda` | 6.5 GB | Text-to-image synthesis |
| `BrushNet` | Diffusers | `cuda` | 4.5 GB | Inpainting & background synthesis |
| `CodeFormer` | PyTorch | `cuda` | 1.0 GB | Face enhancement & restoration |
| `GFPGAN` | PyTorch | `cuda` | 1.0 GB | High-fidelity face restoration |
| `DepthAnything` | PyTorch | `cuda` | 1.5 GB | Monocular depth estimation |

---

## 3. Model Lifecycle & State Machine

```
UNREGISTERED ──► REGISTERED ──► LOADING ──► READY ◄──► IN_USE
                      ▲                       │          │
                      │                       ▼          ▼
                   FAILED ◄────────────── OFFLOADED ◄── UNLOADING
```

1. **`UNREGISTERED`**: Model is unknown to the registry.
2. **`REGISTERED`**: `ModelDescriptor` exists in `ModelRegistry`, but weights are not loaded.
3. **`LOADING`**: Weights are being read from disk / moved into VRAM.
4. **`READY`**: Model is loaded and idle in memory (`ref_count == 0`), eligible for LRU cache eviction if unpinned.
5. **`IN_USE`**: Model has one or more active handles (`ref_count > 0`). Protected from eviction.
6. **`OFFLOADED`**: Model weights have been evicted or moved to CPU RAM to free VRAM.
7. **`UNLOADING`**: Model is actively being purged from memory.
8. **`FAILED`**: Weight loading or CUDA allocation failed.

---

## 4. Cache & VRAM Memory Model

- **Lazy Loading:** Models are loaded on-demand during the first `acquire_model()` call.
- **Reference Counting & RAII:** `ModelHandle` implements Python's context manager protocol (`__enter__` / `__exit__`). Upon exiting the `with` block, `handle.release()` automatically decrements the active reference count.
- **LRU Cache Eviction:** When `max_loaded_models` (default 5) or VRAM memory ceilings are reached, `ModelCache.evict_lru()` unloads the least recently used model where `ref_count == 0` and `is_pinned == False`.
- **Pinned Protection:** Models marked `is_pinned=True` (e.g. core segmenters or critical pipelines) are immune to automatic LRU eviction.
- **Garbage Collection:** Evictions trigger explicit Python `gc.collect()` and `torch.cuda.empty_cache()` / `torch.cuda.ipc_collect()`.

---

## 5. Device Management & Health Diagnostics

- **Device Resolution:** `DeviceManager` resolves requested compute targets (`"cuda"`, `"cpu"`). If CUDA is unavailable, it gracefully degrades to `"cpu"`.
- **Integrity Validation:** `HealthMonitor` validates:
  - Checkpoint file existence and non-zero byte size on disk.
  - Compute device accessibility.
  - Zero-tensor dummy inference warmup (`warmup()`).

---

## 6. Developer Guide

### Acquiring Models in Stage Adapters

```python
from renderer_v2.runtime import ModelRuntimeManager, ModelDescriptor

# 1. Instantiate or inject central ModelRuntimeManager
runtime_manager = ModelRuntimeManager(max_vram_gb=12.0, max_loaded_models=5)

# 2. Acquire model handle using RAII context manager
with runtime_manager.acquire_model("SAM2") as handle:
    print(f"Model: {handle.model_name}, Device: {handle.device}, State: {handle.state.value}")
    # Perform prediction via handle or underlying adapter
    predictions = handle.predict(image_input)

# Handle is automatically released back to READY state on context exit
```

---

## 7. Test Results

Comprehensive unit and integration tests in [`tests/test_model_runtime_manager.py`](file:///D:/Afsar/app%20development/thumbnail-ai/tests/test_model_runtime_manager.py):

- `test_model_registry_pre_registered_models`: PASSED
- `test_custom_model_registration_and_lazy_loading`: PASSED
- `test_reference_counting_and_manual_release`: PASSED
- `test_lru_cache_eviction_and_pinning`: PASSED
- `test_device_manager_resolution`: PASSED
- `test_memory_tracker_stats`: PASSED
- `test_health_monitor_checks`: PASSED
- `test_thread_safe_concurrent_acquisitions`: PASSED
- `test_execution_dispatcher_integration`: PASSED

Full test suite execution: **42 PASSED**, 0 failures in **18.04s**.
