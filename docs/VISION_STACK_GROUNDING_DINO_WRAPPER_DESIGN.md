# VISION_STACK_GROUNDING_DINO_WRAPPER_DESIGN.md

**AI Vision Stack V2.1 — Stage 1: Localization**
**Component: `GroundingDINOWrapper`**
**thumbnail-ai**

Status: **Design specification, v1.0. No implementation.**
Source of truth (verified against the live repository, not against planning docs): `modules/vision_stack/` (`config.py`, `models.py`, `registry.py`, `lifecycle.py`, `resources.py`, `runtime.py`, `loader.py`, `exceptions.py`), `vision_stack.yaml`, `modules/config.py`, `modules/models.py`, `docs/AI_Vision_Stack_V2.1_Architecture.md` (§ "GroundingDINO" model-checkpoint spec, § "GroundingDINO" inter-module contract), `tests/test_vision_stack_runtime.py`, `tests/test_vision_stack_config.py`.

This document specifies the first real model wrapper to sit on top of the completed Vision Stack V2.1 bootstrap infrastructure (Runtime Manager, Registry, Loader, Lifecycle, GPU Resource Manager, Configuration). Every earlier component in `modules/vision_stack/` is metadata-only today — none of it imports `torch`, instantiates a model, or executes CUDA. `GroundingDINOWrapper` is the first component that actually does those things, so this document also fixes the pattern (construction, GPU-lease usage, error translation, logging) that the six remaining wrappers listed in §14 will each replicate with minimal deltas.

---

## 1. Purpose and Scope

### 1.1 Why this component exists

The Vision Stack's Stage 1 (Localization) is GroundingDINO: an open-vocabulary, text-prompted object detector. Per the architecture document's inter-module contract, it takes a thumbnail image plus a text prompt (e.g. `"person . face . logo . arrow . text"`) and returns high-recall bounding boxes that seed SAM 2's instance segmentation in Stage 2. `GroundingDINOWrapper` is the component that turns the registry's `grounding_dino` `VisionModelConfig` entry and the loader's resolved checkpoint metadata into an actual, callable detector — while never touching the registry's bootstrap-time responsibilities (config validation, checkpoint path resolution) or the GPU Resource Manager's lifecycle responsibilities (state transitions, single-active-model enforcement).

### 1.2 Responsibilities

- Own exactly one loaded GroundingDINO model instance (native PyTorch, per the architecture document's "Preferred inference backend") per `RuntimeManager` process.
- Load model weights **lazily**, on first use, never at construction and never at `RuntimeManager.bootstrap()` time — bootstrap remains metadata-only per the existing `ModelLoader` contract.
- Validate input (`image`, `text_prompt`) before any inference call.
- Run inference under a GPU reservation obtained from `GPUResourceManager.reserve("grounding_dino")`, and only inside that reservation's context.
- Convert raw GroundingDINO output tensors into the standard `GroundingDINODetection` schema (§7): pixel-space boxes, confidence-filtered, image-bounds-clamped.
- Report weight-loading and CUDA-execution facts back into the registry's `runtime_state` (`weights_loaded`, `cuda_executed`) so registry state actually reflects reality instead of the resource manager's current placeholder `False` values.
- Translate every third-party failure (`torch` CUDA errors, malformed checkpoint files, GroundingDINO library exceptions) into the typed exception hierarchy defined in §9 — no raw third-party exception may cross this wrapper's public boundary.
- Apply the configured per-model `fallback` policy (`skip_stage` for `grounding_dino` per `vision_stack.yaml`) when inference cannot complete.
- Log every stage of load / inference / eviction through the project's standard rotating Loguru sink convention.

### 1.3 Boundaries — what this component explicitly does not do

- It does not decide *when* to run. Scheduling, GPU-lease acquisition/release, and sequential-execution ordering belong entirely to `RuntimeManager.run_sequential` / `GPUResourceManager.reserve`. The wrapper is called *inside* an already-held reservation; it never acquires or releases the `gpu_lock` itself.
- It does not perform SAM 2 prompting, mask generation, or any Stage 2 responsibility. Its only output is the `List[GroundingDINODetection]` defined in §7 — box prompts for the next stage to consume, nothing more.
- It does not re-validate or duplicate the registry's `VisionModelConfig` (checkpoint path, precision, device, timeout, fallback policy). It reads that config from the `RegisteredVisionModel` handed to it by the reservation context manager and treats it as already-validated.
- It does not manage checkpoint file existence checks — `ModelLoader.bootstrap_metadata` already guarantees (at `RuntimeManager.bootstrap()` time) that `groundingdino_swint_ogc.pth` and `GroundingDINO_SwinT_OGC.py` exist before this wrapper is ever constructed. The wrapper trusts that guarantee and only re-checks the file at the moment of `torch.load`, to catch on-disk corruption/deletion between bootstrap and first use.
- It does not implement retry logic beyond what is specified in §10. There is no Tenacity layer here (unlike Module 7's ComfyUI HTTP/WebSocket transports) — a failed GroundingDINO inference call degrades via the configured `fallback` policy, it is not blindly retried, because a repeated CUDA OOM or a repeated malformed-prompt failure will not resolve itself by retrying with identical inputs.
- It does not know about `ReferenceAssetResolver`, `VisualReferenceManifest`, or any Module 6.5 VRE component. Its only contract-facing collaborator is whatever future orchestrator sequences Stage 1 → Stage 2 (SAM 2) → … per the architecture document; that orchestrator is out of scope for this document.

---

## 2. Architecture

### 2.1 Component placement

`GroundingDINOWrapper` lives in a new module, `modules/vision_stack/grounding_dino.py`, alongside the existing bootstrap infrastructure (`config.py`, `models.py`, `registry.py`, `lifecycle.py`, `loader.py`, `resources.py`, `runtime.py`) inside the `vision_stack` package. This mirrors how `modules/comfyui_client.py` holds every Module 7 Phase 2 ComfyUI-transport class in one file — one wrapper module per external model, not one file per method.

It is a **public** class (unlike `_ComfyUIHTTPTransport`/`_ComfyUIWebSocketTransport`, which are module-private): callers outside `vision_stack` (a future Stage-orchestration module, and this component's own test suite) construct it directly, so it is exported from `modules/vision_stack/__init__.py`'s `__all__` alongside `RuntimeManager`, `ModelRegistry`, etc.

### 2.2 Relationship to existing Vision Stack components

```
                         ┌───────────────────────┐
                         │     RuntimeManager      │
                         │  (bootstrap, schedule)   │
                         └───────────┬─────────────┘
                                     │ owns
                 ┌───────────────────┼────────────────────┐
                 ▼                   ▼                    ▼
        ┌────────────────┐  ┌────────────────────┐ ┌──────────────┐
        │  ModelRegistry   │  │ GPUResourceManager   │ │  ModelLoader   │
        │ (config+state)   │  │ (single-active lease) │ │ (checkpoint    │
        └────────┬─────────┘  └──────────┬───────────┘ │  metadata)     │
                  │                       │             └──────────────┘
                  │ RegisteredVisionModel │ reserve("grounding_dino")
                  │ (config, lifecycle)   │ yields RegisteredVisionModel
                  ▼                       ▼
              ┌─────────────────────────────────────┐
              │        GroundingDINOWrapper           │
              │  (this document)                       │
              │  - lazy weight load                    │
              │  - input validation                    │
              │  - inference under active reservation  │
              │  - output normalization                │
              │  - runtime_state reporting             │
              └─────────────────────────────────────┘
```

The wrapper never imports `RuntimeManager` or `ModelRegistry` directly to *drive* scheduling — it only receives a `RegisteredVisionModel` (already `GPU_ACTIVE`) as a method argument, exactly the shape `GPUResourceManager.reserve()` yields. This keeps the wrapper testable in complete isolation from `RuntimeManager` (§13), the same separation the Module 7 WebSocket transport document applies between transport and `_QueueTracker`.

### 2.3 Interaction sequence with `RuntimeManager` / `GPUResourceManager`

A caller (the future Stage-1 orchestrator, or a test) always drives the wrapper the same way:

```python
with runtime_manager.reserve_model("grounding_dino") as registered_model:
    wrapper.ensure_loaded(registered_model)
    detections = wrapper.detect(image, text_prompt="person . face . logo . text")
```

`ensure_loaded` and `detect` both read `registered_model.config` (a `VisionModelConfig`) for `device`, `precision`, `timeout`, and `fallback`, but the wrapper does not hold a persistent reference to `registered_model` between calls — every call receives it fresh from the caller, because the registry's `RegisteredVisionModel` is an immutable (`frozen=True`) snapshot that changes identity on every `registry.transition(...)`/`update_runtime_state(...)` call (§ registry.py `model_copy(update=...)`).

### 2.4 Internal architecture — two collaborators inside `grounding_dino.py`

Mirroring the project's established pattern of splitting "boot-time metadata" from "runtime behavior" (`loader.py` vs. `resources.py`) and "raw I/O" from "parsed result" (`_ComfyUIHTTPTransport._request` vs. `_request_json`), the module is internally split into two collaborators:

| Class | Responsibility | Public? |
| --- | --- | --- |
| `GroundingDINOWrapper` | Public-facing API: `ensure_loaded`, `detect`, `unload`, `is_loaded`. Owns the loaded model handle and delegates parsing. | Yes |
| `_GroundingDINOOutputParser` | Pure, stateless conversion of raw model output (boxes/logits/phrases tensors) into `List[GroundingDINODetection]`, including confidence filtering and bounds clamping. | No (module-private) |

Splitting the parser out keeps the confidence-floor/bounds-clamping logic unit-testable with plain tensors/arrays, with no `torch.load`, no checkpoint file, and no GPU required — the same testing benefit `_ComfyUIWebSocketTransport.receive()` vs `next_event()` gets from being split.

---

## 3. Class Diagram (ASCII)

```
┌───────────────────────────────────────────────────────────────────────┐
│ GroundingDINOWrapper                                                   │
├───────────────────────────────────────────────────────────────────────┤
│ - _model: Any | None                    # loaded GroundingDINO model    │
│ - _model_config: VisionModelConfig | None                              │
│ - _device: str | None                   # resolved actual device       │
│ - _load_lock: threading.RLock                                          │
│ - _parser: _GroundingDINOOutputParser                                  │
├───────────────────────────────────────────────────────────────────────┤
│ + __init__(checkpoint_root: Path | None = None) -> None                │
│ + is_loaded() -> bool                                                  │
│ + ensure_loaded(registered_model: RegisteredVisionModel) -> None       │
│ + detect(image: np.ndarray, text_prompt: str,                          │
│          registered_model: RegisteredVisionModel,                     │
│          box_threshold: float | None = None,                          │
│          text_threshold: float | None = None) -> List[GroundingDINODetection] │
│ + unload() -> None                                                     │
│ - _build_model(model_config: VisionModelConfig) -> Any                 │
│ - _resolve_device(configured_device: str) -> str                      │
│ - _run_inference(image, text_prompt, box_threshold, text_threshold)    │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ _GroundingDINOOutputParser  (module-private, stateless)                │
├───────────────────────────────────────────────────────────────────────┤
│ + to_detections(boxes_cxcywh, logits, phrases,                        │
│                 image_width: int, image_height: int,                  │
│                 confidence_floor: float) -> List[GroundingDINODetection] │
│ - _cxcywh_to_xyxy_pixels(box, width, height) -> tuple[float, float, float, float] │
│ - _clamp_to_bounds(x0, y0, x1, y1, width, height) -> tuple[...]        │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ GroundingDINODetection  (pydantic, frozen — models.py addition)        │
├───────────────────────────────────────────────────────────────────────┤
│ + label: str                                                           │
│ + confidence: float                                                    │
│ + bounding_box: PixelBoundingBox                                       │
│ + source: Literal["grounding_dino"] = "grounding_dino"                 │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ PixelBoundingBox  (pydantic, frozen — models.py addition)               │
├───────────────────────────────────────────────────────────────────────┤
│ + x0: float   + y0: float   + x1: float   + y1: float  (absolute px)   │
└───────────────────────────────────────────────────────────────────────┘
```

**Relationships:** `GroundingDINOWrapper` *uses* `_GroundingDINOOutputParser` (composition, constructed once in `__init__`). `GroundingDINOWrapper.detect()` *returns* `List[GroundingDINODetection]`. `GroundingDINODetection` *has-a* `PixelBoundingBox`. `GroundingDINOWrapper` *reads but does not own* `RegisteredVisionModel` / `VisionModelConfig` (passed in per call, defined in `vision_stack/models.py`, unchanged by this document).

---

## 4. Naming Conflict — Critical Finding

`modules/models.py` already defines a `DetectedObject` (line ~296) and a `BoundingBox` (line ~162) used by the legacy Module 4/5 YOLO-based `thumbnail_intelligence.py` pipeline. That `BoundingBox` is **normalized** (`x_min`/`y_min`/`x_max`/`y_max` as fractions of image width/height, `[0.0, 1.0]`), while the architecture document's GroundingDINO contract specifies **absolute pixel-space** boxes (`[x0, y0, x1, y1]`). Reusing `DetectedObject`/`BoundingBox` for Stage 1 output would silently mix two incompatible coordinate conventions under the same field names — the single most likely source of a downstream SAM 2 prompting bug if implemented without this document.

**Decision:** this design introduces two new, distinctly-named Pydantic models — `GroundingDINODetection` and `PixelBoundingBox` — in `modules/vision_stack/models.py` (not `modules/models.py`), scoped to the Vision Stack V2.1 package rather than the legacy pipeline. No existing model, field, or import in `modules/models.py` is touched. This also matches the project's "never modify existing files additively" pattern from Module 6 — the new schema is additive, in a new location, not a change to a shared file.

---

## 5. Public API

All signatures below are final for this component. No implementation code — parameter/return/exception/threading/timeout behavior only.

### 5.1 `__init__(self, checkpoint_root: Path | None = None) -> None`

| | |
| --- | --- |
| **Parameters** | `checkpoint_root: Path | None` — root directory containing `groundingdino_swint_ogc.pth` and `GroundingDINO_SwinT_OGC.py`. Defaults to `vision_stack.loader.DEFAULT_CHECKPOINT_ROOT` when omitted, matching `ModelLoader`'s own default so a caller who bootstrapped with a custom `checkpoint_root` can pass the identical value here. |
| **Returns** | `None`. Constructing an instance never loads weights, imports `groundingdino`, or touches CUDA — fully lazy, matching the `_ComfyUIWebSocketTransport` convention of "construction never does I/O." |
| **Raises** | `ValueError` if `checkpoint_root` is given and is not a directory path (fail-fast constructor validation, matching the rest of the codebase's style) — this does not check file *existence*, only that the value is a plausible path; existence is checked at `ensure_loaded()` time. |
| **Thread-safety** | Constructor itself is not thread-safe to call concurrently with itself (no shared state to race, but not a supported pattern). One instance is intended to be owned by exactly one `RuntimeManager`/orchestrator, matching every other Vision Stack singleton-per-model convention. |

### 5.2 `is_loaded(self) -> bool`

| | |
| --- | --- |
| **Parameters** | None. |
| **Returns** | `True` if weights are currently resident (CPU or GPU) in this process; `False` before the first `ensure_loaded()` call or after `unload()`. Pure state read, no I/O. |
| **Raises** | Never raises. |

### 5.3 `ensure_loaded(self, registered_model: RegisteredVisionModel) -> None`

| | |
| --- | --- |
| **Parameters** | `registered_model: RegisteredVisionModel` — the entry yielded by `GPUResourceManager.reserve("grounding_dino")`. Must have `lifecycle_state == GPU_ACTIVE` (the reservation context manager guarantees this before yielding); `registered_model.config` supplies `checkpoint`, `precision`, `device`, `backend`, `timeout`, `fallback`. |
| **Returns** | `None`. No-op if `is_loaded()` is already `True` **and** the previously-loaded config's `checkpoint`/`precision`/`device` match `registered_model.config` — this idempotency mirrors `_ComfyUIWebSocketTransport.ensure_connected()`'s "no-op if already healthy" convention. If a mismatch is detected (e.g. a hot-reloaded config), the wrapper calls `unload()` internally before reloading, rather than raising. |
| **Behavior** | 1) Resolve the actual device via `_resolve_device` (§8). 2) Verify the checkpoint file and its `GroundingDINO_SwinT_OGC.py` config sidecar exist on disk (defense-in-depth re-check — see §1.3). 3) Build the model via `_build_model`: instantiate the GroundingDINO architecture from its config file, `torch.load` the checkpoint's `state_dict` (`weights_only=True`), call `.eval()`, move to the resolved device, cast to the configured precision (`fp16` → `.half()`; `fp32` → left as-is). 4) Store the model, resolved device, and config on `self`. |
| **Raises** | `VisionStackCheckpointError` (re-raised/wrapped from the vision_stack exception hierarchy, §9) if the checkpoint or its sidecar config is missing at load time despite passing bootstrap validation (on-disk corruption/deletion between bootstrap and first use). `GroundingDINOLoadError` (new, §9) if `torch.load` raises (corrupt/truncated checkpoint file, architecture/state-dict mismatch) or if CUDA is requested but unavailable (`torch.cuda.is_available()` is `False`) and the configured `fallback` is not a CPU-capable policy. |
| **Thread-safety** | Guarded internally by `self._load_lock` (an `RLock`) so two threads calling `ensure_loaded` concurrently on the same instance serialize rather than double-load; this is a defense-in-depth guard, not a substitute for the GPU Resource Manager's single-active-model enforcement, which callers must still use. |
| **Timeout behavior** | Bounded by `registered_model.config.timeout` (milliseconds, from `vision_stack.yaml`: `5000` for `grounding_dino`) wrapping the load call. Exceeding it raises `GroundingDINOLoadError` with the elapsed time in the message; there is no partial-load state — either the model is fully swapped in or `self._model` remains at its prior value (or `None`). |

### 5.4 `detect(self, image: np.ndarray, text_prompt: str, registered_model: RegisteredVisionModel, *, box_threshold: float | None = None, text_threshold: float | None = None) -> list[GroundingDINODetection]`

| | |
| --- | --- |
| **Parameters** | `image: np.ndarray` — RGB, `HxWx3`, `uint8`, original resolution (per the architecture contract's `Input: Thumbnail Image (RGB, original resolution)`). `text_prompt: str` — open-vocabulary label list, period-separated (e.g. `"person . face . logo . arrow . text"`), per the architecture contract's `TextPrompt`. `registered_model: RegisteredVisionModel` — same reservation-scoped entry as `ensure_loaded`; must be `GPU_ACTIVE`. `box_threshold` / `text_threshold` — optional per-call overrides of the module defaults defined in `config.py` (§11); when omitted, the configured `GROUNDING_DINO_BOX_THRESHOLD` / `GROUNDING_DINO_TEXT_THRESHOLD` are used. |
| **Returns** | `list[GroundingDINODetection]` (§7). Empty list (not an error) when no boxes clear the confidence floor for the given prompt — this is an explicit "expected guarantee" in the architecture document, not a failure. |
| **Behavior** | 1) Validate inputs (§6). 2) Call `ensure_loaded(registered_model)` if `not self.is_loaded()` — `detect()` is safe to call without a prior explicit `ensure_loaded()`, matching the "idempotent ensure" convention, though callers that want load latency isolated from inference latency (for metrics purposes) may call `ensure_loaded()` explicitly first. 3) Run `_run_inference`, which performs the forward pass under `torch.inference_mode()` (or `torch.no_grad()`) with the model's resolved precision/device. 4) Hand raw boxes/logits/phrases to `_GroundingDINOOutputParser.to_detections()` with the resolved image dimensions and the effective confidence floor. 5) Return the resulting list. |
| **Raises** | `ValueError` for invalid `image`/`text_prompt` (§6 — raised before any model interaction). `GroundingDINOInferenceError` (new, §9) wrapping any `torch` runtime exception during the forward pass, **except** CUDA out-of-memory, which is raised as the more specific `GroundingDINOOutOfMemoryError` (subclass, §9) so callers/orchestrators can apply the configured `fallback` policy (`skip_stage`) distinctly from a generic inference failure. `VisionStackResourceError` propagates unchanged if `registered_model` is not actually `GPU_ACTIVE` (defensive check — should be unreachable if the caller used `GPUResourceManager.reserve()` correctly, but checked explicitly rather than trusted, since this wrapper's contract must hold even if called incorrectly). |
| **Thread-safety** | Not thread-safe for concurrent `detect()` calls on the same instance — GroundingDINO's forward pass is not designed for concurrent invocation on one model instance, and the Vision Stack's single-active-model GPU lease already guarantees only one caller holds the `grounding_dino` reservation at a time, so this is enforced structurally rather than by an additional internal lock (unlike `ensure_loaded`, which does need its own lock because it can race with a `detect()` call from a different, incorrectly-written caller). |
| **Timeout behavior** | Bounded by `registered_model.config.timeout` (`5000` ms). A forward pass exceeding this raises `GroundingDINOInferenceError` with elapsed time in the message. There is no partial-result return on timeout — either `detect()` returns a complete `List[GroundingDINODetection]` or it raises. |

### 5.5 `unload(self) -> None`

| | |
| --- | --- |
| **Parameters** | None. |
| **Returns** | `None`. |
| **Behavior** | Idempotent — no-op if `not is_loaded()`. Otherwise: moves the model to CPU (if it was on GPU), drops the Python reference, calls `torch.cuda.empty_cache()` if CUDA was in use, and clears `self._model`/`self._model_config`/`self._device`. This method does **not** touch the registry's lifecycle state (`GPU_ACTIVE` → `CPU_CACHED` is `GPUResourceManager`'s job, already executed by the `reserve()` context manager's `finally` block on exit); it only releases the wrapper's own in-process handle. It exists primarily for the worker-restart path (§12) and for tests. |
| **Raises** | Never raises — any exception during CUDA cleanup is caught and logged at DEBUG (matching `_ComfyUIWebSocketTransport.close()`'s "never let cleanup mask a real error" convention), since a failure to cleanly free VRAM on an already-discarded model handle is not actionable. |
| **Thread-safety** | Guarded by `self._load_lock`, same lock as `ensure_loaded`. |

### 5.6 Deliberately absent from the public surface

- No `warmup()` method. Lazy loading via `ensure_loaded`/`detect` is the only load path — a separate warmup method would create two code paths for the same operation with no behavioral difference, since `ensure_loaded` is already idempotent and cheap to call ahead of time if a caller wants to isolate load latency.
- No `batch_detect()` / multi-image method. `vision_stack.yaml`'s `batch_size: 1` is enforced by `VisionModelConfig`'s own validator (`batch_size_must_be_one`) — the wrapper's single-image `detect()` signature is the only shape consistent with that constraint. A future architecture revision that lifts the batch-size-1 constraint would need a new design document, not a silent extension of this one.
- No direct `model` property exposing the raw GroundingDINO instance. Callers get detections, not model internals — this keeps the wrapper the single seam through which GroundingDINO's third-party API can change without breaking callers.

---

## 6. Input Validation

Performed in `detect()` before any model interaction (fail fast, no wasted GPU reservation time):

| Check | Failure |
| --- | --- |
| `image` is a `numpy.ndarray` | `ValueError("image must be a numpy.ndarray")` |
| `image.ndim == 3 and image.shape[2] == 3` (RGB, not grayscale/RGBA) | `ValueError("image must be RGB with shape (H, W, 3)")` |
| `image.dtype == np.uint8` | `ValueError("image must be uint8")` |
| `image.shape[0] > 0 and image.shape[1] > 0` | `ValueError("image must have non-zero height and width")` |
| `text_prompt` is a non-empty, non-whitespace `str` | `ValueError("text_prompt must not be empty")` — matches the architecture document's listed failure condition "malformed or empty TextPrompt" |
| `box_threshold`/`text_threshold`, if given, are in `(0.0, 1.0]` | `ValueError("box_threshold/text_threshold must be in (0.0, 1.0]")` |
| `registered_model.lifecycle_state == VisionModelLifecycleState.GPU_ACTIVE` | `VisionStackResourceError("GroundingDINOWrapper.detect called without an active GPU reservation")` |

---

## 7. Output Schema — Standard Detection Object Format

New additions to `modules/vision_stack/models.py` (not `modules/models.py` — see §4):

```
PixelBoundingBox
  x0: float   # absolute pixel, left edge,  0 <= x0 < x1
  y0: float   # absolute pixel, top edge,   0 <= y0 < y1
  x1: float   # absolute pixel, right edge, x1 <= image_width
  y1: float   # absolute pixel, bottom edge, y1 <= image_height

GroundingDINODetection
  label: str                         # matched phrase from text_prompt, lowercased, stripped
  confidence: float                  # raw sigmoid logit score, [0.0, 1.0]
  bounding_box: PixelBoundingBox
  source: Literal["grounding_dino"]  # fixed discriminator, always this literal value
```

Both models are `pydantic.BaseModel` with `model_config = ConfigDict(frozen=True)`, matching every other Vision Stack model (`VisionModelConfig`, `RegisteredVisionModel`, etc.) and the project-wide convention that stage-output objects are immutable value types.

**Expected guarantees** (per the architecture document, restated as this wrapper's contract):

- Every returned box satisfies `0 <= x0 < x1 <= image_width` and `0 <= y0 < y1 <= image_height` — clamped by `_GroundingDINOOutputParser._clamp_to_bounds`, never merely flagged.
- `confidence` is the raw sigmoid score in `[0.0, 1.0]`, not renormalized or rescaled.
- Boxes below the effective confidence floor are excluded from the list entirely — the returned list length is the count of *kept* detections, not the raw model output count.
- `label` values are drawn only from terms present in the caller's `text_prompt`; GroundingDINO's own phrase-grounding output is used as-is (lowercased/stripped for consistency), never post-processed into a different vocabulary.

---

## 8. Model Loading Strategy and Device Resolution

- **Loading trigger:** first `ensure_loaded()` or first `detect()` call after process start (or after a prior `unload()`), never at `RuntimeManager.bootstrap()` or `GroundingDINOWrapper.__init__()`.
- **Backend:** native PyTorch, per the architecture document's "Preferred inference backend" for GroundingDINO (`groundingdino-py`) — no ONNX Runtime path is specified in this phase; community ONNX exports are explicitly called out in the architecture doc as unvalidated for production, so this wrapper does not offer an ONNX branch.
- **Precision:** `registered_model.config.precision` (`fp16` per `vision_stack.yaml`). Weights are loaded in FP32 by `torch.load` (the checkpoint's native format) and cast with `.half()` immediately after `.eval()`, before the first forward pass — never mixed-precision autocast, to keep behavior deterministic and match the architecture document's flat "loaded in FP16" framing.
- **`_resolve_device(configured_device: str) -> str`:** returns `configured_device` unchanged (`"cuda:0"`) if `torch.cuda.is_available()`; otherwise returns `"cpu"` **only if** `registered_model.config.fallback` is a CPU-capable policy (`cpu_fallback` or `cpu_tiled_processing`) **and** logs a WARNING that GPU was unavailable and CPU fallback is being used. For `grounding_dino`'s configured fallback (`skip_stage`, per `vision_stack.yaml`), CUDA unavailability instead raises `GroundingDINOLoadError` immediately — `skip_stage` means "the orchestrator skips this stage entirely," which is a decision this wrapper signals via an exception, not one it silently substitutes for by running on CPU.
- **Checkpoint sidecar:** `GroundingDINO_SwinT_OGC.py` (the architecture's model-definition config, distinct from `vision_stack.yaml`) is loaded via the GroundingDINO library's own config-loading utility to construct the model skeleton before `state_dict` weights are applied — this file's path is `checkpoint_path.parent / "GroundingDINO_SwinT_OGC.py"`, exactly the sidecar path `ModelLoader._required_paths` already resolves and validates at bootstrap.

---

## 9. Error Handling — Exception Hierarchy

New module, `modules/vision_stack/grounding_dino_exceptions.py`, following the project's per-component typed-hierarchy convention (`module7_exceptions.py`, `vre_exceptions.py`):

```
VisionStackError                          (existing, vision_stack/exceptions.py — unchanged)
 └── GroundingDINOError                   (new base for this wrapper)
      ├── GroundingDINOLoadError          # checkpoint/config load failure, device unavailable
      ├── GroundingDINOInferenceError     # forward-pass failure (generic)
      │    └── GroundingDINOOutOfMemoryError   # CUDA OOM specifically — distinct so callers
      │                                          # can apply the configured fallback policy
      │                                          # without string-matching an error message
      └── GroundingDINOParseError         # raw model output could not be converted to
                                            # GroundingDINODetection (malformed tensor shapes)
```

`GroundingDINOError` subclasses `VisionStackError` (not `VisionStackRuntimeError`) because this wrapper's failures are model-level, not runtime-coordination-level — the existing `VisionStackRuntimeError`/`VisionStackResourceError` pair remains reserved for `RuntimeManager`/`GPUResourceManager` lifecycle violations (e.g. "GPU already reserved"), which this wrapper still lets propagate unchanged when it detects them (§5.4).

**Translation rules** (every third-party exception is caught at the point it can occur and re-raised chained, `raise ... from exc`, matching the `_ComfyUIWebSocketTransport` convention of never leaking a raw library exception):

| Origin | Caught | Re-raised as |
| --- | --- | --- |
| `torch.load` in `_build_model` | `RuntimeError`, `pickle.UnpicklingError`, `FileNotFoundError` | `GroundingDINOLoadError` |
| GroundingDINO library's config/model constructor | any `Exception` | `GroundingDINOLoadError` |
| `.eval()` / `.half()` / `.to(device)` | `RuntimeError` (e.g. CUDA init failure) | `GroundingDINOLoadError` |
| Forward pass, generic | `RuntimeError` not matching the OOM signature below | `GroundingDINOInferenceError` |
| Forward pass, CUDA OOM | `torch.cuda.OutOfMemoryError` (or `RuntimeError` containing `"out of memory"` on older torch versions — checked via `isinstance` first, string-match only as a fallback) | `GroundingDINOOutOfMemoryError` |
| Output parsing | `IndexError`, `ValueError`, shape-mismatch `RuntimeError` inside `_GroundingDINOOutputParser` | `GroundingDINOParseError` |

**Fallback policy application:** this wrapper raises typed exceptions; it does **not** itself decide to skip the stage. Applying `VisionModelFallback.SKIP_STAGE` (the configured policy for `grounding_dino`) is the calling orchestrator's responsibility — it catches `GroundingDINOOutOfMemoryError`/`GroundingDINOInferenceError`, reads `registered_model.config.fallback`, and decides whether to proceed with an empty detection list or halt the pipeline. This keeps the fallback *policy* (data, in `vision_stack.yaml`) separate from fallback *mechanism* (code, in the orchestrator), matching how `GPUResourceManager.cpu_fallback_metadata()` already only *reports* fallback eligibility rather than *acting* on it.

---

## 10. Retry Strategy

None, by design. Unlike Module 7's ComfyUI transports (which retry transient network failures against a long-running external server), GroundingDINO inference is in-process and synchronous — a `GroundingDINOOutOfMemoryError` will not resolve on an identical retry without first freeing VRAM (which is the eviction/fallback orchestrator's job, not this wrapper's), and a `GroundingDINOParseError` indicates a code-level shape mismatch that a retry cannot fix. Introducing a Tenacity layer here would mask both failure classes behind unhelpful retry delay. This is a deliberate divergence from Module 7's pattern, called out explicitly so an implementer does not assume Tenacity is required everywhere in the codebase.

---

## 11. Configuration Requirements

No changes to `vision_stack.yaml` or `VisionModelConfig` — GroundingDINO's checkpoint/precision/device/timeout/fallback are already fully specified there and validated by the existing schema.

New additions to `modules/config.py`, in a new section following the existing `# --- AI Vision Stack V2.1 - Configuration Architecture ---` marker (additive only, matching the project's "never modify existing files additively" — beyond appending — convention already used for Module 6):

```python
# ---------------------------------------------------------------------------
# AI Vision Stack V2.1 - GroundingDINO Wrapper
# ---------------------------------------------------------------------------

#: Minimum detection confidence to keep a box, per the architecture document's
#: documented confidence floor for GroundingDINO.
GROUNDING_DINO_BOX_THRESHOLD: float = 0.35

#: Minimum per-token text-grounding confidence (GroundingDINO's own internal
#: text-matching threshold, distinct from the output box_threshold above).
GROUNDING_DINO_TEXT_THRESHOLD: float = 0.25

#: Default open-vocabulary prompt when a caller does not supply one.
GROUNDING_DINO_DEFAULT_PROMPT: str = "person . face . logo . text . arrow"

#: Log file for this wrapper, following the one-log-file-per-component
#: convention already used by every other module (module1.log ... module6_5.log).
VISION_STACK_GROUNDING_DINO_LOG_PATH: Path = LOG_DIR / "vision_stack_grounding_dino.log"
```

`GROUNDING_DINO_BOX_THRESHOLD` is the module-level default consumed by `detect()`'s `box_threshold` parameter when the caller passes `None`; it is not part of `VisionModelConfig`/`vision_stack.yaml` because it is a detection-quality tuning knob specific to this one model's algorithm, not a cross-model bootstrap/lifecycle concern like `precision`/`device`/`timeout` — the same reasoning `VRE_FACE_DETECTION_CONFIDENCE` already follows for the (unrelated) VRE face processor.

**New dependency**, appended to `requirements.txt` under a new `# ── AI Vision Stack V2.1 (GroundingDINO) ──` section:

```
groundingdino-py>=0.4.0
```

(`torch`/`torchvision` are already present from Module 4 and satisfy GroundingDINO's own dependency on them; no version conflict is expected since both already pin `>=2.1.0`/`>=0.16.0`.)

---

## 12. Runtime Lifecycle Integration

- **Registration:** already handled — `grounding_dino` is registered by `ModelRegistry.register_stack()` at `RuntimeManager.bootstrap()` time, before this wrapper exists. No change needed here.
- **Reservation:** the calling orchestrator obtains the lease via `runtime_manager.reserve_model("grounding_dino")` (delegates to `GPUResourceManager.reserve`), which performs the `REGISTERED`/`CPU_CACHED` → `GPU_ACTIVE` transition and sets placeholder `runtime_state` (`gpu_reserved=True, cuda_executed=False, weights_loaded=False`) **before** yielding control to the wrapper.
- **Reporting real state:** inside the reservation, after `ensure_loaded()` succeeds, the wrapper (or the orchestrator on its behalf — see note below) calls `registry.update_runtime_state("grounding_dino", {"weights_loaded": True})`; after a successful `detect()` forward pass, `{"cuda_executed": True}`. This is the first point in the whole Vision Stack where `runtime_state` reflects genuine model activity rather than the resource manager's hardcoded `False` placeholders — a fact worth flagging explicitly since `tests/test_vision_stack_runtime.py`'s existing assertions (e.g. `runtime_state["cuda_executed"] is False`) describe *pre-wrapper* behavior and are expected, correct, and must not be "fixed" by this change; they test the resource manager in isolation, without any wrapper attached.

  *Design note:* whether the wrapper itself calls `registry.update_runtime_state(...)` (requiring it to hold a `ModelRegistry` reference, passed into `__init__` or into `detect()`) or whether the calling orchestrator does so after `detect()` returns (keeping the wrapper registry-agnostic) is an open orchestration-layer decision **deferred to the orchestrator's own design document**, since no such orchestrator exists yet. This document's `detect()`/`ensure_loaded()` signatures (§5) deliberately do not take a `ModelRegistry` parameter, keeping the wrapper testable without any registry at all (§13); the recommended default is that the orchestrator performs the `update_runtime_state` calls, treating the wrapper as a pure function of `(image, text_prompt, registered_model) -> detections` plus incidental in-process caching of loaded weights.
- **Release:** on reservation exit, `GPUResourceManager._release` already transitions `GPU_ACTIVE` → `CPU_CACHED` and resets `runtime_state` to `{"gpu_reserved": False, "cuda_executed": False, "weights_loaded": False}` regardless of what the wrapper reported mid-reservation — this is existing, unmodified behavior and is intentional: `runtime_state` reflects "is this true *right now*, in this reservation," not a persistent history. The wrapper's own `is_loaded()` (§5.2) is the source of truth for "are weights still resident in this process," independent of the registry's per-reservation bookkeeping.
- **Worker restart:** on `RuntimeManager.shutdown()` (which calls `evict_all()`), the orchestrator is responsible for calling `wrapper.unload()` before the process actually recycles, so CUDA memory is released ahead of process exit rather than relying solely on OS-level cleanup — this mirrors the architecture document's "Worker Restart Threshold" flushing "CPU cache, GPU allocator, and Python heap."
- **Sequential execution guarantee:** because `GPUResourceManager.reserve()` raises `VisionStackResourceError` if the GPU is already held by another model, `GroundingDINOWrapper.detect()` can assume — and does not need to separately re-verify — that no other Vision Stack model is concurrently executing on the GPU while it runs. Combined with the `GPU_ACTIVE` check in §6, this is what keeps peak VRAM bounded by one model at a time, per the architecture document's headline VRAM-management strategy.

---

## 13. GPU Memory Management

- **Expected VRAM:** ~2.0 GB FP16, per the architecture document's Model Checkpoint Specification for GroundingDINO. This wrapper does not itself enforce a VRAM budget check (that is the architecture's broader "Health Checks" responsibility, per the document's Worker Lifecycle section, evaluated across the whole worker process, not per-model) — it only ensures its *own* footprint is released promptly via `unload()`/`empty_cache()`.
- **No persistent GPU residency across reservations:** consistent with the single-active-model design, the wrapper does not attempt to keep weights on GPU between `detect()` calls if the reservation has been released — `GPUResourceManager._release` transitions the model back to `CPU_CACHED` at the registry level, and the recommended (though not enforced by this wrapper alone) orchestrator behavior is to call `wrapper.unload()` — or at minimum move the model tensor to CPU — whenever the reservation exits, so VRAM is actually freed rather than merely "logically" marked `CPU_CACHED` while physically still resident on the GPU. This document flags this as a responsibility split worth making explicit in the orchestrator's own design: the registry's lifecycle state and the wrapper's actual VRAM residency are two different things that must be kept in sync by whatever code holds both.
- **OOM handling:** on `GroundingDINOOutOfMemoryError` (§9), the wrapper's `_run_inference` calls `torch.cuda.empty_cache()` in a `finally` block before the exception propagates, to avoid leaving fragmented allocations that could cause the *next* model's reservation to also OOM — this is a defensive best-effort cleanup, not a guarantee, and does not retry the failed call (§10).
- **FP16 only:** no dynamic precision fallback (e.g. dropping to FP32 on OOM, or vice versa) is in scope — `precision` is fixed by `vision_stack.yaml` and treated as configuration, not as a runtime-adjustable OOM mitigation.

---

## 14. Thread Safety

| Aspect | Guarantee |
| --- | --- |
| `ensure_loaded` / `unload` | Internally serialized via `self._load_lock` (`RLock`) — safe to call from multiple threads, though only one load/unload proceeds at a time and the others block. |
| `detect` | Not internally locked. Safe under the *intended* usage pattern (one caller at a time, holding the GPU reservation) because `GPUResourceManager.reserve()` structurally prevents two threads from holding the `grounding_dino` reservation simultaneously (`GPUResourceManager._gpu_lock`, an `RLock` shared across the whole registry). Calling `detect()` on the same `GroundingDINOWrapper` instance from two threads *without* both going through a reservation is explicitly unsupported and unchecked by this class — the same trust boundary the rest of `vision_stack` already places on correct caller usage (e.g. `ModelRegistry`'s own internal `RLock` protects registry *state*, not caller *behavior*). |
| Instance sharing | One `GroundingDINOWrapper` instance is intended to be constructed once per `RuntimeManager` process (a singleton in practice, though this document does not enforce singleton-ness at the class level — that is an orchestrator wiring decision) and reused across every `detect()` call for that worker's lifetime, until `unload()`/process restart. |

---

## 15. Performance Considerations

- **Expected inference speed:** ~80 ms per thumbnail on an RTX 4060-class GPU, per the architecture document — this wrapper introduces no additional per-call overhead beyond ordinary Python/PyTorch dispatch (no extra serialization, no network I/O, unlike the Module 7 ComfyUI path).
- **`torch.inference_mode()`** (preferred over the older `torch.no_grad()` where the installed `torch` version supports it) wraps the forward pass to avoid autograd bookkeeping overhead, since this is inference-only, never training.
- **Lazy load amortization:** the ~694 MB checkpoint load happens once per worker process (or once per config change), not once per thumbnail — `ensure_loaded`'s idempotency (§5.3) is what makes the stated "~80 ms per thumbnail" figure achievable, since a naive re-load-per-call design would dominate wall-clock time with disk I/O instead.
- **No batching:** consistent with the `batch_size: 1` constraint (§5.6), each `detect()` call processes exactly one image; throughput scaling is out of scope for this phase and would require a new `VisionModelConfig` schema revision.

---

## 16. Dependency Management

| Dependency | Already present? | Notes |
| --- | --- | --- |
| `torch>=2.1.0`, `torchvision>=0.16.0` | Yes (Module 4) | No version bump required; GroundingDINO's own `torch` floor is satisfied. |
| `groundingdino-py>=0.4.0` | **No — new** | Added to `requirements.txt` (§11). Pulls in its own transitive deps (`transformers`-adjacent tokenizer utilities, `addict`, `yapf`, `timm`) which are not currently in the project; implementer should pin/verify these resolve cleanly against the existing `torch` version at install time before this wrapper is implemented. |
| `numpy>=1.24.0` | Yes | Already required by Module 4; used for the `image: np.ndarray` input contract. |
| `loguru>=0.7.0` | Yes | Standard project-wide logging. |
| `pydantic>=2.0.0` | Yes | For `GroundingDINODetection`/`PixelBoundingBox`. |

No new dependency on `tenacity` for this wrapper (§10 — no retry layer).

---

## 17. Logging Strategy

Follows the exact `_configure_logger()` pattern already established in `modules/comfyui_client.py` / `modules/image_generator.py` / `modules/visual_reference_engine.py`: a module-level function, called once at import time, attaching a rotating Loguru sink.

```python
_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"

def _configure_logger() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(VISION_STACK_GROUNDING_DINO_LOG_PATH),
        rotation="10 MB",
        retention="30 days",
        format=_LOG_FORMAT,
        level="DEBUG",
        enqueue=True,
    )

_configure_logger()
```

**Critical, project-specific caveat** (from the Module 2 bugfix history — see §1.3's on-disk-corruption re-check and the codebase's known `enqueue=True` pitfall): because `enqueue=True` serializes log records through a background process/thread via `pickle`, **no log call in this wrapper may pass a raw exception object or a raw traceback as a positional/keyword value** (e.g. `logger.error("failed", exc=some_exception)` is forbidden). Every log call must pass only plain strings, numbers, and already-`str()`-converted values — exactly the fix already applied in Module 2's `youtube_metadata.py` for the same class of Loguru serialization failure. Exception *messages* are logged as `str(exc)`, never the exception instance itself.

| Event | Level | Example |
| --- | --- | --- |
| Weight load start/success | INFO | `"Loading GroundingDINO checkpoint device={device} precision={precision}"` |
| Weight load failure | ERROR | `"GroundingDINO load failed: {error_message}"` (string, not the exception object) |
| Inference start | DEBUG | `"GroundingDINO detect prompt='{prompt}' image_shape={shape}"` |
| Inference result | DEBUG | `"GroundingDINO kept {n} of {raw_n} raw detections above threshold={threshold}"` |
| CUDA OOM | ERROR | `"GroundingDINO CUDA OOM during inference: {error_message}"` |
| Fallback-relevant failure | WARNING | `"GroundingDINO inference failed; configured fallback={fallback_policy}"` |
| Unload | DEBUG | `"GroundingDINO weights released, device={device}"` |

No `GenerationMetrics`-style structured metrics file is introduced for this wrapper (unlike Module 7's `_ComfyUIMetricsRecorder`) — that is out of scope until an orchestrator/pipeline-level metrics design exists; this wrapper's logging is Loguru-only.

---

## 18. Integration Points with the Existing Runtime — Summary Table

| Existing component | Integration |
| --- | --- |
| `ModelRegistry` | Wrapper reads `RegisteredVisionModel.config` (never calls `register`/`transition` itself — those remain the registry's and `GPUResourceManager`'s exclusive responsibility). |
| `GPUResourceManager` | Wrapper's `detect`/`ensure_loaded` are only ever called from inside a `reserve()` context; wrapper checks `lifecycle_state == GPU_ACTIVE` defensively. |
| `RuntimeManager` | Not directly referenced by the wrapper; the future orchestrator uses `RuntimeManager.reserve_model` / `run_sequential`, which is unchanged by this document. |
| `ModelLoader` / bootstrap metadata | Wrapper trusts `ModelLoader`'s bootstrap-time checkpoint validation; re-checks file existence only defensively at load time (§1.3, §5.3). |
| `vision_stack/exceptions.py` | `GroundingDINOError` subclasses `VisionStackError`; `VisionStackResourceError` propagates unchanged for misuse. |
| `vision_stack/models.py` | Additive only — new `GroundingDINODetection`/`PixelBoundingBox` models; no existing model changed. |
| `modules/config.py` | Additive only — new constants appended in a new section (§11); no existing constant changed. |
| `modules/models.py` | **Untouched** — the naming-collision finding in §4 means this file is deliberately not modified by this design. |
| `requirements.txt` | Additive only — new `groundingdino-py` line in a new section. |

---

## 19. Sequence Diagram — Complete Execution Flow

```
Orchestrator          RuntimeManager      GPUResourceManager     ModelRegistry      GroundingDINOWrapper        torch/GroundingDINO lib
     │                      │                     │                    │                     │                          │
     │ reserve_model(       │                     │                    │                     │                          │
     │  "grounding_dino")   │                     │                    │                     │                          │
     ├─────────────────────►│                     │                    │                     │                          │
     │                      │ reserve(name)       │                    │                     │                          │
     │                      ├────────────────────►│                    │                     │                          │
     │                      │                     │ get(name)          │                     │                          │
     │                      │                     ├───────────────────►│                     │                          │
     │                      │                     │◄───────────────────┤ RegisteredVisionModel│                         │
     │                      │                     │ transition→GPU_ACTIVE                    │                          │
     │                      │                     ├───────────────────►│                     │                          │
     │                      │                     │ update_runtime_state (weights_loaded=False,│                        │
     │                      │                     │                     cuda_executed=False)  │                         │
     │                      │                     ├───────────────────►│                     │                          │
     │◄─────────────────────┴─────────────────────┤  yields RegisteredVisionModel             │                          │
     │  (inside "with" block)                     │                    │                     │                          │
     │                                                                                        │                          │
     │ ensure_loaded(registered_model) ─────────────────────────────────────────────────────►│                          │
     │                                                                                        │ _resolve_device()        │
     │                                                                                        │ verify checkpoint exists │
     │                                                                                        │ _build_model() ─────────►│
     │                                                                                        │◄───── model instance ────┤
     │                                                                                        │ .eval().half().to(device)│
     │◄───────────────────────────────────────────────────────────────────────────────────────┤ (None — loaded)          │
     │                                                                                        │                          │
     │ detect(image, text_prompt, registered_model) ────────────────────────────────────────►│                          │
     │                                                                                        │ validate inputs (§6)     │
     │                                                                                        │ (already loaded — skip)  │
     │                                                                                        │ torch.inference_mode():  │
     │                                                                                        │  forward pass ──────────►│
     │                                                                                        │◄──── boxes, logits, ─────┤
     │                                                                                        │        phrases          │
     │                                                                                        │ _GroundingDINOOutputParser│
     │                                                                                        │   .to_detections(...)   │
     │◄───────────────────────────────────────────────────────────────────────────────────────┤ List[GroundingDINODetection] │
     │                                                                                        │                          │
     │ (orchestrator, optional) update_runtime_state({weights_loaded: True, cuda_executed: True})                        │
     ├─────────────────────────────────────────────►│───────────────────►│                    │                          │
     │                                                                                        │                          │
     │ ── exit "with" block ──                     │                    │                     │                          │
     │                      │                     │ transition→CPU_CACHED                     │                          │
     │                      │                     ├───────────────────►│                     │                          │
     │                      │                     │ update_runtime_state (all False)          │                          │
     │                      │                     ├───────────────────►│                     │                          │
```

**Error-path variant** (CUDA OOM during forward pass):

```
     │ detect(...) ──────────────────────────────────────────────────────────────────────────►│
     │                                                                                        │ forward pass ───────────►│
     │                                                                                        │◄── torch.cuda.OutOfMemoryError │
     │                                                                                        │ torch.cuda.empty_cache() │
     │◄──────────────────────────────────────────────────────────────────────────── raises ────┤ GroundingDINOOutOfMemoryError │
     │  (orchestrator reads registered_model.config.fallback == "skip_stage")                 │                          │
     │  (orchestrator decides: skip Stage 1, proceed pipeline with empty detections)           │                          │
     │ ── exit "with" block (still releases the reservation normally, via GPUResourceManager's │
     │     finally block, regardless of the exception) ──                                     │                          │
```

---

## 20. Testing Strategy

New test file: `tests/test_grounding_dino_wrapper.py`, following the existing `tests/test_vision_stack_runtime.py` conventions (`sys.path.insert` of the `modules` directory, `tmp_path`-based checkpoint fixtures).

### 20.1 Unit tests — `_GroundingDINOOutputParser` (no GPU, no checkpoint required)

- Given synthetic `boxes_cxcywh`/`logits`/`phrases` tensors (plain lists/arrays, no real model), verify `to_detections` returns the correct pixel-space `PixelBoundingBox` for known cx/cy/w/h → x0/y0/x1/y1 conversions.
- Verify boxes below `confidence_floor` are excluded, not merely flagged (matches the architecture document's expected guarantee).
- Verify a box computed to extend beyond image bounds is clamped, not dropped.
- Verify `label` values are lowercased/stripped and only ever drawn from the supplied `phrases`.
- Verify empty input tensors produce an empty list without raising.

### 20.2 Unit tests — `GroundingDINOWrapper` construction and validation (no GPU, no checkpoint required)

- `__init__` with default and explicit `checkpoint_root`; verify no file I/O or CUDA call occurs (mock `torch`/checkpoint access and assert zero calls).
- `is_loaded()` returns `False` before any load.
- `detect()` raises `ValueError` for each invalid-input case in §6's table (wrong dtype, wrong ndim, empty prompt, out-of-range thresholds) — verify none of these reach `ensure_loaded`/`_build_model` (mock and assert not-called).
- `detect()` raises `VisionStackResourceError` when given a `RegisteredVisionModel` whose `lifecycle_state` is not `GPU_ACTIVE` (construct one directly via `ModelRegistry.register` without transitioning it).

### 20.3 Integration tests — mocked model, real registry/resource-manager (no GPU, no real checkpoint)

Using `unittest.mock.patch` on `_build_model`/the underlying GroundingDINO library import, so these tests run in CI without a GPU or the real ~694 MB checkpoint (mirroring how `tests/test_vision_stack_runtime.py` uses `tmp_path` fixture checkpoint files rather than real weights):

- Full flow: `build_registry` → `GPUResourceManager.reserve("grounding_dino")` → `wrapper.ensure_loaded` → `wrapper.detect` → assert a well-formed `List[GroundingDINODetection]` is returned and the reservation's lifecycle transitions match `test_vision_stack_runtime.py`'s existing assertions (`GPU_ACTIVE` inside the `with`, `CPU_CACHED` after).
- `ensure_loaded` idempotency: call twice with the same config; assert the (mocked) model-construction call happens exactly once.
- `ensure_loaded` reload-on-mismatch: call once, then again with a `RegisteredVisionModel` carrying a different `precision`; assert `unload` is invoked internally before the second load.
- Simulated `torch.cuda.OutOfMemoryError` from the mocked forward pass: assert `GroundingDINOOutOfMemoryError` is raised, `torch.cuda.empty_cache` (mocked) was called, and the GPU reservation is still released cleanly (registry ends in `CPU_CACHED`, not stuck in `GPU_ACTIVE`) — verifies the `reserve()` context manager's `finally` block still runs correctly even though this wrapper raised.
- Simulated malformed checkpoint (`torch.load` mock raises `RuntimeError`): assert `GroundingDINOLoadError`, chained (`__cause__` is the original `RuntimeError`).
- `unload()` after a successful load: assert `is_loaded()` becomes `False` and the mocked CUDA-cleanup call fires.
- Empty-detections path: mock the forward pass to return zero boxes above threshold; assert `detect()` returns `[]`, not an exception (this is a load-bearing test — the architecture doc explicitly calls this an expected, non-error outcome).

### 20.4 What is explicitly **not** tested by this suite

- Real GroundingDINO inference correctness/accuracy against real images — that requires the actual checkpoint and a GPU, and belongs to a separate manual/smoke-test script (mirroring `scripts/verify_comfyui_http.py`'s role for Module 7), not the automated `pytest` suite, consistent with how the project has kept CI GPU-free so far (`onnxruntime-gpu` already documented in `requirements.txt` as falling back to CPU automatically).
- Multi-threaded concurrent `detect()` calls on one instance — explicitly unsupported usage (§14), not a contract this suite needs to defend.

---

## 21. Future Extensibility — The Repeatable Pattern for Remaining Models

The six remaining Vision Stack models (Florence-2, PaddleOCR, OpenCLIP, InsightFace, SAM 2, Depth Anything V2, TEED — seven, matching `VisionModelConfig`'s full field list) each get their own wrapper module (`florence2.py`, `paddleocr.py`, `openclip.py`, `insightface.py`, `sam2.py`, `depth_anything.py`, `teed.py`) inside `modules/vision_stack/`, each following the exact shape fixed by this document:

1. **One public wrapper class** per model (`Florence2Wrapper`, `PaddleOCRWrapper`, `OpenCLIPWrapper`, `InsightFaceWrapper`, `SAM2Wrapper`, `DepthAnythingWrapper`, `TEEDWrapper`), each with `__init__(checkpoint_root=None)`, `is_loaded()`, `ensure_loaded(registered_model)`, `unload()`, and one primary inference method whose name matches the model's role (`caption()`/`analyze()` for Florence-2, `recognize()` for PaddleOCR, `embed()` for OpenCLIP, `identify()` for InsightFace, `segment()` for SAM 2 — taking `List[GroundingDINODetection]` box prompts per the architecture's inter-module contract, `estimate_depth()` for Depth Anything V2, `detect_edges()` for TEED).
2. **One module-private stateless output parser** per model (`_Florence2OutputParser`, etc.), splitting "raw model tensor → typed Pydantic result" from "own the loaded model handle," exactly as `_GroundingDINOOutputParser` does here.
3. **One new, uniquely-named output schema** per model in `vision_stack/models.py` (never reusing/renaming `modules/models.py`'s legacy Module 4/5 schemas — §4's naming-collision finding applies to every future wrapper equally, e.g. SAM 2's `SegmentedObject` per the architecture doc must not collide with any existing segmentation-related model already in `modules/models.py`).
4. **One typed exception hierarchy** per model in `vision_stack/<model>_exceptions.py`, each subclassing `VisionStackError` directly (not `VisionStackRuntimeError`), with a `<Model>LoadError` / `<Model>InferenceError` / `<Model>OutOfMemoryError` (where the model is GPU-resident and OOM is plausible) / `<Model>ParseError` shape.
5. **No retry layer**, for the same reason given in §10, unless a future model's design document identifies a genuinely transient failure mode this document's models don't have (none of the seven remaining models involve network I/O, so this is expected to hold across all of them).
6. **Identical lifecycle integration**: constructed once per `RuntimeManager` process, loaded lazily, used only inside `GPUResourceManager.reserve(name)`, `runtime_state` reporting deferred to the orchestrator (§12's design note applies identically to every wrapper — this is intentionally decided once, here, rather than re-litigated per model).
7. **Identical logging convention**: one new `VISION_STACK_<MODEL>_LOG_PATH` constant per model in `config.py`, one `_configure_logger()` function per wrapper module, same rotation/retention/format, same "no raw exception objects through `enqueue=True`" caveat repeated verbatim in each wrapper's logging section.
8. **Identical testing shape**: parser unit tests (no GPU) → wrapper validation unit tests (no GPU) → mocked integration tests against the real registry/resource-manager (no GPU, no real checkpoint) → an explicit "not tested here" callout for real-checkpoint accuracy, deferred to a manual smoke script.

Model-specific deltas each future design document must still resolve on its own (this document does not prejudge them): SAM 2 and Depth Anything V2's larger VRAM footprints may warrant tiled/CPU-fallback code paths this wrapper's `skip_stage` policy does not need; PaddleOCR's `backend: paddle` means its `_build_model`/dependency section looks structurally different from every `pytorch`-backend wrapper (including this one); OpenCLIP's `embed()` output is a fixed-length vector, not a list of boxes, so its output schema (§7-equivalent) is shaped completely differently from every detection/segmentation-style wrapper; SAM 2's `segment()` takes this wrapper's `List[GroundingDINODetection]` as an *input* (per the architecture's Stage 2 contract), making it the one wrapper in the set with a direct data dependency on another wrapper's output type rather than only on the shared registry/resource-manager infrastructure.
