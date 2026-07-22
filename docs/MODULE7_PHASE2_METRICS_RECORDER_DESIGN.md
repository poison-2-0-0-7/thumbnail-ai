# Module 7 Phase 2 — `_ComfyUIMetricsRecorder` ("MetricsRecorder") Design Specification

**Status:** Draft — implementation-ready
**Component:** `_ComfyUIMetricsRecorder`
**Depends on (complete, do not modify):** `_ComfyUIHTTPTransport`, `_ComfyUIWebSocketTransport`, `ComfyUIEvent`, `_CompletionOutcome`, `_CompletionResult`, `_QueueTracker`, `_ImageCandidate`, `_OutputResult`, `_OutputRetriever`, `module7_exceptions.py`, `models.GenerationMetrics`, `image_generator.MetricsCollector`, `image_generator.utc_now`
**Consumed by (not implemented here):** `ComfyUIClient.generate()`

> **Note on source material.** Unlike the earlier Phase 2 sub-documents, this one was written with direct read access to the live repository (`poison-2-0-0-7/thumbnail-ai`, current `main`). Every interface named below — `_CompletionResult`, `_CompletionOutcome`, `_OutputResult`, `_ImageCandidate`, the `OutputRetrievalError` hierarchy, `GenerationMetrics`, `MetricsCollector`, `utc_now`, and every `config.py` constant cited — was read from the actual source files, not assumed. The only genuinely new things introduced by this document are the `_ComfyUIMetricsRecorder` class itself and its private, module-internal helper types; everything it consumes already exists and is treated as frozen. Two prior documents (`MODULE7_PHASE2_COMFYUI_INTEGRATION_DESIGN.md`, an earlier planning document written before Phase 2 implementation began) sketched a `_ComfyUIMetricsRecorder.record_attempt(video_id, niche, workflow_ref, outcome)` shape and a Tenacity-based OOM retry loop; the codebase as actually implemented does **not** use Tenacity anywhere in `comfyui_client.py` (retries are hand-rolled loops with `time.sleep`, e.g. `_OutputRetriever._download_image`), and no `ComfyUIClient` facade, `_AttemptOutcome` type, or OOM-retry loop exists yet. This document designs `_ComfyUIMetricsRecorder` against what is *actually present* today, and deliberately avoids depending on any shape for `ComfyUIClient.generate()` that hasn't been built, per the task brief's explicit instruction not to design `generate()` yet. Anywhere this document had to make a forward-looking choice because the consuming code doesn't exist yet, it is marked **[ASSUMED]** with its rationale.

---

## 1. Responsibilities

### 1.1 What `_ComfyUIMetricsRecorder` owns

1. Translating the raw, low-level outputs already produced by `_QueueTracker` (`_CompletionResult`, one or more — see §4.2 on retries) and `_OutputRetriever` (`_OutputResult`, on success) plus a small amount of caller-supplied context (video/niche/workflow identity, candidate/retry counts, optional VRAM readings) into exactly one `models.GenerationMetrics` record.
2. Classifying *why* an attempt failed into a short, stable `failure_reason` string, for every failure shape Phase 2's completed components can currently produce (§5.4).
3. Appending that one `GenerationMetrics` record through the existing, unmodified `image_generator.MetricsCollector.append()`.
4. Guaranteeing that recording metrics can never raise an exception that would replace or mask the real generation outcome the caller is returning or raising (§8.4) — matching `MetricsCollector`'s own docstring: *"a passive observer, never a control-flow participant."*
5. Producing exactly one JSONL-appended record per logical `ComfyUIClient.generate()`-style attempt, even when that attempt internally involved multiple submissions (e.g. an OOM retry loop, once built) — see §4.2's `completions: Sequence[_CompletionResult]` design.

### 1.2 What `_ComfyUIMetricsRecorder` does NOT own

- **Does not** submit prompts, track queue state, or poll ComfyUI. It has zero dependency on `_ComfyUIHTTPTransport` or `_ComfyUIWebSocketTransport` — it never imports either.
- **Does not** decide *when* to retry an OOM-classified attempt, how many times, or with which `GenerationProfile`. It only records how many retries *already happened*, as reported by its caller.
- **Does not** decide *whether* a `_CompletionResult`/exception pair is retryable. That classification (`ComfyUIConnectionError` vs. `VRAMExhaustedError` vs. terminal) belongs to the not-yet-built `ComfyUIClient.generate()`. `_ComfyUIMetricsRecorder` only maps whatever outcome/exception it is handed into a `failure_reason` label for observability — it never raises a *different* exception type than the one it was given, and never changes control flow based on retryability.
- **Does not** write `ImageGenerationResult` manifests. `ArtifactWriter` (existing, Phase 1) remains the sole owner of that, invoked once per finished, QA'd result — a Phase 3/4 concern.
- **Does not** measure VRAM itself. `peak_vram_mb`/`gpu_utilization_percent` are accepted as caller-supplied optional values (the caller may separately call `_ComfyUIHTTPTransport.system_stats()` before/after `generate()`); `_ComfyUIMetricsRecorder` never calls `system_stats()` itself.
- **Does not** populate `identity_failures_count`, `qa_failures_count`, `winning_overall_score`, or `winning_signal_scores` on `GenerationMetrics`. These are left at their Pydantic model defaults (`0`, `0`, `None`, `{}`) by Phase 2, for Phase 3/4 to fill in later, matching the field-ownership split the (pre-implementation) integration design document already laid out and which this document does not change.
- **Does not** hold a connection, session, socket, or any per-attempt mutable state across calls. It is safe to construct once per `ComfyUIClient` instance (or once per process) and reuse across every `record_attempt()` call.

---

## 2. Architecture

### 2.1 Dependency graph

```
ComfyUIClient.generate()  (future — not designed here)
        │
        ├── uses ──> _QueueTracker.await_completion()   ──> _CompletionResult (one per submission attempt)
        │
        ├── uses ──> _OutputRetriever.retrieve()          ──> _OutputResult (on success only)
        │
        └── uses ──> _ComfyUIMetricsRecorder.record_attempt(
                          video_id, niche, workflow_version, workflow_hash, profile_name,
                          completions=[...one _CompletionResult per submission attempt...],
                          output=<_OutputResult or None>,
                          exception=<the exception the caller is about to raise, or None>,
                          num_candidates_requested, identity_retry_count,
                          peak_vram_mb, gpu_utilization_percent, attempt_started_at,
                     )
                          │
                          ├── builds ──> models.GenerationMetrics  (existing Pydantic model, unmodified)
                          └── calls  ──> image_generator.MetricsCollector.append(metrics)  (existing, unmodified)
```

- `_ComfyUIMetricsRecorder` depends on: the *shapes* of `_CompletionResult`, `_CompletionOutcome`, `_OutputResult` (read-only — it never constructs or mutates any of them), the `module7_exceptions` hierarchy (read-only, for `isinstance` classification), `models.GenerationMetrics`, and `image_generator.MetricsCollector`.
- It does **not** depend on `_ComfyUIHTTPTransport`, `_ComfyUIWebSocketTransport`, `_QueueTracker`, or `_OutputRetriever` as *classes* — it never imports them, never calls a method on them, and never receives an instance of them. It only receives the plain data (`_CompletionResult`, `_OutputResult`) those classes already produce. This is the same "consumes data, not the producer" boundary `_OutputRetriever`'s own design document establishes for itself relative to `_QueueTracker` (see `MODULE7_PHASE2_OUTPUT_RETRIEVER_DESIGN.md` §2.1–§2.3).
- No component in this graph imports `_ComfyUIMetricsRecorder`. It sits at the end of the dependency chain: `_ComfyUIWebSocketTransport → _QueueTracker → (future) ComfyUIClient → _ComfyUIMetricsRecorder`, and separately `_ComfyUIHTTPTransport → _OutputRetriever → (future) ComfyUIClient → _ComfyUIMetricsRecorder` — exactly the one-directional shape the three existing sub-documents already committed to (`MODULE7_PHASE2_QUEUE_TRACKER_DESIGN.md` §2.5/§11.3, `MODULE7_PHASE2_WEBSOCKET_TRANSPORT_DESIGN.md` §10.3, `MODULE7_PHASE2_OUTPUT_RETRIEVER_DESIGN.md` §1.3/§2.1).

### 2.2 Ownership

- `_ComfyUIMetricsRecorder` owns one injected `image_generator.MetricsCollector` reference, passed at construction (constructor injection — it never constructs its own `MetricsCollector`, mirroring `_OutputRetriever`'s constructor-injection of `_ComfyUIHTTPTransport`).
- It does not own the collector's lifecycle (there is nothing to open/close — `MetricsCollector` is a thin, stateless-except-for-`metrics_path` wrapper that opens/closes the JSONL file per `append()` call).
- It holds no other state. Every value needed to build one `GenerationMetrics` record is either passed into `record_attempt()` or computed synchronously inside that single call (the wall-clock `total_duration_seconds`, §5.3).

### 2.3 Why this is a thin translation layer, not a second metrics system

`models.GenerationMetrics` and `image_generator.MetricsCollector` already exist, are already Pydantic-validated and atomically-appended (`open("a")` + `flush()` + `os.fsync()`), and are explicitly documented (`MetricsCollector`'s own docstring) as the durable-write mechanism for Module 7 monitoring data. `_ComfyUIMetricsRecorder` therefore does not reimplement persistence, retries-on-write-failure, rotation, or file locking — all of that already lives in `MetricsCollector` and is out of scope to modify. `_ComfyUIMetricsRecorder`'s entire job is: **assemble the right `GenerationMetrics` instance from data Phase 2's completed components already produce, then hand it to the existing collector, and never let that hand-off crash the caller.**

---

## 3. Public API

### 3.1 `__init__`

```python
def __init__(self, collector: "MetricsCollector") -> None: ...
```

- **`collector`** (required, positional): an already-constructed `image_generator.MetricsCollector`. `_ComfyUIMetricsRecorder` never constructs its own — matching the constructor-injection convention already used by `_OutputRetriever` (transport) and `_QueueTracker` (transports).
- **Validation:** raises `ValueError` immediately if `collector` is `None`. No other constructor arguments exist, so no other validation is needed.
- **Thread-safety:** the constructor performs no I/O and mutates no shared/global state. Safe to call from any thread.
- **Import-time note:** `image_generator.MetricsCollector` is imported at module scope by `comfyui_client.py`, the same way `_OutputRetriever` imports `PIL.Image`/`PIL.UnidentifiedImageError` at module scope — no lazy import needed. This is the one place Phase 2's transport module (`comfyui_client.py`) gains a dependency on `image_generator.py`; that dependency is one-directional (`comfyui_client.py → image_generator.py`) and does not create a cycle, since `image_generator.py` does not import anything from `comfyui_client.py` today **[VERIFIED: `image_generator.py`'s `__all__`/imports were inspected; it exports `MetricsCollector`, `GenerationMetrics`-adjacent helpers, and the pre-existing exception re-exports, with no import of `comfyui_client`]**.

### 3.2 `record_attempt()`

```python
def record_attempt(
    self,
    *,
    video_id: str,
    niche: str,
    workflow_version: str,
    profile_name: str | None = None,
    workflow_hash: str | None = None,
    completions: Sequence["_CompletionResult"],
    output: "_OutputResult | None" = None,
    exception: BaseException | None = None,
    num_candidates_requested: int = 1,
    identity_retry_count: int = 0,
    peak_vram_mb: float | None = None,
    gpu_utilization_percent: float | None = None,
    attempt_started_at: float | None = None,
) -> None: ...
```

**Inputs, in detail:**

| Parameter | Type | Required? | Source |
|---|---|---|---|
| `video_id` | `str` | yes | Caller's per-video loop context (already threaded through every Module 7 stage today). |
| `niche` | `str` | yes | Same. |
| `workflow_version` | `str` | yes | `WorkflowTemplateRef.workflow_version` or `ComfyUIWorkflowRef.workflow_version` (Phase 1, `models.py`) — `GenerationMetrics.workflow_version` is a required, non-optional field, so this parameter is required too. |
| `profile_name` | `str \| None` | no (default `None`) | `GenerationProfile.name` (Phase 1). Optional on `GenerationMetrics` itself. |
| `workflow_hash` | `str \| None` | no (default `None`) | `ComfyUIWorkflowRef.workflow_hash` (Phase 1). Optional on `GenerationMetrics`. |
| `completions` | `Sequence[_CompletionResult]` | yes, non-empty | One entry per submission attempt for this logical `generate()` call — see §4.2. Ordered oldest → most recent; the **last** entry is treated as the terminal outcome. |
| `output` | `_OutputResult \| None` | no (default `None`) | The `_OutputRetriever.retrieve()` return value, **only** when the terminal `completions[-1].outcome` is `_CompletionOutcome.COMPLETED` and retrieval itself succeeded. `None` on any failure path. |
| `exception` | `BaseException \| None` | no (default `None`) | The exception the caller is about to raise (or has already caught) for this attempt, if any. Used purely for `failure_reason` classification (§5.4) — never re-raised, never logged with its traceback (§6). |
| `num_candidates_requested` | `int` | no (default `1`) | Pass-through to `GenerationMetrics.num_candidates_requested`. Phase 2 calls `generate()` once per candidate (per the master integration document's own future-compatibility note), so this is `1` for every Phase 2-era call; a future Phase 3 orchestrator that requests multiple candidates per video would set this per its own `num_candidates` config. |
| `identity_retry_count` | `int` | no (default `0`) | Pure pass-through to `GenerationMetrics.identity_retry_count`. Phase 2 never sets this to anything but `0` — identity retries are a Phase 3 concern (`IdentityPreservationStage`) that doesn't exist yet. Accepted now purely so the field mapping needs no signature change when Phase 3 starts populating it. |
| `peak_vram_mb` | `float \| None` | no (default `None`) | Caller-measured, e.g. `before.vram_free_mb - after.vram_free_mb` from two `SystemStats` readings around the call. `_ComfyUIMetricsRecorder` never computes this itself (§1.2). |
| `gpu_utilization_percent` | `float \| None` | no (default `None`) | Same shape as `peak_vram_mb`. **[ASSUMED]** — `SystemStats` (as implemented today) exposes `vram_free_mb`, `vram_total_mb`, `device_name`, `comfyui_version` only; it has no utilization field. This parameter therefore has no current data source and will be `None` for every real Phase 2 call until a future change adds one. It is accepted anyway, at zero cost, so `GenerationMetrics.gpu_utilization_percent` needs no signature change later. |
| `attempt_started_at` | `float \| None` | no (default `None`) | A `time.monotonic()` timestamp the caller captured immediately before the *first* submission in `completions`. Used to compute `total_duration_seconds` (§5.3). If omitted, `total_duration_seconds` falls back to `sum(c.queue_wait_seconds + c.generation_seconds for c in completions)`, which under-counts any time spent in `_OutputRetriever.retrieve()` (download/validation) — documented as a known approximation, not a defect, since `attempt_started_at` is cheap for a caller to supply and is the accurate path. |

**Output:** `None`. All effects are side effects (one `GenerationMetrics` record appended, or one ERROR log line if that append could not happen — §8).

**Raises:** `ValueError` for programmer/integration errors only — see §8.1. Never raises `MetricsWriteError` or any other exception arising from the actual metrics-write attempt (§8.4) — those are caught internally.

### 3.3 Why keyword-only, and why `completions` is a sequence rather than a single `_CompletionResult`

Every parameter after `self` is keyword-only (`*`) to avoid positional-argument ordering mistakes at a call site that will eventually live inside a `finally` block in `ComfyUIClient.generate()` — matching `_OutputRetriever.retrieve()`'s own keyword-only `prompt_id` parameter for the same reason.

`completions` is a sequence rather than one `_CompletionResult` specifically so that `_ComfyUIMetricsRecorder`'s public contract does not need to change once a future OOM-retry loop is added to `ComfyUIClient.generate()` (§12). Today, every real call site will pass a one-element sequence (no retry loop exists yet); the sequence shape is forward-compatible scaffolding, not evidence that retries are implemented — it costs nothing today (`len(completions) == 1` degenerates every list-aggregation in §5 to a single value) and avoids a breaking signature change later.

---

## 4. Internal Data Structures

`_ComfyUIMetricsRecorder` introduces **no new public dataclasses and no new Pydantic models**. It consumes the existing `_CompletionResult`/`_CompletionOutcome`/`_OutputResult` types as-is and produces the existing `models.GenerationMetrics` as-is. It introduces exactly one small, private, module-internal constant table:

### 4.1 `_FAILURE_REASON_BY_EXCEPTION_TYPE` (class attribute, `_ComfyUIMetricsRecorder`)

A private, ordered mapping from exception type to a short, stable `failure_reason` string, defined as a class-level constant — the same pattern `_OutputRetriever._PIL_FORMAT_EXTENSIONS` and `_QueueTracker._STATE_QUEUED`/`_STATE_EXECUTING` already use for fixed, non-operator-tunable lookup data (i.e., deliberately **not** promoted to `config.py`, since these strings are part of the module's internal contract, not a deployment knob — see §7 for the reasoning on why no new `config.py` constants are introduced).

```python
_FAILURE_REASON_BY_EXCEPTION_TYPE: ClassVar[tuple[tuple[type[BaseException], str], ...]] = (
    # Order matters: subclasses must be listed before their base classes.
    (MissingOutputFileError, "missing_output_file"),
    (CorruptImageError, "corrupt_image"),
    (UnsupportedImageFormatError, "unsupported_image_format"),
    (NoOutputImageError, "no_output_image"),
    (OutputHistoryError, "output_history_error"),
    (OutputDownloadError, "output_download_error"),
    (VRAMExhaustedError, "vram_exhausted"),
    (ComfyUIQueueError, "queue_error"),
    (ComfyUIConnectionError, "connection_error"),
)
```

This is a tuple, not a `dict`, precisely because ordered `isinstance` matching is required: `MissingOutputFileError` is itself a subclass of `OutputDownloadError` (verified in `module7_exceptions.py`), so a plain `type(exc): reason` dict keyed on `exc.__class__` would silently fail to classify a `MissingOutputFileError` instance correctly if some future code raised it via a base-typed reference, or would require an exact-type dict lookup that breaks the moment a new subclass is added upstream. Ordered `isinstance` checks are robust to that (§5.4 walks through the exact algorithm).

### 4.2 `completions: Sequence[_CompletionResult]` — no new type, but a documented convention

No new wrapper type is introduced for "the list of every submission attempt in one logical `generate()` call." A plain `Sequence[_CompletionResult]` is sufficient because `_CompletionResult` is already frozen, already carries `queue_wait_seconds`/`generation_seconds`/`outcome`/`used_http_fallback`, and `_ComfyUIMetricsRecorder` only ever needs to (a) sum/list two of its numeric fields and (b) read `.outcome` off the last element. Introducing a bespoke aggregate dataclass for this would add a type with no behavior of its own — pure ceremony — so this document deliberately does not add one, consistent with the task brief's constraint to reuse the existing architecture exactly rather than invent parallel structures.

---

## 5. Metrics Collected — field-by-field mapping to `models.GenerationMetrics`

`models.GenerationMetrics` (Phase 1, unmodified, frozen Pydantic model) has these fields today:

```python
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
```

### 5.1 Direct pass-through fields

| `GenerationMetrics` field | Source |
|---|---|
| `video_id` | `record_attempt(video_id=...)`, verbatim. |
| `niche` | `record_attempt(niche=...)`, verbatim. |
| `profile_name` | `record_attempt(profile_name=...)`, verbatim (may be `None`). |
| `workflow_version` | `record_attempt(workflow_version=...)`, verbatim. |
| `workflow_hash` | `record_attempt(workflow_hash=...)`, verbatim (may be `None`). |
| `num_candidates_requested` | `record_attempt(num_candidates_requested=...)`, verbatim. |
| `identity_retry_count` | `record_attempt(identity_retry_count=...)`, verbatim — always `0` from every real Phase 2 call site today. |
| `peak_vram_mb` | `record_attempt(peak_vram_mb=...)`, verbatim (may be `None`). |
| `gpu_utilization_percent` | `record_attempt(gpu_utilization_percent=...)`, verbatim (`None` for every real call today — §3.2). |
| `recorded_at` | `image_generator.utc_now()`, called once at the start of `record_attempt()` — reuses the exact existing helper (`datetime.now(timezone.utc).isoformat()`), not a reimplementation. |

### 5.2 Fields left at their Pydantic defaults (Phase 3/4 concern, unchanged by this document)

`generation_hash`, `identity_failures_count`, `qa_failures_count`, `winning_overall_score`, `winning_signal_scores` are **never set** by `record_attempt()` — they are simply omitted from the `GenerationMetrics(...)` constructor call, so Pydantic applies its declared defaults (`None`, `0`, `0`, `None`, `{}` respectively). This matches the (pre-implementation) integration design document's field-ownership split and requires no schema change for Phase 3/4 to later populate them on their own `GenerationMetrics` records for the same video.

### 5.3 Timing fields — aggregated across `completions`

| `GenerationMetrics` field | Computation |
|---|---|
| `queue_time_seconds` | `sum(c.queue_wait_seconds for c in completions)` — total time spent queued across every submission attempt for this logical call. A retried attempt that queued twice (once per submission) has both queue waits counted, since both really did block the pipeline. |
| `generation_time_seconds` | `[c.generation_seconds for c in completions]` — **one list entry per submission attempt, in order**. For the common (no-retry) case this is a one-element list. This is exactly what the field's `list[float]` type signals it's for: per-attempt generation durations within one logical record, letting an operator see "attempt 1 ran for 4.2s before OOM-erroring, attempt 2 ran for 9.8s and succeeded" from a single JSONL line instead of needing to correlate multiple lines. |
| `total_duration_seconds` | If `attempt_started_at` was supplied: `time.monotonic() - attempt_started_at`, measured at the moment `record_attempt()` runs (i.e., after retrieval/validation has also finished) — the true end-to-end wall time for the whole logical attempt, including `_OutputRetriever` download/validation time that `_CompletionResult` itself does not capture. If `attempt_started_at` is `None`: falls back to `sum(c.queue_wait_seconds + c.generation_seconds for c in completions)`, a documented under-estimate (§3.2) that excludes retrieval time. |
| `generation_retry_count` | `len(completions) - 1`. Zero for the (currently universal) no-retry case; a caller that retried an OOM-classified attempt twice before succeeding passes a 3-element `completions` list, yielding `2`. |

### 5.4 `failure_reason` — classification algorithm

`failure_reason: Optional[str]` is computed by `_classify_failure_reason(exception, terminal_outcome)`, a small `@staticmethod` on `_ComfyUIMetricsRecorder` (private, but intentionally unit-testable in isolation — the same "small pure classification helper, independently tested" pattern Module 2's `_classify_transient_error` already established):

```python
@staticmethod
def _classify_failure_reason(
    exception: BaseException | None,
    terminal_outcome: "_CompletionOutcome",
) -> str | None:
    ...
```

**Algorithm, in order:**

1. If `exception is None` and `terminal_outcome is _CompletionOutcome.COMPLETED`: return `None` (success — no failure to record).
2. If `exception is not None`: walk `_FAILURE_REASON_BY_EXCEPTION_TYPE` in the fixed order shown in §4.1, return the first `reason` string whose `exc_type` matches via `isinstance(exception, exc_type)`.
3. If `exception is not None` but it matched none of the table's known types (i.e., it is some other `Module7Error` subclass, or a wholly unexpected non-`Module7Error` exception — a bug): return the fixed string `"unclassified_error"`, and separately log (§6) the exception's *type name only* (`type(exception).__name__`) at `WARNING` so an operator notices a new failure mode exists that this table doesn't yet know about, without ever logging the exception object itself (§6, pickling-safety) or its message text (which could contain a filename, prompt id, or other detail better kept out of a fixed-vocabulary metrics field).
4. If `exception is None` but `terminal_outcome` is not `COMPLETED`: this is the case where the caller hasn't (yet, or ever, if `ComfyUIClient.generate()` chooses not to always raise) translated a non-terminal `_CompletionOutcome` into a typed exception before calling `record_attempt()`. Return `"execution_error"` if `terminal_outcome is _CompletionOutcome.EXECUTION_ERROR`, or `"timeout"` if `terminal_outcome is _CompletionOutcome.TIMEOUT`. This makes `_ComfyUIMetricsRecorder` correctly classify failures **even before `ComfyUIClient.generate()` exists to do its own exception translation** — a deliberate robustness choice, since `_QueueTracker.await_completion()` already returns a fully-typed `_CompletionOutcome` today, and `_ComfyUIMetricsRecorder` should not have to wait on undesigned facade code to be useful/testable.
5. (Defensive.) If `exception is None` and `terminal_outcome is _CompletionOutcome.COMPLETED` but `output is None` (i.e., the queue tracker says it finished but no output was ever retrieved — this should not happen if the caller wired things correctly, since a `COMPLETED` outcome should always be followed by a `retrieve()` call before `record_attempt()`): return `"output_missing_uncaptured"` rather than silently claiming success. This guards against a future integration bug in `ComfyUIClient.generate()` silently reporting a false success.

**Why this needs both `exception` and `terminal_outcome`, not just one:** `_QueueTracker.await_completion()` never raises — it always returns a typed `_CompletionResult` (§2 of `MODULE7_PHASE2_QUEUE_TRACKER_DESIGN.md`, "does not raise `Module7Error` subclasses itself"). `_OutputRetriever.retrieve()`, by contrast, *does* raise typed exceptions directly (`OutputHistoryError`, `NoOutputImageError`, etc.) rather than returning a typed failure value. So a single logical attempt's failure signal can show up as *either* a non-`COMPLETED` `_CompletionOutcome` (queue-tracker-detected failure, no exception yet) *or* a raised exception (output-retrieval-detected failure, or a facade-level translation of a queue-tracker outcome, once that facade exists). `_ComfyUIMetricsRecorder` has to handle both shapes today because both of its real, already-implemented upstream collaborators use different failure-signaling conventions.

### 5.5 Complete example — success case

```python
recorder.record_attempt(
    video_id="abc123",
    niche="gaming",
    workflow_version="workflow_v1",
    workflow_hash="9f3a...e21",
    profile_name="PROFILE_STANDARD",
    completions=[completion],          # completion.outcome is COMPLETED
    output=output_result,              # the _OutputResult from _OutputRetriever.retrieve()
    exception=None,
    num_candidates_requested=1,
    peak_vram_mb=1024.0,
    attempt_started_at=started_at,
)
```

Produces (conceptually — no implementation shown):

```json
{
  "video_id": "abc123", "niche": "gaming", "profile_name": "PROFILE_STANDARD",
  "workflow_version": "workflow_v1", "workflow_hash": "9f3a...e21", "generation_hash": null,
  "num_candidates_requested": 1, "queue_time_seconds": 2.1, "generation_time_seconds": [9.8],
  "total_duration_seconds": 12.4, "identity_retry_count": 0, "generation_retry_count": 0,
  "failure_reason": null, "identity_failures_count": 0, "qa_failures_count": 0,
  "winning_overall_score": null, "winning_signal_scores": {}, "peak_vram_mb": 1024.0,
  "gpu_utilization_percent": null, "recorded_at": "2026-07-22T10:00:00+00:00"
}
```

### 5.6 Complete example — OOM-retry-then-fail case (illustrates the forward-compatible `completions` shape; the retry loop itself is **not** designed by this document)

```python
recorder.record_attempt(
    video_id="abc123",
    niche="gaming",
    workflow_version="workflow_v1",
    completions=[first_attempt_completion, second_attempt_completion],  # both EXECUTION_ERROR
    output=None,
    exception=vram_exhausted_error_instance,
)
```

Produces `generation_retry_count=1`, `generation_time_seconds=[<first>, <second>]`, `queue_time_seconds=<sum>`, `failure_reason="vram_exhausted"`, `output=None` (omitted → `GeneratedAsset`-adjacent fields simply absent from `GenerationMetrics`, which has none anyway — output metadata is not part of this model; see §5.7).

### 5.7 Note on "output metadata" / "image metadata" (task brief item 4)

`models.GenerationMetrics` has no fields for image path, dimensions, format, or hash — those live on `models.GeneratedAsset` (Phase 3's concern, populated once `ImageGenerationResult` is finalized) and on `_OutputResult` itself (already returned to the caller directly by `_OutputRetriever.retrieve()`). `_ComfyUIMetricsRecorder` accepts `output: _OutputResult | None` **only** to (a) confirm success/failure for `failure_reason` classification (§5.4 step 5) and (b) as a forward-compatible seam — if a future, additive change to `GenerationMetrics` adds an image-metadata field, `_ComfyUIMetricsRecorder` already has the `_OutputResult` in hand to source it from, with no signature change needed. Today, `output`'s width/height/filename/sha256-adjacent fields are **not** copied into any `GenerationMetrics` field, because no such field exists yet — this document does not invent one, per the constraint to reuse the existing architecture exactly.

---

## 6. Logging Strategy

Reuses the exact `_configure_logger()` sink already attached at `comfyui_client.py` import time (`MODULE7_LOG_PATH`, `rotation="10 MB"`, `retention="30 days"`, `enqueue=True`) — no new sink, no new log file. `_ComfyUIMetricsRecorder` is simply another logger call site within the same module.

**Level policy:**

- **DEBUG**: one line per successful `append()`, confirming `video_id` and `failure_reason=None` — mirrors `MetricsCollector.append()`'s own existing DEBUG confirmation, so a successful metrics write is visible at DEBUG in two places (the collector's own log line, unchanged, plus the recorder's) without being noisy at INFO.
- **INFO**: none. Metrics recording is a background bookkeeping concern, not a pipeline-phase-transition worth INFO visibility on its own (the *generation* phase transitions are already logged at INFO by `_QueueTracker`/`_OutputRetriever`, per their own design documents).
- **WARNING**: exactly one case — step 3 of §5.4's classification algorithm, when an exception was supplied but did not match any known entry in `_FAILURE_REASON_BY_EXCEPTION_TYPE`. Logs `type(exception).__name__` only, never `str(exception)` and never the exception object itself.
- **ERROR**: exactly one case — `MetricsCollector.append()` raised `MetricsWriteError` (or, defensively, any other unexpected exception during `GenerationMetrics(...)` construction or `append()`). Logs the failure as a string (`str(exc)`) with `video_id` context, **never** `exc=` / the exception object (§6.1).

### 6.1 Sensitive-information / pickling-safety policy

Following the exact, already-established project convention (`youtube_metadata.py`'s `_before_sleep_log` comment, restated in `MODULE7_PHASE2_COMFYUI_INTEGRATION_DESIGN.md` §6, and already applied by `_OutputRetriever`'s own logging, which logs `str(last_error)` rather than the exception object): because Loguru is configured with `enqueue=True`, every log record is pickled for hand-off to a background writer thread, and traceback objects are not picklable. `_ComfyUIMetricsRecorder` therefore **never** passes an exception instance to `logger.*`, only `str(exc)` or `type(exc).__name__`. It also never logs: raw ComfyUI graph JSON, image bytes, full `history_payload`/`error_payload` dict contents (those belong to `_QueueTracker`'s/`_OutputRetriever`'s own DEBUG-level logging, already specified in their respective documents), or the full `winning_signal_scores`/`GenerationMetrics` JSON body at any level above DEBUG.

---

## 7. Configuration

**No new `config.py` constants are introduced by this document.** `_ComfyUIMetricsRecorder` needs no timeouts, retry budgets, host/port, or file paths of its own — it reuses `MODULE7_METRICS_PATH` exclusively *indirectly*, through the already-injected `MetricsCollector` (which already defaults to `MODULE7_METRICS_PATH` in its own constructor). Nothing about `_ComfyUIMetricsRecorder`'s behavior is meant to be operator-tunable: the failure-reason vocabulary (§4.1) is a fixed part of the module's contract, not a deployment knob, and is therefore defined as a class-level constant on `_ComfyUIMetricsRecorder` itself rather than promoted to `config.py` — the same placement decision already made for `_OutputRetriever._PIL_FORMAT_EXTENSIONS` and `_QueueTracker._STATE_QUEUED`/`_STATE_EXECUTING`.

`MAX_GENERATION_RETRIES: int = 3` (existing, `config.py`, already present today) is **not** read by `_ComfyUIMetricsRecorder` — it's the future OOM-retry loop's concern (bounding how many entries can realistically appear in `completions`, informing §9's performance analysis) but `_ComfyUIMetricsRecorder` itself places no ceiling on `len(completions)`; it aggregates whatever it's given.

---

## 8. Error Handling

### 8.1 Constructor validation

`__init__` raises `ValueError("collector must not be None")` if `collector` is falsy/`None`. This is the only constructor-time validation, matching the minimal-but-present validation style of `_QueueTracker.__init__`/`_OutputRetriever.__init__`.

### 8.2 `record_attempt()` argument validation (fail-fast, raised immediately)

These are **programmer/integration errors** — they indicate a bug in the (future) `ComfyUIClient.generate()` call site, not a runtime/environment condition, and are therefore raised immediately as `ValueError` rather than caught-and-logged, consistent with how `_QueueTracker`/`_OutputRetriever` treat bad constructor arguments today:

- `video_id` is `None`, empty, or not a `str` → `ValueError`.
- `niche` is `None`, empty, or not a `str` → `ValueError`.
- `workflow_version` is `None`, empty, or not a `str` → `ValueError` (it is a required, non-`Optional` field on `GenerationMetrics`; a Pydantic `ValidationError` from the model constructor would be a confusing way to surface this same bug, so it is checked explicitly and early with a clearer message).
- `completions` is `None` or empty → `ValueError("completions must contain at least one _CompletionResult")` — there is no such thing as "an attempt that made zero submissions."
- `num_candidates_requested < 1` or `identity_retry_count < 0` → `ValueError`.

These checks happen **before** any timing computation or `GenerationMetrics` construction, so a caller bug here never partially writes a malformed record.

### 8.3 Failure classification never raises

`_classify_failure_reason()` (§5.4) is a pure function with a defensive catch-all branch (step 3) — it always returns a string or `None`, never raises, and never re-raises whatever exception it was handed (it only inspects it with `isinstance`).

### 8.4 Metrics-write failure — the "what happens if metrics collection itself fails" case

Everything from `GenerationMetrics(...)` construction (which, given the validation in §8.2, should not itself raise a Pydantic `ValidationError` in practice, but is not assumed infallible) through `collector.append(metrics)` is wrapped in one `try/except Exception` block inside `record_attempt()`. On any exception there (most commonly `MetricsWriteError`, raised by `MetricsCollector.append()` on an `OSError` — e.g. a full disk or a permissions problem):

- The exception is **caught inside `_ComfyUIMetricsRecorder`**, not left for the caller to catch. This is a deliberate refinement over the earlier (pre-implementation) integration design document's phrasing, which ambiguously suggested `ComfyUIClient.generate()` would be the one to catch `MetricsWriteError`. Making `_ComfyUIMetricsRecorder` itself the single place that swallows write failures means every future call site — `ComfyUIClient.generate()` or any test harness — gets "never crashes, ever" for free, without needing to remember to wrap the call.
- One `ERROR`-level log line is emitted (`str(exc)` only, §6.1), including `video_id` for correlation.
- `record_attempt()` then returns normally (`None`) — it does **not** re-raise, and does **not** return a sentinel/boolean signaling failure, since nothing upstream is expected to act differently based on whether metrics recording succeeded (matching `MetricsCollector`'s own "passive observer" framing, extended one layer up).

This means: a full disk that breaks metrics logging will **never** cause a video's actual generation result (success or failure) to be lost, misreported, or turned into a different exception than the one already in flight.

### 8.5 Summary table

| Failure | Raised or swallowed? | Caller-visible effect |
|---|---|---|
| `collector is None` at construction | Raised (`ValueError`) | Immediate, loud — a wiring bug caught at startup, not in a `finally` block later. |
| Bad `record_attempt()` arguments (§8.2) | Raised (`ValueError`) | Immediate, loud — a call-site bug, caught by unit tests before reaching production. |
| Unrecognized exception type during classification | Swallowed | `failure_reason="unclassified_error"`, one `WARNING` log line with the type name. `record_attempt()` still completes and appends a record. |
| `MetricsCollector.append()` raises `MetricsWriteError` (or any other exception) | Swallowed | One `ERROR` log line. `record_attempt()` returns normally; no `GenerationMetrics` record was durably written for this attempt, but the caller's own generation outcome is completely unaffected. |

---

## 9. Thread Safety

- `_ComfyUIMetricsRecorder` itself holds no mutable instance state beyond the injected `collector` reference (set once, at construction, never reassigned) — so the object itself is trivially safe to share across threads, the same way `_OutputRetriever` and `_QueueTracker` document themselves as safe to construct once and reuse.
- **Concurrent calls to `record_attempt()` are only as safe as the underlying `MetricsCollector.append()` already is**, which `_ComfyUIMetricsRecorder` does not modify or wrap with any additional locking. `MetricsCollector.append()` opens the file in append mode, writes one line, flushes, and calls `os.fsync()` per call — on POSIX, a single `write()` of a JSONL line under `PIPE_BUF` (historically 4096 bytes on Linux) made against a file opened with `O_APPEND` is atomic with respect to other appenders, but Python's buffered `TextIOWrapper.write()` inside a `with ... open("a")` block does not guarantee that every `.write()` call maps to exactly one atomic `write(2)` syscall of that size, especially once a `GenerationMetrics` record grows past a few hundred bytes (e.g. once `winning_signal_scores` is populated by a future Phase 4). This is pre-existing behavior in `MetricsCollector`, out of scope to change here.
- **Given `MODULE7_MAX_CONCURRENT_GENERATIONS: int = 1`** (existing `config.py` default, already governing Module 7's whole pipeline), concurrent calls to `record_attempt()` from multiple threads/processes are **outside the currently supported operating envelope** — Module 7's own documented concurrency model already assumes one generation in flight at a time, so one `record_attempt()` call in flight at a time follows directly. This document does not add locking to `_ComfyUIMetricsRecorder` or to `MetricsCollector` to support a concurrency level the rest of Module 7 doesn't yet support either.
- **If `MODULE7_MAX_CONCURRENT_GENERATIONS` is ever raised above `1`** in the future, this is flagged as a prerequisite change that belongs to `MetricsCollector` (e.g. a `portalocker`-based advisory lock around the write, mirroring Module 1's own existing file-locking convention in `csv_reader.py`) rather than to `_ComfyUIMetricsRecorder`, since the actual race is on the shared file handle/inode, not on anything `_ComfyUIMetricsRecorder` itself owns.

---

## 10. Performance

- **Memory:** One `GenerationMetrics` instance (a handful of scalars plus a `generation_time_seconds` list bounded, in practice, by `MAX_GENERATION_RETRIES + 1` entries — 4 floats at the existing default of `3`) is constructed, serialized to one JSON line, and immediately discarded after `append()` returns. No accumulation across calls; `_ComfyUIMetricsRecorder` retains no history of past attempts in memory.
- **CPU:** Negligible relative to a ComfyUI generation (seconds-to-tens-of-seconds per `GenerationProfile.expected_generation_seconds`, Phase 1). The dominant cost inside `record_attempt()` is one Pydantic model construction + `model_dump_json()` + one `os.fsync()` syscall (the fsync, not the CPU work, is the real latency floor, and that already exists unchanged inside `MetricsCollector.append()`).
- **No unnecessary allocations:** `completions` is iterated at most twice (once for `queue_time_seconds`'s `sum()`, once for `generation_time_seconds`'s list comprehension) — both single-pass, no intermediate list materialization beyond the one list the field itself requires. The `_FAILURE_REASON_BY_EXCEPTION_TYPE` table is a module-load-time constant tuple, not rebuilt per call.
- **No I/O beyond the one, already-existing `MetricsCollector.append()` call** — `_ComfyUIMetricsRecorder` opens no files, sockets, or connections of its own.

---

## 11. Integration — exactly how, without modifying any of them

| Component | Interaction | What is NOT changed |
|---|---|---|
| `_QueueTracker` | None directly. `_ComfyUIMetricsRecorder` never imports or calls `_QueueTracker`; it only receives the `_CompletionResult` value(s) the caller already obtained from `_QueueTracker.await_completion()`. | `_QueueTracker`'s public API, its state machine, its own logging — all untouched. `_QueueTracker`'s own design document (§2.5, §11.3) already anticipated this exact "produces data, never called by name" relationship and is not contradicted by anything here. |
| `_OutputRetriever` | None directly. Receives the optional `_OutputResult` the caller already obtained from `_OutputRetriever.retrieve()`, purely as a success/failure signal and as a forward-compatible seam (§5.7). | `_OutputRetriever`'s public API, its exception hierarchy, its own logging — all untouched. Its design document (§1.3, §13.3) already anticipated `MetricsRecorder` as a data consumer with "no dependency," which this document fulfills exactly. |
| `_ComfyUIHTTPTransport` | None at all — zero import, zero reference. VRAM figures (`peak_vram_mb`) are supplied by the caller, who is responsible for calling `_ComfyUIHTTPTransport.system_stats()` itself if it wants that data; `_ComfyUIMetricsRecorder` never triggers an HTTP call. | Entirely untouched; not even imported. |
| `_ComfyUIWebSocketTransport` | None at all — zero import, zero reference. All timing data `_ComfyUIMetricsRecorder` needs is already baked into `_CompletionResult` by `_QueueTracker`, which is the component that actually consumes the WebSocket transport. | Entirely untouched; not even imported. |
| `module7_exceptions.py` | Read-only `isinstance` checks against nine already-existing exception classes (§4.1). No new exception classes are added or required by this document — every failure `_ComfyUIMetricsRecorder` needs to classify today already has a concrete, raised-in-practice exception type in the current codebase. | No additions, no modifications. (Contrast with the earlier, pre-implementation integration design document, which proposed adding `ComfyUITimeoutError`/`ComfyUIResponseError` — those do not exist in the repository today and are not required by this document; the timeout case is instead classified directly off `_CompletionOutcome.TIMEOUT`, §5.4 step 4, precisely so `_ComfyUIMetricsRecorder` doesn't have to wait on exceptions that don't exist yet.) |
| `models.GenerationMetrics` | Constructed once per `record_attempt()` call, using only fields/defaults that already exist on the model. | No schema change, no new field, no modified default. |
| `image_generator.MetricsCollector` | `append()` called exactly once per successful `record_attempt()` invocation, with the freshly-built `GenerationMetrics`. | No modification; `_ComfyUIMetricsRecorder` uses its existing public `append(metrics: GenerationMetrics) -> None` signature exactly as-is. |
| `image_generator.utc_now()` | Called once per `record_attempt()` to populate `recorded_at`. | No modification; reused verbatim rather than reimplemented (avoids a second, possibly-drifting timestamp-formatting convention). |

---

## 12. Testing Strategy

Mirrors the existing `tests/test_comfyui_client.py` structure (`TestWebSocketTransport`, `TestQueueTracker`, and — once `_OutputRetriever`'s own tests land — presumably `TestOutputRetriever`), adding a new `TestMetricsRecorder` class in the same file, using a fake/spy `MetricsCollector` (a lightweight stand-in exposing the same `append(metrics) -> None` signature, or the real `MetricsCollector` pointed at a `tmp_path` fixture — both are valid per pytest convention; the real one exercises the genuine file-append path end-to-end and is preferred where it doesn't complicate assertions).

### 12.1 Happy paths

- Single successful completion (`completions=[completion]` with `outcome=COMPLETED`, `output=<_OutputResult>`, `exception=None`) → assert the exact `GenerationMetrics` field values: `failure_reason is None`, `generation_retry_count == 0`, `generation_time_seconds == [completion.generation_seconds]`, `queue_time_seconds == completion.queue_wait_seconds`.
- `attempt_started_at` supplied → assert `total_duration_seconds` reflects wall-clock elapsed time (using `monkeypatch` on `time.monotonic` for a deterministic value, the same technique already used elsewhere in `tests/test_comfyui_client.py`, e.g. `TestQueueTracker`'s `monkeypatch: pytest.MonkeyPatch` fixtures).
- `attempt_started_at` omitted → assert the documented fallback (`sum(queue_wait_seconds + generation_seconds)`) is used instead.
- Multi-element `completions` (simulating a future retry loop) with a successful final entry → assert `generation_retry_count == len(completions) - 1` and `generation_time_seconds` has one entry per `completions` element in order.

### 12.2 Failure paths — one test per `_FAILURE_REASON_BY_EXCEPTION_TYPE` entry

- Nine parametrized cases, one per exception type in §4.1's table, each asserting `record_attempt(..., exception=<instance>)` yields the exact expected `failure_reason` string. Include at least one case using `MissingOutputFileError` specifically to assert it classifies as `"missing_output_file"` and **not** the more general `"output_download_error"` — directly exercising the subclass-ordering correctness called out in §4.1.
- `_CompletionOutcome.EXECUTION_ERROR` with `exception=None` → `failure_reason == "execution_error"`.
- `_CompletionOutcome.TIMEOUT` with `exception=None` → `failure_reason == "timeout"`.
- An unrecognized exception type (e.g. a plain `RuntimeError` not in the table, or a synthetic new `Module7Error` subclass not yet added to `_FAILURE_REASON_BY_EXCEPTION_TYPE`) → `failure_reason == "unclassified_error"`, and assert (via `caplog`/a Loguru sink fixture) that exactly one `WARNING` line was emitted containing the exception's type name and **not** containing `str(exception)`'s message text.
- `outcome=COMPLETED`, `exception=None`, but `output=None` (the defensive §5.4 step 5 case) → `failure_reason == "output_missing_uncaptured"`.

### 12.3 Boundary / malformed-input cases

- `collector=None` at construction → `ValueError`.
- `video_id=""`, `niche=""`, `workflow_version=""` (each independently) → `ValueError`, verifying the exact parameter name is identifiable from the error (either via message content or via a dedicated `pytest.raises(ValueError, match=...)` per case).
- `completions=[]` → `ValueError`.
- `num_candidates_requested=0`, `identity_retry_count=-1` → `ValueError`.

### 12.4 Metrics-write-failure path

- Inject a fake `MetricsCollector` whose `append()` raises `MetricsWriteError` (or a plain `OSError`, to confirm the broad `except Exception` in §8.4 also catches non-`Module7Error` failures) → assert `record_attempt()` returns `None` without raising, and that exactly one `ERROR`-level log line was emitted containing `video_id` and **not** containing a pickled/repr'd exception object (assert on the log record's `extra`/message shape being a plain string, mirroring how `_before_sleep_log` regression tests elsewhere in the project assert "invoked with a string, never an exception object").

### 12.5 Performance-related test

- Construct `completions` with the maximum realistic length implied by `MAX_GENERATION_RETRIES + 1` (i.e. 4 entries) and assert `record_attempt()` completes and produces a `generation_time_seconds` list of exactly that length with no unexpected truncation — a cheap regression guard against an accidental off-by-one or early-break in the aggregation logic, not a real perf/benchmark test (none is warranted for a component whose entire cost is one small object construction plus one already-tested file append).

### 12.6 What is explicitly out of scope for these tests

- No test constructs a real `ComfyUIClient` (it doesn't exist yet) — every test calls `_ComfyUIMetricsRecorder.record_attempt()` directly with hand-built `_CompletionResult`/`_OutputResult` fixtures (both already frozen dataclasses, trivially constructible in a test without any transport/network mocking).
- No `@pytest.mark.gpu` tier is needed for this component — it does no I/O against ComfyUI at all, only against the local metrics JSONL file (or a spy), so every test here belongs in the default, fast, no-GPU suite alongside the rest of `tests/test_comfyui_client.py`.

---

## 13. Future Compatibility — integrating into `ComfyUIClient.generate()` (not designed here)

This section documents the seam `ComfyUIClient.generate()` will use, without specifying anything about `generate()`'s own internal control flow (submission, the OOM-retry decision, profile fallback — all explicitly out of scope per the task brief).

- **Call shape.** Once built, `generate()`'s outline is expected to: submit → `await_completion()` (possibly more than once, if it implements an OOM-retry loop) → on success, `retrieve()` → in a `finally` block (so it runs on every exit path, success or exception), call exactly one `_ComfyUIMetricsRecorder.record_attempt(...)`, passing the full list of `_CompletionResult`s it accumulated across however many submissions it made, the `_OutputResult` if retrieval succeeded, and whatever exception (if any) it is about to propagate.
- **No signature change anticipated.** Every parameter `record_attempt()` accepts today (§3.2) is either something `generate()`'s own inputs already provide (`video_id`, `niche`, workflow identity from `BuiltWorkflow`/`WorkflowTemplateRef`) or something it naturally accumulates during its own execution (`completions`, `output`, `exception`, `attempt_started_at`). Adding a real OOM-retry loop to `generate()` later requires **zero** changes to `_ComfyUIMetricsRecorder` — it already accepts a multi-element `completions` sequence today (§3.3, §5.6) specifically so that future addition is a pure consumer-side change.
- **`peak_vram_mb`/`gpu_utilization_percent` sourcing.** When `generate()` is designed, it will own the decision of *whether* to call `health_check()`/`system_stats()` before and after a generation to compute a peak-VRAM proxy (as the pre-implementation integration document sketched in its own §10) — `_ComfyUIMetricsRecorder` only needs the resulting number, and already accepts it.
- **Multi-candidate orchestration (Phase 3).** A future Phase 3 component that calls `ComfyUIClient.generate()` multiple times per video (once per `candidate_index`) will naturally produce multiple `record_attempt()` calls — one per candidate — each a fully independent `GenerationMetrics` JSONL line, exactly matching `GenerationMetrics`'s own docstring ("one append-only... record for a Module 7 **attempt**"). No aggregation across candidates is `_ComfyUIMetricsRecorder`'s job; that reconciliation, if ever wanted, is a downstream analysis concern (querying the JSONL by `video_id`), not a write-time one.
- **Phase 4 (QA/ranking).** `identity_failures_count`, `qa_failures_count`, `winning_overall_score`, `winning_signal_scores` remain at their Pydantic defaults from every Phase 2 call (§5.2). When Phase 4 exists, it may either extend `_ComfyUIMetricsRecorder`'s signature additively (new optional keyword arguments, defaulting to today's behavior) or append its own follow-up `GenerationMetrics` record independently — that reconciliation is explicitly left as a Phase 4 design decision, not resolved by this document, matching how the pre-implementation integration document already deferred it.

---

## 14. Implementation checklist

1. **No new dependency, no new `config.py` constants** (§7) — nothing to add to `requirements.txt` or `config.py` for this component.
2. **No new `module7_exceptions.py` entries** — `_ComfyUIMetricsRecorder` classifies using the nine exception types that already exist.
3. **Add to `modules/comfyui_client.py`:** the `_FAILURE_REASON_BY_EXCEPTION_TYPE` class constant and the `_ComfyUIMetricsRecorder` class (§3), importing `image_generator.MetricsCollector`, `models.GenerationMetrics`, and `image_generator.utc_now` at module scope alongside the existing imports.
4. **Write `TestMetricsRecorder`** in `tests/test_comfyui_client.py` per §12, using the existing `_CompletionResult`/`_CompletionOutcome`/`_OutputResult` fixtures/constructors already exercised by `TestQueueTracker` and (once written) `TestOutputRetriever` — no new fixture infrastructure should be required beyond a fake/spy `MetricsCollector`.
5. **Full regression run** (`pytest`, default markers) once implemented — confirm every pre-existing test still passes and the new `TestMetricsRecorder` cases are green, with zero changes to any other test file.
6. **No `main.py` change, no `ComfyUIClient` facade change** — neither exists yet; this phase adds one new internal class only.
7. **Docs.** If implementation reveals any deviation from this document (e.g., a different `failure_reason` string chosen, or an additional defensive branch needed in §5.4), update this file's own tables to match — keeping the doc and the code from drifting apart, per the same convention `MODULE7_PHASE2_COMFYUI_INTEGRATION_DESIGN.md` already applies to itself.

---

*End of design specification. No implementation code is included per the task's DESIGN ONLY constraint.*
