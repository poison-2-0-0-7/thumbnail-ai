# MODULE7_PHASE2_COMFYUI_INTEGRATION_DESIGN.md

**Module 7 — Phase 2: ComfyUI Integration (Transport & Execution Layer)**
**thumbnail-ai**

Status: **Design specification, v1.0. No implementation.**
Source of truth: repository `poison-2-0-0-7/thumbnail-ai`, reviewed at the commit with Modules 1–6 complete, Module 7 architecture (`docs/IMAGE_GENERATION_ARCHITECTURE.md`) complete, Module 7 Phase 1 (Foundation) complete, 364 tests passing / 1 deselected (`gpu` marker).
Upstream contracts: this phase consumes `BuiltWorkflow` (Phase 1, `WorkflowBuilder.build()`), `GenerationProfile` (Phase 1/config), and `ImageGenerationResult` / `GenerationMetrics` (Phase 1, `models.py`). It does **not** touch identity preservation, restoration, upscaling, QA scoring, or candidate ranking — those remain Phase 3 (Image Generation) and Phase 4 (Quality Assurance) per the architecture doc's own staging (§2, §15–16).

---

## 0. Scope statement

Phase 2 delivers exactly the "orchestrator and typed client" half of Module 7 described in `IMAGE_GENERATION_ARCHITECTURE.md` §3–4, item **7f (`ComfyUIClient`)**, and nothing past it. Concretely, Phase 2 is responsible for pipeline steps **5→6** in §5 of the architecture doc:

- Take an already-built `BuiltWorkflow` (produced by Phase 1's `WorkflowBuilder`) and get it executed by a locally running ComfyUI server.
- Track it from submission through completion.
- Retrieve the raw output image bytes ComfyUI produced.
- Do this reliably: typed errors, bounded retries, timeouts, structured logs, metrics, and clean resource teardown — with **zero** creative or quality-judgment logic.

Phase 2 explicitly does **not**:
- Decide *what* to generate (that's `WorkflowBuilder`, Phase 1, already done).
- Judge whether a generated candidate is good (`IdentityPreservationStage`, `FaceRestorationStage`, `QualityAssuranceStage`, `CandidateRanker` — Phase 3/4).
- Decide how many candidates to request or drive multi-candidate/retry-with-different-seed business logic (Phase 3 orchestrates *calls into* Phase 2; Phase 2 executes *one submission at a time*).
- Change `main.py`'s pipeline shape. Phase 2 adds a callable component; wiring it into the per-video pipeline stage is a Phase 3 concern (when there's a full stage to wire in), though this document specifies the integration point precisely (§14) so Phase 3 has nothing to redesign.

This scoping matches the architecture doc's own separation of "generation" from "restoration" from "assurance" (§2) and keeps Phase 2 a strict narrowing: a typed transport client around one external process, tested the same way Module 2 tests yt-dlp interaction — with mocks, not a live GPU, in the default suite (§13).

---

## 1. High-level architecture

```
                 ┌──────────────────────────────┐
                 │  Phase 1 output               │
                 │  BuiltWorkflow (graph, ref,    │
                 │  workflow_hash) + GenerationProfile │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
   ┌─────────────────────────────────────────────────────────────┐
   │            modules/comfyui_client.py  (Phase 2, NEW)          │
   │                                                                 │
   │  ComfyUIClient  (public facade, one instance per pipeline run) │
   │   ├─ _ComfyUIHTTPTransport   (requests-based; submit/health/    │
   │   │                           history/view/interrupt/system)    │
   │   ├─ _ComfyUIWebSocketTransport (websocket-client based;         │
   │   │                           progress + completion events)      │
   │   ├─ _QueueTracker            (prompt_id lifecycle bookkeeping)  │
   │   ├─ _OutputRetriever         (downloads finished images)        │
   │   └─ _ComfyUIMetricsRecorder  (per-attempt timing → GenerationMetrics) │
   └───────────────┬─────────────────────────────────────┬─────────┘
                    │ HTTP (localhost:8188)                │ WebSocket (ws://localhost:8188/ws)
                    ▼                                       ▼
   ┌─────────────────────────────────────────────────────────────┐
   │           ComfyUI server (separate OS process, long-lived)    │
   │      python main.py --listen 127.0.0.1 --port 8188            │
   └─────────────────────────────────────────────────────────────┘

   Output of Phase 2, per call to ComfyUIClient.generate():
   ┌──────────────────────────────┐
   │  RawGenerationOutput          │   (internal dataclass, not persisted
   │  - candidate_index             │    as a manifest asset yet — Phase 3
   │  - image_path (staged, local)  │    consumes these before the real
   │  - width / height               │    GeneratedAsset / ImageGenerationResult
   │  - comfyui_prompt_id             │    is finalized)
   │  - node_id                       │
   │  - queue_wait_seconds             │
   │  - generation_seconds              │
   └──────────────────────────────┘
```

Two OS processes, unchanged from the architecture doc (§3): ComfyUI stays a long-lived, externally supervised process; `thumbnail-ai` stays the short-lived driver. Phase 2 adds **no** third process, no database, no message broker — it is a pure-Python client library around HTTP + one WebSocket connection, matching the "typed client around an external tool" pattern Module 2 already established for yt-dlp.

### 1.1 Why `modules/comfyui_client.py` instead of inline in `image_generator.py`

The architecture doc's component table (§4) lists `ComfyUIClient` under `modules/image_generator.py`. Phase 1 already established a precedent for splitting a table entry into its own file when the component has enough independent surface area to warrant isolated testing: `WorkflowLibrary` was specified in the same table row style but shipped as `modules/workflow_library.py`, imported by `image_generator.py`. Phase 2 follows that same precedent for `ComfyUIClient`, for the same reasons:

- HTTP + WebSocket transport code, retry/timeout policy, and queue-tracking bookkeeping is substantial enough (five internal collaborators, §3) that inlining it would roughly double the size of an already large `image_generator.py`.
- It has a fully independent, mockable test surface (§13) that does not depend on any other Phase 1 class — cleaner as its own test file (`tests/test_comfyui_client.py`) mirroring `tests/test_workflow_library.py`.
- `image_generator.py` continues to *import and use* `ComfyUIClient` exactly as the architecture doc's pipeline (§5, steps 5→6) describes; no orchestration behavior changes, only file layout. This is a packaging decision, not an architecture redesign — the public contract (`ComfyUIClient.generate(...)`) is what `image_generator.py` and, later, `main.py` depend on, and that contract is unchanged regardless of which file defines the class.

If preferred, Phase 2 can instead ship `ComfyUIClient` directly inside `image_generator.py` per the doc's literal table with no change to any other part of this design — the two options are interchangeable at the file level and this document's component contracts hold either way. The recommendation above is the default assumed for the rest of this document and the implementation checklist (§16).

---

## 2. Sequence diagrams (text-based)

### 2.1 Startup health check

```
main.py / image_generator.py          ComfyUIClient                ComfyUI server
        │                                    │                            │
        │  ComfyUIClient(host, port)          │                            │
        │───────────────────────────────────►│                            │
        │                                    │  GET /system_stats          │
        │                                    │───────────────────────────►│
        │                                    │◄───────────────────────────│
        │                                    │  200 OK {vram, devices}     │
        │  .health_check() -> SystemStats      │                            │
        │◄───────────────────────────────────│                            │
```
If the GET fails (connection refused / timeout) after the configured connect-retry window (§6), `ComfyUIClient.health_check()` raises `ComfyUIConnectionError` with an operator-facing message ("start ComfyUI first"), matching §28 of the architecture doc. `ProfileSelector` (Phase 1) already expects a VRAM reading from *somewhere*; Phase 2's `health_check()` is that somewhere — it is the first real (non-Phase-1-stubbed) source of `available_vram_gb`.

### 2.2 Successful single-candidate generation (happy path)

```
Caller (Phase 3, or a Phase 2 unit test)     ComfyUIClient          ComfyUI HTTP        ComfyUI WebSocket
        │                                          │                     │                     │
        │ generate(built_workflow, candidate_idx)   │                     │                     │
        │─────────────────────────────────────────►│                     │                     │
        │                                          │ ensure WS connected  │                     │
        │                                          │────────────────────────────────────────────►│ connect + client_id
        │                                          │ POST /prompt {graph, client_id}              │
        │                                          │────────────────────►│                     │
        │                                          │◄────────────────────│                     │
        │                                          │ 200 {prompt_id, number} │                  │
        │                                          │  (queue_wait timer starts)                  │
        │                                          │                     │  "status" {queue_remaining}│
        │                                          │◄─────────────────────────────────────────────│
        │                                          │  (generation timer starts on first "executing")│
        │                                          │                     │  "executing" {node, prompt_id}│
        │                                          │◄─────────────────────────────────────────────│
        │                                          │                     │  "progress" {value, max, node}│
        │                                          │◄─────────────────────────────────────────────│ (repeats)
        │                                          │                     │  "executing" {node: null, prompt_id}│
        │                                          │◄─────────────────────────────────────────────│ ← completion signal
        │                                          │ GET /history/{prompt_id}                       │
        │                                          │────────────────────►│                     │
        │                                          │◄────────────────────│                     │
        │                                          │ 200 {outputs: {node_id: {images:[...]}}}      │
        │                                          │ GET /view?filename&subfolder&type              │
        │                                          │────────────────────►│                     │
        │                                          │◄────────────────────│                     │
        │                                          │ 200 (image bytes)    │                     │
        │  RawGenerationOutput                       │                     │                     │
        │◄─────────────────────────────────────────│                     │                     │
```

### 2.3 VRAM-exhausted (OOM) with profile fallback

```
Caller                          ComfyUIClient              ComfyUI
  │  generate(...)                    │                        │
  │──────────────────────────────────►│  POST /prompt            │
  │                                   │───────────────────────►│
  │                                   │◄────────────────────────│
  │                                   │  WS "execution_error" {exception_type: "OutOfMemoryError", ...} │
  │                                   │  or history status.messages contains an OOM signature           │
  │                                   │  raise VRAMExhaustedError (Tenacity: retryable)                  │
  │  VRAMExhaustedError                │                        │
  │◄──────────────────────────────────│                        │
  │  (Phase 3 catches this, asks Phase 1's ProfileSelector      │
  │   for the next-lighter profile, rebuilds workflow, retries  │
  │   via a NEW ComfyUIClient.generate() call — Phase 2 itself  │
  │   does not know about profile fallback ladders; it only     │
  │   raises the typed signal Phase 3 acts on.)                 │
```
This keeps profile-fallback-on-OOM a Phase 1 (`ProfileSelector`) + Phase 3 (orchestration) concern, per the architecture doc §20: *"VRAMExhaustedError — retryable with Tenacity backoff plus an automatic fallback to a lighter GenerationProfile before failing the video."* Phase 2's job is limited to: (a) retry the *same* submission a bounded number of times via Tenacity for transient OOM (a queue-contention race can look identical to a real capacity OOM on a busy shared GPU), and (b) surface `VRAMExhaustedError` cleanly once retries are exhausted, so the *caller* can decide to downgrade. Phase 2 never silently swaps profiles itself — profile identity is not something the transport layer is allowed to change out from under the caller (that would break the workflow/generation hash contract in §23 of the architecture doc, which is entirely Phase 1/3's responsibility, not Phase 2's).

### 2.4 Connection lost mid-generation (WebSocket drop, HTTP still reachable)

```
ComfyUIClient                     ComfyUI WS                 ComfyUI HTTP
     │  (WS connection drops)          │                          │
     │  before_sleep_log: "WS closed, falling back to HTTP polling" │
     │  begin polling GET /history/{prompt_id} every POLL_INTERVAL   │
     │──────────────────────────────────────────────────────────────►│
     │◄─────────────────────────────────────────────────────────────│
     │  (repeat until outputs present or COMFYUI_EXECUTION_TIMEOUT_SECONDS elapses) │
     │  attempt WS reconnect in the background (best-effort, does not block polling) │
```
Polling is the documented fallback, never the primary path (WebSocket progress is materially more useful for logging/metrics, §9), but Phase 2 must not treat a dropped socket as a fatal error while the HTTP side is still healthy and the job is still in ComfyUI's queue/history.

### 2.5 Shutdown / resource cleanup (success or failure)

```
Caller                         ComfyUIClient
  │  (context manager exit, or explicit .close())│
  │──────────────────────────────────────────────►│
  │                                                │  close WebSocket (if open)
  │                                                │  close requests.Session
  │                                                │  flush + finalize any open GenerationMetrics record
  │                                                │  (queued prompts already submitted to ComfyUI are
  │                                                │   NOT cancelled on close unless the caller explicitly
  │                                                │   requested cancellation via .cancel(prompt_id))
```

---

## 3. Component breakdown

### 3.1 `ComfyUIClient` (public facade)

**Responsibility.** The single public entry point Phase 3 (and Phase 2's own tests) use. Owns one HTTP session, one WebSocket connection (lazily opened, reused across calls within a pipeline run), and coordinates the four internal collaborators below. Mirrors the "thin typed client" framing in the architecture doc §4.

**Public API** (signatures only — no implementation):

```python
class ComfyUIClient:
    def __init__(
        self,
        host: str = COMFYUI_HOST,
        port: int = COMFYUI_PORT,
        *,
        request_timeout_seconds: float = COMFYUI_REQUEST_TIMEOUT_SECONDS,
        execution_timeout_seconds: float = COMFYUI_EXECUTION_TIMEOUT_SECONDS,
        client_id: str | None = None,          # auto-generated uuid4 if omitted
    ) -> None: ...

    def health_check(self) -> SystemStats: ...
        # GET /system_stats -> typed dataclass (vram_free_mb, vram_total_mb, device_name, comfyui_version)
        # Raises ComfyUIConnectionError.

    def generate(
        self,
        built_workflow: BuiltWorkflow,        # from Phase 1 WorkflowBuilder
        candidate_index: int = 0,
        *,
        stage_output_dir: Path,               # e.g. MODULE7_OUTPUT_DIR / video_id / "_raw"
    ) -> RawGenerationOutput: ...
        # One full submit -> track -> retrieve cycle for ONE candidate.
        # Raises: ComfyUIConnectionError, ComfyUIQueueError, ComfyUITimeoutError,
        #         VRAMExhaustedError, ComfyUIOutputMissingError.

    def cancel(self, prompt_id: str) -> None: ...
        # POST /interrupt (current job) or POST /queue {"delete": [prompt_id]} (queued, not yet running).
        # Best-effort; swallows "already finished" as a no-op, not an error.

    def close(self) -> None: ...
        # Idempotent. Closes WS + HTTP session. Safe to call multiple times.

    def __enter__(self) -> "ComfyUIClient": ...
    def __exit__(self, *exc_info) -> None: ...
        # Calls close(); does not suppress exceptions.
```

**Internal workflow.** `generate()` is the orchestrating method and is intentionally the *only* multi-step method on the public surface:
1. Ensure WebSocket is connected (`_ComfyUIWebSocketTransport.ensure_connected()`); reconnect if a previous call left it closed.
2. Submit via `_ComfyUIHTTPTransport.submit_prompt(graph, client_id)` → `prompt_id`. Start `queue_wait` timer.
3. Hand `prompt_id` to `_QueueTracker.await_completion(prompt_id, timeout)`, which consumes WebSocket events (falling back to HTTP polling per §2.4) until a terminal state (`completed`, `execution_error`, `timeout`) is reached. Start `generation` timer on the first `executing` event carrying a non-null node for this `prompt_id`.
4. On `completed`: call `_OutputRetriever.fetch(prompt_id, stage_output_dir, candidate_index)` → `RawGenerationOutput`.
5. On `execution_error`: classify via `_classify_comfyui_error(message, exception_type)` into `VRAMExhaustedError` (retryable, see §6) or `ComfyUIQueueError` (non-retryable, e.g. missing model file / bad node params).
6. On timeout: attempt `cancel(prompt_id)` best-effort, then raise `ComfyUITimeoutError`.
7. `_ComfyUIMetricsRecorder.record_attempt(...)` is called in a `finally` block regardless of outcome, so failed attempts still contribute to `GenerationMetrics` (§10).

**Dependencies.** `requests` (already a project dependency, Module 3), a new dependency `websocket-client` (§7), `tenacity` (already a project dependency, Module 2), `loguru` (already a project dependency), Phase 1's `BuiltWorkflow`/`GenerationProfile`, `module7_exceptions`, `config.py` constants (§8).

**Data flow.** In: `BuiltWorkflow` (already-hashed, already-validated ComfyUI graph JSON from Phase 1) + a staging directory. Out: `RawGenerationOutput` (§4.1) — never a `GeneratedAsset` (that Pydantic model, per the architecture doc's Appendix A, represents a *finished, QA-eligible* asset; Phase 2's output is pre-restoration, pre-QA, and therefore deliberately a different, lighter internal type so nothing downstream can mistake a raw candidate for a finished one).

**Error propagation.** Every method either returns a typed value or raises one of the Phase 2 exceptions in §5 — never a raw `requests.exceptions.*` or a raw `websocket.*` exception escapes `ComfyUIClient`. Low-level exceptions are always caught at the transport boundary and re-raised as the appropriate `Module7Error` subclass, matching Module 2's `_fetch_yt_dlp_info` convention of converting all third-party exceptions to the module's own typed hierarchy before they leave the function that calls the third-party library.

**Configuration usage.** All timeouts, retry counts, host/port, and the WS path come from `config.py` (§8); nothing is hardcoded in the class body, matching every other module's convention.

**Logging behavior.** See §9. `ComfyUIClient` logs one INFO line per phase transition (submitted / queued / executing / progress-milestone / completed / retrieved) and one WARNING per retry (via the shared `_before_sleep_log`-style callback), never raw image bytes or full prompt text.

**Testing strategy.** See §13.1. Fully mocked HTTP (via `responses` or a hand-rolled `requests_mock`-style fixture already idiomatic to `pytest`) and a fake WebSocket server/mock, zero GPU, zero real ComfyUI install required for the default suite.

---

### 3.2 `_ComfyUIHTTPTransport` (internal)

**Responsibility.** Every HTTP call ComfyUI's REST-ish API supports that Phase 2 needs. Owns the single `requests.Session` (connection pooling/keep-alive, matching Module 3's `thumbnail_downloader.py` session-reuse pattern).

**Public API** (module-internal; not exported from `comfyui_client.py`'s `__all__`):

```python
class _ComfyUIHTTPTransport:
    def __init__(self, base_url: str, session: requests.Session, timeout_seconds: float) -> None: ...

    def system_stats(self) -> SystemStats: ...                       # GET /system_stats
    def submit_prompt(self, graph: dict, client_id: str) -> str: ...  # POST /prompt -> prompt_id
    def history(self, prompt_id: str) -> dict | None: ...             # GET /history/{prompt_id}
    def view_image(self, filename: str, subfolder: str, image_type: str) -> bytes: ...  # GET /view
    def interrupt(self) -> None: ...                                  # POST /interrupt
    def delete_from_queue(self, prompt_id: str) -> None: ...          # POST /queue {"delete":[id]}
    def queue_status(self) -> dict: ...                               # GET /queue (pending/running counts)
```

**Internal workflow.** Thin wrappers: build URL, `session.request(...)`, raise `_ComfyUIHTTPError` (a private, non-`Module7Error` transport-local signal — see note below) on non-2xx or connection failure, `response.raise_for_status()`, parse JSON where applicable, return typed/plain data. No retry logic lives here — retry policy is applied by the Tenacity decorator one layer up (§6), keeping this class a pure, easily-mocked I/O boundary, matching Module 2's separation between `_fetch_yt_dlp_info` (the retried call) and the yt-dlp library call itself.

> **Note on the private error type.** `_ComfyUIHTTPError` is intentionally *not* part of the public `module7_exceptions` hierarchy — it exists only so `_ComfyUIHTTPTransport`'s tests can assert precise HTTP-layer failures independently of how `ComfyUIClient` chooses to reclassify them (connection-refused → `ComfyUIConnectionError`, HTTP 400 with a node-validation payload → `ComfyUIQueueError`, etc.). `ComfyUIClient` is the only place `_ComfyUIHTTPError` is caught and translated; it never escapes past the facade, preserving the "typed boundary" project convention from §2 of the architecture doc even for this internal detail.

**Dependencies.** `requests` only.

**Data flow.** In: plain Python primitives + the already-built graph dict. Out: plain dicts/bytes/strings — deliberately *not* Pydantic models, matching Phase 1's own convention of keeping transient, non-persisted internal shapes as plain dataclasses/dicts and reserving `models.py` Pydantic types for things that get validated at a boundary or persisted.

**Error propagation.** Every `requests` exception (`ConnectionError`, `Timeout`, `HTTPError`, `JSONDecodeError`) is caught here and re-raised as `_ComfyUIHTTPError` with the original exception chained (`raise ... from exc`), never left to propagate as a bare `requests` exception into `ComfyUIClient`.

**Configuration usage.** `base_url` built once from `COMFYUI_HOST`/`COMFYUI_PORT`; `timeout_seconds` from `COMFYUI_REQUEST_TIMEOUT_SECONDS`.

**Logging behavior.** DEBUG-level only (each individual HTTP call is noisy; `ComfyUIClient` does the INFO-level phase logging). No response bodies logged beyond truncated previews.

**Testing strategy.** Unit tests mock `requests.Session` (or use `responses`) to assert exact URLs, methods, payloads, and header/timeout usage per call, plus exception translation for every non-2xx and connection-failure case.

---

### 3.3 `_ComfyUIWebSocketTransport` (internal)

**Responsibility.** Owns the single WebSocket connection to `ws://{host}:{port}/ws?clientId={client_id}` and turns raw frames into a small closed set of typed progress/completion events consumed by `_QueueTracker`.

**Public API** (module-internal):

```python
class _ComfyUIWebSocketTransport:
    def __init__(self, ws_url: str, client_id: str, connect_timeout_seconds: float) -> None: ...

    def ensure_connected(self) -> None: ...
        # No-op if already connected. Raises ComfyUIConnectionError on failure to establish.

    def next_event(self, timeout_seconds: float) -> ComfyUIEvent | None: ...
        # Blocking read with timeout; returns None on a read timeout (caller decides
        # whether that's "keep waiting" or "give up", not this method).
        # Parses ComfyUI's JSON text frames into a typed ComfyUIEvent (§4.2);
        # silently ignores/skips binary frames (ComfyUI sends binary preview
        # frames for live-preview images, which Phase 2 does not consume —
        # no live-preview feature in this phase).

    def is_connected(self) -> bool: ...
    def close(self) -> None: ...   # idempotent
```

**Internal workflow.** Uses the `websocket-client` library's `WebSocket` (sync, blocking-with-timeout) client — not `websockets`/`asyncio`, to stay consistent with the rest of the codebase, which is 100% synchronous (confirmed: no `asyncio` usage anywhere in `modules/`). `next_event` sets a per-call socket timeout, attempts `recv()`, and on a `WebSocketTimeoutException` returns `None` rather than raising, so `_QueueTracker` can distinguish "no news yet, keep waiting" from "the connection is actually broken" (`WebSocketConnectionClosedException`, which *is* re-raised as `ComfyUIConnectionError` so `_QueueTracker` can fall back to polling per §2.4).

**Dependencies.** `websocket-client` (new dependency, §7).

**Data flow.** In: nothing beyond timeouts. Out: `ComfyUIEvent | None` per call.

**Error propagation.** `WebSocketConnectionClosedException` / `ConnectionRefusedError` on connect → `ComfyUIConnectionError`. Malformed JSON frame → logged at WARNING and skipped (treated as `None`, not fatal — a single corrupt progress frame should never fail a whole generation when the HTTP history endpoint remains the source of truth for final completion).

**Configuration usage.** `COMFYUI_WS_PATH`, `COMFYUI_WEBSOCKET_TIMEOUT_SECONDS` (per-`next_event` read timeout — short, e.g. 2–5s, so `_QueueTracker` can periodically re-check the overall execution timeout budget), `COMFYUI_STARTUP_TIMEOUT_SECONDS` (initial connect).

**Logging behavior.** DEBUG for every event; `ComfyUIClient`/`_QueueTracker` decide what's INFO-worthy.

**Testing strategy.** A fake local WebSocket server fixture (e.g. `pytest-websocket`-style, or a minimal hand-rolled `threading`-based echo/scripted server) that plays back a scripted sequence of ComfyUI-shaped JSON frames (status → executing → progress×N → executing(node=null)) so `_QueueTracker`'s state machine can be tested end-to-end without any real ComfyUI process. A second test class using the same fixture to simulate a mid-stream disconnect, exercising the polling-fallback path (§2.4).

---

### 3.4 `_QueueTracker` (internal)

**Responsibility.** The state machine that answers "is this `prompt_id` done, still running, queued, or errored?" — the piece that makes queue management and completion detection each independently testable, matching the architecture doc's "each pipeline stage independently testable" philosophy (§2).

**Public API** (module-internal):

```python
class _QueueTracker:
    def __init__(
        self,
        http: "_ComfyUIHTTPTransport",
        ws: "_ComfyUIWebSocketTransport",
        poll_interval_seconds: float,
        execution_timeout_seconds: float,
    ) -> None: ...

    def await_completion(self, prompt_id: str, client_id: str) -> _CompletionOutcome: ...
        # Returns a small closed-set outcome: COMPLETED | EXECUTION_ERROR | TIMEOUT,
        # plus the raw ComfyUI error payload when EXECUTION_ERROR.
```

**Internal workflow.**
1. Loop until `execution_timeout_seconds` elapses (tracked via a monotonic clock, not wall time, to be immune to system clock adjustments):
   a. Try `ws.next_event(COMFYUI_WEBSOCKET_TIMEOUT_SECONDS)`.
   b. If an event for a *different* `prompt_id`/`client_id` arrives (ComfyUI's WS is not per-connection-scoped to one job), ignore it and continue — this matters because ComfyUI is a shared server and other jobs (including from other tools) could be interleaved on the wire.
   c. On `status` events: update queue-position bookkeeping (used for `queue_wait_seconds` metric and WARNING-level "still queued after N seconds" logging).
   d. On `executing` with `node != null` for our `prompt_id`: mark generation as started (ends the queue-wait timer), forward to progress logging.
   e. On `executing` with `node == null` for our `prompt_id`: **completion signal** — break the loop, then do one confirming `http.history(prompt_id)` call to fetch the actual output filenames (the WS event itself does not carry output paths).
   f. On `execution_error` for our `prompt_id`: break the loop with `EXECUTION_ERROR` outcome.
   g. On a `WebSocketConnectionClosedException` surfaced as `ComfyUIConnectionError` from `ws.next_event`: switch to polling mode — call `http.history(prompt_id)` every `COMFYUI_POLL_INTERVAL_SECONDS` instead, and best-effort attempt `ws.ensure_connected()` again every few poll cycles so progress logging can resume if the socket recovers. Polling mode still respects the same overall `execution_timeout_seconds` budget.
   h. If `execution_timeout_seconds` elapses with no terminal event: `TIMEOUT` outcome.
2. Never blocks indefinitely — every blocking call underneath (`ws.next_event`, `http.history`) has its own bounded timeout, and the outer loop enforces the aggregate budget independently, so a hung ComfyUI process cannot hang the pipeline.

**Dependencies.** `_ComfyUIHTTPTransport`, `_ComfyUIWebSocketTransport`.

**Data flow.** In: `prompt_id`, `client_id`. Out: `_CompletionOutcome` (internal enum + optional error payload dict).

**Error propagation.** Does not raise `Module7Error` subclasses itself — it returns a typed outcome and lets `ComfyUIClient.generate()` do the final translation into `VRAMExhaustedError` / `ComfyUIQueueError` / `ComfyUITimeoutError`, keeping the state machine a pure "what happened" component rather than a "what should the caller do about it" component.

**Configuration usage.** `COMFYUI_POLL_INTERVAL_SECONDS`, `COMFYUI_EXECUTION_TIMEOUT_SECONDS`, `COMFYUI_WEBSOCKET_TIMEOUT_SECONDS`.

**Logging behavior.** INFO on phase transitions (queued → executing → completed), INFO with elapsed time on every progress milestone (e.g., every 25% of `progress.max`, not every single `progress` frame — ComfyUI can emit progress events every sampler step, and logging every one of them at INFO would flood `module7.log`; DEBUG gets every raw frame), WARNING on the WS-drop-to-polling fallback, WARNING on "still queued after N seconds" (configurable threshold, helps operators notice a stuck/overloaded ComfyUI instance).

**Testing strategy.** Pure state-machine unit tests driving a scripted fake `_ComfyUIWebSocketTransport`/`_ComfyUIHTTPTransport` pair (both easily fakeable as they're small internal interfaces) — assert the exact outcome and timing bookkeeping for: normal completion, execution error, timeout, WS-drop-then-recover, WS-drop-then-poll-to-completion, and interleaved-other-job-events-ignored.

---

### 3.5 `_OutputRetriever` (internal)

**Responsibility.** Turn a completed `prompt_id`'s history payload into actual image bytes on local disk, staged for Phase 3 to pick up.

**Public API** (module-internal):

```python
class _OutputRetriever:
    def __init__(self, http: "_ComfyUIHTTPTransport") -> None: ...

    def fetch(
        self,
        prompt_id: str,
        history_payload: dict,
        stage_output_dir: Path,
        candidate_index: int,
    ) -> RawGenerationOutput: ...
```

**Internal workflow.**
1. Walk `history_payload["outputs"]` to find the node(s) with an `images` list (the workflow's `SaveImage`-equivalent output node — Phase 1's `WorkflowBuilder`/workflow templates are responsible for ensuring exactly one such node exists per graph; §7.3/§8 of the architecture doc already require templates to be schema-valid, and this is the natural place to add a template-authoring convention documented alongside `_meta`).
2. If zero output-image entries are found: raise `ComfyUIOutputMissingError` (new, §5) — this indicates a template authoring bug (missing/misconfigured save node), not a transient failure, so it is **not** retried by Tenacity.
3. If more than one output-image entry is found: take the first in a deterministic (sorted-by-node-id) order and log a WARNING — multiple save nodes in one graph is unexpected for the shipped templates but should degrade predictably rather than crash, matching the project's "fail loudly and typed, never silently degrade quality" philosophy while still making forward progress on an otherwise-successful generation.
4. For the chosen image entry: `http.view_image(filename, subfolder, type)` → raw bytes.
5. Read image dimensions via Pillow (already a project dependency, Module 3) without doing any other processing — Phase 2 does not resize, crop, or otherwise touch pixels; that begins in Phase 3.
6. Atomic write: temp-file-then-`Path.replace()` into `stage_output_dir / f"candidate_{candidate_index}.png"`, matching the project-wide atomic-write convention.
7. Compute a SHA-256 of the written bytes (reusing `canonical_json_hash`'s sibling pattern from Phase 1's `image_generator.py`, but over raw bytes rather than JSON — a small new `sha256_of_file` helper) for the manifest/metrics trail even at this pre-QA stage, so a later mismatch between "what ComfyUI produced" and "what Phase 3 processed" is detectable.
8. Return `RawGenerationOutput` (§4.1).

**Dependencies.** `_ComfyUIHTTPTransport`, `Pillow` (already a dependency).

**Data flow.** In: history payload (dict), staging directory, candidate index. Out: `RawGenerationOutput`.

**Error propagation.** `ComfyUIOutputMissingError` (non-retryable, template bug) vs. a transport-level failure fetching the bytes (`_ComfyUIHTTPError` → `ComfyUIConnectionError`, retryable via the same Tenacity policy as submission, since a `/view` fetch failing right after a successful `/history` read is exactly the kind of transient network blip Tenacity exists for).

**Configuration usage.** None beyond what's threaded through from `ComfyUIClient` (stage dir comes from the caller, per `MODULE7_OUTPUT_DIR`).

**Logging behavior.** INFO with candidate index, byte size, dimensions, and sha256 on success.

**Testing strategy.** Unit tests over synthetic `history_payload` fixtures (well-formed, missing-images, multi-image) with a mocked `_ComfyUIHTTPTransport.view_image` returning fixed PNG bytes (a tiny real 1×1 or 8×8 PNG fixture, not a mock object, so Pillow's dimension read is exercised for real) — asserts exact staged file path, atomic-write behavior (temp file cleaned up on a simulated write failure), and hash correctness.

---

### 3.6 `_ComfyUIMetricsRecorder` (internal)

**Responsibility.** Assemble the `GenerationMetrics` fields Phase 2 owns (queue/generation timing, retry counts, failure reason) and hand them to Phase 1's existing `MetricsCollector.append()` — Phase 2 does not reimplement metrics persistence, it only *produces* the record.

**Public API** (module-internal):

```python
class _ComfyUIMetricsRecorder:
    def __init__(self, collector: "MetricsCollector") -> None: ...  # Phase 1's MetricsCollector, reused as-is

    def record_attempt(
        self,
        video_id: str,
        niche: str,
        workflow_ref: WorkflowTemplateRef,
        outcome: "_AttemptOutcome",   # success | which exception type | timing breakdown
    ) -> None: ...
```

**Internal workflow.** Builds one `GenerationMetrics` instance (Phase 1 Pydantic model, unchanged — see §10 for the specific field mapping) and calls the existing `MetricsCollector.append()` from Phase 1's `image_generator.py`. This keeps "how metrics get durably written" a single-owner responsibility (Phase 1's `MetricsCollector`, already atomic-append-with-fsync) while Phase 2 only supplies *what* to write for a ComfyUI attempt.

**Dependencies.** Phase 1's `MetricsCollector`, Phase 1's `GenerationMetrics` model.

**Data flow.** In: attempt outcome (timings + optional failure classification). Out: none (side effect: one JSONL line appended).

**Error propagation.** A metrics-append failure raises Phase 1's existing `MetricsWriteError`; per Phase 1's own `MetricsCollector` docstring ("a passive observer, never a control-flow participant"), Phase 2's `ComfyUIClient.generate()` catches `MetricsWriteError` and logs it at ERROR rather than letting a monitoring-write failure mask or replace the *actual* generation outcome being returned/raised to the caller.

**Configuration usage.** None directly — inherits `MODULE7_METRICS_PATH` via the injected `MetricsCollector`.

**Logging behavior.** DEBUG confirmation of append; ERROR on `MetricsWriteError` (caught, not re-raised, per above).

**Testing strategy.** Unit tests asserting the exact `GenerationMetrics` field mapping for a success case and each failure-classification case, using a fake/spy `MetricsCollector` (or the real one pointed at a tmp path, matching how Phase 1's own tests likely already exercise `MetricsCollector`).

---

## 4. Data models

Phase 2 deliberately introduces **no new Pydantic models in `models.py`**. Everything Phase 2 produces that needs to survive past a single method call is either (a) an internal, non-persisted dataclass (this section), or (b) a field-level contribution to a Phase 1 Pydantic model that Phase 3 finalizes and persists (`ImageGenerationResult`, `GenerationMetrics` — §10). This mirrors Phase 1's own pattern: `ReferenceAssets` and `BuiltWorkflow` are plain frozen dataclasses, not Pydantic models, precisely because they're transient shapes passed between in-process stages rather than validated-at-a-boundary or persisted-to-disk types. `RawGenerationOutput` and `SystemStats` below follow that same precedent.

### 4.1 `RawGenerationOutput` (frozen dataclass, `comfyui_client.py`)

```python
@dataclass(frozen=True)
class RawGenerationOutput:
    candidate_index: int
    image_path: Path            # staged local file, e.g. .../<video_id>/_raw/candidate_0.png
    width: int
    height: int
    sha256: str
    comfyui_prompt_id: str
    queue_wait_seconds: float
    generation_seconds: float
```

### 4.2 `ComfyUIEvent` (frozen dataclass, `comfyui_client.py`)

```python
@dataclass(frozen=True)
class ComfyUIEvent:
    event_type: Literal["status", "executing", "progress", "execution_error", "execution_cached"]
    prompt_id: str | None
    node: str | None
    progress_value: int | None
    progress_max: int | None
    queue_remaining: int | None
    error_payload: dict | None
```
A single typed shape for every WS frame Phase 2 cares about, produced by `_ComfyUIWebSocketTransport`'s frame parser. Frame types ComfyUI emits that Phase 2 does not act on (e.g., binary preview images) never reach this dataclass — they're filtered at the parser boundary, keeping `_QueueTracker`'s state machine free of "ignore this" branches for concerns Phase 2 has no opinion on.

### 4.3 `SystemStats` (frozen dataclass, `comfyui_client.py`)

```python
@dataclass(frozen=True)
class SystemStats:
    vram_free_mb: float
    vram_total_mb: float
    device_name: str
    comfyui_version: str
```
Consumed by Phase 1's `ProfileSelector.select(available_vram_gb=...)` — Phase 2's `health_check()` is what finally supplies a *real* VRAM reading (`vram_free_mb / 1024`) in place of whatever placeholder/manual value Phase 1 tests use today. No change to `ProfileSelector`'s signature is required; this is purely "Phase 2 now has a real answer to a question Phase 1 already knew how to ask."

### 4.4 Internal-only enums

`_CompletionOutcome` (`COMPLETED`, `EXECUTION_ERROR`, `TIMEOUT`) and `_AttemptOutcome` (a small dataclass bundling the above with timing) are private to `comfyui_client.py`, not exported, and never appear in a public signature — they exist purely to make `_QueueTracker` and `_ComfyUIMetricsRecorder`'s internals independently unit-testable without stringly-typed state.

---

## 5. Exception hierarchy

Extends the existing `Module7Error` base from `modules/module7_exceptions.py` (Phase 1). Three exceptions already exist and get their **first real raise sites** in Phase 2 (they were declared in Phase 1 as forward-declared/reserved, per their current docstrings):

| Exception | Phase | Raised when | Retryable? |
|---|---|---|---|
| `ComfyUIConnectionError` | *(existing, Phase 1 stub → Phase 2 implements)* | Server unreachable at `health_check()`, WS connect failure, or HTTP connection failure exhausting Tenacity's own connection-retry window (§6). | Retried a bounded number of times by Tenacity *before* this is raised; once raised, it is terminal for that `generate()` call. |
| `ComfyUIQueueError` | *(existing, Phase 1 stub → Phase 2 implements)* | ComfyUI's `execution_error` payload indicates a non-OOM node/graph problem (bad param, missing model file referenced by the profile/template). | No — indicates a `WorkflowBuilder`/config/setup bug, not a transient condition, per the architecture doc §20. |
| `VRAMExhaustedError` | *(existing, Phase 1 stub → Phase 2 implements)* | `execution_error` payload/message matches an OOM signature (`_classify_comfyui_error`, mirroring Module 2's `_classify_transient_error` pattern). | Yes, bounded Tenacity retries of the *same* submission; caller (Phase 3) decides whether to retry again with a lighter profile. |
| `ComfyUITimeoutError` *(NEW, Phase 2)* | `_QueueTracker.await_completion` exceeds `COMFYUI_EXECUTION_TIMEOUT_SECONDS` with no terminal event. | No — a timeout this long indicates a stuck/overloaded server, not a blip; surfaced immediately so Phase 3/`main.py`'s per-video error isolation can move on. |
| `ComfyUIOutputMissingError` *(NEW, Phase 2)* | `_OutputRetriever` finds a completed history payload with zero output-image entries. | No — template authoring bug. |
| `ComfyUIResponseError` *(NEW, Phase 2)* | ComfyUI returns 200 with a payload that fails Phase 2's minimal structural expectations (e.g., `/prompt` response missing `prompt_id`) — distinct from `_ComfyUIHTTPError` in that it's a *protocol contract* violation, not a transport failure. | No — indicates a ComfyUI version mismatch the deployment docs (§28 of the architecture doc) should catch, not something retrying fixes. |

All five extend `Module7Error` directly (flat hierarchy, no sub-bases), matching the existing flat structure in `module7_exceptions.py` rather than introducing a new `ComfyUIError` intermediate base — consistent with how Phase 1 kept `WorkflowTemplateError`, `ReferenceAssetError`, etc. as siblings rather than nesting them.

`_ComfyUIHTTPError` (§3.2) is **not** part of this table — it is a private, transport-internal signal that never escapes `_ComfyUIHTTPTransport`/`_ComfyUIWebSocketTransport` and is always translated into one of the above (or re-raised as one of the above by `ComfyUIClient`) before reaching a caller.

`module7_exceptions.py` requires exactly two additive changes for Phase 2 (both are new class declarations only — no existing class is modified, matching the project's "additive only" convention):

```python
class ComfyUITimeoutError(Module7Error):
    """Raised when a submitted workflow does not reach a terminal state before the execution timeout."""


class ComfyUIOutputMissingError(Module7Error):
    """Raised when a completed ComfyUI execution reports no retrievable output image."""


class ComfyUIResponseError(Module7Error):
    """Raised when ComfyUI returns a structurally invalid response to a known-good request."""
```

The existing docstrings on `IdentityPreservationError`, `QualityAssuranceError`, and `NoEligibleCandidateError` ("Reserved for Phase 2 ...") remain untouched by this phase — those three are Phase 3/Phase 4 concerns per the objectives in the task brief (identity/QA/ranking are explicitly listed as *separate* future phases, not part of "ComfyUI Integration"). Renaming their docstrings from "Phase 2" to "Phase 3"/"Phase 4" for accuracy is a one-line documentation cleanup worth doing during Phase 2 implementation (§16) even though it changes no behavior.

---

## 6. Retry strategy

Follows the exact Tenacity pattern already established in Module 2 (`_before_sleep_log` + `@retry(...)`), reused rather than reinvented, per the architecture doc §21 ("Tenacity's existing `_before_sleep_log` callback pattern is reused for all retryable operations ... so retry visibility is consistent across the whole codebase, not reinvented per module").

**Two independent retry layers**, kept separate because they retry different failure classes with different budgets:

1. **Connection-level retry** (inside `_ComfyUIHTTPTransport`/`_ComfyUIWebSocketTransport` connect paths). Wraps only the *connect*/`submit_prompt` calls. Config: `COMFYUI_CONNECT_RETRY_ATTEMPTS` (default 3), `wait_exponential(min=COMFYUI_CONNECT_RETRY_WAIT_MIN_SECONDS, max=COMFYUI_CONNECT_RETRY_WAIT_MAX_SECONDS)` (defaults 2s/10s, matching Module 2's own min/max exactly for consistency), `retry_if_exception_type(_ComfyUIHTTPError)` for HTTP connect, analogous for the WS connect. Exhausting this raises `ComfyUIConnectionError`.

2. **VRAM-OOM retry** (inside `ComfyUIClient.generate()`, wrapping the whole submit→await_completion cycle). Config: `MODULE7_COMFYUI_OOM_RETRY_ATTEMPTS` (default 2, deliberately smaller than the connection retry budget — an OOM retry re-submits an entire generation, which is expensive, unlike a connection retry which is cheap), same exponential backoff shape, `retry_if_exception_type(VRAMExhaustedError)`. Exhausting this re-raises `VRAMExhaustedError` to the caller (Phase 3 decides on profile fallback, §2.3).

Both layers use a single shared `_before_sleep_log` callback in `comfyui_client.py`, built the same way Module 2's is built: logs `str(exc)` only, **never the exception object itself**, for the exact pickling-safety reason already documented in `youtube_metadata.py`'s comment (Loguru + `enqueue=True` pickles log records for the background writer thread; traceback objects aren't picklable). This is a direct, deliberate copy of an existing, hard-won project convention — not a new invention.

**What is explicitly not retried:** `ComfyUIQueueError`, `ComfyUIOutputMissingError`, `ComfyUIResponseError`, `ComfyUITimeoutError` — each represents either a config/template bug or an already-exhausted wait budget, where retrying identically would not change the outcome (matching the architecture doc §20's own retryable/non-retryable classification for the pre-existing three).

---

## 7. Timeout strategy

Three independent timeout budgets, each protecting a different failure mode, all sourced from `config.py` (§8) rather than hardcoded:

| Timeout | Protects against | Config constant | Default |
|---|---|---|---|
| Per-HTTP-request timeout | A single `requests` call hanging (DNS, TCP connect, slow response). | `COMFYUI_REQUEST_TIMEOUT_SECONDS` *(existing, Phase 1)* | 120s |
| WebSocket per-read timeout | `next_event()` blocking forever when nothing is happening; also the granularity at which `_QueueTracker` re-checks the overall execution budget and can react to a caller-level cancellation. | `COMFYUI_WEBSOCKET_TIMEOUT_SECONDS` *(new)* | 5s |
| Overall execution timeout | The whole submit→complete cycle for one candidate taking unreasonably long (stuck sampler, ComfyUI deadlock, GPU driver hang) — independent of and much larger than the two above. | `COMFYUI_EXECUTION_TIMEOUT_SECONDS` *(new)* | 300s (5 min) — comfortably above `PROFILE_PREMIUM`'s documented `expected_generation_seconds` upper bound of ~70s (architecture doc §6.2) plus generous queue-wait headroom for a busy shared GPU. |

The overall execution timeout is intentionally *not* derived automatically from the selected `GenerationProfile.expected_generation_seconds` (e.g., `expected_time * 3`) even though that's tempting — profile-aware timeout scaling is exactly the kind of "ad hoc combination of raw parameters" the architecture doc's design philosophy (§2, §6) says orchestration code should avoid. Instead, `COMFYUI_EXECUTION_TIMEOUT_SECONDS` is one flat, documented, config-controlled ceiling; if a future profile's expected time approaches it, that is a config change (bump the constant, same as any other profile-driven config tuning), not new branching logic in `_QueueTracker`.

Startup/health-check has its own pre-existing `COMFYUI_STARTUP_TIMEOUT_SECONDS` (Phase 1's config already reserved this), used both for the very first `health_check()` call and as the WS *connect* timeout (distinct from the WS *per-read* timeout above).

---

## 8. Configuration additions

All additive to `modules/config.py`, immediately below the existing `# Module 7 — Local Image Generation Engine (Phase 1 foundation only)` block — Phase 2 implementation should also update that header comment to `(Phase 1 + Phase 2)` since it stops being foundation-only. No existing Phase 1 constant is modified.

```python
# --- Phase 2: ComfyUI transport ---

COMFYUI_WS_PATH: str = "/ws"

COMFYUI_CONNECT_RETRY_ATTEMPTS: int = 3
COMFYUI_CONNECT_RETRY_WAIT_MIN_SECONDS: float = 2.0
COMFYUI_CONNECT_RETRY_WAIT_MAX_SECONDS: float = 10.0

MODULE7_COMFYUI_OOM_RETRY_ATTEMPTS: int = 2
MODULE7_COMFYUI_OOM_RETRY_WAIT_MIN_SECONDS: float = 3.0
MODULE7_COMFYUI_OOM_RETRY_WAIT_MAX_SECONDS: float = 15.0

COMFYUI_WEBSOCKET_TIMEOUT_SECONDS: float = 5.0
COMFYUI_EXECUTION_TIMEOUT_SECONDS: float = 300.0
COMFYUI_POLL_INTERVAL_SECONDS: float = 3.0

MODULE7_PROGRESS_LOG_GRANULARITY_PERCENT: int = 25   # log at every 25% of sampler progress, not every step
MODULE7_STILL_QUEUED_WARNING_SECONDS: float = 30.0    # WARNING if still queued (not yet executing) past this

MODULE7_RAW_STAGE_SUBDIR: str = "_raw"    # under MODULE7_OUTPUT_DIR / <video_id> / — Phase 2 staging only
```

`COMFYUI_HOST`, `COMFYUI_PORT`, `COMFYUI_STARTUP_TIMEOUT_SECONDS`, `COMFYUI_REQUEST_TIMEOUT_SECONDS` are already present from Phase 1 and are reused as-is (§19 of the architecture doc already specified them). No renaming, no value changes.

---

## 9. Logging strategy

Sink, rotation, and format are **unchanged** from Phase 1 — `comfyui_client.py` calls the same `_configure_logger()` pattern already duplicated between `image_generator.py` and `workflow_library.py` (attach to `MODULE7_LOG_PATH`, `rotation="10 MB"`, `retention="30 days"`, same format string, `enqueue=True`). This is a third, small, intentional duplication of that six-line function, consistent with how Phase 1 already duplicated it once (`workflow_library.py`) rather than factoring it into a shared helper — a call this document flags but does not change, since deduplicating it is a cross-cutting refactor of *existing* files, which is explicitly out of scope ("do not redesign previous modules").

**Log level policy** (extending §21 of the architecture doc with Phase 2 specifics):

- **INFO**: health check result (VRAM reading), submission (`prompt_id`, candidate index, workflow name/hash), queue-wait completion, generation start, progress at `MODULE7_PROGRESS_LOG_GRANULARITY_PERCENT` milestones, completion (duration, output filename, sha256), cancellation, close.
- **WARNING**: any retry (via `_before_sleep_log`), WS-drop-to-polling fallback, still-queued-too-long, multiple-output-nodes-found degradation (§3.5).
- **ERROR**: terminal failures right before each `Module7Error` subclass is raised (message includes `prompt_id` and video context when available), `MetricsWriteError` caught-and-logged case (§3.6).
- **DEBUG**: every raw HTTP call, every raw WS frame (pre-filter), per-step progress frames.

**What is never logged**, per the architecture doc §21's existing rule extended to Phase 2's own payloads: raw image bytes (obviously), full ComfyUI graph JSON at INFO (DEBUG may include it truncated), and — new for Phase 2 — full `execution_error` tracebacks from ComfyUI's payload are logged as their string message only (same pickling-safety rationale as `_before_sleep_log`, §6), never the raw payload dict passed through `exc=`.

---

## 10. Manifest & metrics integration

Phase 2 does not write `ImageGenerationResult` manifests itself (`ArtifactWriter` remains Phase 1's, invoked once per *finished, QA'd* result — a Phase 3/4 concern once those stages exist to produce a finishable result). Phase 2's contribution is exclusively the **inputs** those later stages need, plus **`GenerationMetrics`**, which Phase 2 *does* populate per attempt (a `GenerationMetrics` record is defined as "one append-only... record for a Module 7 **attempt**" — Phase 1's own docstring — so a Phase 2 attempt, successful or not, is exactly the right granularity for it, independent of whether a full `ImageGenerationResult` manifest is ever produced for that video).

**Field mapping, `_ComfyUIMetricsRecorder` → `GenerationMetrics`** (existing Phase 1 model, no schema change):

| `GenerationMetrics` field | Phase 2 source |
|---|---|
| `video_id`, `niche` | Passed through from the caller (Phase 3's per-video loop already has these). |
| `profile_name` | From the `GenerationProfile` used to build the workflow. |
| `workflow_version`, `workflow_hash` | From `BuiltWorkflow.workflow_ref.workflow_version` / `BuiltWorkflow.workflow_hash` (Phase 1). |
| `generation_hash` | Left `None` at this stage — it requires model/LoRA/checkpoint hashes not yet resolved until later in the pipeline; Phase 3 fills this in when it finalizes `ImageGenerationResult`. |
| `queue_time_seconds` | Measured by `_QueueTracker` (submission → first "executing" event). |
| `generation_time_seconds` | List, one entry appended per `ComfyUIClient.generate()` call for that video (supports multi-candidate accumulation across sequential Phase 2 calls). |
| `identity_retry_count`, `generation_retry_count` | `generation_retry_count` incremented by Phase 2 on every Tenacity-driven OOM retry it performed; `identity_retry_count` remains `0` from Phase 2 (owned entirely by Phase 3). |
| `failure_reason` | Short string classification (`"connection_error"`, `"queue_error"`, `"vram_exhausted"`, `"timeout"`, `"output_missing"`, `"response_error"`, or `None` on success) — never a full traceback. |
| `peak_vram_mb` | Best-effort: one extra `health_check()` call *after* generation completes, if `MODULE7_MAX_CONCURRENT_GENERATIONS == 1` (the documented default, §19 of the architecture doc) — comparing pre/post free VRAM gives a reasonable peak-usage proxy without ComfyUI exposing true per-job peak tracking. Left `None` if the extra call itself fails (never fail an otherwise-successful generation over a metrics nicety). |
| all other fields (`num_candidates_requested`, `identity_failures_count`, `qa_failures_count`, `winning_overall_score`, `winning_signal_scores`) | Left at their Pydantic defaults by Phase 2 — populated later by whichever Phase 3/4 component owns candidate aggregation. |

This keeps `GenerationMetrics` a single model that different phases contribute different slices of, rather than Phase 2 needing its own parallel metrics type — consistent with "Additive only" and avoiding a second source of truth for per-attempt telemetry.

---

## 11. Error handling summary (end-to-end)

| Failure | Where caught | Exception surfaced | Retried? | Effect on pipeline |
|---|---|---|---|---|
| ComfyUI not running / refuses connection | `_ComfyUIHTTPTransport`/`_ComfyUIWebSocketTransport` connect | `ComfyUIConnectionError` | Yes (connection layer, §6) then terminal | Whole `generate()` call fails; per architecture doc §28, this should read as "start ComfyUI first," not a stack trace, at the log/error-message level. |
| Bad node params / missing model file | `execution_error` WS event or history status | `ComfyUIQueueError` | No | Config/template bug — video marked `status: "error"` once Phase 3 catches it; not something re-submitting fixes. |
| GPU OOM | `execution_error` classified via `_classify_comfyui_error` | `VRAMExhaustedError` | Yes (OOM layer, §6) then terminal | Phase 3 catches, may retry with a lighter `GenerationProfile` (new `ComfyUIClient.generate()` call). |
| Stuck/hung job | `_QueueTracker` budget exceeded | `ComfyUITimeoutError` | No | `cancel()` attempted best-effort; video fails this attempt. |
| Save node missing from template | `_OutputRetriever` finds no images | `ComfyUIOutputMissingError` | No | Template authoring bug; should be caught by `WorkflowLibrary`/golden-manifest tests (§13) before reaching production. |
| Malformed ComfyUI response shape | `_ComfyUIHTTPTransport`/`ComfyUIClient` structural check | `ComfyUIResponseError` | No | Version-mismatch signal; operator-facing, points at deployment docs (§28 of architecture doc). |
| WS drops mid-job, HTTP still up | `_QueueTracker` | *(not an exception — falls back to polling, §2.4)* | N/A | Transparent to the caller; only visible as a WARNING log line. |
| Metrics append fails | `_ComfyUIMetricsRecorder` | Phase 1's `MetricsWriteError`, caught internally | N/A | Logged at ERROR, does not affect the `generate()` call's actual return/raise. |

Every terminal exception, on the way out of `ComfyUIClient.generate()`, is guaranteed (via a `finally` block) to have already triggered one `_ComfyUIMetricsRecorder.record_attempt(...)` call, so failed attempts are never invisible to `module7_metrics.jsonl` even when no manifest is ever written for that video.

---

## 12. Resource cleanup

- `ComfyUIClient` is a context manager (`__enter__`/`__exit__`); Phase 3's per-video loop is expected to use `with ComfyUIClient(...) as client:` for the *whole* per-video Module 7 stage (not re-opened per candidate), so the WebSocket connection and HTTP session are reused across multiple `generate()` calls for the same video's candidates — avoiding reconnect overhead per candidate while still guaranteeing teardown on any exception via the context manager protocol.
- `close()` is idempotent and safe to call from an already-failed state (e.g., WS never successfully connected) — it never raises.
- Submitted-but-abandoned ComfyUI jobs (e.g., the pipeline is killed mid-generation) are **not** Phase 2's responsibility to track across process restarts — ComfyUI's own queue/history is the source of truth for in-flight jobs, and a documented operational note (mirroring §28 of the architecture doc) is added: a leftover queued/running job from a killed pipeline run will simply finish or sit in ComfyUI's history untouched; it does not corrupt subsequent runs since every `generate()` call uses a fresh `client_id`-scoped submission.
- No temp files are left behind on failure: `_OutputRetriever`'s atomic write pattern (temp-then-`Path.replace()`) means a failed retrieval never leaves a partial `candidate_N.png` in the staging directory, matching `ArtifactWriter`'s existing cleanup-on-`OSError` pattern from Phase 1 (`temporary.unlink(missing_ok=True)` in the `except` branch).
- `cancel()` is explicitly available for callers (Phase 3, or a future timeout-driven supervisor) to proactively free GPU resources rather than waiting out a job that's no longer wanted (e.g., after `MAX_GENERATION_RETRIES` is reached at a higher layer and the current in-flight candidate is being abandoned in favor of a fresh attempt).

---

## 13. Testing strategy

Mirrors §22 of the architecture doc precisely, scoped to Phase 2's components only:

### 13.1 Default (fast, CI, no GPU) suite — `tests/test_comfyui_client.py`

- **`_ComfyUIHTTPTransport`**: mocked `requests.Session` (or `responses`) — exact URL/method/payload assertions per endpoint, exception-translation coverage for connection-refused, timeout, non-2xx, malformed JSON.
- **`_ComfyUIWebSocketTransport`**: fake local WebSocket server (threaded, scripted frame playback) — connect success/failure, `next_event` timeout-returns-None, malformed-frame-skipped, connection-closed-mid-read raises `ComfyUIConnectionError`.
- **`_QueueTracker`**: fully scripted fake HTTP+WS pair — every outcome in §11's table exercised as a pure state-machine test, plus the WS-drop-to-polling transition and the ignore-other-jobs'-events case.
- **`_OutputRetriever`**: synthetic `history_payload` fixtures (well-formed / missing-images / multi-image), a tiny real PNG fixture for Pillow dimension reads, mocked `view_image` — atomic-write-cleanup-on-failure asserted the same way Phase 1's `ArtifactWriter` tests presumably already assert it.
- **`_ComfyUIMetricsRecorder`**: field-mapping assertions against a spy/real `MetricsCollector` pointed at a tmp path.
- **`ComfyUIClient` (facade, integration-of-mocks)**: end-to-end `generate()` calls against the full scripted fake HTTP+WS pair, covering: happy path, OOM-then-retry-then-success, OOM-exhausted, queue-error (non-retried), timeout, output-missing, context-manager cleanup on exception, `cancel()` no-op-when-already-finished.
- **Retry/backoff assertions**: reuse the same style Module 2's test suite presumably already uses for its own Tenacity-wrapped function — assert attempt counts and that `_before_sleep_log` is invoked with a string, never an exception object (regression-guarding the exact pickling bug class already fixed once in Module 2).
- All of the above run in the **default** `pytest` invocation (no marker), keeping the project's fast/no-GPU suite fast — Phase 2 adds zero real network calls to the default suite, matching how Module 2's yt-dlp tests are presumably already mocked.

### 13.2 Integration tier — `@pytest.mark.gpu`, skipped by default

- A small number of tests that spin up (or expect already-running) a real local ComfyUI instance and drive one real `ComfyUIClient.generate()` call against `general.json` under `PROFILE_LOW_VRAM` (the cheapest/fastest profile), asserting a real image is retrieved end-to-end. This slots into the *existing* `gpu` marker already declared in `pytest.ini` (`markers = ... gpu: requires a local GPU and ComfyUI; skipped by default.`) — no `pytest.ini` change needed, Phase 2 simply adds tests under an already-provisioned marker.
- A real-WS-drop test (kill/restart ComfyUI mid-generation) is valuable but explicitly optional/manual — documented as a suggested nightly/manual check rather than a required CI-tier test, since reliably triggering a real mid-stream disconnect in an automated environment is brittle; the scripted-fake version in §13.1 already covers the *logic* deterministically.

### 13.3 Golden-fixture note

Phase 2 does not itself need golden-manifest regression tests (those assert final `ImageGenerationResult` fields — a Phase 3/4 concern, §22 of the architecture doc). It does, however, benefit from — and should reuse — Phase 1's existing `WorkflowLibrary` template-validation tests as a **precondition check**: every shipped template must have exactly one recognizable output/save node, or `_OutputRetriever`'s `ComfyUIOutputMissingError` will fire in production. Adding one assertion to the existing `tests/test_workflow_library.py` (or a small addition co-located with Phase 2's tests) — "every template's graph contains at least one node whose `class_type` matches the configured save-node convention" — catches this class of bug at the same fast, no-GPU tier rather than only at integration time.

---

## 14. Integration with Module 6 and Module 7 Phase 1

- **Module 6** (`prompt_compiler.py`) is untouched — Phase 2 never imports from it directly. Its output (`PromptPackage`, persisted under `data/prompt_packages/`) only reaches Phase 2 indirectly, already consumed by Phase 1's `PromptPackageLoader`/`WorkflowBuilder` before a `BuiltWorkflow` exists. Phase 2's `ComfyUIClient` has no knowledge of `PromptPackage` at all — its narrowest possible input surface is `BuiltWorkflow` + `GenerationProfile`, keeping the transport layer fully decoupled from prompt/creative concerns, per the architecture doc's own layering (§2).
- **Phase 1 components reused as-is, unmodified**: `WorkflowBuilder` (produces the `BuiltWorkflow` Phase 2 submits), `ProfileSelector` (Phase 2's `health_check()` finally gives it a real VRAM number to select against — no signature change), `MetricsCollector` (Phase 2 writes through it, doesn't replace it), `WorkflowLibrary` (unchanged; Phase 2 only consumes `WorkflowTemplateRef` values already resolved by it via Phase 1's pipeline), the full `module7_exceptions.py` hierarchy (extended additively, §5), and every relevant `config.py`/`models.py` constant (extended additively, §8/§4).
- **Concrete call shape once Phase 3 exists** (documented here so Phase 3 has a precise contract to build against, without this document specifying any Phase 3 *internal* logic):

  ```python
  # Illustrative shape only — not implementation, and not Phase 2's code to write.
  profile = ProfileSelector(...).select(available_vram_gb=client.health_check().vram_free_mb / 1024)
  workflow_ref = WorkflowLibrary(...).resolve(niche, profile)
  built = WorkflowBuilder().build(package, profile, workflow_ref, reference_assets)
  with ComfyUIClient() as client:
      raw_output = client.generate(built, candidate_index=0, stage_output_dir=staging_dir)
  # raw_output.image_path now feeds Phase 3's IdentityPreservationStage, etc.
  ```

- **`main.py` wiring**: out of scope for Phase 2 itself (the architecture doc's §5 pipeline only becomes fully wireable once Phase 3's stages exist to consume `RawGenerationOutput`), but this document's component contracts (§3, §4) are what Phase 3/`main.py` will import — nothing about them is expected to change shape once Phase 3 exists, only to be *called*.

---

## 15. Future compatibility

### 15.1 With Phase 3 (Image Generation — identity, restoration, background pass, upscale)

- `RawGenerationOutput.image_path` is exactly the input `IdentityPreservationStage` needs (a local file path to run InsightFace embedding comparison against). Phase 2 deliberately returns a **path**, not in-memory bytes, so Phase 3's stages (which the architecture doc describes as running CodeFormer/GFPGAN/ESRGAN as ComfyUI nodes *within* the same graph in later profile iterations, or as separate local CV calls) can each read/write the same file location without Phase 2 holding megabytes of image data in memory across a multi-stage pipeline.
- Multi-candidate generation (`num_candidates` ∈ {1,2,4,8}, §15 of the architecture doc) is achieved by Phase 3 calling `ComfyUIClient.generate()` multiple times with incrementing `candidate_index` (and, per §25's "generated sequentially" note, Phase 2's single-`MODULE7_MAX_CONCURRENT_GENERATIONS`-respecting design is already correct for this — no Phase 2 change needed when Phase 3 starts requesting more than one candidate).
- The identity-retry escalation described in architecture doc §5 step 7 ("resubmit with an incremented internal retry seed") is a Phase 3 concern that will call `WorkflowBuilder.build()` again with a modified seed and then `ComfyUIClient.generate()` again — Phase 2's per-call statelessness (aside from the reused connection) makes this trivial; there is no "generation session" state in `ComfyUIClient` that would need resetting between attempts.
- `VRAMExhaustedError`'s profile-fallback handling (§2.3) is explicitly designed as a Phase 3 responsibility calling back into Phase 1's `ProfileSelector` — Phase 2 needs no changes when that fallback logic is implemented.

### 15.2 With Phase 4 (Quality Assurance — scoring, ranking)

- `_ComfyUIMetricsRecorder`'s `GenerationMetrics` output already has `identity_failures_count`, `qa_failures_count`, `winning_overall_score`, `winning_signal_scores` fields reserved (Phase 1's model, unmodified) for Phase 4 to populate once scoring exists — Phase 2 leaves them at defaults, requiring no schema migration when Phase 4 starts filling them in for the *same* `GenerationMetrics` record Phase 2 already appended per attempt (Phase 4 would append its own follow-up record or, if a single-record-per-video design is preferred, that reconciliation is a Phase 4 design decision, not something this document needs to resolve now).
- `RawGenerationOutput.sha256` gives `QualityAssuranceStage`/`CandidateRanker` a stable audit trail from "what ComfyUI produced" through to "what was scored," satisfying the architecture doc §23's "every artifact traceable" principle even before the full hashing/provenance contract is finalized in a later phase.
- Nothing in Phase 2's public API (`ComfyUIClient.generate()`) needs to change shape to support ranking — ranking operates entirely on `QualityAssuranceReport` objects Phase 4 derives *from* Phase 2's outputs, never on Phase 2's internals directly.

---

## 16. Implementation checklist

A step-by-step checklist for the coding phase that follows this design. Ordered so each step is independently testable before the next begins, matching the project's incremental, test-as-you-go working style.

1. **Dependency.** Add `websocket-client>=1.7.0` to `requirements.txt` under a new `# ── Module 7 Phase 2 (ComfyUI Integration) ──` comment block, alongside a one-line note (mirroring the existing CUDA note style) that it requires no separate native install.
2. **Config.** Add the Phase 2 constants block from §8 to `modules/config.py`; update the existing `# Module 7 — Local Image Generation Engine (Phase 1 foundation only)` header comment to drop "foundation only" phrasing.
3. **Exceptions.** Add `ComfyUITimeoutError`, `ComfyUIOutputMissingError`, `ComfyUIResponseError` to `modules/module7_exceptions.py` (§5); optionally correct the "Reserved for Phase 2" docstrings on `IdentityPreservationError`/`QualityAssuranceError`/`NoEligibleCandidateError` to say Phase 3/4.
4. **Scaffold `modules/comfyui_client.py`.** Module docstring, imports, `_configure_logger()` (copy Phase 1's pattern exactly), the four dataclasses from §4 (`RawGenerationOutput`, `ComfyUIEvent`, `SystemStats`, plus internal `_CompletionOutcome`/`_AttemptOutcome`), and the private `_ComfyUIHTTPError` transport-local exception.
5. **`_ComfyUIHTTPTransport`.** Implement all seven methods (§3.2) against `requests.Session`. Write `tests/test_comfyui_client.py::TestHTTPTransport` first (mocked session), get it green.
6. **`_ComfyUIWebSocketTransport`.** Implement connect/`next_event`/close against `websocket-client`. Write the fake-WS-server test fixture and `TestWebSocketTransport` tests, get them green.
7. **`_QueueTracker`.** Implement the state machine (§3.4) against the two transports (fakeable via the fixtures from steps 5–6). Write `TestQueueTracker` covering every outcome in §11's table plus the WS-drop-to-polling path.
8. **`_OutputRetriever`.** Implement §3.5 including the atomic write and sha256 helper. Write `TestOutputRetriever` with the synthetic `history_payload` fixtures and a tiny real PNG.
9. **`_ComfyUIMetricsRecorder`.** Implement the field mapping from §10 against Phase 1's existing `MetricsCollector`/`GenerationMetrics`. Write `TestMetricsRecorder`.
10. **`_classify_comfyui_error` helper.** A small pure function (mirroring `youtube_metadata.py`'s `_classify_transient_error`) distinguishing OOM-shaped `execution_error` payloads from other execution errors, driven by documented ComfyUI error-message/exception-type signatures. Unit test with fixed sample payloads for both branches.
11. **`ComfyUIClient` facade.** Implement `__init__`, `health_check`, `generate` (wiring steps 5–10 together per the internal workflow in §3.1), `cancel`, `close`, context-manager methods. Apply the two Tenacity retry layers from §6 with the shared `_before_sleep_log`.
12. **Facade-level tests.** `TestComfyUIClient` end-to-end scenarios listed in §13.1, using the fakes built in steps 5–6.
13. **Golden precondition test.** Add the "every template has a recognizable save node" assertion (§13.3) to the existing workflow-library test coverage.
14. **`gpu`-marked integration test(s).** One real end-to-end `generate()` call against a real local ComfyUI + `general.json` + `PROFILE_LOW_VRAM`, marked `@pytest.mark.gpu`, run manually (`pytest -m gpu`) — not part of the default CI run, per the already-existing `pytest.ini` policy.
15. **Full regression run.** `pytest` (default markers, i.e. `-m "not integration and not gpu"`) — confirm all pre-existing 364 tests plus every new Phase 2 test pass, and that the `gpu`-marker deselection count grows by exactly the number of new integration tests added in step 14 (previously 1 deselected).
16. **Docs.** Update `docs/IMAGE_GENERATION_ARCHITECTURE.md`'s status line and this design doc's own header if any deviations were made during implementation (e.g., if `ComfyUIClient` was ultimately inlined into `image_generator.py` per §1.1's stated alternative) — keep the doc and the code from drifting apart, matching how the architecture doc itself already documents its own revision history (v0.9 → v1.0 note at the top of that file).
17. **No `main.py` changes in this phase** — confirm no edits were made there; wiring `ComfyUIClient` into the live per-video pipeline stage is explicitly deferred to Phase 3 per §14 of this document.

---

*End of design specification. No implementation code is included per the task's DESIGN ONLY constraint.*
