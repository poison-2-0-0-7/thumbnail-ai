# MODULE7_PHASE2_QUEUE_TRACKER_DESIGN.md

**Module 7 — Phase 2: ComfyUI Integration**
**Component: `_QueueTracker`**
**thumbnail-ai**

Status: **Design specification, v1.0. No implementation.**
Source of truth: `docs/MODULE7_PHASE2_COMFYUI_INTEGRATION_DESIGN.md` (§1–§3.4, §4, §5, §6, §7, §8, §9, §10, §11, §13 in particular), `docs/MODULE7_PHASE2_WEBSOCKET_TRANSPORT_DESIGN.md` (§2.3, §4, §5, §6, §8), and `modules/comfyui_client.py` as it exists today (Sprint 1 — `_ComfyUIHTTPTransport` and `_ComfyUIWebSocketTransport`, both complete, along with `SystemStats` and `ComfyUIEvent`), `modules/module7_exceptions.py`, `modules/config.py`.

This document narrows and elaborates the master design's §3.4 into an implementation-ready blueprint for `_QueueTracker` alone. It introduces no public interface the master document did not already reserve for this component, and changes nothing in `_ComfyUIHTTPTransport`, `_ComfyUIWebSocketTransport`, or `ComfyUIEvent`.

Assumed complete and unmodified by this document: `_ComfyUIHTTPTransport` and its seven methods, `_ComfyUIWebSocketTransport` and its five methods (`ensure_connected`, `receive`, `next_event`, `is_connected`, `close`), `ComfyUIEvent`, `SystemStats`, and the existing HTTP/WebSocket transport tests in `tests/test_comfyui_client.py`.

---

## 1. Purpose

### 1.1 Why this component exists

`_ComfyUIHTTPTransport.submit_prompt()` hands back only a `prompt_id`; it says nothing about when — or whether — that prompt will ever finish. `_ComfyUIWebSocketTransport.next_event()` hands back a typed stream of events, but a raw stream is not an answer: many events belong to *other* jobs on the same shared ComfyUI server, some events are purely informational, and the stream can go silent forever without ever announcing a failure. `_QueueTracker` exists to turn "a `prompt_id` and a live event stream" into exactly one of three outcomes — completed, errored, or timed out — plus the confirming data (`history` payload or `error_payload`) and timing (`queue_wait_seconds`, `generation_seconds`) that `ComfyUIClient.generate()`, `_OutputRetriever`, and `_ComfyUIMetricsRecorder` each need. It is the one place in Phase 2 that decides "is this job done yet," so nothing else in the module has to.

### 1.2 Responsibilities

- Own the single blocking call, `await_completion()`, that `ComfyUIClient.generate()` invokes once per submitted `prompt_id`.
- Interpret the closed set of `ComfyUIEvent` values arriving on `_ComfyUIWebSocketTransport` for the lifetime of one wait, filtering out events that belong to a different `prompt_id`/`client_id` on the shared server.
- Detect the queued → executing → terminal lifecycle from those events, and fall back to polling `_ComfyUIHTTPTransport.history()` when the WebSocket connection is unavailable, per §6.
- Measure `queue_wait_seconds` (submission to first sign of execution) and `generation_seconds` (first sign of execution to terminal signal) using a monotonic clock, and hand both back as part of its result.
- Enforce exactly one aggregate deadline (`execution_timeout_seconds`) across whichever combination of WebSocket-driven and HTTP-polling-driven waiting actually occurs during one `await_completion()` call.
- Fetch and return the confirming `history` payload on completion, so `_OutputRetriever` never has to talk to either transport itself.
- Translate every transport-level failure it observes (`ComfyUIConnectionError` from the WebSocket, `_ComfyUIHTTPError` from HTTP) into loop behavior (fall back, retry, give up) — never let either escape `await_completion()` as a raised exception.

### 1.3 Boundaries — what this component explicitly does not do

- It does not raise any `Module7Error` subclass. `await_completion()` always returns a typed result object (§3.2); `ComfyUIClient.generate()` alone decides which result maps to `VRAMExhaustedError`, `ComfyUIQueueError`, or `ComfyUITimeoutError` (master document §3.4's "Error propagation," §11). This keeps `_QueueTracker` a pure "what happened" component, testable without ever asserting on exception types.
- It does not classify *why* an `execution_error` happened (OOM vs. bad node params). It hands the raw `error_payload` dict upward unmodified; `_classify_comfyui_error` (a `ComfyUIClient`-level helper, master document §16 step 10) is the only code that inspects its contents for classification purposes.
- It does not retrieve output images or read `history_payload["outputs"]` beyond passing the whole payload through. That is `_OutputRetriever`'s job (master document §3.5) exclusively.
- It does not submit, resubmit, or cancel a prompt. Submission already happened before `await_completion()` is called; cancellation (`ComfyUIClient.cancel()`) is a separate, HTTP-only operation that never touches `_QueueTracker` (§7.5 below elaborates why this is safe given the codebase's single-threaded design).
- It does not open, close, or reconnect `_ComfyUIHTTPTransport`'s session — the HTTP transport has no connection-lifecycle state to manage (it is a thin `requests.Session` wrapper); `_QueueTracker` only ever calls `history()` on it.
- It does not decide the *initial* WebSocket connection state. `ComfyUIClient.generate()` step 1 already calls `ws.ensure_connected()` before `_QueueTracker` is ever invoked (master document §3.1); `_QueueTracker` only calls `ws.ensure_connected()` again, later, as part of its own reconnect-during-fallback behavior (§6.4).
- It does not implement a Tenacity-decorated retry. Every retry-shaped behavior inside `_QueueTracker` (§7.6) is a small, bounded, manually-coded loop local to `await_completion()`, distinct from the two Tenacity layers the master document reserves for `ComfyUIClient` itself (§6 of the master document, connection-level and OOM-level) — `_QueueTracker` is called *inside* the OOM-retry layer's protected region, not itself a retry boundary.

---

## 2. Architecture

### 2.1 Component placement

`_QueueTracker` lives in `modules/comfyui_client.py`, alongside `_ComfyUIHTTPTransport`, `_ComfyUIWebSocketTransport`, `ComfyUIEvent`, and `SystemStats`. Like both transports, it is module-internal — not listed in `__all__` — since nothing outside this module or its tests ever constructs one directly; only `ComfyUIClient` does.

### 2.2 Interaction with `_ComfyUIHTTPTransport`

`_QueueTracker` holds a reference to exactly one `_ComfyUIHTTPTransport` instance (constructor-injected, §3.1) and calls exactly one of its seven methods: `history(prompt_id)`. It never calls `submit_prompt`, `system_stats`, `view_image`, `interrupt`, `delete_from_queue`, or `queue_status` — those remain `ComfyUIClient`'s or `_OutputRetriever`'s exclusive concerns. `history()` is called in two distinct situations, both described in full in §5–§6: once as a **confirming call** immediately after a WebSocket completion signal, and repeatedly as the **polling primitive** during HTTP fallback.

### 2.3 Interaction with `_ComfyUIWebSocketTransport`

`_QueueTracker` holds a reference to exactly one `_ComfyUIWebSocketTransport` instance (constructor-injected) and drives it exclusively through `next_event(timeout_seconds)` and, during fallback recovery only, `ensure_connected()` (§6.4). Per the WebSocket transport document's §2.3, `_QueueTracker` is `_ComfyUIWebSocketTransport`'s only caller besides `ComfyUIClient`'s own initial-connect and `close()` calls. `_QueueTracker` never calls `ws.close()` — connection lifetime is scoped to `ComfyUIClient`, not to one `await_completion()` call, so the same socket keeps serving events across sequential candidates for one video.

### 2.4 Interaction with the future `_OutputRetriever`

Indirect only, through the result `await_completion()` returns. `_QueueTracker` fetches the confirming `history` payload (§5.4) and includes it, unmodified, in its result object; `ComfyUIClient.generate()` passes that same dict straight into `_OutputRetriever.fetch(prompt_id, history_payload, stage_output_dir, candidate_index)` (master document §3.5's signature). `_QueueTracker` never imports `_OutputRetriever` and has no opinion on what a valid `outputs` shape looks like — that validation is entirely `_OutputRetriever`'s (§3.5 step 2's `ComfyUIOutputMissingError`).

### 2.5 Interaction with the future `_ComfyUIMetricsRecorder`

None directly. `_QueueTracker` produces `queue_wait_seconds` and `generation_seconds` as plain floats inside its result object; `ComfyUIClient.generate()` reads them out of that result and forwards them into `_ComfyUIMetricsRecorder.record_attempt()`'s `outcome: _AttemptOutcome` argument (master document §3.6, §10's field-mapping table). `_ComfyUIMetricsRecorder` never references `_QueueTracker` and `_QueueTracker` never references `_ComfyUIMetricsRecorder` or `MetricsCollector` — this keeps the dependency chain the same strict, one-directional shape the WebSocket transport document established for its own siblings (`_ComfyUIWebSocketTransport → _QueueTracker → ComfyUIClient → _ComfyUIMetricsRecorder`).

### 2.6 Interaction with `ComfyUIClient`

`ComfyUIClient` is the only code outside this module (and its tests) that constructs a `_QueueTracker`, and it does so once per `ComfyUIClient` instance (not once per `generate()` call), passing in the same `_ComfyUIHTTPTransport`/`_ComfyUIWebSocketTransport` pair it owns for the lifetime of the `with ComfyUIClient(...)` block. Per the master document's §3.1 internal workflow, `generate()` step 3 is the sole call site: `outcome = self._queue_tracker.await_completion(prompt_id, self._client_id)`. `ComfyUIClient` owns steps 1 (connect), 2 (submit), and 4–7 (retrieve output, classify errors, record metrics); `_QueueTracker` owns only step 3.

### 2.7 Ownership and dependency direction

```
ComfyUIClient
    │  owns (constructor-injected, one instance each, lifetime = the `with` block)
    ├──► _ComfyUIHTTPTransport   ◄──┐
    ├──► _ComfyUIWebSocketTransport ◄┤  both handed to
    └──► _QueueTracker ─────────────┘  _QueueTracker.__init__
              │
              │  returns (per call, never stored)
              ▼
        _CompletionResult (§3.2)
              │
              │  consumed by
              ▼
   ComfyUIClient.generate() steps 4–7
   (→ _OutputRetriever, → _classify_comfyui_error, → _ComfyUIMetricsRecorder)
```

`_QueueTracker` depends on both transports; neither transport depends on `_QueueTracker` or on each other (WebSocket transport document §2.2). This is what lets all three internal collaborators be unit-tested in isolation, and lets `_QueueTracker`'s own tests fake both transports without needing a real socket or a real `requests.Session`.

---

## 3. Public API

All signatures below are final for this phase — nothing here is expected to change shape once implemented, per the master document's "additive only" convention. No implementation code; parameter/return/exception/threading/timeout behavior only.

### 3.1 `__init__(self, http: "_ComfyUIHTTPTransport", ws: "_ComfyUIWebSocketTransport", poll_interval_seconds: float, execution_timeout_seconds: float) -> None`

| | |
|---|---|
| **Parameters** | `http` — the `ComfyUIClient`-owned `_ComfyUIHTTPTransport` instance, reused as-is (§2.2). `ws` — the `ComfyUIClient`-owned `_ComfyUIWebSocketTransport` instance, reused as-is (§2.3). `poll_interval_seconds: float` — the delay between successive `history()` polls once HTTP fallback begins (sourced from `COMFYUI_POLL_INTERVAL_SECONDS`, §8). `execution_timeout_seconds: float` — the single aggregate deadline for one `await_completion()` call, spanning any mixture of WebSocket-driven and polling-driven waiting (sourced from `COMFYUI_EXECUTION_TIMEOUT_SECONDS`, §8). |
| **Returns** | `None`. Construction does no I/O — it does not call `ensure_connected()`, does not call `history()`, and does not start any clock. All timing begins inside `await_completion()` itself. |
| **Raises** | `ValueError` if `poll_interval_seconds <= 0` or `execution_timeout_seconds <= 0`, matching the fail-fast constructor validation style both transports already use. Additionally, `ValueError` if `poll_interval_seconds >= execution_timeout_seconds` — a poll interval at least as large as the whole budget can never produce more than one poll, which is never an intended configuration and is worth rejecting at construction time rather than silently degrading. |
| **Thread-safety** | Not thread-safe by design, matching the rest of `comfyui_client.py` (zero `asyncio` usage anywhere in `modules/`, per the master document §3.3). One `_QueueTracker` instance is owned by exactly one `ComfyUIClient` instance used from one thread. |
| **Timeout behavior** | N/A — no blocking I/O in the constructor. |

### 3.2 `await_completion(self, prompt_id: str, client_id: str) -> _CompletionResult`

The single public method. Blocks until a terminal outcome is reached or `execution_timeout_seconds` elapses — never longer, never shorter, and never indefinitely (§1.2).

| | |
|---|---|
| **Parameters** | `prompt_id: str` — the ComfyUI-assigned identifier returned by `_ComfyUIHTTPTransport.submit_prompt()`, used to filter incoming `ComfyUIEvent`s and to query `history()`. `client_id: str` — the same `uuid4`-derived identifier `ComfyUIClient` passed into `_ComfyUIWebSocketTransport`'s constructor; `_QueueTracker` does not use it to filter WebSocket frames directly (that filtering already happens implicitly — see the note in §4.4 below — via `prompt_id`, since ComfyUI's `executing`/`progress`/`execution_error`/`execution_cached` events already carry `prompt_id`, not `client_id`), but retains it for the one case where it matters: distinguishing "no `status` events with a nonzero `queue_remaining` have arrived at all" (this client's job may not be visible yet) from "a `status` event arrived but for a different client's submission" is not needed, since `status` events carry no `prompt_id` at all (§4.3) and are read purely for queue-length bookkeeping, never for identity — see §4.4. |
| **Returns** | `_CompletionResult` (§3.3, a new frozen dataclass this document introduces — not itself part of the master document's abbreviated `_CompletionOutcome` signature, but a strict elaboration of it: `_CompletionOutcome` becomes the `.outcome` field of a richer object carrying the timing and payload the rest of `ComfyUIClient.generate()` already needs, per the master document's own note in §4.4 that `_AttemptOutcome` is "a small dataclass bundling [`_CompletionOutcome`] with timing" — this document simply moves that bundling to where the timing is actually measured). |
| **Raises** | Never raises. Every failure mode this method can encounter (WebSocket disconnect, HTTP failure, malformed history, timeout) is folded into one of the three `_CompletionOutcome` values inside the returned `_CompletionResult` — see §7 for the exhaustive mapping. |
| **Thread-safety** | Not thread-safe; called exactly once per `generate()` invocation from the single owning thread. |
| **Timeout behavior** | Bounded by `execution_timeout_seconds`, measured via `time.monotonic()` from the first instant inside this method (not from when the prompt was submitted — `ComfyUIClient.generate()` step 2 already started its own `queue_wait` timer at submission time, per the master document §3.1; `_QueueTracker`'s own internal clock, described in §4.5, is what actually produces the `queue_wait_seconds`/`generation_seconds` values returned in `_CompletionResult`, so the two are the same measurement, not two competing ones). Every blocking call underneath — `ws.next_event(timeout_seconds)`, `http.history(prompt_id)` — has its own much smaller, independent timeout, so the outer loop re-checks the aggregate deadline at least once per `COMFYUI_WEBSOCKET_TIMEOUT_SECONDS` (5s default) during WebSocket-driven waiting, and at least once per `poll_interval_seconds` during HTTP-fallback waiting. Retry behavior: none at the Tenacity level (§1.3); the small bounded manual retries described in §7.6 are themselves subject to, and can never exceed, this same outer deadline. |

### 3.3 `_CompletionResult` (frozen dataclass, `comfyui_client.py`, internal — not exported)

```python
@dataclass(frozen=True)
class _CompletionResult:
    outcome: _CompletionOutcome          # COMPLETED | EXECUTION_ERROR | TIMEOUT
    history_payload: dict[str, Any] | None   # populated iff outcome is COMPLETED
    error_payload: dict[str, Any] | None     # populated iff outcome is EXECUTION_ERROR
    queue_wait_seconds: float             # 0.0 if the job was already executing on first observation
    generation_seconds: float             # time from first "executing" sign to the terminal signal
    used_http_fallback: bool              # True if HTTP polling was ever engaged during this call
```

`history_payload` and `error_payload` are mutually exclusive and both `None` when `outcome is _CompletionOutcome.TIMEOUT`. `used_http_fallback` exists purely for the WARNING/metrics visibility the master document's §9 logging policy and §10 `failure_reason` classification want (a completion reached only via polling is diagnostically different from one reached over a healthy socket, even though both map to the same `COMPLETED` outcome).

### 3.4 `_CompletionOutcome` (enum, `comfyui_client.py`, internal — not exported, reproduced from the master document §4.4)

```python
class _CompletionOutcome(Enum):
    COMPLETED = "completed"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
```

Exactly the three values the master document names in §3.4's public API comment. No fourth value (e.g., a distinct `CANCELLED`) is added — §4.6 explains why cancellation never needs its own outcome here.

### 3.5 Helper methods (private, not part of the public surface)

These are internal to `await_completion()`'s implementation; listed here only so the next Codex iteration has named seams to implement against, not because they are part of `_QueueTracker`'s contract with `ComfyUIClient`.

| Helper | Purpose |
|---|---|
| `_run_websocket_phase(self, prompt_id, deadline) -> _PhaseResult` | Drives the `ws.next_event()` loop until a terminal signal, a `ComfyUIConnectionError`, or `deadline` is reached. Returns a small internal sentinel indicating which of the three happened, never a `_CompletionResult` directly (that assembly happens once, in `await_completion()` itself, so timing fields are computed in exactly one place). |
| `_run_polling_phase(self, prompt_id, deadline) -> _PhaseResult` | Drives the `http.history()` polling loop (§6) until a terminal signal, a successful WebSocket reconnect handoff, or `deadline` is reached. |
| `_confirm_completion(self, prompt_id, deadline) -> dict[str, Any] | None` | The bounded-retry confirming-`history()`-call helper described in §5.4 and §7.4. |
| `_classify_history_status(self, history_entry) -> _CompletionOutcome | None` | Reads a ComfyUI history entry's `status` object (present whether the entry arrived via the confirming call or via polling) and maps it to `COMPLETED`, `EXECUTION_ERROR`, or `None` ("not terminal yet — keep waiting/polling"), per §5.5. |

None of these four helpers are directly tested through the public surface described in §10 test IDs, but the testing strategy is written so each is independently exercisable by constructing `_QueueTracker` with hand-scripted fakes and driving it through `await_completion()` — matching how the WebSocket transport document's tests exercise `receive()` versus `next_event()` as two layers of the same public call.

---

## 4. Queue State Machine

### 4.1 States

`_QueueTracker` tracks one internal state variable across a single `await_completion()` call — never persisted, never exposed publicly, reset fresh on every call:

| State | Meaning |
|---|---|
| **Queued** | Submitted; no `executing` event carrying our `prompt_id` (with any `node` value, including `null`) has been observed yet. The job may or may not have started sampling on ComfyUI's side already — `_QueueTracker` only knows what the event stream (or history) has told it so far. |
| **Waiting** | An alias state used only in the state-machine narrative below to describe time spent inside `ws.next_event()`/polling with nothing new to report; it is not a separate value of the tracked variable — both "still Queued, waiting for the first sign of execution" and "Executing, waiting for the next progress/completion event" are "waiting" in the colloquial sense, but the tracked state itself is always exactly one of the other rows in this table. |
| **Executing** | At least one `executing` event with a non-null `node` for our `prompt_id` has been observed (or, in polling mode, `history()`'s status object indicates the prompt has started/is running). This is the instant `generation_seconds` timing begins (§4.5). |
| **Completed** *(terminal)* | An `executing` event with `node: null` for our `prompt_id` was observed (§5.2) **and** confirmed by a subsequent `history()` call returning a payload whose status is not an error (§5.4–§5.5); or, in polling mode, `history()` itself directly reports a non-error terminal status. |
| **Execution Error** *(terminal)* | An `execution_error` event for our `prompt_id` was observed (§5.3); or, in polling mode, `history()`'s status object reports an error terminal status. |
| **Timed Out** *(terminal)* | `execution_timeout_seconds` elapsed with no terminal signal observed, from either transport, in any combination of phases. |

### 4.2 Valid transitions

```
Queued ──(status event, queue_remaining bookkeeping only)──► Queued
Queued ──(executing, node != null, our prompt_id)──────────► Executing
Queued ──(executing, node == null, our prompt_id)──────────► Completed   [rare but valid — see note below]
Queued ──(execution_error, our prompt_id)───────────────────► Execution Error
Executing ──(progress / execution_cached, our prompt_id)───► Executing   (no state change, timing/logging only)
Executing ──(executing, node == null, our prompt_id)────────► Completed
Executing ──(execution_error, our prompt_id)────────────────► Execution Error
{Queued, Executing} ──(execution_timeout_seconds elapses)───► Timed Out
```

**Note on `Queued → Completed` directly:** ComfyUI is not guaranteed to emit a discrete `executing` event with a non-null `node` for every graph — a fully-cached prompt (every node already satisfied by ComfyUI's node cache, signaled by `execution_cached`) can go straight from "queued" to the `executing(node=null)` completion signal with no intervening per-node `executing` event ever naming a real node. `_QueueTracker` treats this as a valid, if unusual, direct transition: `queue_wait_seconds` is measured up to the completion signal itself in this case, and `generation_seconds` is `0.0` (§4.5), which is a correct and informative metric value (an essentially-free cached regeneration), not an error condition.

### 4.3 Invalid transitions

- **Completed → anything.** Once a terminal state is reached, `await_completion()` returns immediately; no further events are read (§4.6's "first terminal signal wins" rule). There is no code path back out of a terminal state within one call.
- **Execution Error → Completed** (or vice versa) for the same `prompt_id` within one call. ComfyUI does not emit both signals for one prompt in practice, but if a malformed/duplicate frame ever produced both in sequence (e.g., a replayed `execution_error` after the queue tracker already saw `executing(node=null)`), the *first* one observed is authoritative and the loop has already exited — the second can never be evaluated because `_QueueTracker` stops reading once terminal. This is a consequence of the "first terminal signal wins" rule, not a separate guard that needs its own conditional.
- **Timed Out → anything.** `Timed Out` is only ever produced by the outer deadline check between reads, at the point `await_completion()` is about to return — there is no event that can arrive "after" a `Timed Out` result has been constructed and returned, since the method has already returned control to `ComfyUIClient`.
- **Executing → Queued.** ComfyUI does not un-queue a job that has begun executing; `_QueueTracker` never reverts `Executing` back to `Queued` on any event, including a `status` event reporting `queue_remaining: 0` after execution has already started (that combination is expected — the job is no longer *in* the queue precisely because it is executing — and is simply further confirmation, logged at DEBUG, not a state change).

### 4.4 Ignoring other jobs' events

ComfyUI's WebSocket endpoint is shared across every client currently connected and every job currently queued or running on that server (WebSocket transport document §4.1). Every `ComfyUIEvent` this component acts on except `status` carries a `prompt_id` field (§4.2 of the master document's dataclass). The filter rule is simple and applied uniformly:

- `executing`, `progress`, `execution_error`, `execution_cached` events whose `event.prompt_id != prompt_id` (the argument `await_completion()` was called with) are ignored — logged at DEBUG with the foreign `prompt_id` for diagnosability, but otherwise treated exactly like a read that timed out (loop again, no state change, no timing side-effect).
- `status` events carry no `prompt_id` at all (they describe ComfyUI's global queue length, not any one job) and are never ignored on identity grounds — they are always read for queue-length bookkeeping (§4.1's `Queued` row), since a `status` event's `queue_remaining` is meaningful background information regardless of whose jobs make up that count. They never drive a state transition by themselves.

### 4.5 Timing measurement

Both timing fields in `_CompletionResult` are measured with `time.monotonic()`, matching the master document §3.4 step 1's explicit rationale ("immune to system clock adjustments"):

- `queue_wait_seconds` = elapsed time from the instant `await_completion()` begins its loop to the instant the state transitions `Queued → Executing`, or to the instant a direct `Queued → Completed`/`Queued → Execution Error` transition occurs (§4.2's note) — in the latter case the value still reflects real elapsed queue time, just with no separate execution phase following it.
- `generation_seconds` = elapsed time from the `Queued → Executing` transition (or `0.0` if that transition never occurred, per §4.2's note) to the terminal transition (`Completed` or `Execution Error`). Left at `0.0`, not measured at all, if the outcome is `TIMEOUT` and execution never began; equal to "time spent executing before the timeout fired" if the outcome is `TIMEOUT` and execution *had* begun (a partial, informative value — a timeout during active generation is diagnostically different from one during queue wait, and this value is what lets `_ComfyUIMetricsRecorder`'s `failure_reason: "timeout"` records still carry a meaningful `generation_time_seconds` entry).
- Both fields are measured **once**, using the single clock instance started at the top of `await_completion()` — there is no separate clock for the WebSocket phase versus the polling phase; a mid-call fallback from WebSocket to HTTP polling (§6) does not reset or restart timing, since it is the same wait for the same job continuing under a different observation mechanism.

### 4.6 Cancellation handling

`_QueueTracker` has no `Cancelled` terminal state, deliberately, for a structural reason specific to this codebase: `ComfyUIClient.cancel(prompt_id)` operates entirely over HTTP (`POST /interrupt` or `POST /queue {"delete": [...]}`, master document §3.1) and is a method on the single-threaded, synchronous `ComfyUIClient` facade. Because nothing in this codebase uses threads or `asyncio` (master document §3.3), `cancel()` can only physically be called by the same thread that is currently blocked inside `await_completion()` — meaning it can never be called *during* one `await_completion()` call, only *before* one starts (a no-op, prompt not yet submitted) or *after* one already returned (§7.5 elaborates the one real use: `ComfyUIClient.generate()` step 6 calling `cancel()` best-effort immediately after `await_completion()` itself already returned `TIMEOUT`). Consequently:

- A cancellation issued between two `generate()` calls (e.g., Phase 3 abandoning a candidate) has no effect on any `_QueueTracker` instance, because no `await_completion()` call is in flight for that `prompt_id` to observe it.
- The only way a cancellation's *effect* becomes visible to `_QueueTracker` at all is if `cancel()` was called for a **different**, earlier `prompt_id` whose job was still technically live on the ComfyUI server (rare, but possible if a prior `generate()` call's `TIMEOUT`-triggered `cancel()` raced with ComfyUI's own processing) — in that case, per the WebSocket transport document §6.6, ComfyUI eventually emits either an `execution_error` (interrupted) or simply stops emitting further `executing`/`progress` events for that old `prompt_id`. Since `_QueueTracker` always filters on the *current* `prompt_id` (§4.4), an old, cancelled job's events are already being ignored as "a different job's events" regardless of *why* they stopped or turned into an error — no special-case branch is needed.
- If the task's illustrative state diagram's `Cancelled` state is ever wanted as a first-class `_CompletionOutcome` in a later phase (e.g., a future concurrent/cancellable redesign), that is out of scope here — it would require `ComfyUIClient`/`_QueueTracker` to support cross-thread signaling, which is an architecture change this document does not make.

---

## 5. Completion Detection

### 5.1 Overview

Two independent signals can indicate completion or failure, and `_QueueTracker` always prefers the WebSocket signal when available, using HTTP `history()` only to *confirm* it (§5.4) or, during fallback, as the *sole* signal (§6).

### 5.2 `executing(node=None)` — the WebSocket completion signal

Per ComfyUI's own protocol (WebSocket transport document §4.1), an `executing` event whose `node` field is `null` — parsed by `_ComfyUIWebSocketTransport._parse_executing()` into a `ComfyUIEvent(event_type="executing", node=None, prompt_id=..., ...)` — means the *entire* prompt has finished executing, not that one particular node ran with no name. `_QueueTracker` treats `event.node is None and event.prompt_id == prompt_id` as the completion signal: it breaks out of the WebSocket read loop immediately (§4.3's "first terminal signal wins") and proceeds to §5.4's confirming call. It does **not** yet know, from this event alone, whether the prompt succeeded or partially failed — ComfyUI emits this same `executing(node=null)` shape both for a fully successful run and, in some ComfyUI versions, immediately after an `execution_error` for the same prompt (the queue always advances). This is exactly why a confirming `history()` call is mandatory rather than optional (§5.4): the WebSocket event alone does not carry a pass/fail verdict, only "no longer running."

### 5.3 `execution_error` — the WebSocket failure signal

An `execution_error` event for our `prompt_id` is unambiguous and requires no confirmation: `_QueueTracker` breaks the loop immediately with outcome `EXECUTION_ERROR`, using `event.error_payload` (the full, unmodified dict `_ComfyUIWebSocketTransport._parse_execution_error()` already captured) as `_CompletionResult.error_payload`. No `history()` call is made in this path — the event itself is ComfyUI's authoritative statement that this node raised, and nothing a subsequent history read could add would change that verdict (history would, at most, additionally list *which* earlier nodes had already produced output before the failing node ran, which is not information `_OutputRetriever` or `_classify_comfyui_error` need).

### 5.4 The confirming `history()` call

Immediately after breaking out of the WebSocket loop on an `executing(node=null)` signal (§5.2), `_QueueTracker` calls `http.history(prompt_id)` once (subject to the small bounded retry in §7.4, not the polling cadence of §6) and inspects the returned dict's `status` object (§5.5) to produce the final verdict:

- If the confirming call's history status indicates success → `COMPLETED`, with `history_payload` set to the returned dict.
- If the confirming call's history status indicates an error → `EXECUTION_ERROR`, with `error_payload` synthesized from the history entry's own error fields (ComfyUI's history `status.messages` array contains the same `execution_error`-shaped node/exception data the WebSocket event would have carried — this is the path that catches the "WS said `executing(node=null)` but the job actually failed" case §5.2 flags). This is the **only** situation in which `error_payload` is derived from `history()` rather than directly from a `ComfyUIEvent`; it is documented here explicitly so the next Codex iteration does not need to guess at the shape — the history entry's `status.messages` list, when non-empty and containing an entry whose first element is `"execution_error"`, has the same second-element dict shape as a WebSocket `execution_error` event's `data`.
- If `history(prompt_id)` returns `None` (per `_ComfyUIHTTPTransport.history()`'s documented contract, this means ComfyUI has no history entry for this `prompt_id` yet — §5.6 covers this as "missing history").

### 5.5 `_classify_history_status()` — the status-object mapping

A ComfyUI history entry's `status` object has (per ComfyUI's own history schema) a `status_str` field taking values including `"success"` and `"error"`, and a `completed` boolean. `_QueueTracker`'s classification helper (§3.5) maps:

| `status.completed` | `status.status_str` | Classification |
|---|---|---|
| `true` | `"success"` | `COMPLETED` |
| `true` | `"error"` | `EXECUTION_ERROR` |
| `false` | any | Not terminal — the entry exists but the job is still recorded as in-progress from ComfyUI's own perspective (this can legitimately happen for a brief window between the `executing(node=null)` WS event and ComfyUI finalizing its own history write; handled by the bounded retry in §7.4, not treated as an inconsistency on the first observation). |
| Missing/malformed `status` object entirely | — | Treated identically to "history missing" (§5.6) — a structurally-invalid history payload from an otherwise-200 response is exactly the kind of case `ComfyUIResponseError` exists for at the `ComfyUIClient` layer (master document §5), but `_QueueTracker` itself does not raise that; it simply cannot classify this call's result as terminal and proceeds per §5.6/§7.4's retry-then-give-up rule. |

### 5.6 Handling missing history

"Missing history" means `http.history(prompt_id)` returned `None`, or returned a dict whose `status` object is absent/malformed (§5.5's last row) at a moment `_QueueTracker` needed a terminal verdict (either the confirming call after a WS completion signal, or a poll during HTTP fallback). This is never immediately fatal:

- During the **confirming call** (§5.4): retried a small, fixed number of times with a short fixed delay (§7.4) — ComfyUI's history write can lag its WebSocket `executing(node=null)` push by a small margin. If still missing after the bounded retries, `_QueueTracker` falls back to polling mode (§6) for the remainder of the overall budget rather than immediately declaring `TIMEOUT` — the job's actual state is still unknown, not necessarily bad, and polling may yet resolve it before the deadline.
- During **HTTP-fallback polling** (§6): missing history is the expected, common case for every poll before the job actually finishes — it is not an error at all, simply "not done yet," and the loop continues to the next scheduled poll.

### 5.7 Handling `execution_cached`

An `execution_cached` event for our `prompt_id` is purely informational (WebSocket transport document §4.1: "some nodes were served from cache rather than recomputed"). `_QueueTracker` logs it at DEBUG and continues the loop with no state transition — it is neither a completion nor a failure signal, and specifically must not be mistaken for `execution_error` (the two event names are visually similar; this is called out explicitly to guard against that class of implementation bug).

---

## 6. HTTP Fallback

### 6.1 When fallback begins

The instant `ws.next_event(COMFYUI_WEBSOCKET_TIMEOUT_SECONDS)` raises `ComfyUIConnectionError` (WebSocket transport document §6.4: the connection was actually closed, not merely quiet). `_QueueTracker` catches this exception at the call site inside `_run_websocket_phase()` — it is the only exception type that method's inner `ws.next_event()` call can raise, per that transport's own contract (§3.4 of the WebSocket transport document) — logs one WARNING (§9), and transitions from the WebSocket-driven phase to the polling phase without returning control to `ComfyUIClient`; this is entirely transparent to the caller (master document §11's error-handling table: "not an exception — falls back to polling").

### 6.2 How polling works

Once in the polling phase, `_QueueTracker` repeats, until a terminal classification (§5.5) or the overall deadline:

1. Call `http.history(prompt_id)`.
2. If it returns `None` or a non-terminal status (§5.5, §5.6) → this poll found nothing new; continue to step 3.
3. Sleep for `poll_interval_seconds`, then repeat from step 1 — unless a reconnect attempt is due (§6.4), in which case the reconnect attempt happens during this same sleep window, not as an additional delay.
4. If `history()` classifies as terminal → break with that outcome, using the returned dict directly as `history_payload` (`COMPLETED`) or deriving `error_payload` from it exactly as in §5.4 (`EXECUTION_ERROR`) — no separate confirming call is needed in polling mode, since the polling call *is* the confirming call.

An `_ComfyUIHTTPError` raised by `http.history()` itself (a transient HTTP-layer failure — connection refused, timeout, malformed JSON, per `_ComfyUIHTTPTransport`'s existing contract) during any single poll is caught, logged at WARNING, and does **not** end the polling loop — it is treated as "no news this poll," identical to a `None` return, and the loop continues on the same `poll_interval_seconds` cadence. This is deliberate: a polling loop that gives up after one transient HTTP blip would be strictly worse than the WebSocket path it is standing in for, which already tolerates arbitrary quiet periods.

### 6.3 Polling interval

`poll_interval_seconds`, injected at construction (§3.1) from `COMFYUI_POLL_INTERVAL_SECONDS` (default `3.0`, master document §8). Not adaptive/backoff-based — a flat interval, matching the master document's own characterization of this as "polling," not a retry policy (§1.3's boundary on Tenacity).

### 6.4 Reconnect attempts

Every `COMFYUI_WS_RECONNECT_POLL_CYCLES` polls (new constant, §8; default `3`, meaning roughly every third poll — about every 9 seconds at the default 3-second interval), `_QueueTracker` makes one best-effort call to `ws.ensure_connected()` **inside** the polling loop's own sleep window (i.e., this does not add extra wall-clock time beyond the normal `poll_interval_seconds` cadence — the reconnect attempt itself is bounded by `_ComfyUIWebSocketTransport`'s own `connect_timeout_seconds`, but that is `COMFYUI_STARTUP_TIMEOUT_SECONDS`, 60s by default, which is far larger than a poll interval; in practice a reconnect either succeeds fast — a network blip resolved — or fails fast — ComfyUI still down — so this is not expected to meaningfully stall polling, but the design does not assume that: see the note below):

- If `ensure_connected()` raises `ComfyUIConnectionError` (still unreachable): caught, logged at DEBUG (not WARNING — a still-failing reconnect during an already-WARNING-logged fallback is expected background noise, not new information), and polling continues unaffected.
- If `ensure_connected()` succeeds: `_QueueTracker` does **not** immediately resume `next_event()`-driven waiting on the next loop iteration. It first makes one immediate, out-of-cadence `http.history(prompt_id)` check (distinct from the regular poll schedule) — this exists specifically because any `ComfyUIEvent`s ComfyUI pushed *during* the outage were never delivered and never will be (WebSocket transport document §1.2: this is a live push stream, not a durable queue); if the job actually completed or failed entirely while the socket was down, only `history()` can reveal that, and checking immediately on reconnect avoids waiting out a full extra `poll_interval_seconds` for information already available. If that immediate check is terminal, `_QueueTracker` returns the classified outcome exactly as in §6.2 step 4. If it is not terminal, `_QueueTracker` resumes the WebSocket-driven phase (`_run_websocket_phase()`) from this point forward — polling stops, and DEBUG-logs the mode switch back.
- A note on bounding the reconnect attempt: because `ensure_connected()`'s own timeout (`COMFYUI_STARTUP_TIMEOUT_SECONDS`, 60s) could in principle exceed the remaining overall `execution_timeout_seconds` budget (300s default, so this is a real if narrow possibility late in a wait), `_QueueTracker` only *attempts* a reconnect if the remaining budget (deadline minus now) is greater than a fixed minimum guard (`COMFYUI_WS_RECONNECT_MIN_BUDGET_SECONDS`, new constant, §8, default `10.0`) — skipping the attempt entirely and logging at DEBUG otherwise, so a reconnect attempt can never itself be the reason the overall deadline is blown past by a large margin.

### 6.5 When WebSocket resumes

Exactly as described in §6.4's second bullet: immediately after a successful `ensure_connected()` call *and* one immediate non-terminal `history()` check. There is no separate "resume" method — this is simply `_run_websocket_phase()` being invoked again from inside the same `await_completion()` loop, with the same `prompt_id` and the same remaining deadline it was already tracking.

### 6.6 When fallback stops

- **Success**: a poll (or the immediate post-reconnect check) classifies as terminal (§6.2 step 4, §6.4). `await_completion()` returns.
- **Resumed WebSocket**: covered by §6.4/§6.5 — fallback stops the moment the socket reconnects and the immediate check is non-terminal; waiting continues, but no longer via polling.
- **Deadline**: `execution_timeout_seconds` elapses while still polling (or between polls). `await_completion()` returns `TIMEOUT`, with `generation_seconds` reflecting whatever partial execution time had already been measured before the WebSocket dropped (§4.5) — the fallback to polling does not reset that clock, and `used_http_fallback=True` is set on the returned result regardless of the outcome, since it accurately describes what happened during the call.

---

## 7. Error Handling

### 7.1 WebSocket disconnect

Not an error from `_QueueTracker`'s caller's perspective — the sole effect is the mode switch described in §6.1. Internally, this is the one and only exception `_run_websocket_phase()` is expected to catch (`ComfyUIConnectionError` from `ws.next_event()`); any other exception type escaping that call would indicate a bug in `_ComfyUIWebSocketTransport` (which is out of scope for this component to guard against defensively — its contract already promises no other exception type, per the WebSocket transport document §3.4).

### 7.2 HTTP failures

Two distinct call sites, two distinct handling rules, already described above and repeated here for completeness as a single reference table:

| Call site | On `_ComfyUIHTTPError` |
|---|---|
| Confirming call after WS completion signal (§5.4) | Bounded retry (§7.4); on exhaustion, fall back to polling (§5.6) rather than giving up outright. |
| Regular poll during HTTP fallback (§6.2) | Logged at WARNING, treated as "no news this poll," loop continues on schedule — never ends the polling loop by itself. |

In neither case does `_QueueTracker` re-raise `_ComfyUIHTTPError` or any `Module7Error` — both are absorbed into loop control flow, consistent with §1.3.

### 7.3 Timeout

The overall `execution_timeout_seconds` budget is the only timeout `_QueueTracker` itself enforces end-to-end; it is checked between every read/poll (never mid-read — `ws.next_event()`'s own per-call timeout and `http.history()`'s own request timeout each bound the individual call, and the outer check happens immediately after each one returns, whether with data or `None`). When the deadline is reached, `_QueueTracker` returns `TIMEOUT` — it does not itself call `ComfyUIClient.cancel()` or `http.interrupt()`; that best-effort cancellation is explicitly `ComfyUIClient.generate()`'s step 6 responsibility (master document §3.1), one layer up, kept there so `_QueueTracker` never needs a reference to the cancel-capable parts of `_ComfyUIHTTPTransport` it otherwise has no reason to touch (§2.2).

### 7.4 Bounded confirming-call retry

A small, fixed, non-Tenacity retry local to `_confirm_completion()` (§3.5): up to `COMFYUI_HISTORY_CONFIRMATION_RETRY_ATTEMPTS` attempts (new constant, §8, default `3`), each separated by a fixed `COMFYUI_HISTORY_CONFIRMATION_RETRY_DELAY_SECONDS` (new constant, §8, default `0.5`) — deliberately much shorter than `poll_interval_seconds`, since this retry exists only to absorb ComfyUI's own history-write lag immediately after a WS completion signal (§5.6), not to wait out a real outage (a real outage during this window is instead handled by falling back to the *polling* cadence once these few fast retries are exhausted, per §5.6). This retry is always subject to the same overall `execution_timeout_seconds` deadline as everything else — if the deadline is closer than the full retry budget would take, the number of attempts actually made is however many fit, not the full configured count.

### 7.5 Cancellation

As established in §4.6, `_QueueTracker` performs no cancellation handling of its own; it has no code path that can observe a `cancel()` call happening concurrently, by construction. The one place cancellation and `_QueueTracker` interact at all is temporal, not code-level: `ComfyUIClient.generate()` calls `self._http.interrupt()`-backed `cancel()` (or the queue-delete variant) only *after* `_QueueTracker.await_completion()` has already returned `TIMEOUT` — at that point `_QueueTracker`'s job for this `prompt_id` is already finished, and the resulting interruption is something a *future* `await_completion()` call (for a different `prompt_id`) might incidentally observe as "some other job's `execution_error`," already covered and correctly ignored by §4.4's filtering.

### 7.6 Malformed events

Never `_QueueTracker`'s concern at the frame-parsing level — `_ComfyUIWebSocketTransport.next_event()` already guarantees that any malformed/unparseable frame becomes a `None` return, never a partial or untrustworthy `ComfyUIEvent` (WebSocket transport document §3.4, §6.3). `_QueueTracker` therefore never needs to validate a `ComfyUIEvent`'s internal field consistency — if `next_event()` returned a non-`None` value, every field on it is already known-valid for that `event_type`. The only "malformed" case `_QueueTracker` itself must handle is a **malformed history payload** (§5.5's last row, §5.6) — a 200-status HTTP response whose JSON body doesn't have the expected `status` shape — which it treats identically to a missing history entry, never as a fatal error.

### 7.7 Inconsistent state

The one inconsistency `_QueueTracker` explicitly guards against is exactly the one §5.2 calls out: a WS `executing(node=null)` signal that history subsequently reveals was actually a failure. This is not a bug or an edge case to special-case defensively — it is the documented, expected reason the confirming call exists at all (§5.4), and the classification in §5.5 already produces the correct outcome (`EXECUTION_ERROR`) without any additional "inconsistency detected" branch. No other inconsistent-state scenario is possible given §4.3's invalid-transition guarantees (a terminal state, once reached, always ends the loop immediately, so there is no window in which two contradictory terminal signals could both be "current").

### 7.8 Retries — summary table

| What | Mechanism | Bound |
|---|---|---|
| WS connect (initial, before `_QueueTracker` is ever called) | Tenacity, one layer up in `ComfyUIClient` | `COMFYUI_CONNECT_RETRY_ATTEMPTS` (master document §6) |
| WS reconnect during fallback | Single best-effort attempt per reconnect cycle, no retry-of-the-retry | `COMFYUI_WS_RECONNECT_POLL_CYCLES` cadence, guarded by `COMFYUI_WS_RECONNECT_MIN_BUDGET_SECONDS` (§6.4) |
| Confirming `history()` call after WS completion | Small fixed manual retry inside `_QueueTracker` | `COMFYUI_HISTORY_CONFIRMATION_RETRY_ATTEMPTS` / `..._DELAY_SECONDS` (§7.4) |
| Polling `history()` calls | Not a retry at all — a poll that errors or finds nothing simply becomes the next scheduled poll | `poll_interval_seconds` cadence, bounded overall by `execution_timeout_seconds` |
| Whole `generate()` call on `VRAMExhaustedError` | Tenacity, one layer up in `ComfyUIClient`, wraps `_QueueTracker` entirely | `MODULE7_COMFYUI_OOM_RETRY_ATTEMPTS` (master document §6) |

---

## 8. Configuration

All constants below are proposed additions to `modules/config.py`'s `# Module 7 — Local Image Generation Engine` block. The first four were already named by the master document's §8 but have not yet been added to `config.py` in Sprint 1 (verified: `COMFYUI_EXECUTION_TIMEOUT_SECONDS`, `COMFYUI_POLL_INTERVAL_SECONDS`, `MODULE7_STILL_QUEUED_WARNING_SECONDS`, and `MODULE7_PROGRESS_LOG_GRANULARITY_PERCENT` are absent from the current file) — this component is what first requires them, so their addition belongs to this implementation step. The remaining three are new, introduced only because §6.4/§7.4 above need a bounded value that no existing constant already covers.

```python
# --- Phase 2: queue tracking (already named by the master design, added here) ---

COMFYUI_EXECUTION_TIMEOUT_SECONDS: float = 300.0
COMFYUI_POLL_INTERVAL_SECONDS: float = 3.0
MODULE7_STILL_QUEUED_WARNING_SECONDS: float = 30.0
MODULE7_PROGRESS_LOG_GRANULARITY_PERCENT: int = 25

# --- Phase 2: queue tracking (new, introduced by this document) ---

COMFYUI_WS_RECONNECT_POLL_CYCLES: int = 3
COMFYUI_WS_RECONNECT_MIN_BUDGET_SECONDS: float = 10.0
COMFYUI_HISTORY_CONFIRMATION_RETRY_ATTEMPTS: int = 3
COMFYUI_HISTORY_CONFIRMATION_RETRY_DELAY_SECONDS: float = 0.5
```

| Constant | Used for |
|---|---|
| `COMFYUI_EXECUTION_TIMEOUT_SECONDS` | `execution_timeout_seconds` constructor argument (§3.1) — the single aggregate deadline. |
| `COMFYUI_POLL_INTERVAL_SECONDS` | `poll_interval_seconds` constructor argument (§3.1, §6.3). |
| `MODULE7_STILL_QUEUED_WARNING_SECONDS` | Threshold for the "still queued after N seconds" WARNING (§9) — read by `_QueueTracker`, not passed through the constructor, since it affects only logging, not control flow. |
| `MODULE7_PROGRESS_LOG_GRANULARITY_PERCENT` | Threshold for INFO-level progress milestone logging (§9) — every N% of a `progress` event's `value/max`, read directly, not constructor-injected, for the same reason as above. |
| `COMFYUI_WS_RECONNECT_POLL_CYCLES` | Cadence of best-effort WS reconnect attempts during HTTP fallback (§6.4). |
| `COMFYUI_WS_RECONNECT_MIN_BUDGET_SECONDS` | Guard preventing a reconnect attempt from being made too close to the overall deadline (§6.4). |
| `COMFYUI_HISTORY_CONFIRMATION_RETRY_ATTEMPTS` / `..._DELAY_SECONDS` | The bounded manual retry around the post-completion-signal confirming `history()` call (§5.6, §7.4). |

`COMFYUI_WEBSOCKET_TIMEOUT_SECONDS` (already present, 5.0 default) is reused as-is for every `ws.next_event()` call `_QueueTracker` makes — no new constant is introduced for that purpose, and no existing constant's value or meaning is changed by this document.

---

## 9. Logging

Sink, rotation, and format are unchanged from every other Module 7 component — `_QueueTracker` calls only `logger.<level>(...)`; `_configure_logger()` is already invoked once at `comfyui_client.py` import time (present in the file today).

| Level | Logged for |
|---|---|
| **INFO** | Phase transitions: `Queued → Executing` (includes `queue_wait_seconds` so far), `→ Completed` (includes `generation_seconds`, `used_http_fallback`), `→ Execution Error` (includes `generation_seconds`, `used_http_fallback`, never the raw `error_payload` dict — see below). Progress milestones at `MODULE7_PROGRESS_LOG_GRANULARITY_PERCENT` boundaries (e.g., a `progress` event crossing the 25/50/75/100% mark of `progress_max`; every intervening frame is DEBUG only, not INFO, per the master document §3.4's own logging-behavior note — "not every single `progress` frame"). |
| **WARNING** | WS-drop-to-polling fallback beginning (§6.1); "still queued after `MODULE7_STILL_QUEUED_WARNING_SECONDS` seconds" (logged once per crossing, not repeated every loop iteration); a poll's `_ComfyUIHTTPError` (§7.2); confirming-call retry exhaustion before falling back to polling (§5.6, §7.4); a WS `executing(node=null)` signal that the confirming history call subsequently classified as `EXECUTION_ERROR` (§7.7 — this specific combination is worth a WARNING beyond the normal `Execution Error` INFO line, since it indicates the WS signal alone would have been misleading). |
| **ERROR** | None raised directly by `_QueueTracker` itself (§1.3 — it never raises); no ERROR-level logging is specified for this component beyond what `ComfyUIClient` logs once it receives a terminal `_CompletionResult` and decides what `Module7Error` subclass, if any, to construct and raise. |
| **DEBUG** | Every ignored foreign-`prompt_id` event (§4.4); every `execution_cached` event (§5.7); every reconnect attempt, successful or not, during fallback (§6.4); every poll's outcome (`None` history / non-terminal status / terminal status); the WS-resumed mode switch (§6.5); every sub-25%-granularity `progress` frame. |

**What is never logged, at any level:** the raw `error_payload` dict's full contents at INFO or WARNING — only `_classify_comfyui_error` (a `ComfyUIClient`-level concern) reads it for classification and, per the master document §9, logs only the extracted string message if it logs anything about it at all. `_QueueTracker` itself never inspects `error_payload`'s contents beyond passing it through, so it has nothing to log about it beyond "an execution error occurred for `prompt_id`." Full `history_payload` contents are never logged either — only its derived `queue_wait_seconds`/`generation_seconds`/`used_http_fallback` summary values.

---

## 10. Testing Strategy

All tests below run in the default `pytest` invocation (no marker) — zero real network calls, zero real ComfyUI process, matching the master document §13.1's stated policy. Location: `tests/test_comfyui_client.py::TestQueueTracker`, driven by fully scripted fake `_ComfyUIHTTPTransport`/`_ComfyUIWebSocketTransport` doubles (hand-rolled test doubles implementing just the methods `_QueueTracker` calls — `history()` and `next_event()`/`ensure_connected()` respectively — not the fake WebSocket server fixture from the WebSocket transport document's §9.1, which exercises a real socket and is unnecessary overhead for testing `_QueueTracker`'s pure state-machine logic).

### 10.1 Fixtures

- A scriptable fake `_ComfyUIWebSocketTransport` double whose `next_event(timeout_seconds)` returns a pre-programmed sequence of `ComfyUIEvent | None` values (or raises `ComfyUIConnectionError` at a programmed point in the sequence), and whose `ensure_connected()` can be programmed to succeed or raise on each call.
- A scriptable fake `_ComfyUIHTTPTransport` double whose `history(prompt_id)` returns a pre-programmed sequence of `dict | None` values per call (or raises `_ComfyUIHTTPError` at a programmed point).
- A fake monotonic clock (patching `time.monotonic`, or a small injectable clock function) so timing-dependent tests (deadline expiry, reconnect cadence, progress-milestone thresholds) run instantly rather than sleeping in real wall-clock time.

### 10.2 Successful execution

- Full happy path: `status` (queue_remaining=2) → `status` (queue_remaining=0) → `executing(node="4")` → `progress` ×N → `executing(node=None)` → confirming `history()` returns a success-status payload. Assert `outcome=COMPLETED`, correct `history_payload`, `queue_wait_seconds`/`generation_seconds` both plausible and nonzero, `used_http_fallback=False`.
- Directly-cached path (§4.2's note): `status` → `executing(node=None)` with no intervening named-node `executing` event. Assert `COMPLETED`, `generation_seconds == 0.0`.

### 10.3 Queue waiting

- Multiple `status` events with decreasing `queue_remaining` before the first `executing` event — assert no state transition occurs on `status` alone, and `queue_wait_seconds` is measured up to the eventual `Executing` transition, not reset by each `status` event.
- A `status`-only stretch long enough to cross `MODULE7_STILL_QUEUED_WARNING_SECONDS` — assert exactly one WARNING is logged (not one per subsequent `status` event).

### 10.4 Execution progress

- A sequence of `progress` events for our `prompt_id` crossing 25/50/75/100% of `progress_max` — assert one INFO log per crossing and DEBUG-only logs for intervening frames (log-capture fixture, asserting counts by level).
- `progress`/`executing` events for a **different** `prompt_id` interleaved throughout — assert they are ignored (§4.4) and do not affect timing or milestone counts.

### 10.5 Completion detection

- `executing(node=None)` followed by a confirming `history()` call whose status is malformed/missing on the first 2 attempts and valid-success on the 3rd — assert `COMPLETED` and assert exactly 3 `history()` calls were made (bounded retry, §7.4).
- `executing(node=None)` followed by a confirming `history()` call whose status indicates `"error"` — assert `outcome=EXECUTION_ERROR` with `error_payload` derived from the history entry, and a WARNING logged for the WS/history mismatch (§7.7).
- `execution_error` event arrives directly (no completion signal at all) — assert `EXECUTION_ERROR` with `error_payload` taken directly from the event, and assert **zero** `history()` calls were made (§5.3 — no confirming call needed for this path).
- `execution_cached` events interleaved before the real completion signal — assert no state transition and DEBUG-only logging (§5.7).

### 10.6 Timeout

- No terminal event ever arrives; fake clock advances past `execution_timeout_seconds` — assert `outcome=TIMEOUT`, `history_payload is None`, `error_payload is None`.
- Timeout occurring after execution had already started (`executing(node="4")` observed, then silence until deadline) — assert `TIMEOUT` with a nonzero, plausible partial `generation_seconds` (§4.5).

### 10.7 Cancellation

- Not independently tested as a `_QueueTracker` outcome, per §4.6 — no `Cancelled` outcome exists to test. Instead: a scripted `execution_error` event for our `prompt_id` immediately following a simulated prior `cancel()` (i.e., the fake WS double is scripted to emit exactly the interruption-shaped `execution_error` a real ComfyUI server would after an interrupt) — assert it is handled identically to any other `EXECUTION_ERROR` path (§10.5), confirming no special-casing is needed or present.

### 10.8 WebSocket disconnect

- `ws.next_event()` raises `ComfyUIConnectionError` mid-loop — assert one WARNING is logged, `_run_polling_phase()` is entered (assertable via a spy on the fake HTTP double receiving its first `history()` call shortly after), and `used_http_fallback=True` on the eventual result regardless of how the call ultimately terminates.

### 10.9 HTTP fallback

- After a WS disconnect, a scripted sequence of `history()` return values (`None`, `None`, non-terminal status, terminal success status) — assert polling continues across the non-terminal responses and terminates correctly on the terminal one, with the expected number of `time.sleep`-equivalent waits at `poll_interval_seconds` cadence (assert via the fake clock advancing by the expected total).
- A `history()` call raising `_ComfyUIHTTPError` mid-polling — assert one WARNING logged, polling continues on the next scheduled cycle rather than aborting (§6.2, §7.2).

### 10.10 Reconnect

- During polling, `ws.ensure_connected()` is scripted to fail on the first two reconnect-cycle attempts and succeed on the third — assert DEBUG (not WARNING) logs for the failed attempts, and assert that after the successful attempt, exactly one *extra*, out-of-cadence `history()` call happens immediately (§6.4) before the fake WS double's `next_event()` is called again.
- The immediate post-reconnect `history()` check itself returns terminal — assert the result is returned directly from that check, without ever re-entering `_run_websocket_phase()`.
- The immediate post-reconnect `history()` check is non-terminal — assert `_run_websocket_phase()` is re-entered and subsequent scripted WS events are consumed normally.
- Remaining budget below `COMFYUI_WS_RECONNECT_MIN_BUDGET_SECONDS` at a scheduled reconnect-cycle point — assert `ensure_connected()` is **not** called that cycle (§6.4's guard), via a call-count assertion on the fake WS double.

### 10.11 Malformed events

- Already exhaustively covered at the transport layer (WebSocket transport document §9.3) — `_QueueTracker`'s own tests only need one confirming case: a fake WS double's `next_event()` returning `None` (standing in for "a malformed frame was already filtered by the transport") interleaved among real events — assert this is handled identically to an ordinary read-timeout `None` (no special branch, per §7.6).

### 10.12 Missing history

- Confirming call returns `None` on every attempt through the retry budget, then `_QueueTracker` falls back to polling and a subsequent poll resolves it — assert the full path: retries exhausted (§7.4) → WARNING logged (§5.6) → polling phase entered → eventual terminal classification.

### 10.13 Retry logic

- Assert the confirming-call retry never exceeds `COMFYUI_HISTORY_CONFIRMATION_RETRY_ATTEMPTS` calls even when every attempt keeps returning `None` and the overall deadline is still far off (proving the retry bound is independent of, and tighter than, the overall timeout).
- Assert the confirming-call retry is cut short (fewer than the configured attempt count) if the overall `execution_timeout_seconds` deadline would otherwise be exceeded by continuing (§7.4's "however many fit" rule) — a deadline-near-exhaustion scenario constructed via the fake clock.

### 10.14 What this test class explicitly does not need to cover

Anything already exhaustively covered by `TestWebSocketTransport` (WebSocket transport document §9) or a future `TestHTTPTransport`/existing HTTP transport tests — frame parsing correctness, binary-frame filtering, raw socket-level reconnection mechanics, or `_ComfyUIHTTPTransport`'s own exception-translation behavior. `TestQueueTracker` only ever interacts with the two transports through hand-rolled doubles implementing their already-tested public contracts, never through the real fake-WebSocket-server fixture — keeping this test class's failures attributable exclusively to `_QueueTracker`'s own state-machine logic.

---

## 11. Integration

### 11.1 With `ComfyUIClient.generate()`

Per the master document §3.1's seven-step internal workflow, `_QueueTracker` participates in exactly step 3: `ComfyUIClient` calls `self._queue_tracker.await_completion(prompt_id, self._client_id)` once, after step 2's `submit_prompt()` has returned a `prompt_id`, and receives one `_CompletionResult` back. `ComfyUIClient` then:

- On `outcome=COMPLETED` (step 4): calls `_OutputRetriever.fetch(prompt_id, result.history_payload, stage_output_dir, candidate_index)`.
- On `outcome=EXECUTION_ERROR` (step 5): calls `_classify_comfyui_error(result.error_payload)` to decide between `VRAMExhaustedError` (retryable, Tenacity OOM layer) and `ComfyUIQueueError` (terminal).
- On `outcome=TIMEOUT` (step 6): calls `self.cancel(prompt_id)` best-effort, then raises `ComfyUITimeoutError`.
- In all three cases (step 7, `finally` block): reads `result.queue_wait_seconds`, `result.generation_seconds`, and `result.used_http_fallback` (the last one feeding into `failure_reason` classification/logging context, not a `GenerationMetrics` field itself) to build the `_AttemptOutcome` passed to `_ComfyUIMetricsRecorder.record_attempt()`.

`_QueueTracker` itself has no awareness of `candidate_index`, `stage_output_dir`, or any Phase-1-originated type (`BuiltWorkflow`, `GenerationProfile`) — its input surface is exactly `prompt_id` and `client_id`, keeping it fully decoupled from everything upstream of submission, matching the architecture document's layering principle the WebSocket transport document already applied to itself.

### 11.2 With `_OutputRetriever`

Indirect, through `_CompletionResult.history_payload` only (§2.4). `_QueueTracker` never imports `_OutputRetriever` and never validates the shape of `history_payload["outputs"]` — that is entirely `_OutputRetriever.fetch()`'s job (master document §3.5 steps 1–2), including raising `ComfyUIOutputMissingError` if no output-image entries are found. This is worth stating plainly here because it is the integration point most likely to be assumed richer than it is: `_QueueTracker` guarantees only that `history_payload` is a dict with a terminal, non-error `status` (per §5.5's classification) — it makes no guarantee about what `history_payload["outputs"]` contains.

### 11.3 With `_ComfyUIMetricsRecorder`

None directly (§2.5) — `_QueueTracker` never imports `_ComfyUIMetricsRecorder` or `MetricsCollector`. The dependency chain remains the strict one-directional shape established by the WebSocket transport document: `_ComfyUIWebSocketTransport → _QueueTracker → ComfyUIClient → _ComfyUIMetricsRecorder`, with `_ComfyUIHTTPTransport` feeding into `_QueueTracker` alongside the WebSocket transport at the same layer.

---

*End of design specification for `_QueueTracker`. No implementation code is included per the task's DESIGN ONLY constraint. This document is intended to be handed to the next Codex iteration alongside `docs/MODULE7_PHASE2_COMFYUI_INTEGRATION_DESIGN.md` and `docs/MODULE7_PHASE2_WEBSOCKET_TRANSPORT_DESIGN.md`; where the three overlap (§3.4, §4.2, §5, §6, §7, §8, §9, §10, §11, §13 of the master document; §2.3 of the WebSocket transport document), this document elaborates rather than supersedes, and any future edit to one should be checked against the other two for drift, per the master document's own §16 step 16 convention.*

Do NOT redesign the completed HTTP transport.

Do NOT redesign the completed WebSocket transport.

Do NOT implement OutputRetriever.

Do NOT implement MetricsRecorder.

Do NOT implement ComfyUIClient.generate().

Keep all changes limited to QueueTracker and its associated tests.

Run the focused QueueTracker tests.

Run the full project test suite.

Do not leave any failing tests.