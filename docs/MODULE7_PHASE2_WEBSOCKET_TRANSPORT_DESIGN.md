# MODULE7_PHASE2_WEBSOCKET_TRANSPORT_DESIGN.md

**Module 7 — Phase 2: ComfyUI Integration**
**Component: `_ComfyUIWebSocketTransport`**
**thumbnail-ai**

Status: **Design specification, v1.0. No implementation.**
Source of truth: `docs/MODULE7_PHASE2_COMFYUI_INTEGRATION_DESIGN.md` (§1, §2.2–2.4, §3.3, §4.2, §5, §6, §7, §8, §9, §13 in particular), `modules/comfyui_client.py` as it exists today (Sprint 1 — `_ComfyUIHTTPTransport` and its seven methods, complete and unmodified by this document), `modules/module7_exceptions.py`, `modules/config.py`, `modules/models.py`.
This document narrows and elaborates the master design's §3.3 into an implementation-ready blueprint for `_ComfyUIWebSocketTransport` alone. It introduces no public interface the master document did not already reserve, and changes nothing in the completed HTTP transport.

Assumed complete and unmodified by this document: `_ComfyUIHTTPTransport` and its seven methods (`system_stats`, `submit_prompt`, `history`, `view_image`, `interrupt`, `delete_from_queue`, `queue_status`), `verify_comfyui_http.py`, and the existing HTTP transport tests.

---

## 1. Purpose

### 1.1 Why this component exists

ComfyUI's HTTP API (`_ComfyUIHTTPTransport`) can submit a prompt and can be polled for its result via `/history/{prompt_id}`, but polling alone is a poor way to know *when* a submission finishes: there is no cheap way to distinguish "still queued behind other jobs," "actively sampling," and "just finished" without hammering `/history` in a tight loop. ComfyUI's own UI solves this by opening a WebSocket connection and listening for push events (`status`, `executing`, `progress`, `execution_error`, `execution_cached`, and others) as they happen. `_ComfyUIWebSocketTransport` exists to give `thumbnail-ai` the same low-latency, low-overhead visibility: a typed, timeout-bounded event stream that `_QueueTracker` (§3.4 of the master document) consumes to detect completion and surface progress, instead of busy-polling HTTP.

### 1.2 Responsibilities

- Own exactly one WebSocket connection to `ws://{host}:{port}{COMFYUI_WS_PATH}?clientId={client_id}` per `ComfyUIClient` instance.
- Establish that connection on demand, idempotently, with a bounded connect timeout.
- Read raw frames off the socket with a bounded per-read timeout, so a caller looping on this transport can always re-check its own outer budget rather than being blocked indefinitely.
- Parse the small, closed set of ComfyUI JSON text-frame shapes that Phase 2 cares about into one typed `ComfyUIEvent` dataclass (§4.2 of the master document), filtering out frame types Phase 2 has no opinion on (binary preview frames, event types outside the tracked set) at the parser boundary.
- Convert every third-party failure (`websocket-client` exceptions, malformed frames, connection loss) into either a `None` return, a logged-and-skipped event, or a typed `Module7Error` subclass — never let a raw `websocket` exception escape this class.
- Close the connection idempotently and release the underlying socket on request.

### 1.3 Boundaries — what this component explicitly does not do

- It does not decide what a `status`/`executing`/`execution_error` event *means* for the lifecycle of a specific `prompt_id` — that interpretation, including the state machine and the polling fallback, belongs entirely to `_QueueTracker` (§3.4 of the master document, elaborated for its interaction with this transport in §3 below). `_ComfyUIWebSocketTransport` is a dumb pipe with a typed exit shape; it has no concept of "my job" versus "someone else's job" on the shared ComfyUI server.
- It does not retry a failed connection itself. Connection-level retry (Tenacity, bounded attempts, exponential backoff) is applied one layer up, exactly as specified in §6 of the master document — this class raises `ComfyUIConnectionError` once and lets the caller's retry decorator decide whether to call `ensure_connected()` again.
- It does not poll HTTP. Falling back to `_ComfyUIHTTPTransport.history()` polling when the socket drops mid-job is `_QueueTracker`'s responsibility (master document §2.4); this transport's only contribution to that fallback is raising a distinguishable `ComfyUIConnectionError` when the read fails because the connection is actually closed, rather than swallowing that failure as an ordinary timeout.
- It does not consume live-preview binary frames (ComfyUI's optional binary image-preview protocol extension). There is no live-preview feature in this phase; binary frames are recognized and discarded, never buffered or decoded.
- It does not touch `_ComfyUIHTTPTransport`, `_OutputRetriever`, or any Pydantic model in `models.py`. It has zero import-time dependency on any of them.

---

## 2. Architecture

### 2.1 Component placement

`_ComfyUIWebSocketTransport` lives in `modules/comfyui_client.py`, alongside `_ComfyUIHTTPTransport` (already present) and the other Phase 2 internal collaborators the master document specifies (`_QueueTracker`, `_OutputRetriever`, `_ComfyUIMetricsRecorder`). It is a module-internal class — not listed in `comfyui_client.py`'s `__all__` — for the same reason `_ComfyUIHTTPTransport` is not: it is an implementation detail of `ComfyUIClient`, never constructed directly by a caller outside this module or its tests.

### 2.2 Interaction with `_ComfyUIHTTPTransport`

None, at the class level. `_ComfyUIWebSocketTransport` holds no reference to `_ComfyUIHTTPTransport` and calls none of its methods. The two transports are siblings, both owned and coordinated by `ComfyUIClient` (and, one level down, by `_QueueTracker`, which is handed both as constructor arguments per the master document's §3.4 signature: `_QueueTracker(http, ws, poll_interval_seconds, execution_timeout_seconds)`). This separation is deliberate: it lets both transports be unit-tested completely independently (§9 of this document), and it is what allows `_QueueTracker` to fail over from one to the other (WS → HTTP polling) without either transport needing to know the other exists.

### 2.3 Interaction with the future `_QueueTracker`

`_QueueTracker` is this transport's only caller. The relationship is a strict consumer/producer pair:

- `_QueueTracker.await_completion()` calls `ws.ensure_connected()` once at the start of a wait cycle (idempotent — a no-op if `ComfyUIClient` already connected the socket for a prior candidate in the same `with ComfyUIClient(...)` block).
- `_QueueTracker`'s loop calls `ws.next_event(COMFYUI_WEBSOCKET_TIMEOUT_SECONDS)` repeatedly. A `None` return means "no event this read cycle" and is not an error — the loop simply re-checks its own outer `execution_timeout_seconds` budget and calls `next_event` again.
- A `ComfyUIConnectionError` raised out of `next_event` is `_QueueTracker`'s signal to switch to the HTTP polling fallback (master document §2.4) and to attempt `ws.ensure_connected()` again on a slower cadence in the background, without that reconnect attempt blocking the polling loop.
- `_QueueTracker` never calls `ws.close()` — connection lifetime is scoped to `ComfyUIClient`, not to a single `await_completion()` call, so the same socket can serve multiple sequential candidates for one video without reconnecting each time (master document §12).

### 2.4 Interaction with the future `_OutputRetriever`

None. `_OutputRetriever` consumes the confirming `_ComfyUIHTTPTransport.history()` call `_QueueTracker` makes after it sees a completion signal (master document §3.4 step "e") — it never touches the WebSocket transport, directly or indirectly. This is intentional: output *bytes* only ever travel over HTTP (`GET /view`); the WebSocket carries no payload data, only event metadata, so there is no scenario in which `_OutputRetriever` would need this class.

### 2.5 Interaction with `ComfyUIClient`

`ComfyUIClient` is the only code outside this module (and its tests) that ever constructs a `_ComfyUIWebSocketTransport`. Per the master document's §3.1 internal workflow, `ComfyUIClient.generate()` step 1 is "ensure WebSocket is connected" — this delegates directly to `ws.ensure_connected()`. `ComfyUIClient.close()` delegates to `ws.close()`. `ComfyUIClient` never calls `next_event` directly; that call is exclusively made from inside `_QueueTracker`. `ComfyUIClient` also owns the `client_id` (a `uuid4` string, generated once per `ComfyUIClient` instance unless the caller supplies one) that is passed into this transport's constructor and appended to the WebSocket URL's `clientId` query parameter — this is what lets ComfyUI's shared, multi-client WebSocket endpoint be meaningfully filtered by `_QueueTracker`'s "ignore other jobs' events" rule (§4.4 below).

---

## 3. Public API

All signatures below are final for this phase; nothing here is expected to change shape once implemented (per the master document's "additive only" convention). No implementation code — parameter/return/exception/threading/timeout behavior only.

### 3.1 `__init__(self, ws_url: str, client_id: str, connect_timeout_seconds: float) -> None`

| | |
|---|---|
| **Parameters** | `ws_url: str` — the fully-formed WebSocket URL including scheme, host, port, and path, e.g. `ws://127.0.0.1:8188/ws` (the `clientId` query parameter is appended internally at connect time from `client_id`, not baked into `ws_url` by the caller, keeping URL construction in one place). `client_id: str` — the `uuid4`-derived identifier `ComfyUIClient` generated or was given; used both in the connect URL and, implicitly, by `_QueueTracker` to filter events. `connect_timeout_seconds: float` — the maximum time `ensure_connected()` will block attempting the initial handshake before raising. |
| **Returns** | `None`. Constructing an instance never opens a socket — connection is fully lazy, established only by the first `ensure_connected()` call. |
| **Raises** | `ValueError` if `ws_url` does not start with `ws://` or `wss://`, if `client_id` is empty/whitespace, or if `connect_timeout_seconds <= 0` — the same fail-fast constructor validation style `_ComfyUIHTTPTransport.__init__` already uses for `base_url` and `timeout_seconds`. Never raises anything socket-related; no I/O happens in `__init__`. |
| **Thread-safety** | Not thread-safe by design (matching the rest of `comfyui_client.py`, which is single-threaded/synchronous throughout, per the master document §3.3's note that the codebase has zero `asyncio` usage). One `_ComfyUIWebSocketTransport` instance is owned by exactly one `ComfyUIClient` instance used from one thread. |
| **Timeout behavior** | N/A — no blocking I/O in the constructor. |

### 3.2 `ensure_connected(self) -> None`

| | |
|---|---|
| **Parameters** | None. |
| **Returns** | `None`. |
| **Behavior** | No-op — returns immediately — if the transport already believes it holds an open, healthy connection (`is_connected()` would return `True`). Otherwise performs the WebSocket handshake against `ws_url` (with `?clientId={client_id}` appended) using `connect_timeout_seconds` as the connect timeout, and on success transitions the transport's internal state to `Connected` (§5). This method is what both the very first connection *and* every reconnection attempt (after a drop, §5's `Failed → Connecting` transition) go through — there is no separate "reconnect" method, keeping the public surface minimal and matching the idempotent-`ensure_*` naming convention `ComfyUIClient.generate()` step 1 already expects (master document §3.1). |
| **Raises** | `ComfyUIConnectionError` if the handshake does not complete within `connect_timeout_seconds`, or if the underlying TCP connect is refused, or if the server responds with a non-101 HTTP status during the WebSocket upgrade (e.g., ComfyUI not actually running at that host/port, or a proxy/firewall rejecting the upgrade). The original `websocket-client` exception (`websocket.WebSocketException` and subclasses, `ConnectionRefusedError`, `socket.timeout`) is always chained via `raise ... from exc`, never left to propagate directly — mirroring `_ComfyUIHTTPTransport._request`'s existing `except requests.RequestException` translation pattern exactly. |
| **Thread-safety** | Not thread-safe; only ever called from the single owning thread (`ComfyUIClient.generate()`, or directly from `_QueueTracker.await_completion()`'s reconnect-attempt step). |
| **Timeout behavior** | Bounded by `connect_timeout_seconds` (sourced from `COMFYUI_STARTUP_TIMEOUT_SECONDS`, §7). This is a *connect-only* timeout — it does not bound how long the connection subsequently stays open or how long `next_event` reads take; those are governed independently by `COMFYUI_WEBSOCKET_TIMEOUT_SECONDS` (§3.4 below). This method itself performs no automatic retry — the connection-level Tenacity retry layer (master document §6, layer 1) wraps *calls to* `ensure_connected()`, not logic inside it, keeping this method a pure single-attempt primitive that is easy to reason about and to fake in tests. |

### 3.3 `receive(self, timeout_seconds: float) -> str | None`

A low-level, transport-only primitive, deliberately separated from `next_event` (§3.4) so the "read one raw frame off the socket" concern and the "parse it into a `ComfyUIEvent`" concern are each independently unit-testable — the same layering principle the master document applies throughout (e.g., `_ComfyUIHTTPTransport._request` versus `_ComfyUIHTTPTransport._request_json`).

| | |
|---|---|
| **Parameters** | `timeout_seconds: float` — the per-call read timeout, set on the underlying socket before each `recv()` attempt (the `websocket-client` library's blocking client supports a settable receive timeout per call; this method sets it every call rather than once at connect time, since different callers — `next_event`'s normal loop versus a future draining/shutdown path — may reasonably want different read timeouts). |
| **Returns** | `str | None`. The raw text-frame payload (still-unparsed JSON string) on a successful read within the timeout window. `None` if the read timed out with no frame arriving — this is the expected, common case while waiting on a still-executing job, not an error. Binary frames (ComfyUI's optional live-preview extension) are detected at this layer and treated identically to a timeout: silently discarded, `None` returned — Phase 2 has no live-preview feature and must not attempt to decode binary payloads as JSON. |
| **Raises** | `ComfyUIConnectionError` if the socket is not currently connected (caller did not call `ensure_connected()` first, or a prior read already discovered the connection was closed) or if the read fails because the connection was actively closed by the peer or the network (`websocket.WebSocketConnectionClosedException`, `ConnectionResetError`, `BrokenPipeError`) — this is the one failure mode `next_event`/`_QueueTracker` must be able to distinguish from an ordinary timeout, since it is what triggers the polling fallback (master document §2.4). A plain read timeout is **not** an exception here; it is the `None` return path. |
| **Thread-safety** | Not thread-safe; single-threaded blocking call, same as every other method on this class. |
| **Timeout behavior** | Exactly `timeout_seconds`, applied to this one call only (not cumulative across calls) — this is the mechanism that lets `_QueueTracker`'s outer loop stay responsive to its own, much larger `execution_timeout_seconds` budget: a 5-second `receive`/`next_event` timeout means the outer loop re-checks its overall deadline at least once every 5 seconds even during a long, quiet stretch of the job. |

### 3.4 `next_event(self, timeout_seconds: float) -> ComfyUIEvent | None`

The primary method `_QueueTracker` calls in its loop. Built directly on top of `receive()`.

| | |
|---|---|
| **Parameters** | `timeout_seconds: float` — forwarded as-is to the internal `receive()` call. |
| **Returns** | `ComfyUIEvent | None` (dataclass defined in §4.2 of the master document, reproduced in §4.5 below for this document's self-containedness). `None` in three cases, all treated identically by the caller: (a) `receive()` itself timed out, (b) `receive()` returned a frame but it was a binary/preview frame already filtered inside `receive()`, or (c) `receive()` returned a text frame that parsed as JSON but whose `type` field is outside the tracked set (§4.3) — an unrecognized-but-well-formed ComfyUI event. Distinguishing these three internally is unnecessary for `_QueueTracker`'s state machine, which only needs "nothing actionable happened this read" — but each case is logged distinctly at DEBUG (§8) so a developer inspecting `module7.log` can tell them apart even though the caller cannot. |
| **Raises** | `ComfyUIConnectionError`, propagated unchanged from `receive()` — this is the only exception type this method can raise; every other failure mode (malformed JSON, a JSON object missing required fields, an unexpected `type` value) is handled internally and converted to a logged-WARNING-and-`None` outcome (§6.3), never an exception, per the master document's explicit design choice that "a single corrupt progress frame should never fail a whole generation." |
| **Thread-safety** | Not thread-safe; single-threaded blocking call. |
| **Timeout behavior** | Identical to `receive()` — bounded by `timeout_seconds` for this one call. |

**Internal workflow** (for implementer clarity; not itself part of the public contract):

1. Call `self.receive(timeout_seconds)`.
2. If `None`, return `None`.
3. Attempt `json.loads(frame)`. On `json.JSONDecodeError`: log WARNING with a truncated frame preview (never the full raw payload at WARNING — see §8), return `None`.
4. Validate the parsed object is a `dict` with a string `"type"` key. If not: log WARNING, return `None`.
5. Dispatch on `type` per the event catalog in §4. Recognized types are mapped into a `ComfyUIEvent`; unrecognized types are logged at DEBUG (not WARNING — an unrecognized-but-well-formed event from a newer ComfyUI version is expected forward-compatible behavior, not a problem) and `None` is returned.
6. For recognized types, validate the type-specific required fields (§4.3's per-type table). A recognized type with a malformed/missing required field is treated the same as an unrecognized type: WARNING log, `None` return — never a partial or best-effort `ComfyUIEvent`, since `_QueueTracker` must never be handed an event it cannot trust the shape of.

### 3.5 `is_connected(self) -> bool`

| | |
|---|---|
| **Parameters** | None. |
| **Returns** | `True` if the transport currently holds what it believes is an open socket (i.e., `ensure_connected()` has succeeded and neither `close()` nor a detected connection-loss has happened since); `False` otherwise, including before the first `ensure_connected()` call. This is a cheap, local state check — it does **not** perform a network round-trip (e.g., no ping/pong probe) to verify the connection is *actually* still alive on the wire; a connection that has silently died since the last successful read will still report `True` here until the next `receive()`/`next_event()` call discovers the closure. This matches how `_ComfyUIHTTPTransport` has no equivalent liveness check either — liveness is only ever discovered by attempting an operation, never pre-flighted. |
| **Raises** | Never raises. |
| **Thread-safety** | Safe to call from the owning thread at any time; it is a pure state read with no I/O. |
| **Timeout behavior** | N/A — synchronous, no blocking. |

### 3.6 `close(self) -> None`

| | |
|---|---|
| **Parameters** | None. |
| **Returns** | `None`. |
| **Behavior** | Idempotent. If not connected, returns immediately without error. If connected, closes the underlying `websocket-client` connection (sends a close frame if the connection is healthy enough to do so; falls back to a hard socket close if it is not) and transitions internal state to `Closed` (§5). Safe to call from any state, including `Failed`. |
| **Raises** | Never raises — any exception encountered while closing an already-unhealthy socket is caught internally and logged at DEBUG (a failure to cleanly close a socket that is being discarded anyway is not actionable and must never mask or replace whatever outcome the caller is already returning/raising, matching the same "never let cleanup mask the real error" principle the master document applies to `_ComfyUIMetricsRecorder`, §11 of the master document). |
| **Thread-safety** | Not thread-safe; called only from the owning thread, typically via `ComfyUIClient.close()` / `ComfyUIClient.__exit__`. |
| **Timeout behavior** | Best-effort; internally bounded by a short, fixed close-handshake timeout (not separately configurable — this is not a caller-tunable operation) so a hung close can never block process shutdown indefinitely. |

### 3.7 Deliberately absent from the public surface

- No `send()` method. `_ComfyUIWebSocketTransport` never writes to the socket — ComfyUI's WebSocket channel is server-push-only for Phase 2's purposes; the only outbound action (`submit_prompt`) happens over HTTP. Omitting `send()` entirely (rather than exposing an unused one) keeps the class honest about being a read-only event stream.
- No `reconnect()` distinct from `ensure_connected()` — see §3.2.
- No context-manager protocol (`__enter__`/`__exit__`) on this class itself. Only `ComfyUIClient` (the public facade) is a context manager, per the master document §3.1/§12; giving every internal collaborator its own context-manager protocol would be redundant surface area for a class nothing outside this module ever holds onto directly.

---

## 4. Event model

### 4.1 ComfyUI's WebSocket message catalog (background)

A ComfyUI server, once a client is connected to `/ws?clientId=...`, pushes JSON text frames of the shape `{"type": "<event_type>", "data": {...}}` for the lifetime of the connection, interleaved across **every** client currently connected and **every** job currently queued or running on that server (the endpoint is not scoped to one submission). The event types ComfyUI is known to emit include: `status` (queue-remaining updates, sent periodically and on every queue-length change), `execution_start` (a prompt has begun executing), `execution_cached` (some nodes were served from cache rather than recomputed), `executing` (a specific node is about to run, or — with `node: null` — the whole prompt has finished), `progress` (a node reporting incremental progress, most commonly a sampler's step count), `executed` (a specific node has finished and is reporting its output, e.g. an image save node), and `execution_error` (a node raised during execution). Binary frames (an opt-in live-preview feature sending intermediate image bytes) may also appear on the same socket.

### 4.2 `ComfyUIEvent` (frozen dataclass — reproduced from the master document §4.2 for reference; not redefined by this document)

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

### 4.3 Per-type payload structure, validation, and mapping

| ComfyUI `type` | Raw `data` shape (as documented by ComfyUI's own `server.py` send sites) | Mapped into `ComfyUIEvent`? | Required-field validation before mapping | Populated `ComfyUIEvent` fields |
|---|---|---|---|---|
| `status` | `{"status": {"exec_info": {"queue_remaining": <int>}}}` | Yes | `data.status.exec_info.queue_remaining` must be present and an `int` ≥ 0. | `event_type="status"`, `queue_remaining=<value>`; `prompt_id`, `node`, `progress_value`, `progress_max`, `error_payload` all `None` — `status` events are server-wide, never scoped to one `prompt_id`. |
| `executing` | `{"node": <str \| null>, "display_node": <str>, "prompt_id": <str>}` | Yes | `data.prompt_id` must be a non-empty string. `data.node` may legitimately be `null` (this is the completion signal, §4.4) or a string. | `event_type="executing"`, `prompt_id=<value>`, `node=<value or None>`; progress/queue/error fields `None`. |
| `progress` | `{"value": <int>, "max": <int>, "prompt_id": <str>, "node": <str>}` | Yes | `data.value`, `data.max` must be non-negative ints with `value <= max`; `data.prompt_id` non-empty string. A payload failing this check is treated as malformed (§3.4 step 6), not silently clamped — Phase 2 never guesses at a corrected value. | `event_type="progress"`, `prompt_id`, `node`, `progress_value=value`, `progress_max=max`; `queue_remaining`, `error_payload` `None`. |
| `execution_error` | `{"prompt_id": <str>, "node_id": <str>, "node_type": <str>, "exception_type": <str>, "exception_message": <str>, "traceback": [<str>, ...], ...}` | Yes | `data.prompt_id` non-empty string required; every other field is passed through best-effort inside `error_payload` — this event type is intentionally the most lenient on shape, since surfacing *some* diagnostic payload to `_classify_comfyui_error` (master document §3.1 step 5, §6) is more valuable than discarding a slightly-off-schema error just because one optional diagnostic field is missing. | `event_type="execution_error"`, `prompt_id`, `error_payload=data` (the full dict, for `_classify_comfyui_error` to inspect — **never logged directly**, see §8); `node`, progress/queue fields `None`. |
| `execution_cached` | `{"nodes": [<str>, ...], "prompt_id": <str>}` | Yes | `data.prompt_id` non-empty string. | `event_type="execution_cached"`, `prompt_id`; `node` set to `None` (the event carries a *list* of cached node IDs, not one — Phase 2 has no per-node cache bookkeeping need in this phase, so the list itself is not retained on the dataclass, only logged at DEBUG). |
| `execution_start` | `{"prompt_id": <str>}` | **No** — filtered at the parser boundary (§3.4 step 5), `next_event` returns `None`. | N/A | N/A — `_QueueTracker`'s state machine does not need a distinct "start" signal beyond the first `executing` event it already sees; adding a sixth `ComfyUIEvent` variant for a signal `_QueueTracker` would immediately discard was rejected as unnecessary surface area, matching the master document's principle that the parser boundary should keep `_QueueTracker` "free of ignore-this branches for concerns Phase 2 has no opinion on" (§4.2 of the master document). Logged at DEBUG only. |
| `executed` | `{"node": <str>, "output": {...}, "prompt_id": <str>}` | **No** — filtered. | N/A | N/A — this event fires once per output-producing node (e.g., a `SaveImage` node reporting its saved filenames) and, in principle, *could* be used to short-circuit `_OutputRetriever` by reading `output.images` straight off the socket instead of doing a confirming `GET /history/{prompt_id}` call afterward. This document deliberately does **not** adopt that optimization: the master document's §3.4 step "e" already specifies the confirming HTTP call as the authoritative source of output filenames precisely because "the WS event itself does not carry output paths" reliably across ComfyUI versions/node types in every configuration, and re-deriving that contract here would be redesigning an already-specified component. Logged at DEBUG only, in case a future phase revisits this tradeoff. |
| *(unrecognized / future ComfyUI event types)* | Unknown | **No** — filtered. | N/A | N/A — logged at DEBUG with the raw `type` string only (never the full `data` payload, in case a future ComfyUI version's unrecognized event happens to carry something sensitive-shaped) so forward compatibility with newer ComfyUI releases never raises or breaks anything; it just does nothing, silently, at the lowest log tier. |
| *(binary frame)* | N/A (opaque bytes) | **No** — filtered inside `receive()`, never reaches the JSON-parsing step at all. | N/A | N/A — logged at DEBUG with byte length only. |

### 4.4 The completion signal

The single most important event for `_QueueTracker` is `executing` with `node: null` for the tracked `prompt_id` — this is ComfyUI's canonical "this prompt is entirely finished" signal (it fires once, after every node in the graph has executed). `_ComfyUIWebSocketTransport` treats this exactly like any other `executing` event structurally (§4.3's row for `executing` covers both cases identically — `node` is simply `None`); it is `_QueueTracker`, not this transport, that gives `node is None` its special "stop waiting, go confirm via HTTP history" meaning (master document §3.4 step "e"). This transport does not special-case it, keeping the parsing layer's logic uniform and the completion-detection *policy* entirely inside `_QueueTracker` where the rest of the state machine already lives.

### 4.5 Malformed and invalid frames — summary

Every failure mode below results in `next_event()` returning `None` plus a log line; **none** of them raise an exception, and none of them are retried at this layer (there is nothing to retry — the next `receive()` call will simply pick up whatever frame arrives next):

- Frame is not valid UTF-8 text (a binary frame, or corrupted text) → filtered in `receive()`.
- Frame is valid text but not valid JSON → `json.JSONDecodeError`, WARNING, `None`.
- Frame is valid JSON but not a JSON object (e.g., a bare string or array) → WARNING, `None`.
- Frame is a JSON object missing a `"type"` key or with a non-string `"type"` → WARNING, `None`.
- Frame has a recognized `"type"` but a missing/malformed `"data"` object or missing/malformed required sub-field per §4.3's table → WARNING, `None`.
- Frame has an unrecognized `"type"` → DEBUG (not WARNING — see §4.3), `None`.

---

## 5. State machine

### 5.1 States

| State | Meaning |
|---|---|
| `Disconnected` | Initial state after `__init__`. No socket exists. Also the state immediately after a clean `close()`... *(see `Closed` below for why these are kept distinct)*. |
| `Connecting` | `ensure_connected()` is actively performing the WebSocket handshake. Transient — this state is never observed by a caller between method calls; it exists only for the duration of one `ensure_connected()` invocation. |
| `Connected` | Handshake succeeded; the socket is open and idle (no read currently in flight). This is the state `is_connected()` reports `True` for. |
| `Receiving` | A `receive()`/`next_event()` call is actively blocked waiting for a frame or its timeout. Transient, like `Connecting` — exists only for the duration of one read call, then returns to `Connected` (frame or timeout received) or transitions to `Failed` (connection-loss detected mid-read). |
| `Closed` | `close()` was called explicitly while the transport was `Connected`/`Receiving`/`Failed`. Terminal for that connection's lifetime, but **not** terminal for the object: `ensure_connected()` called again after `Closed` is a normal, supported reconnection, transitioning back to `Connecting`. |
| `Failed` | A read or the handshake itself detected the connection is actually broken (peer closed it, network error, `WebSocketConnectionClosedException`). Like `Closed`, this is recoverable — `ensure_connected()` from `Failed` is exactly how `_QueueTracker`'s background reconnect-while-polling behavior (master document §2.4) works. |

`is_connected()` returns `True` only in state `Connected` or `Receiving` (mid-read, the socket is still considered "connected" from the caller's perspective — `Receiving` is not a distinct connectivity status, only a concurrency/re-entrancy guard state, §5.3). It returns `False` in every other state.

### 5.2 Valid transitions

```
                    ensure_connected()
                    (handshake succeeds)
  Disconnected ──────────────────────────► Connected
       ▲                                       │  │
       │                                       │  │ receive()/next_event() called
       │ (only via a fresh __init__;           │  ▼
       │  Disconnected is not re-entered       │ Receiving
       │  from any other state)                │  │
       │                                       │  │ frame arrives, or read times out
       │                                       │  └──────────► back to Connected
       │                                       │
       │                                       │  connection-loss detected during
       │                                       │  handshake or during a read
       │                                       ▼
Connecting ◄────────────────────────────── Failed ◄───────────────────────┐
  │     ▲            ensure_connected()          │                         │
  │     │            called again                │ close() called          │
  │     │  handshake fails                        │ while Failed            │
  │     └────────────────────────────────────────►│                         │
  │                                                ▼                         │
  │                                             Closed                       │
  │                                                │                         │
  │  handshake succeeds                            │ ensure_connected()       │
  ▼                                                │ called again             │
Connected ◄──────────────────────────────────────────────────────────────────┘
  │
  │ close() called while Connected/Receiving
  ▼
Closed
```

### 5.3 Transition notes and guards

- **`Disconnected → Connecting → Connected`**: the normal first-connection path, driven by the first `ensure_connected()` call. If the handshake fails, the transition is `Connecting → Failed` (not back to `Disconnected` — `Failed` is the single "not currently usable, but not brand-new either" state, whether the cause was an initial-connect failure or a later drop, so `_QueueTracker`'s reconnect logic has exactly one failure state to check for, not two).
- **`Connected ⇄ Receiving`**: every `receive()`/`next_event()` call transitions `Connected → Receiving` for its duration and back to `Connected` on a normal outcome (frame parsed or timeout). This transition pair exists primarily as a *re-entrancy guard*, not a caller-visible state: because this class is documented as not thread-safe and is only ever driven from one thread by design, `Receiving` is not expected to be observed concurrently with another call — but modeling it explicitly makes the "single blocking read at a time" invariant part of the design rather than an implicit assumption, which matters for the test suite (§9.4 asserts this invariant directly).
- **`Receiving → Failed`**: a read that discovers the connection is actually closed (as opposed to merely timing out) transitions directly to `Failed`, skipping `Connected`. This is the exact moment `receive()` raises `ComfyUIConnectionError` (§3.3).
- **`Failed → Connecting`** and **`Closed → Connecting`**: both are entered only via a fresh `ensure_connected()` call — there is no separate "reconnect" entry point (§3.2, §3.7). From the caller's perspective these two source states are handled identically by `ensure_connected()`; the state machine distinguishes them internally only so logging can say "reconnecting after an explicit close" versus "reconnecting after an unexpected drop" (§8), which is diagnostically useful even though the transition logic is otherwise the same.
- **`close()` is valid from every state** and always ends in `Closed`, including from `Connecting` (an in-flight handshake is abandoned) and from `Failed` (closing an already-broken connection is a normal, expected cleanup path, not an error — §3.6).
- **No transition ever returns to `Disconnected`.** `Disconnected` is exclusively the state a brand-new instance starts in; every subsequent "not connected" state after that first connection attempt is `Closed` or `Failed`, both of which are `ensure_connected()`-recoverable. This asymmetry is deliberate: it means `is_connected() is False` alone is never enough information to know whether reconnecting is meaningful (it always is, past the very first call) versus whether the object was ever used at all (only relevant for a test assertion, never for `_QueueTracker`'s runtime logic).

---

## 6. Error handling

### 6.1 Connection failure (initial handshake)

Handled entirely inside `ensure_connected()` (§3.2): a single attempt, bounded by `connect_timeout_seconds`, raising `ComfyUIConnectionError` on any handshake failure (refused connection, DNS failure, non-101 upgrade response, connect timeout). This method performs **no** internal retry loop — retrying is the caller's decision, applied via the shared connection-level Tenacity layer described in the master document §6 (layer 1: `COMFYUI_CONNECT_RETRY_ATTEMPTS`, `wait_exponential(min=COMFYUI_CONNECT_RETRY_WAIT_MIN_SECONDS, max=COMFYUI_CONNECT_RETRY_WAIT_MAX_SECONDS)`, `retry_if_exception_type` covering both the HTTP transport's and this transport's connection errors, one shared `_before_sleep_log`). Keeping the retry loop outside this class means the exact same Tenacity-wrapped call site can retry an HTTP connect and a WS connect with one consistent policy and one consistent log line shape, rather than duplicating a retry loop inside each transport.

### 6.2 Timeout (read)

Not an error at this layer at all — a `receive()`/`next_event()` call that times out returns `None` (§3.3, §3.4). The *overall* execution timeout (`COMFYUI_EXECUTION_TIMEOUT_SECONDS`, §7) that eventually turns "many consecutive `None`s" into a real `ComfyUITimeoutError` is entirely `_QueueTracker`'s responsibility (master document §3.4 step "h"); this transport has no concept of a cumulative deadline, only a per-call one.

### 6.3 Malformed JSON / invalid payload

Never raised as an exception — always a WARNING log plus a `None` return from `next_event()`, as detailed exhaustively in §4.5. This is a deliberate asymmetry from the HTTP transport, where a malformed JSON *response* to a request the caller is actively waiting on (`_ComfyUIHTTPTransport._request_json`) **is** raised as `_ComfyUIHTTPError`, because an HTTP caller has exactly one response to evaluate and nothing sensible to do but fail that one call. A WebSocket caller, by contrast, is in a long-lived stream where one bad frame among thousands is expected background noise (a truncated frame during a network hiccup, a ComfyUI version emitting one extra field) — failing the entire `generate()` call over one corrupt `progress` frame would be a severe availability regression for no correctness benefit, since the *next* frame (or the confirming HTTP history call) will still deliver the real answer.

### 6.4 Unexpected disconnect

Detected exclusively inside `receive()` (§3.3) by catching `websocket.WebSocketConnectionClosedException` (and the lower-level `ConnectionResetError`/`BrokenPipeError` the `websocket-client` library can also surface depending on platform/socket state) and re-raising as `ComfyUIConnectionError`. The transport transitions to `Failed` (§5) at the same moment. This is the single exception type `next_event()` can propagate, and it is exactly the signal `_QueueTracker` needs to fall back to HTTP polling (master document §2.4) — no other information is needed by that caller, so no richer exception payload (e.g., which specific socket error occurred) is exposed beyond the chained `from exc`.

### 6.5 Retry policy

This class performs zero retries internally, for either connects or reads — see §6.1 (connects retried by the caller's Tenacity layer) and §6.2 (reads are simply re-attempted in `_QueueTracker`'s own loop, which is not a "retry" in the Tenacity sense at all, just the normal shape of polling an event stream).

### 6.6 Cancellation

`_ComfyUIWebSocketTransport` has no cancellation concept of its own — `ComfyUIClient.cancel(prompt_id)` (master document §3.1) operates entirely over HTTP (`POST /interrupt` or `POST /queue {"delete": [...]}`), never touching this transport. If a `cancel()` call succeeds, the natural consequence is that ComfyUI will eventually emit an `execution_error` (interrupted) or simply stop sending `executing`/`progress` events for that `prompt_id` on the existing socket — `_QueueTracker` observes this the same way it observes any other terminal/stalled state, with no special-case code needed in this transport.

### 6.7 Graceful shutdown

`close()` (§3.6) is the sole shutdown path, called by `ComfyUIClient.close()` / `__exit__` (master document §12). It never raises, is safe from any state, and is always paired with the HTTP session's own close inside `ComfyUIClient.close()` so both transports tear down together.

---

## 7. Configuration

All constants below are proposed additions to `modules/config.py`'s `# Module 7 — Local Image Generation Engine` section, already reserved for Phase 2 by the master document's §8 — this document introduces no config constant the master document did not already name, and adds no new one of its own. `_ComfyUIWebSocketTransport` reads only the subset relevant to it:

| Constant | Used for | Default (per master document §8) |
|---|---|---|
| `COMFYUI_HOST`, `COMFYUI_PORT` | Building `ws_url` (`ws://{COMFYUI_HOST}:{COMFYUI_PORT}{COMFYUI_WS_PATH}`) — already present from Phase 1, reused as-is, no changes. | `"127.0.0.1"`, `8188` |
| `COMFYUI_WS_PATH` | The path segment of the WebSocket URL. | `"/ws"` |
| `COMFYUI_STARTUP_TIMEOUT_SECONDS` | `connect_timeout_seconds` for both `ComfyUIClient.__init__`'s construction of this transport and every `ensure_connected()` call — already present from Phase 1 (used for the HTTP `health_check()` startup wait too), reused as-is; this transport introduces no separate connect-timeout constant. | `60.0` |
| `COMFYUI_WEBSOCKET_TIMEOUT_SECONDS` | The `timeout_seconds` argument `_QueueTracker` passes into every `next_event()` call — the per-read timeout. | `5.0` |
| `COMFYUI_CONNECT_RETRY_ATTEMPTS`, `COMFYUI_CONNECT_RETRY_WAIT_MIN_SECONDS`, `COMFYUI_CONNECT_RETRY_WAIT_MAX_SECONDS` | Not read by this class directly — consumed by the Tenacity decorator one layer up (§6.1) that wraps calls to `ensure_connected()`. Listed here because they are the connect-retry policy *for* this transport, even though the constant is referenced at the call site, not inside the class body. | `3`, `2.0`, `10.0` |

No new constant is required beyond what the master document already reserves. `client_id` is **not** a config constant — it is generated per `ComfyUIClient` instance (`uuid4`) or supplied by the caller, per §3.1.

---

## 8. Logging

Sink, rotation, and format are unchanged from every other Module 7 component: the same `_configure_logger()` pattern already duplicated in `image_generator.py` and `workflow_library.py`, attaching to `MODULE7_LOG_PATH` (`rotation="10 MB"`, `retention="30 days"`, `enqueue=True`). `_ComfyUIWebSocketTransport` does not call `_configure_logger()` itself (it is already called once at `comfyui_client.py` import time, per the existing pattern in that file) — it only calls `logger.<level>(...)`.

| Level | Logged for |
|---|---|
| **INFO** | Successful connect (`ensure_connected()` completing the handshake) — includes `ws_url` (without the `clientId` query value at INFO, to keep INFO lines short; full URL at DEBUG) and elapsed handshake time. Explicit `close()` completing. |
| **WARNING** | Every malformed/invalid frame outcome from §4.5's list *except* an unrecognized `type` (which is DEBUG — see §4.3's rationale for why forward-compatible unknown event types are not warning-worthy). A connection-loss detected during a read (immediately before `ComfyUIConnectionError` is raised) — this is the log line `_QueueTracker` correlates with its own "falling back to HTTP polling" WARNING (master document §9), so this transport's log and `_QueueTracker`'s log together tell the full story of a mid-job WS drop. |
| **ERROR** | A handshake failure inside `ensure_connected()`, right before `ComfyUIConnectionError` is raised — includes `ws_url` and the elapsed time up to the timeout, but never the raw underlying exception object (only `str(exc)`, for the same pickling-safety reason as `_before_sleep_log`, §6 of the master document; a `websocket-client` connection exception can carry a non-picklable socket/traceback reference exactly like yt-dlp's `DownloadError` did in Module 2). |
| **DEBUG** | Every raw frame received (truncated preview, not the full payload for `execution_error` frames — see below), every timeout (`receive()` returning `None` because nothing arrived), every recognized-and-parsed `ComfyUIEvent` (by type and `prompt_id`, not full contents), every filtered event (`execution_start`, `executed`, unrecognized types, binary frames) by type only. |

**What is never logged, at any level:** the full raw `execution_error` payload dict (`error_payload`, §4.3) — only `_classify_comfyui_error` (a `ComfyUIClient`-level concern, master document §3.1 step 5 / §6) reads its contents; if that classification needs to log something about the error, it logs the extracted `exception_message` string, never the dict itself, matching the master document §9's explicit rule that ComfyUI's execution-error tracebacks are logged as string messages only. Full ComfyUI graph JSON is never sent over this transport in the first place (it travels one-way, over HTTP, in `submit_prompt`), so there is nothing graph-shaped for this class to accidentally log. Raw image bytes never reach this transport at all (§2.4).

---

## 9. Testing strategy

All tests below run in the default `pytest` invocation (no marker) — zero real network calls, zero real ComfyUI process, matching the master document §13.1's stated policy for the whole Phase 2 default suite. Location: `tests/test_comfyui_client.py::TestWebSocketTransport` (and a `TestWebSocketTransportStateMachine` class if the state-machine assertions in §9.4 warrant separating from the request/response-style tests in §9.1–9.3), mirroring the master document's proposed test-class layout in §13.1.

### 9.1 Fixture: fake local WebSocket server

A minimal, hand-rolled `threading`-based WebSocket server (no external test-only dependency beyond what `websocket-client` itself already needs) bound to `127.0.0.1` on an ephemeral port, capable of:
- Accepting a connection and performing a real WebSocket upgrade handshake (so the handshake path is exercised for real, not mocked away).
- Being scripted, per test, to send a specific ordered sequence of JSON text frames with configurable inter-frame delays (to exercise timeout behavior deterministically).
- Sending a single binary frame on command (to test binary-frame filtering, §4.3's last row).
- Closing the connection on command, either cleanly (close frame) or abruptly (raw socket close, to exercise both categories of "connection loss" §6.4 mentions).
- Refusing the connection entirely, or accepting-then-never-completing the upgrade (to exercise `ensure_connected()`'s timeout path, §3.2, without needing a real unreachable host).

This fixture is the single foundation nearly every test below is built on, and is explicitly designed to also be reusable, unmodified, by `_QueueTracker`'s own test suite (master document §13.1's `TestQueueTracker`) — the same scripted-frame-playback capability both test classes need.

### 9.2 `ensure_connected()` / connection lifecycle

- Successful connection against the fake server → `is_connected()` becomes `True`; state transitions `Disconnected → Connecting → Connected` (assert via a test-only state inspection hook, or indirectly via `is_connected()` plus behavior).
- Calling `ensure_connected()` again while already connected → no-op, no second handshake attempt (assert the fake server saw exactly one connection).
- Connection refused (fake server not listening / immediately closes on accept) → `ComfyUIConnectionError` raised, `connect_timeout_seconds` not exceeded on the wall clock (this should fail fast on an actual refusal, not wait out the full timeout — refusals and true hangs are different failure shapes worth asserting differently).
- Connection accepted but the WebSocket upgrade never completes → `ComfyUIConnectionError` raised only after approximately `connect_timeout_seconds` has elapsed (bounded-wait assertion, with a generous tolerance).
- `close()` then `ensure_connected()` again → succeeds, second real handshake occurs (assert the fake server saw two connections total) — proves the `Closed → Connecting → Connected` reconnection path (§5.2).
- `close()` called twice in a row → second call is a no-op, does not raise.
- `close()` called before any `ensure_connected()` call → no-op, does not raise (covers `Disconnected → Closed` implicitly, or rather, confirms `close()` is safe pre-connection).

### 9.3 `receive()` / `next_event()` — event parsing

For each recognized event type in §4.3's table (`status`, `executing` with a node, `executing` with `node: null`, `progress`, `execution_error`, `execution_cached`): script the fake server to send one well-formed frame of that shape, assert `next_event()` returns a `ComfyUIEvent` with exactly the expected field values, and assert nothing else on the dataclass is populated beyond what §4.3 specifies (e.g., a `status` event's `prompt_id` is `None`, not accidentally carried over from a prior event).

For each malformed-input case in §4.5: not-JSON text, JSON-but-not-an-object, missing `type`, recognized `type` with a missing/malformed required field (one dedicated test per required field per type listed in §4.3's validation column) — assert `next_event()` returns `None` and does not raise, and (where feasible with a log-capture fixture) assert a WARNING was logged.

Unrecognized-but-valid `type` (e.g., `execution_start`, `executed`, and a synthetic made-up future type) → assert `next_event()` returns `None` and logs at DEBUG, not WARNING (a regression guard for the WARNING/DEBUG distinction in §4.3).

Binary frame sent by the fake server → assert `receive()`/`next_event()` returns `None` without attempting `json.loads` on it (verifiable by ensuring no JSON-decode-error log line appears, only the binary-frame DEBUG line).

Read timeout (fake server configured to delay past `timeout_seconds`) → assert `next_event()` returns `None` after approximately `timeout_seconds` has elapsed, not immediately and not after a much longer wait — a bounded-wait assertion, matching the connection-timeout test style in §9.2.

### 9.4 Disconnect / reconnect / concurrency-guard

- Fake server closes the connection cleanly mid-wait (client is blocked inside `receive()`) → `ComfyUIConnectionError` raised from that in-flight call; state transitions to `Failed`; a subsequent `is_connected()` returns `False`.
- Fake server closes the connection abruptly (raw socket close, no close frame) → same assertion as above, covering the `ConnectionResetError`/`BrokenPipeError` code path distinctly from the clean-close `WebSocketConnectionClosedException` path (§6.4).
- After a detected disconnect, `ensure_connected()` called again against a still-listening fake server → succeeds, proving `Failed → Connecting → Connected` recovery (§5.2) — this is the unit-level counterpart to `_QueueTracker`'s own higher-level "WS-drop-then-recover" test (master document §3.4's testing strategy).
- (If the implementation adds any internal re-entrancy guard per §5.3's `Receiving` state note) a test asserting that calling `next_event()` from a second thread while one is already blocked in `receive()` on the first thread raises a clear internal error rather than corrupting socket state — included for defensive-coding value even though the class is documented single-threaded-only; this is the one test in this section that is about *catching a misuse bug early*, not about a supported usage pattern.

### 9.5 Resource cleanup

- `close()` while a `receive()` call is blocked on another thread (simulating the shutdown-during-active-read scenario `ComfyUIClient.close()` could trigger if called from a signal handler or a second thread in some future usage) — assert the blocked call unblocks promptly (via a `ComfyUIConnectionError` or a clean `None` return, whichever the underlying `websocket-client` behavior naturally produces) rather than hanging forever. This test exists specifically to guard against the "hung close" scenario §3.6 already commits to bounding.
- No lingering OS-level socket/file-descriptor leak across repeated `ensure_connected()` / `close()` cycles in a single test (a loop of, e.g., 20 connect/close cycles against the fake server, asserting the fake server's connection count matches exactly and the process's open-file-descriptor count does not grow unbounded) — a lightweight regression guard rather than a strict leak-detector.

### 9.6 What this test class explicitly does not need to cover

Anything belonging to `_QueueTracker`'s own state machine (deciding *when* a job is done, the polling fallback's HTTP-side behavior, ignoring other jobs' events) — those are `_QueueTracker`'s tests (master document §13.1's `TestQueueTracker`), which will *use* this fixture but assert different things on top of it. This document's test list is scoped strictly to what `_ComfyUIWebSocketTransport` itself is responsible for (§1.3's boundaries), keeping the two test classes as cleanly separated as the two production classes are.

---

## 10. Integration

### 10.1 With `_QueueTracker`

Already detailed in §2.3 above; summarized as a contract: `_QueueTracker` is constructed with a `_ComfyUIWebSocketTransport` instance (alongside a `_ComfyUIHTTPTransport` instance) and drives it exclusively through `ensure_connected()` and `next_event()`. `_QueueTracker` is the only code that interprets `ComfyUIEvent.event_type`/`node`/`prompt_id` values into lifecycle meaning; this transport is purely a typed, filtered, timeout-bounded frame source to it.

### 10.2 With `_OutputRetriever`

None (§2.4). Explicitly called out here again because it is the integration point most likely to be mistakenly assumed to exist (since `_OutputRetriever` also deals with "the job is done" data) — it does not, and should not, import or reference `_ComfyUIWebSocketTransport` at all.

### 10.3 With `_ComfyUIMetricsRecorder`

None directly. `_ComfyUIMetricsRecorder` records timing (`queue_time_seconds`, `generation_time_seconds`) that `_QueueTracker` measures *using* the timestamps of events this transport delivers (specifically: the first `status`/`executing` event marks the end of queue-wait, per the master document §10's field-mapping table), but the metrics recorder itself never touches this transport — it only ever receives already-computed timing values from `_QueueTracker`/`ComfyUIClient`. This keeps the dependency graph a strict chain (`_ComfyUIWebSocketTransport → _QueueTracker → ComfyUIClient → _ComfyUIMetricsRecorder`) rather than a web, matching every other Phase 2 component's single-direction dependency style (§3 of the master document).

### 10.4 With `ComfyUIClient.generate()`

Per the master document §3.1's seven-step internal workflow, this transport participates in exactly step 1 (`ensure_connected()`, called once per `generate()` invocation, no-op after the first) and, indirectly through `_QueueTracker`, step 3 (event consumption during `await_completion`). It has no role in steps 2 (HTTP submit), 4 (output retrieval), 5–6 (error classification — the classifier reads `ComfyUIEvent.error_payload` that already passed through this transport, but the classification logic itself lives in `ComfyUIClient`/a module-level helper, not here), or 7 (metrics recording). `ComfyUIClient.close()`/`__exit__` is this transport's only other integration point, via `close()` (§3.6, §2.5).

---

*End of design specification for `_ComfyUIWebSocketTransport`. No implementation code is included per the task's DESIGN ONLY constraint. This document is intended to be handed to the next Codex iteration alongside `docs/MODULE7_PHASE2_COMFYUI_INTEGRATION_DESIGN.md`; where the two overlap (§3.3, §4.2, §5, §6, §7, §8, §9, §13 of the master document), this document elaborates rather than supersedes, and any future edit to one should be checked against the other for drift, per the master document's own §16 step 16 convention.*
