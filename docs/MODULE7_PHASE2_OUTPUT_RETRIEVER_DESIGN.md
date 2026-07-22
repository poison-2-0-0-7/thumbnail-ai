# Module 7 Phase 2 — `_OutputRetriever` Design Specification

**Status:** Draft — implementation-ready
**Component:** `_OutputRetriever`
**Depends on (complete, do not modify):** `_ComfyUIHTTPTransport`, `_ComfyUIWebSocketTransport`, `ComfyUIEvent`, `_QueueTracker`, `_CompletionResult`
**Consumed by (not implemented here):** `ComfyUIClient.generate()`, `MetricsRecorder`

> **Note on source material:** This document was written from the interface names, responsibilities, and constraints supplied in the task brief, plus the standard ComfyUI `/history/{prompt_id}` REST schema. I did not have direct read access to `MODULE7_PHASE2_COMFYUI_INTEGRATION_DESIGN.md`, `MODULE7_PHASE2_WEBSOCKET_TRANSPORT_DESIGN.md`, or `MODULE7_PHASE2_QUEUE_TRACKER_DESIGN.md` when producing this. Before handing this to Codex, verify:
> - the exact field names/shape returned by `_CompletionResult` (assumed: `prompt_id: str`, `history: dict | None`, `status: str`, `error: str | None`)
> - whatever exception base classes already exist in the Module 7 hierarchy (assumed: `ComfyUIError` root, with `ComfyUITransportError`, `ComfyUITimeoutError` already defined by the HTTP/WS transports)
> - the existing constants module path (assumed: `modules.comfyui.constants` or similar, referred to below as `_constants`)
>
> Anywhere this document made an assumption to fill a gap, it is marked **[ASSUMED]**. Codex should treat those as defaults to confirm against the real codebase, not as fixed requirements.

---

## 1. Purpose

### 1.1 Why `_OutputRetriever` exists

`ComfyUIClient.generate()` orchestrates a full request: submit a prompt, wait for completion via `_QueueTracker`, and hand back a usable image. The "wait for completion" and "get me an image" halves of that job are architecturally distinct — one is about *queue state and timing*, the other is about *interpreting ComfyUI's output artifacts and fetching bytes*. `_OutputRetriever` is the second half, extracted into its own component so that:

- `_QueueTracker` stays focused purely on prompt lifecycle/state tracking and never touches image bytes, file formats, or the `/view` endpoint.
- `ComfyUIClient.generate()` stays a thin orchestrator that composes tracker + retriever, rather than embedding history-parsing and download logic inline.
- History parsing and image selection logic — which is genuinely fiddly (multiple output nodes, multiple images per node, format filtering) — has one owner and one test suite.

### 1.2 Responsibilities

`_OutputRetriever` is responsible for, and only for:

1. Accepting a completed `_CompletionResult` (or equivalent: a `prompt_id` plus its ComfyUI history payload).
2. Parsing the ComfyUI history structure to locate candidate output images.
3. Deterministically selecting **one** image when multiple candidates exist.
4. Downloading the selected image's bytes via `_ComfyUIHTTPTransport`.
5. Validating that the downloaded bytes constitute a usable image.
6. Returning a structured `_OutputResult` (or raising a Module 7 exception) to its caller.

### 1.3 Architectural boundaries — what it does NOT do

- **Does not** poll, subscribe to, or otherwise wait for completion. It assumes the prompt is *already* complete when invoked. Waiting is `_QueueTracker`'s job.
- **Does not** open or manage WebSocket connections. It has no dependency on `_ComfyUIWebSocketTransport` or `ComfyUIEvent` at all.
- **Does not** submit prompts to ComfyUI's `/prompt` endpoint. Submission is `ComfyUIClient`'s job.
- **Does not** record metrics/timings itself — it exposes enough information (via logging and return values / exceptions) for `MetricsRecorder` to do that from the outside. It has no dependency on `MetricsRecorder`.
- **Does not** perform any image *transformation* (resize, crop, format conversion, thumbnailing). Its validation step confirms the bytes decode as an image; it does not modify them. Downstream thumbnail-processing modules own transformation.
- **Does not** cache retrieved images across calls or manage a local filesystem cache. Each `retrieve()` call is a fresh fetch. (If caching becomes a requirement later, it belongs in a decorator/wrapper, not in this class.)
- **Does not** retry prompt submission or re-queue failed generations. Its retry scope is limited strictly to the HTTP download of an already-known image reference.

### 1.4 Separation from neighboring components

| Component | Owns | Does not own |
|---|---|---|
| `_QueueTracker` | Prompt lifecycle state, waiting/polling/WS-event correlation, producing `_CompletionResult` | Image bytes, history schema interpretation beyond "did it finish" |
| `_ComfyUIHTTPTransport` | Raw HTTP request/response mechanics (session, base URL, auth, low-level retry-on-connection-error) | Knowledge of ComfyUI's history/output JSON schema, image selection logic |
| `_ComfyUIWebSocketTransport` | Live event stream from ComfyUI | Anything about final output retrieval |
| `_OutputRetriever` (this doc) | History parsing, output/image selection, download orchestration, image validation | Waiting for completion, prompt submission, metrics recording, image transformation |
| `MetricsRecorder` (future) | Aggregating timings/counters emitted by other components | Any retrieval logic |
| `ComfyUIClient.generate()` | Wiring: submit → track → retrieve → return to caller | Any of the above internals |

---

## 2. Architecture

### 2.1 Dependency graph

```
ComfyUIClient
    │
    ├── uses ──> _QueueTracker ──> produces _CompletionResult
    │
    └── uses ──> _OutputRetriever
                       │
                       ├── uses ──> _ComfyUIHTTPTransport   (download only: GET /view, GET /history if needed)
                       └── consumes ──> _CompletionResult   (input, not owned/mutated)
```

- `_OutputRetriever` depends on `_ComfyUIHTTPTransport` and on the shape of `_CompletionResult`. It does **not** depend on `_QueueTracker`, `_ComfyUIWebSocketTransport`, `ComfyUIEvent`, `MetricsRecorder`, or `ComfyUIClient`.
- `ComfyUIClient` depends on `_OutputRetriever` (and separately on `_QueueTracker`), not the reverse.
- No component in this graph imports `ComfyUIClient`. This keeps the dependency direction strictly "leaf components ← client," with zero cycles.

### 2.2 Ownership

- `_OutputRetriever` owns a reference to a `_ComfyUIHTTPTransport` instance, injected at construction (constructor injection, not created internally). It does not own the transport's lifecycle (connection pool open/close) — that remains `ComfyUIClient`'s or the transport's own responsibility.
- `_OutputRetriever` holds no ComfyUI-connection state of its own beyond the injected transport reference and its own configuration (timeouts, retry counts). It is safe to construct once and reuse across many `retrieve()` calls.
- `_OutputRetriever` does not hold a reference to `_QueueTracker`. The caller (`ComfyUIClient.generate()`) is responsible for obtaining a `_CompletionResult` from the tracker first, then passing it (or its relevant fields) into `retrieve()`.

### 2.3 Data ownership of `_CompletionResult`

`_OutputRetriever` treats `_CompletionResult` as **read-only input**. It never mutates it and never constructs one. If `_CompletionResult` already embeds the raw history JSON **[ASSUMED: field `history: dict`]**, `_OutputRetriever` parses that embedded payload directly rather than re-fetching `/history/{prompt_id}` itself. If it does not embed the history payload, `_OutputRetriever` is responsible for issuing the `GET /history/{prompt_id}` call itself via `_ComfyUIHTTPTransport` — see §4, Step 2.

---

## 3. Public API

### 3.1 `__init__`

```
def __init__(
    self,
    transport: _ComfyUIHTTPTransport,
    *,
    preferred_output_nodes: Sequence[str] | None = None,
    allowed_image_formats: Sequence[str] | None = None,
    download_timeout: float | None = None,
    download_retries: int | None = None,
    download_retry_backoff: float | None = None,
) -> None
```

- **`transport`** (required, positional): an already-constructed `_ComfyUIHTTPTransport`. `_OutputRetriever` never constructs its own transport.
- **`preferred_output_nodes`** (keyword, optional): ordered list of ComfyUI node IDs/titles to prefer when multiple output nodes exist in history (see §6.4). Defaults to `_constants.DEFAULT_PREFERRED_OUTPUT_NODES` **[ASSUMED constant name]** if not supplied.
- **`allowed_image_formats`** (keyword, optional): lowercase file extensions without dots, e.g. `("png", "jpg", "jpeg", "webp")`. Defaults to `_constants.SUPPORTED_IMAGE_FORMATS`.
- **`download_timeout`**, **`download_retries`**, **`download_retry_backoff`**: override the module-level defaults (`_constants.OUTPUT_DOWNLOAD_TIMEOUT_SECONDS`, `_constants.OUTPUT_DOWNLOAD_MAX_RETRIES`, `_constants.OUTPUT_DOWNLOAD_RETRY_BACKOFF_SECONDS`) for this instance. `None` means "use the constant."

**Validation:** raises `ValueError` immediately if `transport` is `None`, or if any numeric override is negative. All other arguments are optional and independently defaulted — no cross-field validation needed.

**Thread-safety:** the constructor performs no I/O and mutates no shared/global state. Safe to call from any thread.

### 3.2 `retrieve()`

```
def retrieve(
    self,
    completion: _CompletionResult,
    *,
    prompt_id: str | None = None,
) -> _OutputResult
```

- **`completion`** (required): the `_CompletionResult` produced by `_QueueTracker` for a prompt that has already reached a terminal *successful* state. Passing a `_CompletionResult` whose status is not "success" **[ASSUMED status enum: `"success" | "failed" | "cancelled" | "timeout"`]** is a programmer error, not a retrieval-time condition — see §9.
- **`prompt_id`** (keyword, optional): explicit override of the prompt ID to use, for callers that want to decouple ID from the `completion` object's own ID field (e.g. testing). Defaults to `completion.prompt_id`.

**Returns:** `_OutputResult`, a small immutable data object:

```
@dataclass(frozen=True)
class _OutputResult:
    prompt_id: str
    output_node_id: str
    filename: str
    subfolder: str
    image_type: str          # ComfyUI's "type": "output" | "temp" | "input"
    format: str               # normalized lowercase extension, e.g. "png"
    content: bytes
    width: int | None         # populated during validation if decodable; None if format doesn't expose it cheaply
    height: int | None
```

**Raises** (all from the existing Module 7 exception hierarchy — see §9 for the full mapping):
- `OutputRetrievalError` (and subclasses) for any failure in history parsing, selection, download, or validation.
- `ValueError` if `completion` is `None`, or if `completion.status` is not a successful terminal state (programmer error, checked eagerly, not retried).

**Retry behaviour:** `retrieve()` itself does not retry at the whole-method level. Internally, only the HTTP *download* sub-step (§7) is retried, up to `download_retries` attempts, with backoff. History parsing and selection are pure/local and are not retried (a malformed history payload won't fix itself on retry).

**Timeout behaviour:** the overall `retrieve()` call has no separate wall-clock timeout of its own; its duration is bounded by the sum of the download attempts' individual `download_timeout` values. If the caller needs an overall deadline, that's applied by the caller wrapping the call (consistent with how `_QueueTracker` presumably applies its own polling timeout).

**Validation responsibilities:** `retrieve()` is responsible for validating (a) history shape, (b) selected output reference shape, (c) downloaded bytes. It is **not** responsible for validating that `completion.prompt_id` actually matches what ComfyUI thinks completed — that correlation is `_QueueTracker`'s job upstream.

**Thread-safety:** `_OutputRetriever` holds no mutable instance state that `retrieve()` writes to (no counters, no cache dict). Multiple threads may call `retrieve()` concurrently on the same instance, provided the injected `_ComfyUIHTTPTransport` is itself safe for concurrent use (assumed true, since `ComfyUIClient` presumably already shares it across concurrent generations). If the transport is *not* thread-safe, that is a pre-existing constraint of `_ComfyUIHTTPTransport`, not something `_OutputRetriever` introduces or must work around.

### 3.3 Helper methods (private, `_`-prefixed — documented for Codex's benefit, not part of the public contract)

```
def _extract_history_payload(self, completion: _CompletionResult) -> dict
```
Returns the raw history dict for the prompt, either from `completion.history` if present, or by calling `_ComfyUIHTTPTransport.get(f"/history/{prompt_id}")` **[ASSUMED transport method name/signature — confirm against actual `_ComfyUIHTTPTransport` public API]**. Raises `OutputHistoryError` on missing/malformed payload (§5).

```
def _collect_candidate_images(self, history_payload: dict, prompt_id: str) -> list[_ImageCandidate]
```
Walks `history_payload[prompt_id]["outputs"]`, producing a flat, order-stable list of candidate images across all output nodes. Raises `OutputHistoryError` if no `outputs` key exists or it is not a dict.

```
def _select_image(self, candidates: list[_ImageCandidate]) -> _ImageCandidate
```
Applies the deterministic selection algorithm (§6). Raises `NoOutputImageError` if `candidates` is empty after format filtering.

```
def _download_image(self, candidate: _ImageCandidate) -> bytes
```
Calls the HTTP transport's `/view` endpoint with retry/backoff (§7). Raises `OutputDownloadError` / subclasses on exhaustion.

```
def _validate_image_bytes(self, content: bytes, candidate: _ImageCandidate) -> tuple[int | None, int | None]
```
Confirms non-empty, confirms decodability, extracts `(width, height)` when cheaply available. Raises `CorruptImageError` / `UnsupportedImageFormatError` (§8).

Internal `_ImageCandidate` is a private dataclass: `{output_node_id, filename, subfolder, image_type, format}` — it is the pre-download counterpart of `_OutputResult` and is not exposed publicly.

---

## 4. Retrieval Workflow

`retrieve()` executes these steps strictly in order, failing fast on the first error:

```
1. Validate `completion` (status must be terminal-success; prompt_id resolvable)
        │
        ▼
2. Obtain history payload
   (from completion.history, else GET /history/{prompt_id})
        │
        ▼
3. Locate `outputs` block in history[prompt_id]
        │
        ▼
4. Collect all image candidates across all output nodes
        │
        ▼
5. Filter candidates to allowed_image_formats
        │
        ▼
6. Deterministically select ONE candidate
        │
        ▼
7. Download image bytes via HTTP transport (/view), with retry
        │
        ▼
8. Validate downloaded bytes (non-empty, decodable, format matches)
        │
        ▼
9. Construct and return _OutputResult
```

Each numbered step corresponds to a private helper in §3.3 except steps 1 and 9, which live directly in `retrieve()`. Every step raises a specific exception subtype on failure (§9) rather than a generic one, so callers/tests can discriminate failure modes.

---

## 5. History Parsing

### 5.1 Expected schema

ComfyUI's `/history/{prompt_id}` response (and, by extension, whatever `completion.history` embeds) has the shape:

```json
{
  "<prompt_id>": {
    "prompt": [...],
    "outputs": {
      "<node_id>": {
        "images": [
          {"filename": "ComfyUI_00001_.png", "subfolder": "", "type": "output"}
        ]
      },
      "<other_node_id>": { "...": "..." }
    },
    "status": {"status_str": "success", "completed": true, "messages": [...]}
  }
}
```

### 5.2 Required fields

For `_OutputRetriever` to proceed, the payload must have:
- Top-level key equal to the resolved `prompt_id`.
- Under that key, an `outputs` dict.
- At least one entry under `outputs` that itself contains an `images` list with at least one dict having non-empty `filename`.

`subfolder` and `type` are read defensively: if absent, `subfolder` defaults to `""` and `type` defaults to `"output"` **[ASSUMED — matches ComfyUI's own default behavior]**, rather than treating their absence as an error. Their absence is logged at `DEBUG`, not treated as malformed.

### 5.3 Missing keys / malformed history

| Condition | Handling |
|---|---|
| Top-level `prompt_id` key missing entirely from the payload | Raise `OutputHistoryError` — "prompt not found in history" |
| `outputs` key missing or not a dict | Raise `OutputHistoryError` — "history payload missing outputs" |
| `outputs` present but empty dict | Raise `NoOutputImageError` — "no output nodes produced any output" |
| A node entry under `outputs` is not a dict, or has no `images` key | That node is **skipped** (logged at `DEBUG`), not fatal — other nodes may still have valid images |
| `images` present but not a list | That node is **skipped** (logged at `WARNING` — this is more surprising than a missing key) |
| An individual image entry missing `filename` or with empty `filename` | That single image entry is **skipped** (logged at `DEBUG`) |
| History payload is not a dict at all (e.g. `None`, string, list) | Raise `OutputHistoryError` — "history payload malformed" |

The rule of thumb: structural problems at the *payload* or *outputs-block* level are fatal (`OutputHistoryError`); problems scoped to a *single node* or *single image entry* are non-fatal skips, because ComfyUI workflows commonly have auxiliary nodes (previews, masks, intermediate saves) that are not the intended thumbnail output and should not abort retrieval.

### 5.4 Multiple output nodes / missing outputs / unsupported outputs

- **Multiple output nodes with images:** all are collected as candidates (§6); selection happens in a separate, deterministic step — history parsing itself does not pick a winner.
- **Missing outputs** (an output node referenced by the workflow graph never appears under `outputs`, e.g. because it was bypassed): not an error at parse time — `_OutputRetriever` only looks at what history actually reports, never at the submitted workflow graph. It has no visibility into the graph and does not attempt to reconcile "expected nodes" vs "actual nodes."
- **Unsupported outputs** (video, latent-only, or other non-image output types that ComfyUI may report under `outputs` in other list keys such as `"gifs"` or `"files"`): `_OutputRetriever` only reads the `images` key of each output node. Other keys are ignored entirely — not logged as errors, since they're simply out of scope, not malformed. **[ASSUMED — if Module 7 needs video/gif thumbnail support later, that is a new sibling method or a new component, not a silent extension of this one.]**

---

## 6. Image Selection

Selection must be a pure function of `(candidates)` — same input, same output, every time, with no reliance on dict-iteration order beyond what Python already guarantees (insertion order, which itself is driven by the deterministic order the history payload was parsed in).

### 6.1 Candidate collection order

Candidates are collected by iterating `history[prompt_id]["outputs"].items()` in the order the payload dict presents them (Python dicts preserve insertion order; the JSON parser preserves the source document's key order), and within each node, `images` in list order. This gives a stable, deterministic candidate list for a given history payload.

### 6.2 Format filtering

Before selection, candidates are filtered to those whose filename extension (lowercased, no leading dot) is in `allowed_image_formats`. Extension is derived from `filename.rsplit(".", 1)[-1]` when a `.` is present; candidates with no extension are dropped. This filtering happens *before* node-preference logic so that an unsupported-format image on a "preferred" node doesn't win over a supported-format image on a non-preferred node.

### 6.3 Selection algorithm

```
selected = None
for preferred_node_id in preferred_output_nodes:
    matches = [c for c in filtered_candidates if c.output_node_id == preferred_node_id]
    if matches:
        selected = matches[0]   # first image on the first matching preferred node
        break

if selected is None:
    if filtered_candidates:
        selected = filtered_candidates[0]   # first candidate in collection order
    else:
        raise NoOutputImageError(...)
```

In words:
1. If `preferred_output_nodes` is configured, try each preferred node ID in the order given; the first preferred node that has at least one surviving candidate wins, and its **first** image (list order) is selected.
2. If no preferred node matched (either the list is empty, or none of the preferred IDs appear in this history), fall back to the very first candidate in overall collection order (§6.1).
3. If there are no candidates left after format filtering, raise `NoOutputImageError`.

### 6.4 Preferred output node matching

`preferred_output_nodes` entries are matched against `output_node_id` (the ComfyUI node ID string, e.g. `"9"`) exactly, as strings. **[ASSUMED — if the underlying workflow JSON gives nodes human-readable titles that ComfyUI surfaces elsewhere in history, matching against title is out of scope for this version; node ID is the only stable identifier guaranteed to be present.]** This keeps matching simple and avoids depending on workflow-authoring conventions this component has no visibility into.

### 6.5 Duplicate images

If the same `(filename, subfolder, type)` tuple appears more than once across candidates (which can legitimately happen if a node lists the same image twice, or two differently-configured nodes happen to save under the same filename), no de-duplication occurs before selection — the first occurrence in collection order is what gets selected per §6.3, and later duplicates are simply never reached. No special-casing is needed because selection already stops at the first winner.

### 6.6 Determinism guarantee

Given an identical history payload and identical configuration (`preferred_output_nodes`, `allowed_image_formats`), `_select_image` always returns the same candidate. This is guaranteed because: (a) collection order is derived purely from the payload's own structure, (b) filtering is a pure predicate, (c) the preferred-node loop is a deterministic linear scan, (d) the fallback is "first in list." No randomness, no wall-clock, no set-based (unordered) operations are used anywhere in the selection path.

---

## 7. Image Download

### 7.1 Interaction with HTTP Transport

`_OutputRetriever` calls a single existing method on `_ComfyUIHTTPTransport` to fetch image bytes from ComfyUI's `/view` endpoint, passing `filename`, `subfolder`, and `type` as query parameters, e.g. conceptually:

```
response_bytes = self._transport.get_binary(
    "/view",
    params={"filename": candidate.filename, "subfolder": candidate.subfolder, "type": candidate.image_type},
    timeout=self._download_timeout,
)
```

**[ASSUMED method name `get_binary` / parameter shape — `_OutputRetriever` must call whatever binary-fetch method `_ComfyUIHTTPTransport` already exposes; if it only exposes a JSON-decoding `get()`, Codex must use the transport's raw/bytes variant instead, or flag that transport needs a small compatible addition (a new method, not a redesign, if one doesn't already exist).]**

`_OutputRetriever` does not construct URLs, manage the base host/port, or manage auth headers — all of that is transport-owned. It supplies only the ComfyUI-domain-specific query parameters.

### 7.2 Download workflow

1. Build query parameters from the selected `_ImageCandidate`.
2. Call the transport's binary-fetch method with `download_timeout`.
3. On success: pass bytes to validation (§8).
4. On transport-level exception (connection error, non-2xx HTTP status, timeout): treat as a retryable failure, up to `download_retries` additional attempts, unless the transport raises a class already known to be non-retryable (e.g. a 404 indicating the file plainly doesn't exist — see §7.4).

### 7.3 Retry strategy

- Retries apply **only** to the download step, not to history parsing or selection.
- Backoff: fixed or exponential per `download_retry_backoff` **[ASSUMED: exponential, `backoff * (2 ** attempt_index)`, capped at a reasonable ceiling like `_constants.OUTPUT_DOWNLOAD_RETRY_BACKOFF_MAX_SECONDS` — confirm this matches whatever pattern `_QueueTracker`/HTTP transport already use elsewhere in the module, and reuse that pattern rather than inventing a new one]**.
- `download_retries` counts *additional* attempts after the first; e.g. `download_retries=2` means up to 3 total attempts.
- Each attempt is logged (§11).
- If all attempts are exhausted, raise `OutputDownloadError` wrapping the last underlying exception (`raise OutputDownloadError(...) from last_exc`).

### 7.4 Non-retryable failures

An HTTP 404 (file genuinely not present at that filename/subfolder/type) is treated as **non-retryable** — retrying will not make ComfyUI produce a file it already reported as an output but that isn't actually on disk. This raises `OutputDownloadError` immediately (specifically a `MissingOutputFileError` subclass, §9) without consuming retry attempts, since the failure is deterministic, not transient.

Connection errors, timeouts, and 5xx responses are treated as transient/retryable, consistent with how `_ComfyUIHTTPTransport` presumably already classifies its own errors elsewhere **[ASSUMED — confirm the transport's existing exception taxonomy already distinguishes retryable vs non-retryable, and reuse that classification rather than re-implementing it here]**.

### 7.5 Timeout behaviour

Each individual download attempt is bounded by `download_timeout` (passed straight to the transport call). There is no separate outer timeout across all retry attempts in this component; the practical upper bound is `download_timeout * (download_retries + 1)` plus backoff sleep time, which is an acceptable, explicit cost of the retry policy.

### 7.6 Corrupt downloads / empty responses

- A response with `len(content) == 0` is **not** passed to image validation — it is treated as a download-layer failure (empty payload), logged, and retried like any other transient failure (server may have returned an incomplete write). If retries are exhausted with persistent empty responses, raise `OutputDownloadError`.
- A response with non-zero length that nonetheless fails to decode as an image is **not** a download failure — it progresses past download and fails at the validation step instead (§8), because "bytes arrived but aren't a valid image" is a data-integrity concern, distinct from "bytes didn't arrive."

---

## 8. Validation

Validation happens once, after a successful download, before constructing `_OutputResult`.

| Rule | Check | Failure |
|---|---|---|
| Image exists | Non-`None`, non-empty `bytes` object | `CorruptImageError` — "empty image payload" (should already be caught at download layer per §7.6, but checked here too as a defensive invariant) |
| Non-empty file | `len(content) > 0` | Same as above |
| Supported format | File's magic bytes/signature match a known image format, or the format asserted by the filename extension is one of `allowed_image_formats` | `UnsupportedImageFormatError` |
| Readable image | Bytes can be opened/decoded by the project's existing image-decoding library (e.g. Pillow, if already a dependency elsewhere in the codebase) without raising | `CorruptImageError` — "image failed to decode" |
| Dimensions available | `(width, height)` extracted from the decoded image when decoding succeeded | Not independently fatal — if decode succeeded, dimensions should always be available; if the decode library can't report them for some exotic format, `width`/`height` are set to `None` on `_OutputResult` rather than failing retrieval outright |
| Metadata validation | Not required for this component. `_OutputRetriever` does not parse or validate embedded PNG workflow-metadata (ComfyUI embeds the generating workflow in PNG tEXt chunks) — that is out of scope; downstream consumers who need it can read `_OutputResult.content` themselves | N/A |

**[ASSUMED image-decoding dependency:** the design assumes the codebase already has an image-decoding library available (most likely Pillow, given `yolo11n.pt` in the repo root suggests a vision-processing stack is already present). If none is currently a dependency, introducing one is a config/dependency decision for `ComfyUIClient`'s broader design, not something `_OutputRetriever`'s design should silently assume — flag this before implementation.]

**When retrieval fails due to validation:** the exception propagates out of `retrieve()` immediately; no fallback to a "next-best candidate" is attempted. If the single selected/downloaded image is corrupt, that is surfaced as a hard failure to the caller (`ComfyUIClient.generate()`), which is free to decide whether to retry the *entire generation* — that policy decision belongs to the client, not to this component (see §13).

---

## 9. Error Handling

### 9.1 Exception hierarchy

Reusing the existing Module 7 root, `_OutputRetriever` introduces a small subtree under it **[ASSUMED root name `ComfyUIError` — substitute the actual root class]**:

```
ComfyUIError                              (existing root)
└── OutputRetrievalError                  (new — base for everything this component raises)
    ├── OutputHistoryError                (history payload missing/malformed structurally)
    ├── NoOutputImageError                (parsed successfully, but zero usable image candidates)
    ├── OutputDownloadError               (download step failed after retries exhausted)
    │   └── MissingOutputFileError        (non-retryable 404 — file genuinely absent)
    ├── CorruptImageError                 (bytes downloaded but fail to decode / are empty at validation time)
    └── UnsupportedImageFormatError       (bytes decode fine, but format is not in allowed_image_formats)
```

All new exception classes accept and forward `prompt_id` and, where relevant, `output_node_id`/`filename`, as structured attributes (not just message text), so `MetricsRecorder` and logging can key off them without string-parsing.

### 9.2 Handling table

| Failure | Raised as | Retried? | Propagates to caller? |
|---|---|---|---|
| Malformed history (§5.3, payload-level) | `OutputHistoryError` | No | Yes |
| Missing outputs entirely | `OutputHistoryError` | No | Yes |
| No candidates after filtering | `NoOutputImageError` | No | Yes |
| Transient download failure (connection/timeout/5xx) | (internal retry loop) then `OutputDownloadError` if exhausted | Yes, up to `download_retries` | Yes, after exhaustion |
| 404 on download | `MissingOutputFileError` | No | Yes, immediately |
| HTTP transport error not otherwise classified | Wrapped as `OutputDownloadError` (`raise ... from original`) | Follows same policy as transient failures unless the transport's own exception type is already known non-retryable | Yes |
| Corrupt/undecodable image | `CorruptImageError` | No | Yes |
| Format not allowed (post-decode) | `UnsupportedImageFormatError` | No | Yes |
| `completion` is `None` or non-terminal-success status | `ValueError` | No (programmer error) | Yes |

### 9.3 What does NOT propagate as `OutputRetrievalError`

Any exception raised by `_ComfyUIHTTPTransport` that is already part of the existing transport exception hierarchy (e.g. an auth failure, a malformed base-URL configuration error) is **not** re-wrapped if it's clearly a transport/configuration problem rather than an output-retrieval-domain problem — it propagates as-is. Only failures that are meaningfully about *this component's* job (parsing, selecting, validating, or the retry-exhausted download outcome) get wrapped in the new `OutputRetrievalError` subtree. This avoids masking transport-layer problems as if they were output-layer problems.

---

## 10. Configuration

All configuration is sourced from existing project constants where possible; new constants are introduced only where no equivalent already exists.

| Name | Purpose | Source |
|---|---|---|
| `SUPPORTED_IMAGE_FORMATS` | Default `allowed_image_formats` | **Reuse if it already exists** (likely already defined for thumbnail output elsewhere in the codebase, given the project is thumbnail-focused); else new constant, e.g. `("png", "jpg", "jpeg", "webp")` |
| `DEFAULT_PREFERRED_OUTPUT_NODES` | Default `preferred_output_nodes` | New constant if none exists; default to `()` (empty — pure fallback-to-first-candidate behavior) unless the workflow JSON convention used elsewhere in the project already designates a canonical "final output" node ID/title convention worth hardcoding |
| `OUTPUT_DOWNLOAD_TIMEOUT_SECONDS` | Default `download_timeout` | Reuse existing HTTP timeout constant if `_ComfyUIHTTPTransport` already defines one; else new, e.g. `30.0` |
| `OUTPUT_DOWNLOAD_MAX_RETRIES` | Default `download_retries` | Reuse existing retry-count constant if `_QueueTracker`/transport already define one (for consistency of retry philosophy across the module); else new, e.g. `3` |
| `OUTPUT_DOWNLOAD_RETRY_BACKOFF_SECONDS` | Default `download_retry_backoff` | Reuse existing backoff constant if present; else new, e.g. `1.0` |

New constants, if introduced, should live alongside whatever existing constants module the HTTP/WS transports and `_QueueTracker` already use, not in a new file — keeping all Module 7 tunables in one place.

---

## 11. Logging

Follow the existing project-wide Loguru conventions (structured `logger.bind(...)` context, consistent level usage). Every log line should be bound with at least `prompt_id`, and `output_node_id`/`filename` once known.

| Event | Level | Example message shape |
|---|---|---|
| Retrieval started | `INFO` | `"Starting output retrieval"` bound with `prompt_id` |
| History payload obtained (from `completion` vs fetched) | `DEBUG` | `"History payload source: {embedded\|fetched}"` |
| Node/image skipped during parsing (non-fatal) | `DEBUG` or `WARNING` per §5.3 | `"Skipping output node with no images"` bound with `output_node_id` |
| Selected output node | `INFO` | `"Selected output node"` bound with `output_node_id`, `via` = `"preferred"` or `"fallback"` |
| Selected image | `INFO` | `"Selected image"` bound with `filename`, `subfolder`, `format` |
| Download attempt started | `DEBUG` | `"Downloading image"` bound with `attempt`, `max_attempts` |
| Download success | `INFO` | `"Image downloaded"` bound with `filename`, `bytes=len(content)` |
| Download failure (single attempt, will retry) | `WARNING` | `"Download attempt failed, retrying"` bound with `attempt`, `error` |
| Download failure (retries exhausted) | `ERROR` | `"Download failed after {n} attempts"` |
| Non-retryable download failure (404) | `ERROR` | `"Output file missing on server"` |
| Validation failure | `ERROR` | `"Image validation failed"` bound with `reason` |
| Retrieval succeeded (overall) | `INFO` | `"Output retrieval complete"` bound with `filename`, `width`, `height` |

No image bytes, and no full history payload, are ever logged — only metadata (filenames, node IDs, sizes, counts) — to keep logs bounded and avoid dumping large binary/JSON blobs into log storage.

---

## 12. Testing Strategy

All tests use a mocked `_ComfyUIHTTPTransport` (no real ComfyUI instance) and hand-constructed fake history payload dicts / fake `_CompletionResult` instances. Suggested `pytest` structure: `tests/module7/test_output_retriever.py`, organized into classes mirroring the sections below.

### 12.1 Happy-path retrieval
- Single output node, single image → returns expected `_OutputResult`.
- History embedded in `completion.history` is used directly (transport's history-fetch method is **not** called).
- History not embedded → transport's history-fetch method **is** called exactly once with the right `prompt_id`.

### 12.2 Selection behavior
- Single output node → that image selected.
- Multiple output nodes, no `preferred_output_nodes` configured → first candidate in collection order selected.
- Multiple output nodes, `preferred_output_nodes` configured and one matches → preferred node's first image selected, even though it's not first in raw collection order.
- Multiple output nodes, `preferred_output_nodes` configured but none match any present node → falls back to first-in-collection-order.
- Multiple images on the same node → first image in the node's `images` list selected.
- Mixed formats where the first candidate is an unsupported format → filtered out, next candidate selected.
- Determinism: calling `_select_image` twice with an identical candidate list produces an identical result (guards against any accidental reliance on set/dict iteration nondeterminism).

### 12.3 History error handling
- `completion.history` missing and transport's history-fetch call raises/returns malformed data → `OutputHistoryError`.
- Payload missing the `prompt_id` key → `OutputHistoryError`.
- Payload's `outputs` key missing → `OutputHistoryError`.
- `outputs` present but empty dict → `NoOutputImageError`.
- A node with non-dict value → skipped, not fatal (retrieval still succeeds if another node has valid images).
- A node with `images` not a list → skipped with `WARNING` log, not fatal.
- An image entry missing `filename` → skipped, not fatal.
- History payload is not a dict at all (`None`, `"garbage"`, `[]`) → `OutputHistoryError`.

### 12.4 Missing / unsupported outputs
- All candidates filtered out by `allowed_image_formats` → `NoOutputImageError`.
- Output node present with only non-image keys (e.g. only `"gifs"`, no `"images"`) → treated as no candidates from that node.

### 12.5 Download behavior
- Successful download on first attempt → no retry logging, single transport call.
- Transient failure then success → retried the expected number of times, eventual success returned, each attempt logged.
- Transient failure exhausting all retries → `OutputDownloadError` raised, transport called exactly `download_retries + 1` times.
- 404 response → `MissingOutputFileError` raised immediately, transport called exactly once (no retry).
- Empty byte response → treated as retryable failure per §7.6, eventually `OutputDownloadError` if it never recovers.

### 12.6 Validation
- Valid PNG/JPEG/WebP bytes (small fixture images) → `_OutputResult.width`/`height` populated correctly.
- Empty bytes reaching validation directly (defensive path) → `CorruptImageError`.
- Garbage bytes that don't decode as any image → `CorruptImageError`.
- Bytes that decode but whose format isn't in `allowed_image_formats` → `UnsupportedImageFormatError`.

### 12.7 Determinism across repeated calls
- Given the same fake history payload and same config, two separate `retrieve()` calls (with fresh mock transports returning identical bytes) produce `_OutputResult`s equal in every field except none that would legitimately vary (there are none — no timestamps in `_OutputResult`).

### 12.8 Logging behavior
- Assert (via caplog / Loguru sink capture) that key events fire at the expected levels: retrieval started, image selected, download success, and — separately — that a validation failure logs at `ERROR` and a retried download logs at `WARNING` per attempt.
- Assert that no test triggers a log line containing raw image bytes or the full history dict (guard against accidental verbose logging regressions).

### 12.9 Programmer-error guards
- `retrieve(None)` → `ValueError`.
- `retrieve(completion)` where `completion.status` is not the successful terminal value → `ValueError`.
- `_OutputRetriever(transport=None)` → `ValueError`.

---

## 13. Integration

### 13.1 With `_QueueTracker`

`ComfyUIClient.generate()` is expected to call `_QueueTracker` first to obtain a terminal `_CompletionResult`, then pass that result into `_OutputRetriever.retrieve()`. `_OutputRetriever` never talks to `_QueueTracker` directly and has no import of it. The hand-off is a single data object (`_CompletionResult`) — the two components are otherwise fully decoupled, which is what makes each independently testable.

### 13.2 With `ComfyUIClient.generate()`

Expected (not implemented here) call shape inside `generate()`:

```
completion = self._queue_tracker.wait_for_completion(prompt_id)   # existing, complete
if completion.status != "success":
    # existing error handling for failed/cancelled/timeout generations
    ...
output = self._output_retriever.retrieve(completion)
return output   # or a further-transformed object built from output.content
```

`ComfyUIClient` owns the decision of what to do when `retrieve()` raises — e.g. whether a `MissingOutputFileError` should trigger a whole-generation retry (re-submit the prompt) versus surfacing straight to the caller. That policy is out of scope for `_OutputRetriever`, which only reports *why* retrieval failed, not what to do about it.

### 13.3 With `MetricsRecorder` (future)

`_OutputRetriever` does not call `MetricsRecorder` directly (no dependency, per §1.3/§2.1). The expected integration pattern is that `ComfyUIClient` (or a thin decorator around `_OutputRetriever`) wraps calls to `retrieve()` and records:
- retrieval duration (wall-clock around the whole `retrieve()` call),
- outcome (success / which exception subtype on failure),
- selected image format and byte size on success,
- number of download attempts consumed (derivable from log records, or — if a cleaner integration is wanted later — `_OutputResult` could be extended with a non-breaking optional field such as `download_attempts: int` in a future revision; not part of this version's contract).

This keeps `_OutputRetriever` metrics-agnostic today while leaving an obvious, low-friction seam for `MetricsRecorder` to attach to later without requiring a redesign.

---

## 14. Summary of new public surface

For quick reference when implementing:

- `class _OutputRetriever`
  - `__init__(transport, *, preferred_output_nodes=None, allowed_image_formats=None, download_timeout=None, download_retries=None, download_retry_backoff=None)`
  - `retrieve(completion, *, prompt_id=None) -> _OutputResult`
- `@dataclass(frozen=True) class _OutputResult` — `prompt_id, output_node_id, filename, subfolder, image_type, format, content, width, height`
- New exceptions: `OutputRetrievalError`, `OutputHistoryError`, `NoOutputImageError`, `OutputDownloadError`, `MissingOutputFileError`, `CorruptImageError`, `UnsupportedImageFormatError` (all under existing `ComfyUIError` root).

No existing public interface (`_ComfyUIHTTPTransport`, `_ComfyUIWebSocketTransport`, `ComfyUIEvent`, `_QueueTracker`, `_CompletionResult`) is modified by this design.
