# MODULE7_PHASE2_GENERATE_DESIGN.md

**Module 7 — Phase 2: ComfyUI Integration**
**Component: `ComfyUIClient.generate()`**
**thumbnail-ai**

Status: **Design specification, v1.0. No implementation.**

Source of truth: `modules/comfyui_client.py` as it exists today (all of Sprint 1 and
Sprint 2 — `_ComfyUIHTTPTransport`, `_ComfyUIWebSocketTransport`, `_QueueTracker`,
`_OutputRetriever`, `_ComfyUIMetricsRecorder`, `_CompletionResult`, `_OutputResult`,
`SystemStats`, `ComfyUIEvent`, `_ComfyUIHTTPError`, all complete and unmodified by
this document), `modules/image_generator.py` (`BuiltWorkflow`, `WorkflowBuilder`,
`MetricsCollector`, Phase 1, complete and unmodified), `modules/module7_exceptions.py`,
`modules/config.py`, `modules/models.py`, and the four prior Phase 2 sub-documents
(`MODULE7_PHASE2_COMFYUI_INTEGRATION_DESIGN.md`, `..._WEBSOCKET_TRANSPORT_DESIGN.md`,
`..._QUEUE_TRACKER_DESIGN.md`, `..._METRICS_RECORDER_DESIGN.md`).

Assumed complete and unmodified by this document: every one of the six ✅ components
listed under Module 7 Phase 1/Phase 2 in the task brief, plus `BuiltWorkflow` /
`WorkflowBuilder` / `MetricsCollector` (Phase 1, `image_generator.py`), plus the
existing test suites for all of the above.

---

## 0. Reconciliation note — where this document departs from earlier drafts

`MODULE7_PHASE2_COMFYUI_INTEGRATION_DESIGN.md` ("the master document") was written
*before* Phase 2 implementation began, and its §3.1 sketch of `ComfyUIClient.generate()`
predates the actual shape of `_QueueTracker`, `_OutputRetriever`, and
`_ComfyUIMetricsRecorder` as they were ultimately built. Per this task's instruction to
treat the repository as the source of truth, this document supersedes the master
document's §3.1/§6 wherever the two disagree. The concrete deltas, each verified
against `modules/comfyui_client.py` directly:

| Master document assumed | Repository actually has | Effect on this design |
|---|---|---|
| `ComfyUIClient.generate()` wrapped in two Tenacity retry layers (connection-level, OOM-level), config constant `MODULE7_COMFYUI_OOM_RETRY_ATTEMPTS` | No `tenacity` import anywhere in `comfyui_client.py`; no such config constant exists in `config.py`; every existing retry in the file (`_OutputRetriever._download_image`, `_QueueTracker`'s reconnect/poll loop) is a hand-rolled `time.sleep` loop | §4 below documents that `generate()` implements **no retry loop of its own** — see §4 for the full rationale |
| Return type `RawGenerationOutput` | No such type exists anywhere in the codebase. `_OutputRetriever.retrieve()` already returns `_OutputResult`, a frozen dataclass with exactly the fields a raw-candidate return value needs | §2 below specifies `generate()` returns `_OutputResult` directly rather than inventing a new, functionally-identical public type |
| `generate(built_workflow, candidate_index=0, *, stage_output_dir: Path)` | `_OutputRetriever.retrieve()` never writes to disk — it returns in-memory `content: bytes`. No component in Phase 1 or Phase 2 owns disk persistence of a generated candidate | §2 drops `stage_output_dir` from the signature; persisting `_OutputResult.content` to disk is out of `generate()`'s scope (see §1) |
| Exception `ComfyUITimeoutError` | Not defined in `module7_exceptions.py` | §5 flags this as a required, minimal, pre-implementation addition — see §5's **[ASSUMED — NEW]** entry |
| `_ComfyUIMetricsRecorder.record_attempt(video_id, niche, workflow_ref, outcome)` (4-arg sketch) | `record_attempt(*, video_id, niche, workflow_version, profile_name=None, workflow_hash=None, completions, output=None, exception=None, num_candidates_requested=1, identity_retry_count=0, peak_vram_mb=None, gpu_utilization_percent=None, attempt_started_at=None)` | §6 is written against the real signature |
| "`generate()` catches `MetricsWriteError`" | `_ComfyUIMetricsRecorder.record_attempt()` already wraps its own `GenerationMetrics` construction *and* `self._collector.append(...)` in a single `try/except Exception: logger.error(...)` — it **never raises** | §6 documents that `generate()` needs no exception handling around the `record_attempt()` call; it is already fail-safe by construction |

Every other interface referenced below (method names, parameter names, field names,
config constant names) was read directly from `modules/comfyui_client.py`,
`modules/module7_exceptions.py`, `modules/config.py`, and `modules/image_generator.py`
in this session, not inferred from the master document.

---

## 1. Responsibilities

### 1.1 What `generate()` owns

`generate()` is the single orchestrating method that turns one already-materialized
`BuiltWorkflow` into one retrieved candidate image, by driving the five completed
collaborators through exactly one submit → track → retrieve → record cycle:

1. Ensuring the WebSocket transport is connected before submission.
2. Submitting the workflow graph via `_ComfyUIHTTPTransport.submit_prompt()`.
3. Handing the resulting `prompt_id` to `_QueueTracker.await_completion()` and
   waiting for a terminal `_CompletionResult`.
4. Classifying an `EXECUTION_ERROR` outcome's `error_payload` into either
   `VRAMExhaustedError` or `ComfyUIQueueError` (this classification is not owned by
   any existing component — see §1.3 and §10).
5. On a `COMPLETED` outcome, calling `_OutputRetriever.retrieve()` to obtain the
   downloaded, validated image.
6. On a `TIMEOUT` outcome, best-effort cancelling the prompt on ComfyUI and raising a
   typed timeout exception.
7. Calling `_ComfyUIMetricsRecorder.record_attempt()` exactly once per `generate()`
   call, unconditionally, regardless of which of the above paths was taken.
8. Translating every transport-internal failure (`_ComfyUIHTTPError`) it observes
   directly (i.e., not already translated by a collaborator) into the appropriate
   public `Module7Error` subclass before it can leave `generate()`.
9. Returning the collaborators' own result value (`_OutputResult`) unmodified on
   success, or raising a typed exception on every failure path.

### 1.2 What `generate()` does not own

- **Connection lifecycle beyond "ensure connected."** Opening the HTTP session and
  closing either transport is `ComfyUIClient.__init__`/`close`/`__exit__` — out of
  scope for this document, per the task brief.
- **Deciding *when* the WebSocket vs. HTTP-polling fallback happens.** That state
  machine lives entirely inside `_QueueTracker.await_completion()` (§6 of
  `MODULE7_PHASE2_QUEUE_TRACKER_DESIGN.md`); `generate()` calls
  `await_completion()` exactly once and receives one terminal answer.
- **Selecting which output image/node wins when a workflow has multiple output
  nodes.** That is `_OutputRetriever._select_image()`'s job entirely.
- **Retrying the HTTP download of a selected image.** That is
  `_OutputRetriever._download_image()`'s job (`COMFYUI_OUTPUT_DOWNLOAD_MAX_RETRIES`).
- **Deciding whether to retry a whole failed generation, or fall back to a lighter
  `GenerationProfile`.** Per the master document §2.3 and
  `MODULE7_PHASE2_METRICS_RECORDER_DESIGN.md`'s own framing, that ladder is a Phase 3
  concern: Phase 3 catches `VRAMExhaustedError`, asks Phase 1's `ProfileSelector` for
  the next-lighter profile, rebuilds the workflow via `WorkflowBuilder`, and calls
  `generate()` again. `generate()` itself makes exactly one submission attempt.
- **Writing anything to disk.** `_OutputResult.content` is in-memory bytes; no Phase 1
  or Phase 2 component persists a candidate image to a file, and `generate()`
  introduces no new file-writing responsibility (see §0's table).
- **Appending to `GenerationMetrics.jsonl` itself.** `generate()` calls
  `_ComfyUIMetricsRecorder.record_attempt()`; it does not touch `MetricsCollector`
  directly and does not construct a `GenerationMetrics` instance itself.
- **Multi-candidate orchestration, identity retries, QA, or ranking.** See §12.

### 1.3 A responsibility this document must explicitly assign

Every one of the four component-level design documents (§1.3 of the queue-tracker
doc, §1.3-equivalent notes in the output-retriever doc) explicitly disclaims
responsibility for classifying *why* an `EXECUTION_ERROR` happened (VRAM exhaustion
vs. a bad node parameter vs. a missing model file), and explicitly assigns that
decision to `ComfyUIClient` — but no such classification helper exists anywhere in
`comfyui_client.py` today. This document therefore specifies it as part of
`generate()`'s own internal control flow (§4, §10) rather than as a new named
component, since the task brief scopes this document to `generate()` alone and this
classification has no independent surface area or testable public contract of its
own — it is a private conditional inside the method that owns the decision.

---

## 2. Public API

```python
def generate(
    self,
    built_workflow: BuiltWorkflow,
    *,
    video_id: str,
    num_candidates_requested: int = 1,
    identity_retry_count: int = 0,
    peak_vram_mb: float | None = None,
    gpu_utilization_percent: float | None = None,
) -> _OutputResult:
    ...
```

### 2.1 Inputs

| Parameter | Type | Source | Notes |
|---|---|---|---|
| `built_workflow` | `BuiltWorkflow` (`image_generator.py`, Phase 1, frozen dataclass: `graph: dict[str, Any]`, `workflow_ref: WorkflowTemplateRef`, `workflow_hash: str`) | Caller (Phase 3) via `WorkflowBuilder.build()` | `generate()` submits `built_workflow.graph` unmodified. `workflow_ref.niche`, `workflow_ref.profile_name`, `workflow_ref.workflow_version`, and `workflow_hash` are read for the `record_attempt()` call in §6 — `generate()` never re-derives these. |
| `video_id` | `str`, keyword-only, required | Caller | Not carried by `BuiltWorkflow`/`WorkflowTemplateRef` at all (verified: neither dataclass has a `video_id` field), so it must be passed explicitly. Required (`_ComfyUIMetricsRecorder._required_text` raises `ValueError` on empty/blank), matching the collaborator's own validation. |
| `num_candidates_requested` | `int`, keyword-only, default `1` | Caller (Phase 3, when generating more than one candidate for the same video) | Pure pass-through to `record_attempt(num_candidates_requested=...)`. `generate()` itself always requests and retrieves exactly one candidate per call regardless of this value — it is metadata about the caller's broader batch, not an instruction to this call. |
| `identity_retry_count` | `int`, keyword-only, default `0` | Caller (Phase 3's identity-preservation retry ladder, not yet built) | Pure pass-through to `record_attempt(identity_retry_count=...)`. `generate()` has no identity-retry concept of its own. |
| `peak_vram_mb` | `float \| None`, keyword-only, default `None` | Caller | Pure pass-through. No component in this codebase currently measures VRAM during generation (verified: no `pynvml`/`nvidia-smi` usage anywhere in `modules/`); `generate()` does not attempt to measure it either. |
| `gpu_utilization_percent` | `float \| None`, keyword-only, default `None` | Caller | Same as above. |

**[ASSUMED]** `built_workflow` and `video_id` are the only two arguments a Phase 3
call site is guaranteed to already have in hand at the point it calls `generate()`
(`PromptPackage.video_id` is known well before `WorkflowBuilder.build()` runs). The
remaining three keyword arguments default to values that make `generate()` fully
usable with just the first two, which matches this method's "thin orchestration,
one candidate per call" framing. Confirm with Phase 3's actual call site once it
exists that no additional per-call context needs threading through.

### 2.2 Output

Returns `_OutputResult` (already defined in `comfyui_client.py`, §4 below) on success.
`generate()` does not construct, wrap, or copy this value — it returns exactly what
`_OutputRetriever.retrieve()` gave it.

`_OutputResult` is not currently exported via `__all__` (only `SystemStats` is,
today). Verified against the repository: `_OutputResult` is referenced nowhere
outside `modules/comfyui_client.py` and `tests/test_comfyui_client.py` — no other
module imports it. Its leading underscore already follows this file's established
convention for types that cross internal collaborator boundaries without being part
of the module's public surface (`_CompletionResult` is passed between `_QueueTracker`,
`_OutputRetriever`, and `_ComfyUIMetricsRecorder` today the same way, and is not
exported either). A caller receiving `generate()`'s return value does not need to
import the class name to use it — attribute access on the returned instance works
regardless of export status, and Python does not enforce `__all__` as an import
restriction.

This document therefore makes **no recommendation to add `_OutputResult` to
`__all__`**. There is no concrete implementation requirement forcing it, and adding
one would be an unnecessary public API change for a need that does not yet exist. If
a future caller outside this module's own tests needs to `isinstance()`-check or
directly import `_OutputResult` by name, that is a real, then-current requirement to
revisit at that time — not something to pre-emptively export now.

### 2.3 Exceptions

`generate()` raises exactly one of the following per call — never more than one,
never a bare/untranslated exception:

| Exception | Raised when |
|---|---|
| `ComfyUIConnectionError` | `ws.ensure_connected()` fails; or `http.submit_prompt()` raises `_ComfyUIHTTPError` (translated — see §5) |
| `ComfyUIQueueError` | `_QueueTracker` returns `EXECUTION_ERROR` and §10's classification does **not** match an OOM signature |
| `VRAMExhaustedError` | `_QueueTracker` returns `EXECUTION_ERROR` and §10's classification **does** match an OOM signature |
| `ComfyUITimeoutError` **[ASSUMED — NEW, see §5]** | `_QueueTracker` returns `TIMEOUT` |
| `OutputHistoryError`, `NoOutputImageError`, `OutputDownloadError` (incl. `MissingOutputFileError`), `CorruptImageError`, `UnsupportedImageFormatError` | Raised by, and propagated unmodified from, `_OutputRetriever.retrieve()` |
| `ValueError` | Raised by, and propagated unmodified from, argument validation already performed by a collaborator (e.g. `_ComfyUIMetricsRecorder._required_text` on a blank `video_id` — though see §3 step 1, `generate()` validates this itself, earlier, for a clearer stack trace) |

No other exception type is permitted to leave `generate()`. Any unexpected exception
from a collaborator (a bug, not a documented failure mode) is allowed to propagate
as-is rather than being silently swallowed — but per §6, the metrics `finally` block
still runs first.

---

## 3. Complete control flow

```
 1. Validate video_id is non-empty (str, .strip()); raise ValueError immediately if not.
    (Fails fast, before any network I/O, with a clear message — matches every other
    Module 7 component's convention of validating inputs at the top of the call.)

 2. attempt_started_at = time.monotonic()
    (Captured once, passed to record_attempt() in the finally block, §6.)

 3. try:
 4.     self._ws.ensure_connected()
        # Raises ComfyUIConnectionError directly (already public/typed) — let it
        # propagate untranslated. No submission has happened yet, so nothing to
        # clean up.

 5.     try:
 6.         prompt_id = self._http.submit_prompt(built_workflow.graph, self._client_id)
 7.     except _ComfyUIHTTPError as exc:
 8.         raise ComfyUIConnectionError(...) from exc
            # Translation happens HERE and only here for submit_prompt — see §5.

 9.     completion = self._queue_tracker.await_completion(prompt_id, self._client_id)
        # Blocks. Never raises (§1.3/§6 of the queue-tracker doc). Always returns a
        # terminal _CompletionResult: outcome ∈ {COMPLETED, EXECUTION_ERROR, TIMEOUT}.

10.     if completion.outcome is _CompletionOutcome.TIMEOUT:
11.         self._best_effort_cancel(prompt_id)          # §11, never raises
12.         raise ComfyUITimeoutError(f"ComfyUI generation timed out for prompt_id={prompt_id}")

13.     if completion.outcome is _CompletionOutcome.EXECUTION_ERROR:
14.         if self._is_oom(completion.error_payload):    # §10
15.             raise VRAMExhaustedError(self._error_message(completion.error_payload))
16.         raise ComfyUIQueueError(self._error_message(completion.error_payload))

17.     # completion.outcome is _CompletionOutcome.COMPLETED
18.     output = self._output_retriever.retrieve(completion, prompt_id=prompt_id)
        # May raise OutputHistoryError / NoOutputImageError / OutputDownloadError /
        # MissingOutputFileError / CorruptImageError / UnsupportedImageFormatError.
        # Propagated unmodified — no translation needed, these are already the
        # correct public exception types.

19.     return output

20. finally:
21.     self._metrics_recorder.record_attempt(
            video_id=video_id,
            niche=built_workflow.workflow_ref.niche,
            workflow_version=built_workflow.workflow_ref.workflow_version,
            profile_name=built_workflow.workflow_ref.profile_name,
            workflow_hash=built_workflow.workflow_hash,
            completions=(completion,) if "completion" in locals() else (),
            output=output if "output" in locals() else None,
            exception=<the exception about to propagate, if any — see §6>,
            num_candidates_requested=num_candidates_requested,
            identity_retry_count=identity_retry_count,
            peak_vram_mb=peak_vram_mb,
            gpu_utilization_percent=gpu_utilization_percent,
            attempt_started_at=attempt_started_at,
        )
```

Steps 20–21 are pseudocode-shaped only to the extent necessary to specify *which*
values are threaded into `record_attempt()` and *when*; §6 specifies the exact
mechanism (no bare variable-existence checks — an explicit result/exception tracking
pattern is used instead, since relying on `locals()` is exactly the kind of
implementation decision this document must not leave ambiguous). See §6.

### 3.1 Step-by-step summary against the twelve pipeline stages requested

| Stage | Step(s) above | Owning call |
|---|---|---|
| Input validation | 1 | `generate()` itself |
| Workflow submission | 5–8 | `_ComfyUIHTTPTransport.submit_prompt()` |
| Queue tracking | 9 | `_QueueTracker.await_completion()` |
| Completion waiting | 9 (blocks until terminal) | `_QueueTracker.await_completion()` |
| Output retrieval | 18 | `_OutputRetriever.retrieve()` |
| Metrics recording | 20–21 | `_ComfyUIMetricsRecorder.record_attempt()` |
| Result creation | 19 | none — `_OutputResult` returned unmodified |
| Return path | 19 | — |
| Failure path | 4, 7–8, 10–16 | see §5 |

---

## 4. Retry behaviour

**`generate()` implements no retry loop of its own.** This is a deliberate,
verified-against-the-repository design decision, not an omission:

- `tenacity` (a project dependency per `requirements.txt`) is imported and used
  **nowhere** in `modules/comfyui_client.py` today — confirmed by direct search.
  Every retry-shaped behavior that already exists in this file (`_QueueTracker`'s
  WebSocket-to-HTTP-polling fallback and reconnect attempts; `_QueueTracker`'s
  bounded history-confirmation retries via `COMFYUI_HISTORY_CONFIRMATION_RETRY_ATTEMPTS`;
  `_OutputRetriever`'s bounded image-download retries via
  `COMFYUI_OUTPUT_DOWNLOAD_MAX_RETRIES`/`COMFYUI_OUTPUT_DOWNLOAD_RETRY_BACKOFF_SECONDS`)
  is a small, local, hand-rolled loop, fully owned by the component that needs it.
- `config.py` defines no `MODULE7_COMFYUI_OOM_RETRY_ATTEMPTS`-shaped constant (or any
  connection-retry-attempt constant scoped to `ComfyUIClient` itself) — confirmed by
  direct search. Introducing a retry loop in `generate()` today would mean inventing
  both the loop and its governing config constant with no repository precedent for
  either, which the task brief explicitly forbids ("Do not duplicate retry logic
  already owned by another component" / "Do NOT invent repository interfaces").
- Every prior Phase 2 sub-document frames the OOM-retry-with-profile-fallback ladder
  as **Phase 3's** responsibility, not Phase 2's: `MODULE7_PHASE2_METRICS_RECORDER_DESIGN.md`
  states plainly that "no `ComfyUIClient` facade... or OOM-retry loop exists yet" and
  designs `record_attempt()`'s `completions: Sequence[_CompletionResult]` parameter to
  be forward-compatible with a *future* retry loop without needing a signature change
  — it does not require `generate()` to build that loop now.

**What this means concretely:**

- One call to `generate()` == exactly one `submit_prompt()` call == exactly one
  `await_completion()` call == exactly one `_CompletionResult`.
- `completions=(completion,)` — a one-element tuple — is passed to
  `record_attempt()` every time (see §6). This is not a placeholder; it is the
  correct, permanent shape for as long as `generate()` makes a single attempt per
  call, and it costs nothing today per `_ComfyUIMetricsRecorder`'s own design (its
  list-aggregation logic degenerates cleanly for a one-element sequence).
- `VRAMExhaustedError` is raised to the *caller* immediately upon detection (§3 step
  15), not retried internally. A caller (Phase 3, once built) that wants OOM-retry-
  with-profile-downgrade achieves it by catching `VRAMExhaustedError`, asking
  `ProfileSelector` for a lighter profile, calling `WorkflowBuilder.build()` again,
  and calling `generate()` again — each such call is independently metered by its
  own `record_attempt()` invocation with its own one-element `completions` tuple.
  (If a future revision of this document wants `generate()` itself to retry
  internally across multiple `_CompletionResult`s in one call, that is an explicit,
  separate design change to `record_attempt()`'s call site — not something this
  version does silently.)
- **Nothing here retries** malformed HTTP responses at the `generate()` level either
  (a malformed `/prompt` response is a `_ComfyUIHTTPError`, translated once to
  `ComfyUIConnectionError`, and raised — not retried, since `_ComfyUIHTTPTransport`
  itself owns zero retry logic for `submit_prompt` and none is added here).

**Retry limits / ordering / backoff strategy:** N/A at the `generate()` level, for
the reasons above. See the cited component design documents for the retry limits,
ordering, and backoff strategy owned by `_QueueTracker` and `_OutputRetriever`
respectively — `generate()` does not duplicate, wrap, or re-document those.

---

## 5. Exception flow

### 5.1 Propagate unmodified (no translation)

- `ComfyUIConnectionError` from `self._ws.ensure_connected()` — already the correct
  public type.
- `OutputHistoryError`, `NoOutputImageError`, `OutputDownloadError`,
  `MissingOutputFileError`, `CorruptImageError`, `UnsupportedImageFormatError` from
  `self._output_retriever.retrieve()` — already the correct public types.
- `ValueError` from `generate()`'s own step-1 input validation.

### 5.2 Translate

| Caught | Raised instead | Where |
|---|---|---|
| `_ComfyUIHTTPError` from `self._http.submit_prompt(...)` | `ComfyUIConnectionError` | §3 steps 5–8 |

This is the **only** translation point `generate()` performs. `_ComfyUIHTTPError` is
explicitly documented (both in the master document §3.2 and directly in
`comfyui_client.py`'s own docstring for the class) as "never escapes past the
facade" — `ComfyUIClient` is the sole place that catches it. Since `_QueueTracker`
already fully absorbs every `_ComfyUIHTTPError` it might see from its own internal
`history()` polling calls (per its own design doc §1.3 — "translate every
transport-level failure it observes... into loop behavior — never let either escape
`await_completion()`"), and `_OutputRetriever` already translates every HTTP failure
during download into `OutputDownloadError`/`MissingOutputFileError` before returning,
`submit_prompt()` is the only call site inside `generate()` itself where a raw
`_ComfyUIHTTPError` can reach this method directly.

**[ASSUMED]** A submission-time `_ComfyUIHTTPError` is classified as
`ComfyUIConnectionError` rather than `ComfyUIQueueError`, on the reasoning that a
failure at `/prompt` (connection refused, timeout, malformed JSON response, non-2xx
status) reflects an inability to reach or converse with ComfyUI at all, not a
problem with the *content* of an accepted, queued job — `ComfyUIQueueError` is
reserved for failures ComfyUI itself reports about a job it accepted (§5.4 below).
If a future revision wants to distinguish "ComfyUI rejected this graph as invalid"
(e.g. an HTTP 400 with a node-validation payload) from "ComfyUI was unreachable"
at the `/prompt` call specifically, that requires `_ComfyUIHTTPTransport` itself to
preserve the HTTP status code on `_ComfyUIHTTPError` — a change to a completed
component, out of scope here, and not required by the current exception hierarchy
(`ComfyUIQueueError`'s own docstring, "Reserved for Phase 2 ComfyUI execution
failures," covers execution outcomes, not malformed submissions).

### 5.3 Constructed directly by `generate()` (not a translation of a caught exception)

| Exception | Constructed when |
|---|---|
| `VRAMExhaustedError` | `completion.outcome is EXECUTION_ERROR` and §10 classifies `error_payload` as OOM |
| `ComfyUIQueueError` | `completion.outcome is EXECUTION_ERROR` and §10 does not classify it as OOM |
| `ComfyUITimeoutError` **[ASSUMED — NEW]** | `completion.outcome is TIMEOUT` |

### 5.4 **[ASSUMED — NEW]** `ComfyUITimeoutError` does not yet exist

`module7_exceptions.py` today has no exception dedicated to a `_QueueTracker`
`TIMEOUT` outcome. `_CompletionOutcome.TIMEOUT` is a real, already-implemented
enum member that `await_completion()` can and does return — `generate()` must map it
to *something* typed. Two options were weighed:

1. **Reuse `ComfyUIQueueError`** for both `EXECUTION_ERROR`-non-OOM and `TIMEOUT`.
   Requires no new exception class, but conflates two operationally distinct
   failures (a node genuinely erroring vs. ComfyUI simply never finishing within
   `COMFYUI_EXECUTION_TIMEOUT_SECONDS`) under one type, which would force Phase 3 to
   inspect the exception's message string to tell them apart if it ever needs
   different handling (e.g., a timeout might warrant a longer-timeout retry on the
   same profile, whereas a queue error should not).
2. **Add `ComfyUITimeoutError(Module7Error)`** to `module7_exceptions.py` — a
   minimal, additive, one-class change, following the exact pattern already used by
   every sibling exception in that file (a one-line docstring, no constructor
   override needed since `_QueueTracker` supplies no structured timeout payload
   beyond the message).

**This document recommends option 2** and treats it as a required prerequisite for
implementing `generate()` — flagged here per the task brief's instruction to mark
uncertain interfaces `[ASSUMED]` rather than silently invent them. This is the one
place this document asks for a (trivial, additive) change outside `generate()`
itself; everything else it depends on already exists. Confirm this addition before
implementation begins.

**Scope of this prerequisite, stated explicitly:**

- `ComfyUITimeoutError` is a **minimal, additive** exception — one new class, nothing
  else touched.
- It subclasses `Module7Error` directly, exactly as every existing exception in
  `module7_exceptions.py` does (`ComfyUIConnectionError`, `ComfyUIQueueError`,
  `VRAMExhaustedError`, etc. — verified in source, all inherit `Module7Error`
  directly, none inherit each other except the documented `OutputRetrievalError`
  subtree).
- It contains **no custom logic** — no overridden `__init__`, no extra attributes, no
  behavior beyond what `Module7Error`/`Exception` already provide. This matches the
  majority of the file's existing exceptions (e.g. `ComfyUIConnectionError`,
  `ComfyUIQueueError`, `VRAMExhaustedError` are each a bare `class Name(Module7Error):`
  with only a one-line docstring — no constructor override). `ComfyUITimeoutError`
  follows that same bare pattern, not the small minority (`OutputRetrievalError` and
  its subclasses) that carry structured fields, since `_QueueTracker`'s `TIMEOUT`
  outcome supplies no structured payload for it to carry (§5.4 above).
- It follows the exact stylistic convention already used throughout the file: a
  `class ComfyUITimeoutError(Module7Error):` declaration immediately followed by a
  single one-line docstring describing when it is raised — matching, verbatim in
  form, lines such as `class ComfyUIQueueError(Module7Error): """Reserved for Phase
  2 ComfyUI execution failures."""`.
- **No existing exception in the hierarchy is modified, renamed, reparented, or
  removed.** `ComfyUITimeoutError` is a pure addition alongside the existing classes;
  every current `isinstance()` check, `except` clause, and test assertion elsewhere
  in the codebase continues to behave exactly as it does today.
- **No existing exception behavior changes.** Nothing about how `ComfyUIConnectionError`,
  `ComfyUIQueueError`, `VRAMExhaustedError`, or any `OutputRetrievalError` subclass is
  raised, caught, or classified (including `_ComfyUIMetricsRecorder`'s
  `_FAILURE_REASON_BY_EXCEPTION_TYPE` mapping) is touched by this addition.

This is a tiny, mechanical prerequisite to unblock `generate()`'s implementation —
not an architectural change to the exception hierarchy, and not a redesign of any
completed component.

### 5.5 Swallowed

Nothing is swallowed by `generate()` itself. The one place an exception is
deliberately *not* allowed to propagate is inside `_best_effort_cancel()` (§11) and
inside `_ComfyUIMetricsRecorder.record_attempt()` (already self-swallowing, §6) —
both are documented explicitly below rather than left as an implicit "nothing here
raises" assumption.

---

## 6. Metrics integration

### 6.1 Exactly when

`_ComfyUIMetricsRecorder.record_attempt()` is called **exactly once per `generate()`
call**, in a `finally` block wrapping the entire body from just after step 1's input
validation through the return, so it runs on every exit path — success, every typed
exception in §5, and (if it ever happened) an untranslated/unexpected exception.

Input validation (§3 step 1, blank `video_id`) is the one exit path that happens
*before* the `try`/`finally`, and therefore does **not** trigger a `record_attempt()`
call — there is no meaningful attempt to record yet (no workflow was submitted, no
niche/profile/hash are even in hand), and `_ComfyUIMetricsRecorder._required_text`
would reject the same blank `video_id` internally in any case. This matches the
existing collaborator's own validation, just surfaced earlier for a cleaner error.

### 6.2 Exactly what data it receives, and how outcome-dependent values are tracked

Rather than relying on `locals()` introspection (used only as illustrative
pseudocode in §3), `generate()`'s implementation tracks two local variables,
initialized before the `try` block:

```python
completion: _CompletionResult | None = None
output: _OutputResult | None = None
```

`completion` is assigned as soon as `await_completion()` returns (§3 step 9, before
any outcome branching). `output` is assigned only on the `COMPLETED` path, after
`_OutputRetriever.retrieve()` succeeds (§3 step 18). The exception (if any) about to
propagate is captured with `except BaseException as exc: raise` wrapping the whole
try body one level in — or, equivalently and more idiomatically, `sys.exc_info()`
is not needed at all: the `finally` block's `record_attempt()` call is placed
*inside* a single `try/except Exception as exc: raise` that re-raises after
recording, so `exc` is in scope for the `record_attempt(exception=exc)` argument.
This mirrors the existing project convention (`_OutputRetriever`'s own methods pass
already-caught exception objects downward rather than re-inspecting the exception
state).

Concretely, the values passed are:

| `record_attempt()` parameter | Value |
|---|---|
| `video_id` | the validated `video_id` argument |
| `niche` | `built_workflow.workflow_ref.niche` |
| `workflow_version` | `built_workflow.workflow_ref.workflow_version` |
| `profile_name` | `built_workflow.workflow_ref.profile_name` |
| `workflow_hash` | `built_workflow.workflow_hash` |
| `completions` | `(completion,)` if `completion is not None`, else `()` — see §6.3 for why the empty case must never actually occur |
| `output` | `output` (may be `None`) |
| `exception` | the exception being raised, or `None` on success |
| `num_candidates_requested` | pass-through argument |
| `identity_retry_count` | pass-through argument |
| `peak_vram_mb` | pass-through argument |
| `gpu_utilization_percent` | pass-through argument |
| `attempt_started_at` | the `time.monotonic()` value captured at the very top of `generate()` (§3 step 2) |

### 6.3 The one edge case: `completions` must never be empty

`_ComfyUIMetricsRecorder._validated_completions()` raises `ValueError` if
`completions` is empty. The only way `generate()` could reach its `finally` block
with `completion is None` is if `ensure_connected()` or `submit_prompt()` raised
*before* `await_completion()` was ever called (§3 steps 4–8). This is a real, valid
failure path — and it means `record_attempt()` cannot be called with a real
`_CompletionResult` in that case, because none exists yet.

**Resolution:** for this specific pre-submission failure window, `generate()`
constructs a synthetic `_CompletionResult` representing zero elapsed queue/generation
time, so the metrics record still gets written (satisfying "ensure metrics are always
recorded") without inventing a fake outcome value:

```python
_CompletionResult(
    outcome=_CompletionOutcome.EXECUTION_ERROR,   # nearest-fit; queue/generation never started
    history_payload=None,
    error_payload=None,
    queue_wait_seconds=0.0,
    generation_seconds=0.0,
    used_http_fallback=False,
)
```

**[ASSUMED]** Using `_CompletionOutcome.EXECUTION_ERROR` as the nearest-fit
placeholder outcome for a pre-submission connection failure is a judgment call —
`_CompletionResult`/`_CompletionOutcome` (owned by `_QueueTracker`, not modifiable
here) has no `outcome` value that literally means "never submitted." This choice is
inert in practice: `_ComfyUIMetricsRecorder._classify_failure()` checks
`exception is not None` *first* (§4.2's own precedence, verified in
`_classify_failure`'s source), and a real `exception` (the `ComfyUIConnectionError`
being raised) is always present on this path — so the synthetic outcome's value is
never actually read for `failure_reason` classification; it exists purely to satisfy
`_validated_completions()`'s "must not be empty" and "must have valid dataclass
values" checks. Confirm this reasoning holds if `_classify_failure()`'s precedence
ever changes.

### 6.4 Recording on each specific case

| Case | `completions` | `output` | `exception` |
|---|---|---|---|
| Success | `(completion,)`, `outcome=COMPLETED` | the returned `_OutputResult` | `None` |
| `EXECUTION_ERROR` → `VRAMExhaustedError` | `(completion,)`, `outcome=EXECUTION_ERROR` | `None` | the `VRAMExhaustedError` instance |
| `EXECUTION_ERROR` → `ComfyUIQueueError` | `(completion,)`, `outcome=EXECUTION_ERROR` | `None` | the `ComfyUIQueueError` instance |
| `TIMEOUT` | `(completion,)`, `outcome=TIMEOUT` | `None` | the `ComfyUITimeoutError` instance |
| Retrieval failure (any `OutputRetrievalError` subclass) | `(completion,)`, `outcome=COMPLETED` | `None` (retrieval never completed) | the retrieval exception instance |
| Pre-submission connection failure | synthetic (§6.3) | `None` | the `ComfyUIConnectionError` instance |

In every row, `_ComfyUIMetricsRecorder._classify_failure()` (already implemented,
unmodified) derives the correct `failure_reason` string from whichever of
`exception`/`completion.outcome`/`output is None` it is given — `generate()` supplies
raw values only and performs no failure-reason string logic itself, per the "do not
duplicate logic already owned by another component" constraint.

### 6.5 Ensuring metrics are always recorded

Because `record_attempt()` is called from a `finally` block (§6.1) and because
`_ComfyUIMetricsRecorder.record_attempt()` already wraps its own body (`GenerationMetrics`
construction and `self._collector.append(...)`) in `try/except Exception:
logger.error(...)` — verified directly in the component's source — **no
`MetricsWriteError` or any other metrics-layer exception can ever propagate out of
`generate()`**, regardless of what fails inside `record_attempt()`. `generate()`
therefore needs no additional `try/except` around the `record_attempt()` call itself;
it is already fail-safe by construction, and adding a redundant `try/except` there
would itself be the kind of unnecessary duplication the task brief warns against.

---

## 7. Logging strategy

`generate()` logs at the **phase-transition level only** — one INFO line for entering
generation and one for each terminal outcome — and deliberately does not re-log
anything already logged by a collaborator at DEBUG/INFO/WARNING inside their own
calls (submission detail, WebSocket event parsing, queue-wait/generation timing
breakdown, download attempts, history-confirmation retries — all already logged by
`_ComfyUIHTTPTransport`, `_ComfyUIWebSocketTransport`, `_QueueTracker`, and
`_OutputRetriever` respectively).

| Level | When | Example |
|---|---|---|
| INFO | Entering `generate()`, after validation, before submission | `"Starting ComfyUI generation: video_id={video_id} niche={niche} profile={profile} workflow_hash={hash}"` |
| INFO | After `submit_prompt()` succeeds | `"ComfyUI prompt submitted: video_id={video_id} prompt_id={prompt_id}"` |
| INFO | On `COMPLETED` + successful retrieval, just before `return` | `"ComfyUI generation complete: video_id={video_id} prompt_id={prompt_id} queue_wait={q:.2f}s generation={g:.2f}s used_http_fallback={f}"` |
| WARNING | On `TIMEOUT`, before raising `ComfyUITimeoutError` | `"ComfyUI generation timed out: video_id={video_id} prompt_id={prompt_id}"` |
| WARNING | On `EXECUTION_ERROR` classified as OOM, before raising `VRAMExhaustedError` | `"ComfyUI generation exhausted VRAM: video_id={video_id} prompt_id={prompt_id}"` |
| ERROR | On `EXECUTION_ERROR` classified as a non-OOM queue error, before raising `ComfyUIQueueError` | `"ComfyUI generation failed with execution error: video_id={video_id} prompt_id={prompt_id} message={msg}"` |
| ERROR | On a pre-submission `ComfyUIConnectionError` | `"ComfyUI generation could not be started: video_id={video_id} error={error}"` |

Every `error_payload`/`history_payload` dict is logged, at most, as its already-
extracted string message (`exception_message`, per §10) — never the raw dict.
This matches the explicit "never logged directly" rule stated in both
`MODULE7_PHASE2_WEBSOCKET_TRANSPORT_DESIGN.md` §8 and
`MODULE7_PHASE2_QUEUE_TRACKER_DESIGN.md` §8. No raw image bytes, no full prompt
graph JSON, and no full workflow text are ever logged by `generate()` — only IDs,
hashes, timings, and short extracted messages.

`_OutputRetriever.retrieve()` failures are **not** re-logged at `generate()`'s level
beyond the ERROR line above for `EXECUTION_ERROR`-classified failures; retrieval
failures (`OutputHistoryError` etc.) are already logged with full context by
`_OutputRetriever` itself before it raises, so `generate()` does not add a second,
duplicate ERROR line for that specific path — it relies on the exception's own
message when it needs to log anything about that path at all (and per the table
above, it does not add a distinct line for retrieval failures specifically, to avoid
duplicating `_OutputRetriever`'s own logging).

---

## 8. Thread safety

- **Concurrency assumption:** `MODULE7_MAX_CONCURRENT_GENERATIONS = 1` (verified in
  `config.py`) — the existing project convention, inherited from Phase 1's design, is
  that Module 7 generates candidates **sequentially**, never concurrently, within one
  pipeline run. `generate()` is designed against this assumption and is not required
  to be safe for concurrent invocation from multiple threads on the same
  `ComfyUIClient` instance.
- `_QueueTracker` mutates its own instance state (`_call_started_at`,
  `_execution_started_at`, `_state`, etc.) on every `await_completion()` call via
  `_reset_call_state()` — it is explicitly **not** safe for two concurrent
  `await_completion()` calls on the same tracker instance (confirmed by its own
  design doc's "one blocking call" framing). Since `generate()` is Module 7's only
  caller of `await_completion()`, and Module 7 never calls `generate()` concurrently
  per the bullet above, this is consistent, not a new constraint introduced here.
- `_OutputRetriever` holds no mutable instance state (confirmed in its own design
  doc §"Thread-safety") and would tolerate concurrent `retrieve()` calls if the
  underlying `_ComfyUIHTTPTransport`/`requests.Session` did — but this is moot under
  the sequential-generation assumption above.
- `_ComfyUIMetricsRecorder.record_attempt()` appends to a JSONL file via
  `MetricsCollector.append()`, which itself opens, writes, and closes the file
  handle per call (no persistent open handle held across calls) — safe for
  sequential reuse across many `generate()` calls on the same client instance, which
  is exactly the pattern Phase 3 is expected to use (one `ComfyUIClient` per
  pipeline run, many sequential `generate()` calls for multiple candidates).
- `generate()` introduces no locks, no new mutable state on `ComfyUIClient` beyond
  what `__init__` already establishes (out of scope here), and relies entirely on
  the existing single-threaded, sequential-generation model already assumed by
  every completed Phase 1/Phase 2 component.

---

## 9. Performance

- **Blocking operations.** `generate()` is fully synchronous and blocking end-to-end
  — `ensure_connected()` (network), `submit_prompt()` (network), `await_completion()`
  (blocks for up to `COMFYUI_EXECUTION_TIMEOUT_SECONDS`, by design — this *is* the
  generation), and `retrieve()` (network, image decode). This matches every existing
  component in the file; no component in this codebase uses `asyncio`, and
  introducing async here would be a scope-exceeding architectural change this
  document explicitly does not make.
- **Memory.** `generate()` holds at most one `_CompletionResult` (a handful of
  primitives and one `history_payload`/`error_payload` dict, already bounded by
  ComfyUI's own history response size) and one `_OutputResult` (whose `content:
  bytes` is a single decoded image, typically low-single-digit megabytes) in memory
  at a time. `generate()` does not accumulate state across calls — no cache, no
  growing list — so its own memory footprint does not grow with repeated calls.
- **CPU.** `generate()` itself performs no CPU-bound work — no image decoding
  (`_OutputRetriever` does that, once, via Pillow), no hashing (`built_workflow.workflow_hash`
  is already computed by `WorkflowBuilder`), no JSON re-serialization of the graph
  (submitted as-is). The only CPU-bound work `generate()` triggers is what its
  collaborators already do internally.
- **Avoiding duplicated work.** `generate()` calls each collaborator's public method
  exactly once per `generate()` call (§3), with no redundant re-fetches: it does not
  re-fetch `/history/{prompt_id}` itself (already returned inside `completion.history_payload`
  by `_QueueTracker`, and consumed directly by `_OutputRetriever` from that same
  object rather than being re-fetched — confirmed via `_OutputRetriever._resolve_prompt_id`/
  `_extract_history_payload` reading from the passed-in `completion`), and does not
  re-download or re-validate the image bytes `_OutputRetriever` already validated.

---

## 10. Component interaction

### 10.1 `_ComfyUIHTTPTransport`

`generate()` calls exactly one method directly: `submit_prompt(built_workflow.graph,
self._client_id)`. (`interrupt()`/`delete_from_queue()` are called only from the
timeout-cancellation path, §11 — still direct calls, no wrapping.) `generate()` never
calls `history()`, `view_image()`, `queue_status()`, or `system_stats()` directly —
those are exclusively `_QueueTracker`'s and `_OutputRetriever`'s concerns (or, for
`system_stats()`, out of `generate()`'s scope entirely — that belongs to a
`health_check()`-shaped method this document does not design).

### 10.2 `_ComfyUIWebSocketTransport`

`generate()` calls exactly one method directly: `ensure_connected()`, once, at the
top of the method, before submission. `generate()` never calls `receive()`,
`next_event()`, or `close()` directly — event consumption is exclusively
`_QueueTracker`'s job; connection teardown is exclusively `ComfyUIClient.close()`'s
job (out of scope here).

### 10.3 `_QueueTracker`

`generate()` constructs no `_QueueTracker` state and calls exactly one method:
`await_completion(prompt_id, self._client_id)`, once. `generate()` reads
`completion.outcome`, `completion.error_payload` (only on `EXECUTION_ERROR`, for
classification, §10.6), and passes the whole `completion` object through to
`_OutputRetriever.retrieve()` and `_ComfyUIMetricsRecorder.record_attempt()`
unmodified — it never reads or mutates `completion.history_payload`,
`completion.queue_wait_seconds`, `completion.generation_seconds`, or
`completion.used_http_fallback` directly (those are read by the INFO log line in §7
for display only, and by the two downstream collaborators for their own purposes).

### 10.4 `_OutputRetriever`

`generate()` calls exactly one method: `retrieve(completion, prompt_id=prompt_id)`,
once, only when `completion.outcome is _CompletionOutcome.COMPLETED`. `generate()`
passes `prompt_id` explicitly (rather than relying on `_OutputRetriever`'s own
`_resolve_prompt_id` fallback-from-history-payload logic) since `generate()` already
has the authoritative `prompt_id` from `submit_prompt()`'s own return value — this
avoids `_OutputRetriever` needing to re-derive it from `history_payload` on the
common path, while still tolerating a `None` fallback per its own documented
contract if a future caller ever omits it.

### 10.5 `_ComfyUIMetricsRecorder`

`generate()` calls exactly one method: `record_attempt(...)`, exactly once, in a
`finally` block, per §6.

### 10.6 The one piece of logic `generate()` owns outright: OOM classification

No existing component classifies `error_payload` as VRAM-exhaustion vs. any other
execution error (§1.3). `generate()` implements this as a small private helper,
`_is_oom(error_payload: dict[str, Any] | None) -> bool`, operating only on the two
string fields the WebSocket transport's own design doc (§4.3) documents as present
on an `execution_error` event's `data` — and therefore on `error_payload` — today:
`exception_type` and `exception_message`.

```python
_OOM_SIGNATURES = ("outofmemoryerror", "cuda out of memory", "out of memory")

def _is_oom(self, error_payload: dict[str, Any] | None) -> bool:
    if not error_payload:
        return False
    haystack = " ".join(
        str(error_payload.get(field, ""))
        for field in ("exception_type", "exception_message")
    ).lower()
    return any(signature in haystack for signature in _OOM_SIGNATURES)
```

**[ASSUMED]** The exact set of OOM signature substrings above is inferred from the
master document's own sequence diagram (§2.3: `exception_type: "OutOfMemoryError"`)
and from ComfyUI/PyTorch's well-known real-world error text ("CUDA out of memory").
No test fixture or real ComfyUI OOM payload was available to verify this exhaustively
against. This must be confirmed against a real captured ComfyUI OOM `execution_error`
payload (or the `gpu`-marked integration test tier, once it exists) before this
classification is trusted in production — a false negative here means a real OOM is
misreported as a generic `ComfyUIQueueError`, which is a silent classification
degradation, not a crash, so it is safe to ship provisionally but must be verified.

A companion helper, `_error_message(error_payload) -> str`, extracts
`error_payload.get("exception_message")` (falling back to a generic
`"ComfyUI execution error"` string if absent/malformed) for use in the exception
message and the ERROR log line in §7 — this is the only part of `error_payload`
ever surfaced in a message or log line, per §7's "extracted string message only"
rule.

---

## 11. Best-effort cancellation on timeout

`_best_effort_cancel(prompt_id: str) -> None` is a small private helper invoked only
from §3 step 11 (the `TIMEOUT` path). It issues both of ComfyUI's cancellation
primitives, since `generate()` cannot know from the timeout alone whether the prompt
is still queued or actively (hung) executing:

```python
def _best_effort_cancel(self, prompt_id: str) -> None:
    for action in (self._http.interrupt, lambda: self._http.delete_from_queue(prompt_id)):
        try:
            action()
        except _ComfyUIHTTPError as exc:
            logger.warning(
                "ComfyUI best-effort cancellation failed for prompt_id={prompt_id}: {error}",
                prompt_id=prompt_id, error=str(exc),
            )
```

Both calls are attempted (interrupt covers "currently executing"; delete-from-queue
covers "still queued") and any failure is logged at WARNING and swallowed — this
path must never mask or replace the `ComfyUITimeoutError` that is about to be raised
regardless of whether cancellation succeeded, matching the exact "cleanup must never
mask the caller's real outcome" convention already used verbatim inside
`_ComfyUIWebSocketTransport.close()` (`comfyui_client.py`, confirmed in source).

---

## 12. Testing strategy

New test class `TestGenerate` in `tests/test_comfyui_client.py`, following the
existing file's established pattern (`Mock`/hand-rolled fakes for
`_ComfyUIHTTPTransport`/`_ComfyUIWebSocketTransport`, direct construction of
`_CompletionResult`/`_OutputResult` fixtures, `pytest.raises` for exception paths —
no new test infrastructure introduced). `generate()` is exercised by mocking its
five collaborators directly (constructor-injected fakes, matching every existing
collaborator's own "constructor injection, not created internally" convention) —
never a real ComfyUI instance, matching the file's existing `gpu`-marker split.

### 12.1 Success path

- Happy path: `ensure_connected()` → `submit_prompt()` → `await_completion()`
  returns `COMPLETED` → `retrieve()` returns a fixture `_OutputResult` → `generate()`
  returns that exact object; `record_attempt()` called once with
  `completions=(completion,)`, `output=<the result>`, `exception=None`.
- Assert `record_attempt()` receives the correct `video_id`, `niche`,
  `workflow_version`, `profile_name`, `workflow_hash` derived from a fixture
  `BuiltWorkflow`.
- Assert `retrieve()` is called with the exact `prompt_id` returned by
  `submit_prompt()`, not a re-derived one.
- Assert `attempt_started_at` passed to `record_attempt()` is a `time.monotonic()`-
  shaped float captured before any collaborator call (mockable via `monkeypatch` on
  `time.monotonic`, matching how `_QueueTracker`'s own tests already do this).

### 12.2 Every failure path

- **Pre-submission connection failure:** `ensure_connected()` raises
  `ComfyUIConnectionError` → propagates unmodified; `submit_prompt()` never called;
  `record_attempt()` called once with the synthetic `_CompletionResult` (§6.3) and
  `exception=<that ComfyUIConnectionError instance>`.
- **Submission failure:** `submit_prompt()` raises `_ComfyUIHTTPError` →
  `generate()` raises `ComfyUIConnectionError` (translated, §5.2), chained (`.__cause__`
  is the original `_ComfyUIHTTPError`); same synthetic-completion metrics assertion
  as above.
- **`EXECUTION_ERROR`, OOM-classified:** `await_completion()` returns a
  `_CompletionResult(outcome=EXECUTION_ERROR, error_payload={"exception_type":
  "OutOfMemoryError", ...})` → `generate()` raises `VRAMExhaustedError`;
  `record_attempt()` receives `exception=<VRAMExhaustedError>`, `output=None`.
  Parametrize this case across every string in `_OOM_SIGNATURES` (§10.6) to lock in
  the classification boundary, plus one case with a *non*-matching
  `exception_message` (e.g. `"KeyError: 'checkpoint'"`) asserting `ComfyUIQueueError`
  instead — this is the single highest-value test in this whole suite, since it is
  the one piece of logic this document invents outright (§1.3).
- **`EXECUTION_ERROR`, non-OOM:** as above but asserting `ComfyUIQueueError` and its
  message contains the extracted `exception_message`.
- **`EXECUTION_ERROR`, empty/missing `error_payload`:** `_is_oom(None)` and
  `_is_oom({})` both return `False` → `ComfyUIQueueError`, generic fallback message
  (no `KeyError`/`AttributeError` from missing dict keys).
- **`TIMEOUT`:** `await_completion()` returns `_CompletionResult(outcome=TIMEOUT, ...)`
  → asserts both `self._http.interrupt()` and
  `self._http.delete_from_queue(prompt_id)` were called before `ComfyUITimeoutError`
  is raised; `record_attempt()` receives `exception=<ComfyUITimeoutError>`.
- **`TIMEOUT` with cancellation itself failing:** `interrupt()`/`delete_from_queue()`
  both raise `_ComfyUIHTTPError` → asserts `ComfyUITimeoutError` (not the
  cancellation failure) is what ultimately propagates, and that both cancellation
  attempts were still made despite the first one failing.
- **Retrieval failure, each subtype:** `await_completion()` returns `COMPLETED`,
  `retrieve()` raises each of `OutputHistoryError`, `NoOutputImageError`,
  `OutputDownloadError`, `MissingOutputFileError`, `CorruptImageError`,
  `UnsupportedImageFormatError` in turn → each propagates unmodified;
  `record_attempt()` receives `completions=(completion,)` with `outcome=COMPLETED`,
  `output=None`, `exception=<that specific instance>` in every case.
- **Blank `video_id`:** `generate(built_workflow, video_id="")` (and `"   "`) raises
  `ValueError` before `ensure_connected()` is ever called (assert the mock transport
  saw zero calls) and before `record_attempt()` is ever called.

### 12.3 Retry behaviour

Per §4, there is no retry loop to test at this level. Instead, assert the **absence**
of retry behavior directly: `await_completion()` and `retrieve()` are each called
**exactly once** per `generate()` call in every scenario above (`assert_called_once`
on both mocks), including on `VRAMExhaustedError` — this is the regression test that
locks in "`generate()` does not retry OOM internally" as a deliberate contract, not
an accident, and would immediately fail if someone later adds an internal retry loop
without also updating this document.

### 12.4 Metrics recording

- For every scenario in §12.1/§12.2, assert `record_attempt()` is called **exactly
  once** (never zero, never more than once) — this is the "ensure metrics are always
  recorded" regression test.
- Assert `record_attempt()`'s keyword arguments match exactly the table in §6.2 for
  a representative subset of the scenarios above (success, one exception-path, the
  pre-submission synthetic-completion path specifically, since it is the least
  obvious of the three).

### 12.5 Exception propagation

- For each exception type in §5, assert the raised exception's type is exactly the
  documented one (not a subclass masquerading, not the raw untranslated original) —
  `pytest.raises(ExactType)`, and for the translated `_ComfyUIHTTPError` →
  `ComfyUIConnectionError` case, additionally assert `exc.value.__cause__` is the
  original `_ComfyUIHTTPError` instance (`raise ... from exc` chaining preserved).

### 12.6 Boundary cases

- `built_workflow.graph` is an empty-but-valid dict (`{}`) — `generate()` performs
  no validation of the graph's contents itself (that already happened inside
  `WorkflowBuilder.build()`/`WorkflowLibrary.validate()`, upstream, per §1.2); assert
  `generate()` passes it through to `submit_prompt()` unmodified without raising.
- `num_candidates_requested`, `identity_retry_count`, `peak_vram_mb`,
  `gpu_utilization_percent` all omitted (defaults) — assert `record_attempt()`
  receives `num_candidates_requested=1`, `identity_retry_count=0`,
  `peak_vram_mb=None`, `gpu_utilization_percent=None`.
- Two sequential `generate()` calls on the same `ComfyUIClient`/mocked collaborators
  — assert `ensure_connected()` is called both times (it is idempotent/no-op if
  already connected, per its own contract) and that the second call's
  `attempt_started_at` differs from the first's, confirming no stale timing state
  leaks between calls (this is really a `_QueueTracker`/`ensure_connected`
  regression, but worth asserting once here since it is exactly the "many
  sequential `generate()` calls per pipeline run" pattern §8 assumes).

### 12.7 Regression tests

- The OOM-signature parametrization in §12.2 doubles as the regression suite for
  §10.6's classification boundary — any future change to `_OOM_SIGNATURES` must keep
  every case in that parametrization passing.
- The "called exactly once" assertions in §12.3/§12.4 are themselves the permanent
  regression guard against silently reintroducing retry logic or duplicate metrics
  calls in a future edit.

---

## 13. Future compatibility

- **Multiple candidate generation.** No change needed. Per the master document's own
  §15.1-equivalent framing (confirmed consistent with this document's §1.2/§4),
  Phase 3 achieves multi-candidate generation by calling `generate()` multiple times
  with the same `video_id` and an incremented awareness of `num_candidates_requested`
  — `generate()`'s per-call statelessness (aside from the reused HTTP session/WebSocket
  connection, owned by `__init__`/`close`, out of scope here) makes this trivial.
  Nothing in §2's signature needs to change; `candidate_index`-shaped bookkeeping
  (which candidate this is, for file-naming purposes) belongs to whichever future
  component persists `_OutputResult.content` to disk — not to `generate()`, per §0's
  correction that no disk-writing responsibility exists here today.
- **Generation profiles.** Already fully supported — `built_workflow.workflow_ref.profile_name`
  flows straight through to `record_attempt(profile_name=...)` today. A future
  profile-fallback ladder (§4) requires zero changes to `generate()`'s signature or
  body — it is entirely a new call pattern from Phase 3 (catch `VRAMExhaustedError`,
  rebuild, call `generate()` again), not a new parameter or branch inside `generate()`.
- **QA stages.** `generate()`'s return type, `_OutputResult`, is deliberately the
  "pre-QA, pre-restoration raw candidate" type (matching the master document's own
  framing that Phase 2's output must never be mistaken for a finished,
  QA-eligible `GeneratedAsset`). A future QA stage consumes `_OutputResult.content`
  as its input and produces a `QualityAssuranceReport` — it does not require
  `generate()` to change shape, since `generate()` was never going to run QA itself
  (§1.2).
- **Ranking.** Operates entirely on `QualityAssuranceReport`/`CandidateScore` objects
  a future Phase 4 derives *from* multiple `generate()` calls' outputs — never on
  `generate()`'s internals. No change needed.
- **Orchestration.** A future Phase 3 orchestrator wraps `generate()` in exactly the
  retry/fallback/multi-candidate/identity-retry logic enumerated above, entirely from
  the outside, using only `generate()`'s existing public contract (§2) and the typed
  exceptions it already raises (§5). This document's explicit decision *not* to build
  any of that logic inside `generate()` itself (§1.2, §4) is precisely what keeps this
  future extension additive rather than requiring a redesign of this method.

---

## 14. Prerequisites before implementation (from §0/§5.4/§10.6)

This design is implementation-ready modulo two small, explicitly-flagged, additive
items — neither of which touches a completed component's existing behavior:

1. Add `ComfyUITimeoutError(Module7Error)` to `module7_exceptions.py` — a minimal,
   additive, bare exception class with a one-line docstring and no custom logic,
   following the file's existing style exactly, with no change to any existing
   exception in the hierarchy (§5.4).
2. Verify the `_OOM_SIGNATURES` substring list (§10.6) against a real captured
   ComfyUI OOM `execution_error` payload before relying on it in production; treat
   it as provisional until then.

No `__all__`/export changes are required for this design (§2.2) — `_OutputResult`
remains module-internal, matching how `_CompletionResult` is already used across
this file's collaborators today.
