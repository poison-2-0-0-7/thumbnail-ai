# Module 7 — ControlNet Compatibility Architecture

**thumbnail-ai**
**Status:** Architecture only. Zero implementation code, zero tests, zero repository modification. Written for implementation by a separate agent (per the brief: Gemini CLI).
**Scope:** `workflows/fragments/controlnet_*.json`, `modules/generation_components/capability_probe.py`, `modules/workflow_library.py`/`WorkflowBuilder._slots()`, `modules/config.py`, `observability/generation_trace.py`. Does not redesign `WorkflowGraphAssembler`'s attachment mechanism, `WorkflowLibrary`'s niche/edit-mode resolution, or any Module 7 V2 staged-edit stage — all treated as fixed, correct, reused contracts, consistent with every prior document in this repository's `docs/` tree.

---

## 0. Grounding note

Per the brief's own background, three prior architectures are cited as "already implemented": `MODULE7_PHASE2_COMFYUI_INTEGRATION.md`, `MODULE7_RENDER_EXECUTION_ARCHITECTURE.md`, `MODULE7_EDIT_MODE_ACTIVATION_FIX.md`. Checked directly against the current repository rather than assumed:

- A file named `MODULE7_PHASE2_COMFYUI_INTEGRATION_DESIGN.md` exists under `docs/` (a near-match to the brief's filename) and a file named `MODULE7_EDIT_MODE_ACTIVATION_FIX_ARCHITECTURE.md` exists under `docs/` (also a near-match) — both real.
- **No file named `MODULE7_RENDER_EXECUTION_ARCHITECTURE.md` exists under `docs/` on `main`.** That document was produced in this session but was never committed to the repository — this document does not treat its findings as verified-on-`main` fact, and re-confirms independently, below, whichever of its claims are load-bearing for this document's own scope.
- **The specific claim "the renderer now correctly operates in staged_edit mode" / "image staging has already been fixed" is checked directly and found to be only partially true.** `modules/config.py`'s `MODULE7_PROFILE_PREFERENCE` does include `PROFILE_STANDARD_EDIT` (the profile-reachability fix is real, on `main`), and `effective_edit_mode` is genuinely threaded into `_process_single_candidate()` and `WorkflowLibrary.resolve()` (the mode-forwarding fix is real, on `main`). **However, every `workflows/*_edit.json` template's `KSampler` node still hardcodes `"denoise": 1.0`** (`grep -n "denoise" workflows/general_edit.json` → `"denoise": 1.0,`, confirmed identically across all eleven edit templates) — the specific defect that nullifies source-image conditioning regardless of correct template/fragment selection is still present on `main` as of this review.
- This matters directly for this document's scope: **this document's ControlNet fix is independent of, and does not depend on, that unresolved defect being fixed first.** ControlNet conditioning attaches at `positive_conditioning` (§3), not at the `latent_image`/`denoise` edge — so the ControlNet capability-resolution problem this document solves is real and worth solving regardless of whether the separate `denoise` defect has shipped. This document does not attempt to fix that defect (out of scope, per the brief's "do not redesign the editing pipeline" boundary already established by prior documents) and flags it here only so a reader does not assume "ControlNet works once this document ships" implies "the renderer now edits correctly" — it does not, independently, per §0's finding.

---

## 1. Executive Summary

Every ControlNet fragment (`workflows/fragments/controlnet_depth.json`, `controlnet_canny.json`, `controlnet_segmentation.json`) hardcodes a literal `control_net_name` string — `"controlnet_depth_sdxl.safetensors"` and equivalents — inside a `ControlNetLoader` node. This value is not a parameter of thumbnail-ai's own logic; it is the exact filename ComfyUI's `ControlNetLoader` node expects to find, byte-for-byte, in that installation's `models/controlnet/` directory. Different ComfyUI installations populate that directory differently — different official re-uploads, different community naming conventions, Control-LoRA variants, T2I-Adapter variants — so a filename that is correct on one machine is routinely absent on another, and the failure only manifests as a ComfyUI-side node execution error at generation time, not as a thumbnail-ai-side validation error before generation is attempted.

This document designs a **Capability Resolution Layer**: the pipeline requests a semantic capability ("depth conditioning"), a new discovery component reads what ComfyUI's own `/object_info` endpoint already reports as actually installed on that machine (an HTTP call thumbnail-ai already makes today, for a narrower purpose — §3), and a small, deterministic, config-defined resolver maps the requested capability to whichever installed model best satisfies it, in a documented, transparent, PORCE-traceable priority order. Workflow fragments stop naming files and start naming capabilities; the pipeline validates at startup that every capability it needs is satisfiable and fails loudly, with a specific remediation message, before any generation attempt — rather than failing inside a ComfyUI node mid-run.

---

## 2. Root Cause Analysis

**Why hardcoded model filenames are architecturally incorrect — precisely, not generally:**

1. **ComfyUI's `ControlNetLoader` node does not have a fixed, versioned vocabulary of valid `control_net_name` values.** Its `INPUT_TYPES()` populates that field's combo list dynamically, at `/object_info` request time, from whatever files are physically present in the server's configured `models/controlnet/` search path(s) at that moment. This is not a thumbnail-ai integration detail — it is how ComfyUI itself is designed to work, precisely because model filenames are a local installation concern, not a workflow-graph concern. A workflow JSON that embeds a literal filename is, by ComfyUI's own design intent, embedding an installation-specific fact into a portable artifact.
2. **The filename space is not just "different names for the same file" — it spans genuinely different model families.** The brief's own examples — `controlnet-sd-xl-1.0-depth`, Control-LoRA variants, T2I-Adapter variants — are not merely renamed copies of the same weights. Official SDXL ControlNet and Stability's Control-LoRA are both, in ComfyUI, typically loaded via the same `ControlNetLoader`/`ControlNetApply` node pair (Control-LoRA ships as a `.safetensors` compatible with the same loader), so for those two, a filename-level substitution is sufficient. **T2I-Adapter is not** — ComfyUI exposes T2I-Adapter models through a distinct node class family (`T2IAdapterLoader`, and in unified/current ComfyUI builds, adapter-specific apply semantics that differ from `ControlNetApply`'s conditioning-strength model). This is a real, load-bearing constraint on the design (§5): **capability resolution cannot always be pure filename substitution inside a fixed graph shape** — for node-class-incompatible variants, the *fragment itself* (not just one field inside it) must vary. Any design that only threads a resolved filename into an otherwise-fixed fragment (as the brief's own Phase 3 example literally shows) is correct for same-node-class variants and silently wrong for different-node-class variants; §5 designs for this explicitly rather than glossing over it.
3. **thumbnail-ai already fetches the exact data needed to solve this, but currently discards the relevant part of it.** `CapabilityProbe.installed_node_types()` (`modules/generation_components/capability_probe.py:167-196`) already calls `self.client.object_info()` — which already returns, per node class, the full `input.required.<field>` combo enumeration, including `ControlNetLoader`'s live `control_net_name` file list — but the method immediately reduces the response to `frozenset(info.keys())` (line 184), keeping only top-level node class names and discarding every combo list. This is precisely why `is_fragment_supported()` (line 198) currently validates "is `ControlNetLoader` installed as a node class" (almost always true if the ControlNet extension is installed at all) but never validates "does the specific `control_net_name` value this fragment hardcodes actually appear in that node's live combo list" — the exact gap this document closes, reusing data thumbnail-ai already retrieves rather than adding a new API call.
4. **The failure mode this produces is specifically bad because it is silent until late.** `is_fragment_supported()` passing does not mean the fragment will execute — it means only that the *node class* exists. The filename mismatch is discovered by ComfyUI at graph-execution time, after a video has gone through Modules 1–10.5, after a ComfyUI queue slot was consumed, as a mid-run execution error — exactly the "runtime failures despite the environment being correctly configured" symptom in the brief, and exactly the failure-timing problem §4/§8 of this document are designed to move earlier.

---

## 3. Current Architecture

- **`workflows/fragments/controlnet_depth.json` / `controlnet_canny.json` / `controlnet_segmentation.json`** — each attaches at `positive_conditioning`, each contains a `LoadImage` (the conditioning map, e.g. `{{depth_map_path}}`), a `ControlNetLoader` with a literal `control_net_name`, and a `ControlNetApply` wiring the loader's output and the loaded map into the conditioning chain at `{{controlnet_{type}_strength}}`. All three fragments are structurally identical apart from the literal filename and strength placeholder name (§2's finding 3 confirms this literal is the only non-portable part of an otherwise well-built fragment).
- **`CapabilityProbe`** (`modules/generation_components/capability_probe.py`) — real, working, node-class-level capability checking, gated by `MODULE7_CAPABILITY_PROBE_ENABLED` (default `True`), cached for `MODULE7_CAPABILITY_PROBE_CACHE_SECONDS` (default 300s) via `time.monotonic()`-based TTL. `client.object_info()` is the single HTTP call this whole layer is built on (`ComfyUIClient.object_info()`, `modules/comfyui_client.py:1714-1719`, itself wrapping `_HTTPTransport.object_info()` at line 1519, a plain `GET /object_info`). `is_fragment_supported()` drops a fragment (with a logged warning and a `UnsupportedNodeTypeWarning`) if any required node class is missing — but, per §2, never inspects combo-list contents.
- **`WorkflowBuilder._select_fragments()` / `_slots()`** (`modules/image_generator.py`) — the existing, established extension points for "which fragments attach" and "what values fill their placeholders" respectively; already used by every other conditioning mechanism (§7 of `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md`'s conditioning-role table). This document's Phase 3 (§7) extends both, adding no new mechanism.
- **`validate_module7_edit_reachability()`** (`modules/config.py:463-473`) — the existing precedent for "raise a typed `Module7Error` at startup if a configured capability is structurally unreachable," introduced by `MODULE7_EDIT_MODE_ACTIVATION_FIX_ARCHITECTURE.md` for edit-mode/profile reachability. §8 of this document adds a sibling validator following the identical shape, not a new validation paradigm.
- **`GenerationTraceRecord`** (`observability/models.py`, populated by `observability/generation_trace.py::GenerationTraceFactory.create()`) — already carries `fragments_attached: list[FragmentAttachmentRecord]` with a `strength_or_weight` field per attached fragment (per `PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md`'s original `GenerationTraceRecord` design). §11 of this document extends `FragmentAttachmentRecord`, additively, with capability-resolution fields.

---

## 4. Proposed Architecture

```
                                    ┌────────────────────────────────────────┐
                                    │   ModelDiscoveryService (Phase 1, §5)    │
                                    │   reads ComfyUIClient.object_info(),     │
                                    │   extracts per-node-class combo lists    │
                                    │   (not just class names, unlike today's  │
                                    │   CapabilityProbe.installed_node_types)  │
                                    └───────────────────┬──────────────────────┘
                                                        │ installed models, per node class
                                                        ▼
                                    ┌────────────────────────────────────────┐
                                    │   ControlNetCapabilityResolver           │
                                    │   (Phase 2, §6)                          │
                                    │   capability name → ordered pattern list │
                                    │   → (node_class_family, resolved_filename,│
                                    │      resolution_source)                  │
                                    └───────────────────┬──────────────────────┘
                                                        │
                        ┌───────────────────────────────┼──────────────────────────────┐
                        ▼                               ▼                              ▼
        ┌─────────────────────────┐    ┌─────────────────────────────┐   ┌─────────────────────────┐
        │ Startup Validation        │    │ Workflow Resolver (Phase 3)   │   │ GenerationTrace (Phase 5) │
        │ (Phase 4, §8)              │    │ fragment selection + slot     │   │ resolution decision       │
        │ fail fast, typed error,    │    │ substitution (§7)             │   │ recorded per attempt (§11)│
        │ PORCE-reportable (§9/§10)  │    │                               │   │                           │
        └─────────────────────────┘    └─────────────────────────────┘   └─────────────────────────┘
```

Every box is either new (`ModelDiscoveryService`, `ControlNetCapabilityResolver`) or an additive extension of an existing, real component (`CapabilityProbe`, `WorkflowBuilder`, `config.py`'s validator family, `GenerationTraceRecord`). No existing component's existing responsibility is redesigned.

---

## 5. Component Design

### 5.1 `ModelDiscoveryService` (new, `modules/generation_components/model_discovery_service.py`)

A sibling to `CapabilityProbe`, not a replacement — same file location convention, same `client`/`enabled`/`cache_ttl_seconds` constructor shape, same fail-soft posture (§2's finding 3 means this component reuses `CapabilityProbe`'s *already-cached* `object_info()` payload rather than issuing a second HTTP call; the two components should share one cache, not duplicate the round-trip — see §12).

**Responsibility:** given the full `/object_info` payload, return, per relevant loader node class, the list of installed model filenames ComfyUI itself reports for that node's relevant combo field:

```
installed_models_for(node_class: str, field_name: str) -> tuple[str, ...]
```

e.g. `installed_models_for("ControlNetLoader", "control_net_name")` reads `object_info["ControlNetLoader"]["input"]["required"]["control_net_name"][0]` (ComfyUI's own `INPUT_TYPES()` convention: a combo field's value is a two-element list, `[options_list, config_dict]`) and returns it as a tuple. Also exposes `installed_models_for("T2IAdapterLoader", "t2i_adapter_name")` and equivalents for other node-class families a capability might resolve to (§5.2), so the design is not ControlNet-specific in shape even though this document's scope is ControlNet-specific in application — a future LoRA/upscale-model discovery need (§16) reuses this exact method, not a new one.

**Failure posture:** if `node_class` is absent from the payload (extension not installed at all) or the field is absent/malformed, return an empty tuple — never raise. Distinguishing "extension not installed" from "installed but this specific file missing" is the resolver's job (§6), not the discovery service's — the discovery service's contract is strictly "report what ComfyUI itself reports, faithfully, or report nothing."

### 5.2 `ControlNetCapabilityResolver` (new, `modules/generation_components/controlnet_capability_resolver.py`)

**Responsibility:** given a capability name (`"depth"`, `"canny"`, `"segmentation"`) and `ModelDiscoveryService`'s output, return a `ResolvedCapability`:

```
ResolvedCapability:
  capability: str                         # "depth"
  node_class: str                         # "ControlNetLoader" | "T2IAdapterLoader" | ...
  filename_field: str                     # "control_net_name" | "t2i_adapter_name"
  resolved_filename: str | None           # the actual filename to use, or None if unresolved
  resolution_source: Literal[
      "legacy_exact_match",               # matched the historical hardcoded literal exactly — §9's
                                            # backward-compatibility guarantee made visible/auditable
      "pattern_match",                    # matched a configured pattern for this capability
      "unresolved",                       # no installed model satisfied any configured pattern
  ]
  matched_pattern: str | None             # which pattern (by name, not raw regex) matched, for diagnostics
  fragment_variant: str                   # which fragment file family to attach, §7
```

**Resolution algorithm — deterministic, table-driven, no ML/fuzzy matching (Design Principle already established by every prior document in this repository: "closed, config-defined table, not open-ended inference"):**

```
CONTROLNET_CAPABILITY_TABLE: dict[str, tuple[CapabilityCandidate, ...]]
```

Each `CapabilityCandidate` is `(pattern_name: str, node_class: str, filename_field: str, fragment_variant: str, filename_regex: Pattern[str])`, and the tuple for a capability is **priority-ordered** — first match wins. For `"depth"`:

```
1. ("legacy_sdxl_official", "ControlNetLoader", "control_net_name", "controlnet_depth",
    re.compile(r"^controlnet_depth_sdxl\.safetensors$"))          # today's exact literal, first —
                                                                     # guarantees zero behavior change
                                                                     # for any installation that already
                                                                     # has exactly this file (§9)
2. ("sdxl_1_0_official_alt_naming", "ControlNetLoader", "control_net_name", "controlnet_depth",
    re.compile(r"controlnet.?sd.?xl.?1\.0.?depth", re.I))
3. ("control_lora_depth", "ControlNetLoader", "control_net_name", "controlnet_depth",
    re.compile(r"control.?lora.?depth", re.I))
4. ("t2i_adapter_depth", "T2IAdapterLoader", "t2i_adapter_name", "controlnet_depth_t2iadapter",
    re.compile(r"t2i.?adapter.?depth", re.I))
```

Analogous tables for `"canny"` and `"segmentation"`, each seeded first with today's exact literal (`controlnet_canny_sdxl.safetensors`, `controlnet_seg_sdxl.safetensors`). The resolver walks the tuple in order, checks `ModelDiscoveryService.installed_models_for(candidate.node_class, candidate.filename_field)` for any installed filename matching `candidate.filename_regex`, and returns the first hit. If none of a capability's candidates match, `resolution_source="unresolved"`, `resolved_filename=None` — never a silent guess.

**Why priority order encodes real preference, not just fallback order:** official SDXL ControlNet and Control-LoRA are trained differently and generally warrant different `strength` defaults for comparable conditioning fidelity — placing official-model patterns first is a real, if approximate, quality preference and not merely "prefer whatever thumbnail-ai historically assumed." This document does not attempt to auto-tune per-variant strength (out of scope — a future extension, §16); it only orders *which model is selected*, not what strength it's applied at, which remains `{{controlnet_{capability}_strength}}`'s existing package/profile-driven value, unchanged.

---

## 6. Data Flow

```
ComfyUI /object_info  ──▶  CapabilityProbe.installed_node_types() [existing, unchanged]
        │                        (top-level class names only, existing use unchanged)
        │
        └────────────────▶  ModelDiscoveryService.installed_models_for(node_class, field)
                                    │  (new — reads the SAME cached payload, deeper)
                                    ▼
                            ControlNetCapabilityResolver.resolve("depth")
                                    │
                                    ▼
                            ResolvedCapability{node_class, resolved_filename, resolution_source, fragment_variant}
                                    │
                    ┌───────────────┼───────────────────────┐
                    ▼               ▼                       ▼
        WorkflowBuilder      Startup Validation      GenerationTraceRecord
        ._select_fragments()  (config.py validator,  .fragments_attached[i]
        picks fragment_variant  Phase 4, §8)          (Phase 5, §11)
        ._slots() sets
        "{{resolved_depth_controlnet}}"
        = resolved_filename
```

One discovery call per pipeline run (cached, per `CapabilityProbe`'s existing TTL convention, §12), reused across every candidate/fragment resolution within that run — no per-candidate re-fetch.

---

## 7. Runtime Flow — Workflow Resolver (Phase 3)

**Fragment change:** each existing fragment's literal is replaced with a placeholder, exactly per the brief's own example:

```
Current  (workflows/fragments/controlnet_depth.json, node "20"):
  "control_net_name": "controlnet_depth_sdxl.safetensors"

Proposed:
  "control_net_name": "{{resolved_depth_controlnet}}"
```

**Fragment-variant change (§2 finding 2, §5.2):** because T2I-Adapter requires a different node class, a pure placeholder swap inside the *existing* `controlnet_depth.json` is insufficient for that variant. This document proposes a **new sibling fragment file**, `workflows/fragments/controlnet_depth_t2iadapter.json`, structurally parallel to `controlnet_depth.json` but built around `T2IAdapterLoader`/its apply node instead of `ControlNetLoader`/`ControlNetApply`, attaching at the same `positive_conditioning` point with the same `{{controlnet_depth_strength}}` slot name (so downstream slot-value computation in `_slots()` does not need to know which variant was chosen). `WorkflowBuilder._select_fragments()` — already the component responsible for choosing *which* fragment file to attach (§3) — is extended to choose between `controlnet_depth.json` and `controlnet_depth_t2iadapter.json` based on `ResolvedCapability.fragment_variant`, the exact same kind of decision it already makes for `is_edit_workflow` (§6 of `MODULE7_RENDER_EXECUTION_ARCHITECTURE.md`'s finding, independently re-confirmed here: `_select_fragments()` already branches on structural conditions to choose among named fragments — this is additive use of an existing branch point, not a new mechanism).

**Slot substitution:** `WorkflowBuilder._slots()` gains one new computed value per resolved capability actually needed by the current package/profile (mirroring exactly how `denoise_strength` is already computed as a slot value, §5 of `MODULE7_RENDER_EXECUTION_ARCHITECTURE.md`): `slots["resolved_depth_controlnet"] = resolved.resolved_filename` (and `resolved_canny_controlnet`, `resolved_segmentation_controlnet` analogously), sourced from a `ControlNetCapabilityResolver.resolve(...)` call made once per `build()` invocation (or reused from the run-level cache, §12) — not recomputed per placeholder.

**If unresolved:** per §8, an unresolved capability that a selected profile/niche actually requires should already have raised at startup, before any `build()` call — `_slots()`/`_select_fragments()` are not the layer responsible for handling an unresolved capability gracefully at generation time; they assume, by the time they run, that validation already guaranteed resolution succeeded for every capability this run will need (a clean separation of "fail early" from "build correctly," consistent with `RegionPlanValidator`'s own documented role in `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` §8 of doing all upfront validation before any stage touches GPU-bound work).

---

## 8. Validation Rules (Phase 4)

New `validate_controlnet_capability_availability()` in `modules/config.py`, structurally identical to the existing `validate_module7_edit_reachability()` (§3):

```
def validate_controlnet_capability_availability(
    resolver: ControlNetCapabilityResolver,
    required_capabilities: frozenset[str],   # derived from which profiles have controlnet_enabled=True
                                               # and which niches' templates reference which capabilities —
                                               # a static, config-derivable set, not a runtime guess
) -> None:
    unresolved = [cap for cap in required_capabilities if not resolver.resolve(cap).resolved_filename]
    if unresolved:
        raise Module7Error(
            f"ControlNet capabilities {sorted(unresolved)} could not be resolved to any installed model. "
            f"Checked patterns: {resolver.describe_patterns(unresolved)}. "
            f"Suggested fix: install one of the compatible model files listed above into ComfyUI's "
            f"models/controlnet directory, or disable controlnet_enabled for affected profiles."
        )
```

Called at the same startup checkpoint `validate_module7_edit_reachability()` is already called from (wherever `main.py`/a `tai doctor`-equivalent entry point performs pre-flight checks, per §3's precedent) — additive, not a new checkpoint. **This is the mechanism that converts §2 finding 4's "silent until mid-run" failure into a startup-time, human-readable failure** — the core deliverable of this document, stated plainly: a missing capability is now a `tai doctor`-visible fact, not a ComfyUI execution-log fact discovered after a queue slot was already spent.

**Graceful mode:** if `MODULE7_CAPABILITY_PROBE_ENABLED=False` or ComfyUI is unreachable at validation time, this validator — like `CapabilityProbe.is_fragment_supported()` (§3) — fails soft (skips the check, logs a warning) rather than blocking startup entirely, preserving the existing project-wide "capability probing is advisory when the server can't be reached, mandatory when it can" posture already established.

---

## 9. Diagnostics / PORCE Integration (Phases 4–5, §9–§11 combined per the brief's overlap)

**New PORCE rule, `RULE-EDIT-04` (following `RULE-EDIT-02`'s exact class/interface shape in `observability/diagnostics/rules/edit_mode_resolution_rules.py`, or a new sibling file `controlnet_capability_rules.py` in the same `observability/diagnostics/rules/` package, §5 of `PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md`'s file layout):**

```
RULE-EDIT-04 — ControlNet Capability Resolution Integrity
  Fires FAIL if: a GenerationTraceRecord's fragments_attached contains a ControlNet-family
    fragment whose resolution_source == "unresolved" (i.e., generation proceeded anyway,
    meaning §8's startup validation was bypassed, disabled, or stale relative to a
    since-changed ComfyUI installation)
  Fires WARNING if: resolution_source == "pattern_match" with matched_pattern outside the
    top-priority ("legacy_exact_match") tier — informational: this run used a fallback
    model, not the historically-assumed one; not a failure, but worth surfacing so a
    developer comparing output quality across machines knows the models actually differed
  Fires INFO if: resolution_source == "legacy_exact_match" — the common, expected case,
    confirming zero behavioral change from pre-this-document behavior
```

**Human-readable report (§13 of `PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md`'s existing renderer, reused, not replaced):** the "missing capability / unavailable compatible models / suggested fixes" text the brief's Phase 4 asks PORCE to report is exactly `Module7Error`'s message from §8, surfaced through PORCE's existing `human_report_renderer.py` the same way any other typed exception already becomes a `Finding` (per `PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md` §17's error-handling contract: a caught, typed error becomes a `Finding` with `severity` reflecting its blocking nature, not a silent log line) — no new reporting format needed.

---

## 10. Root Cause / Failure Classification

Reusing `MODULE7_RENDER_EXECUTION_ARCHITECTURE.md` §9's taxonomy directly, this defect classifies as:

| Class (per existing taxonomy) | This defect |
|---|---|
| Template/graph-definition defect | `control_net_name` literal hardcoded in fragment JSON, inconsistent with the fragment's own portability intent |
| Observability blind spot | `CapabilityProbe.installed_node_types()` discards the combo-list data needed to detect this, by construction (§2 finding 3) |

No new taxonomy category is required — this document's defect fits the existing classification scheme exactly, which is itself a small piece of evidence that the scheme (§9 of the prior document) generalizes correctly across investigations, as intended.

---

## 11. GenerationTrace Integration (Phase 5)

`FragmentAttachmentRecord` (existing model, `observability/models.py`, already carrying `fragment_name`, `attach_point`, `strength_or_weight` per `PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md`'s original design) gains four new, optional, additive fields — schema-compatible with every already-persisted trace (old records simply have these fields `None`, exactly the pattern `PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md` §21 already used for its own additive migrations):

```
requested_capability: str | None       # "depth"
resolved_model: str | None             # the actual filename used
resolution_source: str | None          # "legacy_exact_match" | "pattern_match" | "unresolved"
fallback_path: bool                    # True if resolution_source != "legacy_exact_match"
compatibility_decision: str | None     # one-sentence human-readable rationale, e.g.
                                         # "matched pattern 'control_lora_depth': official
                                         #  SDXL depth model not found, using Control-LoRA
                                         #  equivalent 'control-lora-depth-rank128.safetensors'"
```

Populated inside `GenerationTraceFactory.create()` (`observability/generation_trace.py`) from the `ResolvedCapability` objects `WorkflowBuilder` already computed during `_slots()`/`_select_fragments()` (§7) — passed through the same `fragments_attached` parameter the factory already accepts (§3, `observability/generation_trace.py:43`), not a new parameter. **This composes directly with `MODULE7_RENDER_EXECUTION_ARCHITECTURE.md`'s own Phase 2 recommendation** (fixing `GenerationTraceFactory` to read `built_wf.graph` truthfully instead of hardcoding literals) — that fix and this document's trace extension touch the same function and should land together or in either order without conflict, since one corrects existing hardcoded fields (`denoise`, `latent_source`) and this one adds new fields entirely.

---

## 12. Configuration Changes

- `CONTROLNET_CAPABILITY_TABLE: dict[str, tuple[CapabilityCandidate, ...]]` — new, `modules/config.py`, following the existing "small, closed, config-defined table" convention (`MODULE7_QA_WEIGHTS`, `MODULE7_PROFILE_PREFERENCE`, `MODULE7_GENERATION_PROFILES` are all this same pattern).
- `MODULE7_CAPABILITY_DISCOVERY_ENABLED: bool = True` — new, mirrors `MODULE7_CAPABILITY_PROBE_ENABLED`'s exact shape; deliberately a *separate* flag from `MODULE7_CAPABILITY_PROBE_ENABLED` rather than reusing it, so an operator can disable node-class probing and model-filename resolution independently if one needs to be temporarily bypassed without the other.
- **`ModelDiscoveryService` and `CapabilityProbe` should share one cached `/object_info` payload**, not issue two independent HTTP calls with two independent TTL clocks — the cleanest implementation is `ModelDiscoveryService` accepting a `CapabilityProbe` instance in its constructor and reading `probe._cached_types`-equivalent raw payload (requiring `CapabilityProbe` to additionally cache the *raw* `object_info()` dict, not just its derived `frozenset(keys())`, an additive change to an existing private attribute, not a signature change) rather than being handed a `client` directly and fetching independently. This is called out explicitly because it is the one place this document's design could accidentally double the number of `/object_info` round-trips per run if implemented carelessly.
- No changes to `MODULE7_GENERATION_PROFILES`, `MODULE7_PROFILE_PREFERENCE`, or any edit-mode-related configuration — this document's scope is orthogonal to profile/edit-mode selection (§0).

---

## 13. Migration Plan

| Phase | Change | Breaking? |
|---|---|---|
| 1 | `ModelDiscoveryService` — new, unreachable until wired | No |
| 2 | `ControlNetCapabilityResolver` + `CONTROLNET_CAPABILITY_TABLE` (seeded first-priority with today's exact literals, §5.2/§9) — new, unreachable until wired | No |
| 3 | `controlnet_{depth,canny,segmentation}.json` fragments: literal → `{{resolved_*_controlnet}}` placeholder; `WorkflowBuilder._slots()`/`_select_fragments()` extended to populate it | No, provided §5.2's first-priority pattern is the exact legacy literal — any installation with the original file installed resolves to `resolution_source="legacy_exact_match"` and an identical build to today's |
| 4 | New `controlnet_depth_t2iadapter.json` (+ canny/segmentation equivalents) — new files, additive | No |
| 5 | `validate_controlnet_capability_availability()` wired into startup checks | No — a machine that already has the legacy files installed continues to validate successfully; only a machine with none of a capability's candidates installed newly fails fast, which is strictly better than today's silent-until-runtime failure, not a regression |
| 6 | `FragmentAttachmentRecord` extension + `GenerationTraceFactory` population | No — additive optional fields |
| 7 | `RULE-EDIT-04` registered in PORCE's rule engine | No — additive rule |

Every phase independently shippable and testable, following the exact phasing discipline every prior document in this repository's `docs/` tree has used.

---

## 14. Testing Strategy

- `ModelDiscoveryService`: fixture-based, synthetic `/object_info` payloads (including malformed/missing-field cases), asserting correct extraction and empty-tuple fail-soft behavior.
- `ControlNetCapabilityResolver`: table-driven per capability, asserting priority order is honored (a fixture with both the legacy filename and a Control-LoRA filename installed must resolve to the legacy one), and asserting `resolution_source="unresolved"` when no candidate matches.
- **Regression test, directly reproducing the reported symptom**: a fixture where only `controlnet-sd-xl-1.0-depth.safetensors` is "installed" (no legacy filename present) — assert the pre-this-document code path would fail (documented, not executed, since the legacy code has no resolution step to test) and the new resolver correctly resolves via the `sdxl_1_0_official_alt_naming` pattern — the concrete case named in the brief.
- `validate_controlnet_capability_availability()`: asserts it raises `Module7Error` with a message containing the missing capability name and at least one suggested-fix string, for an all-unresolved fixture; asserts it passes silently for an all-resolved fixture.
- Backward-compatibility test (mirrors `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` §16's precedent exactly): given a fixture where the legacy filename is present, assert the fully-built graph is byte-for-byte identical to the pre-this-document build.
- `RULE-EDIT-04`: table-driven, one fixture per severity tier (FAIL/WARNING/INFO), per PORCE's established rule-testing convention.

---

## 15. Risks

| Risk | Detail | Mitigation |
|---|---|---|
| Pattern table drifts out of date as new naming conventions appear | A regex-based table is only as current as its last edit | §16 — table is data, not code; a new entry is a low-cost, reviewable, non-breaking change; no different in kind from `MODULE7_QA_WEIGHTS` or `MODULE7_PROFILE_PREFERENCE` already requiring occasional manual updates |
| T2I-Adapter fragment variant (§7) is new, untested-in-production code, not merely a filename change | Genuinely more implementation risk than the ControlNet/Control-LoRA filename-substitution path | Flagged explicitly in §7 rather than presented as equivalent-risk to the filename-only cases; §13 Phase 4 is separable and can ship after Phases 1–3/5–7 if T2I-Adapter support needs more validation time |
| Double `/object_info` HTTP round-trips if `ModelDiscoveryService` and `CapabilityProbe` aren't wired to share a cache | Minor performance/latency risk, not a correctness risk | Called out explicitly in §12 as an implementation constraint, not left implicit |
| Startup validation (§8) could block a legitimate run on a machine where ComfyUI is temporarily still loading its model directory index | False-positive startup failure | Mirrors `validate_module7_edit_reachability()`'s existing fail-soft-when-unreachable posture (§8) — the validator does not run at all if ComfyUI can't be reached, only when it can be reached and genuinely reports no matching model |

---

## 16. Future Work

- Per-variant `strength` auto-tuning (Control-LoRA vs. official ControlNet may warrant different default strengths for comparable visual effect) — explicitly deferred, §5.2.
- Reusing `ModelDiscoveryService.installed_models_for()` for non-ControlNet model families already hinted at by its capability-agnostic method shape (§5.1): checkpoint discovery (`CheckpointLoaderSimple`), LoRA discovery (`LoraLoader`), IPAdapter model discovery (`IPAdapterModelLoader`) — the same silent-hardcoded-filename risk plausibly exists for `PROFILE_*`'s `checkpoint` field (`juggernautXL.safetensors`, confirmed as a literal in real `generation_metadata.json` per `MODULE7_RENDER_EXECUTION_ARCHITECTURE.md`'s evidence) and is a natural, structurally-supported follow-on, not designed here.
- Automatic Control-LoRA vs. official-ControlNet A/B quality comparison, feeding back into `CONTROLNET_CAPABILITY_TABLE`'s priority order empirically rather than by hand — a PVQEF-adjacent extension (`PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md` §24's "cross-video pattern mining" future extension is the natural home for this).
