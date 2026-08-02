# Module 7 — Render Execution Architecture (Investigation)

**thumbnail-ai**
**Status:** Architecture / investigation document only. Zero implementation code, zero tests, zero repository modification.
**Scope constraint honored throughout:** this document does not redesign Module 7, PORCE, or the editing pipeline. It documents, with cited evidence, exactly where in the already-built execution path the renderer stops conditioning on the source thumbnail — and it designs the *investigation methodology* (decision tree, evidence artifacts, diagnostic rules) as a reusable, permanent capability, not a one-off script.
**Repository state this document is grounded against:** `poison-2-0-0-7/thumbnail-ai` @ `main`, commit `09b8f97` ("feat(module7): complete staged edit activation and PORCE validation"), the tip of the branch at the time of this review.

---

## 0. Grounding note — what actually exists, verified against `main`

Per instructions, four prior documents are treated as source material where they genuinely exist on `main`:

- `docs/MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` — **exists**, real, and its §9 staged-edit design (`RegionPlanValidator` → `BaseLatentStage` → `BackgroundEditStage`/`ObjectEditStage` → `MaskedCompositeStage` → `TypographyStage` → `HarmonizationStage`) is the specification against which §6/§7 of this document check actual code.
- `docs/PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md` — **exists**, real, and its `GenerationTraceRecord`/rule-engine design is the specification against which §7/§9 of this document check the real `observability/` package.
- `docs/MODULE7_EDIT_MODE_ACTIVATION_FIX_ARCHITECTURE.md` — **exists**, real, and is the most directly relevant prior document: it identified and scoped a fix for "Gap A" (edit-capable profile excluded from `MODULE7_PROFILE_PREFERENCE`) and "Gap B" (resolved `effective_edit_mode` never forwarded past `run()`), explicitly deferred "Gap C" (the seven staged-edit stages constructed but never invoked) to future work, and — critically — treated the `*_edit.json` templates and `WorkflowBuilder`'s fragment-assembly mechanics as "already correct, reused contracts" (its own §5/header framing) rather than verifying their contents. §5 of this document shows that assumption was incorrect.
- `docs/MODULE7_RENDERING_ROOT_CAUSE_INVESTIGATION_ARCHITECTURE.md` — **does not exist under `docs/` on `main`** at the time of this review (confirmed: `ls docs/ | grep -i RENDERING_ROOT_CAUSE` returns no match). This document does not rely on it, does not assume its contents, and re-derives every finding attributed to "the investigation" in the brief directly from the current repository instead, exactly as the brief's own instruction to not assume anything requires.

Separately, three status claims in the brief were checked, not assumed:

- **"PROFILE_STANDARD_EDIT Activation ✓" and "RULE-EDIT-02 ✓"** — **verified true.** `modules/config.py:454-456` now reads `MODULE7_PROFILE_PREFERENCE = ("PROFILE_PREMIUM", "PROFILE_STANDARD_EDIT", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM")`, and `image_generator.py`'s `effective_edit_mode` is genuinely forwarded into `_process_single_candidate(effective_edit_mode=effective_edit_mode)` (lines 1284, 1313) and on into `self.workflow_library.resolve(niche, profile, edit_mode=effective_edit_mode)` (lines 1471, 1493). Gaps A and B, as scoped by the Activation Fix document, are closed.
- **"921 repository tests passing" / "tai doctor healthy"** — **not independently verifiable in this sandbox** (no GPU, no ComfyUI, no vendored checkpoints are present here) and this document does not assert or deny the figure. What matters for this investigation is narrower and *is* verified directly: no test file in `tests/` asserts on the numeric `denoise` value inside a built `staged_edit` workflow graph (`grep -rn "denoise" tests/test_module7_phase2_edit_mode_threading.py tests/test_module7_phase3_activation.py tests/test_module7_reachability_validation.py` returns zero matches) — so a green test suite and a healthy `tai doctor` are consistent with the defect found in §5 continuing to exist; they were never checking for it.
- **"The pipeline reports healthy"** — verified narrowly true in the sense that `RULE-EDIT-02` (`observability/diagnostics/rules/edit_mode_resolution_rules.py`) checks exactly one thing — config-level reachability of an edit-capable profile — and that check now passes. §9 of this document shows this is not the same claim as "the renderer edits the source image," and documents the specific reason PORCE cannot currently distinguish the two.

---

## 1. Problem Statement

With Gaps A and B (profile reachability, edit-mode forwarding) closed, `edit_mode="auto"` now genuinely resolves to `"staged_edit"` for VRAM budgets that fit `PROFILE_STANDARD_EDIT`, and `WorkflowLibrary.resolve()` genuinely selects `{niche}_edit.json` instead of the legacy template. Despite this, generated thumbnails continue to bear no structural resemblance to their source — random portraits, unrelated vehicles, grayscale office scenes, unrelated landscapes. This document's objective is to identify, with repository evidence rather than inference, the exact remaining point(s) in the execution path where source-image conditioning is discarded, and to design a permanent, reusable investigation capability (decision tree + evidence artifacts + diagnostic rules) so this class of defect does not require a manual, from-scratch repository archaeology exercise the next time it happens.

**Finding, stated up front and evidenced in full below:** the renderer stops behaving like an editor at a single, precise, statically-verifiable location — **the literal JSON value `"denoise": 1.0` hardcoded into node `"5"` (`KSampler`) of every one of the eleven `workflows/*_edit.json` templates** — reached correctly (Gaps A/B are real fixes), fed a correctly VAE-encoded source latent (the `inpaint_base` fragment is real and correctly built), and then immediately discarded by full-strength resampling before the graph is even submitted to ComfyUI. A second, independent, already-known-and-documented gap (Gap C: the seven staged-edit Python stages are constructed but never called) compounds this. A third finding, specific to this investigation, is that the observability layer meant to catch exactly this cannot currently do so, by construction — detailed in §5, §6, and §9.

---

## 2. Current Renderer Architecture

Unchanged from, and consistent with, `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md`'s own description of what exists versus what was designed — restated here only to the depth needed to ground §3–§9's evidence:

- **Profile layer** (`modules/config.py`, `modules/models.py`) — five `GenerationProfile` instances; one, `PROFILE_STANDARD_EDIT`, declares `edit_mode_default="staged_edit"`.
- **Selection layer** (`modules/image_generator.py::ProfileSelector`) — VRAM-based, preference-ordered, edit-mode-agnostic by design (per the Activation Fix document's Design Goal 2, which this document does not revisit).
- **Mode-resolution layer** (`ImageGeneratorPipeline.run()`, lines 1200–1220) — computes `effective_edit_mode` once per `run()` call, from the selected profile.
- **Template-resolution layer** (`WorkflowLibrary.resolve()`, `modules/workflow_library.py:82-128`) — resolves `{niche}_edit.json` when `effective_edit_mode == "staged_edit"`.
- **Graph-assembly layer** (`WorkflowBuilder.build()` → `WorkflowGraphAssembler.assemble()`, `generation_components/workflow_graph_assembler.py`) — merges fragments from `workflows/fragments/*.json` into the base template via a namespaced-node, `ATTACHMENT_PREVIOUS`-sentinel splice mechanism.
- **Per-candidate orchestration layer** (`ImageGeneratorPipeline._process_single_candidate()`, lines 1449–1560+) — the actual, single, linear sequence of operations executed per generated candidate: resolve template → build graph → submit to ComfyUI → identity check/retry → face restoration → background composite → upscale → (QA scoring, ranking, downstream of the excerpt shown).
- **Trace-capture layer** (`observability/generation_trace.py::GenerationTraceFactory.create()`) — constructs the `GenerationTraceRecord` PORCE persists per attempt.

---

## 3. Execution Flow

Traced directly against the code, for a single candidate, under `effective_edit_mode == "staged_edit"` (i.e., the now-activated case):

```
1. ImageGeneratorPipeline.run()
     → profile = ProfileSelector.select(...)                          [may now select PROFILE_STANDARD_EDIT]
     → effective_edit_mode = "staged_edit"                            [Gap B: now correctly computed AND forwarded]
2. ImageGeneratorPipeline._process_single_candidate(..., effective_edit_mode="staged_edit")
     → workflow_ref = WorkflowLibrary.resolve(niche, profile, edit_mode="staged_edit")
         → resolves "{niche}_edit.json" (e.g. general_edit.json)       [confirmed correct, §7]
     → built_wf = WorkflowBuilder.build(package, profile, workflow_ref, ...)
         → base_unsubstituted = WorkflowBuilder.build_base(...)        [loads the JSON file from disk verbatim]
         → fragments = WorkflowBuilder._select_fragments(profile, conditioning, workflow_ref)
             → is_edit_workflow = template_name.endswith("_edit")      [True → confirmed correct, §6]
             → fragments = ["inpaint_base", "edit_region_mask", ...]   [confirmed correct, §6]
         → slots = WorkflowBuilder._slots(package, profile, references, conditioning, plan)
             → slots["denoise_strength"] = 0.75                        [computed — confirmed correct, §5]
         → final_graph = WorkflowBuilder._substitute(base_unsubstituted, slots)
             → node "5" ("KSampler") .inputs.denoise remains the LITERAL 1.0 from the on-disk JSON
               — {{denoise_strength}} is never referenced anywhere in general_edit.json's text,
               so string-substitution has nothing to substitute; the literal survives unchanged.
               ★★★ THIS IS THE POINT OF DIVERGENCE — see §5 for the exact evidence ★★★
     → raw_output = client_obj.generate(built_wf, ...)                  [submits the graph — with denoise=1.0 — to ComfyUI]
3. ComfyUI executes the submitted graph exactly as submitted:
     → LoadImage(source_thumbnail_path) → VAEEncodeForInpaint(pixels, vae, mask) → a real, correctly-encoded latent
     → KSampler(latent_image=<that encoded latent>, denoise=1.0) → samples as if the input were pure noise
       (denoise=1.0 is defined, in ComfyUI/Stable-Diffusion sampling semantics, as "ignore the input latent's
       content entirely and generate purely from the conditioning/noise schedule" — the specific numeric value
       that makes an encoded-but-then-fully-resampled latent behave, for practical output purposes, identically
       to EmptyLatentImage's noise latent)
     → VAEDecode → a decoded image with no structural dependency on the LoadImage'd source thumbnail
4. self.identity_stage.verify(...) → self.restoration_stage.restore(...) →
   self.background_compositor.composite(...)                            [confirmed no-op, §6] →
   self.upscale_stage.upscale(...)
   — none of RegionPlanValidator / BaseLatentStage / MaskedCompositeStage / BackgroundEditStage /
     ObjectEditStage / TypographyStage / HarmonizationStage participate at any point — confirmed §6.
5. GenerationTraceFactory.create(..., built_wf=built_wf, ...) records
   latent_source="noise", denoise=1.0, edit_mode="txt2img" — literal Python constants,
   never read from built_wf.graph — confirmed §7. The one artifact designed to make step 2's
   defect visible cannot see it, by construction.
```

---

## 4. Runtime Pipeline Analysis

Per the brief's "trace the complete execution path" table (Inputs / Outputs / Runtime State / Expected Behaviour / Actual Behaviour / Failure Modes), condensed to the stages where Expected and Actual diverge — every stage not listed here was checked and found to match its documented design (§2's layers 1–4 for Gaps A/B are confirmed fixed, §0):

| Stage | Inputs | Expected Behaviour (per `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md`) | Actual Behaviour (verified) | Failure Mode |
|---|---|---|---|---|
| Workflow template selection | `niche`, `profile`, `effective_edit_mode="staged_edit"` | Select an image-to-image/inpaint-capable graph | **Matches.** `general_edit.json` (or niche equivalent) is genuinely selected. | None — this stage is correct. |
| Fragment selection | `workflow_ref`, `conditioning` | Attach `inpaint_base` (real `VAEEncodeForInpaint`) and `edit_region_mask` | **Matches.** `_select_fragments()` correctly detects `is_edit_workflow` and appends both fragments (`image_generator.py:337-342`). | None — this stage is correct. |
| Slot computation | `package`, `profile` | Compute a partial-denoise strength for the sampler | **Matches.** `_slots()` computes `"denoise_strength": 0.75` (`image_generator.py:408`). | None — this stage is correct. |
| **Graph substitution / KSampler denoise** | `base_unsubstituted` (the on-disk `_edit.json`), `slots` | The KSampler node's `denoise` should reflect `slots["denoise_strength"]` | **Diverges.** The literal `"denoise": 1.0` in every `_edit.json` template (§5) is never a `{{denoise_strength}}` placeholder, so substitution has nothing to act on; the literal `1.0` is submitted to ComfyUI unchanged. | **Root cause.** See §5, §10. |
| ComfyUI sampling | The submitted graph, including the correctly-encoded latent and `denoise=1.0` | Partial resample, structurally anchored to the source | Full resample from the conditioning/noise schedule, structurally decoupled from the source | Direct consequence of the row above — not a separate defect. |
| Per-candidate stage orchestration | Raw ComfyUI output | Route through `MaskedCompositeStage` (paste-back guarantee) etc. per the five-stage §9 design | **Diverges.** `_process_single_candidate()` calls only `identity_stage` → `restoration_stage` → `background_compositor` → `upscale_stage`; the seven staged-edit objects (§2) are never referenced (`grep -n "self\.<attr>\." modules/image_generator.py` returns zero hits for all seven, §6). | **Compounding cause — Gap C**, already known and explicitly deferred by the Activation Fix document (§0). |
| Background compositing | Restored candidate image, `ReferenceAssets` | Composite preserved subject over new background (per its own docstring) | **Diverges.** `BackgroundCompositor.composite()` (`image_generator.py:690-714`) opens the already-generated image and re-saves it unchanged whenever a source thumbnail exists — no mask lookup, no pixel operation. | **Independent, pre-existing defect**, first identified in `PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md` §A.2, unaffected by any fix to date. |
| Generation trace capture | `built_wf`, `profile`, `conditioning_ctx` | Record the actual `latent_source`/`denoise`/`edit_mode` used | **Diverges.** `GenerationTraceFactory.create()` hardcodes `latent_source="noise"`, `denoise=1.0`, `edit_mode="txt2img"` as Python literals (`observability/generation_trace.py:126-135`), never reading `built_wf.graph`. | **Observability blind spot** — see §7, §9. |

---

## 5. ComfyUI Workflow Analysis

Every workflow JSON reachable from `WorkflowLibrary` was inspected directly (not sampled) — eleven legacy templates (`workflows/{niche}.json` / `general.json`) and eleven edit templates (`workflows/{niche}_edit.json` / `general_edit.json`), plus all nine files in `workflows/fragments/`.

**Legacy templates** (`workflows/general.json` and ten niche equivalents) — unchanged since `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md`'s original review: node `"4"` is `EmptyLatentImage`; node `"5"` (`KSampler`) has `"denoise": 1.0` and `"latent_image": ["4", 0]`; no `LoadImage`, no `VAEEncode` of any kind exists anywhere in these files. These are not in scope for the "editor" claim — they are `legacy_txt2img`'s templates by design and are working exactly as designed.

**Edit templates** (`workflows/general_edit.json`, verified in full; the other ten `{niche}_edit.json` files verified via `grep -n "denoise" workflows/*_edit.json`, all eleven returning `"denoise": 1.0` at an identical structural position, line 52 or 72 depending on file length):

- Node `"4"` is **still `EmptyLatentImage`**, with `_meta.attachment_points.latent_source` pointing at `["5", "latent_image"]` — the *same* edge as the plain `latent` attachment point. Declaring a `latent_source` attachment point is meaningful only if a fragment attaching there actually changes what feeds `KSampler.inputs.latent_image` at graph-assembly time — which requires `WorkflowGraphAssembler` to *rewrite* that edge when a fragment attaches (its documented behavior, §7 of `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md`) rather than merely *add* new, disconnected nodes alongside the untouched base graph.
- **`workflows/fragments/inpaint_base.json`** — real and correctly built: node `"10"` is `LoadImage(source_thumbnail_path)`, node `"20"` is `LoadImage(edit_mask_path)`, node `"30"` is `VAEEncodeForInpaint(pixels=["10",0], vae=["1",2], mask=["20",1], grow_mask_by=6)`, with `_attach.point="latent_source"`, `_attach.output_node="30"`. This fragment, in isolation, is a genuine, well-formed inpainting subgraph.
- **`workflows/fragments/edit_region_mask.json`** — also real and correctly built: `LoadImage(edit_mask_path)` → `SetLatentNoiseMask(samples=["ATTACHMENT_PREVIOUS",0], mask=[...])`.
- **The KSampler node itself, node `"5"`, is untouched by either fragment.** Both fragments attach at `latent_source`, meaning (per §7 of `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md`'s design and consistent with `WorkflowGraphAssembler`'s attachment mechanism) node `"5"`'s `latent_image` input is rewired from `["4",0]` to the fragment chain's final output (`inpaint_base`'s node `"30"`, or, if `edit_region_mask` is also attached, its node `"20"` chained on top). **`node "5"`'s `denoise` field is never part of any fragment's `_attach` target and is never referenced by `{{denoise_strength}}` anywhere in the base template's own text** — it remains the literal `1.0` written into the JSON file on disk, regardless of which fragments attach.

**Direct answers to the brief's ComfyUI-analysis questions, per template class:**

| Question | Legacy (`{niche}.json`) | Edit (`{niche}_edit.json`) |
|---|---|---|
| Load Image present? | No | **Yes** — via `inpaint_base` fragment, when attached |
| VAE Encode present? | No | **Yes** — `VAEEncodeForInpaint`, via `inpaint_base` fragment, when attached |
| Mask inputs present? | No | **Yes** — via `edit_region_mask` fragment, when attached |
| EmptyLatentImage still used? | Yes (node `"4"`) | **Yes — node `"4"` is still present in the base template file itself**; whether it is actually the latent that reaches KSampler depends on whether `latent_source`-attaching fragments were selected (they are, per §6, when `is_edit_workflow` is true) |
| Which latent reaches KSampler? | `EmptyLatentImage`'s noise (`["4",0]`), always | **The VAE-encoded, masked source latent** (`inpaint_base`/`edit_region_mask`'s chain), when fragments attach — **but immediately, fully overwritten by `denoise: 1.0` at the same node** |
| Which image reaches ControlNet? | The source thumbnail, via `controlnet_depth`/`controlnet_canny`/`controlnet_segmentation` fragments, when `profile.controlnet_enabled` and the relevant conditioning map exists (unchanged mechanism, both template classes) | Same as legacy |
| Which image reaches IPAdapter? | Face/object reference crops, via `ipadapter_reference`, when `profile.ipadapter_enabled` (unchanged mechanism, both template classes) | Same as legacy |
| Which latent reaches VAE Decode? | `KSampler`'s output, which (denoise=1.0, noise input) has no structural dependency on any source image | **`KSampler`'s output — which, despite receiving an encoded source latent as input, has denoise=1.0 and therefore also has no structural dependency on the source image.** This is the crux finding: **the edit template's `VAEDecode` output is, for practical purposes, exactly as decoupled from the source thumbnail as the legacy template's is**, because the one parameter that would make the difference between "encoded latent, partially resampled" and "encoded latent, immediately discarded" is hardcoded to the "discarded" value in every edit template. |

---

## 6. Python Orchestration Analysis

Traced directly against `modules/image_generator.py`, per the brief's explicit stage list:

| Stage | Instantiated in `__init__`? (`image_generator.py:1162-1198`) | Ever called via `self.<attr>.<method>(...)` anywhere in the file? | Verdict |
|---|---|---|---|
| `RegionPlanValidator` | Yes (`self.region_plan_validator`) | **No** — `grep -n "self\.region_plan_validator\." modules/image_generator.py` returns zero matches | Constructed, unreachable |
| `BaseLatentStage` | Yes (`self.base_latent_stage`) | **No** | Constructed, unreachable |
| `MaskedCompositeStage` | Yes (`self.masked_composite_stage`) | **No** | Constructed, unreachable |
| `BackgroundEditStage` | Yes (`self.background_edit_stage`) | **No** | Constructed, unreachable |
| `ObjectEditStage` | Yes (`self.object_edit_stage`) | **No** | Constructed, unreachable |
| `TypographyStage` | Yes (`self.typography_stage`) | **No** | Constructed, unreachable |
| `HarmonizationStage` | Yes (`self.harmonization_stage`) | **No** | Constructed, unreachable |
| `IdentityPreservationStage` | Yes (`self.identity_stage`) | **Yes** (`_process_single_candidate`, line 1516, plus the identity-retry loop 1520-1534) | Real, executes every run |
| `FaceRestorationStage` | Yes (`self.restoration_stage`) | **Yes** (line 1540) | Real, executes every run |
| `BackgroundCompositor` | Yes (`self.background_compositor`) | **Yes** (line 1545) — **but its own body is a no-op, §4** | Executes, but does nothing |
| `UpscaleStage` | Yes (`self.upscale_stage`) | **Yes** (line 1550) | Real, executes every run |

**This confirms, independently of the Activation Fix document's own admission (§0), that Gap C is fully open.** Whatever `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` §9 designed those seven stages to do — per-region masked sampling, base-latent anchoring, deterministic typography, seam harmonization, and, most importantly for this investigation, the pixel-exact `KEEP`-region paste-back guarantee (`MaskedCompositeStage`) — none of it executes in `_process_single_candidate()`'s actual, linear call sequence. The entire "staged edit" behavior that reaches the renderer today is limited to *which ComfyUI graph gets built* (§5) — no Python-side per-region logic runs before or after that single graph submission.

`WorkflowBuilder._select_fragments()` (`image_generator.py:327-358`) — the one piece of Gap-C-adjacent logic that *does* run — is correctly implemented: `is_edit_workflow` is computed from `workflow_ref.template_name.endswith("_edit")` (line 338), and when true, both `inpaint_base` and `edit_region_mask` are unconditionally appended (lines 341-342) before the usual ControlNet/IPAdapter/mask-fragment checks. This is the one component in the whole staged-edit chain that behaves exactly as `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` §7 specified, without qualification.

---

## 7. Runtime Graph Analysis

Comparing the brief's three named artifacts directly:

**1. Workflow JSON stored on disk** (`workflows/general_edit.json` et al.) — `EmptyLatentImage` present at node `"4"`; `KSampler` at node `"5"` with `denoise: 1.0` and `latent_image: ["4",0]`; no `LoadImage`/`VAEEncode` node present in the base file itself (§5).

**2. Workflow graph after Python modification** (`WorkflowBuilder.build()`'s `final_graph`, i.e. `built_wf.graph`) — base graph + attached fragment nodes, per `WorkflowGraphAssembler.assemble()`'s splice mechanism. Node `"5"`'s `latent_image` input is (per the attachment-point rewrite mechanism, §5) redirected to the fragment chain's output when `inpaint_base`/`edit_region_mask` attach. **Node `"5"`'s `denoise` field is unaffected by this process** — no fragment declares `denoise` as an attachment target, and no slot substitution reaches it, so it remains `1.0` in `final_graph` exactly as it was on disk.

**3. Workflow graph actually submitted to ComfyUI** (`client_obj.generate(built_wf, ...)`'s payload) — `ComfyUIClient.generate()` submits `built_wf.graph` as-is (this document does not re-verify `ComfyUIClient`'s own transport code, which is out of scope per the "do not redesign" instruction and was not implicated by any evidence found); it is, by construction, identical to artifact 2.

**The three artifacts differ exactly where expected (node `"5"`'s `latent_image` edge, between artifacts 1 and 2, when fragments attach) and are identical exactly where the defect lives (node `"5"`'s `denoise` value, across all three artifacts, always `1.0`).** This is the single cleanest, most falsifiable statement of the root cause this investigation produced: **diffing artifact 1 against artifact 2 for any `staged_edit` run will show a changed `latent_image` edge and an unchanged `denoise` value, every time, deterministically, with no dependency on prompt content, seed, or niche** — because the change (fragment attachment) and the non-change (denoise) are both static properties of the on-disk JSON files, not runtime-dependent computations.

**A fourth artifact the brief implicitly asks for — what the trace-capture layer *reports* happened — was checked and found to be actively misleading, not merely incomplete.** `GenerationTraceFactory.create()` (`observability/generation_trace.py:118-135`) accepts `built_wf` as a parameter but only reads `built_wf.workflow_hash` and `built_wf.workflow_ref.template_name` from it (lines 54-58) — never `built_wf.graph`. Its returned `GenerationTraceRecord` unconditionally sets `latent_source="noise"`, `denoise=1.0`, `edit_mode="txt2img"` (lines 126-127, 135) as **Python literal constants**, regardless of whether `effective_edit_mode` was `"legacy_txt2img"` or `"staged_edit"`, and regardless of what `final_graph` actually contained. This means artifact 4 (the trace) is currently **incapable of ever reporting anything other than "legacy txt2img"**, even on a `staged_edit` run where artifacts 1–3 show real fragment attachment. This is not a rounding-error or edge-case gap — it is a hardcoded constant that makes the one artifact designed to answer this investigation's central question structurally unable to answer it.

---

## 8. Evidence Collection Strategy

Two categories of evidence, kept explicitly distinct because they have different reliability characteristics and different verification costs — a distinction this investigation's own findings (§5–§7) depend on:

**A. Static evidence** — direct inspection of files that exist on disk independent of any pipeline run: workflow JSON templates, fragment JSON files, `config.py`'s profile/preference definitions, `image_generator.py`'s/`workflow_library.py`'s source code. This category requires no GPU, no ComfyUI process, no live pipeline execution, and is **fully deterministic and reproducible by any future investigator with only `git clone` and `grep`/`cat`** — every finding in §5, §6, and most of §7 of this document is static evidence, precisely because it is the strongest, cheapest, and most falsifiable evidence available. This document's central finding (the hardcoded `denoise: 1.0`) was established entirely as static evidence, without running the pipeline once.

**B. Runtime evidence** — artifacts produced only by actually executing the pipeline: the real `GenerationTraceRecord` per attempt, `module7_metrics.jsonl`, generated images, PORCE's `PipelineTrace`/`RootCauseReport`. This category is necessary to confirm that static findings actually manifest in production (e.g., confirming a real `PROFILE_STANDARD_EDIT` run really does produce a visually-unrelated image, not merely that the template says it will) and to catch defects that are inherently runtime-dependent (VRAM fallback paths, ComfyUI-side node execution failures, race conditions) that static inspection cannot see. **Runtime evidence for the specific fix under investigation here is currently unavailable**: the one real, on-disk `root_cause_report.json` found in this repository (`data/observability/traces/vIWkN-2J0ic/root_cause_report.json`) is timestamped from a `PROFILE_LOW_VRAM` (legacy) run, predating the Gap A/B fix — no `PROFILE_STANDARD_EDIT` run has yet produced a persisted trace in this repository, which §11/§14 flag as the first action item, not an assumption to route around.

**The investigation methodology this document establishes as reusable practice:** for any future "renderer produces X instead of Y" report, static evidence should be gathered *first* (it is cheap, deterministic, and — as this investigation demonstrates — was sufficient on its own to find and prove the entire root-cause chain in §5–§7 without a single pipeline run), and runtime evidence gathered *second*, specifically to confirm static findings manifest and to catch the residual class of defects static inspection structurally cannot reach (§9's failure classification formalizes this split).

---

## 9. Failure Classification

A taxonomy for classifying *why* a renderer output diverges from source-image conditioning, derived from the concrete failure chain found in §3–§7, generalized so it applies to future investigations of the same symptom class:

| Class | Definition | Detectable via static evidence? | Instance found in this investigation |
|---|---|---|---|
| **Configuration unreachability** | The correct code path exists but is never selected due to config data (not logic) excluding it | Yes | Gap A (fixed) — `PROFILE_STANDARD_EDIT` excluded from `MODULE7_PROFILE_PREFERENCE` |
| **Value non-propagation** | A correctly-computed value is computed but never forwarded to the component that needs it | Yes | Gap B (fixed) — `effective_edit_mode` computed but not threaded to `_process_single_candidate` |
| **Template/graph-definition defect** | The on-disk artifact that defines runtime behavior (here, a ComfyUI workflow JSON) contains a value inconsistent with its own stated purpose | Yes | **This investigation's primary finding** — `denoise: 1.0` hardcoded in every `*_edit.json`, `{{denoise_strength}}` never referenced |
| **Orchestration non-invocation** | Correctly-built components exist and are instantiated but no code path ever calls them | Yes | Gap C (open, deferred) — the seven staged-edit stages |
| **Silent no-op implementation** | A component executes without error but its implementation does not perform its documented function | Yes | `BackgroundCompositor.composite()` |
| **Observability blind spot** | The telemetry/diagnostic layer meant to detect a defect cannot detect it, by construction | Yes | `GenerationTraceFactory.create()`'s hardcoded `latent_source`/`denoise`/`edit_mode` literals |
| **Runtime-only defect** | A defect that only manifests under live execution (VRAM exhaustion mid-run, ComfyUI node execution error, nondeterministic race) and cannot be found by reading source/config alone | No — requires runtime evidence (§8B) | None found in this investigation; flagged as the remaining unverified category — §14 |

Every class found in this investigation (rows 1–6) was static evidence alone. This is itself a finding: **the reported symptom did not require a single pipeline execution to root-cause**, which materially changes the cost/turnaround of the next phase (§12) and is the primary justification for prioritizing the template fix (§12 Phase 1) over any runtime-instrumentation work.

---

## 10. Root Cause Decision Tree

A reusable tree for this class of investigation ("renderer completes without error, output doesn't resemble expected conditioning"), with this investigation's actual path through it marked `[THIS CASE]`:

```
Q1: Does the resolved profile/edit_mode actually select an editing-capable template?
    (Check: config.py profile definitions + preference ordering; run() effective_edit_mode
     computation; WorkflowLibrary.resolve()'s edit_mode argument at its actual call site —
     not just that it exists as a parameter.)
    │
    ├─ NO  → Configuration unreachability or value non-propagation (Gaps A/B class).
    │         Fix: config/wiring change only. [Confirmed already fixed for A/B, §0]
    │
    └─ YES [THIS CASE] → continue
         │
         Q2: Does the selected template file, on disk, actually differ structurally from
             the legacy template — i.e., does it contain LoadImage/VAEEncode nodes at all?
             (Check: read the JSON file directly, do not trust the filename.)
             │
             ├─ NO  → Template/graph-definition defect: an "_edit" file that isn't actually
             │         an edit graph. [Not this case — inpaint_base/edit_region_mask fragments
             │         are real and correctly built, §5]
             │
             └─ YES [THIS CASE] → continue
                  │
                  Q3: Does the KSampler node's `denoise` value allow the encoded latent's
                      content to survive sampling (i.e., denoise meaningfully < 1.0)?
                      (Check: the literal value in the template file AND whether any slot/
                      fragment mechanism overrides it at build time — do not assume a computed
                      "partial denoise" slot value is actually wired to the node that needs it.)
                      │
                      ├─ NO [THIS CASE] → ★ ROOT CAUSE FOUND ★ — denoise=1.0 nullifies the
                      │         encoded latent regardless of correct upstream wiring.
                      │         [This investigation stops here with a confirmed, sufficient,
                      │         static-evidence root cause — §12 Phase 1 fixes exactly this.]
                      │
                      └─ YES → continue
                           │
                           Q4: Do the per-region Python orchestration stages (mask-based
                               compositing, staged sampling) actually execute?
                               (Check: grep for self.<stage>.<method>( call sites, not just
                               __init__ construction.)
                               │
                               ├─ NO → Orchestration non-invocation (Gap C class).
                               │        [Also true in this case — a real, compounding,
                               │         already-documented, independently-sufficient
                               │         second defect, §6]
                               │
                               └─ YES → continue to Q5 (ControlNet/IPAdapter reference
                                        image correctness, prompt/negative-prompt
                                        contradiction, QA scoring correctness — none of
                                        these were implicated by evidence found in this
                                        investigation and are not elaborated further here,
                                        per the "do not redesign" scope boundary)
```

**Both Q3's "NO" branch (denoise hardcoding) and Q4's "NO" branch (Gap C) are independently true for the current repository state** — either alone would be sufficient to explain "renderer produces unrelated images despite correct profile/template selection," and both must be addressed (§12) because fixing only one leaves the other as a full, independent cause of the identical symptom.

---

## 11. Runtime Artifacts

Artifacts this investigation relied on or identified as needed, organized by whether they already exist correctly, exist but are unreliable, or don't yet exist:

| Artifact | Status | Notes |
|---|---|---|
| `workflows/*.json`, `workflows/fragments/*.json` (static) | **Exists, reliable** | Primary evidence source for this entire investigation |
| `modules/config.py` profile/preference definitions (static) | **Exists, reliable** | Confirms Gaps A/B fixed |
| `data/observability/traces/{video_id}/root_cause_report.json` (runtime, PORCE) | **Exists, but stale** | Only one instance in the repository, predates the Gap A/B fix, cannot speak to the current `PROFILE_STANDARD_EDIT` path — §12 Phase 0 generates a fresh one |
| `data/observability/generation_traces/{video_id}/*.json` (runtime, `GenerationTraceRecord`) | **Exists, but unreliable by construction** | Hardcoded `latent_source`/`denoise`/`edit_mode` fields (§7) — cannot currently be used as evidence for or against this investigation's finding; must be fixed to read `built_wf.graph` before it can serve its designed purpose |
| A `staged_edit`-run trace for a video where the built graph's actual `denoise` value is captured | **Does not yet exist** | The single most valuable missing artifact — §12 Phase 2 |
| A rule that cross-checks "`effective_edit_mode == staged_edit`" against "the actually-built graph's `denoise` value" | **Does not yet exist** | `RULE-EDIT-02` checks config reachability only (§0); this investigation's finding is currently undetectable by any existing PORCE rule — §12 Phase 3 |

---

## 12. Incremental Implementation Plan

Per the brief's explicit format (`Implementation → Tests → tai → Commit` per phase), each phase independently testable and independently valuable even if later phases are deferred:

**Phase 0 — Produce fresh runtime evidence.** Run the pipeline once for one `video_id` with the now-fixed `PROFILE_STANDARD_EDIT` path active, and persist the resulting `GenerationTraceRecord`/PORCE trace. *Tests:* none new — this is data collection, using existing, already-tested code paths. *tai:* `tai doctor` (existing). *Commit:* the resulting trace artifact only, as a checked-in fixture for Phase 1–3's tests to reference, mirroring how `data/generated_thumbnails/vIWkN-2J0ic/` is already a checked-in real-run fixture in this repository.

**Phase 1 — Fix the template defect (§5, §10 Q3).** Change every `workflows/*_edit.json` template's node `"5"` `"denoise": 1.0` literal to the placeholder `"{{denoise_strength}}"`, matching how every other per-run-variable value in these templates (`seed`, `steps`, `cfg`, etc.) is already expressed as a slot placeholder rather than a literal — this is a data-file change following an existing, established convention, not new logic. *Tests:* a new test asserting the *built* graph's `final_graph["5"]["inputs"]["denoise"]` equals `slots["denoise_strength"]` (0.75, per current `_slots()`) for a `staged_edit` build, and equals `1.0` for a `legacy_txt2img` build (regression guard) — this is the single test this whole investigation shows was missing. *tai:* full suite + `tai doctor`. *Commit:* template files + the one new test file.

**Phase 2 — Fix the observability blind spot (§7, §11).** `GenerationTraceFactory.create()` reads `built_wf.graph["5"]["inputs"]["denoise"]` (and checks for the presence of a `VAEEncodeForInpaint`/`LoadImage` node in `built_wf.graph` to set `latent_source` truthfully) instead of hardcoding literals — an additive change to an already-existing, already-isolated factory function, not a redesign of PORCE's trace model (`GenerationTraceRecord`'s schema is unchanged; only how its fields are populated changes). *Tests:* fixture-based, asserting the factory produces `denoise=0.75`/`latent_source="vae_encoded_source"` for a synthetic `built_wf` containing the relevant nodes, and the current hardcoded values for one that doesn't. *tai:* full suite + `tai doctor`. *Commit:* `observability/generation_trace.py` + tests.

**Phase 3 — Add the missing PORCE rule (§9, §11).** A new rule (e.g. `RULE-EDIT-03`, following `RULE-EDIT-02`'s exact class/interface shape in `observability/diagnostics/rules/edit_mode_resolution_rules.py`) that fires `FAIL` when `effective_edit_mode`/`GenerationTraceRecord.edit_mode == "staged_edit"` but `GenerationTraceRecord.denoise >= <configured threshold, e.g. 0.95>` — the exact condition this investigation found manually, made permanent and automatic. *Tests:* table-driven, per PORCE's own established rule-testing convention (`PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md` §20) — including a fixture reproducing this investigation's exact finding as a named regression test, following the precedent that document already set for the `BackgroundCompositor` case. *tai:* full suite + `tai doctor`. *Commit:* new rule file + tests + registry entry.

**Phase 4 — Fix `BackgroundCompositor` and wire Gap C.** Explicitly out of this document's scope to design (the brief: "do not redesign Module 7," "do not redesign the editing pipeline") — flagged here only as the necessary next phase after 1–3 make staged editing *actually* condition on the source image, since Gap C's absence (§6) remains a second, independent, sufficient cause of the reported symptom even after Phase 1 ships. A dedicated architecture document, following this document's own evidence-first methodology, is the appropriate vehicle for that phase — not an extension of this one.

---

## 13. Testing Strategy

Each phase's tests are specified inline in §12; the cross-cutting principle, consistent with every prior architecture document in this repository's `docs/` tree, is: **one new regression test per confirmed finding, asserting on the built artifact (the graph dict, the trace record), not on log output or side effects** — this is precisely the class of test that was missing (§0's `grep -rn "denoise" tests/...` returning zero matches) and precisely why this defect shipped past "921 tests passing." A repository-wide policy recommendation (not a code change, and therefore appropriate to state here): any test asserting a `WorkflowBuilder.build()` result for a `staged_edit` template should assert on at least one concrete node value proving the source image was actually going to influence the output (`denoise < 1.0`, or a `VAEEncode`-family node present) — asserting only that the correct *template file* was selected, as the current `test_module7_phase3_activation.py`/`test_module7_reachability_validation.py` do, is necessary but not sufficient, and this investigation is the concrete proof of that gap.

---

## 14. Risk Assessment

| Risk | Detail | Mitigation |
|---|---|---|
| **Fixing `denoise` alone, without Gap C, may still not produce acceptable output quality** | A single global `denoise=0.75` KSampler pass over the whole frame (no per-region masking beyond `edit_region_mask`'s single mask) is a materially weaker guarantee than `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md`'s full per-region staged design | Explicitly flagged in §12 Phase 4 as a necessary follow-on, not silently treated as "done" once Phase 1 ships — §1's stated finding is careful to say Phase 1 fixes the *first* divergence point, not the *only* one |
| **No runtime evidence yet exists for the post-Gap-A/B `staged_edit` path** (§8, §11) | This document's conclusions rest on static evidence, which is strong but has not yet been confirmed against a real, fresh pipeline run | §12 Phase 0 is sequenced first specifically to close this gap before Phase 1's fix is evaluated |
| **`0.75` as the target `denoise_strength` was read from existing code (`_slots()`), not independently re-derived or tuned** | This document reuses an existing, already-computed value rather than proposing a new one, per the "do not redesign" scope boundary — but its suitability (too high, too low) for actual quality was not evaluated here | Out of scope for this document; a quality-tuning pass belongs to Phase 4 or a dedicated follow-on, informed by PVQEF's existing quality dimensions (`PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md` §11) |
| **`GenerationTraceFactory` fix (Phase 2) could itself be wrong if `built_wf.graph`'s node numbering isn't stable across templates** | All eleven `_edit.json` templates were verified to use node `"5"` for `KSampler` consistently (§5), but a future template addition could use a different node ID | Phase 2's implementation should key off `class_type == "KSampler"` rather than a hardcoded node ID `"5"`, a detail for the implementation phase, noted here as a design constraint rather than deferred silently |

---

## 15. Migration Plan

Fully additive, matching the phasing discipline every prior document in this repository's `docs/` tree already established (§0's four source documents, all phased the same way): each of §12's four phases ships independently, is individually revertible, and — critically for Phases 1–3 specifically — **touches only data files and one isolated factory function**, not `ImageGeneratorPipeline`'s control flow, `WorkflowGraphAssembler`'s attachment mechanism, or PORCE's `RuleEngine`/`PipelineTrace` schema. No phase in §12 requires any existing test to change; Phase 1's new test is additive, Phase 2's factory change preserves `GenerationTraceRecord`'s existing schema (only populates it more truthfully), and Phase 3 is a new rule registered alongside `RULE-EDIT-02`, not a replacement for it. Phase 4 (Gap C, `BackgroundCompositor`) is explicitly out of this document's migration scope and belongs to its own architecture document, to be authored once Phases 1–3 have produced the fresh runtime evidence (§12 Phase 0/§14) needed to determine whether Gap C's fix is still necessary at the same priority once denoise=1.0 alone is corrected.
