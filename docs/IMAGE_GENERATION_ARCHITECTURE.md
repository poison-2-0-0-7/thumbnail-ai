# IMAGE_GENERATION_ARCHITECTURE.md

**Module 7 — Local Image Generation Engine**
**thumbnail-ai**

Status: **Design specification, v1.0. No implementation.**
Source of truth: repository `poison-2-0-0-7/thumbnail-ai` at the commit reviewed for this document (Modules 1–6 complete, 351 tests passing).
Upstream contract: this module consumes `PromptPackage` objects produced by Module 6 (`modules/prompt_compiler.py`) and persisted under `data/prompt_packages/`.

Revision note (v1.0): this revision preserves the v0.9 architecture's pipeline shape, module boundaries, ComfyUI-as-local-orchestrator design, and every existing section. It adds ten additive capabilities — generation profiles, workflow versioning, workflow/generation hashing, multi-candidate generation with deterministic ranking, niche-specific workflow selection, a ComfyUI workflow-template library, weighted quality scoring, a documented model-swap contract, expanded GPU resource management, and production monitoring — without changing what Module 7 is responsible for or how it talks to ComfyUI.

---

## 1. Goals

**Primary goal.** Given a `PromptPackage` (Module 6 output, itself derived deterministically from Module 5's `RedesignSpecification`), produce one or more redesigned YouTube thumbnail images that:

1. Are competitive in visual quality and composition with top-performing creators in the same niche.
2. Preserve the creator's identity — specifically the face(s) that Module 4 (`thumbnail_intelligence.py`) detected and flagged for preservation via `elements_to_preserve` and `ObjectDirective(action="preserve")`.
3. Improve estimated click-through potential relative to the source thumbnail, consistent with the CTR/curiosity-gap signals already computed in `RedesignSpecification.source_ctr_potential_score` and `source_curiosity_gap_score`.
4. Run entirely on local hardware — no network calls, no paid services, no telemetry.
5. Are deterministic and reproducible given the same `PromptPackage`, matching the project-wide "deterministic compiler, not a reasoning system" philosophy already established in Modules 5 and 6.

**Secondary goals.**

- Fit comfortably inside an 8 GB VRAM budget without manual intervention per run, while degrading gracefully — never silently — on smaller cards and taking advantage of larger ones (§6, §25).
- Fail loudly and typed (via the project's existing exception-hierarchy convention), never silently degrade quality.
- Integrate as a drop-in module in `main.py`'s pipeline with the same architectural shape as Modules 1–6: a `modules/image_generator.py` with a corresponding `ImageGenerationResult` Pydantic model in `models.py`, config constants in `config.py`, and Loguru logging with `enqueue=True`.
- Be swappable at the model level (checkpoint, LoRA, ControlNet, upscaler) without touching orchestration code — formalized in v1.0 as an explicit **generation profile** contract (§6) rather than scattered config constants.
- Select among several candidate images, several niche-tuned workflow templates, and several hardware profiles, all **without changing Module 7's call signature or `main.py`'s orchestration code** (§6, §7, §15).

**Non-goals.**

- This is not a general-purpose image editor. It only redesigns thumbnails per `PromptPackage` instructions.
- This module does not reason about creative direction. All creative decisions were already made deterministically by Module 5; Module 7 only *renders* them.
- This module does not write new prompt copy. `PromptPackage.positive_prompt` / `negative_prompt` are treated as final text, exactly like Module 6 treats `RedesignSpecification` as final.
- Multi-candidate generation and niche-specific workflows (new in v1.0) do not introduce creative reasoning either — candidate ranking is a deterministic scoring function over machine-checkable signals (§15–16), and workflow selection is a deterministic lookup keyed off data Module 4 already computed (§7), not a new inference step.

---

## 2. Design philosophy

The project's existing modules share a consistent philosophy that this design continues rather than breaks from:

- **Module 5 reasons. Module 6 compiles. Module 7 renders.** Each module is a strict narrowing of degrees of freedom. Module 7 must not reintroduce creative freedom that Modules 5–6 deliberately removed (e.g., it must not "improve" the prompt, invent subjects, or ignore `negative_prompt`/`safety_constraints`).
- **Determinism by default.** `GenerationParameters.seed` and `GenerationParameters.sampler` (currently `"deterministic"` as a placeholder value from Module 6) become real, binding execution parameters in Module 7. The same `PromptPackage` + the same model/LoRA/ControlNet file hashes must produce the same image, bit-for-bit where the underlying sampler allows it (see §23).
- **Configuration over code — profiles, not scattered flags.** v1.0 formalizes this further: hardware/quality trade-offs are expressed as a small, closed set of named **generation profiles** (§6), and creative/niche trade-offs are expressed as a small, closed set of named **workflow templates** (§7). Module 7's orchestration code selects *among* profiles and templates; it never branches on ad hoc combinations of raw parameters. This keeps the "narrowing of degrees of freedom" property intact even as the system grows more capable.
- **Typed boundaries, atomic writes, structured logging.** Continue the project convention: Pydantic models for every stage's output, atomic temp-file-then-`Path.replace()` writes for all persisted artifacts, Loguru with `enqueue=True`, and Tenacity retry with the existing `_before_sleep_log` callback pattern for transient failures (VRAM allocation races, ComfyUI queue contention).
- **Local-first, offline-first.** ComfyUI is treated as a local inference engine, not a service. Module 7 talks to a ComfyUI instance over `localhost` only; there is no code path that can reach the public internet at generation time. Model *acquisition* (downloading checkpoints) is a separate, explicit, one-time setup step — not something Module 7 does at runtime.
- **Graceful hardware degradation, not silent quality loss.** On an 8 GB card, some stages (FLUX.1-dev at full precision, for instance) do not fit. The architecture must choose quantized/optimized variants up front, in config, rather than downgrading silently mid-run. In v1.0 this is expressed as automatic **profile selection** bounded by measured VRAM (§6, §25), always logged, never silent.
- **Separation of "generation" from "restoration" from "assurance."** Three distinct, independently testable pipeline stages: (a) generate the redesigned composition, (b) restore/preserve the identity-critical regions, (c) verify the result meets the machine-checkable bar (resolution, face similarity, NSFW/safety, OCR sanity) before it is considered a finished asset. This mirrors Module 4's own internal staging (OCR stage, face stage, object stage, color stage) — Module 7 is structured the same way. v1.0 adds a fourth, equally separated stage: (d) **rank** — when more than one candidate is generated, ranking is its own deterministic, independently testable stage that consumes (c)'s scores rather than re-deriving them.
- **Every artifact is traceable to the exact code and weights that produced it.** v1.0 elevates this from "record the seed" to a full reproducibility contract: `workflow_version`, `workflow_hash`, `PromptPackage` hash, model/LoRA/ControlNet hashes, and a top-level `generation_hash` are recorded for *every* generation, not only for regression-test fixtures (§23).

---

## 3. Overall architecture

```
                       ┌────────────────────────────────────────┐
                       │           Module 6 output               │
                       │   data/prompt_packages/<video_id>.json   │
                       │            (PromptPackage)               │
                       └─────────────────────┬────────────────────┘
                                              │
                                              ▼
                       ┌────────────────────────────────────────┐
                       │         MODULE 7 — image_generator.py   │
                       │                                           │
                       │  7a. PromptPackageLoader                 │
                       │  7b. ReferenceAssetResolver               │
                       │  7c. ProfileSelector (§6)                 │
                       │  7d. WorkflowLibrary lookup (§7)          │
                       │  7e. WorkflowBuilder (ComfyUI graph)      │
                       │  7f. ComfyUIClient (HTTP + WebSocket)     │
                       │  7g. IdentityPreservationStage            │
                       │  7h. FaceRestorationStage                 │
                       │  7i. UpscaleStage                         │
                       │  7j. QualityAssuranceStage (scoring, §16) │
                       │  7k. CandidateRanker (§15, if N>1)        │
                       │  7l. ArtifactWriter (+ hashes, §23)       │
                       │  7m. MetricsCollector (§26)               │
                       └─────────────────────┬────────────────────┘
                                              │
                                              ▼
                       ┌────────────────────────────────────────┐
                       │      data/generated_thumbnails/           │
                       │        <video_id>/ (images + manifest)    │
                       │         ImageGenerationResult (JSON)       │
                       └────────────────────────────────────────┘
```

Module 7 does not embed a diffusion runtime itself. It is an **orchestrator and typed client** around a locally running **ComfyUI server** (a separate OS process, started once, reused across runs — analogous to how Module 2 treats yt-dlp as an external tool it shells out to / drives, not a library it reimplements).

Two processes:

1. **ComfyUI server** — long-lived, `python main.py --listen 127.0.0.1 --port 8188` (or equivalent), holding models in VRAM/RAM across requests to avoid reload cost.
2. **thumbnail-ai Module 7** — short-lived per pipeline run, talks to ComfyUI over `http://127.0.0.1:8188` (queueing a workflow) and a WebSocket (progress + completion events). No other network egress.

The additions in this revision (7c, 7d, 7k, 7m) are **pure-Python orchestration logic that sits around the same two-process model** — they do not add a third process, a database, or a service. `ProfileSelector` and `WorkflowLibrary` are lookups against static, version-controlled config/JSON; `CandidateRanker` is a pure function over `QualityAssuranceReport` scores already computed in-process; `MetricsCollector` appends to the same structured log/manifest outputs Module 7 already produces.

---

## 4. Module responsibilities

| Component | File (proposed) | Responsibility |
|---|---|---|
| Config | `modules/config.py` (extended) | ComfyUI host/port, model filenames, VRAM profile, output paths, timeouts, log paths — additive constants only, no changes to existing Module 1–6 constants. |
| Models | `modules/models.py` (extended) | New Pydantic models: `ComfyUIWorkflowRef`, `GeneratedAsset`, `FaceMatchResult`, `QualityAssuranceReport`, `ImageGenerationResult`, and (v1.0) `GenerationProfile`, `WorkflowTemplateRef`, `CandidateScore`, `GenerationMetrics`. Additive only. |
| `PromptPackageLoader` | `modules/image_generator.py` | Reads a persisted `PromptPackage` JSON, validates it against the Module 6 schema, rejects `status == "error"` packages. |
| `ReferenceAssetResolver` | `modules/image_generator.py` | Locates the source thumbnail (Module 3 output) and, if `elements_to_preserve` references a face, locates/derives the face crop and embedding needed for identity conditioning. |
| `ProfileSelector` *(v1.0)* | `modules/image_generator.py` | Pure function: requested quality tier (or `"auto"`) + measured VRAM/config → a concrete `GenerationProfile` (§6). Encapsulates all hardware-driven decisions so `WorkflowBuilder` never has to. |
| `WorkflowLibrary` *(v1.0)* | `modules/workflow_library.py` (NEW) | Loads and validates the versioned ComfyUI workflow templates under `workflows/` (§7); resolves `(niche, profile) → template path` via a deterministic lookup table, falling back to `general.json`. No I/O beyond reading these static, version-controlled files. |
| `WorkflowBuilder` | `modules/image_generator.py` | Pure function: `PromptPackage` + `GenerationProfile` + resolved workflow template → a concrete ComfyUI workflow graph (JSON), with `workflow_version` and `workflow_hash` attached. No network I/O. Fully unit-testable like `prompt_compiler.py` is. |
| `ComfyUIClient` | `modules/image_generator.py` | Thin typed HTTP/WebSocket client: submit workflow, poll/stream progress, retrieve output images, handle queue and OOM errors. Tenacity-wrapped. |
| `IdentityPreservationStage` | `modules/image_generator.py` | Confirms the InsightFace embedding of the generated face matches the source face within a configured threshold; triggers regeneration or restoration fallback if not. |
| `FaceRestorationStage` | `modules/image_generator.py` | Invokes GFPGAN/CodeFormer (as ComfyUI nodes, see §9) on the finished composite to correct diffusion-induced facial artifacts without altering identity. |
| `UpscaleStage` | `modules/image_generator.py` | ESRGAN-family upscale + deterministic Lanczos downscale to the exact target resolution from `GenerationParameters`/`QualityParameters`. |
| `QualityAssuranceStage` | `modules/image_generator.py` | Deterministic, non-AI checks plus one AI check (NSFW/safety classifier), each producing a named, weighted sub-score (§16) that both gates acceptance and feeds `CandidateRanker`. |
| `CandidateRanker` *(v1.0)* | `modules/image_generator.py` | Pure function: `list[QualityAssuranceReport]` (+ `FaceMatchResult`s) → an ordered ranking and a selected winner, using the weighted score in §16. Runs only when `GenerationParameters` requests more than one candidate (§15); a no-op pass-through at N=1. |
| `ArtifactWriter` | `modules/image_generator.py` | Atomic writes of final PNGs + a JSON manifest (`ImageGenerationResult`, including workflow/model/generation hashes per §23) to `data/generated_thumbnails/<video_id>/`, following the temp-file-then-`Path.replace()` pattern used elsewhere in the project. |
| `MetricsCollector` *(v1.0)* | `modules/image_generator.py` | Aggregates per-stage timings, retry counts, failure reasons, and resource-usage samples into a `GenerationMetrics` record appended to `logs/module7_metrics.jsonl` (§26) — a passive observer, never a control-flow participant. |
| Orchestration | `main.py` (extended) | Adds a Module 7 stage after Module 6 in the existing pipeline sequence, same error-isolation-per-video pattern already used for Modules 2–6. |

---

## 5. Complete image generation pipeline

End-to-end, one `video_id` at a time (the pipeline stays per-video to match the existing per-video error isolation in `main.py`):

1. **Load** `PromptPackage` for `video_id`; reject if `status != "success"`.
2. **Resolve references**: locate source thumbnail file (Module 3), locate `FaceAnalysis`/`FaceDetail` data already computed by Module 4 for that video (re-used, not recomputed), extract the reference face crop and compute (or reuse a cached) InsightFace embedding.
3. **Select generation profile** *(v1.0, §6)*: `ProfileSelector` resolves the requested/auto profile against measured VRAM and config, producing a concrete `GenerationProfile` (checkpoint, sampler, scheduler, steps, CFG, ControlNet/IPAdapter toggles, restoration/upscaler choice).
4. **Select workflow template** *(v1.0, §7)*: `WorkflowLibrary` resolves the creator's niche (already classified by Module 4) plus the chosen profile to a concrete workflow template file (e.g. `workflows/gaming.json`), falling back to `workflows/general.json` if no niche-specific template exists.
5. **Build workflow graph** from `PromptPackage` + reference assets + profile + template (pure function, no I/O); the resulting graph is hashed (`workflow_hash`, §23) before submission.
6. **Submit** to ComfyUI; stream progress; retrieve raw generated image(s) — `GenerationParameters.num_candidates` controls candidate count (1, 2, 4, or 8 — see §15), generated sequentially per §25.
7. **Identity check** *(per candidate)*: compare generated face embedding(s) to the reference embedding. If below threshold, either (a) resubmit with an incremented internal retry seed (bounded retries) using stronger identity conditioning, or (b) fall back to the highest-similarity candidate and flag it in the manifest as `identity_confidence: low`.
8. **Face restoration** *(per surviving candidate)*: run each candidate through GFPGAN/CodeFormer to correct local artifacts (eyes, teeth, skin texture) without shifting identity.
9. **Background/composition pass** (only if `RedesignSpecification`/`PromptPackage` calls for background replacement — see §13): inpaint background region only, subject region masked out, so the restored subject is never touched again after step 8.
10. **Upscale** each surviving candidate to at least `QualityParameters.min_resolution_px`, then deterministically resize/crop to exactly `GenerationParameters.width` × `GenerationParameters.height` (default 1280×720, 16:9).
11. **Quality assurance & scoring** *(v1.0, §16)*: run deterministic structural checks + one NSFW/safety classifier pass per candidate, producing a weighted `QualityAssuranceReport.overall_score` for each. Reject and (bounded) retry a candidate on hard-gate failure; otherwise it becomes eligible for ranking.
12. **Rank candidates** *(v1.0, §15 — no-op if `num_candidates == 1`)*: `CandidateRanker` orders eligible candidates by weighted score and selects the winner; all candidate scores are retained in the manifest for auditability even though only the winner (and, if `MODULE7_SAVE_CANDIDATES`, the runners-up) is persisted as an image.
13. **Persist**: atomic write of the winning PNG (+ optional runner-up PNGs) plus a JSON `ImageGenerationResult` manifest recording every parameter used — seed, profile name, workflow name/version/hash, model/LoRA/ControlNet hashes, `PromptPackage` hash, `generation_hash`, per-candidate QA scores, and the ranking outcome — for reproducibility and audit (§23).
14. **Log & record metrics**: structured success/failure per stage via Loguru, matching the granularity of Module 2/4's per-stage logging, plus a `GenerationMetrics` record appended for dashboards (§26).

---

## 6. Generation profiles

v1.0 replaces "configure the checkpoint/sampler/etc. directly" with a small, closed set of named **generation profiles**. A profile is the single unit of hardware/quality trade-off; Module 7's orchestration code (and `main.py`) only ever selects a profile by name — it never assembles ad hoc parameter combinations.

### 6.1 Profile contract

Every profile is a `GenerationProfile` Pydantic model (additive, in `models.py`) with exactly these fields, all required (no silently-optional hardware behavior):

| Field | Meaning |
|---|---|
| `name` | Stable identifier, e.g. `PROFILE_STANDARD`. Recorded in every manifest. |
| `checkpoint` | Checkpoint filename + declared family (`sdxl`, `flux`), resolved against `comfyui/models/checkpoints/`. |
| `sampler` / `scheduler` | Concrete (sampler, scheduler) pair — never ComfyUI defaults (§23). |
| `steps` | Integer step count for the main `KSampler`. |
| `cfg` | CFG scale. |
| `controlnet_enabled` | Whether the ControlNet composition-guidance branch (§9) is included. |
| `ipadapter_enabled` | Whether InstantID/IP-Adapter-FaceID identity conditioning (§9) is included. Profiles intended for non-face content may disable this; §10's identity gate is skipped cleanly per its existing "no face detected" rule when this is `False`. |
| `restoration` | Which restorer(s) run (`codeformer`, `gfpgan`, `both`, or `none`) and their fidelity weight. |
| `upscaler` | Which upscale path runs (`real_esrgan_x4`, `lanczos_only`). |
| `expected_vram_gb` | Documented, measured peak VRAM for this profile at the target checkpoint/resolution — not a theoretical figure; re-measured whenever a component model changes. |
| `expected_generation_seconds` | Documented, measured single-candidate wall-clock time on the reference RTX 4060 Laptop GPU (8 GB) baseline (§25), for capacity planning and the throughput metrics in §26. |

### 6.2 Default profile set

| Profile | Checkpoint family | Steps / CFG | ControlNet / IPAdapter | Restoration | Upscaler | Expected VRAM | Expected time |
|---|---|---|---|---|---|---|---|
| `PROFILE_STANDARD` | SDXL / Juggernaut XL, fp16 | 30 / 6.5 | On / On | CodeFormer (w=0.35) | Real-ESRGAN x4 | ~7.5 GB | ~15–35 s |
| `PROFILE_FAST` | SDXL / Juggernaut XL, fp16 | 16 / 6.0 | On / On | CodeFormer (w=0.35) | Lanczos only | ~7 GB | ~6–12 s |
| `PROFILE_PREMIUM` | FLUX.1-schnell, GGUF Q5_K_M, T5-XXL on CPU/RAM | 20 / 1.0 (FLUX-native guidance) | Off (FLUX ControlNet ecosystem still maturing; documented gap, §24) / On | CodeFormer (w=0.35) + GFPGAN blend | Real-ESRGAN x4 | ~7.8 GB | ~40–70 s |
| `PROFILE_LOW_VRAM` | SDXL / Juggernaut XL, fp16, `--lowvram` | 20 / 6.0 | Off / On | CodeFormer (w=0.4) | Lanczos only | ~4.5–5.5 GB | ~20–45 s (slower per-step under `--lowvram`) |

These four are the v1.0 shipped defaults; §6.3 documents how an operator or a future model change adds a fifth without touching Module 7 orchestration code. `expected_vram_gb`/`expected_generation_seconds` are re-measured, not re-guessed, whenever `checkpoint`, `restoration`, or `upscaler` changes for a profile — stale numbers here are a documentation bug, tracked the same way a stale docstring would be.

### 6.3 Profile selection

- `MODULE7_PROFILE` config default is `"auto"`. Under `"auto"`, `ProfileSelector` measures available VRAM at ComfyUI startup (via ComfyUI's own `/system_stats` endpoint, not a guess) and picks the richest profile whose `expected_vram_gb` fits within a configurable headroom margin (`MODULE7_VRAM_HEADROOM_GB`, default 0.5 GB) — falling back down the list `PROFILE_PREMIUM → PROFILE_STANDARD → PROFILE_FAST → PROFILE_LOW_VRAM` never silently, always with a logged reason (§19).
- An explicit `GenerationParameters`-level override (e.g., a creator/niche combination Module 5 flagged as quality-critical) may request a specific profile by name; `ProfileSelector` honors it and only overrides down to `PROFILE_LOW_VRAM` if the requested profile genuinely does not fit, logging a `ProfileDowngradedWarning`.
- **Adding a profile is a config-and-data change, not a code change.** A new `GenerationProfile` entry (e.g., a future `PROFILE_12GB`) is added to `config.py`'s profile registry dict; `ProfileSelector`, `WorkflowBuilder`, and every downstream stage consume it through the same typed `GenerationProfile` interface, so nothing in `image_generator.py` branches on profile name beyond the selection step itself.

---

## 7. Niche-specific workflows and the ComfyUI workflow library

Module 4 (`thumbnail_intelligence.py`) already classifies the creator's content niche as part of its existing analysis. v1.0 makes that classification actionable for Module 7 without adding a new reasoning step: niche selection is a **deterministic lookup**, exactly like profile selection.

### 7.1 Workflow library layout

```
thumbnail-ai/
└── workflows/
    ├── general.json          # NEW — fallback template, used when no niche match
    ├── gaming.json            # NEW
    ├── finance.json           # NEW
    ├── education.json         # NEW
    ├── podcast.json            # NEW
    ├── tech.json               # NEW
    ├── lifestyle.json          # NEW
    ├── vlog.json               # NEW
    ├── fitness.json            # NEW
    ├── reaction.json           # NEW
    └── documentary.json        # NEW
```

Each file is a **ComfyUI-native workflow-graph template**: the same node-graph JSON format ComfyUI already exports/imports, with Module 7's variable slots (prompt text, seed, checkpoint name, ControlNet strength, etc.) marked as named placeholders rather than hardcoded values. These are version-controlled, human-reviewable, and diffable in normal code review — not opaque binary blobs.

Per-niche templates differ only in the parameters that empirically matter per niche (documented per template in a `_meta` block inside each JSON file, e.g. `gaming.json` biases ControlNet strength higher to preserve HUD/composition elements common in gaming thumbnails, while `podcast.json` biases toward simpler backgrounds and a lower ControlNet weight since podcast thumbnails are typically face-forward with minimal scene complexity). This is graph-level and parameter-level variation only — the *node types* (checkpoint loader, CLIP encode, KSampler, etc.) stay identical across templates so `WorkflowBuilder`'s slot-filling logic does not need per-niche branches.

### 7.2 Selection logic — how it stays outside Module 7's core logic

- `WorkflowLibrary.resolve(niche: str, profile: GenerationProfile) -> WorkflowTemplateRef` is the entire selection surface. It is a pure dict lookup (niche → filename), with `general.json` as the unconditional fallback for any niche not present in the table (including future niches Module 4 might add) — so `WorkflowLibrary` never raises `KeyError` for an unrecognized niche, it degrades to `general.json` and logs the fallback.
- **Module 7's core logic (`WorkflowBuilder`, `ComfyUIClient`, identity/restoration/upscale/QA stages) is entirely unaware of niches.** They only ever see the already-resolved `WorkflowTemplateRef` + `GenerationProfile` + `PromptPackage`. This is what "workflow selection happens without changing Module 7 logic" means concretely: niche awareness lives in one small lookup table, not threaded through the pipeline.
- The niche value itself comes from Module 4's existing classification field (already present in `RedesignSpecification`'s upstream context) — Module 7 does not re-classify content or call any additional model to determine niche.
- Because template selection and profile selection are independent axes, the resolved workflow is really `(niche, profile)`, e.g. `gaming.json` rendered under `PROFILE_FAST` — `WorkflowBuilder` merges the niche template's graph shape with the profile's checkpoint/sampler/steps/CFG values at build time.

### 7.3 Adding or editing a niche template

Adding a niche (or retuning an existing one) is: add/edit one JSON file under `workflows/`, add one line to the lookup table, add one golden-manifest regression fixture (§22). No changes to `image_generator.py`. This mirrors how Module 6 already treats its own template strings as swappable-without-code-change data.

---

## 8. ComfyUI workflow graph (structure)

The workflow graph shape below is the *general-purpose* structure that `general.json` and every niche template share; niche templates vary node parameters (weights, strengths, prompt-slot composition) within this shape, not the shape itself.

```
[CheckpointLoaderSimple]  (checkpoint per GenerationProfile, §6)
        │
        ▼
[CLIPTextEncode ×2]   (positive / negative, from PromptPackage, §10)
        │
        ▼
[rembg / mask node]   (subject mask from source thumbnail)
        │
        ├──────────────► [InstantID / IPAdapterFaceID]  (identity conditioning, if profile.ipadapter_enabled)
        │
        ▼
[ControlNet preprocessor + apply]   (if profile.controlnet_enabled)
        │
        ▼
[KSampler] (main)   (seed / steps / cfg / sampler+scheduler from GenerationProfile)
        │
        ▼
[VAEDecode]
        │
        ├── (background-change requested?) ──► [VAEEncodeForInpaint] → [KSampler] (inpaint) ──┐
        │                                                                                        │
        ▼                                                                                        ▼
[ImageCompositeMasked]  (recombine restored subject + new background) ◄────────────────────────┘
        │
        ▼
[CodeFormer / GFPGAN node]   (face restoration, per GenerationProfile.restoration)
        │
        ▼
[Real-ESRGAN Upscale x4]   (skipped if GenerationProfile.upscaler == "lanczos_only")
        │
        ▼
[ImageScale → exact W×H]   (deterministic Lanczos, from GenerationParameters)
        │
        ▼
[SaveImage → staging dir]
```

**Off-graph, post-ComfyUI (Python side, not ComfyUI nodes):** the `IdentityPreservationStage` face-embedding comparison and `QualityAssuranceStage` checks run in Module 7's own process against the file(s) ComfyUI wrote, not as ComfyUI nodes — keeping the "AI reasoning about acceptance" boundary in typed, testable Python rather than opaque graph nodes, consistent with how Module 4 keeps its Gemini-reasoning boundary explicit (`GeminiReasoning` model) rather than implicit. `CandidateRanker` (§15) likewise runs off-graph, over the collected `QualityAssuranceReport`s.

**Multi-candidate note (§15):** when `num_candidates > 1`, this same graph is submitted once per candidate with an incremented seed; ComfyUI is not asked to batch multiple images through one `KSampler` call (§25 explains why batch size stays 1).

---

## 9. Node-by-node explanation

| Node | Purpose | Driven by |
|---|---|---|
| `CheckpointLoaderSimple` | Loads the base checkpoint once per ComfyUI process lifetime — not per request, to avoid reload latency. | `GenerationProfile.checkpoint` (§6) |
| `CLIPTextEncode` ×2 | Encodes `positive_prompt` and `negative_prompt` verbatim from the `PromptPackage`. Module 7 does not concatenate, reorder, or edit this text beyond the deterministic template joins Module 6 already performed. | `PromptPackage.positive_prompt` / `.negative_prompt` |
| `rembg`/mask node | Produces a binary subject mask from the source thumbnail so identity-preserving conditioning and later compositing only ever touch the correct region. | `SubjectTreatment.target_bbox`, `has_subject` |
| `InstantID`/`IPAdapterFaceID` | Injects the InsightFace embedding of the *source* face as a conditioning signal so the generated face is anchored to the creator's real identity rather than invented. Included only when `GenerationProfile.ipadapter_enabled`. | `elements_to_preserve`, Module 4's cached `FaceAnalysis`, `GenerationProfile` |
| ControlNet preprocessor + apply | Constrains overall composition/edges/depth to match `LayoutDirection.focal_zone` and `SubjectTreatment.target_bbox`. Included only when `GenerationProfile.controlnet_enabled`; niche templates (§7) tune its strength. | `LayoutDirection`, `SubjectTreatment`, `GenerationProfile`, niche template |
| `KSampler` (main) | The actual diffusion denoising loop. Seed comes from `GenerationParameters` (incremented per candidate, §15); steps/CFG/sampler/scheduler come from `GenerationProfile`, never randomized or left to ComfyUI defaults. | `GenerationParameters`, `GenerationProfile` |
| `VAEDecode` | Converts latents to pixel space. | — |
| Inpaint branch (`VAEEncodeForInpaint` + second `KSampler`) | Only included when `RedesignSpecification`/`PromptPackage` calls for background change while preserving the subject — regenerates *only* the masked-out background region, subject latents are never re-noised. | `ObjectDirective` (background-related), mask from rembg |
| `ImageCompositeMasked` | Recombines the untouched/identity-locked subject with the newly generated background using the same mask, guaranteeing pixel-level subject preservation outside the diffusion process itself. | mask from rembg |
| CodeFormer/GFPGAN node | Local artifact correction, run with a fidelity bias toward the original identity, not toward generic "prettification." | `GenerationProfile.restoration` |
| Real-ESRGAN upscale | 4x super-resolution pass to recover fine detail lost in the base generation before the final deterministic resize. Skipped entirely for profiles/requests with `upscaler == "lanczos_only"` or `QualityParameters.upscale_requested == False`. | `GenerationProfile.upscaler`, `QualityParameters.upscale_requested` |
| `ImageScale` | Deterministic Lanczos resize/crop to the exact `width`×`height` (default 1280×720) so the output is pixel-exact to YouTube's thumbnail spec, not merely "close." | `GenerationParameters.width/height` |
| `SaveImage` | Writes to a ComfyUI-local staging directory; Module 7's `ArtifactWriter` then atomically moves/renames into the project's own `data/generated_thumbnails/` tree. | — |

Every resolved graph — after niche-template selection, profile parameter injection, and `PromptPackage` slot-filling — is hashed as `workflow_hash` (§23) immediately before submission, so the manifest records the *exact* graph that ran, not just which template/profile were nominally selected.

---

## 10. Prompt injection strategy

"Prompt injection" here means *how PromptPackage fields are mapped into ComfyUI node inputs* — not the security meaning of the term (that's covered separately in §27).

- **No new text generation, ever.** `WorkflowBuilder` treats every string field on `PromptPackage` (`positive_prompt`, `negative_prompt`, `subject_instructions`, `background_instructions`, `typography_instructions`, `composition_instructions`, `lighting_instructions`, `color_instructions`, `object_placement`, `rendering_constraints`, `safety_constraints`) as **already-final text**. Module 7's only job is concatenation/routing into the correct node slot, using the same "fixed template, not reasoning" posture Module 6 uses when compiling `RedesignSpecification` → `PromptPackage`.
- **Deterministic field routing**, e.g.:
  - `positive_prompt` + `subject_instructions` + `lighting_instructions` + `color_instructions` → main `CLIPTextEncode` (positive).
  - `negative_prompt` + `rendering_constraints` + `safety_constraints` → `CLIPTextEncode` (negative), weighted by `ModelSettings.negative_prompt_weight`.
  - `background_instructions` → the inpaint branch's positive prompt only (never injected into the subject-region conditioning).
  - `typography_instructions` → **not** sent to the diffusion model at all by default (SDXL/FLUX render legible in-image text unreliably); instead it is retained in the manifest for a possible future typography-overlay module (see §29) and only used by `WorkflowBuilder` to compute `TextOverlaySpec.avoid_zones` as an inpaint no-go mask, preventing the diffusion model from painting content into safe-zones text will later occupy.
- **No prompt is executed as a template with user-controlled substitution at generation time.** All substitution already happened, immutably, in Module 6 (`PromptPackage.model_config = ConfigDict(frozen=True)`). Module 7 reads a frozen object; it cannot mutate or reinterpret it.
- **Truncation/token-limit handling is explicit, not silent.** SDXL's CLIP encoder has a real token ceiling (~77 tokens per chunk before ComfyUI's built-in chunking kicks in). `WorkflowBuilder` logs (not silently truncates) when a compiled prompt exceeds the safe single-chunk length, and relies on ComfyUI's standard long-prompt chunking node rather than hand-truncating text, so nothing from `safety_constraints` is ever silently dropped for being "too late" in the string.

---

## 11. Identity preservation strategy

This is the single highest-priority quality bar in the spec, because it's explicitly named in the project goal ("preserving the creator's identity").

**Layered defense, not a single mechanism:**

1. **Conditioning-time**: InstantID/IP-Adapter-FaceID injects the *source* face's InsightFace embedding directly into the diffusion process, biasing generation toward the real face from the start rather than hoping restoration fixes it after the fact.
2. **Compositing-time**: when only the background changes (the common case per `SubjectTreatment`/`ObjectDirective(action="preserve")`), the subject region is masked out of the inpaint pass entirely — the original pixels (or the InstantID-conditioned regenerated subject, if a full redesign was requested) are composited back in, so identity drift from the diffusion process is architecturally bounded, not merely "usually low."
3. **Verification-time**: `IdentityPreservationStage` recomputes an InsightFace embedding on the *generated* face and compares it via cosine similarity to the *source* embedding (already available from Module 4's `FaceAnalysis`, no recomputation of the source needed). A configurable threshold (recommended default: cosine similarity ≥ 0.45–0.55, InsightFace's typical same-identity range) gates acceptance, and this same similarity score is one of the weighted signals `CandidateRanker` uses (§15).
4. **Bounded retry, not infinite regeneration.** On a failed identity check, the pipeline retries up to `config.MAX_IDENTITY_RETRIES` (default 2) with an incremented seed and, if still failing, escalates identity-conditioning strength (e.g., raises InstantID's `ip_weight`) before finally falling back to the best-of-N candidate with an explicit `identity_confidence: low` flag written into the manifest — never a silent "good enough" acceptance.
5. **Multi-face handling.** If `FaceAnalysis.face_count > 1`, only the face(s) referenced in `elements_to_preserve`/marked `is_largest` (per Module 4's existing convention) are subject to the identity gate; secondary/background faces are treated as ordinary `ObjectDirective` subjects.
6. **No identity preservation is attempted for videos where `SubjectTreatment.has_subject` is `False` or no face was detected** — the stage is skipped cleanly rather than run against nothing, avoiding false-negative QA failures. Profiles with `ipadapter_enabled == False` skip step 1 the same way.

---

## 12. Face restoration pipeline

Runs **after** the identity check passes (restoration must not be allowed to fix an already-wrong-identity face into a *confident-looking* wrong-identity face).

1. **Input**: the accepted composite (post `ImageCompositeMasked`).
2. **CodeFormer**, `fidelity` (`w`) parameter set low per `GenerationProfile.restoration` (recommended default `0.3–0.4`) — biases the network toward preserving the input's actual identity/likeness over generic beautification, directly serving the identity-preservation goal rather than fighting it.
3. **GFPGAN v1.4 as a secondary pass/blend option**, enabled per-profile (`PROFILE_PREMIUM` enables it by default; others leave it off) for cases where CodeFormer alone leaves visible diffusion artifacts that a second, differently-trained restorer catches; when enabled, the two outputs are blended at a configurable ratio rather than one replacing the other outright.
4. **Region-limited application.** Restoration runs only inside the face bounding box (with a small padding margin), not full-frame, so background/typography-safe-zone content the diffusion model already got right isn't perturbed by a second network pass.
5. **Re-verification.** After restoration, `IdentityPreservationStage`'s embedding comparison runs *again* on the restored face (restoration networks can, rarely, shift identity slightly) — this is a cheap re-check, not a full regeneration, and only fails the pipeline if similarity drops below threshold post-restoration.

---

## 13. Background generation strategy

Driven by `ObjectDirective` entries and `LayoutDirection`/`ColorDirection` from the (pass-through) `RedesignSpecification` context embedded in the `PromptPackage`.

- **Preserve-subject, replace-background** is the default and most common mode: `rembg` (or the InsightFace/YOLO-derived bbox, whichever gives a tighter mask — `DetectedObject` bboxes from Module 4 are reused here rather than recomputed) produces the subject mask; only the inverse region is denoised via the inpaint branch.
- **Full redesign** (both subject and background regenerated) is used only when `SubjectTreatment.has_subject is False` or when Module 5 explicitly directed full recomposition — in that case the entire canvas goes through the main `KSampler`, with ControlNet still constraining overall composition to `LayoutDirection` targets (when `GenerationProfile.controlnet_enabled`) so the result doesn't ignore the deterministic layout guidance.
- **Object directives are enforced structurally where possible**: objects marked `"remove"` are excluded from the inpaint-region prompt and explicitly added to the negative prompt for that pass; objects marked `"include"` are named in the positive background prompt; objects marked `"preserve"` fall under the same masking logic as the subject.
- **Color direction is enforced via prompt + a deterministic post-process nudge**: `ColorDirection.target_brightness/contrast/saturation` are both (a) worded into `color_instructions` (already done by Module 6) and (b) applied as a small, deterministic, non-AI histogram/curve adjustment in `QualityAssuranceStage` if the generated image's measured brightness/contrast/saturation falls outside a tolerance band around the target — giving a guaranteed-compliant result rather than hoping the diffusion model honored the prompt. Color-compliance is also one of the weighted QA sub-scores (§16).
- **Negative-space and clutter targets** (`LayoutDirection.target_negative_space_ratio`, `target_clutter_score`) are enforced primarily through ControlNet composition guidance and the object-count directives above; `QualityAssuranceStage` computes the actual negative-space ratio on the output (reusing Module 4's composition-analysis approach) and logs — but does not hard-fail on — deviation, since this is a softer target than identity or resolution.

---

## 14. Upscaling pipeline

1. Base generation resolution is intentionally **not** the final 1280×720 — SDXL/FLUX are natively trained around ~1024px on the long edge; generating directly at low-height 720p thumbnails produces worse anatomy/detail than generating near-native resolution and downscaling.
2. **Generate** at the checkpoint's native-friendly resolution closest to the target aspect ratio (e.g., 1344×768 for 16:9 SDXL) rather than exactly 1280×720.
3. **Real-ESRGAN x4** upscales the generated image for detail recovery (particularly important for text-safe-zone edges and face restoration fidelity), when `GenerationProfile.upscaler == "real_esrgan_x4"`.
4. **Deterministic downscale** via Lanczos resampling to *exactly* `GenerationParameters.width` × `height` (1280×720 default) — this final step is not AI-based, is fully deterministic, and guarantees pixel-exact output dimensions regardless of any upstream rounding.
5. **`QualityParameters.upscale_requested`** is honored as a gate on top of the profile setting: if `False`, the Real-ESRGAN step is skipped even for a profile that would otherwise run it (faster, used for draft/preview generations — see §17 configuration options).
6. **`QualityParameters.min_resolution_px`** is validated post-pipeline in `QualityAssuranceStage`, not just assumed from configuration — actual output dimensions are measured and checked.

---

## 15. Multi-candidate generation and ranking

v1.0 formalizes candidate generation — previously a forward-looking note in the extensibility section — as a first-class, config-driven capability.

### 15.1 Candidate counts

`GenerationParameters.num_candidates` accepts `1`, `2`, `4`, or `8` (validated; other values are rejected at `PromptPackageLoader` with a clear error rather than silently clamped). Candidates are generated **sequentially**, each with the base seed incremented by a fixed stride (`seed`, `seed+1`, `seed+2`, ...), through the identical resolved workflow graph (§8) — no change to `WorkflowBuilder` is needed to support N>1; it is purely a loop in the orchestration layer around steps 6–11 of §5.

### 15.2 Ranking pipeline

`CandidateRanker` is a pure function: `list[(GeneratedAsset, QualityAssuranceReport, FaceMatchResult)] → RankedCandidates`. It never re-runs a model or recomputes a score; it only consumes scores already produced by `IdentityPreservationStage` and `QualityAssuranceStage` for each candidate.

1. Any candidate failing a **hard gate** (NSFW/safety, corrupt frame, resolution mismatch) is excluded from ranking entirely, regardless of its other scores — a hard gate failure is disqualifying, not merely penalized.
2. Remaining candidates are ordered by `QualityAssuranceReport.overall_score` (§16), a single weighted scalar, descending.
3. Ties (identical `overall_score` to the configured precision) are broken deterministically by `FaceMatchResult.similarity` descending, then by lowest candidate index (i.e., earliest/lowest seed wins) — never by an arbitrary or time-dependent order, preserving reproducibility (§23).
4. The winner becomes the persisted `<video_id>.png`; if `MODULE7_SAVE_CANDIDATES` is enabled, runners-up are persisted under `<video_id>_candidates/` with their rank and score in the filename for easy manual review.
5. Every candidate's scores (not just the winner's) are retained in `ImageGenerationResult.candidate_scores` — this is cheap (a handful of floats each) and makes it possible to audit *why* a candidate won without regenerating anything.

### 15.3 Extensibility of ranking signals

The ranking module is intentionally structured so new signals are additive:

- `IdentityScore` (from §11) — always available when a face is preserved.
- `CompositionScore` — reuses Module 4's composition-analysis utilities (negative-space ratio, focal-zone adherence) against `LayoutDirection` targets.
- `OCRQualityScore` — reuses Module 4's OCR utilities to confirm the reserved text-safe-zone is genuinely clean (no stray generated marks) for a future typography pass (§29).
- `ObjectPreservationScore` — YOLO11n-based compliance check from §16's Tier 1 object-directive logic, expressed as a 0–1 score rather than a boolean.
- `TextSafeZoneScore` — bbox-overlap check between generated content and `TextOverlaySpec.avoid_zones`.
- `ColorComplianceScore` — the brightness/contrast/saturation tolerance check from §13, expressed as a 0–1 score.

Each signal is implemented as a small, independently unit-testable function with the shape `(image, PromptPackage, context) -> float in [0, 1]`. Adding a new signal means adding one such function plus one entry in the weight table (§16) — `CandidateRanker` itself does not need to change, since it only ever sees the resulting `overall_score`.

---

## 16. Quality assurance and scoring pipeline

Two tiers, matching the project's existing "deterministic first, AI second" bias, now feeding a single weighted score rather than only pass/fail booleans.

**Tier 1 — deterministic, non-AI checks (always run, cheap, each also produces a score):**
- Exact resolution/aspect-ratio match to `GenerationParameters` — hard gate (pass/fail), not scored.
- File integrity (valid, non-corrupt PNG; not a solid/blank frame) — hard gate.
- Text-safe-zone collision (`TextSafeZoneScore`, §15.3) — reuses Module 4's OCR/bbox utilities.
- Object-directive compliance (`ObjectPreservationScore`, §15.3) — reuses the vendored YOLO11n model.
- Color-direction tolerance (`ColorComplianceScore`, §15.3), with deterministic correction if within a fixable band.
- Composition/negative-space adherence (`CompositionScore`, §15.3).

**Tier 2 — AI-assisted checks (run once per accepted candidate):**
- NSFW/safety classifier pass (Falconsai/CLIP-NSFW) — hard gate, no override, never scored/weighted (a safety failure disqualifies regardless of every other score, §15.2).
- Identity similarity (`IdentityScore`) — already computed in §11, surfaced here as a QA field.
- Face quality — a lightweight, deterministic sharpness/artifact heuristic (Laplacian-variance-based) on the restored face region, distinguishing "restoration ran and helped" from "restoration ran and the result still looks soft/artifacted."

### 16.1 Weighted overall score

`QualityAssuranceReport.overall_score` is a configurable weighted sum of the non-hard-gate signals, normalized to `[0, 1]`:

| Signal | Default weight |
|---|---|
| Identity score | 0.30 |
| Face quality | 0.15 |
| Composition score | 0.15 |
| Text-safe-zone score | 0.15 |
| Object-preservation score | 0.15 |
| Color-compliance score | 0.10 |

Weights are a named config dict (`MODULE7_QA_WEIGHTS`), not hardcoded arithmetic — retuning the balance (e.g., a niche where composition matters more than color) is a config change. Weights must sum to 1.0; `ProfileSelector`-adjacent validation at config-load time rejects a misconfigured weight table rather than silently normalizing it, so a typo doesn't quietly change scoring behavior.

**Failure handling:** any Tier 1 or Tier 2 hard-gate failure routes back to a bounded retry (new seed, or escalated conditioning per §11) rather than either (a) silently shipping a non-compliant asset or (b) crashing the whole per-video pipeline stage — matching the existing per-video error-isolation pattern in `main.py`. After `config.MAX_GENERATION_RETRIES` (default 3) exhausted attempts, the video's Module 7 result is written with `status: "error"` and a typed exception reason, exactly like a Module 2/4 failure is recorded today, and the pipeline proceeds to the next video. For multi-candidate requests, a candidate that exhausts its own retries is simply excluded from ranking (§15.2) rather than failing the whole video, as long as at least one candidate ultimately passes.

---

## 17. Output assets

Per successful `video_id`, Module 7 writes:

- `<video_id>.png` — the final, QA-accepted, top-ranked thumbnail (1280×720 by default).
- `<video_id>_candidates/` (optional, config-gated, off by default to save disk) — persisted rejected/lower-ranked candidates, useful for debugging identity-preservation and ranking-weight tuning.
- `<video_id>_manifest.json` — the `ImageGenerationResult` Pydantic model, serialized: every parameter used (seed, profile name, workflow name/version/`workflow_hash`, checkpoint/LoRA/ControlNet/IPAdapter weight hashes and versions, `PromptPackage` hash, `generation_hash`, QA scores per candidate, identity-similarity score, ranking outcome, retry count, durations per stage, timestamps) — sufficient to reproduce the exact run, per §23.

---

## 18. Folder structure

```
thumbnail-ai/
├── modules/
│   ├── config.py                 # extended, additive only
│   ├── models.py                 # extended, additive only
│   ├── workflow_library.py       # NEW — WorkflowLibrary (§7)
│   └── image_generator.py        # NEW — Module 7
├── workflows/                    # NEW — versioned ComfyUI workflow templates (§7)
│   ├── general.json
│   ├── gaming.json
│   ├── finance.json
│   ├── education.json
│   ├── podcast.json
│   ├── tech.json
│   ├── lifestyle.json
│   ├── vlog.json
│   ├── fitness.json
│   ├── reaction.json
│   └── documentary.json
├── comfyui/                      # NEW — local ComfyUI install (not committed; gitignored)
│   ├── models/
│   │   ├── checkpoints/          # Juggernaut XL, FLUX.1-schnell-GGUF
│   │   ├── controlnet/
│   │   ├── ipadapter/            # InstantID / IP-Adapter-FaceID weights
│   │   ├── facerestore/          # CodeFormer, GFPGAN
│   │   ├── upscale_models/       # Real-ESRGAN
│   │   └── insightface/          # shared antelopev2, same weights Module 4 uses
├── data/
│   ├── generated_thumbnails/     # NEW
│   │   └── <video_id>/
│   │       ├── <video_id>.png
│   │       ├── <video_id>_candidates/     # optional
│   │       └── <video_id>_manifest.json
│   └── qa_reports/               # NEW — optional detailed QA logs per video
├── logs/
│   ├── module7.log               # NEW, same Loguru convention as module1.log/module2.log
│   └── module7_metrics.jsonl     # NEW — structured metrics stream (§26)
└── tests/
    ├── test_image_generator.py   # NEW
    ├── test_workflow_library.py  # NEW
    └── test_candidate_ranker.py  # NEW
```

---

## 19. Configuration options

All additive constants in `config.py`, following the existing `DEFAULT_*`/`MODULE*_LOG_PATH` naming convention:

- `COMFYUI_HOST`, `COMFYUI_PORT` (default `127.0.0.1`, `8188`).
- `COMFYUI_STARTUP_TIMEOUT_SECONDS`, `COMFYUI_REQUEST_TIMEOUT_SECONDS`.
- `MODULE7_PROFILE` (default `"auto"`; accepts a specific profile name to override, §6.3).
- `MODULE7_GENERATION_PROFILES` — the profile registry dict keyed by name → `GenerationProfile` (§6.2); the closed set Module 7 selects from.
- `MODULE7_VRAM_HEADROOM_GB` (default `0.5`) — safety margin `ProfileSelector` reserves below measured free VRAM.
- `MODULE7_WORKFLOW_LIBRARY_DIR` (default `PROJECT_ROOT / "workflows"`).
- `MODULE7_NICHE_WORKFLOW_MAP` — niche → template filename lookup table (§7.2); unmapped niches fall back to `general.json`.
- `MODULE7_WORKFLOW_VERSION` (e.g. `"workflow_v3"`) — the current default workflow schema version recorded in every manifest (§23); bumped when the graph *shape* (not just parameters) changes.
- `MODULE7_QA_WEIGHTS` — the weighted-scoring dict from §16.1.
- `MODULE7_IDENTITY_SIMILARITY_THRESHOLD` (default `0.5`).
- `MODULE7_CODEFORMER_FIDELITY` (default `0.35`).
- `MAX_IDENTITY_RETRIES` (default `2`), `MAX_GENERATION_RETRIES` (default `3`).
- `MODULE7_SAVE_CANDIDATES` (bool, default `False`).
- `MODULE7_LOG_PATH = LOG_DIR / "module7.log"`; `MODULE7_METRICS_PATH = LOG_DIR / "module7_metrics.jsonl"` (§26).
- `MODULE7_OUTPUT_DIR = PROJECT_ROOT / "data" / "generated_thumbnails"`.
- `MODULE7_NSFW_THRESHOLD` (default conservative, e.g. `0.15` score to trigger rejection).
- `MODULE7_MAX_CONCURRENT_GENERATIONS` (default `1`) — queue/scheduling ceiling, §25.
- Draft mode: `MODULE7_DRAFT_STEPS`/`MODULE7_DRAFT_UPSCALE_SKIP` for fast low-fidelity previews during development, distinct from the production profile — never used for a final accepted asset.

---

## 20. Error handling

Continues the project's existing typed-exception-hierarchy convention (as established in Module 2's `AuthenticationError`, etc.):

- `ComfyUIConnectionError` — server unreachable/not started. Non-retryable beyond Tenacity's connection-retry window; surfaces a clear operator message ("start ComfyUI first") rather than a stack trace.
- `ComfyUIQueueError` — workflow submitted but ComfyUI reported an execution error (bad node params, missing model file). Non-retryable without a config/code fix — this indicates a `WorkflowBuilder` or setup bug, not a transient condition.
- `VRAMExhaustedError` — OOM signal from ComfyUI. Retryable with Tenacity backoff plus an automatic fallback to a lighter `GenerationProfile` (§6.3) before failing the video.
- `IdentityPreservationError` — raised after `MAX_IDENTITY_RETRIES` exhausted with no passing candidate; video marked `status: "error"`, pipeline continues to next video.
- `QualityAssuranceError` — raised after `MAX_GENERATION_RETRIES` exhausted on Tier 1/2 checks for every candidate.
- `PromptPackageInvalidError` — malformed/`status: "error"` input from Module 6; fails fast, no generation attempted.
- `WorkflowTemplateError` *(v1.0)* — a resolved niche template fails JSON-schema validation or references a node type the installed ComfyUI doesn't have; non-retryable, falls back to `general.json` once, then fails the video if `general.json` also errors (which would indicate a broader ComfyUI/install problem, not a per-niche issue).
- `NoEligibleCandidateError` *(v1.0)* — every generated candidate (of however many were requested) failed a hard gate or exhausted retries; distinct from `QualityAssuranceError` so metrics (§26) can separate "one candidate failed" from "the whole request produced nothing usable."

All of the above extend a common `Module7Error` base, mirroring how Module 2/4 exceptions extend their own module-level bases, and every failure path writes a partial `ImageGenerationResult` with `status: "error"` and a populated `error_message`, matching `RedesignSpecification`/`PromptPackage`'s existing `status`/`error_message` fields.

---

## 21. Logging

- Loguru, `enqueue=True`, sink at `MODULE7_LOG_PATH`, same rotation/retention conventions as `module1.log`/`module2.log`.
- Per-stage structured log lines: profile selection (and any downgrade reason), workflow template resolution (and any niche-fallback reason), workflow build (+ `workflow_hash`), ComfyUI submission, per-node progress (from the WebSocket progress events), identity-check result, restoration result, upscale result, QA result per candidate, ranking outcome, final write.
- Tenacity's existing `_before_sleep_log` callback pattern is reused for all retryable operations (ComfyUI connection, VRAM-OOM retry, identity-retry) so retry visibility is consistent across the whole codebase, not reinvented per module.
- No image bytes or raw prompt text are logged at INFO level by default (prompt text can be long and is already durably persisted in the `PromptPackage`/manifest JSON); DEBUG level may include truncated prompt previews for troubleshooting.

---

## 22. Testing strategy

Mirrors the existing test suite's structure (`tests/test_prompt_compiler.py`, `tests/test_thumbnail_intelligence.py`, etc.) — 351 passing tests today, Module 7 should add its own fully isolated suite without touching existing ones.

- **`WorkflowBuilder` — pure unit tests, no ComfyUI required.** Given a fixed `PromptPackage` fixture, assert the exact graph structure/node parameters produced (analogous to how `test_prompt_compiler.py` asserts exact `PromptPackage` field values from a fixed `RedesignSpecification`). This is the bulk of the coverage and requires no GPU.
- **`ProfileSelector` — unit tests over synthetic VRAM readings.** Assert the fallback ladder (`PROFILE_PREMIUM → STANDARD → FAST → LOW_VRAM`) resolves correctly at documented VRAM thresholds, and that an explicit override is honored unless it genuinely doesn't fit.
- **`WorkflowLibrary` — unit tests over the niche lookup table.** Assert every entry in `MODULE7_NICHE_WORKFLOW_MAP` resolves to a template file that actually exists and passes schema validation; assert an unmapped niche falls back to `general.json` without raising.
- **`ComfyUIClient` — mocked HTTP/WebSocket tests.** Use a fake ComfyUI server (or `responses`/`aioresponses`-style mocking) to test retry behavior, timeout handling, and progress-event parsing without a real GPU or ComfyUI install — CI-friendly.
- **`IdentityPreservationStage` — unit tests against fixed embedding pairs.** Precomputed embedding fixtures (not live InsightFace calls) verify the cosine-similarity threshold logic and retry/escalation branching deterministically.
- **`QualityAssuranceStage` — unit tests against fixed synthetic images.** Small deterministic test PNGs (wrong resolution, blank frame, text-zone collision, correct case) verify each Tier 1 check independently; the NSFW classifier is mocked in unit tests (not invoked live) and covered separately by a small, explicitly-marked "slow"/integration test.
- **`CandidateRanker` — pure unit tests over fixed score tuples.** Given synthetic `QualityAssuranceReport` lists, assert ordering, hard-gate exclusion, and the deterministic tie-break rule (§15.2) — no images or models involved.
- **Integration tests (marked `@pytest.mark.gpu`, skipped by default in CI, run manually/nightly on the dev machine).** A small fixed set of real `PromptPackage` fixtures, across at least one niche template and each profile, run through a real local ComfyUI instance, asserting the pipeline completes and produces a QA-passing image — this is the only tier that actually needs the RTX 4060.
- **Golden-manifest regression tests.** For a fixed seed/model-hash/`workflow_version` combination, assert the `ImageGenerationResult` manifest's non-image fields (parameters, hashes, QA scores, ranking outcome) match a stored golden file — catches accidental parameter, profile, or template drift without requiring pixel-identical image comparison across environments (see §23 on determinism limits). Each niche template and each profile gets at least one golden fixture.
- **`pytest.ini`** gains a `gpu` marker alongside any existing markers, deselected by default (`-m "not gpu"`), matching how a resource-heavy modern test suite typically isolates hardware-dependent tests from the fast default run.

---

## 23. Reproducibility, workflow versioning, and hashing

v1.0 elevates reproducibility from "record the seed" (v0.9) to a full, always-on provenance contract, matching the project-wide "deterministic compiler, not a reasoning system" philosophy.

### 23.1 Seed and sampler are binding, not advisory

- `GenerationParameters.seed` is passed directly to `KSampler`; Module 7 never substitutes a random seed unless a retry explicitly and loggedly increments it per §11/§16, or a multi-candidate request increments it per a fixed, documented stride (§15.1).
- **Sampler/scheduler pair is fixed per profile in config** (§6.1), not left to ComfyUI defaults — `GenerationParameters.sampler` currently arrives from Module 6 as the placeholder value `"deterministic"`; each `GenerationProfile` resolves this to one concrete, documented (sampler, scheduler) pair (e.g., `dpmpp_2m` + `karras` for `PROFILE_STANDARD`) so "deterministic" has one unambiguous real meaning per profile across the whole system.

### 23.2 Workflow versioning

- `MODULE7_WORKFLOW_VERSION` (e.g. `workflow_v3`) identifies the **graph-shape schema** — bumped whenever a node is added/removed/rewired in a way that changes what the graph *can* do, independent of which niche template or profile is in use. `workflow_v1`, `workflow_v2`, `workflow_v3`, ... are tagged, changelog-documented releases of the underlying graph shape shared by every template in `workflows/`.
- Every `ImageGenerationResult` records `workflow_version`, letting a future audit or regression test immediately tell whether an old manifest's graph shape is still current, without diffing JSON by hand.
- A workflow-shape change (new `workflow_version`) requires new golden-manifest fixtures (§22) for every affected niche/profile combination before it can ship — this is the mechanism that keeps versioning meaningful rather than aspirational.

### 23.3 Hashing — what gets hashed and why

Every `ImageGenerationResult` (successful or failed after retries) records:

| Field | What it hashes | Why |
|---|---|---|
| `workflow_hash` | The fully resolved ComfyUI graph JSON (post niche-template + profile + `PromptPackage` slot-filling), immediately before submission. | Confirms *exactly* what graph ran — catches drift even within the same `workflow_version` (e.g., a niche template edited without a version bump would be caught here first). |
| `prompt_package_hash` | The `PromptPackage` JSON as loaded (it is already `frozen=True`, so this is stable for the run). | Ties the output back to an exact, unmodified Module 6 artifact. |
| `checkpoint_hash`, `lora_hashes`, `controlnet_hashes`, `ipadapter_hash`, `restoration_model_hashes`, `upscaler_hash` | SHA-256 of each model weight file actually loaded for this run (computed once at ComfyUI startup and cached, not re-hashed per request — these are multi-GB files). | Distinguishes "the pipeline changed" from "someone swapped a checkpoint file on disk" during later audits (§27 covers provenance/licensing implications). |
| `generation_hash` | A single top-level hash over `(workflow_hash, prompt_package_hash, checkpoint_hash, lora_hashes, controlnet_hashes, seed, GenerationProfile.name)`. | One value that answers "would re-running this produce the same result," without an auditor needing to compare six separate fields by hand. |

Hashing is cheap relative to generation time (model-file hashes are cached; the graph/`PromptPackage` hashes are small JSON) and runs unconditionally — it is not a debug-only or opt-in feature, because a manifest without hashes cannot support a real reproducibility claim.

### 23.4 Known determinism limits, documented rather than hidden

GPU floating-point non-associativity means bit-exact pixel reproduction across different GPU models/driver versions/cuDNN builds is not guaranteed even with an identical seed and graph — this is a property of CUDA kernels generally, not a defect in this design. The architecture's determinism guarantee is scoped to: *same seed + same graph (same `workflow_hash`) + same model files (same hashes) + same GPU/driver/ComfyUI version → same output*, which is the practically achievable and industry-standard bar, and is what the golden-manifest regression tests (§22) validate (parameters, hashes, and QA scores, not raw pixels, across environments).

### 23.5 Non-deterministic stages are isolated and flagged

The NSFW classifier and any future best-of-N human-selection step are explicitly excluded from the "deterministic core" — they gate acceptance but do not alter pixels, keeping the boundary between "generation is deterministic" and "acceptance may involve a threshold call" clean and auditable, matching Module 4's own separation of deterministic OCR/face/object stages from its LLM-reasoning stage. `CandidateRanker`'s tie-break rule (§15.2) is deliberately deterministic for the same reason — ranking must not be a hidden source of irreproducibility.

---

## 24. Future model swapping

Formalizes and expands the model-swap contract only sketched in v0.9. The explicit goal: SDXL/Juggernaut, FLUX, and any future checkpoint family must be swappable **without changing Module 7 code**, only config/profile/template data.

### 24.1 What makes a model swappable under this architecture

A checkpoint (or LoRA, ControlNet, restorer, upscaler) is swappable in place as long as:

1. It is loadable via a ComfyUI-native node already present in the workflow-graph shape (§8) — `CheckpointLoaderSimple` or a documented equivalent (e.g., a GGUF loader node for quantized FLUX). Adding a genuinely new *node type* (not just a new weights file) is a `workflow_version` bump (§23.2), not a Module 7 code change — it's still config/template data, just data that also touches the graph shape.
2. Its expected VRAM/generation-time characteristics are measured and captured in a `GenerationProfile` entry (§6.1) — never assumed from a vendor's marketing numbers.
3. Its prompt-conditioning contract matches what `WorkflowBuilder` already routes (positive/negative text, optional ControlNet/IPAdapter conditioning) — a model needing an entirely different conditioning paradigm (e.g., a future model with no text-conditioning at all) would be an out-of-scope architecture change, not a swap, and this document says so explicitly rather than implying otherwise.

### 24.2 The swap procedure

1. Add the new checkpoint file to `comfyui/models/checkpoints/` (out-of-band, per §28's model-asset-management process) and record its hash.
2. Add or edit a `GenerationProfile` entry in `MODULE7_GENERATION_PROFILES` pointing at the new checkpoint, with freshly measured `expected_vram_gb`/`expected_generation_seconds`.
3. Run the golden-manifest regression suite (§22) for that profile across at least the `general.json` template and one niche template, to confirm the swap didn't silently break prompt routing or output shape.
4. Optionally, retune `MODULE7_NICHE_WORKFLOW_MAP`/individual `workflows/*.json` parameter values (e.g., a new model might want different CFG defaults per niche) — still a data change, not a code change.

No step in this procedure touches `image_generator.py`. This is the concrete mechanism behind the v0.9 promise ("swappable at the model level... without touching orchestration code") and directly satisfies the requirement that SDXL, Juggernaut, and FLUX can be replaced by future models on the same basis.

### 24.3 Documented gaps (honesty over false completeness)

- FLUX's ControlNet ecosystem is less mature than SDXL's as of this writing; `PROFILE_PREMIUM` ships with `controlnet_enabled: False` for that reason (§6.2), not because the architecture can't support it — when FLUX ControlNet tooling matures, this is a one-line profile edit.
- A model swap that changes the *native* aspect ratio/resolution sweet spot (§14) needs its `GenerationProfile`-adjacent generation-resolution constant reviewed; this is called out here so it isn't missed during a future swap.

---

## 25. Resource management and GPU scheduling

Expands v0.9's performance notes (previously "§22") into an explicit resource-management contract, now that profiles and multi-candidate generation both compete for the same VRAM budget.

### 25.1 VRAM estimation and automatic profile selection

- VRAM is **measured, not assumed**, at ComfyUI startup via its `/system_stats` endpoint (§6.3); `ProfileSelector` uses this measurement, not a hardcoded "assume 8 GB" constant, so the same code behaves correctly on a smaller or larger card without a manual config edit.
- Automatic selection always logs which profile was chosen and why (fits, downgraded, or explicit override) — never a silent choice (§21).
- `expected_vram_gb` per profile (§6.1/6.2) is re-measured whenever a component model changes; stale VRAM numbers here would defeat the point of measuring anything.

### 25.2 Memory cleanup

- ComfyUI's default execution model already unloads/reloads models between disjoint graph sections when needed, but Module 7's config explicitly avoids holding both the main checkpoint and a second heavy model (e.g., a second SDXL for the inpaint pass) resident simultaneously — reuse the same loaded checkpoint object for both the main and inpaint `KSampler` calls.
- **`--lowvram`/`--novram` ComfyUI launch flags** as a documented fallback within `PROFILE_LOW_VRAM` (not the default) for users hitting OOM even at fp16 — degrades speed, not correctness.
- After each candidate in a multi-candidate run, Module 7 explicitly requests ComfyUI to free any per-request tensors it can (via ComfyUI's queue/history cleanup endpoints) rather than letting VRAM usage creep upward across a batch of sequential candidates — a `VRAMExhaustedError` mid-batch is treated the same as any other OOM (§20), with the automatic profile-downgrade fallback applying to the *remaining* candidates in that batch.

### 25.3 Batch scheduling and queue management

- **Batch size = 1 at the ComfyUI level**, always. No multi-image-per-request batching; VRAM headroom on an 8 GB laptop GPU is better spent on resolution/ControlNet/IPAdapter than parallelism. Multiple candidates (§15) are generated **sequentially**, each a separate ComfyUI submission.
- **`MODULE7_MAX_CONCURRENT_GENERATIONS`** (default `1`) governs how many `video_id`s Module 7 will have in flight against the single ComfyUI server at once — kept at 1 by default because a single 8 GB card cannot usefully run two diffusion jobs concurrently; documented as raisable only alongside a genuinely multi-GPU or much higher-VRAM deployment, not as a knob to "go faster" on the reference hardware.
- **Real-ESRGAN tiling.** Run the upscaler in tiled mode (e.g., 512px tiles) rather than full-frame, which keeps its VRAM footprint small and largely independent of final output resolution.
- **CPU-friendly stages stay on CPU.** `rembg`, EasyOCR re-check, and (optionally) the NSFW classifier are all light enough to run on the i9-13900HX without contending for VRAM the diffusion stages need — explicit device placement in config, not left to library defaults. This also holds for FLUX's T5-XXL text encoder under `PROFILE_PREMIUM` (§6.2), which is deliberately kept off-GPU.

### 25.4 Expected throughput

Indicative only, not a guaranteed SLA, and now expressed per profile (§6.2) rather than as a single number:

- `PROFILE_STANDARD`: ~15–35 s/thumbnail once the checkpoint is warm in VRAM.
- `PROFILE_FAST`: ~6–12 s/thumbnail (draft/preview use).
- `PROFILE_PREMIUM`: ~40–70 s/thumbnail.
- `PROFILE_LOW_VRAM`: ~20–45 s/thumbnail (slower per-step under `--lowvram`).

First-request latency per process lifetime is higher than these figures due to model load time, which is why the ComfyUI server is kept long-lived (§3) rather than restarted per video. For an N-candidate request, expected wall-clock time is approximately N × the per-profile figure above (sequential generation, §25.3), before ranking overhead (negligible — ranking is a pure in-memory computation over already-computed scores).

---

## 26. Production monitoring and metrics

Expands v0.9's single monitoring bullet into a concrete metrics contract, designed to support a future dashboard without introducing a telemetry service (still "no external services," §2).

### 26.1 What is measured

Per generation attempt, `MetricsCollector` (§4) appends one `GenerationMetrics` record to `logs/module7_metrics.jsonl` (structured JSON Lines, easy to tail/ingest into any future local dashboard or `pandas` notebook) with:

- `video_id`, `niche`, `profile_name`, `workflow_version`, `workflow_hash`, `generation_hash`, `num_candidates_requested`.
- **Timing**: `queue_time_seconds` (time waiting for ComfyUI to start the job), `generation_time_seconds` per candidate, `total_duration_seconds` for the whole video.
- **Retries**: `identity_retry_count`, `generation_retry_count`, and, for a failed video, `failure_reason` (the specific typed exception name from §20 — not a free-text string, so failure reasons can be aggregated exactly).
- **Quality outcomes**: `identity_failures_count` (candidates that failed the identity gate before any retry succeeded), `qa_failures_count` (candidates that failed any Tier 1/2 hard gate), winning candidate's `overall_score` and per-signal sub-scores (§16.1).
- **Resource usage**: `peak_vram_mb` (sampled from ComfyUI's `/system_stats` during the run), `gpu_utilization_percent` (sampled, best-effort — not all driver/OS combinations expose this reliably, and the field is nullable rather than defaulting to a misleading `0`).

### 26.2 Derived/aggregate metrics for dashboards

Because each record is self-contained and timestamped, standard aggregate metrics are simple derived queries over the JSONL stream rather than something Module 7 needs to compute itself:

- Average throughput (thumbnails/hour) and average `generation_time_seconds`, sliceable by `profile_name` and `niche`.
- Success rate and retry rate over a trailing window.
- Identity-failure and QA-failure rate by niche — the concrete signal that would tell an operator "the `finance.json` template needs retuning."
- VRAM headroom trend over time — an early warning if a model update starts pushing a profile closer to its documented `expected_vram_gb` ceiling (§6.1), prompting a re-measurement before it causes OOM in production.

### 26.3 Design constraints

- **Metrics collection is a passive observer, never a control-flow participant.** `MetricsCollector` never influences retries, profile selection, or ranking — it only records what already happened, keeping the pipeline's decision logic independently testable without needing to mock a metrics system (§22).
- **No external services, no new process.** The metrics stream is a local, append-only file, consistent with §2's "no telemetry, no paid services" constraint; a future dashboard reads this file (or a `main.py`-adjacent aggregation script), it is not pushed anywhere.
- **Metrics never include image bytes or raw prompt text**, consistent with §21's logging policy — only structured, small, aggregatable fields.

---

## 27. Security considerations

- **No network egress at generation time.** ComfyUI is bound to `127.0.0.1` only; Module 7's `ComfyUIClient` is hardcoded to talk to the configured local host/port and nothing else — no code path constructs a request to any external URL during a generation run.
- **Model provenance.** All recommended model files are widely-mirrored, checksum-verifiable community weights (Hugging Face / Civitai / GitHub releases); the architecture calls for a one-time, explicit, operator-run download step with hash verification, never an automatic runtime fetch — consistent with "no code path can reach the public internet at generation time" above. The model-hash fields introduced in §23.3 give this verification a durable, per-run audit trail rather than only a one-time setup check.
- **Prompt/text sanitization boundary.** Because `PromptPackage` fields are frozen and already validated upstream by Module 6, Module 7 does not need to re-sanitize prompt text for injection-style attacks against the diffusion model itself (there is no equivalent of SQL injection here) — but it *does* validate that file-path-like fields (e.g., any future reference-image path) are constrained to the project's own `data/` tree, preventing path traversal if `PromptPackage` were ever loaded from an untrusted source. The same constraint applies to `workflows/*.json` template paths resolved by `WorkflowLibrary` (§7.2) — resolution is restricted to `MODULE7_WORKFLOW_LIBRARY_DIR`.
- **NSFW/safety gate is non-bypassable in code** — there is no configuration flag that disables the Tier 2 safety check; only the identity/QA *retry counts* and *thresholds* are configurable, not whether the check runs at all. This holds per-candidate in multi-candidate mode too — a candidate cannot reach `CandidateRanker` without passing the safety gate (§15.2).
- **License compliance is itself a security/compliance consideration for a "production" system**: FLUX.1-dev's non-commercial license is why this spec defaults to FLUX.1-**schnell** (Apache-2.0) for `PROFILE_PREMIUM` (§6.2); similarly, YOLO11n's AGPL-3.0 license (already vendored in the repo as `yolo11n.pt`) obligates source-availability if the pipeline's output is offered as a network service to others — worth flagging explicitly since it's an existing dependency being reused here, not a new one introduced by this design, but its license terms apply regardless of which module invokes it.
- **Generated-content provenance.** Recommend embedding a C2PA-style or simple custom metadata tag (e.g., PNG `tEXt` chunk noting "AI-assisted redesign, source video_id, generation timestamp, `generation_hash`") into output files — cheap, local, no dependency needed beyond Pillow (already a project dependency), and aligns with emerging platform expectations around AI-generated thumbnail disclosure. Including `generation_hash` here means provenance is verifiable against the manifest even if the PNG is later separated from its JSON sidecar.

---

## 28. Production deployment considerations

- **Two-process operational model** (§3): ComfyUI as a supervised long-lived local service (e.g., via a simple process manager or a documented `systemd`/Windows-service-equivalent setup for the target Windows machine), and the `thumbnail-ai` pipeline as the short-lived driver — Module 7 should health-check ComfyUI on startup (`ComfyUIConnectionError` if not reachable) rather than assume it's running.
- **Model asset management is out-of-band.** A documented, one-time setup script/checklist (not part of the automated pipeline) for downloading and hash-verifying the checkpoint/ControlNet/IPAdapter/restoration/upscale weights into `comfyui/models/...`, since these are multi-gigabyte assets that don't belong in version control or in the automated per-run path. Hashes recorded here feed directly into the manifest fields of §23.3.
- **Disk footprint planning.** Model weights alone (Juggernaut XL + FLUX-schnell-GGUF + ControlNets + InstantID + CodeFormer/GFPGAN + Real-ESRGAN) total roughly 15–25 GB; combined with per-video generated assets and optional candidate retention (larger now that up to 8 candidates can be retained, §15), deployment documentation should call out disk budget explicitly, and `MODULE7_SAVE_CANDIDATES=False` should be the documented production default.
- **Windows-specific packaging notes.** ComfyUI runs natively on Windows with an NVIDIA CUDA-enabled PyTorch build; deployment docs should pin a known-good CUDA/PyTorch/ComfyUI version triple (recorded in `requirements.txt`-equivalent for the ComfyUI environment, kept separate from the main project's `requirements.txt` since ComfyUI manages its own dependency set) rather than "latest," to keep the determinism guarantees in §23 meaningful over time.
- **Graceful startup/shutdown in `main.py`.** The pipeline's Module 7 stage should detect an already-running ComfyUI instance (preferred, avoids repeated model-load latency across pipeline runs) versus needing to launch one itself (documented as an optional convenience, with a config flag, not the default path — production deployments should manage the ComfyUI process lifecycle independently of the Python pipeline for reliability).
- **Monitoring hooks.** The structured `module7.log` output, the `module7_metrics.jsonl` stream (§26), and per-run `ImageGenerationResult` manifests are together designed to be sufficient for both basic and moderately detailed production monitoring (success rate, retry rate, identity-confidence distribution, mean duration, per-niche/per-profile throughput) without requiring a separate telemetry system — consistent with the "no external services" constraint.

---

## 29. Future extensibility

Explicitly designed as extension points, not implemented now (model swapping, previously listed here, is now its own formalized section — §24):

- **Typography overlay module (Module 8, potential).** `typography_instructions`/`TextOverlaySpec` are already captured and safe-zoned by Module 7 but not rendered as text; a future deterministic text-rendering pass (Pillow-based, matching Module 3's existing Pillow dependency) could composite real, crisp text into the reserved zones — sidestepping diffusion models' unreliable text rendering entirely, and could reuse the `OCRQualityScore` signal (§15.3) to validate the reserved zone stayed clean.
- **LoRA-based creator-specific style locking.** For creators with many videos, a per-creator LoRA (trained offline, one-time, on their thumbnail history) could further improve identity/style consistency beyond what IPAdapter/InstantID alone provide — pluggable via a new optional `ModelSettings` field and a new `lora_hashes` entry already reserved for it in the manifest (§23.3).
- **Alternate ComfyUI-compatible backends.** Because `WorkflowBuilder` only depends on ComfyUI's node-graph JSON contract, swapping checkpoints requires only a config/profile change (§24), not an architecture change.
- **Batch/queue mode** for processing an entire `creators.csv` run's worth of `PromptPackage`s against a persistent ComfyUI server without per-video process startup cost — the per-video metrics stream (§26) would become the natural input to a batch-run summary report.
- **Human-in-the-loop candidate selection.** With `MODULE7_SAVE_CANDIDATES` and multi-candidate generation (§15) already producing ranked, scored alternatives, a future lightweight review UI could surface the top-2/3 candidates for manual override of `CandidateRanker`'s automatic pick, without any change to the generation pipeline itself — the ranking data needed already exists in the manifest.
- **Additional niche templates.** The workflow library (§7) is designed so a new niche is config/data-only (§7.3); creator-taxonomy growth in Module 4 does not require a Module 7 release.

---

## Appendix A — Upstream contract summary (for implementer reference)

Module 7 consumes, unmodified and read-only, the following existing types from `modules/models.py`:

- `PromptPackage` (and its nested `GenerationParameters`, `QualityParameters`, `ModelSettings`) — the direct input.
- `RedesignSpecification` (and its nested `ColorDirection`, `SubjectTreatment`, `TextOverlaySpec`, `ObjectDirective`, `LayoutDirection`) — referenced for context/tracing, not re-parsed as the primary input.
- `FaceAnalysis` / `FaceDetail` (Module 4) — reused for identity-embedding source data, avoiding recomputation.
- `DetectedObject` (Module 4) — reused for object-directive compliance checks in QA.
- The creator-niche classification already present in Module 4's analysis output — reused by `WorkflowLibrary` (§7), not recomputed.

Module 7 introduces, additively, the following new types (design intent, not implementation):

- `ImageGenerationResult` — top-level manifest, mirrors the `status`/`error_message`/`duration_seconds`/`generated_at` convention already established by `RedesignSpecification` and `PromptPackage`; extended in v1.0 with `workflow_version`, `workflow_hash`, `prompt_package_hash`, `generation_hash`, model/LoRA/ControlNet hash fields, `candidate_scores`, and ranking outcome.
- `GeneratedAsset` — path, dimensions, hashes for one output image.
- `FaceMatchResult` — identity-similarity score, threshold, pass/fail.
- `QualityAssuranceReport` — per-check pass/fail and scores for Tier 1/Tier 2 checks, plus the weighted `overall_score` (§16.1).
- `GenerationProfile` *(v1.0)* — the checkpoint/sampler/scheduler/steps/CFG/ControlNet/IPAdapter/restoration/upscaler/expected-VRAM/expected-time bundle (§6.1).
- `WorkflowTemplateRef` *(v1.0)* — resolved `(niche, profile) → template path` reference plus the template's own `workflow_version` (§7, §23.2).
- `CandidateScore` *(v1.0)* — per-candidate score record retained for audit even when the candidate image itself is not persisted (§15.2).
- `GenerationMetrics` *(v1.0)* — the structured per-attempt record appended to `module7_metrics.jsonl` (§26.1).

This appendix exists so implementation work can proceed without re-deriving the upstream contract from scratch.
