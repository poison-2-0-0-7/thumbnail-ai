# Module 7 V2 — Professional AI Thumbnail Editing Engine Architecture

**thumbnail-ai**
**Status:** Design only. No implementation code, no pseudocode. Handoff artifact for the implementation agent.
**Author role:** Principal AI Image Generation Architect
**Source of truth:** `poison-2-0-0-7/thumbnail-ai` @ `main`, read in full for this document — `main.py`; `modules/image_generator.py`; `modules/workflow_library.py`; `modules/models.py`; `modules/config.py`; `modules/generation_components/*` (`workflow_graph_assembler.py`, `conditioning_asset_resolver.py`, `candidate_strategy_planner.py`, `node_fragment_library.py`, `capability_probe.py`, `strategy_pack_resolver.py`, `workflow_graph_cache.py`, `workspace_loader.py`); `modules/composition_engine.py` + `composition_components/`; `modules/thumbnail_planner.py` + `planner_components/`; `modules/decision_engine.py` + `decision_components/`; `modules/asset_extraction_engine.py` + `asset_extraction_components/`; `workflows/*.json`; `workflows/fragments/*.json`; `docs/IMAGE_GENERATION_ARCHITECTURE.md`; `docs/MODULE_10_5_THUMBNAIL_PLANNER_AND_CONDITIONING_PIPELINE_ARCHITECTURE.md`; `docs/MODULE_8_9_INTEGRATION_ARCHITECTURE.md`; `tests/test_image_generator.py`, `tests/test_generation_components`.
**Constraint honored throughout:** this document does not redesign Modules 1–6, 8, 9, 10, or 10.5. It treats their outputs — `PromptPackage`, `AssetExtractionManifest`, `DecisionManifest`, `CompositionWorkspace`/`GenerationBundle`, `GenerationPlan` — as fixed, correct, upstream contracts, exactly as `IMAGE_GENERATION_ARCHITECTURE.md` §2 already establishes ("Module 5 reasons. Module 6 compiles. Module 7 renders."). Every change proposed here is internal to Module 7's own execution strategy.

---

## 1. Executive Summary

The pipeline is structurally complete. Modules 1–6, 8, 9, 10, and 10.5 correctly produce a `PromptPackage`, an `AssetExtractionManifest`, a `DecisionManifest`, a `CompositionWorkspace`/`GenerationBundle`, and a `GenerationPlan` — a full, per-element, per-pixel record of what should be kept, replaced, enhanced, removed, or added, and where. Module 7 receives all of this. **It does not use most of it as an editing constraint — it uses it as a prompting hint.**

The current Module 7 (`image_generator.py::ImageGeneratorPipeline`, `workflows/*.json`) is, underneath its profile/workflow/candidate/QA scaffolding, a **text-to-image generator conditioned by ControlNet and IPAdapter**. Every workflow template starts from `EmptyLatentImage` (pure Gaussian noise) and runs `KSampler` at `denoise: 1.0`. The source thumbnail is never `VAEEncode`d into the pipeline. No inpainting or outpainting fragment exists in `workflows/fragments/`. There is no node, at any point in any of the eleven JSON graphs read for this document, that consumes the original pixels as anything other than a ControlNet *hint image* (edge/depth/segmentation maps) or an IPAdapter *style reference*. Both of those conditioning mechanisms bias a fresh render toward resembling the source — neither of them **is** the source.

This is the root cause of every symptom listed in the brief: identity drift, wrong expressions, changed hairstyles, wrong or missing objects, wrong poses, background over-replacement, unreadable text, vanished brand elements, layout drift, and a generic-SD-art look. All of these are exactly what a strong text-to-image model does when it is only *nudged* toward a reference instead of *constrained* to preserve it pixel-for-pixel where preservation was decided.

A second, independent contributing cause was found during this review: four of the six Tier-1 QA scoring functions in `image_generator.py` — `_calculate_text_safe_zone_score`, `_calculate_object_preservation_score`, `_calculate_color_compliance_score`, `_calculate_composition_score` — are stubs that unconditionally `return 1.0`. Only identity similarity and a generic Laplacian-variance sharpness score are real measurements. `CandidateRanker` and the QA hard-gate logic in `QualityAssuranceStage` are therefore making accept/reject and ranking decisions against a score that is 4/6ths constant. The system is not failing to detect bad candidates — it is structurally unable to.

**Module 7 V2 addresses both causes with one governing idea: the original thumbnail is the base layer, not a hint.** Generation becomes a bounded set of localized edits — driven directly by the `DecisionManifest`'s already-computed per-element `KEEP` / `REPLACE` / `ENHANCE` / `REMOVE` / `ADD` decisions and the `CompositionWorkspace`'s already-computed masks — composited back onto a pixel-exact copy of the source for every region marked `KEEP`. Nothing is regenerated that was already decided to be preserved. This is implemented as an additive execution mode inside the existing `ImageGeneratorPipeline`, gated by a new profile-level flag, so `PROFILE_LOW_VRAM`/legacy behavior is unchanged unless explicitly opted into V2.

---

## 2. Current Module 7 Review

What is real and works, verified against the code (not the design docs, which in places describe intent the implementation does not yet fully match):

- **Two-process model** (`ComfyUIProcessManager`, `ComfyUIClient`) — solid, unchanged in V2.
- **`ProfileSelector`** — four real profiles (`PROFILE_STANDARD`, `PROFILE_FAST`, `PROFILE_PREMIUM`, `PROFILE_LOW_VRAM`) in `config.py::MODULE7_GENERATION_PROFILES`, correctly differentiated by checkpoint (SDXL Juggernaut vs. FLUX.1-schnell GGUF), steps, CFG, and VRAM budget. Reused as-is in V2 — see §5.1.
- **`WorkflowLibrary`** — niche-to-template resolution (`workflows/{niche}.json`, fallback `general.json`) — reused as-is.
- **`WorkflowGraphAssembler`** — a genuinely well-built declarative fragment-merge mechanism: fragments declare an `_attach.point` that must exist in the base template's `_meta.attachment_points`, and namespaced fragment nodes splice into the graph via an `ATTACHMENT_PREVIOUS` sentinel. This mechanism is **exactly** what V2's new inpainting/masking fragments need, and it is reused unchanged — see §7.
- **`IdentityPreservationStage`** — real InsightFace cosine-similarity check with a documented deterministic-hash fallback embedding when InsightFace is unavailable. Real, but currently only a post-hoc *detector*, never a generation-time *constraint*. V2 repositions it as a stage-exit gate on an already-preserved region (§9), not the only line of defense.
- **`FaceRestorationStage`, `UpscaleStage`** — real, reused unchanged as the final two stages of the V2 pipeline (§9, Stage 6/7).
- **`CandidateRanker`** — the ranking logic (hard-gate exclusion, `overall_score` descending, deterministic tie-break by identity similarity then lowest seed) is sound. What feeds it is not — see §3.
- **Reproducibility/hashing** (`generation_hash`, `prompt_package_hash`, `canonical_json_hash`) — real, correctly implemented, reused unchanged.

What is aspirational or absent, verified against the same code:

- **No image-to-image path.** Every `workflows/*.json` base template's node `4` is `EmptyLatentImage`; node `5`'s `KSampler` runs at `"denoise": 1.0` with no exception. `VAEEncode` does not appear in any base template.
- **No inpainting/outpainting fragment.** `workflows/fragments/` has seven fragments; none contains an `VAEEncodeForInpaint`, `InpaintModelConditioning`, or equivalent node. `MODULE_8_9_INTEGRATION_ARCHITECTURE.md` §7 independently reaches the same conclusion and flags it as the single highest-value gap — this document is the "future, separate Module 7 extension document" that doc explicitly deferred to.
- **`text_exclusion_mask.json` cannot do what its name implies.** It attaches `SetLatentNoiseMask` at the `latent` attachment point — but that point is `EmptyLatentImage`'s output (pure noise). Masking a noise latent doesn't protect existing text pixels; there are no existing pixels in the latent to protect. The fragment is well-formed but structurally inert given the current base templates.
- **Four of six Tier-1 QA scores are hardcoded to `1.0`** (§3).
- **`DecisionManifest` and `GenerationPlan` are read but not enforced as pixel constraints.** `ImageGeneratorPipeline.run()` accepts `generation_plan`/`generation_bundle` and uses them to select ControlNet/IPAdapter fragments and populate prompt-adjacent slots (per `MODULE_10_5...` §8) — but nothing in the current graph *guarantees* that a `KEEP`-decided region in the output matches the source region. A ControlNet-depth map biases geometry; it does not pin pixels.

---

## 3. Root Cause Analysis

| Symptom (from brief) | Mechanism |
|---|---|
| Face identity changes | No latent is ever seeded from the source face region; IPAdapter (weight-based style transfer) and ControlNet (structure-only) are both approximations, not constraints. `IdentityPreservationStage` only measures the drift after the fact — measuring identity loss is not the same as preventing it. |
| Wrong expressions, different hairstyle, incorrect poses | Same root cause as above — anything not literally pinned by ControlNet's edge/depth map (which does not encode expression, hair texture, or fine pose) is left to the diffusion model's prior. |
| Incorrect/missing objects, missing products | `ObjectPreservationScore` — the one signal designed to catch this — unconditionally returns `1.0` (`_calculate_object_preservation_score`, `image_generator.py:597`), so a candidate that drops a product is scored identically to one that doesn't. Combined with no inpainting fragment, there is also no mechanism to *edit only the object region* even if the score existed. |
| Different background (when unintended) | Every profile's `DecisionResolver` default for a background layer is `REPLACE` (per `MODULE_10_5...` §6.5), which is correct when a redesign wants a new background — but today's REPLACE is full-frame regeneration at `denoise=1.0`, so anything else in frame is exposed to the same regeneration, not just the background region. |
| Text becomes unreadable | Text is diffusion-rendered (implicitly, via the base prompt) rather than deterministically composited. `docs/IMAGE_GENERATION_ARCHITECTURE.md` §29 already identifies this and defers it as a "future extensibility" item; V2 pulls it forward because it is now blocking quality, not a nice-to-have (§9, Stage 4). `TextSafeZoneScore` is also stubbed to `1.0`, so unreadable text in the safe zone is currently unscored. |
| Brand elements disappear | Same as "incorrect/missing objects" — a logo/watermark asset with no `preserve` directive honored at the pixel level is subject to full-frame regeneration odds, and `ObjectPreservationScore` cannot catch it because it isn't measuring anything. |
| Layout drifts | `CompositionScore` is stubbed to `1.0`; `LayoutDirection`/`TextOverlaySpec.avoid_zones` are prompt-level instructions to a model that has no mechanism to obey spatial instructions precisely — this is a known, general weakness of prompt-only spatial control in diffusion models, not a config error. |
| "Generic Stable Diffusion art rather than an edited YouTube thumbnail" | The single most direct symptom of the txt2img-from-noise architecture. There is no artifact in the pipeline that *forces* the output to visually anchor to the input beyond conditioning-strength hyperparameters (`controlnet_*_strength`, `ipadapter_weight`) which are global sliders, not per-region contracts. |

**Conclusion:** this is not a tuning problem (raising ControlNet/IPAdapter strength further trades identity preservation for visible ControlNet artifacting and IPAdapter over-fitting — a well-known ceiling in SDXL/FLUX ControlNet workflows) and not a prompt-engineering problem (Module 6 already compiles deterministic, well-structured prompts). It is an **architecture problem**: the pipeline never actually edits the source image. Module 7 V2 fixes the architecture, not the parameters.

---

## 4. Design Principles

1. **The source thumbnail is the base layer.** Every pixel not explicitly decided `REPLACE`/`ENHANCE`/`REMOVE`/`ADD` by the (already-computed) `DecisionManifest` is copied into the output byte-for-byte. This is enforced by post-generation masked compositing (§9), not merely by conditioning strength — conditioning biases a model; compositing guarantees an outcome.
2. **Prefer editing over regeneration; generate only inside a decided region.** A background `REPLACE` edits the background mask's region. A `KEEP` face is never inside a diffusion sampling region at all after Stage 1 (§9).
3. **Reuse, don't rebuild.** `WorkflowGraphAssembler`, `WorkflowLibrary`, `ProfileSelector`, `ArtifactWriter`, `MetricsCollector`, the hashing/reproducibility contract, and the `IdentityPreservationStage`/`FaceRestorationStage`/`UpscaleStage` trio are all sound and are reused, not replaced. V2 changes *what graph gets built and in how many passes*, not the surrounding orchestration.
4. **Every element already has a decision; V2's job is to execute it faithfully, not to make a new one.** `DecisionManifest.ResolvedDecision` (KEEP/REMOVE/REPLACE/ENHANCE/ADD, per `models.py:1384-1386`) and `CompositionWorkspace`'s per-layer masks already exist. V2 introduces zero new reasoning about *what* to change — only new machinery for *how* to change exactly and only that.
5. **Additive, flag-gated, backward compatible.** Every change is a new optional code path selected by a new config flag/profile field, following the exact pattern `MODULE_8_9_INTEGRATION_ARCHITECTURE.md` and `MODULE_10_5...` already established for this codebase (`ASSET_EXTRACTION_ENABLED`, `DECISION_ENGINE_ENABLED`, `THUMBNAIL_PLANNER_ENABLED`). Existing profiles keep today's exact txt2img behavior unless a profile explicitly opts into V2.
6. **Fix measurement before trusting ranking.** The QA stubs (§3) are corrected as part of this design, not left as a known gap — a staged, mask-aware pipeline is only verifiably better than the current one if the scores used to compare candidates actually measure what they claim to.
7. **Deterministic where diffusion is unreliable.** Text rendering and color/lighting harmonization are moved to deterministic, non-diffusion post-processing wherever the project already has the tooling to do so (Pillow, Module 4's OCR/color utilities), consistent with the project-wide "deterministic compiler, generative renderer only where necessary" philosophy already stated in `IMAGE_GENERATION_ARCHITECTURE.md` §2.

---

## 5. Module 7 V2 Architecture

### 5.1 Position and entry point

`ImageGeneratorPipeline.run()` gains one new parameter: `edit_mode: Literal["legacy_txt2img", "staged_edit"] = "legacy_txt2img"`. This is not a new top-level function — it is a branch inside the existing pipeline entry point, matching the precedent already set by `generation_plan: GenerationPlan | None = None` in `MODULE_10_5...`'s Module 7 extension (§8 of that document). `main.py`'s existing call site is unaffected until it is explicitly updated to pass `edit_mode="staged_edit"`.

`GenerationProfile` (in `models.py`) gains one new optional field: `edit_mode_default: Literal["legacy_txt2img", "staged_edit"] | None = None`. When `edit_mode="auto"` is requested at the pipeline level, `ProfileSelector` resolves it from the selected profile's `edit_mode_default`, following the exact same "profile encapsulates the decision" pattern §6 of `IMAGE_GENERATION_ARCHITECTURE.md` already uses for hardware trade-offs. No new top-level flag is needed beyond this — profiles remain the single seam where hardware/quality/mode trade-offs live.

### 5.2 Structural shape

```
                    PromptPackage · GenerationBundle · GenerationPlan · DecisionManifest
                                          │
                                          ▼
                    ┌──────────────────────────────────────────────┐
                    │   7.0  RegionPlanValidator (§8, new)          │
                    │        — turns already-decided elements into  │
                    │          a concrete, ordered EditPlan          │
                    └───────────────────────┬────────────────────────┘
                                          │
                                          ▼
                    ┌──────────────────────────────────────────────┐
                    │   7.1  BaseLatentStage (§9 Stage 1, new)       │
                    │        VAEEncode(source thumbnail) → base      │
                    │        latent; establishes the pixel anchor    │
                    └───────────────────────┬────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
       7.2 BackgroundEditStage (§9 Stage 2)         7.3 ObjectEditStage (§9 Stage 3, per-object loop)
       inpaint, masked to background region          inpaint, masked to each REPLACE/ENHANCE/ADD object
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                    ┌──────────────────────────────────────────────┐
                    │   7.4  MaskedCompositeStage (§9, new)          │
                    │        paste-back: every KEEP-region pixel     │
                    │        is the original, byte-for-byte          │
                    └───────────────────────┬────────────────────────┘
                                          ▼
                    ┌──────────────────────────────────────────────┐
                    │   7.5  TypographyStage (§9 Stage 4, new)       │
                    │        deterministic Pillow text compositing   │
                    └───────────────────────┬────────────────────────┘
                                          ▼
                    ┌──────────────────────────────────────────────┐
                    │   7.6  HarmonizationStage (§9 Stage 5, new)    │
                    │        deterministic color/lighting match at   │
                    │        edit-region seams                       │
                    └───────────────────────┬────────────────────────┘
                                          ▼
              (existing, unchanged) FaceRestorationStage → UpscaleStage
                                          ▼
                    ┌──────────────────────────────────────────────┐
                    │   7.7  QualityAssuranceStage (§11, corrected)  │
                    │   7.8  CandidateRanker (§10, reused)           │
                    │   7.9  ArtifactWriter / MetricsCollector       │
                    │        (existing, unchanged)                   │
                    └──────────────────────────────────────────────┘
```

Stages 7.1–7.6 are new. Everything below the horizontal line is the existing, working Module 7 machinery, unmodified in shape — only its inputs change (it now receives an already-mostly-correct composite instead of a raw diffusion output).

---

## 6. Workflow Selection Engine

V2 does not replace `WorkflowLibrary`'s niche-based template lookup — it adds a second, orthogonal selection axis: **edit scope**, derived mechanically from the `DecisionManifest`, not inferred.

**Selection is a lookup table, not a classifier**, consistent with Design Principle 3 (reuse `WorkflowLibrary`'s existing "config/data lookup, not inference" character, `IMAGE_GENERATION_ARCHITECTURE.md` §7):

| Condition (computed directly from `DecisionManifest` + `CompositionWorkspace`, zero new inference) | Edit scope selected |
|---|---|
| No `REPLACE`/`ENHANCE`/`ADD` decisions at all (i.e., `DecisionManifest` is all-`KEEP`, or absent and `DECISION_ENGINE_ENABLED=False`) | **Minimal edit** — harmonization/upscale/restoration only, no diffusion sampling pass at all. Cheapest possible path; also the correct behavior for a thumbnail that Module 5 already judged near-optimal. |
| Only the background layer is `REPLACE`, zero object-level decisions | **Background-only edit** — Stage 2 runs, Stage 3 is a no-op |
| One or more `ObjectAsset`-scoped decisions (`REPLACE`/`ENHANCE`/`ADD`), background `KEEP` | **Object-only edit** — Stage 3 runs (one inpaint pass per decided object, §9), Stage 2 is a no-op |
| Both background and ≥1 object decided non-`KEEP` | **Heavy redesign** — Stage 2 and Stage 3 both run, in that order (background first, so object edits composite against the *new* background, not the stale one) |
| Any decision's target has no resolvable mask in `CompositionWorkspace` (a genuine data gap, not a "nothing to do" case) | **Fallback to `legacy_txt2img`** for that element only, logged as a typed warning (`EditPlanFallbackWarning`, §14) — never silently dropped, never silently upgraded to full-frame regeneration for the whole image |

This table is implemented as `RegionPlanValidator.classify()` (§8) — a pure function over already-computed data, matching Design Principle 4. It requires no new model, no new heuristic weight, and no new LLM call.

**Face-only edit, expression enhancement, and lighting enhancement**, named explicitly in the brief, map onto this same table rather than needing separate scopes: a face-only edit is an object-only edit where the sole decided element is the `PersonAsset`; expression/lighting enhancement is the `ENHANCE` case within object-only or background-only edit respectively (§9 Stage 3/5 handle `ENHANCE` at lower denoise than `REPLACE`, §9.3).

---

## 7. Conditioning Engine

Every conditioning mechanism the current system has — ControlNet (canny/depth/segmentation), IPAdapter (single and, per the multi-object fix already specified in `MODULE_10_5...` §5.3, multi-reference), regional masking, text exclusion — is **kept**, because they are correctly implemented; V2's fix is architectural (edit vs. regenerate), not conditioning-level. Two additions are required, both expressed as new fragments following the exact schema every existing fragment already uses (`_attach.point`, `_attach.output_node`, `graph`), so `WorkflowGraphAssembler` (§2) requires zero changes:

### 7.1 New fragment: `inpaint_base.json`

Attaches at a new attachment point, `"latent_source"`, declared in each edit-capable base template's `_meta.attachment_points` (a template-level additive change — `general.json` and niche templates gain one new `_meta` entry; their `graph` node `4` changes from `EmptyLatentImage` to `VAEEncode` + `InpaintModelConditioning`-equivalent when, and only when, the template is invoked in `staged_edit` mode — the *existing* `legacy_txt2img` templates are untouched copies retained under their current filenames; new templates are added alongside as `{niche}_edit.json`, resolved by `WorkflowLibrary` per the edit-scope table in §6, not by overwriting anything).

- Loads the source thumbnail region (already available — `GenerationBundle.source_thumbnail_path`) and the region's binary mask (already available per-element from `CompositionWorkspace`, the exact masks Module 10 already wrote to disk).
- `VAEEncode`s the source region into a base latent, then applies `VAEEncodeForInpaint`-equivalent masking so `KSampler`'s `denoise` parameter becomes a **true partial-denoise strength** over real content, not a placebo strength over noise.
- `denoise` is threaded from the `DecisionManifest` decision type (§9.3), not hardcoded — `ENHANCE` uses a low value (light repaint, most structure retained), `REPLACE`/`ADD` use a higher value (near-full repaint within the masked region only).

### 7.2 New fragment: `edit_region_mask.json`

A generalization of the existing-but-inert `text_exclusion_mask.json` pattern, corrected to attach at the *new* `latent_source` point (where a real, VAE-encoded latent exists to mask) instead of the old `latent` point (which only ever held noise). Loads a binary region mask (from `CompositionWorkspace`, one per decided element) and applies it via `SetLatentNoiseMask` so only the masked pixels are subject to resampling at all — everything outside the mask is, at the latent level, already anchored to the source's own encoding. This is the belt; §9's post-generation paste-back (Design Principle 1) is the suspenders — masking alone is not fully trusted, because diffusion models are known to bleed slightly across mask boundaries, which is exactly why compositing is also enforced.

### 7.3 Conditioning role summary (reused mechanisms, restated for completeness per the brief's request)

| Mechanism | Role in V2 |
|---|---|
| IPAdapter (single/multi) | Style/identity *bias* during an object or background edit pass — now operating on a masked, partial-denoise region instead of a full frame, so its influence is spatially bounded to where it's needed |
| ControlNet — depth/canny/segmentation | Structural guidance *within* an edit region (e.g., depth-guided background replacement keeps horizon/perspective consistent with the preserved foreground) |
| Face crops / object crops (Module 8 `PersonAsset`/`ObjectAsset`) | IPAdapter reference images, unchanged role, now scoped per-region rather than globally |
| Foreground/background extraction | Defines the inpaint mask boundary directly — Module 8's `SceneAsset.foreground`/`.background` (or VRE fallback) *is* the region mask input to `edit_region_mask.json` |
| Masks (person/object, from `CompositionWorkspace`) | As above — now a first-class generation input via `inpaint_base.json`, not merely a QA/ranking input |
| OCR masks | No longer feed the diffusion graph at all — text moves to deterministic post-compositing (§9 Stage 4), which sidesteps the "inert mask on noise" problem entirely rather than trying to fix it in-graph |
| Negative prompts | Unchanged role, still unioned from `PromptPackage` + `DecisionManifest` `remove`-decisions per `MODULE_10_5...` §6.5's Gap F fix |
| Regional prompts | Unchanged role (`regional_mask_conditioning.json`), now composed with `edit_region_mask.json` when a region needs both semantic (prompt) and spatial (mask) conditioning simultaneously — no conflict, both attach at different points (`positive_conditioning` vs. `latent_source`) |
| Reference images (general) | Unchanged — IPAdapter reference role, per element |

---

## 8. Generation Pipeline

**7.0 `RegionPlanValidator`** (new, `generation_components/region_plan_validator.py`) runs first, before any GPU work, matching the project's "fail loudly and typed, before spending compute" convention already used by `PromptPackageLoader`'s upfront rejection of `status != "success"` packages.

Inputs: `DecisionManifest`, `CompositionWorkspace`, `GenerationPlan` (all already materialized by the time Module 7 runs).
Output: an `EditPlan` — a new, small, frozen Pydantic model: `edit_scope: Literal["none","background_only","object_only","heavy_redesign"]`, `regions: list[EditRegion]` where `EditRegion = {element_id, decision_type, mask_path, denoise_strength, stage: Literal["background","object"]}`.

This is a pure, deterministic mapping (Design Principle 4) — it performs the §6 lookup and resolves each region's `denoise_strength` from a small, named config table (`MODULE7_V2_DENOISE_BY_DECISION`, §19) keyed by `ResolvedDecision.decision_type`, not a per-call heuristic. Any element whose mask cannot be resolved from `CompositionWorkspace` is excluded from `regions` and recorded in `EditPlan.fallback_elements` (§6's fallback row) rather than causing a hard failure — matching the graceful-degradation posture the rest of the pipeline already uses for missing optional data.

Steps 7.1–7.6 execute in the order shown in §5.2. Full step detail is in §9. The pipeline remains **per-video, sequential**, matching `main.py`'s existing per-video error isolation — no change to that boundary.

---

## 9. Multi-stage Editing

### 9.1 Stage 1 — Identity/Base Latent Anchor (`BaseLatentStage`)

`VAEEncode`s the full source thumbnail once (not per-region — the encode is shared and cached in-memory for the remainder of this video's generation, since every subsequent stage reads from the same base latent). This single encode is the pixel anchor referenced throughout the rest of this document. No sampling happens in this stage — it is pure setup, and is why it is listed separately from Stage 2/3 rather than folded into them: its output (the base latent + its VAE-decoded reference copy) is reused by every later stage and by the final paste-back (§9.4), so it must exist exactly once, deterministically, before any stochastic sampling begins.

### 9.2 Stage 2 — Background Edit

Runs only when `EditPlan.edit_scope` includes a background region (§6). Uses `inpaint_base.json` + `edit_region_mask.json` masked to the background region, with ControlNet depth/segmentation (when available, per `MODULE_10_5...` §5.1's Module 8 depth/segmentation sourcing) to keep replaced background geometrically consistent with the preserved foreground's perspective. `denoise` resolved per §8's table — high, since "background replace" is semantically a full repaint of that region, but *only* that region.

### 9.3 Stage 3 — Object Edit (per-object loop)

Runs once per element in `EditPlan.regions` with `stage == "object"`, in the order Module 9's `DecisionManifest` lists them (stable, deterministic — no reordering introduced here). Each iteration is a full `inpaint_base.json` pass masked to that single object's region, so a `REPLACE` on one product does not touch a `KEEP` product two inches away in the frame — this is the direct fix for "incorrect objects" and "missing products," because each object's fate is now an independent, spatially bounded operation instead of a shared consequence of one global sample.

- `REPLACE` → high denoise, full IPAdapter/ControlNet conditioning from the *new* reference asset.
- `ENHANCE` → low denoise, IPAdapter conditioned on the *existing* crop (Module 8's own `ObjectAsset.crop` for that element) — a light repaint that improves quality/lighting without changing content, directly serving the brief's "expression enhancement" / "lighting enhancement" workflow types when applied to a `PersonAsset`.
- `ADD` → high denoise within a *new* mask region (from `CompositionWorkspace`'s placement engine, which already computes where a new element should go per `composition_components/placement_engine.py`) — no existing pixels are touched because the mask, by construction, covers previously-empty canvas.
- `REMOVE` is **not** a Stage 3 operation — a `REMOVE` decision is executed as: exclude the element's region from every other region's mask (so nothing else's inpaint accidentally reconstructs it), then let Stage 2's background inpaint (if the removed element overlapped the background) or a small dedicated erase-inpaint (background-conditioned, no reference image) fill the hole. This directly fixes "brand elements disappear" *as an unwanted side effect* — a deliberate `REMOVE` still works exactly as before; an *undesired* disappearance can no longer happen to an element that was never decided `REMOVE`, because non-`REMOVE` elements are never inside anyone else's sampling mask.

### 9.4 Stage 3.5 — Masked Composite (`MaskedCompositeStage`, new)

Not a sampling stage — a deterministic Pillow/`numpy` alpha composite: `output = source_pixels` everywhere, overwritten only inside the union of masks that were actually sampled in Stage 2/3. This is Design Principle 1 made literal. It is the single most important new component in this document, because it is what converts "the model was strongly biased to preserve X" into "X is guaranteed byte-identical to the source." It runs even when `edit_mode="staged_edit"` produces zero diffusion passes at all (the "minimal edit" scope from §6) — in that case it is a no-op copy, which is correct: nothing was decided to change.

### 9.5 Stage 4 — Typography (`TypographyStage`, new)

Deterministic Pillow text rendering into `TextOverlaySpec`/`GenerationPlan.headline_placement_zone`'s reserved region (already computed, already safe-zoned, per `MODULE_10_5...` §6.4/§7), using the font/weight/color already specified by `DesignBlueprint`. This directly and completely resolves "text becomes unreadable" — it removes text from the diffusion model's responsibility entirely, exactly as `IMAGE_GENERATION_ARCHITECTURE.md` §29 already proposed as a future Module 8 extension. V2 promotes it from "future extensibility" to "required, because it is now blocking output quality," and implements it as a Module 7 internal stage (not a new pipeline module) since it operates on Module 7's own in-flight composite, not on a new standalone artifact.

### 9.6 Stage 5 — Harmonization (`HarmonizationStage`, new)

Deterministic seam correction at every edit-region boundary: local histogram/luminance matching (Lab-space mean/std transfer, a well-understood, cheap, non-AI technique) between each edited region and its immediately surrounding preserved pixels, so a background-replace or object-edit doesn't produce a visible lighting/color seam against the untouched majority of the frame. This is the direct mechanism behind `ColorComplianceScore` once that score is corrected (§11) — the stage's own before/after Lab-distance measurement *is* the score, not a separate computation.

### 9.7 Stage 6/7 — Face Restoration, Upscale (existing, unchanged)

`FaceRestorationStage` and `UpscaleStage` run exactly as they do today, but now operate on a composite where the face region is, for `KEEP`, already pixel-identical to source (restoration only needs to correct genuine diffusion artifacts in `ENHANCE`/`REPLACE` face regions, if any — a strictly smaller job than today's "correct whatever the full-frame model produced").

---

## 10. Candidate Generation

Candidate counts (`1`/`2`/`4`/`8`, `GenerationParameters.num_candidates`) and the seed-stride mechanism are **unchanged** — reused exactly as specified in `IMAGE_GENERATION_ARCHITECTURE.md` §15.1. What changes is *what varies between candidates*:

- **Stage 1's base latent is identical across all candidates of a video** — it is not stochastic (VAE encoding is deterministic), so it is computed once and shared, not once per candidate. This is a direct VRAM/time saving unavailable to the legacy pipeline (§13).
- **Only Stage 2/3's sampling passes vary by seed.** Because `MaskedCompositeStage` (§9.4) guarantees `KEEP` regions are byte-identical regardless of seed, candidate diversity is concentrated entirely in the regions that were actually decided to change — which is also exactly where diversity is *useful*. The legacy pipeline's candidate diversity was spread across the entire frame, including regions the user never wanted to vary in the first place.
- **Diversification axes**, in priority order: (1) seed stride (existing mechanism, reused), (2) for `num_candidates >= 4`, alternate between the `REPLACE`-path's two available reference/IPAdapter weight settings (a "closer to reference" vs. "more creative" pair, both already valid per §7.3's IPAdapter role — no new asset needed, just two config-defined weight presets), (3) for `num_candidates == 8`, additionally alternate ControlNet strength between two presets (tighter/looser structural guidance). This is a small, closed, config-defined set (Design Principle 4) — never an open-ended parameter search.
- Because Stage 1 is shared and `KEEP` regions never enter sampling, **N candidates are not N full-frame generations** — they are N partial-region generations sharing one base encode, which is the primary lever making 4–8 candidates practical on the target hardware (§13).

---

## 11. Candidate Evaluation

The stub scores identified in §3 are corrected as part of this document, because §10's diversity strategy is only meaningful if ranking can actually tell candidates apart on the dimensions that matter:

| Score | Current state | V2 implementation |
|---|---|---|
| `ObjectPreservationScore` | Stub, `1.0` | Run the already-vendored `yolo11n.pt` over each `KEEP`/`ENHANCE` object's region in the candidate output; compare detected class + bbox IoU against Module 8's `ObjectAsset` record for that element. Score = fraction of decided-preserve objects still detected above an IoU/confidence threshold. Reuses an existing model dependency — no new weight file. |
| `TextSafeZoneScore` | Stub, `1.0` | Simplifies almost to a formality once Stage 4 (§9.5) owns text rendering deterministically — becomes a check that no Stage 2/3 sampling mask overlapped `headline_placement_zone` (a bbox-overlap check against data already computed by `GenerationPlan`), not an OCR-on-pixels check. Cheap, and structurally guaranteed to pass except in a genuine placement-logic bug — which is exactly what a hard gate should be for. |
| `ColorComplianceScore` | Stub, `1.0` | Direct output of Stage 5's harmonization measurement (§9.6) — the Lab-space distance already computed to *perform* harmonization is reused, not recomputed, as the score (Design Principle 3: reuse). |
| `CompositionScore` | Stub, `1.0` | Reuses Module 8's `CompositionAsset.eye_flow_map`/`.negative_space_mask` (already computed, already available per `MODULE_10_5...` §5.1) compared against `DesignBlueprint`'s target layout — a masked-region-overlap metric, not a new CV pass. |
| `IdentityScore`, face quality | Already real | Unchanged — reused as-is (§2). |

`CandidateRanker`'s weighted-sum logic and hard-gate/tie-break rules (`IMAGE_GENERATION_ARCHITECTURE.md` §16.1) are **reused unchanged** — only their inputs stop being constants. `MODULE7_QA_WEIGHTS`' existing validate-sums-to-1.0 contract (`validate_qa_weights()`, already implemented in `image_generator.py:511`) is unchanged.

One new hard gate is added, specific to staged editing: **paste-back integrity** — a byte-level check that every pixel outside the union of sampled masks is in fact identical to the source (a cheap `numpy` array-equality check against the known mask union). This can only fail from a `MaskedCompositeStage` bug, not from model behavior — it exists as a regression guard for Design Principle 1 itself, disqualifying a candidate immediately (no scoring needed) if the core guarantee of this entire document was somehow violated.

---

## 12. Iterative Refinement

Refinement in V2 is **per-region, not per-frame** — a direct consequence of the staged architecture (§9) and a deliberate improvement over a hypothetical "regenerate the whole image again" loop, which would reintroduce the exact problem this document exists to fix (re-exposing `KEEP` regions to fresh sampling risk).

```
For each region in EditPlan.regions (§8):
    generate (Stage 2 or 3, per region)
        ↓
    evaluate that region only (region-scoped ObjectPreservationScore / IdentityScore / ColorComplianceScore, §11)
        ↓
    hard gate failed? ──No──▶ keep, proceed to next region
        │Yes
        ▼
    identify which conditioning knob is implicated:
      - identity/object drift → increase IPAdapter weight one preset step, reduce denoise one preset step
      - structural drift → increase ControlNet strength one preset step
      - both already at max preset → escalate: shrink the region's mask by one erosion step (tighter, safer edit) and retry
        ↓
    retry that region only (new seed, same base latent — Stage 1 is not redone)
        ↓
    max 2 retries per region ──▶ exhausted: exclude this region's edit from the candidate (fall back to KEEP for
                                  that specific element — i.e., paste back the original for that region rather
                                  than shipping a low-quality edit), log a typed partial-failure reason on the
                                  result, continue to next region
```

**Stopping conditions:** 2 retries per region (a small, config-defined `MODULE7_V2_MAX_REGION_RETRIES`, not a global retry budget — bounding retries per-region keeps worst-case total attempts proportional to the number of *decided* elements, not the whole frame, which for a typical thumbnail with 1–3 non-`KEEP` elements is a small, predictable number). **Performance constraint:** total wall-clock budget per video is still governed by `GenerationProfile.expected_generation_seconds` (existing field) — a region that would exceed the remaining budget after its first attempt skips retry and falls back to paste-back-original immediately, favoring a correct-but-unedited region over a slow, possibly-still-wrong one. This fallback-to-original behavior is only safe *because* of Design Principle 1 — an unedited region is never a broken region, merely an unimproved one, which is a strictly better failure mode than what the legacy pipeline could offer (a full-frame regeneration has no equivalent "just don't" fallback).

---

## 13. Performance Optimization

Target: RTX 4060 Laptop GPU, 16 GB system RAM, matching `IMAGE_GENERATION_ARCHITECTURE.md`'s existing 8 GB VRAM budget framing.

- **Stage 1 amortization** (§10) — one `VAEEncode` per video, not per candidate or per region. On an 8-region, 4-candidate video, this is a ~32x reduction in encode calls versus a naive "encode per attempt" approach, though the legacy pipeline had zero encodes to begin with — the honest comparison is that V2 adds exactly one cheap encode op per video in exchange for removing many expensive full-frame sampling passes.
- **Smaller sampling regions, not smaller images.** A masked inpaint over, say, a 400×300px background region at the same step count is meaningfully cheaper per-step (proportional to the number of latent tokens inside the mask, for the attention layers that dominate SDXL/FLUX cost) than a full 1344×768 frame — this is the primary throughput win, independent of any step-count reduction.
- **Per-region step budgets, tuned lower than legacy's single global pass.** Because each region's edit is narrower in scope (one object, or the background, not "the whole creative direction"), fewer steps are needed to converge — `MODULE7_V2_STEPS_BY_DECISION` (new, small config table, mirroring `MODULE7_V2_DENOISE_BY_DECISION`'s structure) sets `ENHANCE` lower than `REPLACE`, both lower than legacy's flat 30/16/20-step profiles.
- **Reuse `WorkflowGraphAssembler`'s existing per-run caching seam** (`generation_components/workflow_graph_cache.py` already exists) — the assembled graph *structure* for a given `(niche, profile, edit_scope)` triple is cacheable across videos with the same shape, since only slot values (paths, seeds, strengths) differ between runs, not graph topology. This is an existing component, not a new one — V2 simply exercises it against more, smaller graphs instead of one large one.
- **ComfyUI server stays warm across regions and candidates within a video**, exactly as it already does across videos (§2's two-process model) — no new process-lifecycle work needed, this is a direct consequence of Module 7 already treating ComfyUI as a persistent service.
- **VRAM ceiling check before Stage 3's per-object loop begins** — if `EditPlan.regions` has many object-scoped entries (a thumbnail with many decided products, say), `capability_probe.py` (already exists) is consulted once up front to decide whether all objects can be queued in one ComfyUI batch or must be strictly sequential; this reuses existing capability-probing machinery rather than adding new VRAM-measurement code.
- **`PROFILE_LOW_VRAM` gets its own `edit_mode_default`** — for the tightest hardware tier, background-only edit scope may be forced regardless of what `EditPlan.classify()` would otherwise select (a profile-level override, not a change to the classification logic itself), trading some object-level fidelity for guaranteed headroom — an explicit, logged, config-driven choice, not a silent degrade (Design Principle 5/`IMAGE_GENERATION_ARCHITECTURE.md`'s existing "graceful degradation, never silent" principle, §2).

---

## 14. Failure Recovery

Every case named in the brief, mapped to the existing typed-exception convention (`module7_exceptions.py`'s hierarchy, extended additively — no existing exception type is renamed or removed):

| Failure | V2 handling |
|---|---|
| Missing face | `RegionPlanValidator` finds no `PersonAsset` for a `PERSON`-typed decision → that decision moves to `EditPlan.fallback_elements` (§8) rather than raising; downstream, Stage 1's identity anchor step is skipped for that video, and `IdentityPreservationStage.verify()` already returns a `skipped=True` no-op result for this exact case (existing code, `image_generator.py:636-638`) — reused unchanged. |
| Tiny face (below a usable-crop-size threshold) | Same `EditPlan.fallback_elements` path, with a specific `reason="face_too_small"` — object-only/background-only scope proceeds for other elements; the face region defaults to `KEEP` (paste-back), which is always safe per Design Principle 1. |
| Multiple faces | Already handled upstream by Module 8's `PersonAsset[]` (plural) and Module 9's per-person decisions — V2's per-element Stage 3 loop (§9.3) naturally handles N people as N independent regions; no special-casing needed beyond what §9.3 already describes. |
| Missing objects (an `ObjectAsset` the `DecisionManifest` references isn't actually in `CompositionWorkspace`) | `RegionPlanValidator`'s existing fallback-element path (§8) — same mechanism as missing face, `reason="object_asset_unresolved"`. |
| Bad OCR | No longer a generation-time risk once Stage 4 (§9.5) owns text deterministically from `DesignBlueprint`'s already-validated copy — an OCR failure upstream (Module 4) is a Module 4/5 concern, unchanged by this document, and was already out of scope per Design Principle 3. |
| ControlNet failure (node error, missing model file) | New typed `ControlNetFragmentError` (extends existing `Module7Error`) — caught per-region (not per-video): that region's edit falls back to IPAdapter-only conditioning (already-existing fragment, just drop the ControlNet fragment from that region's fragment list) before falling back further to paste-back-original if IPAdapter-only also fails its own gate. |
| IPAdapter failure | Symmetric to above — falls back to ControlNet-only, then to paste-back-original. Two independent conditioning mechanisms failing simultaneously for the same region is treated as that region exhausting its retries (§12), not a whole-video failure. |
| Workflow (graph assembly) failure | Existing `FragmentAttachmentError` (already implemented in `WorkflowGraphAssembler`, §2) — unchanged, still fatal for the affected region's edit, still non-fatal for the video (falls back to paste-back for that region only, which is a *stronger* recovery position than the legacy pipeline had, since legacy had no "just don't edit" option). |
| Out of VRAM | Existing `ComfyUIClient` OOM handling (Tenacity-wrapped retry) is reused; V2 adds one new recovery step ahead of a full retry — first attempt to shrink the *current region's* batch/resolution (a region is already smaller than a full frame, §13, so this has headroom the legacy pipeline's full-frame OOM handling didn't), only falling through to `ProfileSelector`'s existing profile-downgrade path (§6 of `IMAGE_GENERATION_ARCHITECTURE.md`) if the region-level shrink doesn't recover. |
| Poor candidate quality (all candidates fail hard gates) | Existing behavior, unchanged in shape: `status: "error"` after `MAX_GENERATION_RETRIES` exhausted (`IMAGE_GENERATION_ARCHITECTURE.md` §16.1) — but because paste-back-original is always a valid per-region fallback (Design Principle 1), the realistic worst case in V2 is "a mostly-original thumbnail with the safest subset of edits applied," not a hard video-level failure, for any video where at least the `RegionPlanValidator` step itself succeeded. |
| Timeouts | Existing per-stage timeout config (implied by `expected_generation_seconds` per profile) — extended with a per-region soft-timeout that triggers the "skip retry, paste back original" path described in §12, rather than only a single video-level timeout. |

---

## 15. Migration Plan

Every phase independently shippable, additive, defaulting to a no-op, matching the phasing discipline `MODULE_8_9_INTEGRATION_ARCHITECTURE.md` §8 already established for this project. No existing test should need to change.

| Phase | Change | Flag / gate | Default | Breaking? |
|---|---|---|---|---|
| **0** | Fix the four stub QA scores (§11) | — | always on | No — scores were previously constant; any test asserting `overall_score` against a fixture will need its fixture's expected value recomputed, which is a test-data update, not a behavioral break in Module 7 itself |
| **1** | Add `EditPlan`/`EditRegion` models, `RegionPlanValidator` (§8), no wiring into generation yet — pure new code, unreachable from `main.py` | — | inert | No |
| **2** | Add `inpaint_base.json` / `edit_region_mask.json` fragments and `{niche}_edit.json` base templates alongside existing templates (§7) — new files only, zero existing files modified | — | inert until selected | No |
| **3** | Add `BaseLatentStage`, `MaskedCompositeStage` (§9.1/§9.4), wire into `ImageGeneratorPipeline.run()` behind `edit_mode` parameter, defaulting to `"legacy_txt2img"` | `edit_mode` param, default `"legacy_txt2img"` | off | No |
| **4** | Add `BackgroundEditStage`/`ObjectEditStage` (§9.2/§9.3), reachable only when `edit_mode="staged_edit"` | same | off | No |
| **5** | Add `TypographyStage`/`HarmonizationStage` (§9.5/§9.6) | same | off | No |
| **6** | Add `GenerationProfile.edit_mode_default` field; introduce one new profile, `PROFILE_STANDARD_EDIT`, as a copy of `PROFILE_STANDARD` with `edit_mode_default="staged_edit"` — existing profiles' fields are untouched | new profile entry, opt-in | existing profiles unaffected | No |
| **7** | `main.py` updated to pass `edit_mode="auto"` (resolves via profile, §5.1) — the one call-site change that actually turns V2 on for anyone using `PROFILE_STANDARD_EDIT` | — | only affects the new profile | No |
| **8** (future, separate doc) | Outpainting fragment, for canvas-extension use cases — explicitly out of scope here per the brief | new | off | No |

**Rollout order matters:** Phase 0 should ship first and alone — correcting QA scoring changes what "good" looks like for *every* future phase's own validation, so shipping it first means Phases 3–7 can be evaluated against real signal from the start rather than needing a re-evaluation pass later. Phases 1–2 have no dependency on each other and can ship together. Phase 3 depends on 1–2 (needs the plan model and the new fragments to exist). Phases 4–5 each depend on 3 but not on each other. Phase 6–7 depend on all prior phases being complete.

---

## 16. Testing Strategy

Mirrors the project's existing 1:1 module-to-test-file convention (`tests/test_image_generator.py`, `tests/test_generation_components/`):

- **`RegionPlanValidator`** — pure-function unit tests, table-driven directly off §6's classification table: given a synthetic `DecisionManifest` + `CompositionWorkspace`, assert the exact `edit_scope` and `EditRegion[]` produced, including the fallback-element path for unresolvable masks. No GPU, no ComfyUI — same style as existing `test_prompt_compiler.py`/`test_redesign_spec_engine.py`.
- **`MaskedCompositeStage`** — the highest-value test in this entire document: given a synthetic "generated" image that is *deliberately wrong* everywhere outside a known mask, assert the composite output is byte-identical to the source outside that mask and equals the "generated" pixels only inside it. This is a regression guard for Design Principle 1 itself and should be treated as a release-blocking test, not an optional one.
- **QA score corrections (§11)** — each of the four previously-stubbed functions gets its own fixture-based unit test with known-good and known-bad synthetic inputs (e.g., an image with a product literally deleted, asserting `ObjectPreservationScore` drops below threshold) — these did not exist before (a stub needs no test) and are new, not modified, test files.
- **Fragment schema tests** — `inpaint_base.json`/`edit_region_mask.json` validated against the same fragment-schema assertions `WorkflowGraphAssembler`'s existing tests already apply to the seven current fragments (attachment point exists, required placeholders present) — extend the existing parametrized test list, do not write a new test harness.
- **Staged pipeline integration test** — one synthetic end-to-end run per §6 edit-scope category (minimal / background-only / object-only / heavy-redesign), asserting the correct stages ran (via `MetricsCollector`'s existing per-stage timing records, reused as an assertion surface) and skipped stages truly produced zero ComfyUI queue submissions — matching `test_main_pipeline.py`'s existing integration-test style.
- **Failure-injection tests**, one per row of §14's table — reusing the project's existing pattern (seen in `test_comfyui_client.py`) of mocking `ComfyUIClient` to raise each typed error and asserting the documented fallback occurs, not a crash.
- **Backward-compatibility test** — explicit assertion that `edit_mode="legacy_txt2img"` (the default) produces a byte-identical `ImageGenerationResult` to the pre-V2 pipeline given the same `PromptPackage`/seed, guarding Phase 3–7's "no breaking change" claim mechanically, not just by code review.
- **Performance regression test** (marked slow/optional in CI, per existing `pytest.ini` marker conventions if present) — asserts a staged-edit run's total ComfyUI queue time for a representative fixture is less than a legacy full-frame run's, guarding §13's throughput claims over time.

---

## 17. Future Extensions

Explicitly deferred, not designed here, consistent with the brief's scope boundary ("do not redesign unrelated modules"):

- **Outpainting** (§15 Phase 8) — no current taxonomy entry in Modules 5/9/10.5 calls for canvas extension; add only if that changes upstream.
- **LoRA-based per-creator identity locking** — already reserved as a future extension point in `IMAGE_GENERATION_ARCHITECTURE.md` §29 (`lora_hashes` field already reserved in the manifest); V2's per-region IPAdapter conditioning is a natural place to also apply a per-creator LoRA once trained, but training/management of those LoRAs is out of scope here.
- **Human-in-the-loop region override** — with `EditPlan` now an explicit, inspectable, per-region artifact (unlike the legacy pipeline's implicit whole-frame decision), a future lightweight review step could let an operator override a single region's decision (e.g., force a `REPLACE` back to `KEEP`) before generation runs, without touching Module 9's own reasoning — this document's `EditPlan` model is designed to make that a config/data change later, not an architecture change.
- **Generated (non-preserved) headline copy** — still deferred exactly as `MODULE_10_5...` §6.4 already defers it; Stage 4 (§9.5) renders whatever headline `GenerationPlan.headline` already contains, it does not write new copy.
- **Shared model registry across Modules 4/6.5/8** — flagged as out of scope by `MODULE_8_9_INTEGRATION_ARCHITECTURE.md`'s own Risks table; V2 does not add a fifth call site to that same redundant-inference problem (Stage 1's VAE encode is a different model family — the checkpoint's own VAE — not InsightFace/YOLO, so it does not worsen this existing, separately-tracked issue), but does not fix it either.
- **Segmentation-driven region masks beyond what Module 8 already computes** — V2 consumes `CompositionWorkspace`'s existing masks as-is (Design Principle 3); finer-grained, generation-time segmentation (e.g., SAM2-refined masks per candidate) is a plausible V3 idea, explicitly not designed here.
