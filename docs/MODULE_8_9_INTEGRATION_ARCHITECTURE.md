# Module 8/9 Integration Architecture
## `thumbnail-ai` — Unifying Asset Extraction, Decision Engine, Composer, and Planner into One Live Pipeline

**Status:** Design only. No implementation code, no pseudocode. Handoff artifact for Codex.
**Source of truth:** `poison-2-0-0-7/thumbnail-ai` @ `main`, cloned and read in full for this document — `main.py`, `modules/models.py`, `modules/config.py`, `modules/asset_extraction_engine.py`, `modules/decision_engine.py` (+ `decision_components/`), `modules/composition_engine.py` (+ `composition_components/`), `modules/thumbnail_planner.py` (+ `planner_components/`), `modules/visual_reference_engine.py`, `modules/image_generator.py`, `workflows/*.json`, `workflows/fragments/*.json`, `pytest.ini`.
**Constraint honored throughout:** nothing below redesigns a module that already works. Every change is additive, optional-parameter, or feature-flag-gated, matching the pattern already established by `THUMBNAIL_PLANNER_ENABLED` and `DECISION_ENGINE_ENABLED` in `config.py`.

---

## 0. Executive Summary

The live pipeline in `main.py` today is:

```
1 (CSV) → 2 (Metadata) → 3 (Thumbnail) → 4 (Intelligence) → 5 (Redesign Spec)
  → 5.5 (Design Blueprint) → 6 (Prompt Compiler) → 10 (Asset Composer)
  → 10.5 (Thumbnail Planner, flag-gated) → 7 (Image Generation)
```

Module 8 (Asset Extraction Engine) and Module 9 (AI Decision Engine) are fully implemented, fully tested, and **never called** from this chain. This is not a small oversight — both were built with integration in mind:

- Module 9's `load_input_bundle()` already reads Module 8's manifest **if present**, degrading gracefully when absent.
- Module 10.5's `load_planner_input_bundle()` already reads **both** Module 8's and Module 9's manifests optionally, and a `PrecedenceResolver` already prefers Module 9's decisions over Module 10's own fallback decisions **when computing generation strategy**.

So the artifacts, schemas, and precedence logic for this integration are already written. What's missing is narrower than a new architecture: (1) two function calls in `main.py`, (2) one new config flag, (3) one new optional parameter on Module 10 so its physically-baked masks/layers respect Module 9's decisions and not just Module 10.5's derived strategy text, and (4) a fix to how Module 6.5 and Module 8 currently do redundant CV inference over the same image. This document specifies exactly those four things and nothing else.

---

## 1. Verified Current State (repository facts, not assumptions)

| Fact | Evidence |
|---|---|
| Live pipeline order is 1→2→3→4→5→5.5→6→10→10.5→7 | `main.py::_run_pipeline_creators`, module-boundary comments |
| Module 8 and Module 9 are never imported or called in `main.py` | `grep -n "Module 8\|Module 9\|asset_extraction_engine\|decision_engine" main.py` → no matches |
| `DECISION_ENGINE_ENABLED: bool = False` already exists as a config flag | `modules/config.py:628` |
| No equivalent `ASSET_EXTRACTION_ENABLED` flag exists yet | absent from `config.py`; only `ASSET_EXTRACTION_CACHE_ENABLED` (a caching switch, not an on/off switch) exists |
| Module 8's `extract_assets(video_id, source_image_path, intelligence)` depends only on Module 3 (thumbnail path) + Module 4 (`ThumbnailIntelligence`) | `asset_extraction_engine.py::AssetExtractionEngine.extract` signature and body |
| Module 9's `load_input_bundle()` requires Module 4, Module 5, and Module 6 artifacts (hard dependency) and Module 8's manifest (soft — degrades to `None` if missing) | `decision_components/io.py::load_input_bundle`, comments "Degrades gracefully if missing" |
| Module 10 (`AssetComposer.prepare_generation_workspace`) reads only Module 5's `RedesignSpecification` and internally instantiates `VisualReferenceEngine` (Module 6.5) — it does not import or reference Module 8 or Module 9 anywhere | `grep` over `composition_engine.py` + `composition_components/` returned zero hits for `AssetExtractionManifest` / `DecisionManifest` |
| Module 10's own `DecisionResolver` re-derives KEEP/REMOVE/REPLACE/ENHANCE/ADD decisions directly from `RedesignSpecification`, using a hardcoded rule table, independent of Module 9 | `composition_components/decision_resolver.py` |
| Module 10.5 (`ThumbnailPlanner`) already optionally loads both Module 8's `AssetExtractionManifest` and Module 9's `DecisionManifest` from disk | `planner_components/io.py::load_planner_input_bundle`, §5/§6 |
| Module 10.5's `PrecedenceResolver` already prefers Module 9's `DecisionManifest` over Module 10's baked-in layer decisions, **but only for deriving `FaceStrategy` / `BackgroundStrategy` / `preserve_objects` / lighting** — it does not and cannot retroactively change the masks/crops Module 10 already wrote to disk | `planner_components/precedence_resolver.py`, `planner_components/strategy_deriver.py` |
| Module 7's `ImageGeneratorPipeline.run()` already derives ControlNet/IPAdapter fragment selection internally from `generation_plan`/`generation_bundle` — this wiring is real and live, not aspirational | `image_generator.py::_select_fragments`, `run_image_generation_pipeline` |
| No inpainting or outpainting ComfyUI fragment exists | `workflows/fragments/` contains only `controlnet_canny.json`, `controlnet_depth.json`, `controlnet_segmentation.json`, `ipadapter_reference.json`, `multi_object_reference.json`, `regional_mask_conditioning.json`, `text_exclusion_mask.json` |
| `PROFILE_PREMIUM` (Flux checkpoint) runs with `controlnet_enabled=False` | `config.py::MODULE7_GENERATION_PROFILES` |
| Module 8/9 use package-qualified imports (`from modules.config import ...`); every other pipeline module uses flat imports (`from config import ...`) | `asset_extraction_engine.py`, `decision_engine.py` vs. `composition_engine.py` |
| `main.py` puts only `modules/` on `sys.path`; Module 8/9's package-qualified imports resolve today only because `python main.py` implicitly adds the *project root* (the script's own directory) to `sys.path[0]` — this is fragile, not broken, but must not be assumed to survive refactors | `main.py` sys.path setup vs. `pytest.ini`'s `pythonpath = modules .` (which explicitly adds both) |

---

## 2. Canonical Ownership — Resolving the Module 4 / 6.5 / 8 Overlap

**The problem, stated precisely:** Module 6.5 (Visual Reference Engine) and Module 8 (Asset Extraction Engine) both run face detection and object detection over the *same source thumbnail image*, using overlapping vision-stack models (InsightFace, YOLO/GroundingDINO), and both produce their own independent output artifacts. If both run in the live pipeline as currently written, GPU inference for faces and objects happens twice per creator — a real cost against the RTX 4060 / 16 GB VRAM constraint that governs every performance decision in this project.

**Resolution — ownership table:**

| Responsibility | Canonical owner | Why |
|---|---|---|
| Semantic understanding of the source thumbnail (what's present, roughly where, sentiment, composition grid, LLM reasoning) | **Module 4** (unchanged) | Already the pipeline's only reasoning stage; nothing here overlaps with 6.5 or 8, which are both non-reasoning. |
| Comprehensive, multi-family, reusable pixel-asset catalog — every face, every person, every object, typography regions, effects, scene graph, full segmentation/depth/composition metadata | **Module 8** (becomes canonical) | Module 8 is the only one of the two designed for *multiple* people/objects, typography extraction, and effects detection. It is also the one Module 9 already depends on for reasoning, and the one Module 10.5 already optionally consumes. Making it canonical means every downstream consumer (9, 10.5, evaluation/QA, future analytics) reads from one asset source instead of several disagreeing ones. |
| The narrow, ComfyUI-conditioning-shaped subset of that catalog — `face_crop_path`, `face_mask_path`, `foreground.png`/`background.png`, one hero `object_crop.png`/`object_mask.png`, `depth_map.png`, `canny_map.png` | **Module 6.5 becomes a thin projection over Module 8**, not an independent extractor | Module 6.5's *public interface* — `VisualReferenceEngine.prepare_assets()`, its output file paths, its `VisualReferenceManifest` schema — does not change. What changes is only its internal data source: when a Module 8 manifest exists for the `video_id`, Module 6.5 projects the relevant fields from it (the largest/primary `PersonAsset` → face crop+mask, the primary `ObjectAsset` → object crop+mask, `SceneAsset`'s depth/canny if present) instead of re-running InsightFace/YOLO itself. When no Module 8 manifest exists (Module 8 disabled, or this `video_id` hasn't been processed by it yet), Module 6.5 falls back to its current direct-CV-extraction behavior, byte-for-byte identical to today. This is purely additive and fully backward compatible — no caller of Module 6.5 (i.e., Module 10) needs to change at all for this specific fix. |

This resolves the duplication without touching Module 10's call site, without changing Module 6.5's schema, and without asking Module 8 to do anything it doesn't already do.

---

## 3. Integration Point 1 — Module 8 into the Live Pipeline

**Where it runs:** immediately after Module 4, in parallel conceptually with Modules 5/5.5/6 (it has no dependency on any of them — only on Module 3's thumbnail path and Module 4's `ThumbnailIntelligence`). Placing it right after Module 4 keeps the topological order clean and means its manifest is available in time for Module 9 later in the same creator's iteration.

**Call added to `main.py::_run_pipeline_creators`, immediately after the existing Module 4 block:**

- New optional block, gated by a new `ASSET_EXTRACTION_ENABLED` flag (added to `config.py`, default `False`, matching the existing `DECISION_ENGINE_ENABLED` / `THUMBNAIL_PLANNER_ENABLED` pattern).
- Calls the existing public API `extract_assets(video_id, source_image_path=str(thumbnail.thumbnail_path), intelligence=intelligence)` — no new function needed, this top-level helper already exists and already persists its manifest atomically to disk internally.
- On `AssetExtractionBaseError` (its existing exception hierarchy root — verify exact name against `asset_extraction_exceptions.py` at implementation time), logs and increments `skipped`, **only if** `ASSET_EXTRACTION_REQUIRED` is also introduced as a stricter sub-flag defaulting `False` — by default, a Module 8 failure should **not** abort the creator, because Module 9/10.5 already treat its absence as a soft dependency. This mirrors the existing `THUMBNAIL_PLANNER_ENABLED` block's degrade-not-abort posture, except even softer: log a warning and continue with `asset_extraction=None` implicitly (nothing downstream needs the in-memory object; everything reads it back from disk by `video_id`).

**New parameter threaded through:** `asset_extraction_dir: Path = DEFAULT_ASSET_EXTRACTION_DIR` added to `run_pipeline()` / `_run_pipeline_creators()`'s signature, following the exact pattern of `redesign_spec_dir`, `design_blueprint_dir`, `prompt_package_dir` already there. Defaults to the existing `DEFAULT_ASSET_EXTRACTION_DIR` constant, so omitting it changes nothing.

---

## 4. Integration Point 2 — Module 9 into the Live Pipeline

**Where it runs:** after Module 6 (hard dependency: needs `PromptPackage`) and after Module 8 (soft dependency, for full-context reasoning) — i.e. immediately before the existing Module 10 block.

**Call added to `main.py::_run_pipeline_creators`, between the Module 6 block and the Module 10 block:**

- Gated by the config flag that **already exists**: `DECISION_ENGINE_ENABLED`. No new flag needed here — it was reserved and never wired up.
- Calls the existing public API `run_decision_engine(video_id, decision_dir=decision_dir, analysis_dir=analysis_dir, redesign_spec_dir=redesign_spec_dir, prompt_package_dir=prompt_package_dir, asset_extraction_dir=asset_extraction_dir)`.
- On failure (its `MissingArtifactError` / `ArtifactValidationError` / decision-engine-specific error hierarchy), logs and — **must abort this creator**, unlike Module 8: Module 9's own `load_input_bundle` treats Module 4/5/6 artifacts as hard requirements and will raise if they're missing, which by pipeline position they never will be (they were just produced). A raised error here signals a real defect (e.g. corrupted JSON), not a normal degrade case, so it should be treated the same severity as an existing Module 5/6 failure: `skipped += 1; continue`.
- **New parameter:** `decision_dir: Path = DEFAULT_DECISION_DIR` added to the pipeline signature, same pattern as above.

**Import-path fix required before this wiring is safe:** Module 9 (and Module 8) use `from modules.config import ...` / `from modules.decision_components... import ...`. This works today only because `python main.py`, run from the project root, implicitly makes the project root `sys.path[0]`. This is exactly the mechanism `pytest.ini`'s `pythonpath = modules .` also relies on for tests. It is fragile, not broken — but before wiring these two modules into `main.py`'s call graph, the safest fix (zero behavior change, matches existing convention) is to make `main.py`'s own `sys.path` setup insert the **project root** in addition to `_MODULES_DIR`, exactly mirroring what `pytest.ini` already does. This is a one-line addition, not a refactor of Module 8/9's own import style (which stays as-is — not "fixing" a working component, just making the entry point's path setup match the test suite's).

---

## 5. Interface Change — Module 10 Must Accept Module 9's Decisions Before It Bakes Masks

This is the one genuine architectural gap in the whole system, and it's narrower than it first appears.

**The problem:** Module 10.5's `PrecedenceResolver` already gives Module 9's `DecisionManifest` precedence over Module 10's fallback decisions — but only when deriving the generation **strategy** (`FaceStrategy`, `BackgroundStrategy`, `preserve_objects`, lighting). By the time Module 10.5 runs, Module 10 has *already* built the `GenerationBundle` — the actual masks, crops, and layer placements — using only its own internal `DecisionResolver`, which never sees Module 9's output. So today, even with Module 9 fully wired per §4, a case where Module 9's LLM-assisted reasoning disagrees with Module 10's simple `RedesignSpecification`-only rule mapping (e.g. Module 9 decides `REPLACE` for an object Module 10's naive resolver decided to `KEEP`) produces an inconsistency: Module 10.5's derived strategy will correctly reflect Module 9's REPLACE decision, but the physical mask/crop for that object — already baked into the workspace by Module 10 — will still be the KEEP-shaped asset, because nothing told Module 10 to build it differently.

**The fix — additive, optional parameter, zero breaking change:**

`AssetComposer.prepare_generation_workspace(video_id, decision_manifest: Optional[DecisionManifest] = None)` — new optional keyword parameter, default `None`, so every existing call site (and every existing test) is untouched.

- When `decision_manifest` is `None` (today's default, and the case for anyone with `DECISION_ENGINE_ENABLED=False`), `AssetComposer` behaves exactly as it does now — internal `DecisionResolver` maps `RedesignSpecification` directly.
- When `decision_manifest` is provided, `composition_components/decision_resolver.py::DecisionResolver.resolve()` is extended (additively — new branch, not a rewrite) to reuse the **exact same precedence rule already implemented and tested in Module 10.5's `PrecedenceResolver`**: prefer a matching `ResolvedDecision` from the manifest by `target.element_id`/`element_type` where `decision_manifest.status != "error"`, else fall back to today's `RedesignSpecification`-only mapping per element. This is literally copying an already-proven precedence rule one layer earlier in the pipeline, not inventing a new one.
- `main.py`'s new Module 10 call site (§3/§4 above) passes `decision_manifest=decision_manifest if DECISION_ENGINE_ENABLED else None`, where `decision_manifest` is the object returned by the new Module 9 call.

With this one change, Module 10.5's existing `PrecedenceResolver` becomes a **consistency check**, not the sole enforcement point — it will always agree with what Module 10 already built, because both now consult the same manifest with the same precedence rule. If it ever disagrees, that's a signal of a genuine bug, not an expected steady state.

---

## 6. End-to-End Data Flow (Post-Integration)

```
Module 3 (thumbnail.jpg)
      │
      ▼
Module 4 — Thumbnail Intelligence
  reads:  thumbnail.jpg, VideoMetadata
  writes: data/analysis/{video_id}.json          (ThumbnailIntelligence)
      │
      ├─────────────────────────────────────────────────────────┐
      ▼                                                          ▼
Module 5 — Redesign Specification                    Module 8 — Asset Extraction Engine  [NEW CALL]
  reads:  ThumbnailIntelligence                         reads:  thumbnail.jpg, ThumbnailIntelligence
  writes: data/redesign_specs/{video_id}.json            writes: data/asset_extraction/{video_id}/asset_manifest.json
          (RedesignSpecification)                                (AssetExtractionManifest: people[], scene,
      │                                                            objects[], typography[], visual_properties,
      ▼                                                            composition, effects)
Module 5.5 — Design Blueprint                                    │
  reads:  ThumbnailIntelligence, RedesignSpecification,           │
          VideoMetadata                                           │
  writes: data/design_blueprints/{video_id}.json                  │
          (DesignBlueprint)                                       │
      │                                                            │
      ▼                                                            │
Module 6 — Prompt Compiler                                        │
  reads:  RedesignSpecification, DesignBlueprint                  │
  writes: data/prompt_packages/{video_id}.json                    │
          (PromptPackage)                                         │
      │                                                            │
      └──────────────────────┬─────────────────────────────────────┘
                              ▼
                Module 9 — AI Decision Engine  [NEW CALL]
                  reads:  ThumbnailIntelligence (M4), RedesignSpecification (M5),
                          PromptPackage (M6), AssetExtractionManifest (M8, optional)
                  writes: data/decisions/{video_id}/decision_manifest.json
                          (DecisionManifest: ResolvedDecision[] — KEEP/REMOVE/
                           REPLACE/ENHANCE/ADD per element, rule+LLM sourced,
                           conflict-resolved)
                              │
                              ▼
                Module 10 — Asset Composer  [EXTENDED CALL]
                  reads:  RedesignSpecification (M5), DecisionManifest (M9, optional,
                          NEW PARAMETER — see §5), and internally invokes
                          Module 6.5 (VisualReferenceEngine — now optionally
                          projecting from AssetExtractionManifest per §2)
                  writes: data/composition_workspaces/{video_id}/  (GenerationBundle,
                          CompositionWorkspace: masks, crops, layers, layer placement)
                              │
                              ▼
                Module 10.5 — Thumbnail Planner  (flag: THUMBNAIL_PLANNER_ENABLED,
                                                    unchanged wiring)
                  reads:  CompositionWorkspace (M10), AssetExtractionManifest (M8,
                          optional), DecisionManifest (M9, optional — now consistent
                          with M10's own resolution per §5), PromptPackage (M6),
                          ThumbnailIntelligence (M4), RedesignSpecification (M5)
                  writes: data/strategy_packs/{video_id}.json  (GenerationPlan:
                          FaceStrategy, BackgroundStrategy, PlanConditioningAsset[])
                              │
                              ▼
                Module 7 — Image Generation  (unchanged wiring)
                  reads:  PromptPackage, GenerationBundle, GenerationPlan,
                          DesignBlueprint
                  writes: data/generated_thumbnails/{video_id}/{video_id}.png
```

**Artifact canonical-ownership summary:**

| Artifact | Written by | Consumed by |
|---|---|---|
| `ThumbnailIntelligence` | Module 4 | 5, 5.5, 8, 9 |
| `RedesignSpecification` | Module 5 | 5.5, 6, 9, 10 |
| `DesignBlueprint` | Module 5.5 | 6, 7 |
| `PromptPackage` | Module 6 | 9, 10.5, 7 |
| `AssetExtractionManifest` | **Module 8 (canonical, per §2)** | 6.5 (projection), 9, 10.5 |
| `VisualReferenceManifest` | Module 6.5 (now projects from Module 8 when present) | 10 |
| `DecisionManifest` | Module 9 | **10 (new, §5)**, 10.5 |
| `GenerationBundle` / `CompositionWorkspace` | Module 10 | 10.5, 7 |
| `GenerationPlan` | Module 10.5 | 7 |
| Generated thumbnail | Module 7 | — |

---

## 7. ComfyUI Conditioning Gap Analysis (Objective 6)

Verified against `workflows/fragments/` and `image_generator.py::_select_fragments`:

| Conditioning type | Fragment exists? | Wired into fragment selection? | Gap |
|---|---|---|---|
| ControlNet — depth | ✅ `controlnet_depth.json` | ✅ (`profile.controlnet_enabled and conditioning.depth_path`) | None |
| ControlNet — canny | ✅ `controlnet_canny.json` | ✅ | None |
| ControlNet — segmentation | ✅ `controlnet_segmentation.json` | ✅ | None |
| IPAdapter (single reference) | ✅ `ipadapter_reference.json` | ✅ (`profile.ipadapter_enabled and conditioning.ip_adapter_reference_paths`) | None |
| Multi-object reference | ✅ `multi_object_reference.json` | ✅ (gated on `len(role_image_paths) > 1` and `object_` role keys) | None |
| Regional prompting / masking | ✅ `regional_mask_conditioning.json` | Present but not directly confirmed in `_select_fragments` snippet above — verify at implementation time | Low — likely wired, needs direct confirmation |
| Text-region exclusion mask | ✅ `text_exclusion_mask.json` | Present, same caveat | Low |
| **Inpainting** (localized edit of REPLACE/ENHANCE regions while leaving the rest of the source pixels untouched) | ❌ **No fragment exists** | N/A | **Real gap.** This is the mechanism that would let Module 9's `REPLACE`/`ENHANCE` decisions on specific elements (a logo, a product, a hand) actually edit just that region instead of the current approach of regenerating the whole background via ControlNet-guided txt2img. |
| **Outpainting** | ❌ **No fragment exists** | N/A | Real gap, lower priority — nothing in Modules 5/9/10.5's taxonomy currently calls for canvas extension. |
| ControlNet on `PROFILE_PREMIUM` (Flux checkpoint) | N/A | `controlnet_enabled=False` for this profile | **Not a bug** — Flux's ControlNet integration path differs from SDXL's and the profile was deliberately configured without it. Flagging so it isn't mistaken for an oversight during migration; identity preservation on `PROFILE_PREMIUM` currently relies on IPAdapter alone. |

**Recommendation:** adding an `inpainting` fragment is the highest-value remaining gap for identity preservation specifically, because it's the only mechanism that would let a `REPLACE`/`ENHANCE` decision on one element (say, a low-quality logo) be executed without regenerating the entire frame around a preserved face/subject. This is flagged as a candidate for a **future, separate** Module 7 extension document — out of scope for this integration architecture, which is explicitly about wiring existing pieces together, not adding new generation capability.

---

## 8. Migration Plan

Each phase is independently shippable, additive, and defaults to a no-op if its flag is off. No existing test should need to change.

| Phase | Change | Flag | Default | Breaking? |
|---|---|---|---|---|
| **0** | Add `ASSET_EXTRACTION_ENABLED: bool = False` to `config.py`, alongside existing `DECISION_ENGINE_ENABLED` | new | `False` | No |
| **0** | `main.py`: add project root to `sys.path` alongside `_MODULES_DIR`, matching `pytest.ini`'s `pythonpath = modules .` | — | — | No (additive path entry) |
| **1** | Wire Module 8 call into `_run_pipeline_creators`, right after Module 4, gated by `ASSET_EXTRACTION_ENABLED`, non-fatal on failure | `ASSET_EXTRACTION_ENABLED` | off | No |
| **2** | Wire Module 9 call into `_run_pipeline_creators`, right after Module 6 / before Module 10, gated by `DECISION_ENGINE_ENABLED` (already exists), fatal-per-creator on failure (matches Module 5/6 severity) | `DECISION_ENGINE_ENABLED` | off | No |
| **3** | Add optional `decision_manifest` parameter to `AssetComposer.prepare_generation_workspace`; extend `DecisionResolver.resolve()` with a manifest-precedence branch (reusing Module 10.5's existing precedence rule); `main.py` passes the Module 9 result through | — | inert when `None` | No |
| **4** | Convert `VisualReferenceEngine` to optionally project from an `AssetExtractionManifest` when present for the `video_id`, falling back to direct CV extraction otherwise; gated internally by presence-check, no new top-level flag needed since it degrades automatically | — | inert when no Module 8 manifest exists | No |
| **5** (future, separate doc) | Add inpainting fragment + `WorkflowBuilder` selection rule for element-level `REPLACE`/`ENHANCE` | new | off | No |

**Rollout order matters:** Phase 3 depends on Phase 2 (needs a `DecisionManifest` to pass); Phase 4 depends on Phase 1 (needs an `AssetExtractionManifest` to project from). Phases 0–2 can ship together as one PR; Phase 3 and Phase 4 are independent of each other and can ship separately once 0–2 are live.

---

## 9. Risks

| Risk | Detail | Mitigation |
|---|---|---|
| **Redundant GPU inference** | Until Phase 4 ships, enabling both `ASSET_EXTRACTION_ENABLED` and the existing VRE call inside Module 10 means InsightFace/YOLO run twice per creator (once for Module 8, once for Module 6.5) | Documented as a known, temporary cost of Phase 1–3; Phase 4 removes it. Operators should not enable `ASSET_EXTRACTION_ENABLED=True` in production until Phase 4 ships, or should accept the extra ~1 face-detection + 1 object-detection pass per creator in the interim. |
| **Model-loading cost on RTX 4060 / 16GB** | Module 8's docstring already notes a "single-model GPU lock" for its own multi-family sequential extraction, but that lock does not span across Module 4 / Module 6.5 / Module 8 — three separate call sites can each independently load/unload the same underlying weights (InsightFace, YOLO) | Out of scope to fix in this document (would touch `ModelBridge` internals across three modules); flagged for a future "shared model registry" investigation. Not a blocker for wiring per §3/§4, since Modules 8/9/10 already run sequentially per creator, not concurrently. |
| **Import-path fragility** | Module 8/9's package-qualified imports (`modules.config`) only resolve today via `sys.path[0]` being the script directory when run as `python main.py` from repo root | Fixed by the one-line Phase 0 `sys.path` addition; verify Codex adds the project root **before** `_MODULES_DIR` is inserted a second time redundantly, or dedupes, to avoid shadowing issues if `modules/config.py` and a hypothetical root-level `config.py` both existed (they don't today — checked). |
| **Module 9 failure severity choice** | Treating Module 9 failures as creator-fatal (§4) means a corrupted or missing upstream artifact now blocks generation for that creator when `DECISION_ENGINE_ENABLED=True`, where previously (flag off) the creator would proceed | This is the same severity Module 5/6 already have today — consistent with treating Module 9 as a required stage once opted in, not a soft enhancement. If Afsar wants non-fatal behavior instead, change `continue` to a fallback of `decision_manifest=None` passed to Module 10 — a one-line alternative, called out here so the choice is explicit rather than assumed. |
| **Ollama dependency growth** | Module 9 has its own `MODULE9_OLLAMA_MODEL = "qwen3:8b"` LLM reasoning stage, separate from Module 4's Ollama call — enabling `DECISION_ENGINE_ENABLED` adds a second local LLM call per creator | Already accounted for in Module 9's own design (timeout/retry config exists); flagged here only as an added per-creator latency cost once enabled, not a new risk this document introduces. |
| **`DecisionManifest` / Module 10 disagreement during rollout** | Between Phase 2 and Phase 3 shipping, Module 9 will be producing decisions that Module 10 still ignores | Ship Phase 2 and Phase 3 in the same PR/release to avoid a window where this inconsistency is observable in production. |

---

## 10. Backward Compatibility Statement

- No existing public function signature loses a parameter or changes a default.
- The only signature change is one new **optional, defaulted** parameter on `AssetComposer.prepare_generation_workspace` (§5) — every existing call site compiles and behaves identically without modification.
- No existing JSON schema (`RedesignSpecification`, `DesignBlueprint`, `PromptPackage`, `GenerationBundle`, `GenerationPlan`) changes shape.
- No existing test in `tests/` should require modification; new tests are additive (Module 8/9 wiring paths, the new `DecisionResolver` branch, the new `sys.path` entry).
- Both new pipeline stages are opt-in via flags that default to today's exact behavior (`False`/off).
- Module 6.5's public interface and output file paths are unchanged by §2's internal-source change.

---

## 11. Open Items for Afsar

1. **Module 9 failure severity** (Risks table) — confirm creator-fatal is the intended behavior, or specify the non-fatal alternative.
2. **Regional-mask / text-exclusion fragment wiring** (§7) — flagged as "likely wired, needs direct confirmation" rather than verified with full certainty; worth a quick direct check against `_select_fragments`'s full body before Codex implements, since this document's audit focused on the depth/canny/segmentation/IPAdapter branches shown in the grep.
3. **Shared model registry across Module 4 / 6.5 / 8** (Risks table) — explicitly out of scope here; confirm whether it should be a follow-on document.
4. **Inpainting fragment** (§7) — confirmed as the highest-value remaining identity-preservation gap; confirm whether this should be scoped as the next document after this one ships.
