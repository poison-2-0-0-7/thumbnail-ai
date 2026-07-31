# Module 10.5 — Thumbnail Planner & Full Conditioning Pipeline Architecture

**Design document — architecture only. No implementation code, no pseudocode.**
**Source of truth verified against:** `poison-2-0-0-7/thumbnail-ai` @ `main` (cloned and reviewed in full prior to writing this document — `main.py`, `modules/`, `modules/*_components/`, `modules/vision_stack/`, `tests/`, `docs/`, `workflows/`, `config.py`, `models.py`).
**Status of this document:** extension only. Nothing described below requires editing an existing file's *behavior*; every existing module, public API, test, and JSON artifact keeps working exactly as it does today, byte-for-byte, when the new pieces are absent or disabled.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Pipeline Analysis](#2-current-pipeline-analysis)
3. [Architectural Gap Analysis (Verified, Not Assumed)](#3-architectural-gap-analysis-verified-not-assumed)
4. [Design Goals](#4-design-goals)
5. [Generation Conditioning Pipeline](#5-generation-conditioning-pipeline)
6. [Thumbnail Planner Architecture](#6-thumbnail-planner-architecture)
7. [Generation Plan Data Model](#7-generation-plan-data-model)
8. [Module 7 Extension](#8-module-7-extension)
9. [Data Flow](#9-data-flow)
10. [Dependency Diagram](#10-dependency-diagram)
11. [Interfaces](#11-interfaces)
12. [Public APIs](#12-public-apis)
13. [Internal APIs](#13-internal-apis)
14. [Folder Structure](#14-folder-structure)
15. [Configuration Changes](#15-configuration-changes)
16. [Error Handling](#16-error-handling)
17. [Logging](#17-logging)
18. [Testing Strategy](#18-testing-strategy)
19. [Migration Strategy](#19-migration-strategy)
20. [Phase-by-Phase Implementation Plan](#20-phase-by-phase-implementation-plan)

---

## 1. Executive Summary

The originating brief for this document assumes that "large amounts of intelligence are extracted but are not guaranteed to influence the generated thumbnail," and asks for a new conditioning pipeline plus a Thumbnail Planner to fix that. A full repository read shows that premise is **half true, in a very specific and fixable way** — and the fix is much smaller than "build a conditioning pipeline from scratch."

Three independent things are true simultaneously in this codebase today:

1. **A real, working, tested conditioning pipeline already exists.** `GenerationConditioningContext` / `ConditioningAssetResolver` (`modules/generation_components/conditioning_asset_resolver.py`), the fragment-based `WorkflowBuilder` (`modules/image_generator.py`), and six ComfyUI graph fragments (`workflows/fragments/*.json` — depth, canny, segmentation, IP-Adapter, text-exclusion mask, regional mask) are fully implemented, wired into `ImageGeneratorPipeline.run()`, and exercised by `tests/test_generation_components/`. This is Module 7 Phase 3 work, already done.
2. **Two entire modules of rich intelligence — Module 8 (Asset Extraction Engine) and Module 9 (AI Decision Engine) — are fully implemented, fully tested, and never invoked anywhere.** Not in `main.py`, not in `composition_engine.py`, not in `evaluation/pipeline_runner.py`. `grep` for `AssetExtractionEngine` and `DecisionEngine` across every orchestration entry point returns zero hits. Module 10 (`composition_engine.py`) instead calls the much smaller Module 6.5 (Visual Reference Engine — one face, one generic background/foreground pair, a placeholder depth map, a Canny map) and has its own private, rule-only `DecisionResolver` (`composition_components/decision_resolver.py`) that duplicates a fraction of what Module 9 already does correctly, with conflict resolution, LLM-adjudicated ambiguity handling, and a validated reasoning trace, stripped out.
3. **There is one concrete, provable "dead data" bug inside the wiring that does exist.** `GenerationBundleBuilder.build_generation_bundle()` (`modules/composition_components/generation_bundle_builder.py`) never sets `GenerationBundle.canny_path` — the field exists, `ConditioningAssetResolver` is ready to consume it, the `controlnet_canny` fragment is ready to fire, but the one line that would populate it from `CompositionLayer` was never written, and `CompositionLayer` itself has no `canny_hint_path` field to carry it even if it were. Canny maps are computed by VRE and then discarded. Separately, `reference_image_paths` and `mask_paths` in `GenerationBundle` are `dict[role_value, path]`, keyed by `LayerRole` (`"object"`, `"person"`, `"background"`) rather than by `asset_id` — so if a `RedesignSpecification` ever preserves more than one object (which `ObjectDirective` lists explicitly support), every object after the first silently overwrites the previous one's path in the flattened bundle before Module 7 ever sees it.

So the correct scope for this document is **not** "invent a conditioning pipeline" — it is: (a) wire the two orphaned modules into the live pipeline behind a feature flag, preserving every existing code path as the default; (b) fix the two verified plumbing gaps in `GenerationBundleBuilder`/`CompositionLayer` so extracted conditioning data stops being silently dropped; and (c) introduce the Thumbnail Planner as the single deterministic, cached, versioned artifact that the brief asked for — not as a *replacement* for `GenerationConditioningContext` (which is correct and stays exactly as it is, doing path resolution) but as the missing **semantic decision layer** above it, because today decision semantics (what strategy, what emphasis, what to say in the headline zone) are scattered as implicit side effects across Module 5's heuristics, Module 6's prompt strings, and Module 10's placement geometry, with no single artifact anyone (a human, Module 9's reasoning trace, or an evaluator) can point to and say "this is why the thumbnail looks the way it does."

Everything below is designed so that a single boolean flag, `THUMBNAIL_PLANNER_ENABLED` (default `False`), governs whether any of this runs. With it off, `main.py`, every existing test, every existing JSON artifact schema, and every existing hash/cache key computation is untouched.

---

## 2. Current Pipeline Analysis

### 2.1 What `main.py` actually runs today (verified line-by-line)

```
Module 1  csv_reader.load_all_creators()
    ↓
Module 2  youtube_metadata.process_video()            → VideoMetadata
    ↓
Module 3  thumbnail_downloader.process_thumbnail()     → ThumbnailData
    ↓
Module 4  thumbnail_intelligence.analyze_thumbnail()    → ThumbnailIntelligence
    ↓
Module 5  redesign_spec_engine.build_redesign_specification() → RedesignSpecification
    ↓
Module 6  prompt_compiler.compile_prompt_package()      → PromptPackage
    ↓
Module 10 composition_engine.AssetComposer
             .prepare_generation_workspace(video_id)     → GenerationBundle
    ↓
Module 7  image_generator.run_image_generation_pipeline() → generated PNG + manifest
```

Modules 8 and 9 are **absent from this list**. `main.py` never imports `asset_extraction_engine` or `decision_engine`.

### 2.2 What each already-wired module actually contributes to generation

| Module | Produces | Consumed by | Verified reach into Module 7 |
|---|---|---|---|
| 4 — Thumbnail Intelligence | OCR text regions, face detail(s), detected objects, `ColorProfile`, `CompositionAnalysis`, Ollama (`qwen3:8b`) reasoning | Module 5 | Indirect only, via Module 5/6 text fields |
| 5 — Redesign Spec | `ColorDirection`, `SubjectTreatment`, `TextOverlaySpec` (placement-only — *"this model never contains new copy"*, verbatim docstring), `ObjectDirective[]`, `LayoutDirection` | Module 6, Module 10's private `DecisionResolver` | Yes, via prompt strings and layer decisions |
| 6 — Prompt Compiler | `PromptPackage` (positive/negative prompt strings, generation parameters, seed) | Module 7 `WorkflowBuilder._slots` | Yes, directly |
| 6.5 — VRE | One face crop + mask, one background, one foreground, one placeholder depth map, one Canny map, all under `data/visual_references/<video_id>/` | Module 10's `AssetRegistry` | Yes, via Module 10 |
| 10 — Composition (`AssetComposer`) | `CompositionWorkspace` → flattened `GenerationBundle` | Module 7 `ConditioningAssetResolver` | Yes, directly |
| 7 — Generation Integration | Final generated image + `ImageGenerationResult` manifest | End of pipeline | — |

### 2.3 What each already-built-but-unwired module *would* contribute

| Module | Produces (fully implemented today) | Currently reaches Module 7? |
|---|---|---|
| 8 — Asset Extraction Engine | `AssetExtractionManifest`: **multiple** `PersonAsset` (face, face mask, embedding, landmarks, body mask, pose keypoints, clothing/hair masks), `SceneAsset` (real background/foreground, depth map, segmentation map, sky/ground masks), **multiple** `ObjectAsset` (crop, mask, bbox, parent/child hierarchy — from SAM2), `TypographyAsset[]` (per-text-region crop, font/alignment/color estimate), `VisualPropertiesAsset` (extended palette, gradients, lighting direction, shadow/highlight regions, blur/focus map), `CompositionAsset` (eye-flow map, negative-space mask, visual-hierarchy overlay), `EffectsAsset` (glow/outline/shadow/motion-blur/particle flags) | **No** |
| 9 — AI Decision Engine | `DecisionManifest`: per-element `ResolvedDecision` (keep/remove/replace/enhance/add), each with confidence, `DecisionSource` (rule / LLM / rule-LLM-agreement / conflict-resolution), rationale, and a full `ReasoningTraceEntry` history — arbitrated across rule engine + `qwen3:8b`-backed ambiguity routing + conflict resolution + validation | **No** |

Module 9's own ingestion code (`decision_components/io.py::load_input_bundle`) is already written to accept an `asset_extraction_dir` and gracefully treat it as absent — so Module 9 was built *expecting* Module 8 to exist upstream of it, and Module 10 was never updated to sit downstream of either.

### 2.4 A repository-numbering note worth flagging to the team

`docs/MODULE8_ASSET_EXTRACTION_ENGINE_ARCHITECTURE.md`'s own pipeline diagram labels the ComfyUI generation stage "Module 11." The actually-implemented generation stage is, and remains, "Module 7" throughout the codebase (`module7_exceptions.py`, every `MODULE7_*` config constant, `MODULE7_PHASE2_*` / `MODULE7_PHASE3_*` doc filenames). This document uses the code's numbering (Module 7) throughout and does not attempt a renumbering; it is called out here only so nobody "fixes" imports based on the docs' numbering.

---

## 3. Architectural Gap Analysis (Verified, Not Assumed)

Ordered by how directly each one blocks "every extracted artifact has a downstream consumer":

### Gap A — Module 8 and Module 9 are fully built and completely disconnected
Confirmed by `grep -rn "AssetExtractionEngine\|DecisionEngine" main.py modules/composition_engine.py evaluation/pipeline_runner.py` → zero matches. This is the single largest source of "dead data": nearly everything Section headers "OCR becomes headline planning," "object crops become generation guidance," "creator faces become reference conditioning" etc. in the original brief already has a *model* and a *processor* built for it in Module 8 — it is simply never run.

### Gap B — Module 10 reimplements a strictly worse version of Module 9's job
`composition_components/decision_resolver.py` maps `RedesignSpecification` fields to `LayerDecision` with four hardcoded rules (background→REPLACE always, person→ENHANCE/KEEP, object action string match, text→ADD if any OCR text existed). It has no confidence, no LLM adjudication for ambiguous cases, no conflict resolution across competing signals, and no persisted rationale trail beyond a single free-text string. Module 9 already does all of this correctly and is sitting unused one file away.

### Gap C — `GenerationBundle.canny_path` is unreachable dead code (verified, not theoretical)
`GenerationBundleBuilder.build_generation_bundle()` initializes `canny_path: Optional[str] = None` and **never assigns it** — there is no code path that sets it, because `CompositionLayer` (the only structure the builder iterates) has a `depth_hint_path` field but no `canny_hint_path` field. VRE (`vre_components/topology_processor.py`) computes a real Canny edge map (`_apply_canny_edge_detection`, genuine OpenCV Canny, not a placeholder) every single run, and it is written to disk and then never referenced again. Meanwhile `ConditioningAssetResolver` is fully ready to consume `bundle.canny_path`, and `workflows/fragments/controlnet_canny.json` is fully ready to attach. The wire is cut in exactly one place.

### Gap D — Multi-object preservation collapses in the bundle flatten step
`GenerationBundle.reference_image_paths` and `mask_paths` are `dict[str, str]` keyed by `layer.placement.role.value` (e.g., `"object"`), not by `layer.placement.asset_id` (e.g., `"object_0_microphone"`, `"object_1_laptop"`). `ObjectDirective` already supports a list of independently-actioned objects, and `DecisionResolver.resolve()` already emits one `LayerRole.OBJECT` layer per directive with a unique `element_key` — but the moment those layers are flattened into a `GenerationBundle`, every object after the first overwrites the one before it in the dict, because they share the same role key. A creator with two preserved objects (e.g., a mic and a laptop) will only ever get one of them into Module 7 today, non-deterministically (whichever iterates last).

### Gap E — VRE's "depth map" is a placeholder, and the four real Module 8 vision-stack wrappers it would need to stop being one are unbuilt
`vre_components/topology_processor.py::_apply_monocular_depth` is a Gaussian-blur-then-normalize intensity heuristic — not a learned depth model. `modules/vision_stack/` declares `sam2`, `birefnet`, `bisenet`, `depth_anything`, and `teed` as stack members via YAML, and `GPUResourceManager` / `ModelRegistry` / `ModelLoader` boot-time plumbing exists for all of them, but the *only* implemented inference wrapper today is `grounding_dino.py`. Module 8's design doc (§2, finding 3) already documents this precisely and correctly scopes those five wrappers as vision-stack deliverables, not Module 8 internals. This document does not re-scope that work; it only notes that wiring Module 8 in (Gap A) does not by itself upgrade depth-map quality until those wrappers land — Module 8 will degrade those specific families to `AssetExtractionStatus.PARTIAL` exactly as it's designed to, and the Thumbnail Planner (§6) must read that status rather than assume success.

### Gap F — No single deterministic artifact represents "what strategy is this thumbnail using"
Decision-relevant information is spread with no unifying record: `RedesignSpecification.subject_treatment.rationale` is a free string; `PromptPackage.subject_instructions` is another free string built from it; `CompositionWorkspace.layers[i].placement.rationale` is a third, independently-worded string for the same decision. None of them are structured, none of them are validated against each other, and there is no single object a human, Codex, or an evaluator (Module PVQEF) can read to get "headline text plan, face strategy, background strategy, camera distance, lighting, palette, negative constraints, conditioning asset list" as one deterministic, hashable, cacheable record. This is the literal gap the brief's `ThumbnailPlanner` example schema is describing, and it is real — it is just adjacent to, not a replacement for, the already-correct plumbing in `GenerationConditioningContext`.

### Gap G — OCR never becomes new headline copy anywhere, by explicit design
`TextOverlaySpec`'s own docstring states it "never contains new copy" — Module 5 only ever computes a placement zone for *existing* on-thumbnail text. No module today decides what a *redesigned* headline should say. The brief's "OCR becomes thumbnail headline planning" is aspirational, not a wiring bug — this is new scope, and is handled explicitly and conservatively in §6.4.

### Gap H — Transcript stops at Module 4
`transcript` fields exist only in `youtube_metadata.py`, `thumbnail_intelligence.py`, and `models.py`; nothing downstream of Module 4's Ollama reasoning stage re-reads the transcript. Any semantic enrichment from transcript content today is filtered entirely through whatever Module 4's `GeminiReasoning` free-text summary happened to capture, with no guarantee any of it survives into `PromptPackage`.

---

## 4. Design Goals

1. **Wire, don't rebuild.** Module 8 and Module 9 are correct, tested, and idiomatic to this codebase. The job is integration, not redesign.
2. **Every new consumer is additive and optional.** `main.py`'s existing call sequence keeps working verbatim when the new flag is off; every existing JSON schema on disk is unchanged; every existing test keeps passing unmodified.
3. **Fix the two proven plumbing bugs (Gaps C and D) independently of the flag**, because they are pure bugs, not scope changes — `canny_path` should populate and multi-object bundles should not collapse regardless of whether the Planner or Modules 8/9 are enabled. These are the only two changes in this document that touch already-shipped file *behavior* (not signatures), and both are additive fields / additional assignment lines, never removals.
4. **One deterministic, cached, hashable Generation Plan artifact**, matching every existing module's `build_X()` / `save_X()` / cache-by-hash shape (`redesign_spec_engine.py`, `prompt_compiler.py`, `thumbnail_intelligence.py` all follow this exact pattern; the Planner does too).
5. **Degrade the same way every other module already degrades.** `IntelligenceStatus.PARTIAL`, `AssetExtractionStatus.PARTIAL`, `DecisionManifestStatus.PARTIAL` all exist today with the same shape — a failed vision-stack family, a low-confidence decision, or a missing Module 8/9 artifact must produce a `PARTIAL` plan and safe fallbacks, never a hard pipeline stop.
6. **RTX 4060 / 16GB RAM budget is inherited, not re-litigated.** The Planner performs no model inference of its own (it is a pure aggregation/decision-formatting stage over already-computed artifacts); GPU budget concerns belong entirely to Module 8's existing sequential-GPU-lock design.
7. **No module is rewritten.** `composition_engine.py`, `image_generator.py`, `decision_engine.py`, `asset_extraction_engine.py` keep every existing public method signature. New behavior is reached exclusively through new optional constructor parameters (mirroring the existing `bundle_loader` / `workspace_loader` / `conditioning_resolver` optional-injection pattern already used in `ImageGeneratorPipeline.__init__`) and new call sites in `main.py`, gated by config.

### Non-Goals
- Implementing the five missing vision-stack inference wrappers (`sam2.py`, `birefnet.py`, `bisenet.py`, `depth_anything.py`, `teed.py`) — out of scope here exactly as Module 8's own doc already scopes them out; the Planner is designed to work correctly whether they exist or not.
- Generating new headline *copy* via an LLM — §6.4 defines a conservative, deterministic-only headline plan (existing-text passthrough or a template-derived placeholder), not a generative writing feature. That is future scope, flagged explicitly as out of bounds here.
- Renumbering Module 7 to "Module 11" to match the Module 8 doc's diagram (§2.4) — a documentation inconsistency, not an architecture task.

---

## 5. Generation Conditioning Pipeline

This section documents the pipeline **as it should exist after this extension**, correcting Gaps C and D, and routing Module 8/9 output into the same `GenerationConditioningContext` seam that Module 7 already trusts. It deliberately does *not* redesign `ConditioningAssetResolver`, `WorkflowBuilder`, or the fragment library — those are correct today and are reused as-is.

### 5.1 Per-artifact conditioning mapping (grounded in what's actually implemented)

| Extracted artifact | Model that already holds it | New/fixed path to `GenerationConditioningContext` |
|---|---|---|
| Creator face(s) | `PersonAsset.face`, `.face_embedding` (Module 8) | `role_image_paths["person_<i>"]`, new `ip_adapter_reference_paths["face_<i>"]` when `IPAdapter` is enabled — both fields already exist on the context |
| Object crops (plural) | `ObjectAsset.crop`, `.mask` per `object_index` (Module 8) | `role_image_paths["object_<object_index>_<label>"]` and `role_mask_paths["object_<object_index>_<label>"]` — **keyed by `asset_id`, not role** (fixes Gap D) |
| Foreground extraction | `SceneAsset.foreground` (Module 8) or VRE foreground (fallback) | `role_image_paths["foreground"]` |
| Background extraction | `SceneAsset.background` (Module 8) or VRE background (fallback) | `role_image_paths["background"]` |
| Depth map | `SceneAsset.depth_map` (Module 8, real model when wrappers land) or VRE placeholder depth (fallback) | `depth_path` (already wired end-to-end; no change needed once Module 8 is the source) |
| Canny map | VRE Canny (already computed today, currently discarded) or `SceneAsset` derivative | `canny_path` — **fixed to actually populate** (Gap C fix, §5.2) |
| Segmentation map | `SceneAsset.segmentation_map` (Module 8) | `segmentation_path` (field already exists on `GenerationConditioningContext`, already unused because nothing populates `bundle.segmentation_path` today either — same class of bug as Gap C, same fix pattern) |
| OCR text regions | `TypographyAsset[]` (Module 8) / `OCRResult` (Module 4) | `text_exclusion_mask_path` (already wired) plus new plan field `headline_placement_zone` (§7) |
| Transcript | `VideoMetadata.transcript` (Module 2) | New Planner input only (§6.4) — never a pixel-conditioning asset |
| Thumbnail intelligence (CTR/composition scores) | `ThumbnailIntelligence.composition`, `.color_profile` | New Planner input, drives `composition_strategy` / `color_palette` plan fields (§7) — not new pixel conditioning, semantic weighting only |
| Redesign specification | `RedesignSpecification` (Module 5, unchanged) | Continues to drive `PromptPackage` exactly as today; Planner reads it too, for cross-validation only |
| Composition analysis | `CompositionAsset.eye_flow_map`, `.negative_space_mask` (Module 8) | New optional `role_image_paths["composition_hint"]`, consumed only if a future workflow fragment declares it — not force-fit into an existing fragment |
| Color analysis | `VisualPropertiesAsset` (Module 8) / `ColorProfile` (Module 4) | Planner `color_palette` field only (semantic, not pixel-conditioning) |

### 5.2 Fixing Gap C and Gap D — the only two behavior-affecting changes in this document

Both are additive-field changes to existing frozen Pydantic models plus one new assignment line each in `GenerationBundleBuilder`; neither changes any existing field's meaning or removes anything.

- **`CompositionLayer` gains an optional `canny_hint_path: Optional[str] = None`**, populated by Module 10's layer-building loop the same way `depth_hint_path` already is (via `self._asset_registry.resolve("canny_map")`, a lookup key VRE's manifest already supports since it writes a Canny asset today — it is simply never `resolve()`d). `GenerationBundleBuilder` then sets `canny_path` from the first layer carrying a `canny_hint_path`, mirroring the existing `depth_path` logic exactly.
- **`GenerationBundle.reference_image_paths` and `.mask_paths` change their dict key from `role.value` to `layer.placement.asset_id`** inside `GenerationBundleBuilder` only (the model's type, `dict[str, str]`, does not change — only which string is used as the key). `ConditioningAssetResolver.role_image_paths` / `role_mask_paths` then naturally carry one entry per preserved object instead of collapsing to one. `WorkflowBuilder._slots()`'s two single-object placeholder lookups (`foreground_image_path`, `background_image_path`, `person_mask_path`, `object_mask_path`) are preserved unchanged for backward compatibility (they read fixed keys `"foreground"`, `"background"`, `"person"`, `"object"` which continue to exist whenever a *single* object is preserved, matching every currently-passing test); a new fragment, `multi_object_reference` (§5.3), is added for the ≥2-object case rather than overloading the four legacy slots.

Both fixes are backward compatible: any workspace with exactly one preserved object of a role produces identical `GenerationBundle` output before and after (the asset_id-keyed dict has exactly one entry, same value, just a different — but never previously observed by any test — dict key), and any workspace with zero Canny data behaves exactly as it does today (`canny_path` stays `None` if VRE isn't run or the registry lookup misses).

### 5.3 One new fragment: `multi_object_reference.json`

Follows the exact schema every existing fragment already uses (`_attach.point`, `_attach.output_node`, `graph`) — a small IP-Adapter/reference-latent chain that accepts N image inputs instead of one. `WorkflowBuilder._select_fragments()` gains one new condition: `if profile.ipadapter_enabled and len(conditioning.role_image_paths) > 1 and any(k.startswith("object_") for k in conditioning.role_image_paths)`. This is additive to the existing `if`/`elif` chain in that method — no existing condition changes.

---

## 6. Thumbnail Planner Architecture

### 6.1 Position in the pipeline

Per the brief, the Planner is a layer that turns upstream analysis into deterministic generation decisions. Given what Module 10 and Module 9 actually produce today, the correct seam is **after Module 10 has resolved geometry (it needs canvas size, placement zones, and layer decisions to reason about layout) and before Module 7 consumes the result** — not strictly "between Prompt Compiler and Generation Integration" as originally phrased, because that gap is currently occupied by Composition, whose geometric output the Planner needs as an input. Concretely:

```
Module 6  Prompt Compiler         → PromptPackage
Module 8  Asset Extraction Engine → AssetExtractionManifest   (newly wired)
Module 9  AI Decision Engine      → DecisionManifest          (newly wired, consumes Module 8's manifest)
Module 10 Composition (AssetComposer) → CompositionWorkspace → GenerationBundle
                                                     │
                                                     ▼
                        Module 10.5 — Thumbnail Planner
              (PromptPackage + AssetExtractionManifest + DecisionManifest
               + CompositionWorkspace + ThumbnailIntelligence → GenerationPlan)
                                                     │
                                                     ▼
                             Module 7 — Generation Integration
              (GenerationPlan folds into GenerationConditioningContext
               as additional, optional slot values — no existing slot removed)
```

This mirrors the existing "X.5" naming precedent set by Module 6.5 sitting between Module 6 and Module 7.

### 6.2 What the Planner does and does not do

**Does:**
- Reads the (by now materialized) outputs of Modules 4, 5, 6, 8, 9, and 10 for one `video_id`.
- Resolves any remaining disagreement between them using a **fixed, documented precedence order** (§6.3) — it performs no new inference and calls no model; it is a pure aggregation/normalization stage, matching Module 6's own "deterministic-first" philosophy.
- Emits one `GenerationPlan` (§7), cached and hashed exactly like every other module's output.
- Degrades to `PLANNER_STATUS.PARTIAL` when an upstream artifact is missing (e.g., Module 8/9 disabled) and falls back to Module 10/legacy fields for that portion of the plan — never blocks the pipeline.

**Does not:**
- Perform CV, embeddings, or any GPU work (that's Modules 4/8's job, already done by the time the Planner runs).
- Decide keep/remove/replace (that's Module 9's job; the Planner reads `DecisionManifest`, it doesn't recompute it).
- Generate new headline copy via an LLM (§6.4, explicitly conservative/deterministic only in this scope).
- Replace `GenerationConditioningContext` (that stays as the pixel/path-resolution layer; the Planner is the semantic-decision layer one level above it).

### 6.3 Precedence order when signals disagree

Because Module 9 may be disabled (feature flag off) while Module 5's simpler heuristics always run, the Planner needs one documented tie-breaking rule per field family, so its output is deterministic regardless of which upstream modules happen to be enabled for a given run:

1. **Decision semantics (keep/remove/replace/enhance/add per element):** `DecisionManifest.decisions` (Module 9) if present and `status != ERROR`, else `CompositionWorkspace.layers[*].placement.decision` (Module 10's existing resolver) as the fallback. This is a strict override, not a merge — Module 9, when present, is trusted over Module 10's simpler rule set because it already incorporates Module 10's same input plus LLM adjudication and conflict resolution.
2. **Asset provenance (which pixel file to point at for a role):** `AssetExtractionManifest` (Module 8) if present and the specific family's `extraction_status != "skipped"`, else the VRE-derived path already present in `GenerationBundle`. Family-by-family, not all-or-nothing — e.g., Module 8 people data can be used while scene data falls back to VRE if only the scene family degraded.
3. **Prompt text (positive/negative/subject/background instructions):** always `PromptPackage` (Module 6) verbatim, unchanged. The Planner never edits prompt strings — it only adds structured fields alongside them.
4. **Layout/geometry (canvas, focal zone, safe margins):** always `CompositionWorkspace` (Module 10), unchanged — the Planner has no independent geometry engine.

### 6.4 Headline planning (Gap G), handled conservatively

Given `TextOverlaySpec`'s explicit "never contains new copy" contract, the Planner's `headline` field is populated by exactly one of, in order:
1. The existing OCR text (`TypographyAsset.text` from Module 8, or `OCRResult.text_regions` from Module 4) verbatim, if `RedesignSpecification.text_overlay.include_text` is true and Module 5 decided to preserve existing text — this is a passthrough, not generation.
2. An empty string with `headline_source = "none"`, if no existing text is being preserved.

Generating *new* headline copy from title/transcript is explicitly deferred as a documented future extension point (a `headline_source = "generated"` value is reserved in the enum for that future work) and is **not implemented by this design** — doing so correctly needs its own LLM-adjudication and copy-length/legibility validation loop, analogous to Module 9's, and mixing that scope into this document would violate "do not redesign existing modules" by implicitly demanding new Module 5 behavior it doesn't have today.

### 6.5 Strategy fields derived from already-computed intelligence (no new inference)

- `face_strategy`: derived directly from `DecisionManifest`/`CompositionWorkspace` person-layer decision (`keep` → `"preserve_as_is"`, `enhance` → `"enhance_existing"`, absent → `"none"`) plus `PersonAsset.face_embedding is not None` → `"identity_locked"` suffix, reusing Module 7's own existing `IdentityPreservationStage` semantics rather than inventing new ones.
- `background_strategy`: derived from the background layer's `LayerDecision` (always `REPLACE` under today's `DecisionResolver` rule; may vary once Module 9 is live) plus whether `depth_path`/`segmentation_path` are available, to record `"structure_guided_replace"` vs `"unguided_replace"`.
- `composition_strategy` / `camera_distance` / `lighting` / `color_palette`: read straight from `ThumbnailIntelligence.composition`, `.face_analysis` (for camera distance, from face bbox size), `RedesignSpecification.color_direction`, and `VisualPropertiesAsset.lighting_direction` (Module 8) when present. All are string-enum summaries of numbers that already exist somewhere in the pipeline — no new measurement is introduced.
- `negative_constraints`: unions `PromptPackage.rendering_constraints`, `.safety_constraints`, and — new — any Module 9 `remove`-decision targets restated as explicit negative constraints (e.g., a removed watermark becomes `"no watermark"`), so a decision to remove something is guaranteed to also suppress its regeneration, which is not guaranteed today (Module 10's `DecisionResolver` marks a layer `REMOVE` but nothing propagates that into `PromptPackage.negative_prompt`, itself compiled *before* Module 10 ever runs, since Module 6 precedes Module 10 in the pipeline — this is Gap F made concrete: removal decisions currently cannot reach the negative prompt at all because of pipeline ordering, and the Planner is the first point in the pipeline positioned after both prompt compilation and decision resolution where that union is even possible).
- `conditioning_assets`: a flat, ordered list of every `role_image_paths`/`role_mask_paths`/`depth_path`/`canny_path`/`segmentation_path`/`ip_adapter_reference_paths` entry the resolved `GenerationConditioningContext` will carry — i.e., the plan's own manifest of what pixel data is about to influence the image, which is exactly the auditability the brief is asking for ("every extracted artifact has a clearly defined downstream consumer" becomes a field you can literally print and check).

---

## 7. Generation Plan Data Model

New frozen Pydantic models in `models.py`, appended after the existing `GenerationBundle` class (§ "Module 10.5 — Thumbnail Planner Models," following the same section-comment convention already used for the Module 8/9 sections). No existing model is modified.

- **`HeadlineSource`** (`str, Enum`): `PRESERVED_OCR`, `NONE`, `GENERATED` (reserved, unused by this design per §6.4).
- **`FaceStrategy`** (`str, Enum`): `NONE`, `PRESERVE_AS_IS`, `ENHANCE_EXISTING`, `PRESERVE_AS_IS_IDENTITY_LOCKED`, `ENHANCE_EXISTING_IDENTITY_LOCKED`.
- **`BackgroundStrategy`** (`str, Enum`): `STRUCTURE_GUIDED_REPLACE`, `UNGUIDED_REPLACE`, `KEEP`.
- **`PlanConditioningAsset`** (frozen `BaseModel`): `role: str`, `asset_id: str`, `path: str`, `kind: Literal["reference_image","mask","depth","canny","segmentation","ip_adapter_reference","text_exclusion_mask"]`, `source_module: Literal["module8","vre","module10"]` — the provenance tag that makes §6.3's precedence auditable after the fact.
- **`GenerationPlan`** (frozen `BaseModel`, top-level artifact):
  - `video_id: str`
  - `headline: str`, `headline_source: HeadlineSource`, `headline_placement_zone: Optional[BoundingBox]`
  - `face_strategy: FaceStrategy`
  - `background_strategy: BackgroundStrategy`
  - `preserve_objects: list[str]` (asset_ids, resolving Gap D's ambiguity into an explicit list)
  - `composition_strategy: str`, `camera_distance: str`, `lighting: str`, `color_palette: list[str]`
  - `negative_constraints: list[str]`
  - `conditioning_assets: list[PlanConditioningAsset]`
  - `decision_manifest_hash: Optional[str]` (sha256 of the Module 9 `DecisionManifest` used, `None` if Module 9 was unavailable and Module 10's fallback resolver was used instead — the plan is honest about its own provenance)
  - `asset_extraction_manifest_hash: Optional[str]` (same pattern for Module 8)
  - `prompt_package_hash: str`, `workspace_hash: str` (ties the plan to the exact upstream artifacts it was built from, matching `WorkspaceMetadata`'s existing hash-chaining pattern)
  - `status: Literal["success","partial","error"] = "success"`
  - `partial_failure_reasons: list[str] = []`
  - `engine_version: str`
  - `generated_at: str`

All hash fields validate as SHA-256 hex digests using the exact same `field_validator` pattern already copy-pasted across `WorkspaceMetadata`, `GenerationBundle`, and `AssetExtractionManifest` — this document does not introduce a new hashing convention.

---

## 8. Module 7 Extension

Module 7's actual consumption surface (`ConditioningAssetResolver`, `WorkflowBuilder`) is not rewritten. `ImageGeneratorPipeline.run()` gains one new optional parameter, `generation_plan: GenerationPlan | None = None`, following the exact precedent already set by its existing `generation_bundle` and `composition_workspace` optional parameters. When provided:

- `ConditioningAssetResolver.resolve()` gains an optional `plan: GenerationPlan | None = None` parameter; when present, it cross-checks that every `PlanConditioningAsset` the plan lists still resolves to an existing file (reusing its existing `_verify_file_exists` helper) and uses the plan's asset-id-keyed list as the authoritative source for `role_image_paths`/`role_mask_paths` (implementing the Gap D fix's final consumer-side step), falling back to `bundle`/`workspace`-derived values field-by-field exactly as it does today when `plan` is `None`.
- `WorkflowBuilder._slots()` gains a handful of new plan-derived slot values (`headline_text`, `headline_zone_x`/`y`/`w`/`h`, `negative_prompt` gets the plan's `negative_constraints` unioned in) purely as additional dict keys — every existing slot key, and every existing template that doesn't declare those new placeholders, is unaffected, because `_substitute()` only ever fills placeholders a template explicitly names.
- `run_image_generation_pipeline()` (the top-level helper `main.py` calls) gains a matching optional `generation_plan` parameter, defaulting to `None`, so `main.py`'s existing call site keeps compiling and behaving identically until it's explicitly updated to pass one (§19).

Every dependency Module 7 has on upstream data remains explicit and unchanged in kind, just larger in what it's allowed to be populated from:

| Module 7 dependency | Source before this doc | Source after this doc |
|---|---|---|
| Face references | VRE face crop only | Module 8 `PersonAsset[]` when available, VRE fallback |
| Object references | Single VRE `object_crop` | Module 8 `ObjectAsset[]`, asset-id-keyed (Gap D fix) |
| Depth | VRE placeholder | Module 8 `SceneAsset.depth_map` when real wrappers land; VRE fallback until then |
| Canny | *(nothing — Gap C)* | VRE Canny, now actually wired (Gap C fix) |
| Masks | VRE only | Module 8 per-family masks when available, VRE fallback |
| Generation Plan | *(did not exist)* | New, optional |
| Prompt Package | Module 6, unchanged | Module 6, unchanged |
| Workspace Assets | Module 10, unchanged | Module 10, unchanged |
| Workflow Template | `WorkflowLibrary`, unchanged | `WorkflowLibrary`, unchanged |

---

## 9. Data Flow

```
CSV → Metadata → Thumbnail Download
                        │
                        ▼
              Thumbnail Intelligence (Module 4)
             (OCR, faces, objects, color, composition, Ollama reasoning)
                        │
          ┌─────────────┼─────────────────────────┐
          ▼                                         ▼
  Redesign Specification (5)          Asset Extraction Engine (8)  [newly wired]
          │                                         │
          ▼                                         │
  Prompt Compiler (6) ──► PromptPackage              │
          │                                         │
          │                    ┌────────────────────┘
          │                    ▼
          │        AI Decision Engine (9)  [newly wired, consumes 4+5+6+8]
          │                    │
          └──────────┬─────────┘
                      ▼
        Composition / AssetComposer (10)
    (VRE for pixels it still owns + Module 8 assets when present +
     Module 9 decisions when present, else its own existing resolver)
                      │
                      ▼
           CompositionWorkspace → GenerationBundle
                      │
                      ▼
        Thumbnail Planner (10.5)  [new]
   PromptPackage + AssetExtractionManifest + DecisionManifest +
     CompositionWorkspace + ThumbnailIntelligence → GenerationPlan
                      │
                      ▼
      Generation Integration (7)
  ConditioningAssetResolver(bundle, workspace, plan) → GenerationConditioningContext
                      │
                      ▼
              WorkflowBuilder → ComfyUI → generated thumbnail
```

Every arrow that exists in the current pipeline (§2.1) is preserved unchanged; the new arrows are Module 8 → Module 9, Module 8/9 → Module 10 (as optional inputs, additive to what Module 10 already reads), and the new Module 10.5 box.

---

## 10. Dependency Diagram

```
config.py ──────────────────────────────────────────────────┐
   │  (THUMBNAIL_PLANNER_ENABLED, MODULE10_5_* constants)    │
   ▼                                                          │
models.py ── GenerationPlan, HeadlineSource, FaceStrategy,   │
              BackgroundStrategy, PlanConditioningAsset       │
   │                                                          │
   ▼                                                          ▼
thumbnail_planner.py (new orchestrator, modules/)   thumbnail_planner_exceptions.py (new, leaf module,
   │   imports:                                       zero project-internal deps, mirrors
   │   - models (GenerationPlan, DecisionManifest,     decision_exceptions.py / asset_extraction_exceptions.py)
   │     AssetExtractionManifest, CompositionWorkspace,
   │     PromptPackage, ThumbnailIntelligence)
   │   - planner_components/ (new subpackage, mirrors
   │     decision_components/ / asset_extraction_components/
   │     structure: precedence_resolver.py, headline_planner.py,
   │     strategy_deriver.py, conditioning_manifest_builder.py,
   │     interfaces.py, io.py)
   ▼
main.py (new call site, additive, after Module 10 / before Module 7)
   │
   ▼
image_generator.py (ImageGeneratorPipeline.run(generation_plan=...))
   │
   ▼
generation_components/conditioning_asset_resolver.py (extended resolve() signature, backward compatible)
```

`thumbnail_planner.py` depends on `decision_engine.py` and `asset_extraction_engine.py` only through their **already-public** `DecisionEngine.run()` / `AssetExtractionEngine.extract()` (or their cached-load helpers, matching the existing `load_cached_redesign_spec`-style pattern) — it never reaches into their internal components, exactly as every other orchestrator in this codebase only imports its peers' public entry points.

---

## 11. Interfaces

New interfaces, in `planner_components/interfaces.py`, following the existing `I<Name>` protocol convention used throughout `decision_components/interfaces.py` and `composition_components/interfaces.py`:

- `IPrecedenceResolver` — one method, resolving a field family per §6.3's documented order, given optional Module 8/9/10 artifacts.
- `IHeadlinePlanner` — one method implementing §6.4's conservative passthrough-or-empty logic.
- `IStrategyDeriver` — one method implementing §6.5's derivations.
- `IConditioningManifestBuilder` — one method producing the `list[PlanConditioningAsset]` from a resolved `GenerationConditioningContext`, so the Planner's own manifest and Module 7's actual runtime context are built from the same resolution logic and cannot drift apart.
- `IPlanCache` — mirrors `IDecisionCache` (`decision_components/interfaces.py`) exactly: `load(video_id) -> GenerationPlan | None`, `save(plan) -> Path`.

Every one of these takes and returns only the models defined in §7 plus the already-existing upstream models — no interface here reaches past the public models any other module already exposes.

---

## 12. Public APIs

`modules/thumbnail_planner.py`, mirroring `decision_engine.py`'s and `asset_extraction_engine.py`'s existing public shape exactly:

- `class ThumbnailPlanner` — orchestrator, constructor takes optional injected components (same DI pattern as `AssetComposer.__init__`, `DecisionEngine.__init__`).
- `ThumbnailPlanner.plan(video_id: str, *, force_recompute: bool = False) -> GenerationPlan` — the single public entry point, following `DecisionEngine.run()`'s exact signature shape.
- `build_generation_plan(video_id, ...) -> GenerationPlan` and `save_generation_plan(plan, plan_dir=...) -> Path` — free-function wrappers, matching the `build_X()`/`save_X()` shape used by every other module (`build_redesign_specification`/`save_redesign_spec`, `compile_prompt_package`/`save_prompt_package`) so `main.py`'s new call site reads identically to its five neighbors.
- `load_cached_generation_plan(video_id, plan_dir=...) -> GenerationPlan | None` — matching `load_cached_redesign_spec`'s existing shape.

`modules/thumbnail_planner_exceptions.py`:
- `ThumbnailPlannerError` (base), `UpstreamArtifactMissingError`, `PlanValidationError`, `PlanCacheError`, `PlanPersistError` — same naming and hierarchy depth as `decision_exceptions.py`.

---

## 13. Internal APIs

`planner_components/`:
- `precedence_resolver.py::PrecedenceResolver` — implements §6.3, pure functions over already-loaded artifacts, no I/O.
- `headline_planner.py::HeadlinePlanner` — implements §6.4.
- `strategy_deriver.py::StrategyDeriver` — implements §6.5.
- `conditioning_manifest_builder.py::ConditioningManifestBuilder` — wraps the existing `ConditioningAssetResolver` (imported, not duplicated) and converts its resolved `GenerationConditioningContext` into the `list[PlanConditioningAsset]` the plan persists.
- `io.py::load_planner_input_bundle(video_id, ...)` — mirrors `decision_components/io.py::load_input_bundle` exactly: best-effort loads of `ThumbnailIntelligence`, `RedesignSpecification`, `PromptPackage`, `AssetExtractionManifest` (optional), `DecisionManifest` (optional), `CompositionWorkspace`, returning `None` for any optional artifact that isn't on disk rather than raising, so a Planner run never hard-fails just because Module 8/9 are disabled.
- `io.py::PlanCache` — mirrors `decision_components/io.py::DecisionCache`'s existing atomic-write-plus-hash-verification cache pattern.

---

## 14. Folder Structure

```
modules/
  thumbnail_planner.py                  (new)
  thumbnail_planner_exceptions.py       (new)
  planner_components/                   (new)
    __init__.py
    interfaces.py
    precedence_resolver.py
    headline_planner.py
    strategy_deriver.py
    conditioning_manifest_builder.py
    io.py

data/
  generation_plans/                     (new, mirrors data/decisions/, data/asset_extraction/)
    <video_id>/
      generation_plan.json

tests/
  test_thumbnail_planner.py             (new, mirrors test_decision_engine.py)
  test_planner_components/              (new, mirrors tests/decision_components/)
    test_precedence_resolver.py
    test_headline_planner.py
    test_strategy_deriver.py
    test_conditioning_manifest_builder.py
    test_io.py

workflows/fragments/
  multi_object_reference.json           (new, §5.3)
```

No existing directory is renamed, moved, or restructured.

---

## 15. Configuration Changes

Appended to `config.py` in a new, clearly delimited section following every existing module's section-comment convention (`# --- Module N — Name ---`):

```
# ---------------------------------------------------------------------------
# Module 10.5 — Thumbnail Planner
# ---------------------------------------------------------------------------

THUMBNAIL_PLANNER_ENABLED: bool = False   # master flag; main.py checks this before calling Module 10.5

MODULE10_5_LOG_PATH: Path = LOG_DIR / "module10_5.log"
DEFAULT_GENERATION_PLAN_DIR: Path = PROJECT_ROOT / "data" / "generation_plans"
GENERATION_PLAN_FILENAME: str = "generation_plan.json"
PLANNER_ENGINE_VERSION: str = "1.0.0"
PLANNER_CACHE_ENABLED: bool = True

# Toggles for the two already-built-but-unwired modules; independent of the
# planner flag so Module 8/9 can be enabled for evaluation without the Planner,
# and vice versa is intentionally NOT allowed (Planner requires attempting 8/9
# load, even if both are absent on disk and it falls back per §6.3).
ASSET_EXTRACTION_ENGINE_ENABLED: bool = False
DECISION_ENGINE_ENABLED: bool = False

# §5.2 Gap C/D fixes — always active regardless of the flags above, since they
# are bug fixes, not new scope.
COMPOSITION_RESOLVE_CANNY_ASSET_KEY: str = "canny_map"   # matches VRE's existing manifest key
```

`main.py`'s existing `DEFAULT_*` imports are untouched; the new constants are additive imports only in the new call site.

---

## 16. Error Handling

The Planner's exception hierarchy (`thumbnail_planner_exceptions.py`, §12) follows the same **recoverable-by-default** philosophy already established by every other module in this codebase: `main.py`'s existing try/except-per-module-then-`continue` loop pattern (visible for every module from 2 through 7 today) gets one more matching block:

```
try:
    plan = ThumbnailPlanner().plan(video_id)
except ThumbnailPlannerError as exc:
    logger.error(...)
    skipped += 1
    continue
```

Internally:
- Missing Module 8/9 artifacts (flags off, or files absent) → `status="partial"`, `partial_failure_reasons` populated, plan still produced using §6.3 fallbacks. This never raises.
- A referenced conditioning asset that Module 8/9/10 claim exists but is actually missing on disk at plan-build time → `UpstreamArtifactMissingError`, matching `ConditioningResolutionError`'s existing behavior in Module 7 for the identical failure mode.
- A `DecisionManifest` or `AssetExtractionManifest` with `status == "error"` → treated as absent (fall back per §6.3), not as a hard failure, exactly as `PromptPackageLoader.load()` already treats `package.status == "error"` as a raise-worthy condition only for its *own* artifact, never for an *optional upstream* one — the Planner never raises just because an optional upstream module errored; it only raises if *no* usable source exists for a required field (e.g., no `PromptPackage` at all, which is already a hard Module 7 precondition today, unchanged).

---

## 17. Logging

One new log file, `logs/module10_5.log`, `_configure_logger()` called once at import time with `enqueue=True`, 10 MB rotation, 30-day retention — identical to every other module's logging setup (verified pattern in `visual_reference_engine.py`, `decision_engine.py`, `asset_extraction_engine.py`, all reviewed above). Key log lines mirror existing modules' info/warning/error split:
- `INFO` on plan cache hit/miss, on which precedence branch (§6.3) was taken per field family (so a human can grep "used Module 9 decision" vs "fell back to Module 10 resolver" per run).
- `WARNING` when an optional upstream artifact (Module 8 or 9) is absent and a fallback is used — this is the log line that makes Gap A's current silent bypass visible going forward, whether or not the flags are enabled.
- `ERROR` only for the hard-failure cases in §16.

---

## 14. Operational Staging & Rollout

The rollout of Module 10.5 follows a staged, zero-downtime procedure:

1. **Default State**:
   - `THUMBNAIL_PLANNER_ENABLED = False` in `config.py`.
   - The main pipeline continues running Modules 1–10 + Module 7 without invoking the Thumbnail Planner.
   - All tests pass with zero behavior change.

2. **Staged Activation**:
   - **Step A (Upstream Verification)**: Enable `ASSET_EXTRACTION_ENGINE_ENABLED = True` and `DECISION_ENGINE_ENABLED = True` in the target operational environment and verify Modules 8 and 9 produce valid manifests on real creator inputs.
   - **Step B (Planner Activation)**: Set `THUMBNAIL_PLANNER_ENABLED = True` in `config.py`. `main.py` will now invoke `ThumbnailPlanner().plan(video_id)` to produce a deterministic `GenerationPlan` artifact prior to Module 7 image generation.
   - **Step C (Monitoring & Profiling)**: Monitor `logs/module10_5.log` for precedence resolution decisions, degradation warnings, and cache hits. Record performance metrics (VRAM usage, generation latency) on the target rig.

---

## 18. Testing Strategy

Mirrors the existing per-module test structure exactly (`tests/test_decision_engine.py` + `tests/decision_components/`, `tests/test_asset_extraction_engine.py` + `tests/asset_extraction_components/`):

1. **Unit tests per `planner_components/` file**, each testing one precedence rule, headline case, or strategy derivation in isolation with hand-built minimal fixtures — no real Module 4-10 execution required, matching how `tests/decision_components/` already unit-tests `RuleEngine`/`ConflictResolver` against constructed `InputBundle`s rather than live pipeline runs.
2. **`test_thumbnail_planner.py`** integration-style tests using the same fixture-directory pattern already used by `tests/test_main_pipeline.py` — one fixture set with Module 8/9 artifacts present (flags on), one fixture set with them absent (flags off), asserting the two documented behaviors from §6.3 side by side.
3. **Regression tests for Gaps C and D specifically**: a test asserting `GenerationBundleBuilder.build_generation_bundle()` populates `canny_path` when a workspace layer carries `canny_hint_path`, and a test asserting a two-object `CompositionWorkspace` produces a `GenerationBundle.reference_image_paths` with two distinct keys, not one — these two tests alone should have existed already (the current test suite has no assertion that `canny_path` is ever non-`None`, and no fixture with two simultaneously-preserved objects), and adding them is valuable independent of everything else in this document.
4. **Backward-compatibility regression suite**: every existing test file that currently exercises `image_generator.py`, `composition_engine.py`, and `main.py` must continue to pass completely unmodified — this is verified, not just asserted, by running the full existing `pytest.ini`-configured suite both before and after the change with `THUMBNAIL_PLANNER_ENABLED=False` and diffing the results (must be zero diff).
5. **Determinism test**: run `ThumbnailPlanner().plan(video_id)` twice against identical fixture input and assert byte-identical `GenerationPlan` JSON (same pattern already used by Module 6's and Module 9's own determinism tests) — required because the Planner's whole value proposition is being the one deterministic, cacheable artifact the brief asked for.

---

## 19. Migration Strategy

Zero-downtime, flag-gated, in this order:

1. Land `models.py` additions (§7) and `thumbnail_planner_exceptions.py` — pure additions, no risk, can merge immediately.
2. Land the Gap C / Gap D fixes to `CompositionLayer`, `GenerationBundleBuilder`, and `composition_engine.py`'s layer-building loop, plus their regression tests (§18.3) — these are bug fixes; they change output only for workspaces that previously silently lost canny/multi-object data, so the risk is "more correct output," not behavior change for any currently-passing test or currently-generated single-object thumbnail.
3. Land `planner_components/` and `thumbnail_planner.py` fully, with `THUMBNAIL_PLANNER_ENABLED = False` — dead code path, mergeable and testable in isolation, zero effect on `main.py`'s current behavior.
4. Land the `main.py` call site (new `if THUMBNAIL_PLANNER_ENABLED:` block calling the Planner and passing its result into `_run_module7_generation`'s new optional parameter) with the flag still `False` in committed config — verified no-op by the backward-compat regression suite (§18.4).
5. In a separate, explicitly-opted-in environment (not committed config default), flip `ASSET_EXTRACTION_ENGINE_ENABLED` and `DECISION_ENGINE_ENABLED` to `True` first, alone, and confirm Module 8/9 run correctly end-to-end against real creator data on the RTX 4060 rig (this exercises code that has never run in production despite being fully tested in isolation — the first real-world validation belongs here, not bundled with the Planner's own rollout).
6. Only then flip `THUMBNAIL_PLANNER_ENABLED = True`, so the Planner's first live runs have real (not fallback-path) Module 8/9 data available to validate the §6.3 precedence logic against, not just its fallback branch.

Each step is independently revertible by flipping its flag back, with no data migration required at any step (nothing already on disk under `data/composition_workspaces/`, `data/prompt_packages/`, etc. changes shape).

---

## 20. Phase-by-Phase Implementation Plan (for autonomous coding agents / Codex handoff)

**Phase 1 — Models & Exceptions.** Add every model in §7 to `models.py`, and `thumbnail_planner_exceptions.py`, with unit tests validating field constraints only (no orchestration logic). Deliverable: green tests, zero other files touched.

**Phase 2 — Gap C / Gap D fixes.** Add `CompositionLayer.canny_hint_path`, update `composition_engine.py`'s layer-building loop to resolve and set it, update `GenerationBundleBuilder` to (a) set `canny_path` from it and (b) key `reference_image_paths`/`mask_paths` by `asset_id`. Add the two regression tests from §18.3. Deliverable: green tests including every pre-existing test in `tests/test_composition_engine.py`, `tests/test_composition_components/`, and `tests/test_image_generator.py` unmodified and passing.

**Phase 3 — `planner_components/` unit-level pieces.** `PrecedenceResolver`, `HeadlinePlanner`, `StrategyDeriver`, `ConditioningManifestBuilder`, each with isolated unit tests per §18.1. No orchestrator yet.

**Phase 4 — `io.py` (planner-side ingestion + cache).** `load_planner_input_bundle`, `PlanCache`, mirroring `decision_components/io.py`'s existing structure precisely enough that a reviewer can diff the two files side by side.

**Phase 5 — `ThumbnailPlanner` orchestrator + public API.** `thumbnail_planner.py` per §12, wired to Phases 3-4's components via constructor injection, `THUMBNAIL_PLANNER_ENABLED` added to `config.py` (default `False`). Integration tests per §18.2.

**Phase 6 — Module 7 extension.** `ConditioningAssetResolver.resolve(..., plan=None)`, `WorkflowBuilder._slots()` new keys, `ImageGeneratorPipeline.run(..., generation_plan=None)`, `run_image_generation_pipeline(..., generation_plan=None)` — all additive optional parameters. Full existing Module 7 test suite (`tests/test_image_generator.py`, `tests/test_generation_components/`, `tests/test_comfyui_client.py`) must remain green unmodified.

**Phase 7 — `multi_object_reference.json` fragment.** New fragment file plus one new condition in `WorkflowBuilder._select_fragments()`. Test against a synthetic two-object `GenerationConditioningContext` fixture.

**Phase 8 — `main.py` wiring (flag off by default).** New import block, new `if THUMBNAIL_PLANNER_ENABLED:` branch calling `ThumbnailPlanner().plan(...)` between the existing Module 10 and Module 7 call sites, passing the result into `_run_module7_generation`'s new parameter. Full `tests/test_main_pipeline.py` must remain green with the flag at its default (`False`).

**Phase 9 — Staged flag flip (non-code, ops task).** Per §19 steps 5-6, in a non-default environment — validate Module 8/9 against real data first, then the Planner. Document actual VRAM/duration numbers observed on the RTX 4060 rig back into `config.py`'s comments (matching how `ASSET_EXTRACTION_MODEL_TIMEOUT_SECONDS` etc. are already annotated with their rationale).

**Phase 10 — Headline-generation follow-up (explicitly out of scope here).** Only after Phase 9 is stable: a *separate* design document for `HeadlineSource.GENERATED`, since it requires new Module 5/9-adjacent decision logic this document deliberately does not introduce.

---

## Appendix — Summary of Every File Touched

| File | Change type |
|---|---|
| `modules/models.py` | Additive (new models, §7) |
| `modules/thumbnail_planner.py` | New file |
| `modules/thumbnail_planner_exceptions.py` | New file |
| `modules/planner_components/*.py` | New files |
| `modules/composition_engine.py` | Additive (one new `resolve("canny_map")` call + assignment in existing loop) |
| `modules/composition_components/generation_bundle_builder.py` | Additive (one new field read, one changed dict-key expression) |
| `modules/generation_components/conditioning_asset_resolver.py` | Additive (`resolve()` gains one optional trailing parameter) |
| `modules/image_generator.py` | Additive (`WorkflowBuilder._slots()` new dict keys; `ImageGeneratorPipeline.run()` / `run_image_generation_pipeline()` new optional trailing parameter) |
| `modules/config.py` | Additive (new delimited section, §15) |
| `main.py` | Additive (new flag-gated call site between existing Module 10 and Module 7 calls) |
| `workflows/fragments/multi_object_reference.json` | New file |
| `tests/test_thumbnail_planner.py`, `tests/test_planner_components/*` | New files |
| `tests/test_composition_*`, `tests/test_image_generator.py`, `tests/test_main_pipeline.py` | Additive (new regression cases only) |

No file in this table has an existing public function signature removed or an existing field's meaning changed.
