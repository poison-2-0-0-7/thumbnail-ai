# MODULE 5.5 — THUMBNAIL COPYWRITER & LAYOUT PLANNER
## Architecture Design Document
### `thumbnail-ai` — Deterministic Design Blueprint Stage

**Status:** Design only. No implementation code, no pseudocode.
**Source of truth:** `github.com/poison-2-0-0-7/thumbnail-ai` (cloned and read in full before writing this document — `main.py`, all of `modules/models.py`, `modules/config.py`, `modules/redesign_spec_engine.py`, `modules/prompt_compiler.py`, `modules/thumbnail_intelligence.py`, `modules/decision_engine.py` + `decision_components/`, `modules/composition_engine.py` + `composition_components/`, and the existing Module 5, 6.5, 7, 8, 9, 10, and Pipeline-Validation architecture documents in `docs/`).

---

## 0. Grounding note — read this first

Before proposing anything, the following load-bearing facts were verified directly in the repository rather than assumed:

1. **The live pipeline, per `main.py`, is:** CSV Reader (1) → YouTube Metadata (2) → Thumbnail Downloader (3) → Thumbnail Intelligence (4) → Redesign Specification (5) → Prompt Compiler (6) → `AssetComposer.prepare_generation_workspace` (labelled "Module 10" in `main.py`'s comments, which internally instantiates `VisualReferenceEngine` — Module 6.5 — to build the ControlNet/IP-Adapter reference pack) → Image Generation (7, `image_generator.py` + `comfyui_client.py`). Module 8 (`asset_extraction_engine.py`) and Module 9 (`decision_engine.py`) exist as fully-built, independently tested engines but are **not yet wired into `run_pipeline()`** — `main.py`'s own module list ends at "Future modules …" after Module 7.

2. **Module 5 (`redesign_spec_engine.py`) and Module 6 (`prompt_compiler.py`) are both explicitly non-reasoning.** Their docstrings state, respectively, that they "never … invoke an AI/LLM, or invent creative content" and are "a compiler, not a reasoning or generation system." Module 4 (`thumbnail_intelligence.py`, via `GeminiReasoning`) is documented as the pipeline's **only** reasoning stage.

3. **`RedesignSpecification.text_overlay` (`TextOverlaySpec`) is explicitly placement-only.** Its docstring reads: *"this model never contains new copy."* There is today no field anywhere in `models.py` that holds an actual headline string. `PromptPackage.typography_instructions` is a `str` built by `prompt_compiler.py` purely from `TextOverlaySpec.include_text` / `placement_zone` — it currently emits generic instructions like "reserve top-left region for text" with no copy, so any headline text seen in a generated thumbnail today is either absent or hallucinated by the diffusion model's own text-rendering behavior, not authored anywhere in the pipeline. This is the exact gap the brief describes.

4. **A numbering collision already exists in the repo and is already precedent for fractional/decimal insertion.** `docs/Module_6.5_Visual_Reference_Engine_Architecture.md` establishes that new stages inserted between two existing integer-numbered modules take the decimal of the module they feed into. `docs/module5_architecture_v2.md` additionally shows the project's *original* roadmap numbered differently from what was actually built (its Module 8–11 were "Redesign QA / Outreach Copywriter / Email Assembler / Gmail Sender" — an outreach pipeline that was never built; the repo instead built Module 7 Generation, Module 8 Asset Extraction, Module 9 Decision Engine, Module 10 Asset Composer). Both existing precedents mean: **this document does not invent a numbering convention, it follows one already in use twice.**

5. **The task brief's input list includes two items that do not exist yet in the form implied.** "Thumbnail Planner output" is not a module in this repo — the closest existing artifact is `RedesignSpecification.layout_direction` (`LayoutDirection`, Module 5). "Visual References" (Module 6.5, VRE) run **after** Module 6 in the live pipeline (inside the Module 10 workspace-prep step), so a stage inserted between Module 5 and Module 6 architecturally *cannot* consume VRE output without either running it twice or reordering three existing, tested modules — which §Compatibility of the brief forbids. §2 and §3 below resolve this rather than silently ignoring it.

---

## 1. Executive Summary

The pipeline analyzes thumbnails exceptionally well (Module 4), converts that analysis into deterministic redesign targets (Module 5), and compiles those targets into a ComfyUI-ready prompt package (Module 6) — but nothing in that chain ever **decides what the thumbnail should say**, or **arbitrates where competing visual elements go** the way a human designer would. `TextOverlaySpec` reserves a rectangle; it never fills it. `LayoutDirection` names a focal zone; it never resolves what happens when the focal zone, the safest text zone, and the largest detected face all overlap.

This document specifies **Module 5.5 — Thumbnail Copywriter & Layout Planner**, a new deterministic stage inserted between Module 5 (Redesign Specification) and Module 6 (Prompt Compiler). It consumes the existing Module 2 (video metadata/transcript), Module 4 (Thumbnail Intelligence), and Module 5 (Redesign Specification) outputs — all already computed and already available at this point in the pipeline — and produces a new frozen artifact, the **Design Blueprint**, containing an authored, scored headline; explicit face/object/background/color/lighting/camera strategy; and a fully arbitrated layout (text position, subject position, visual-priority ranking, safe margins) with zero geometric conflicts.

Module 6 is extended, not rewritten, to prefer the Design Blueprint's authored headline and resolved zones over its current generic placement-only logic, while remaining fully backward-compatible when no blueprint exists. No existing module's public interface, stored schema, or test suite changes.

---

## 2. Current Pipeline Analysis

| Stage | File | Reasoning? | Produces | Persists to |
|---|---|---|---|---|
| 1. CSV Reader | `csv_reader.py` | No | `list[Creator]` | — |
| 2. YouTube Metadata | `youtube_metadata.py` | No | `VideoMetadata` (title, description, transcript, categories) | — |
| 3. Thumbnail Downloader | `thumbnail_downloader.py` | No | `ThumbnailData` | `data/thumbnails/` |
| 4. Thumbnail Intelligence | `thumbnail_intelligence.py` | **Yes** (local reasoning stage; OCR/face/object/color/composition are deterministic CV, but `GeminiReasoning` is the one interpretive step) | `ThumbnailIntelligence` | `data/analysis/` |
| 5. Redesign Specification | `redesign_spec_engine.py` | No — pure rule tables over Module 4 output | `RedesignSpecification` | `data/redesign_specs/` |
| 6. Prompt Compiler | `prompt_compiler.py` | No — pure template compilation over Module 5 output | `PromptPackage` | `data/prompt_packages/` |
| 6.5 Visual Reference Engine | `visual_reference_engine.py` | No | `VisualReferenceManifest` (crops, masks, depth/canny maps) | `data/visual_references/` — invoked from *inside* Module 10's workspace prep, i.e. after Module 6 |
| 10. Asset Composer | `composition_engine.py` | No | `CompositionWorkspace` / `GenerationBundle` | `data/composition_workspaces/` (per `MODULE7_COMPOSITION_WORKSPACE_ROOT`) |
| 7. Image Generation | `image_generator.py` + `comfyui_client.py` | No | `ImageGenerationResult` | `data/generated_thumbnails/` |
| 8. Asset Extraction | `asset_extraction_engine.py` | No | `AssetExtractionManifest` | built, tested, not yet called from `main.py` |
| 9. Decision Engine | `decision_engine.py` | Yes (rule engine + optional local-LLM adjudication via `llm_reasoner.py`, Ollama-backed) | `DecisionManifest` | built, tested, not yet called from `main.py` |

Every stage validates `video_id` non-empty via the same `@field_validator` pattern, is a frozen (`ConfigDict(frozen=True)`) Pydantic model, persists via an atomic write-then-rename to a `DEFAULT_{X}_DIR / {video_id}.json` path governed by a `{X}_FILENAME_TEMPLATE` constant in `config.py`, and logs to its own rotating Loguru sink at `LOG_DIR / module{N}.log`. Module 5.5 follows this convention exactly (§9–§13).

---

## 3. Architectural Gap Analysis

**Gap A — No authored copy anywhere in the deterministic path.** `TextOverlaySpec` is geometry-only by explicit design. There is no headline string, no scoring, no candidate selection anywhere upstream of Module 7. Whatever text appears on a generated thumbnail today is an artifact of the diffusion model's own (unreliable) text rendering, not a pipeline decision.

**Gap B — No cross-element arbitration.** `LayoutDirection.focal_zone`, `SubjectTreatment.target_bbox`, and `TextOverlaySpec.placement_zone` / `avoid_zones` are each computed independently in `redesign_spec_engine.py` from thresholds in `config.py` (`RULE_OF_THIRDS_LOW_THRESHOLD`, `CLUTTER_HIGH_THRESHOLD`, `MIN_SUBJECT_AREA_RATIO`, `MIN_NEGATIVE_SPACE_RATIO`). Nothing checks whether the resulting zones collide. A large detected face and a wide `placement_zone` can and do overlap in the current schema with no resolution step.

**Gap C — No visual-hierarchy ranking.** `CompositionAnalysis.visual_hierarchy_score` (Module 4) *measures* whether one focal point dominates a thumbnail; nothing downstream *decides* which element (face, headline, object) should be that dominant point, or by how much it should be enlarged/de-emphasized relative to the others.

**Gap D — No object-level emphasis/removal strategy beyond keep/remove/preserve.** `ObjectDirective.action` (Module 5) is a three-way enum with no scale, no rank, and no interaction with the headline or face. A professional designer's "shrink this, cut that one entirely, blow this one up behind the headline" judgment has no home in the current schema.

**Gap E — Prompt Compiler has nothing better to compile.** Because Gaps A–D exist upstream, `prompt_compiler.py`'s `typography_instructions` / `composition_instructions` builders (lines building strings such as "reserve region for text overlay") are already about as good as they can be *given their current inputs*. This is not a Module 6 defect; it is a Module 6 starved of a Module 5.5 it doesn't yet have.

**Resolution:** all four gaps are closed by inserting one new deterministic stage that (a) authors and scores headline candidates from data Module 4 already computed, and (b) runs a single conflict-resolution pass over every existing per-element zone/action before anything reaches Module 6.

---

## 4. Module Responsibilities

| Module | Responsibility | Change |
|---|---|---|
| 1–4 | Unchanged | None |
| **5.5 (new)** | Author and score headlines; decide hook type, emotion, face/object/background/camera/lighting strategy; resolve all zone conflicts into one non-overlapping layout; rank visual priority | New |
| 5 | Unchanged — remains the sole source of `ColorDirection`, raw `SubjectTreatment`/`LayoutDirection` targets, and `ObjectDirective` base actions, which 5.5 consumes and refines, never recomputes | None |
| 6 | Gains one new optional constructor/argument path: when a `DesignBlueprint` is supplied, prefer its authored headline and resolved zones over the current placement-only derivation; when absent, fall back to today's exact behavior byte-for-byte | Additive only |
| 6.5, 7, 10 | Unchanged. `VisualReferenceManifest`'s crop/mask assets remain generation-time conditioning inputs; the Blueprint's zones are expressed in the same `BoundingBox` normalized-fraction coordinate system already used by Module 4/5, so Module 10's placement engine can consume either without translation, but Module 10's existing consumption of `RedesignSpecification` is not modified by this document | None |
| 8, 9 | Unchanged | None |

---

## 5. Design Blueprint Model

New frozen Pydantic models in `modules/models.py`, appended after the existing `# Module 6 — Prompt Compiler` section and before `# Module 6.5`, following the file's existing `model_config = ConfigDict(frozen=True)` + `@field_validator("video_id")` convention exactly.

**`HeadlineCandidate`**
| Field | Type | Notes |
|---|---|---|
| `text` | `str` | The candidate headline |
| `template_id` | `str` | Which deterministic template produced it (§6) |
| `curiosity_score` | `float [0,1]` | Lexicon/pattern-based |
| `emotional_impact_score` | `float [0,1]` | Lexicon-based, seeded from `GeminiReasoning.emotional_impact` |
| `readability_score` | `float [0,1]` | Flesch-style heuristic, deterministic |
| `ctr_potential_score` | `float [0,1]` | Weighted composite (see below) |
| `character_count` | `int` | |
| `mobile_readability_score` | `float [0,1]` | Penalizes candidates exceeding `MODULE55_MOBILE_CHAR_SOFT_LIMIT` |
| `brand_consistency_score` | `float [0,1]` | Checks against `elements_to_preserve` / banned-term list |
| `composite_score` | `float [0,1]` | Fixed-weight sum, tie-break order defined in §6 |

**`DesignBlueprint`** (the top-level artifact, structurally mirroring the brief's example JSON, using existing types wherever one already exists rather than duplicating it):

| Field | Type | Reuses |
|---|---|---|
| `video_id` | `str` | validated non-empty, same pattern as every other manifest |
| `headline` | `str` | selected `HeadlineCandidate.text` |
| `headline_variants` | `list[HeadlineCandidate]` | all scored candidates, ranked |
| `headline_score` | `float` | selected candidate's `composite_score` |
| `hook_type` | `Literal["curiosity","shock","controversy","benefit","authority","fomo","question","how_to"]` | new enum, §6 |
| `emotion` | `str` | derived from `GeminiReasoning.emotional_impact` + `FaceDetail.emotion`, not re-reasoned |
| `face_strategy` | `Literal["smile","neutral","shock","exaggerate","remove","preserve"]` | new enum |
| `object_strategy` | `list[ObjectLayoutDirective]` | **wraps**, does not replace, Module 5's `ObjectDirective` — adds `scale_factor: float` and `emphasis_rank: int` |
| `background_strategy` | `Literal["keep","replace","blur","darken","simplify"]` | refines `RedesignSpecification`'s implicit background handling (today only inferable from `ObjectDirective` actions) |
| `text_position` | `TextPlacement` | **reused verbatim** from `composition_components`'s existing `TextPlacement` model — no new placement type invented |
| `subject_position` | `BoundingBox` | reused type |
| `camera_distance` | `Literal["close_up","medium","wide"]` | new enum, derived from `SubjectTreatment.crop_tighter` + face bbox area |
| `lighting` | `str` | derived from `ColorProfile.brightness`/`contrast`, phrase table |
| `color_palette` | `list[str]` | pass-through of `RedesignSpecification.color_direction`, surfaced for compiler convenience — not recomputed |
| `visual_priority` | `list[str]` | ranked element identifiers: subset/ordering of `{"headline","face","primary_object","background"}` |
| `branding_constraints` | `list[str]` | pass-through of `RedesignSpecification.elements_to_preserve` plus any hard constraints from `SAFETY_CONSTRAINTS` |
| `conflicts_resolved` | `int` | audit count, same naming convention as `DecisionManifest.conflicts_resolved` |
| `status` | `Literal["success","partial","error"]` | matches `IntelligenceStatus`/`DecisionManifestStatus` three-way convention |
| `partial_failure_reasons` | `list[str]` | |
| `error_message` | `Optional[str]` | |
| `duration_seconds` | `float` | |
| `generated_at` | `str` | ISO-8601 UTC, same as every other artifact |

This is a specification, not a prompt: every field is a closed enum, a bounded score, a normalized bounding box, or a pass-through — never free-form generation instructions.

---

## 6. Copywriter Architecture

**Non-negotiable constraint carried over from Module 5/6's own docstrings:** the copywriter must not become a second LLM-reasoning stage. It must not call Ollama, Gemini, or any network model. Module 4 already ran the one interpretive step the pipeline permits; Module 5.5 **consumes** `GeminiReasoning.emotional_impact`, `.redesign_recommendations`, `.elements_to_preserve`, and `VideoMetadata.title`/`.transcript`, and turns them into copy through fixed templates and lexicon tables — architecturally identical in spirit to how `redesign_spec_engine.py` turns `CompositionAnalysis` scores into `LayoutDirection` via fixed thresholds, never by re-reasoning.

**Candidate generation** — a fixed **Headline Template Library**, keyed by `hook_type`, analogous to `workflows/*.json`'s niche-keyed template files and to `decision_components/rules/*.py`'s one-file-per-action-type organization:
- `templates/curiosity.py`, `templates/shock.py`, `templates/benefit.py`, `templates/question.py`, `templates/how_to.py`, `templates/authority.py`, `templates/fomo.py`, `templates/controversy.py`
- Each template is a slot-fill pattern (e.g. `"The {subject} Nobody Talks About"`, `"I Tried {subject} For {duration}"`) whose slots are filled deterministically from `VideoMetadata.title` keyword extraction (simple TF-based noun-phrase pull, not an LLM call) and `GeminiReasoning.elements_to_preserve`.
- `hook_type` selection itself is rule-based: `GeminiReasoning.content_mismatch_detected` and `.curiosity_gap_score` map to specific hook types via a threshold table in `config.py` (`MODULE55_HOOK_TYPE_THRESHOLDS`), mirroring `redesign_spec_engine.py`'s existing threshold-table style exactly.
- Every applicable template for the selected `hook_type` (plus the two next-best hook types, for variant diversity) produces one `HeadlineCandidate` each, giving 3–6 variants per run — bounded and deterministic given identical inputs.

**Scoring** — each sub-score is a pure function of the candidate string and already-known context, no model calls:
- `curiosity_score`: presence/count of curiosity markers (open loops, numbers, contrast words, "?", withheld information patterns) against a fixed lexicon, `MODULE55_CURIOSITY_LEXICON`.
- `emotional_impact_score`: lexicon match against the `emotion` field selected in §7, weighted by whether the candidate reinforces or contradicts the detected facial emotion (contradiction is penalized — a "shock" headline over a smiling face scores lower).
- `readability_score`: syllable/word-length heuristic (no external NLP model), same "no network, no invented content" spirit as Module 5/6.
- `character_count` / `mobile_readability_score`: direct length check against `MODULE55_MOBILE_CHAR_SOFT_LIMIT` (default 40, matching common thumbnail-text legibility guidance) and `MODULE55_MOBILE_CHAR_HARD_LIMIT`.
- `brand_consistency_score`: penalizes any candidate containing a term from `RedesignSpecification.elements_to_preserve`'s *negative* space (i.e. terms Module 4 flagged as weaknesses) or from a configurable banned-term list.
- `ctr_potential_score`: seeded from Module 4's own `GeminiReasoning.ctr_potential_score` rather than re-deriving CTR from scratch — this candidate-level score expresses "how much does *this specific headline* move the needle relative to the source thumbnail's already-measured baseline," computed as a bounded delta, not a fresh absolute estimate.
- `composite_score = w1·curiosity + w2·emotional_impact + w3·readability + w4·ctr_potential + w5·mobile_readability + w6·brand_consistency`, fixed weights in `config.py` (`MODULE55_HEADLINE_SCORE_WEIGHTS`), ties broken by shortest `character_count` then lexical order — fully deterministic, reproducible given identical Module 4/5 input.

Best candidate by `composite_score` becomes `DesignBlueprint.headline`; all candidates are retained in `headline_variants` for auditability and for a future A/B or human-review workflow, without the pipeline itself branching on them.

---

## 7. Layout Planner Architecture

The Layout Planner is a **single conflict-resolution pass**, not a fresh spatial-reasoning system — it takes zones that Module 4/5 already computed and arbitrates between them, using the same normalized-`BoundingBox` coordinate system used everywhere else in the repo (no new coordinate convention introduced).

**Inputs already on hand at this point in the pipeline (no new upstream data required):**
- Face zone(s): `ThumbnailIntelligence.faces` (`FaceDetail.bbox`, `.is_largest`, `.position_label`)
- Object zones: `ThumbnailIntelligence.objects` (`DetectedObject.bbox`) crossed with `RedesignSpecification.object_directives` (which ones survive)
- Text zone candidate: `RedesignSpecification.text_overlay.placement_zone` / `.avoid_zones`
- Subject target: `RedesignSpecification.subject_treatment.target_bbox`
- Focal zone: `RedesignSpecification.layout_direction.focal_zone`
- Negative space: `ThumbnailIntelligence.composition.negative_space_ratio`, rule-of-thirds intersections implied by `composition.rule_of_thirds_score`

**Resolution algorithm (rule table, not a solver):**
1. Compute pairwise IoU (intersection-over-union) between every retained zone (face, each kept object, text placement zone).
2. Any pair exceeding `MODULE55_MAX_ZONE_OVERLAP` (default 0.15) is a conflict. Conflicts are resolved by a fixed precedence order — `face > headline_text > primary_object > secondary_object > background` — identical in spirit to `decision_components/conflict_resolver.py`'s existing precedence-table pattern for resolving competing rule/LLM decisions in Module 9.
3. The lower-precedence zone is shifted to the nearest of the four rule-of-thirds-adjacent safe quadrants that does not conflict with anything already placed; if no quadrant is free, the zone's `emphasis_rank` (for objects) is demoted or the element is marked for `background_strategy = "simplify"` rather than left overlapping.
4. `text_position` (`TextPlacement`) is the final, conflict-free text zone; `subject_position` is the final, conflict-free subject/face zone.
5. Safe margins are enforced as a fixed percentage inset (`MODULE55_SAFE_MARGIN_RATIO`, default 0.05) applied to every zone before step 1, so nothing the Layout Planner outputs touches the frame edge — a professional-designer convention the current schema has no equivalent for.
6. `visual_priority` is the final precedence-ordered list, truncated to elements actually present (a thumbnail with no detected object omits `"primary_object"` rather than emitting a placeholder).
7. `camera_distance` and `subject_position`'s implied scale are derived from the resolved face/subject bbox area post-resolution, not the raw Module 4 measurement, so they reflect what the layout actually decided rather than what the source thumbnail happened to contain.

`conflicts_resolved` on the `DesignBlueprint` records how many pairwise conflicts step 2 found, giving the same audit signal `DecisionManifest.conflicts_resolved` already provides for Module 9 — consistent observability across both conflict-resolution stages in the codebase.

---

## 8. Design Decision Engine

Rather than inventing a third decision-making pattern, Module 5.5's decision engine **reuses the two-phase candidate → resolved shape already proven in `decision_components`** (Module 9), scoped down to the design-strategy fields that aren't headline text or layout geometry: `hook_type`, `emotion`, `face_strategy`, `background_strategy`, `camera_distance`, `lighting`.

- **Phase 1 — candidate generation:** for each of the six fields above, a small rule table (one file per field, mirroring `decision_components/rules/{add,enhance,keep,remove,replace}_rules.py`'s one-file-per-concern layout) proposes a value from thresholds over `GeminiReasoning`, `ColorProfile`, `CompositionAnalysis`, and `FaceAnalysis` — e.g. `face_strategy_rules.py`: if `FaceDetail.smile_detected` is `True` and `hook_type != "shock"`, propose `"preserve"`; if `emotion` resolves to a high-arousal label and no smile was detected, propose `"exaggerate"` only when `RedesignSpecification.subject_treatment.crop_tighter` is already `True` (never *invents* a crop decision Module 5 didn't already lean toward).
- **Phase 2 — resolution:** because every field here has exactly one rule source (there is no separate LLM adjudicator for design strategy — that stays exclusive to Module 9's post-generation decisions per §Compatibility), resolution is deterministic table lookup, not conflict adjudication. This keeps Module 5.5 fully rule-based and network-call-free, distinct from Module 9 which legitimately calls a local Ollama model for post-generation edit decisions.
- All six fields, plus the Copywriter (§6) and Layout Planner (§7) outputs, are assembled into one `DesignBlueprint` by a single `build_design_blueprint()` entry point — the direct sibling of `build_redesign_specification()` and `compile_prompt_package()`.

---

## 9. Data Flow

```
VideoMetadata (Module 2)  ─┐
ThumbnailIntelligence      ├──►  build_design_blueprint()  ──►  DesignBlueprint  ──►  save_design_blueprint()
(Module 4)                 │        │            │                                        │
RedesignSpecification      ┘        │            │                                        ▼
(Module 5)                          │            │                          data/design_blueprints/{video_id}.json
                          Copywriter │      Layout Planner
                          (§6)       │         (§7)
                                     ▼
                          Design Decision Engine (§8)
```

`main.py` gains one new stage inserted between the existing Module 5 and Module 6 calls:

```
redesign_spec  = build_redesign_specification(intelligence)     # unchanged
save_redesign_spec(redesign_spec, ...)                          # unchanged
                                                                  # NEW ↓
design_blueprint = build_design_blueprint(intelligence, redesign_spec, metadata)
save_design_blueprint(design_blueprint, ...)
                                                                  # NEW ↑
prompt_package = compile_prompt_package(redesign_spec, design_blueprint=design_blueprint)  # extended signature, optional kwarg
save_prompt_package(prompt_package, ...)                        # unchanged
```

A `Module5_5Error` (skip, log, `continue`) failure follows the exact `except (...) as exc: logger.error(...); skipped += 1; continue` pattern already used for every other stage in `run_pipeline()` — the creator's run is skipped, not the whole batch.

---

## 10. Interfaces

New public functions in `modules/design_blueprint_engine.py`, matching the existing module-level (not class-level) function-export pattern used by `redesign_spec_engine.py` and `prompt_compiler.py`:

- `build_design_blueprint(intelligence: ThumbnailIntelligence, redesign_spec: RedesignSpecification, metadata: VideoMetadata) -> DesignBlueprint`
- `save_design_blueprint(blueprint: DesignBlueprint, *, blueprint_dir: Path = DEFAULT_DESIGN_BLUEPRINT_DIR) -> Path`
- `load_design_blueprint(video_id: str, *, blueprint_dir: Path = DEFAULT_DESIGN_BLUEPRINT_DIR) -> DesignBlueprint` (mirrors the load-side helper already present for other manifests)

Exceptions in `modules/design_blueprint_exceptions.py`, matching `redesign_spec_exceptions`-style naming:
- `DesignBlueprintError` (base)
- `InvalidRedesignSpecError` — raised when the incoming `RedesignSpecification.status == "error"` (mirrors Module 6's `InvalidRedesignSpecError` guard against a bad Module 5 output)
- `DesignBlueprintCacheError` — atomic-write failure, same semantics as `RedesignSpecCacheError` / `PromptPackageCacheError`

`compile_prompt_package()`'s signature gains **one optional keyword-only parameter**, `design_blueprint: Optional[DesignBlueprint] = None`. When present, `typography_instructions` is built from `blueprint.headline` + `blueprint.text_position` instead of the current placement-only string; every other Module 6 code path, and every call site that does not pass the new argument, is byte-for-byte unchanged. This satisfies the brief's "preserve existing APIs" requirement literally: the existing signature remains valid, callable, and behaviorally identical when the new argument is omitted.

---

## 11. Configuration

New constants in `modules/config.py`, appended after the existing `# Module 6 — Prompt Compiler` block, following the file's `MODULE{N}_*` / `DEFAULT_{X}_DIR` naming exactly:

```
MODULE55_LOG_PATH: Path = LOG_DIR / "module5_5.log"
DEFAULT_DESIGN_BLUEPRINT_DIR: Path = PROJECT_ROOT / "data" / "design_blueprints"
DESIGN_BLUEPRINT_FILENAME_TEMPLATE: str = "{video_id}.json"

MODULE55_MOBILE_CHAR_SOFT_LIMIT: int = 40
MODULE55_MOBILE_CHAR_HARD_LIMIT: int = 60
MODULE55_MAX_ZONE_OVERLAP: float = 0.15
MODULE55_SAFE_MARGIN_RATIO: float = 0.05
MODULE55_HOOK_TYPE_THRESHOLDS: dict[str, float] = {...}
MODULE55_CURIOSITY_LEXICON: frozenset[str] = frozenset({...})
MODULE55_HEADLINE_SCORE_WEIGHTS: dict[str, float] = {...}
```

No existing constant is renamed, retyped, or removed; `PROJECT_ROOT` and `LOG_DIR` are imported, not redefined, exactly as every other module-specific config block already does.

---

## 12. Logging

A dedicated rotating Loguru sink at `MODULE55_LOG_PATH`, `10 MB` rotation / `30 days` retention, `enqueue=True`, `_LOG_FORMAT` string — copied verbatim from `redesign_spec_engine.py`'s `_configure_logger()` / module-import-time `_configure_logger()` call pattern, so log format and rotation policy are indistinguishable from every other module's logs in `logs/`.

Log lines follow the existing structured-kwarg style: `logger.info("Design blueprint built video_id={vid} headline_score={score}", vid=..., score=...)`; `logger.warning(...)` on any candidate falling below `MODULE55_MOBILE_CHAR_SOFT_LIMIT` before selection; `logger.error(...)` on `DesignBlueprintError` subclasses, matching Module 5/6's exact severity conventions.

---

## 13. Error Handling

Three-way `status` outcome (`success` / `partial` / `error`), identical semantics to `IntelligenceStatus` and `DecisionManifestStatus`:

- **`error`** — `redesign_spec.status == "error"` (nothing valid to build from) or zero headline candidates survive scoring (e.g. every template produced an empty slot-fill). Raises `InvalidRedesignSpecError`; caught in `main.py` exactly like every other stage, incrementing `skipped` and `continue`-ing to the next creator.
- **`partial`** — e.g. no face detected so `face_strategy` defaults to a documented safe value (`"neutral"`), or fewer than 3 headline variants were producible from the available title/transcript keywords. `partial_failure_reasons` records why, same pattern as `ThumbnailIntelligence.partial_failure_reasons`.
- **`success`** — full blueprint, all sub-scores populated, zero unresolved zone conflicts.

`save_design_blueprint()` uses the identical atomic write-then-`Path.replace()` pattern already used by every other `save_*` function in the codebase (write to `.tmp`, then rename) — no new persistence mechanism introduced.

---

## 14. Testing Strategy

Mirrors the existing per-module test-file convention (`tests/test_redesign_spec_engine.py`, `tests/test_prompt_compiler.py`) with a new `tests/test_design_blueprint_engine.py`, plus a `tests/decision_components/`-style `tests/design_blueprint_components/` directory for the per-field rule files from §8, matching `tests/decision_components/test_rule_engine.py`'s structure:

- **Copywriter unit tests**: fixed `ThumbnailIntelligence` + `RedesignSpecification` fixtures → assert deterministic, byte-identical `headline_variants` and score ordering across repeated runs (no run-to-run drift, since there is no randomness or network call anywhere in this stage).
- **Layout Planner unit tests**: synthetic overlapping-bbox fixtures (face + text zone deliberately overlapping) → assert `conflicts_resolved >= 1` and post-resolution IoU is under `MODULE55_MAX_ZONE_OVERLAP`.
- **Golden-sample regression**: extend `evaluation/benchmarking/golden_sample_manager.py`'s existing golden-manifest mechanism (`data/evaluation/golden/golden_manifest.json`) with a `design_blueprints` golden set, reusing the already-built `RegressionDetector` rather than writing a second regression harness.
- **`evaluation/module_validators/`**: add `design_blueprint_validator.py`, matching the one-validator-per-module pattern already present for every other module (`redesign_spec_validator.py`, `prompt_compiler_validator.py`, etc.), registered in `evaluation/module_validators/__init__.py` the same way its siblings are.
- **Integration test**: extend `tests/test_main_pipeline.py`'s existing end-to-end fixture run to assert the new stage's artifact is written and that `compile_prompt_package()` called *without* `design_blueprint` still produces the exact prompt package the current test suite already asserts — a direct backward-compatibility regression guard.

---

## 15. Migration Strategy

No data migration is required — Module 5.5 introduces a new artifact directory (`data/design_blueprints/`) and does not alter or require re-processing of any existing `data/redesign_specs/` or `data/prompt_packages/` file. Rollout is purely additive and can be staged:

1. Land `models.py` additions + `design_blueprint_engine.py` + `design_blueprint_exceptions.py` + `config.py` constants, fully unit-tested, **not yet called from `main.py`** — this is a zero-risk merge, identical in spirit to how Module 8/9 already sit fully built but uncalled today.
2. Land the additive `design_blueprint: Optional[...] = None` parameter on `compile_prompt_package()`, tested for both branches (with/without blueprint), still not called from `main.py`.
3. Wire the two new calls into `run_pipeline()` (§9) behind no flag — since the new stage fails soft (`skipped += 1; continue`, same as every existing stage) and Module 6's new parameter defaults to `None`-safe behavior, this step is a normal, revertible one-line-diff deployment rather than a flagged rollout.
4. Optional follow-up (out of scope for this document, flagged the way §0.5 flags scope boundaries): once Module 8/9 are wired into `main.py`, the Design Blueprint's `object_strategy`/`face_strategy` fields become natural additional inputs to Module 9's rule tables for post-generation touch-up decisions — but that is Module 9's extension to make, not this document's, per "do not redesign existing modules."

---

## 16. Phase-by-Phase Implementation Plan

**Phase 1 — Models & Config (no behavior change).**
Add `HeadlineCandidate`, `ObjectLayoutDirective`, `DesignBlueprint` to `models.py`; add `MODULE55_*` constants to `config.py`. Deliverable: schema compiles, no engine logic yet.

**Phase 2 — Copywriter subsystem.**
Implement the template library, keyword extraction, lexicon scoring, and headline selection (§6) as pure functions with 100% deterministic-output unit test coverage before any layout logic is written.

**Phase 3 — Layout Planner subsystem.**
Implement zone collection, IoU conflict detection, precedence-based resolution, and safe-margin inset (§7), unit-tested against synthetic overlapping/non-overlapping fixtures independent of the Copywriter.

**Phase 4 — Design Decision Engine.**
Implement the six per-field rule tables (§8) and the single `build_design_blueprint()` entry point that assembles Copywriter + Layout Planner + Decision Engine output into one `DesignBlueprint`, plus `save_design_blueprint()` / `load_design_blueprint()`.

**Phase 5 — Module 6 extension.**
Add the optional `design_blueprint` parameter to `compile_prompt_package()`; extend `typography_instructions`/`composition_instructions` builders to prefer blueprint data when present; regression-test the `None` path against the current test suite unchanged.

**Phase 6 — Pipeline wiring.**
Insert the two new calls into `main.py` between Module 5 and Module 6 (§9); extend `tests/test_main_pipeline.py`.

**Phase 7 — Evaluation framework integration.**
Add `design_blueprint_validator.py` to `evaluation/module_validators/`; extend the golden-sample manifest and regression detector (§14).

**Phase 8 — Documentation.**
This document is the Phase 8 deliverable for its own module, matching the repo's existing one-document-per-module convention in `docs/`.

Each phase is independently mergeable and independently testable, and no phase before Phase 6 touches `main.py` or any existing module's stored schema — satisfying the brief's compatibility requirements throughout the rollout, not just at the end state.
