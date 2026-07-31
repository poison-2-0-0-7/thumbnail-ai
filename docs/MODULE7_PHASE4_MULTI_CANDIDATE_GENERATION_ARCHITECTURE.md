# Module 7 — Phase 4: Multi-Candidate Generation Architecture

**Enhancement to the existing Module 7 (Local Image Generation Engine).**
**Repository:** `poison-2-0-0-7/thumbnail-ai`
**Scope:** `modules/image_generator.py`, `modules/models.py`, `modules/config.py`, `workflows/*.json`, `main.py` (wiring only). Modules 1–6, 8, 9, 10, 10.5, the ComfyUI Phase 2 transport layer, and the Evaluation framework are treated as fixed, reused contracts.

---

## 0. Repository Grounding — What Actually Exists Today

This section states, factually, what the current `main` branch already does, so the design below extends real code instead of inventing a parallel system. This grounding matters more than usual here, because **the skeleton of multi-candidate generation is already present but inert**.

**Confirmed, as of the current `main` branch:**

- `ImageGeneratorPipeline.run()` (`modules/image_generator.py`) already contains a `for cand_idx in range(num_candidates):` loop. For each iteration it: derives `cand_seed = base_seed + cand_idx`; clones the `PromptPackage` via `model_copy(update=...)` with the new seed; resolves a `WorkflowTemplateRef` and calls `WorkflowBuilder.build()`; calls `client.generate()` with VRAM-exhaustion fallback; runs `IdentityPreservationStage`, `FaceRestorationStage`, `BackgroundCompositor`, `UpscaleStage`, and `QualityAssuranceStage` per candidate; and appends `(cand_idx, path, QualityAssuranceReport, FaceMatchResult)` to `candidate_results`.
- `CandidateRanker.rank()` already exists and is already called after the loop. It filters candidates on `QualityAssuranceReport.hard_gate_passed`, sorts eligible candidates by `(-overall_score, -identity_similarity, candidate_index)`, and returns the winner plus a full `list[CandidateScore]` audit trail. It already raises `NoEligibleCandidateError` when every candidate fails hard gates.
- `models.py` already defines `CandidateScore` (`candidate_index`, `overall_score`, `identity_similarity`, `hard_gate_passed`, `rank`, `selected`), `GeneratedAsset.candidate_index`, `ImageGenerationResult.candidate_scores` / `.selected_candidate_index`, and `GenerationMetrics.num_candidates_requested`. The manifest schema was clearly designed with multiple candidates in mind from the start.
- `config.py` already defines `MODULE7_SAVE_CANDIDATES: bool = False`. When `True`, `ImageGeneratorPipeline.run()` copies every candidate into `{video_id}_candidates/candidate_{idx}_score_{score:.2f}.png` before deleting `tmp_candidates/`.
- **The gap:** `num_candidates` is read as `getattr(package.generation_parameters, "num_candidates", 1)`. `GenerationParameters` (in `models.py`) has **no `num_candidates` field**, so this `getattr` always falls through to its default of `1`. Every run today produces exactly one candidate. The entire loop, ranker, and manifest schema described above execute correctly but are permanently exercised with `num_candidates == 1`, making `CandidateRanker` a single-element no-op and `MODULE7_SAVE_CANDIDATES` a dead flag in practice.
- **The second gap:** even if `num_candidates` were wired to a value `> 1`, every candidate in the current loop is generated from an **identical `PromptPackage`** except for `generation_parameters.seed`. `positive_prompt`, `subject_instructions`, `background_instructions`, `typography_instructions`, `composition_instructions`, `lighting_instructions`, and `color_instructions` are copied verbatim into every candidate. Candidates therefore differ only by diffusion noise, not by any of the seven dimensions this document's brief asks for (framing, composition, lighting, typography, emphasis, background treatment, color grading, object prominence). There is no concept of a "strategy" anywhere in the codebase today.
- `DesignBlueprint` (Module 5.5, `models.py`) already carries exactly the dimensions a strategy needs to perturb: `camera_distance` (`close_up`/`medium`/`wide`), `background_strategy` (`keep`/`replace`/`blur`/`darken`/`simplify`), `face_strategy`, `object_strategy: list[ObjectLayoutDirective]` (each with `scale_factor` and `emphasis_rank`), `color_palette`, `lighting`, and `text_position`. `PromptPackage` (Module 6, `prompt_compiler.py`) already renders each of these into its own free-text instruction field (`composition_instructions`, `lighting_instructions`, `typography_instructions`, `color_instructions`, `background_instructions`). This is the exact seam a strategy layer needs — it already exists, one level up from where the candidate loop currently lives.
- `WorkflowBuilder.build()` (`modules/image_generator.py`) reloads the template from disk (`WorkflowLibrary.load()`, uncached) and rebuilds the full graph — including fragment selection and `graph_assembler.assemble()` — on every call. Inside today's candidate loop this happens once per candidate even though `profile`, `workflow_ref`, and `conditioning` are identical across candidates within a strategy; only the prompt slots and seed change. This is unnecessary rebuild work, not unnecessary *model* reload — `ComfyUIClient` does not itself reload checkpoints between calls to the same profile (that is ComfyUI server-side node caching, outside this repository's control), but the Python-side graph materialization is redone from scratch each time.
- The conditioning resolution step (`ConditioningAssetResolver.resolve()`, called once per `ImageGeneratorPipeline.run()`, **outside** the candidate loop) is already correctly cached at the run level — depth maps, masks, and reference images are resolved once and reused by every candidate. This part of the design brief ("reuse cached conditioning assets") is already satisfied and requires no change.
- `main.py`'s `_run_module7_generation()` calls `run_image_generation_pipeline()` (the real, active code path). A second function, `_persist_generated_thumbnail()`, exists lower in `main.py` and independently constructs an `ImageGenerationResult` with `candidate_index=0` hardcoded — but it is not called anywhere in `main.py`. It is dead legacy code from before `run_image_generation_pipeline()` existed and is explicitly out of scope here (removal, if desired, is a separate cleanup, not part of this architecture).

**Conclusion driving this design:** Phase 4 is not a candidate-loop redesign — the loop, the ranker, and the manifest schema already exist and are structurally correct. Phase 4's job is three things, all additive: (1) give `GenerationParameters`/`PromptPackage` a real, validated `num_candidates` and a `strategy` identity so the existing loop actually runs more than once; (2) insert a small, deterministic **Strategy Pack** layer between the `PromptPackage` produced by Module 6 and the per-candidate package used inside the loop, so each candidate's `*_instructions` fields are a bounded, deterministic rewrite of Module 6's originals rather than a byte-identical copy; and (3) extend output management, configuration, and workflow-reuse around the loop that already exists, without touching its control flow, its collaborators' constructor signatures, or `CandidateRanker`'s ranking algorithm.

---

## 1. Executive Summary

Module 7 already contains a working, tested multi-candidate execution engine — a per-candidate loop, hard-gated quality ranking, and an audit-complete manifest schema — that has never been exercised beyond one candidate because nothing populates `num_candidates`, and that has never produced *meaningfully different* candidates because nothing varies the prompt instructions per candidate. Phase 4 activates this dormant machinery rather than building a new one.

The design adds:

1. A validated `num_candidates` field and a new `strategy_pack` configuration surface, wired from `config.py` through `PromptPackage.generation_parameters` into the existing loop.
2. A `StrategyPack` / `CandidateStrategy` component that deterministically derives a per-candidate `PromptPackage` from Module 6's original package plus a `DesignBlueprint`-bounded perturbation (framing, lighting, typography emphasis, background treatment, color grading, object prominence) — reusing `DesignBlueprint`'s existing enums and `ObjectLayoutDirective.scale_factor`/`emphasis_rank` fields rather than introducing new vocabulary.
3. A `WorkflowGraphCache` inside `WorkflowBuilder`'s call site that reuses the base graph materialization across candidates sharing the same `(profile, workflow_ref, conditioning)` triple, re-substituting only the per-candidate slots (prompt text, seed) that actually change — cutting redundant template loads and fragment assembly without changing `WorkflowBuilder`'s public contract.
4. Output management that writes `candidate_manifest.json` (per-candidate strategy/params/scores) and `generation_metadata.json` (run-level provenance) alongside the existing `ImageGenerationResult` manifest — additive files, not a replacement of `ArtifactWriter`'s current format — plus zero-padded `candidate_01.png..candidate_NN.png` naming when `MODULE7_SAVE_CANDIDATES` is enabled.
5. Configuration for max candidates, strategy packs, parallel generation, per-candidate VRAM budgeting, timeouts, and retries — all additive keys in `config.py` following existing naming conventions (`MODULE7_*`), all defaulted to today's single-candidate behavior so existing deployments are unaffected until explicitly opted in.

No existing public method signature loses a parameter or changes meaning. No existing test should need to change its assertions for the `num_candidates=1` (default) case; new tests cover `num_candidates > 1`.

---

## 2. Current Pipeline Analysis

The full pipeline, as implemented and wired in `main.py`:

```
CSV (csv_reader)
  → Metadata Extraction (youtube_metadata)
  → Thumbnail Download (thumbnail_downloader)
  → Thumbnail Intelligence (thumbnail_intelligence)          [Module 4]
  → Redesign Specification (redesign_spec_engine)            [Module 5]
  → Design Blueprint (design_blueprint_engine)                [Module 5.5]
  → Prompt Compilation (prompt_compiler)                      [Module 6]
  → Visual Reference Engine (visual_reference_engine)         [Module 6.5]
  → Asset Composer / Composition Workspace (composition_engine) [Module 10]
  → Thumbnail Planner (thumbnail_planner)                     [Module 10.5]
  → Module 7: ImageGeneratorPipeline.run()
        → ConditioningAssetResolver.resolve()      (once per run)
        → for cand_idx in range(num_candidates):   (currently always 1)
              WorkflowBuilder.build()
              ComfyUIClient.generate()
              IdentityPreservationStage.verify()  (+ bounded retries)
              FaceRestorationStage.restore()
              BackgroundCompositor.composite()
              UpscaleStage.upscale()
              QualityAssuranceStage.evaluate()
        → CandidateRanker.rank()
        → ArtifactWriter.write_manifest()
        → MetricsCollector.append()
  → Evaluation framework (evaluation/pipeline_runner.py, module_validators)
```

Within Module 7, the object graph is entirely constructor-injected (`ImageGeneratorPipeline.__init__` accepts every collaborator as an optional override), which is the same dependency-injection convention used by `composition_components/`, `decision_components/`, and `planner_components/`. This convention is preserved — the new Phase 4 collaborators (`StrategyPackResolver`, `CandidateStrategyPlanner`, `WorkflowGraphCache`) are injected the same way, with concrete defaults.

The `PromptPackage` produced by Module 6 (`prompt_compiler.compile_prompt_package()`) is a `frozen` Pydantic model — every generation input is already immutable and reproducible from `RedesignSpecification` + `DesignBlueprint`. This determinism guarantee is the one hard constraint Phase 4 must not weaken: a strategy's transformation of a `PromptPackage` must itself be a pure function of `(package, blueprint, strategy_id, candidate_index)`, exactly as deterministic as everything upstream of it.

---

## 3. Architectural Gap Analysis

| Design objective (brief) | Current state | Gap |
|---|---|---|
| Generate multiple candidates | Loop exists, fully wired end-to-end | `num_candidates` never resolves above `1` — no model field, no config, no CLI/pipeline wiring |
| Vary framing/composition/lighting/typography/emphasis/background/color/object prominence | `DesignBlueprint` already models all seven dimensions; `PromptPackage` already has a matching instruction field per dimension | No component rewrites those instruction fields per candidate — every candidate gets an identical copy |
| Preserve DesignBlueprint / creator identity / required objects / headline intent | `IdentityPreservationStage`, `QualityAssuranceStage.object_preservation_score`, `_calculate_text_safe_zone_score` already enforce this per candidate | Fully satisfied already; a strategy layer must operate *within* these existing gates, not around them |
| Configurable strategy packs (Faithful / Higher emotion / Cleaner composition / Higher contrast / Aggressive CTR) | No concept of "strategy" exists anywhere | New, additive |
| Reuse workflow efficiently, avoid unnecessary reloads/duplicated preprocessing | Conditioning resolved once per run (good); `WorkflowBuilder.build()` rebuilds the full graph from disk every candidate | Partial gap — template/fragment materialization is redundant across candidates sharing profile+conditioning |
| Candidate output files (`candidate_01.png`...), `candidate_manifest.json`, `generation_metadata.json` | `MODULE7_SAVE_CANDIDATES` writes `candidate_{idx}_score_{score:.2f}.png` (non-padded, score baked into filename); one combined `ImageGenerationResult` manifest, no split files | Naming convention and file-splitting gap |
| Config: max candidates, parallel generation, VRAM limits, timeouts, retry, caching, strategy packs | `MODULE7_VRAM_HEADROOM_GB`, `COMFYUI_EXECUTION_TIMEOUT_SECONDS`, `MAX_IDENTITY_RETRIES` exist for single-candidate concerns | No candidate-count cap, no parallelism switch, no strategy-pack registry |
| Preserve existing APIs, modules, tests, config, DI, logging, evaluation, output structure | — | This is the primary constraint on *how* every gap above is closed, not a gap itself |

---

## 4. Candidate Generation Architecture

### 4.1 Where Phase 4 sits

Phase 4 inserts exactly one new step between "resolve conditioning" and "enter the per-candidate loop" inside `ImageGeneratorPipeline.run()`, and modifies the loop body to consult a resolved strategy list instead of a bare `range(num_candidates)`:

```
ConditioningAssetResolver.resolve()          [unchanged, still once per run]
StrategyPackResolver.resolve(config)          [NEW — resolves configured pack → ordered list[CandidateStrategy]]
for cand_idx, strategy in enumerate(strategies):
    CandidateStrategyPlanner.derive_package(base_package, blueprint, strategy)  [NEW]
        → cand_package  (frozen PromptPackage, instructions rewritten within bounds)
    WorkflowBuilder.build(cand_package, ..., cache=WorkflowGraphCache)  [build() unchanged; caller reuses cache]
    ComfyUIClient.generate()                  [unchanged]
    IdentityPreservationStage.verify()        [unchanged]
    FaceRestorationStage.restore()            [unchanged]
    BackgroundCompositor.composite()          [unchanged]
    UpscaleStage.upscale()                    [unchanged]
    QualityAssuranceStage.evaluate()          [unchanged]
CandidateRanker.rank()                        [unchanged]
```

`num_candidates` becomes `len(strategies)`, derived from the resolved strategy pack rather than a raw integer — this keeps "how many candidates" and "what makes each candidate different" as a single source of truth instead of two configs that can drift out of sync.

### 4.2 Preserving the Design Blueprint as an invariant, not a per-candidate input

The brief requires every candidate to "preserve the deterministic Design Blueprint while varying" the seven listed dimensions. Concretely, this means a strategy is only ever permitted to move a value **within** the value space `DesignBlueprint` already declares as legal for that video — it never introduces a value the blueprint forbids:

- `camera_distance` — a strategy may only select among `close_up`/`medium`/`wide` (the blueprint's own `Literal`), and only through a strategy-defined adjacency (e.g. "cleaner composition" may step one notch wider; it may not jump `close_up → wide` in one candidate).
- `background_strategy` — a strategy may only select among the blueprint's five literals (`keep`/`replace`/`blur`/`darken`/`simplify`); it may not invent a sixth.
- `object_strategy` (`list[ObjectLayoutDirective]`) — a strategy may scale `scale_factor` within a bounded multiplier band (e.g. ±15%) and re-order `emphasis_rank` among objects already marked `include`/`preserve`; it may never change an object's `action` (an object the blueprint says to `remove` stays removed in every candidate; an object marked `preserve` is never scaled below a floor that would make it inconspicuous).
- `color_palette` — a strategy may request a grading instruction ("higher contrast", "punchier saturation") layered onto the existing palette; it may never substitute a different palette.
- `lighting`, `text_position`, `headline` — `lighting` may be re-worded by a strategy's instruction template but must still resolve to a description compatible with the blueprint's `lighting` string; `text_position` and `headline` are **never** touched by any strategy — headline intent and typography placement are the two elements the brief calls "headline intent" and are treated as strategy-invariant by construction, not by convention.

This is enforced structurally: `CandidateStrategyPlanner.derive_package()` takes the `DesignBlueprint` as a required argument (not optional), and every rewrite rule is expressed as a transformation of a blueprint field, never as free-form prompt text unconstrained by it. A strategy that would move a blueprint field outside its declared literal/bound is a configuration error, validated at strategy-pack load time (see §5.3), not a runtime surprise.

### 4.3 Preserving creator identity and required objects

No change to `IdentityPreservationStage` or `QualityAssuranceStage.object_preservation_score`. Every candidate — regardless of strategy — passes through the exact same identity-retry loop and the exact same QA hard gates that exist today. A strategy can change *emphasis and framing*; it cannot change *whether* the creator's face matches the reference or *whether* a `preserve`-marked object is present. This is why Phase 4 does not touch `CandidateRanker` at all: ranking already operates purely on `QualityAssuranceReport`/`FaceMatchResult`, which are strategy-agnostic by design. A "faithful" strategy candidate and an "aggressive CTR" strategy candidate compete for the win on identical, unmodified quality terms.

---

## 5. Strategy Pack Architecture

### 5.1 `CandidateStrategy` — a named, bounded transformation

A `CandidateStrategy` is a frozen, declarative record — not code — describing *how far* and *in which direction* each of the seven dimensions may move for one candidate slot. It is data, validated once at load time by Pydantic, the same pattern `GenerationProfile` already uses for hardware/quality contracts. It carries:

- `name` (e.g. `"faithful"`, `"higher_emotion"`, `"cleaner_composition"`, `"higher_contrast"`, `"aggressive_ctr"`)
- `camera_distance_shift`: `-1` / `0` / `+1` (step within the blueprint's close_up→medium→wide ordering; `0` for faithful)
- `object_emphasis_bias`: a small signed multiplier band applied to `ObjectLayoutDirective.scale_factor` for objects ranked `emphasis_rank == 1`
- `background_intensity_bias`: an instruction-strength adjustment applied on top of the blueprint's `background_strategy` (e.g. push a `blur` further, or a `darken` further, without changing the strategy literal itself)
- `color_grade_bias`: an instruction-strength adjustment layered onto `color_instructions` (e.g. "increase contrast and saturation moderately") — never a palette substitution
- `typography_weight_bias`: an instruction-strength adjustment to `typography_instructions` only (e.g. bolder stroke emphasis) — never a position change
- `emotion_bias`: an instruction addition consistent with the blueprint's existing `emotion` string and `face_strategy` (e.g. amplify an already-selected `shock`/`exaggerate` face strategy; a strategy cannot invent emotion the blueprint didn't already select)
- `description` — a short human-readable rationale, surfaced in `candidate_manifest.json` and useful for evaluation/debugging

### 5.2 The five default strategies from the brief

| Strategy | camera_distance_shift | object_emphasis_bias | background_intensity_bias | color_grade_bias | typography_weight_bias | emotion_bias |
|---|---|---|---|---|---|---|
| Variant A — Faithful | 0 | none | none | none | none | none |
| Variant B — Higher emotion | 0 | none | none | none | none | amplify blueprint's existing `face_strategy`/`emotion` |
| Variant C — Cleaner composition | +1 (wider, more breathing room) | reduce secondary-object scale | simplify/soften | none | none | none |
| Variant D — Higher contrast | 0 | none | none | increase contrast/saturation | none | none |
| Variant E — Aggressive CTR | 0 | increase primary-object scale | none | increase contrast/saturation | bolder | amplify blueprint's existing `face_strategy`/`emotion` |

Variant A ("faithful") is the *identity transformation* — it is the exact behavior the pipeline has today, guaranteeing that enabling multi-candidate generation with the default pack never regresses the single-candidate case; it simply becomes "one of N" rather than "the only one."

### 5.3 Strategy packs as configuration, not code

A `StrategyPack` is an ordered `list[CandidateStrategy]`, defined as data (YAML/JSON, mirroring how `workflows/*.json` are already data, not code) under a new `data/strategy_packs/` directory, discovered and validated by a `StrategyPackLibrary` that mirrors `WorkflowLibrary`'s existing `discover()`/`load()`/`validate()` shape exactly. `config.py` gains `MODULE7_STRATEGY_PACK: str = "default_five"` (name, not path — same convention as `MODULE7_PROFILE`) and `MODULE7_STRATEGY_PACK_DIR: Path`. A pack's length **is** the run's `num_candidates` once resolved — see §4.1 — subject to the `MODULE7_MAX_CANDIDATES` ceiling in §11.

Custom packs (e.g. a 2-strategy "fast preview" pack, or a niche-specific pack for `gaming` vs `finance`) are added by dropping a new JSON file in `data/strategy_packs/` — no code change, following the exact extensibility model `workflows/*.json` already established for niches.

---

## 6. Candidate Data Models

All new models live in `modules/models.py` beside the existing Module 7 section, following its existing conventions (`frozen=True`, `field_validator` for non-empty strings, `Literal` for closed enums).

- **`CandidateStrategy`** (new, frozen) — the fields from §5.1, plus a `field_validator` that rejects an empty `name` and bounds every bias field to a declared numeric range (e.g. `-0.3..+0.3` for multiplier biases) so an invalid strategy pack fails at load time, not mid-run.
- **`StrategyPack`** (new, frozen) — `name: str`, `strategies: list[CandidateStrategy]`, `pack_version: str`. Mirrors `WorkflowTemplateRef`'s versioning convention.
- **`GenerationParameters`** (existing, extended) — adds `num_candidates: int = 1` (validated `1 <= n <= MODULE7_MAX_CANDIDATES`) and `strategy_pack: Optional[str] = None` (`None` preserves today's exact single-candidate behavior with no strategy applied — the true backward-compatible default). This is the **only** modification to an existing frozen model in this design, and it is purely additive with safe defaults; no existing `GenerationParameters(...)` construction site breaks.
- **`GeneratedAsset`** (existing, unchanged) — `candidate_index` already exists and already means exactly what Phase 4 needs.
- **`CandidateScore`** (existing, unchanged) — already sufficient; Phase 4 adds no new scoring dimension, because strategy does not change how quality is judged (§4.3).
- **`ImageGenerationResult`** (existing, unchanged) — `candidate_scores`/`selected_candidate_index` already carry everything the winner-selection story needs. Phase 4 adds a **new**, separate model for the richer per-candidate record written to `candidate_manifest.json`:
- **`CandidateManifestEntry`** (new, frozen) — `candidate_index: int`, `strategy_name: str`, `seed: int`, `workflow_hash: str`, `generation_parameters: GenerationParameters` (the per-candidate resolved copy), `qa_report: QualityAssuranceReport`, `face_match: FaceMatchResult`, `candidate_score: CandidateScore`, `stage_durations_seconds: dict[str, float]`, `output_path: str`.
- **`CandidateManifest`** (new, frozen) — `video_id: str`, `entries: list[CandidateManifestEntry]`, `winning_candidate_index: int`, `strategy_pack_name: Optional[str]`, `generated_at: str`. This is the model serialized to `candidate_manifest.json` (§10).
- **`GenerationRunMetadata`** (new, frozen) — run-level provenance, serialized to `generation_metadata.json` (§10): `video_id`, `profile_name`, `workflow_version`, `workflow_hash`, `conditioning_asset_hashes: dict[str, str]` (reusing the hash fields `GenerationBundle`/conditioning already compute), `model_versions: dict[str, str]` (checkpoint/restoration/upscaler identifiers already available on `GenerationProfile`), `num_candidates_requested`, `num_candidates_completed`, `total_duration_seconds`, `parallel_generation_used: bool`, `retry_summary: dict[str, int]`.

None of these new models replace or narrow an existing one; `CandidateManifest`/`GenerationRunMetadata` are strictly additive views over data the pipeline already computes today (QA reports, face-match results, stage durations) plus the new strategy identity.

---

## 7. Module 7 Extension

### 7.1 `StrategyPackResolver` (new, `modules/generation_components/strategy_pack_resolver.py`, mirroring the existing `generation_components/` package introduced in Phase 3)

Constructor-injected with a `StrategyPackLibrary` (default concrete implementation reading `data/strategy_packs/`). `resolve(requested_pack: Optional[str], max_candidates: int) -> list[CandidateStrategy]`:

- `requested_pack is None` → returns `[CandidateStrategy.faithful_default()]`, a single-element in-code fallback identical to today's behavior — no file I/O, no config dependency, guaranteeing that a `PromptPackage` with `strategy_pack=None` behaves byte-identically to the pipeline before Phase 4.
- Otherwise loads the named pack, truncates to `max_candidates` (logging a warning if the pack was longer), and returns its `strategies` list in file-declared order (deterministic — order is data, not a runtime sort).

### 7.2 `CandidateStrategyPlanner` (new, same package)

`derive_package(base_package: PromptPackage, blueprint: DesignBlueprint, strategy: CandidateStrategy, candidate_index: int) -> PromptPackage`. Pure function: given the same four inputs, always returns the same output. Internally it:

1. Copies `base_package` via `model_copy(update=...)` (the same immutable-update pattern the current loop already uses for seed).
2. Computes a bounded `camera_distance` per §4.2 and re-renders `composition_instructions` using the **same instruction-template mechanism `prompt_compiler.py` already uses** for Module 6 (Phase 4 calls into `prompt_compiler`'s existing template functions rather than inventing new prompt-string logic — no duplicated templating).
3. Applies `object_emphasis_bias` to a working copy of `blueprint.object_strategy` and re-renders `object_placement` the same way.
4. Applies `background_intensity_bias`/`color_grade_bias`/`typography_weight_bias` as bounded instruction-strength adjustments to `background_instructions`/`color_instructions`/`typography_instructions` respectively, via small, declarative phrase-append rules (e.g. append `", increased contrast and saturation"` capped at one applied phrase per dimension) — not free-text generation, keeping determinism and auditability.
5. Sets `generation_parameters.seed = base_package.generation_parameters.seed + candidate_index` — preserving the existing seed-increment convention exactly.
6. Leaves `positive_prompt`, `subject_instructions`, `lighting_instructions`' core content, `headline`-derived text, `negative_prompt`, `rendering_constraints`, and `safety_constraints` untouched, per §4.3.

### 7.3 `ImageGeneratorPipeline.run()` changes

Three surgical changes, no signature change:

1. After conditioning resolution, resolve `strategies = self.strategy_pack_resolver.resolve(package.generation_parameters.strategy_pack, MODULE7_MAX_CANDIDATES)`.
2. Replace `num_candidates = getattr(...)` with `num_candidates = len(strategies)`.
3. Inside the loop, replace the bare `cand_package = package.model_copy(update={"generation_parameters": ...})` with `cand_package = self.strategy_planner.derive_package(package, design_blueprint, strategies[cand_idx], cand_idx)`. This requires `run()` to also accept the `DesignBlueprint` it currently does not load — see §7.4.

`self.strategy_pack_resolver` and `self.strategy_planner` are new constructor parameters on `ImageGeneratorPipeline.__init__`, both optional with concrete defaults, following the exact pattern every other collaborator already uses.

### 7.4 Sourcing the `DesignBlueprint` inside Module 7

Module 7 currently never loads a `DesignBlueprint` — it only ever sees the already-compiled `PromptPackage`. Phase 4 adds an optional `design_blueprint: DesignBlueprint | None = None` parameter to `ImageGeneratorPipeline.run()` and `run_image_generation_pipeline()`, defaulted to `None`. `main.py` already holds the `design_blueprint` object in scope at the point it calls `_run_module7_generation()` (§ pipeline trace above) and is extended to pass it through — a one-line addition at the existing call site, not a new load path. If `design_blueprint is None` (e.g. a caller that only ever had a `PromptPackage`), `StrategyPackResolver` silently falls back to the single faithful strategy regardless of configured pack, since strategy bounds cannot be computed without it — graceful degradation, matching Phase 3's existing philosophy (§2, Phase 3 doc, goal 3).

---

## 8. Workflow Reuse

### 8.1 `WorkflowGraphCache` (new, scoped to one `ImageGeneratorPipeline.run()` call — never persisted across runs or across videos)

A small, non-collaborator-facing cache object, constructed once per `run()` call and passed to each loop iteration's `WorkflowBuilder.build()` call site (not into `WorkflowBuilder` itself, keeping `WorkflowBuilder.build()`'s signature and its unit tests untouched). It memoizes the **template-load + fragment-selection + fragment-assembly** portion of `build()` — the part keyed by `(workflow_ref, profile, conditioning)`, which is identical across every candidate in a run — while still calling `_slots()` and `_substitute()` fresh per candidate, since those are the only parts that legitimately vary (prompt text, seed).

Concretely: `WorkflowBuilder` gains one new, purely internal method, `build_base(profile, workflow_ref, conditioning, library) -> _BaseGraphMaterialization` (template dict + assembled fragments, pre-slot-substitution), and `build()` is refactored to call `build_base()` then apply `_slots()`/`_substitute()` — a pure internal decomposition of existing logic, not new behavior. The candidate loop's call site checks `WorkflowGraphCache` for a `(workflow_ref, profile, conditioning)` key before calling `build_base()`, reusing the cached materialization for every candidate in the run and only re-running `_slots()`/`_substitute()` per candidate.

This eliminates the redundant `WorkflowLibrary.load()` disk read and `graph_assembler.assemble()` fragment work identified in §0 as the real inefficiency, **without** touching `canonical_json_hash()`'s determinism guarantee — each candidate's `workflow_hash` is still computed from its own fully-substituted graph, so two candidates with different prompt text still get correctly different hashes, and the cache is invalidated automatically the moment `profile` changes (e.g. the existing VRAM-fallback path already picks a different `profile`, which naturally produces a new cache key).

### 8.2 What is explicitly *not* cached

- `ComfyUIClient.generate()` calls are never cached or deduplicated — each candidate is a genuinely distinct image generation and must hit ComfyUI.
- Per-candidate stages (`IdentityPreservationStage`, `FaceRestorationStage`, `BackgroundCompositor`, `UpscaleStage`, `QualityAssuranceStage`) operate on the raw output bytes of each candidate and cannot be shared across candidates by definition.
- Conditioning resolution (`ConditioningAssetResolver.resolve()`) was already run-scoped, not candidate-scoped, before Phase 4 (§0) — no change needed.

---

## 9. Cache Strategy

Phase 4 introduces exactly one new cache (`WorkflowGraphCache`, §8.1), deliberately scoped to a single `run()` invocation:

- **Lifetime:** process memory, one call to `ImageGeneratorPipeline.run()`. Never written to disk, never shared across videos or across pipeline runs — this avoids any staleness/invalidation design burden, since the cache dies with the call that created it.
- **Key:** `(workflow_ref.template_path, workflow_ref.workflow_version, profile.name, conditioning_hash)`, where `conditioning_hash` is derived the same way `GenerationBundle`/`CompositionWorkspace` already derive their own content hashes (reusing existing hashing utilities, e.g. `canonical_json_hash` already in `image_generator.py`) — no new hashing scheme.
- **Size bound:** implicitly bounded by `MODULE7_MAX_CANDIDATES` (§11) since at most one entry per distinct profile is created per run, and a run typically uses one profile unless the VRAM-fallback path forces a mid-run switch (at most two entries in that case).
- **Existing caches, unaffected:** `VRE_CACHE_ENABLED`, `ASSET_EXTRACTION_CACHE_ENABLED`, `DECISION_CACHE_ENABLED`, `COMPOSITION_CACHE_ENABLED`, `PLANNER_CACHE_ENABLED`, and `MODULE7_CAPABILITY_PROBE_CACHE_SECONDS` are all upstream-of or orthogonal-to Module 7's candidate loop and require no change. `StrategyPackLibrary`'s pack-loading is deliberately **not** cached beyond normal filesystem caching — strategy packs are small, rarely-changing config files read at most once per run, and adding a cross-run cache for them would reintroduce the staleness-invalidation complexity this design otherwise avoids.

---

## 10. Output Management

### 10.1 File layout (extends, does not replace, today's layout)

```
data/generated_thumbnails/{video_id}/
├── {video_id}.png                     # unchanged — winning candidate, existing name/location
├── {video_id}_manifest.json           # unchanged — existing ImageGenerationResult (ArtifactWriter)
├── candidate_01.png                   # NEW naming when MODULE7_SAVE_CANDIDATES=True
├── candidate_02.png
├── candidate_03.png
├── candidate_04.png
├── candidate_manifest.json            # NEW — CandidateManifest (§6)
└── generation_metadata.json           # NEW — GenerationRunMetadata (§6)
```

`candidate_NN` numbering is 1-indexed and zero-padded to `len(str(MODULE7_MAX_CANDIDATES))` digits (2 digits by default), replacing today's `candidate_{idx}_score_{score:.2f}.png` convention — the score moves into `candidate_manifest.json` where it belongs alongside the rest of that candidate's audit trail, keeping filenames stable across re-runs that produce slightly different scores for the same strategy (useful for diffing generated_thumbnails/ across runs, and for the Evaluation framework's existing file-based comparisons in `evaluation/`).

`{video_id}.png` and `{video_id}_manifest.json` — the two artifacts every existing downstream consumer (Evaluation framework, `main.py`'s return value, tests) already depends on — are untouched in name, location, and schema. Phase 4 is purely additive at the filesystem level.

### 10.2 `ArtifactWriter` extension

`ArtifactWriter` (existing class) gains two new methods, `write_candidate_manifest(manifest: CandidateManifest) -> Path` and `write_generation_metadata(metadata: GenerationRunMetadata) -> Path`, following the exact pattern of its existing `write_manifest()` (atomic tmp-file-then-replace write, same as `main.py`'s existing `_persist_generated_thumbnail` write pattern and `image_generator.py`'s own artifact writes). `write_manifest()` itself is unchanged.

### 10.3 What every candidate retains (per the brief)

Satisfied entirely by `CandidateManifestEntry` (§6): generation parameters (`generation_parameters`), workflow hash (`workflow_hash`), seed (embedded in `generation_parameters.seed`), strategy (`strategy_name`), timing (`stage_durations_seconds`), model versions (via `GenerationRunMetadata.model_versions`, run-scoped since model versions don't vary by candidate within a run), and conditioning assets (via `GenerationRunMetadata.conditioning_asset_hashes`, likewise run-scoped since conditioning is resolved once — §8.2).

---

## 11. Configuration

All new keys added to `modules/config.py`, following existing `MODULE7_*` naming and existing default-safe conventions (every new default preserves today's single-candidate behavior):

| Key | Default | Purpose |
|---|---|---|
| `MODULE7_MAX_CANDIDATES` | `1` | Hard ceiling on `num_candidates`; a `strategy_pack` longer than this is truncated with a logged warning (§7.1). Operators opt into >1 explicitly. |
| `MODULE7_STRATEGY_PACK` | `None` | Default `strategy_pack` value threaded into `GenerationParameters` when Module 6 compiles a package, if not overridden per-request. `None` preserves current behavior exactly. |
| `MODULE7_STRATEGY_PACK_DIR` | `data/strategy_packs/` | Discovery root for `StrategyPackLibrary`, mirroring `MODULE7_WORKFLOW_LIBRARY_DIR`. |
| `MODULE7_PARALLEL_CANDIDATES` | `False` | See §11.1. |
| `MODULE7_MAX_PARALLEL_CANDIDATES` | `2` | Concurrency cap when parallel generation is enabled. |
| `MODULE7_CANDIDATE_VRAM_BUDGET_GB` | `None` (falls back to `expected_vram_gb` of the selected profile) | Optional override for how much VRAM one in-flight candidate is assumed to consume, used to derive a safe parallelism ceiling (§11.1). |
| `MODULE7_CANDIDATE_TIMEOUT_SECONDS` | `COMFYUI_EXECUTION_TIMEOUT_SECONDS` (existing value, reused as default) | Per-candidate generation timeout; a candidate that exceeds it is treated as a failed candidate (excluded from ranking, logged), not a run-aborting error. |
| `MODULE7_CANDIDATE_RETRY_ATTEMPTS` | `0` | Number of full-candidate retries (distinct from the existing `MAX_IDENTITY_RETRIES`, which retries *within* a candidate for identity failure only) if a candidate errors before reaching the QA stage (e.g. a transient ComfyUI error unrelated to identity). |
| `MODULE7_SAVE_CANDIDATES` | `False` (existing key, unchanged default) | Now also gates the zero-padded candidate PNGs described in §10.1, in addition to its existing behavior. |
| `MODULE7_WORKFLOW_GRAPH_CACHE_ENABLED` | `True` | Escape hatch to disable §8.1's cache (e.g. for isolating a suspected caching bug during debugging), matching the existing `*_CACHE_ENABLED` convention used by every other module. |

### 11.1 Parallel generation — design, not default-on

`MODULE7_PARALLEL_CANDIDATES` governs whether the candidate loop in `ImageGeneratorPipeline.run()` dispatches candidates concurrently (via a bounded worker pool, e.g. `concurrent.futures.ThreadPoolExecutor` sized to `min(MODULE7_MAX_PARALLEL_CANDIDATES, num_candidates)`) or sequentially as it does today. This is additive and defaults to `False` for two concrete reasons grounded in the existing codebase, not caution for its own sake:

1. `ComfyUIClient` (Phase 2) is a single HTTP/WebSocket client against one local ComfyUI server process. Concurrent `generate()` calls contend for the same GPU and the same ComfyUI execution queue; Phase 2's `QueueTracker` (`docs/MODULE7_PHASE2_QUEUE_TRACKER_DESIGN.md`) already models one queue, not N. Parallelism here means concurrently *submitting and awaiting* jobs that ComfyUI itself will still serialize on the GPU — the benefit is overlapping the Python-side post-processing (identity check, restoration, compositing, upscale, QA) of one candidate with the ComfyUI-side generation of the next, not true simultaneous GPU work.
2. `expected_vram_gb` on `GenerationProfile` already models single-job VRAM headroom (`MODULE7_VRAM_HEADROOM_GB`). Enabling parallel candidates without a VRAM-aware ceiling risks the exact `VRAMExhaustedError` the existing fallback ladder (§0, `ImageGeneratorPipeline.run()`) already handles for the *single-job* case, but now compounded by concurrent jobs. `MODULE7_CANDIDATE_VRAM_BUDGET_GB` exists specifically to let an operator declare a safe per-candidate budget so `MODULE7_MAX_PARALLEL_CANDIDATES` can be derived/validated against available VRAM rather than guessed.

Given these constraints, Phase 4 ships parallel generation as a configuration option with conservative defaults (off, cap of 2) rather than the default execution mode — sequential-by-default is the safe, proven behavior; parallelism is an opt-in throughput optimization for operators who have profiled their own hardware.

---

## 12. Logging

No new logging framework or sink — Phase 4 emits through the existing Loguru sink already configured by `image_generator.py`'s `_configure_logger()` (rotating file sink at `MODULE7_LOG_PATH`, same format string used across every module). New log points, all at existing log levels used elsewhere in `image_generator.py`:

- `INFO`, on strategy pack resolution: `"Resolved strategy_pack={pack} → {n} candidate(s): {names}"`.
- `INFO`, per candidate start: `"Generating candidate idx={idx} strategy={name} seed={seed}"` (parallels the existing `CandidateRanker` winner log already at `INFO`).
- `DEBUG`, on workflow graph cache hit/miss: `"WorkflowGraphCache hit for key={key}"` / `"...miss, materializing base graph"` — mirrors the existing `DEBUG`-level verbosity used for per-node/per-fragment decisions in `WorkflowBuilder`.
- `WARNING`, on strategy-pack truncation (`MODULE7_MAX_CANDIDATES` exceeded) and on individual candidate timeout/failure (excluded from ranking) — matching the existing `WARNING` used for VRAM-fallback and identity-retry events.
- `ERROR`, only escalated to the existing `NoEligibleCandidateError` path (unchanged) when *every* candidate — across every strategy — fails hard gates or times out; a single candidate's failure is logged at `WARNING` and the run proceeds with the remaining candidates, consistent with the brief's spirit that multi-candidate generation should be *more* resilient than single-candidate, not equally fragile per-candidate.

---

## 13. Error Handling

Phase 4 introduces two new, narrowly-scoped exceptions in `module7_exceptions.py`, alongside the existing `Module7Error` hierarchy (`ComfyUIConnectionError`, `VRAMExhaustedError`, `NoEligibleCandidateError`, etc.):

- **`StrategyPackError(Module7Error)`** — raised by `StrategyPackLibrary`/`StrategyPackResolver` for a malformed or missing strategy pack file, or a strategy whose bias fields fall outside validated bounds (§6). Raised at `run()` start, before any generation work begins — fails fast, exactly like today's existing `PromptPackageInvalidError`/`WorkflowTemplateError` fail-fast behavior for other malformed inputs.
- **`CandidateGenerationTimeoutError(Module7Error)`** — raised internally when a candidate exceeds `MODULE7_CANDIDATE_TIMEOUT_SECONDS`; caught inside the loop (not propagated to the caller) and converted into a `WARNING`-logged, excluded candidate, exactly as a QA hard-gate failure is handled today (§12). The run only propagates a hard error via the existing `NoEligibleCandidateError` if zero candidates survive.

`MODULE7_CANDIDATE_RETRY_ATTEMPTS` (§11) governs whether a timed-out or errored candidate is retried in place (same strategy, incremented sub-seed) before being counted as excluded — reusing the exact retry-then-give-up shape `IdentityPreservationStage` already uses for identity failures (`MAX_IDENTITY_RETRIES`), rather than introducing a new retry idiom.

No change to the existing `VRAMExhaustedError` fallback ladder — it continues to operate exactly as today, per-candidate, with the sole addition that a profile downgrade forced mid-run invalidates the relevant `WorkflowGraphCache` entry (§8.1, automatic via cache key) rather than requiring explicit cache-busting logic.

---

## 14. Testing Strategy

Following the existing test-suite convention (one `tests/test_<module>.py` per module, plus a `tests/test_<engine>_components/` directory for component-level units, exactly as `tests/test_generation_components` already exists for Phase 3's collaborators):

- **`tests/test_generation_components/test_strategy_pack_resolver.py`** (new) — `strategy_pack=None` yields exactly the single faithful strategy; a named pack resolves its declared strategies in file order; truncation at `MODULE7_MAX_CANDIDATES` logs a warning and truncates deterministically (keeps the first N, does not reorder).
- **`tests/test_generation_components/test_candidate_strategy_planner.py`** (new) — for each of the five default strategies (§5.2), asserts the derived `PromptPackage` differs from the base package only in the fields that strategy is permitted to touch, and that `headline`/`text_position`/`positive_prompt`/`subject_instructions`/safety and rendering constraints are always byte-identical to the base package regardless of strategy (this is the direct test of the §4.2/§4.3 invariants). A dedicated test asserts the "faithful" strategy produces a package identical to today's pre-Phase-4 seed-increment-only copy, byte for byte.
- **`tests/test_image_generator.py`** (existing, extended) — new cases for `num_candidates > 1` end-to-end through `ImageGeneratorPipeline.run()` with a fake/stub `ComfyUIClient` (the existing test file already stubs this collaborator for the `num_candidates == 1` case; extended, not replaced), asserting: correct candidate count generated, `CandidateRanker` receives all of them, `candidate_manifest.json`/`generation_metadata.json` are written with the right entry count, and — critically — a regression test that `num_candidates=1`/`strategy_pack=None` produces an `ImageGenerationResult` identical in every field to the pre-Phase-4 baseline fixture already used by this test file.
- **`tests/test_workflow_library.py`** / a new `tests/test_generation_components/test_workflow_graph_cache.py` — cache-hit reuse across candidates sharing a profile; cache-miss/new-entry on a mid-run profile downgrade (VRAM fallback); confirms `workflow_hash` still differs correctly between two cache-sharing candidates whose prompt text differs (i.e., proves the cache never causes two genuinely different candidates to collapse to the same hash).
- **`tests/test_evaluation`** (existing) — extended with one new module validator case confirming the Evaluation framework can locate and parse `candidate_manifest.json`/`generation_metadata.json` when present, without requiring them (backward-compatible parsing — files absent for `num_candidates=1` runs where `MODULE7_SAVE_CANDIDATES=False`, matching current default output).
- **Determinism regression suite** — a repository-wide invariant test (extending the pattern already used by `canonical_json_hash`/`generation_hash` tests) asserting that running the full candidate-derivation pipeline twice with identical inputs (`package`, `blueprint`, `strategy`, `candidate_index`) produces byte-identical `PromptPackage` and `workflow_hash` output — the same determinism guarantee Phase 3 already tests for its own conditioning-fragment injection.

No existing test file needs an assertion changed for the default (`num_candidates=1`, `strategy_pack=None`) path — every new behavior is reached only through new, explicit configuration.

---

## 15. Migration Strategy

Phase 4 ships fully backward-compatible and **off by default**, so migration is a matter of sequencing rollout, not data migration (no existing persisted artifact — `PromptPackage`, `ImageGenerationResult`, or otherwise — needs to be rewritten or re-validated; new `GenerationParameters` fields have safe defaults that Pydantic applies transparently to any already-serialized package missing them).

1. **Land the schema additions** (§6: `num_candidates`, `strategy_pack` on `GenerationParameters`; new `CandidateStrategy`/`StrategyPack`/`CandidateManifest`/`GenerationRunMetadata` models) with defaults that reproduce today's behavior exactly. Deployable with zero behavior change.
2. **Land `StrategyPackResolver`/`CandidateStrategyPlanner`/`WorkflowGraphCache`** wired into `ImageGeneratorPipeline`, still gated by `MODULE7_MAX_CANDIDATES=1` default. Deployable with zero behavior change — the new code paths exist but `num_candidates` still can never exceed 1 by default config.
3. **Land output-management additions** (`ArtifactWriter.write_candidate_manifest`/`write_generation_metadata`, zero-padded filenames), gated by the existing `MODULE7_SAVE_CANDIDATES` flag remaining `False` by default. Deployable with zero behavior change for default-configured deployments.
4. **Ship the default strategy pack** (`data/strategy_packs/default_five.json`, §5.2) as repository data, unreferenced until `MODULE7_STRATEGY_PACK` is set. Deployable with zero behavior change.
5. **Opt-in rollout**: operators raise `MODULE7_MAX_CANDIDATES` and set `MODULE7_STRATEGY_PACK="default_five"` in a specific environment (e.g. a staging profile or a single niche via `MODULE7_NICHE_WORKFLOW_MAP`-style per-niche override, which is a natural follow-on config extension but not required for v1). Evaluation framework results and generation-time/VRAM metrics from `MetricsCollector` (already collecting `num_candidates_requested` — an existing field, previously always `1`) are compared against the single-candidate baseline before wider rollout.
6. **Optional cleanup** (out of scope for this document, but noted per §0): remove the dead `_persist_generated_thumbnail`/hardcoded-`candidate_index=0` path in `main.py`, since it is superseded by `run_image_generation_pipeline()` and never exercised. This is a housekeeping change independent of Phase 4's functional additions.

At no point does an existing consumer of `{video_id}.png` or `{video_id}_manifest.json` need to change; both remain schema- and path-identical throughout every phase above.

---

## 16. Phase-by-Phase Implementation Plan

**Phase 4.1 — Data model foundation**
Add `num_candidates`/`strategy_pack` to `GenerationParameters`; add `CandidateStrategy`, `StrategyPack`, `CandidateManifestEntry`, `CandidateManifest`, `GenerationRunMetadata` to `models.py`; add `StrategyPackError`/`CandidateGenerationTimeoutError` to `module7_exceptions.py`. No behavior change. Unit tests for model validation only.

**Phase 4.2 — Strategy resolution and planning**
New `modules/generation_components/strategy_pack_resolver.py` and `candidate_strategy_planner.py`; `StrategyPackLibrary` mirroring `WorkflowLibrary`; ship `data/strategy_packs/default_five.json` (§5.2) plus a JSON-schema-style validation path reused from `WorkflowLibrary.validate()`'s pattern. Unit tests per §14 (strategy planner determinism and invariant tests). Not yet wired into `ImageGeneratorPipeline`.

**Phase 4.3 — Workflow graph cache**
Refactor `WorkflowBuilder.build()` into `build_base()` + slot-substitution (§8.1), purely internal, with a regression test proving `build()`'s external output is unchanged for every existing caller. Add `WorkflowGraphCache`, unwired. Unit tests per §14 (cache-hit/miss/downgrade-invalidation).

**Phase 4.4 — Pipeline wiring**
Wire `StrategyPackResolver`/`CandidateStrategyPlanner`/`WorkflowGraphCache` into `ImageGeneratorPipeline.__init__`/`run()` (§7.3), add the optional `design_blueprint` parameter to `run()` and `run_image_generation_pipeline()`, and thread it from `main.py`'s existing in-scope `design_blueprint` object at the existing `_run_module7_generation()` call site (§7.4). End-to-end tests per §14 with `num_candidates > 1` via stubbed `ComfyUIClient`. `MODULE7_MAX_CANDIDATES` still defaults to `1` in `config.py` — merged dark.

**Phase 4.5 — Output management**
`ArtifactWriter.write_candidate_manifest()`/`write_generation_metadata()`; zero-padded candidate filenames; Evaluation-framework backward-compatible parsing of the two new JSON files. Tests per §14.

**Phase 4.6 — Configuration and operational rollout**
Add remaining `config.py` keys (§11: max candidates, parallel generation, VRAM budget, timeout, retry, cache toggle) with production-safe defaults. Enable `MODULE7_MAX_CANDIDATES > 1` and a non-`None` `MODULE7_STRATEGY_PACK` in a single controlled environment; compare `MetricsCollector` output and Evaluation-framework scores against the single-candidate baseline (§15, step 5).

**Phase 4.7 — Parallel generation (optional, separately gated)**
Implement the bounded worker pool described in §11.1 behind `MODULE7_PARALLEL_CANDIDATES=False` default; VRAM-budget-aware concurrency sizing; load-tested on representative hardware before any default-on consideration. This phase is intentionally decoupled from 4.1–4.6 and can ship later, or not at all, without blocking the rest of the architecture — sequential multi-candidate generation alone already satisfies the brief's core goal (multiple high-quality candidates instead of one).
