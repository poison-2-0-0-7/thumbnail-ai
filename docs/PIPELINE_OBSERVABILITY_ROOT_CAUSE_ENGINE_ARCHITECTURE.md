# Pipeline Observability & Root Cause Engine — Architecture

**thumbnail-ai**
**Status:** Design only. No implementation code, no pseudocode, no tests.
**Author role:** Lead Software Architect
**Repo studied:** `poison-2-0-0-7/thumbnail-ai` @ `main` (live clone, `git log` verified — 34 commits, most recent: `e37afbb feat: integrate Module 8 asset extraction and Module 9 decision engine`)

---

# PART A — REPOSITORY ANALYSIS

*(Required before any architecture, per instructions. Every claim below is sourced from a specific file, read in full, not inferred.)*

## A.1 Current pipeline diagram (verified against `main.py`, not assumed)

The task brief's stated pipeline order is close but not exact. `main.py::_run_pipeline_creators` (lines 214–553) was read in full; the **actual, live** order is:

```
1  CSV Reader            (csv_reader.py)                     — always runs
2  YouTube Metadata      (youtube_metadata.py)                — always runs
3  Thumbnail Downloader  (thumbnail_downloader.py)             — always runs
4  Thumbnail Intelligence(thumbnail_intelligence.py)            — always runs
8  Asset Extraction      (asset_extraction_engine.py)          — flag: ASSET_EXTRACTION_ENABLED (default False)
5  Redesign Spec         (redesign_spec_engine.py)             — always runs
5.5 Design Blueprint     (design_blueprint_engine.py)          — always runs
6  Prompt Compiler       (prompt_compiler.py)                  — always runs
9  Decision Engine       (decision_engine.py)                  — flag: DECISION_ENGINE_ENABLED (default False)
10 Asset Composer        (composition_engine.py)               — always runs (consumes Module 9's output if present)
10.5 Thumbnail Planner   (thumbnail_planner.py)                — flag: THUMBNAIL_PLANNER_ENABLED (default False)
7  Image Generation      (image_generator.py)                  — always runs
```

Two corrections to the brief's diagram:

1. **Module 6.5 (Visual Reference Engine) has no call site in `main.py` at all.** It is invoked *internally* by `AssetComposer.prepare_generation_workspace()` (Module 10), confirmed by `composition_engine.py` importing and instantiating `VisualReferenceEngine` directly — it is not a top-level pipeline stage, it is Module 10's own dependency. The brief's diagram lists it as a peer stage; the code does not.
2. **Module 8 runs immediately after Module 4, before Module 5** — not where the brief's diagram places it (after 6.5, before 9). This ordering exists because Module 8 depends only on Module 4's output and Module 9 needs Module 8's manifest, so it is scheduled as early as its dependencies allow.

## A.2 Current Module 7 implementation summary

The brief's own pipeline diagram labels the final stage **"Module 7 Editing Engine (V2)."** This label describes a *design*, not the current code. Verified directly against `modules/image_generator.py` and every file in `workflows/`:

- Every one of the eleven base workflow templates (`workflows/general.json`, `gaming.json`, `tech.json`, `finance.json`, `fitness.json`, `podcast.json`, `vlog.json`, `documentary.json`, `education.json`, `lifestyle.json`, `reaction.json`) builds its latent from node `EmptyLatentImage`, and every one of them runs `KSampler` at `"denoise": 1.0`. No `VAEEncode` node exists in any base template. Confirmed by `grep -n "VAEEncode|EmptyLatentImage|denoise" workflows/*.json` — eleven `EmptyLatentImage` hits, eleven `"denoise": 1.0` hits, zero `VAEEncode` hits.
- `BackgroundCompositor` (`image_generator.py:690-714`), whose docstring reads *"Composite preserved subject over newly generated background,"* does not composite anything: its `composite()` method opens the already-generated image and writes it back out unchanged (`Image.open(generated_image_path) ... bg_img.save(temp_target, ...)`) whenever a source thumbnail exists. It performs no mask lookup, no paste-back, no pixel operation of any kind beyond a re-save. This is a **verified, concrete instance** of the failure category the brief calls "renderer ignored edit plan" — a class exists that names the correct responsibility but doesn't implement it.
- Four of the six Tier-1 QA scoring functions (`_calculate_text_safe_zone_score`, `_calculate_object_preservation_score`, `_calculate_color_compliance_score`, `_calculate_composition_score`, all in `image_generator.py:592-609`) are unconditional `return 1.0` stubs.
- `WorkflowGraphAssembler.assemble()` (`generation_components/workflow_graph_assembler.py`) correctly merges ControlNet/IPAdapter/masking fragments into the base graph via a namespaced-node, `ATTACHMENT_PREVIOUS`-sentinel mechanism — but it only logs a **count** of fragments attached (`"Assembled workflow graph with {n_fragments} fragment(s) attached"`), never *which* fragments, at what strength, for which element. This is the exact data a root-cause engine needs and the exact data that is currently discarded after being logged as a bare integer.
- `DecisionManifest` (Module 9) and `GenerationPlan` (Module 10.5) are accepted as optional parameters by `ImageGeneratorPipeline.run()` and used to select ControlNet/IPAdapter fragments and prompt-adjacent slot values — but nothing in the graph or in any persisted artifact records whether a given `KEEP` decision's element actually ended up outside every sampling mask, or inside one. The decision is read; whether it was *honored* is never checked or recorded.
- A separate design document, `docs/MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md`, already exists in this repository specifying exactly the image-to-image/inpainting architecture that would fix the above — but it is a design document only; none of it is implemented in `modules/image_generator.py` or `workflows/` as of the commit reviewed for this document. This document treats that gap as a **live, current-state fact**, and — per §A.10 and the Diagnostic Engine section — designs the new observability subsystem to work correctly against **today's** txt2img-based Module 7, while remaining valid without modification once that other document's V2 is implemented (a required non-functional goal in the brief: "Support future renderers").

## A.3 Current logging system

`modules/config.py` defines one `MODULE{N}_LOG_PATH` constant per module (`MODULE1_LOG_PATH` through `MODULE10_5_LOG_PATH`, plus `EVAL_LOG_PATH`, plus `COMFYUI_PROCESS_LOG_PATH`) — **twelve separate log files under `logs/`, one per module, each configured independently inside that module's own file** via a repeated `logger.add(MODULE{N}_LOG_PATH, rotation="10 MB", retention="7 days", level="INFO", enqueue=True)` call. This is a real, consistent, well-implemented convention (Loguru, async-safe `enqueue=True`, sane rotation/retention) — but it is **twelve independent streams with no cross-file correlation**. A `video_id` appears in most log lines (confirmed by `grep -n "video_id" modules/*.py` returning matches in every stage's logging calls) but there is no single command, index, or artifact that assembles "everything that happened for `video_id=X`" across all twelve files. This is precisely the manual, slow, inconsistent process the brief describes developers doing today.

## A.4 Current artifact generation

Every module writes its own typed, frozen-Pydantic-model artifact to its own directory, via the project-wide atomic temp-file-then-`Path.replace()` pattern, keyed by `video_id`:

| Module | Directory | Artifact |
|---|---|---|
| 2 | (in-memory, not persisted separately) | `VideoMetadata` |
| 3 | `data/thumbnails/` | `{video_id}.jpg` |
| 4 | `data/analysis/` | `{video_id}.json` (`ThumbnailIntelligence`) |
| 8 | `data/asset_extraction/{video_id}/` | `asset_manifest.json` (`AssetExtractionManifest`) |
| 5 | `data/redesign_specs/` | `{video_id}.json` (`RedesignSpecification`) |
| 5.5 | `data/design_blueprints/` | `{video_id}.json` (`DesignBlueprint`) |
| 6 | `data/prompt_packages/` | `{video_id}.json` (`PromptPackage`) |
| 9 | `data/decisions/{video_id}/` | `decision_manifest.json` (`DecisionManifest`) |
| 10 | `data/composition_workspaces/{video_id}/` | `CompositionWorkspace` + `GenerationBundle` (masks, crops, layers) |
| 10.5 | `data/strategy_packs/` | `{video_id}.json` (`GenerationPlan`) |
| 7 | `data/generated_thumbnails/{video_id}/` | `{video_id}.png` + manifest (`ImageGenerationResult`) |
| 7 (metrics) | `logs/module7_metrics.jsonl` | append-only `GenerationMetrics` per attempt |
| Evaluation | `data/evaluation/runs/{run_id}/` | `run_manifest.json`, `module_results/{video_id}.json`, `quality_reports/{video_id}.json` |

Every artifact with a hash field (`ImageGenerationResult.generation_hash`, `GenerationPlan.workspace_hash`/`prompt_package_hash`, etc.) uses `canonical_json_hash()`/SHA-256, validated via a repeated `field_validator` pattern — a real, working reproducibility chain, confirmed in `image_generator.py:86-97` and referenced consistently across `docs/MODULE_10_5...`'s data model section.

## A.5 Existing debug capabilities

Beyond the twelve raw log files (§A.3) and manually opening JSON artifacts by hand (§A.4), the only structured debug capability that exists is `evaluation/module_validators/` (real, implemented code — nine validator files: `csv_reader_validator.py`, `youtube_metadata_validator.py`, `thumbnail_downloader_validator.py`, `thumbnail_intelligence_validator.py`, `redesign_spec_validator.py`, `design_blueprint_validator.py`, `prompt_compiler_validator.py`, `asset_composer_validator.py`, `module7_validator.py`). Each performs **schema and invariant validation** of one module's persisted artifact — "is this JSON well-formed and internally consistent." **None of these validators exist for Module 8 (asset extraction), Module 9 (decision engine), or Module 10.5 (thumbnail planner)** — a real, verified gap in the existing evaluation framework itself (three of twelve live pipeline stages have no validator). Schema validity is a different and much narrower question than "why did this specific element end up wrong in the final image" — a `DecisionManifest` can be perfectly schema-valid and still be silently ignored by Module 10/7 (as `BackgroundCompositor`, §A.2, demonstrates it currently is).

## A.6 Existing report generation

`evaluation/reporting/` is real, implemented code: `report_builder.py` assembles a `PipelineRunReport`, and `report_renderer.py` implements three `IReportRenderer`s (JSON — canonical, is the manifest; Markdown — human-readable table; HTML — styled). These operate at the **batch/run level** — "how did this whole CSV run of N creators do, and is it better or worse than last time" — not at the **single-video, single-generation, causal-explanation** level the brief is asking for. `PipelineRunReport` does contain a `quality_reports: dict[str, QualityEvaluationReport]` keyed by `video_id`, so per-video *quality scores* already exist in a reportable form — but not per-video *causal trace* or *root-cause findings*.

## A.7 Existing manifests (schema/hash conventions already established, reused rather than reinvented — see Part B)

Confirmed, exact field names, all frozen Pydantic (`ConfigDict(frozen=True)`), all with `status: Literal["success","partial","error", ...]`, `*_at` ISO-8601 timestamp fields, and (where applicable) SHA-256 hash fields validated by a repeated `field_validator`:

`VideoMetadata`, `ThumbnailData`, `ThumbnailIntelligence`, `AssetExtractionManifest` (`PersonAsset`, `ObjectAsset`, `SceneAsset`, `TypographyAsset`, `CompositionAsset`, `VisualPropertiesAsset`), `RedesignSpecification`, `DesignBlueprint`, `PromptPackage` (nested `GenerationParameters`, `QualityParameters`, `ModelSettings`), `DecisionManifest` (`ResolvedDecision`, decision types `KEEP`/`REMOVE`/`REPLACE`/`ENHANCE`/`ADD` — `models.py:1384-1386`), `CompositionWorkspace`/`GenerationBundle`, `GenerationPlan` (`FaceStrategy`, `BackgroundStrategy`, `PlanConditioningAsset`), `ImageGenerationResult` (`GeneratedAsset`, `FaceMatchResult`, `QualityAssuranceReport`, `CandidateScore`, `GenerationMetrics`), and the evaluation framework's own `DimensionScore`, `QualityEvaluationReport`, `ModuleValidationResult`, `PipelineRunReport`.

## A.8 Current strengths

- A genuinely consistent, project-wide architectural discipline: frozen Pydantic models, atomic writes, typed exception hierarchies per module, `ABC` component interfaces, Loguru with `enqueue=True`, feature-flag-gated additive rollout (`ASSET_EXTRACTION_ENABLED`, `DECISION_ENGINE_ENABLED`, `THUMBNAIL_PLANNER_ENABLED`). This discipline is what makes a non-invasive observability layer *possible* — every stage's input and output is already a typed, persisted, hashable artifact at a predictable path.
- A real, working reproducibility/hash-chaining contract already exists end-to-end (source artifact hash → derived artifact hash), which the new system can read and re-expose, not invent.
- PVQEF (`evaluation/`) already solves batch-level quality scoring and run-over-run regression detection well, and already reuses Module 7's own inline scores rather than duplicating them (`evaluation/quality/inline_qa_scorer.py`) — the precedent this document follows for its own relationship to PVQEF (§B.11).

## A.9 Current weaknesses (the ones this document exists to fix)

1. **No cross-module, per-video trace exists anywhere.** Twelve log files, twelve-plus artifact directories, zero index that links them by `video_id`.
2. **No visual trace / storyboard artifact** showing the thumbnail's journey through the pipeline.
3. **No rule-based root-cause engine.** PVQEF's validators answer "is this artifact well-formed"; nothing answers "why did the *renderer* produce this specific pixel result," which requires reasoning across artifacts, not validating one at a time.
4. **No record of the renderer's own execution choices** — which fragments were attached (only a count is logged, §A.2), what `denoise`/latent-source mode was used (always noise today, never recorded as a queryable field), what ControlNet/IPAdapter strengths were actually sent to ComfyUI for a given run.
5. **`BackgroundCompositor`'s no-op behavior (§A.2) is itself currently undetectable** — it logs a success message identical in shape to what a real compositing pass would log. A root-cause engine needs to be able to catch exactly this class of "ran without error, but didn't do what its own name claims" defect.
6. **Three of twelve live stages (Module 8, 9, 10.5) have no artifact validator** in the existing evaluation framework (§A.5).
7. **No timeline view** correlating stage durations, dependencies, and errors/warnings in one place — durations exist only inside Module 7's own `module7_metrics.jsonl` and (if PVQEF is run) `performance_profiler.py`'s per-stage timings; Modules 1–6/8/9/10/10.5 record no per-stage duration at all outside log-timestamp deltas.

## A.10 Exact integration points

This document's new subsystem needs exactly **one** touchpoint that is not purely read-only, identified precisely and justified against the "do not change existing APIs" constraint in §B.3/§B.5. Every other integration point is a read of an already-existing, already-documented, already-stable artifact or log file.

---

# PART B — ARCHITECTURE

## 1. Executive Summary

Every module in this pipeline already produces a rich, typed, persisted artifact. The problem is not missing data — the problem is that the data is scattered across twelve directories and twelve log files with no correlation key applied, no causal reasoning layered on top, and (per §A.2/§A.9.4–5) a small number of specific renderer-level facts that are computed transiently but never recorded at all. This document specifies the **Pipeline Observability & Root Cause Engine**, referred to throughout as **PORCE** — a new, additive, read-mostly subsystem, sibling to `modules/` and `evaluation/`, whose job is to answer, for any single `video_id`, mechanically: *what happened, in what order, using what inputs, producing what outputs, honoring or violating which decisions, and why did the final thumbnail turn out the way it did.*

PORCE does not replace PVQEF (`evaluation/`) — it consumes PVQEF's per-video `QualityEvaluationReport` as one input among several (§11) and adds the two capabilities PVQEF was never designed to provide: a full causal **trace** linking every module's artifacts together, and a deterministic **rule engine** that turns "these two facts are inconsistent" into a structured, severity-ranked, evidence-linked finding.

## 2. Problem Analysis

The pipeline's own logs and artifacts already contain the answer to "why did this thumbnail turn out wrong" in the overwhelming majority of cases — §A.2's `BackgroundCompositor` finding and §A.2's "denoise always 1.0" finding were both discovered in this repository review using nothing but `grep` and direct file reads, no new tooling. The actual problem is **retrieval and correlation cost**, not missing signal: an engineer today must know to check twelve files, in the right order, and manually hold enough of the pipeline's own architecture in their head to notice that `BackgroundCompositor`'s log line doesn't imply what it appears to imply. PORCE's job is to make that inference mechanical, repeatable, and instant per video, instead of expert-dependent and slow.

## 3. Architectural Goals

1. **Zero modification of Modules 1–6, 8, 9, 10, 10.5, and PVQEF.** Read their persisted artifacts and log files only.
2. **One narrow, precisely-scoped, additive integration point into Module 7** (§B.5), following the exact precedent `MetricsCollector` already set inside `ImageGeneratorPipeline` — a new internal call, zero signature change, zero behavior change to existing callers.
3. **Explain causally, not just report status.** A finding names a root cause and points at the evidence, not just a pass/fail.
4. **Deterministic-first.** Every rule in §10 is a pure function over already-typed data — no LLM call, no new ML model, matching the "deterministic compiler" ethos already established project-wide.
5. **Correct against today's Module 7, valid against tomorrow's.** Rules that reason about `latent_source`/`edit_mask_honored` (§9) degrade gracefully to "not applicable" facts on the current txt2img renderer rather than false-failing, and activate meaningfully once `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` ships.
6. **Additive storage only** — a new `data/observability/` tree, never touching any existing `data/*` directory.
7. **Extensible without redesign** — a new quality dimension, root-cause rule, or module validator is one new class behind an existing interface (mirrors PVQEF's own Design Goal 7, intentionally, for consistency across the two sibling subsystems).

## 4. System Placement

```
thumbnail-ai/
├── modules/            (Modules 1–10.5 — untouched, except §B.5's one hook)
├── evaluation/          (PVQEF — untouched, consumed read-only, §11)
├── observability/        (PORCE — new, this document)
├── data/
│   ├── analysis/ ... generated_thumbnails/   (untouched, read-only inputs)
│   ├── evaluation/                            (untouched, read-only input)
│   └── observability/                         (new, PORCE's own output tree)
├── logs/
│   ├── module1.log ... module10_5.log, evaluation.log   (untouched, read-only inputs)
│   └── module7_metrics.jsonl                             (untouched, read-only input)
└── docs/
    └── PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md   (this document)
```

`observability/` depends on `modules/` (typed artifact classes, for parsing) and optionally on `evaluation/` (for §11) — the dependency arrow points one way, exactly as PVQEF already established for its own relationship to `modules/` (`docs/PIPELINE_VALIDATION...` §3).

## 5. Component Architecture

```
observability/
├── __init__.py
├── cli.py                          # python -m observability.cli trace|explain|report <video_id>
├── config.py                       # OBS_* constants, mirrors evaluation/config.py's style
├── observability_exceptions.py     # PORCEError hierarchy
├── trace/
│   ├── __init__.py
│   ├── interfaces.py               # IArtifactCollector (ABC)
│   ├── artifact_index_builder.py   # walks known DEFAULT_{X}_DIR paths for one video_id
│   ├── log_correlator.py           # greps the twelve MODULE{N}_LOG_PATH files for one video_id,
│   │                                #  parses Loguru's structured line format, orders by timestamp
│   ├── generation_trace_reader.py  # reads the new GenerationTraceRecord (§B.5) when present
│   └── trace_assembler.py          # merges the above into one PipelineTrace (§8)
├── diagnostics/
│   ├── __init__.py
│   ├── interfaces.py               # IDiagnosticRule (ABC)
│   ├── rules/                      # one file per rule family (§10.2)
│   │   ├── latent_initialization_rules.py
│   │   ├── conditioning_rules.py
│   │   ├── decision_honoring_rules.py
│   │   ├── asset_provenance_rules.py
│   │   ├── prompt_consistency_rules.py
│   │   └── composition_rules.py
│   └── rule_engine.py              # runs every registered IDiagnosticRule, collects Findings
├── quality/
│   ├── __init__.py
│   └── quality_bridge.py           # reads PVQEF's QualityEvaluationReport (§11) — no new scoring code
├── visual_trace/
│   ├── __init__.py
│   └── storyboard_renderer.py      # composes the Section-8 visual trace image/HTML
├── reporting/
│   ├── __init__.py
│   ├── interfaces.py               # IObservabilityReportRenderer (ABC)
│   ├── human_report_renderer.py    # Markdown/HTML per-video explanation
│   ├── machine_report_renderer.py  # canonical JSON (RootCauseReport verbatim)
│   └── timeline_renderer.py        # §12
└── timeline/
    ├── __init__.py
    └── timeline_builder.py         # merges trace_assembler's ordering + log_correlator's timestamps

data/observability/
├── traces/{video_id}/
│   ├── pipeline_trace.json          # PipelineTrace (§8)
│   ├── artifact_index.json          # ArtifactIndex (§8.1)
│   ├── timeline.json                # Timeline (§12)
│   ├── storyboard.html              # visual trace (§8.2)
│   └── root_cause_report.json       # RootCauseReport (§10.3) — canonical
├── reports/{video_id}/
│   ├── report.md
│   └── report.html
└── generation_traces/{video_id}/
    └── generation_trace_record.json # written by §B.5's one hook, read by generation_trace_reader.py
```

## 6. Data Flow

```
video_id
   │
   ├──▶ ArtifactIndexBuilder ──▶ walks DEFAULT_ANALYSIS_DIR, DEFAULT_ASSET_EXTRACTION_DIR,
   │                              DEFAULT_REDESIGN_SPEC_DIR, DEFAULT_DESIGN_BLUEPRINT_DIR,
   │                              DEFAULT_PROMPT_PACKAGE_DIR, DEFAULT_DECISION_DIR,
   │                              composition_workspaces/, DEFAULT_STRATEGY_PACK_DIR (if exists),
   │                              generated_thumbnails/{video_id}/  — for each, records path + sha256
   │                              + whether the file exists at all (a missing file is itself a fact,
   │                              §9.2's "asset extraction missing" rule reads exactly this)
   │                                    │
   ├──▶ LogCorrelator ──────────▶ greps all twelve MODULE{N}_LOG_PATH files + module7_metrics.jsonl
   │                              for lines containing video_id, parses Loguru's line format
   │                              (timestamp, level, module, message), sorts chronologically
   │                                    │
   ├──▶ GenerationTraceReader ──▶ reads data/observability/generation_traces/{video_id}/... (§B.5)
   │                              if present; records absence (not an error) if the video predates
   │                              this document's rollout or Module 7 skipped generation entirely
   │                                    │
   │                                    ▼
   │                         TraceAssembler.assemble() ──▶ PipelineTrace (§8)
   │                                    │
   │                     ┌──────────────┼───────────────────┐
   │                     ▼              ▼                   ▼
   │              RuleEngine    QualityBridge        StoryboardRenderer
   │           (§10, reads    (§11, reads PVQEF's   (§8.2, composes the
   │            PipelineTrace) QualityEvaluationReport) visual trace)
   │                     │              │                   │
   │                     └──────────────┴───────────────────┘
   │                                    ▼
   │                        RootCauseReport (§10.3) ──▶ ReportRenderers ──▶ report.md / report.html /
   │                                                                          RootCauseReport JSON (canonical)
   ▼
data/observability/traces/{video_id}/  (all of the above, persisted)
```

Every arrow above is a **read** except the one narrow write inside Module 7 (§B.5) and PORCE's own writes into `data/observability/` — no existing `data/*` directory is ever written to by this system.

## 7. Artifact Storage

Follows the exact `DEFAULT_{X}_DIR` / atomic-write / `video_id`-keyed convention already used by every module (§A.4), added to `modules/config.py` as new, additive constants only (`OBS_TRACES_DIR`, `OBS_REPORTS_DIR`, `OBS_GENERATION_TRACES_DIR`), mirroring exactly how PVQEF added `EVAL_*` constants to the same file without touching any existing constant. All PORCE artifacts are frozen Pydantic models, written temp-file-then-`Path.replace()`, matching `main.py::_persist_generated_thumbnail`'s existing pattern.

## 8. Trace Model

### 8.1 `PipelineTrace` / `ArtifactIndex`

```
ArtifactRef:      module: str, artifact_type: str, path: Optional[str], exists: bool,
                   sha256: Optional[str], size_bytes: Optional[int]

ArtifactIndex:     video_id: str, refs: list[ArtifactRef], built_at: str

ModuleTraceEntry:  module: str, stage_order: int, status: Literal["success","partial","error","skipped","not_run"],
                   started_at: Optional[str], completed_at: Optional[str], duration_seconds: Optional[float],
                   # duration recovered from module7_metrics.jsonl when module == "module7";
                   # for all other modules, recovered as a best-effort delta between this module's
                   # first and last log line for the video_id (LogCorrelator, §6) — explicitly labeled
                   # `duration_source: Literal["exact","log_derived","unavailable"]` so a report never
                   # silently presents an approximation as measured fact
                   inputs: list[ArtifactRef], outputs: list[ArtifactRef],
                   config_snapshot: dict[str, Any],     # the relevant MODULE{N}_* / *_ENABLED flags
                                                          # active for this run, read from modules/config.py
                                                          # at trace-build time
                   log_lines: list[LogLineRef],          # pointers (file + line number), not copies —
                                                          # avoids duplicating twelve logs' content into
                                                          # every trace
                   errors: list[str], warnings: list[str]

PipelineTrace:     video_id: str, modules: list[ModuleTraceEntry],  # ordered per §A.1's verified order
                   artifact_index: ArtifactIndex, generation_trace: Optional[GenerationTraceRecord],
                   overall_status: Literal["success","partial","error"], assembled_at: str
```

`ModuleTraceEntry` for a flag-gated module (8/9/10.5) that didn't run has `status: "not_run"` (not `"skipped"`, which is reserved for "attempted but the per-creator loop chose to `continue`") — this distinction matters directly for the rule engine (§10.2's "asset extraction missing" rule must not fire against a video that never had `ASSET_EXTRACTION_ENABLED=True` to begin with; that's expected configuration, not a failure).

### 8.2 Visual trace (`storyboard.html`)

One HTML artifact per video, `StoryboardRenderer` (HTML, not a new image-generation dependency — reuses Pillow, already a project dependency, only to thumbnail/resize existing images, never to synthesize new ones) laying out, in the exact order the brief specifies:

```
Original Thumbnail → Module 4 Analysis (key fields: objects, faces, composition,
color) → Module 5 Copy (headline text) → Module 5.5 Layout (blueprint's zone map,
rendered as bounding boxes over a copy of the source) → Module 6 Prompt (positive/
negative prompt text) → Module 8 Assets (thumbnail crops of each extracted person/
object/typography asset) → Module 9 Decisions (a table: element → decision type)
→ Module 10 Composition (layer masks, rendered as translucent overlays) → Module
10.5 Plan (face/background strategy strings) → Module 7 Inputs (the
GenerationTraceRecord's fragment list, denoise/latent-source mode, seed) →
Generated Thumbnail
```

Each stage's panel links to that stage's exact `ArtifactRef` (§8.1) so an engineer can click through to the raw JSON, never forcing the storyboard to be the only view. Any stage with `status: "not_run"` or `"error"` renders as a visibly distinct (grayed-out / red-bordered) panel rather than a silent gap — the storyboard's own layout is itself a first, glanceable root-cause signal before the rule engine's text findings are even read.

## 9. Diagnostic Engine

### 9.1 `GenerationTraceRecord` — the one new data point Module 7 doesn't currently produce

Per §A.9.4/§A.9.5, the renderer's own execution choices are not currently recorded anywhere. This is fixed with **one new, additive, optional model and one new, additive, internal call inside `image_generator.py`**, specified precisely so an implementer can evaluate exactly how narrow it is:

```
GenerationTraceRecord:
  video_id: str
  attempt_index: int                 # candidate index, matches ImageGenerationResult.candidate_scores ordering
  latent_source: Literal["noise", "vae_encoded_source"]   # today: always "noise" (§A.2) — this field's
                                                            # entire value, honestly, is currently constant;
                                                            # it becomes meaningful the moment
                                                            # MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md ships
  denoise: float
  fragments_attached: list[FragmentAttachmentRecord]   # {fragment_name, attach_point, strength_or_weight}
  controlnet_enabled: bool           # lifted from the resolved GenerationProfile, not re-derived
  ipadapter_enabled: bool            # same
  seed: int
  workflow_template: str             # e.g. "gaming.json" — resolved WorkflowLibrary choice
  workflow_hash: str                 # already computed by existing code, just also written here
  edit_mask_paths: list[str]         # empty today (no inpainting fragment exists, §A.2) — populated
                                      # once an inpainting fragment exists
  recorded_at: str
```

**Integration point, precisely scoped:** one call, `GenerationTraceRecorder().record(...)`, added inside `WorkflowGraphAssembler.assemble()` (right after its existing `logger.info("Assembled workflow graph with ...")` call, §A.2) and one more inside `WorkflowBuilder`'s existing profile/fragment-resolution path, each populating the fields already computed at that point in the existing code (fragment list, profile flags, resolved template name, hash) — **no new computation, only new persistence of values that already exist in local variables at that point in the existing function.** This mirrors, exactly, how `MetricsCollector.append()` is already called from inside `ImageGeneratorPipeline.run()` today: a side-effect write, zero change to any function's signature, zero change to any function's return value, zero change to any existing test's expected behavior (an existing test asserting on `ImageGeneratorPipeline.run()`'s return value sees no change at all). This is the one integration point flagged in §A.10.

### 9.2 Root-cause fact extraction

Before any rule runs, `RuleEngine` derives a small set of named booleans/values from the `PipelineTrace` — a "facts" layer, so every rule (§10.2) is a pure function over facts, not over raw artifacts (this indirection is what makes rules independently unit-testable without constructing a full trace fixture each time):

| Fact | Derived from |
|---|---|
| `source_never_encoded` | `generation_trace.latent_source == "noise"` (today: always true, §9.1) |
| `controlnet_missing_but_expected` | `profile.controlnet_enabled == False` while `PROFILE_PREMIUM` was *not* selected (i.e., unexpectedly off, not the documented Flux exception already noted in `MODULE_8_9_INTEGRATION_ARCHITECTURE.md` §1) |
| `ipadapter_disabled_but_reference_exists` | `generation_trace.ipadapter_enabled == False` while `AssetExtractionManifest`/VRE produced a usable face/object crop |
| `edit_mask_ignored` | any `DecisionManifest` entry with `decision_type in {KEEP, REMOVE}` whose `element_id` does not appear in `generation_trace.edit_mask_paths` **when** `edit_mask_paths` is non-empty (i.e., this fact is only meaningful once an inpainting-capable renderer exists — on today's renderer it is always vacuously true and explicitly annotated `not_applicable_reason: "no inpainting fragment exists in this renderer version"`, per Design Goal 5) |
| `asset_extraction_missing` | `ArtifactIndex` entry for Module 8 has `exists: False` while `ASSET_EXTRACTION_ENABLED == True` |
| `object_mapping_incorrect` | cross-reference `AssetExtractionManifest.objects[].asset_id` against `DecisionManifest.resolved_decisions[].target.element_id` — any decision target with no matching asset_id |
| `renderer_ignored_edit_plan` | `GenerationPlan.face_strategy`/`background_strategy` imply a specific action, but no corresponding `FragmentAttachmentRecord` exists in `generation_trace.fragments_attached` for that role (this is the fact that, applied to today's code, **flags `BackgroundCompositor` directly** — its "composite" produces an output identical in hash to its input whenever a source thumbnail exists, which is independently checkable via `ArtifactIndex`'s sha256 of the pre- and post-compositor image, §9.3) |
| `background_regenerated_unnecessarily` | `DecisionManifest`'s background layer decision is `KEEP`, but the generated image's background region hash differs materially (perceptual hash distance, not exact, since restoration/upscale legitimately alter pixels) from the source's |
| `identity_drift` | `ImageGenerationResult.candidate_scores[selected].identity_score` below `MODULE7_IDENTITY_SIMILARITY_THRESHOLD` — read directly, not recomputed (Design Goal 4/consistency with §11's reuse posture) |
| `prompt_contradiction` | `PromptPackage.positive_prompt` and `.negative_prompt` share an n-gram (a cheap, deterministic string-level check — not a semantic one, deliberately, to stay in the "deterministic-first" lane) |
| `composition_mismatch` | `GenerationPlan.composition_strategy` string disagrees with the independently-computed layout category PVQEF's `composition_scorer.py` already assigns to the generated image (§11's cross-check reuse pattern) |
| `mask_overlap_problem` | any two `CompositionWorkspace` layer masks with IoU above a configured threshold, where both layers are decided `REPLACE`/`ADD` — a genuine upstream Module 10 data-quality signal, computed once here rather than duplicated per-rule |
| `conditioning_failure` | a `ControlNetFragmentError`/`FragmentAttachmentError`-shaped line found by `LogCorrelator` in `module7.log` for this `video_id` |

### 9.3 Concrete worked example (grounded, not hypothetical)

Because `BackgroundCompositor` (§A.2) is a verified, real, current defect, PORCE's design is validated against it directly rather than only against constructed scenarios: for any video where Module 9's background-layer decision is `REPLACE` and a source thumbnail exists, `ArtifactIndex` will show the generated PNG's sha256 as byte-identical (via the intermediate temp-file copy §A.2 traced) to whatever `WorkflowBuilder`'s raw ComfyUI output was *before* `BackgroundCompositor` ran — because that function never modifies the file. The `renderer_ignored_edit_plan` fact fires, `RULE-DEC-04` (§10.2) fires at `FAIL`/high confidence, and the recommended action reads, verbatim, the exact gap identified in §A.2 — this is the mechanism by which the engine "explains WHY" rather than only detecting "something is different."

## 10. Rule Engine

### 10.1 `Finding` shape

```
Finding:
  finding_id: str                    # e.g. "RULE-DEC-04"
  severity: Literal["FAIL","WARNING","INFO","PASS"]
  confidence: float                  # [0.0, 1.0] — deterministic rules report 1.0 unless the underlying
                                      # fact itself was derived from a probabilistic signal (e.g. a
                                      # perceptual-hash distance, §9.2's background_regenerated_unnecessarily),
                                      # in which case confidence reflects that signal's own reliability band
  affected_module: str
  root_cause: str                    # one sentence, plain language, references the fact(s) that fired
  recommended_action: str            # one concrete, actionable sentence
  supporting_evidence: list[ArtifactRef | LogLineRef]   # pointers, never copied content (§8.1)
  rule_version: str                  # for reproducibility as rules evolve, mirrors DimensionScore.scorer_version
```

### 10.2 Rule catalogue (deterministic, table-driven — Design Goal 4)

Six families, one file each (§5's `diagnostics/rules/`), each rule a small class implementing `IDiagnosticRule.check(facts: TraceFacts) -> Finding | None`:

- **`latent_initialization_rules.py`** — `source_never_encoded` (INFO on today's renderer, annotated per Design Goal 5; would be FAIL on a future renderer if `latent_source` disagreed with the selected `edit_mode`, once that field exists).
- **`conditioning_rules.py`** — `controlnet_missing_but_expected`, `ipadapter_disabled_but_reference_exists`, `conditioning_failure`.
- **`decision_honoring_rules.py`** — `edit_mask_ignored`, `renderer_ignored_edit_plan` (the `BackgroundCompositor` case, §9.3), `background_regenerated_unnecessarily`.
- **`asset_provenance_rules.py`** — `asset_extraction_missing`, `object_mapping_incorrect`.
- **`prompt_consistency_rules.py`** — `prompt_contradiction`.
- **`composition_rules.py`** — `composition_mismatch`, `mask_overlap_problem`, `identity_drift`.

Every rule file's rules are individually registered in `rule_engine.py`'s registry (a plain list, not a plugin-discovery mechanism — matching the project's preference for explicit, static composition over dynamic discovery, consistent with `MODULE7_QA_WEIGHTS`'s static dict convention) — adding rule #23 is adding one class and one registry line, never editing `RuleEngine` itself (Architectural Goal 7).

### 10.3 `RootCauseReport`

```
RootCauseReport:
  video_id: str
  findings: list[Finding]
  fail_count: int, warning_count: int, info_count: int, pass_count: int
  top_root_causes: list[str]         # findings sorted by severity then confidence, top 3, deduplicated
                                       # by affected_module — the "read this first" summary
  generated_from_trace_hash: str      # sha256 of PipelineTrace, ties the report to an exact trace snapshot
  engine_version: str
  status: Literal["success","partial","error"]
  generated_at: str
```

## 11. Quality Evaluation (reuse, not reinvention)

Per Architectural Goal 1/§A.6/§A.8, PORCE does **not** implement a second set of quality scorers. `quality_bridge.py` reads PVQEF's already-persisted `data/evaluation/runs/{run_id}/quality_reports/{video_id}.json` (`QualityEvaluationReport`, §A.6) when it exists, and folds its `dimension_scores`/`hard_gate_passed` into `PipelineTrace` as read-only context available to `composition_rules.py`'s `composition_mismatch` check (§9.2). If no PVQEF run exists yet for a given `video_id`, `quality_bridge.py` records `status: "unavailable"` rather than failing — PORCE's own trace/rule-engine value stands on its own without requiring PVQEF to have run first, matching Design Goal 5's "degrade gracefully" posture. The brief's "Automatic Quality Analysis" objective (identity, object, layout, background, typography, hierarchy, color, composition, brand, mask coverage, edit-region correctness) maps directly onto PVQEF's existing fourteen dimensions (§7 of `PIPELINE_VALIDATION...`) plus two PORCE-native facts not covered there (`mask_overlap_problem`, `edit_mask_ignored`, both structurally about the renderer's *process*, not the output image, and therefore correctly homed in PORCE rather than PVQEF).

## 12. Root Cause Analysis

Root cause analysis is not a separate component from the Rule Engine (§10) — `RootCauseReport.top_root_causes` **is** the root-cause-analysis deliverable, by construction: severity- and confidence-ranked, deduplicated by module, each entry traceable through `supporting_evidence` back to an exact artifact or log line (§8.1/§10.1). This satisfies the brief's requirement to "explain WHY the failure occurred instead of only reporting that it failed" directly — every `Finding.root_cause` is a plain-language sentence naming the mechanism (e.g. "Module 9 decided KEEP for element `person_0`, but Module 7 recorded no fragment referencing `person_0`'s mask" — the `BackgroundCompositor` case, §9.3), never a bare status code.

## 13. Reporting

Four formats, all derived from `RootCauseReport` + `PipelineTrace` without recomputing anything (mirrors PVQEF's own "renderers derive from the builder's output only" discipline, §A.6):

- **Human-readable** (`reporting/human_report_renderer.py`) — Markdown/HTML, one page per video: storyboard link (§8.2), top-3 root causes, full findings table, module timeline (§14).
- **Machine-readable** (`reporting/machine_report_renderer.py`) — `RootCauseReport` JSON verbatim, the canonical form every other renderer derives from.
- **Visual** — the storyboard (§8.2), a distinct artifact from the human-readable report, cross-linked from it.
- **Artifact index** — `ArtifactIndex` (§8.1) is independently retrievable, not only embedded in the trace, so tooling that only needs "where are this video's files" doesn't need to parse the full trace.
- **Failure/pipeline/execution/quality/root-cause summaries** — these are not five new artifacts; they are five pre-filtered *views* over the same `RootCauseReport`/`PipelineTrace` pair (`RootCauseReport` filtered to `severity == "FAIL"` is the failure summary; `PipelineTrace.modules` filtered to `status != "success"` is the execution summary; etc.), implemented as small query functions in `cli.py`, not as separate stored artifacts — avoiding five redundant persisted copies of the same underlying facts (Architectural Goal 6's storage discipline).

## 14. Timeline View

`timeline_builder.py` merges `PipelineTrace.modules[].started_at/completed_at/duration_seconds` (§8.1) with `LogCorrelator`'s ordered, per-video log lines into one `Timeline`:

```
Timeline:
  video_id: str
  entries: list[TimelineEntry]   # {module, event: Literal["started","completed","error","warning"],
                                  #  timestamp, message, source: "artifact"|"log"}
  total_duration_seconds: Optional[float]
  critical_path: list[str]        # modules ordered by actual measured duration, descending —
                                    # "where did the time go" at a glance
```

`timeline_renderer.py` produces a simple horizontal-bar visualization (reuses the Visualizer-style SVG generation already familiar in this project's tooling, or a plain HTML `<table>` fallback — either way, no new charting dependency required) showing every module, its duration, and any error/warning markers inline — directly satisfying the brief's Timeline View requirement (module, execution time, inputs, outputs, dependencies, errors, warnings) using data already assembled in §8/§9, not a new capture mechanism.

## 15. Configuration

`observability/config.py` (new file, mirrors `evaluation/config.py`'s existing pattern exactly): `OBS_TRACES_DIR`, `OBS_REPORTS_DIR`, `OBS_GENERATION_TRACES_DIR` (also added to `modules/config.py` as thin re-exports, matching how `EVAL_*` path constants are additively present in `modules/config.py` today per §A.4/PVQEF's own §3), `OBS_RULE_REGISTRY_ENABLED: dict[str, bool]` (per-rule on/off, default all `True` — lets an operator silence a noisy rule without a code change), `OBS_MASK_IOU_THRESHOLD` (for `mask_overlap_problem`), `OBS_PERCEPTUAL_HASH_THRESHOLD` (for `background_regenerated_unnecessarily`), `OBS_LOG_CORRELATION_WINDOW_HOURS` (bounds how far back `LogCorrelator` scans, since 7-day log retention, §A.3, is already a hard ceiling). No secrets, no network config — this system is entirely local-filesystem-based, matching the project's offline-first posture.

## 16. Logging

`observability.log` (new `OBS_LOG_PATH` constant, `logs/observability.log`), same Loguru `rotation="10 MB", retention="7 days", enqueue=True` pattern as every other module — PORCE logs its *own* execution (trace-build failures, rule-engine exceptions) to this thirteenth log file; it does not write into any existing module's log file, and (deliberately) is not itself included in `LogCorrelator`'s per-video correlation target list, to avoid a trivial self-referential loop.

## 17. Error Handling

New `PORCEError` hierarchy in `observability_exceptions.py`, following the exact `Module{N}Error`/`PVQEFError` shape: `TraceAssemblyError`, `RuleEngineError`, `ReportRenderingError`. Best-effort, non-fatal by construction (Architectural Goal/PVQEF Design Goal 4 precedent, reused deliberately for consistency): a single rule raising is caught, logged, and recorded as a `Finding` with `severity: "INFO"`, `root_cause: "rule execution failed: {typed exception}"` rather than aborting the whole `RootCauseReport` — an engine designed to explain failures should not itself become an opaque failure. A missing artifact (§8.1's `exists: False`) is never an exception — it's a fact the rule engine consumes, exactly as `ArtifactIndex` is designed to represent.

## 18. Performance

PORCE runs **on demand, per video_id**, not inline with the production pipeline (Architectural Goal 1's "zero modification" extends to "zero runtime cost when not invoked") — it is not called from `main.py::_run_pipeline_creators` at all. `ArtifactIndexBuilder`'s file-existence/hash checks and `LogCorrelator`'s grep-and-parse are both cheap, I/O-bound, sub-second-per-video operations against already-small JSON files and log lines (twelve files, 7-day/10 MB rotation cap, §A.3 — a bounded, small corpus per video). The one non-trivial cost is `StoryboardRenderer`'s image compositing (§8.2), which is deliberately deferred (built lazily, only when `cli.py trace --with-storyboard` or the human report is explicitly requested) rather than always generated — matching PVQEF's own "expensive checks are opt-in" precedent (§7.11's determinism checker). The one true integration cost (§9.1's `GenerationTraceRecord` write inside Module 7) is a single small JSON write per generation attempt — negligible relative to a 9–55 second ComfyUI generation pass (`GenerationProfile.expected_generation_seconds`, already documented per-profile in `config.py`), satisfying "must not reduce performance significantly."

## 19. Security

Entirely local-filesystem, no network calls, no new external dependency beyond what's already vendored (Pillow for the storyboard, already present). `data/observability/` contains no credentials and only derived/pointer data (paths, hashes, short text) — raw image bytes are referenced by path, never embedded (mirrors `GeneratedAsset`'s existing "image bytes never embedded" convention, §A.6/PVQEF §9). Log correlation reads existing log files read-only; PORCE never writes into `logs/module{1-10.5}.log` or any `data/{module}` directory, eliminating any risk of this new subsystem corrupting a production artifact.

## 20. Testing Strategy

Mirrors the project's 1:1 module-to-test-file convention (`tests/test_evaluation/` precedent, §A.6):

- `tests/test_observability/test_artifact_index_builder.py` — fixture-based, synthetic `data/` tree with some files present/absent, assert `ArtifactRef.exists`/`sha256` correctness.
- `tests/test_observability/test_log_correlator.py` — synthetic multi-file Loguru-format fixtures, assert correct per-video_id extraction and chronological ordering.
- `tests/test_observability/test_generation_trace_recorder.py` — the one new Module 7 hook (§9.1): assert `ImageGeneratorPipeline.run()`'s existing return value and existing test suite (`tests/test_image_generator.py`) are byte-for-byte unaffected by the new call, plus a new, additive test that the `GenerationTraceRecord` itself is written correctly.
- `tests/test_observability/test_rules/` — one file per rule family (§10.2), table-driven, synthetic `TraceFacts` fixtures with known-true and known-false cases for every rule, **including a fixture that directly reproduces the `BackgroundCompositor` case (§9.3)** as a regression test proving the rule catches the exact, real, currently-existing defect this document found — the highest-value test in the whole suite, since it validates the engine against a real bug rather than only a synthetic one.
- `tests/test_observability/test_reporting.py` — snapshot-style tests per renderer, asserting each format derives correctly from a fixed `RootCauseReport` fixture.
- `tests/test_observability/test_e2e_trace.py` — integration test running the full assembly against a small, checked-in synthetic multi-module artifact fixture set (not live data), marked with the project's existing `@pytest.mark.integration` convention.

## 21. Migration Strategy

Fully additive, no phase requires any existing test to change (Architectural Goal 1), following the exact phasing discipline `MODULE_8_9_INTEGRATION_ARCHITECTURE.md` (§8 of that document) and `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md` (§15 of that document) already established for this project:

| Phase | Change | Gate | Breaking? |
|---|---|---|---|
| 0 | `observability/` package skeleton, `config.py`/`observability_exceptions.py`, no wiring | — | No |
| 1 | `ArtifactIndexBuilder`, `LogCorrelator` — pure read-side components, fully testable against fixtures, no live-repo dependency | — | No |
| 2 | `TraceAssembler` → `PipelineTrace` (without `GenerationTraceRecord`, which doesn't exist yet) | — | No |
| 3 | **The one Module 7 hook** (§9.1) — additive `GenerationTraceRecorder.record()` call inside `WorkflowGraphAssembler`/`WorkflowBuilder` | new, always-on (cheap, §18) but wrapped in try/except so a recorder failure never affects generation itself | No — verified via §20's byte-for-byte return-value test |
| 4 | `RuleEngine` + all six rule families (§10.2) | — | No |
| 5 | `QualityBridge` (§11) | — | No, degrades gracefully if PVQEF hasn't run |
| 6 | `StoryboardRenderer` (§8.2), `TimelineBuilder`/`timeline_renderer.py` (§14) | — | No |
| 7 | `cli.py` — the one user-facing entry point tying everything together (`python -m observability.cli trace <video_id>`) | — | No |
| 8 (future, separate doc) | Module 8/9/10.5 validators for PVQEF (§A.5's identified gap) — explicitly PVQEF's own gap, not this document's scope, flagged here only as a natural follow-on | — | No |

**Rollout order matters:** Phase 3 should ship as early as practical relative to Phases 4+, since every rule in §9.2 that reasons about `generation_trace` is meaningless without it — Phases 0–3 form one coherent first release; Phases 4–7 can each ship independently after that.

## 22. Phase-by-Phase Implementation Plan (for autonomous coding agents / Codex handoff)

Same eight phases as §21, restated as concrete file-level tasks per phase — omitted here for brevity beyond §21's table and §5's file tree, since §5 already enumerates every file an implementer needs to create, and §21 already specifies dependency order; a separate, dedicated task-breakdown document (matching this project's existing `MODULE7_PHASE2_*` document-per-sprint pattern, §A itself) is the appropriate place for line-item task tickets once this architecture is approved, not this document.

## 23. Risks

| Risk | Detail | Mitigation |
|---|---|---|
| **Log correlation is best-effort, not exact** | `LogCorrelator`'s per-module duration for non-Module-7 stages is derived from log timestamp deltas, not a measured start/stop (§8.1) — could be misleading if a module's logging is sparse | `duration_source` field (§8.1) makes this explicit in every report; never presented as exact without the caveat |
| **`GenerationTraceRecord` write could, in principle, affect Module 7 timing/behavior** | Any new code inside a hot path carries nonzero risk | Wrapped in try/except (§21 Phase 3), write is synchronous-but-tiny JSON, and §20's byte-for-byte return-value test is a release gate, not a nice-to-have |
| **Rule false positives on legitimate configuration** | E.g., `controlnet_missing_but_expected` firing on `PROFILE_PREMIUM` if the Flux-exception carve-out (§9.2, already documented in `MODULE_8_9_INTEGRATION_ARCHITECTURE.md`) isn't correctly encoded | Rule explicitly checks profile identity before firing, tested against a `PROFILE_PREMIUM` fixture in §20 |
| **PVQEF and PORCE reporting the same video could confuse readers about which is authoritative for what** | Two sibling subsystems, two report trees | §13's explicit "PORCE explains causes, PVQEF measures quality" framing, cross-linked (§11) rather than merged into one report, so each retains a single clear responsibility |
| **Twelve-log-file grep cost grows with corpus size over time** | 7-day retention (§A.3) bounds this today, but a future retention-policy change could silently invalidate `OBS_LOG_CORRELATION_WINDOW_HOURS`'s assumption | Flagged in §15's config; `LogCorrelator` should warn (not fail) if a video's earliest expected log timestamp falls outside the configured window |

## 24. Future Extensions

- **Module 8/9/10.5 validators for PVQEF** (§21 Phase 8) — a natural, explicitly out-of-scope-here follow-on to the gap identified in §A.5.
- **Live/streaming trace mode** — today's design is on-demand/post-hoc (§18); a future extension could tail the twelve log files in real time during a live pipeline run for a "watch this video's trace build as it happens" mode, without changing the underlying `PipelineTrace`/`Finding` data model at all.
- **Cross-video pattern mining** — once enough `RootCauseReport`s accumulate, a batch job could surface "which root cause fires most often across the last N videos" (a natural extension of PVQEF's own `HistoricalStore`/`benchmark_history.jsonl` pattern, §A.6, reused rather than reinvented) — explicitly deferred, since it requires no new capture mechanism, only a new aggregation over data this document already collects.
- **Automatic remediation suggestions beyond text** — `Finding.recommended_action` is a sentence today; a future extension could link directly to the specific config flag or code location implicated (e.g., a `Finding` about `BackgroundCompositor` linking directly to `image_generator.py:690` — an IDE-openable reference), once a stable line-number/commit-hash convention is established project-wide.
- **Integration with `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md`** — once that document's staged-edit renderer ships, `GenerationTraceRecord.edit_mask_paths`/`latent_source` become live, meaningful fields rather than constants, and `edit_mask_ignored` (§9.2) becomes an active, non-vacuous rule — no schema change required, exactly per Design Goal 5's intent.
