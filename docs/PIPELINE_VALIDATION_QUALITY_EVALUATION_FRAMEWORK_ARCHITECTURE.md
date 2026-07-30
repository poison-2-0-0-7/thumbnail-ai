# Pipeline Validation & Quality Evaluation Framework — Architecture

**Status:** Design only. No implementation code, no pseudocode, per instructions.
**Author role:** Lead Software Architect
**Repo studied:** `poison-2-0-0-7/thumbnail-ai` (branch `main`, live clone)

---

## 0. Grounding note — read this first

Before designing, the full repository was cloned and read, including: `main.py`
(the eight-stage orchestration loop: CSV Reader → YouTube Metadata → Thumbnail
Downloader → Thumbnail Intelligence → Redesign Spec → Prompt Compiler →
Asset Composer → Module 7 generation); `modules/models.py` in full (all 60+
frozen Pydantic models); `modules/config.py` (the `MODULE{N}_*` / `DEFAULT_{X}_DIR`
constant convention); every `*_exceptions.py` file; every `*_components/`
directory and its `interfaces.py`; `modules/vision_stack/` in full (config,
models, registry, loader, lifecycle, runtime, and all seven implemented model
wrappers); `pytest.ini`; `tests/test_main_pipeline.py`; and the existing
Module 6.5, 7, 8, 9, 10 architecture documents in `docs/` to match their
structure, tone, and level of rigor.

This framework is **not Module 11 and not a ninth pipeline stage**. Every
existing module already writes its own artifact to its own directory and logs
its own errors; this framework never touches that logic. It is a **read-only
observer that sits beside the pipeline**: it invokes the same public functions
`main.py` invokes (or `main.run_pipeline` itself), reads the same persisted
JSON artifacts every module already writes, and adds nothing to the hot path.
If this framework's process is never started, the production pipeline behaves
identically to how it behaves today.

Four discrepancies between the task brief and the actual repository state are
surfaced here rather than papered over, exactly as the Module 9 document
did for the (at-the-time) missing Module 8:

1. **No evaluation, validation, benchmarking, or reporting code exists
   anywhere in the repository today.** There is no `evaluation/`, `qa/`, or
   `benchmarks/` package, no golden-sample fixtures, and no report renderer.
   This is a greenfield design with no prior art to extend — Sections 3–19
   below are the first specification of this surface.

2. **Module 7 already contains a per-candidate quality gate that this
   framework must not duplicate.** `modules/models.py::QualityAssuranceReport`
   and `CandidateScore` are computed *inside* `image_generator.py` during
   generation, per candidate, to pick a winner. This framework evaluates the
   **one already-selected, already-persisted** thumbnail per video *after the
   fact*, at a different cadence (batch/CI, not per-request), for a different
   audience (an engineer or Afsar reviewing pipeline health), and with a
   richer, non-real-time metric set (prompt adherence, attractiveness, color
   harmony, determinism-across-runs) that Module 7 has no reason to compute
   inline. Section 7.2 specifies exactly how this framework *reads* Module 7's
   existing scores instead of recomputing them.

3. **`vision_stack.yaml` / `VisionStackConfig` already reserve model slots for
   `openclip`, `florence2`, and `paddleocr`** (see
   `modules/vision_stack/config.py::VISION_STACK_MODEL_ORDER`), but **no
   wrapper file exists for any of the three** — only `grounding_dino.py`,
   `insightface_multi.py`, `bisenet.py`, `birefnet.py`, `sam2.py`,
   `depth_anything.py`, and `teed.py` are implemented. Prompt-adherence and
   attractiveness/aesthetic scoring (Section 7) are the first consumers that
   actually need OpenCLIP. This document treats **`OpenCLIPWrapper`** as a
   **Phase 0 prerequisite** (Section 19), built by an implementation engineer
   strictly following the existing wrapper pattern (`birefnet.py`'s shape:
   `ModelLoader`-backed construction, `VisionStackRuntimeError` on failure,
   lifecycle registration) — not a redesign of `vision_stack`, an extension
   of an already-provisioned but unbuilt slot.

4. **`modules/vre_components` populates `ReferenceAssets.face_crop_path`
   from the *source* thumbnail, and Module 7 checks identity similarity
   against that crop.** Face-preservation scoring in this framework
   (Section 7.1) reuses that exact same source-face crop and the same
   `InsightFaceMultiWrapper` — it does not re-run face detection with new
   logic.

Everything below — naming, file layout, logging, caching, exception style,
DI pattern, testing layout, config conventions — mirrors what the repository
already does, not a novel style introduced for this framework.

---

## 1. Purpose

The **Pipeline Validation & Quality Evaluation Framework** (referred to
throughout as **PVQEF**) is an independent, on-demand harness that:

- Drives the production pipeline (`main.run_pipeline`, or any subset of its
  stages) against a controlled input set (a small CSV, a "golden" fixed set
  of creators, or the live `data/creators.csv`).
- After each stage runs, validates the artifact that stage persisted against
  a schema- and invariant-level contract, independent of whether the stage
  itself raised an exception.
- After Module 7 (and Module 10) complete, evaluates the **final generated
  thumbnail** along fourteen quality dimensions (Section 7).
- Aggregates all of this into one **`PipelineRunReport`** per run, persists
  it, and compares it against the previous run to surface regressions.
- Exposes batch execution, golden-sample regression testing, and historical
  benchmarking as first-class, separately invokable capabilities.

It exists to answer, mechanically and repeatably, the question every module's
own test suite cannot answer alone: *"does the pipeline, taken end to end,
still produce a production-quality thumbnail, and is it better or worse than
last week?"*

---

## 2. Design Goals

1. **Zero modification of Modules 1–10.** PVQEF imports and calls existing
   public functions; it does not edit `main.py`, any `*_engine.py`, any
   `*_components/`, `models.py`, or `config.py` beyond the additive,
   append-only sections this document proposes (new constants, new model
   classes — never edits to existing fields).
2. **Read-only with respect to production data.** PVQEF never deletes or
   mutates `data/analysis/`, `data/redesign_specs/`, `data/prompt_packages/`,
   `data/generated_thumbnails/`, etc. It writes exclusively to its own
   `data/evaluation/` tree.
3. **Deterministic-first scoring, ML-assisted only where necessary** — the
   same stance Module 5 and Module 9 already take. Hash/dimension/schema
   checks are exact. Perceptual dimensions (attractiveness, composition) use
   models already in (or provisioned for) `vision_stack`, never a new
   external API call.
4. **Best-effort, non-fatal by construction.** One creator's evaluation
   failure never aborts the batch, mirroring `main.run_pipeline`'s own
   `skipped += 1; continue` pattern.
5. **Same conventions as every other module**: frozen Pydantic models with
   `video_id` / `status` / `error_message` / `duration_seconds` /
   `*_at` fields, `Module{N}Error`-style typed exception hierarchies,
   `ABC`-based component interfaces, `loguru` with `enqueue=True`, atomic
   temp-file-then-`replace()` writes, `pytest` markers for anything that
   needs a live model or GPU.
6. **Hardware-aware.** Full-suite runs (fourteen quality dimensions × batch
   of N) are the most GPU/CPU-intensive thing in the repository outside of
   Module 7 itself. Every scorer must declare its VRAM footprint and support
   a `lightweight` config profile for Afsar's RTX 4060 (8 GB VRAM) laptop
   (Section 13).
7. **Extensible without redesign.** Adding a fifteenth quality dimension or a
   new regression rule must be possible by adding one new class behind an
   existing interface — never by editing the orchestrator.

---

## 3. Folder Structure

PVQEF is a new top-level package, sibling to `modules/`, not nested inside
it — it depends on `modules/` (imports pipeline functions and shared models)
but nothing in `modules/` depends on it. This mirrors the existing "no image
I/O in Module 9" boundary discipline: the dependency arrow points one way.

```
thumbnail-ai/
├── evaluation/
│   ├── __init__.py
│   ├── cli.py                          # entry point: python -m evaluation.cli ...
│   ├── config.py                       # EVAL_* constants (mirrors modules/config.py style)
│   ├── evaluation_exceptions.py        # PVQEFError hierarchy
│   ├── pipeline_runner.py              # drives main.run_pipeline stage-by-stage
│   ├── module_validators/
│   │   ├── __init__.py
│   │   ├── interfaces.py               # IModuleValidator (ABC)
│   │   ├── csv_reader_validator.py
│   │   ├── youtube_metadata_validator.py
│   │   ├── thumbnail_downloader_validator.py
│   │   ├── thumbnail_intelligence_validator.py
│   │   ├── redesign_spec_validator.py
│   │   ├── prompt_compiler_validator.py
│   │   ├── asset_composer_validator.py
│   │   └── module7_validator.py
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── interfaces.py               # IQualityScorer (ABC)
│   │   ├── prompt_adherence_scorer.py  # OpenCLIP text-image similarity
│   │   ├── face_preservation_scorer.py # reuses InsightFaceMultiWrapper
│   │   ├── object_preservation_scorer.py # reuses GroundingDINOWrapper
│   │   ├── background_quality_scorer.py  # reuses BiRefNetWrapper / BiSeNetWrapper
│   │   ├── composition_scorer.py       # reuses DepthAnythingWrapper + rule-of-thirds heuristics
│   │   ├── text_readability_scorer.py  # reuses existing OCR path (Module 4's PaddleOCR usage)
│   │   ├── color_harmony_scorer.py     # reuses Module 4's ColorProfile logic
│   │   ├── attractiveness_scorer.py    # OpenCLIP aesthetic-embedding proxy score
│   │   ├── determinism_checker.py      # re-runs generation N times, diffs seeds/hashes
│   │   ├── performance_profiler.py     # wall-clock, peak VRAM, peak RAM per stage
│   │   └── aggregator.py               # combines all scorer outputs into one report
│   ├── benchmarking/
│   │   ├── __init__.py
│   │   ├── historical_store.py         # append-only JSONL, mirrors module7_metrics.jsonl
│   │   ├── regression_detector.py      # statistical comparison vs. rolling baseline
│   │   └── golden_sample_manager.py    # loads/validates the fixed golden creator set
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── report_builder.py           # assembles PipelineRunReport
│   │   └── report_renderer.py          # JSON (canonical) + Markdown + HTML views
│   └── batch/
│       ├── __init__.py
│       └── batch_executor.py           # concurrency-bounded multi-creator runs
│
├── data/
│   └── evaluation/
│       ├── runs/{run_id}/
│       │   ├── run_manifest.json               # PipelineRunReport
│       │   ├── module_results/{video_id}.json  # ModuleValidationResult[] per creator
│       │   ├── quality_reports/{video_id}.json # QualityEvaluationReport per creator
│       │   └── comparisons/{video_id}_before_after.json
│       ├── golden/
│       │   ├── golden_manifest.json            # pinned creators + expected-shape hashes
│       │   └── baselines/{video_id}.json        # last-known-good QualityEvaluationReport
│       └── history/
│           └── benchmark_history.jsonl          # one row per run, append-only
│
├── docs/
│   └── PIPELINE_VALIDATION_QUALITY_EVALUATION_FRAMEWORK_ARCHITECTURE.md  # this document
│
├── tests/
│   └── test_evaluation/
│       ├── test_pipeline_runner.py
│       ├── test_module_validators/
│       ├── test_quality_components/
│       ├── test_benchmarking.py
│       ├── test_reporting.py
│       └── test_e2e_golden.py           # @pytest.mark.integration / @pytest.mark.gpu
│
└── modules/
    ├── models.py     # + one new additive section: "# Evaluation Framework"
    └── config.py     # + one new additive section: "# Evaluation Framework"
```

`evaluation/` gets its own `config.py` (rather than folding everything into
`modules/config.py`) because PVQEF's configuration surface (report formats,
regression thresholds, golden-sample paths, scorer weights) is large and
orthogonal to pipeline configuration — but the **directory-path constants**
(`EVAL_RUNS_DIR`, `EVAL_GOLDEN_DIR`, `EVAL_HISTORY_PATH`, `EVAL_LOG_PATH`)
are added to `modules/config.py` following the exact `DEFAULT_{X}_DIR` /
`MODULE{N}_LOG_PATH` naming convention already used for every other module,
so that `PROJECT_ROOT`-relative paths have exactly one source of truth
across the whole repo, matching the "config.py has zero dependencies, single
predictable place to look" principle stated in `modules/config.py`'s own
docstring.

---

## 4. Public APIs

All entry points are plain functions (no framework classes exposed at the
boundary), matching the calling convention `main.py` uses for every module
(`process_video(...)`, `analyze_thumbnail(...)`, `build_redesign_specification(...)`).

```python
# evaluation/pipeline_runner.py
def run_full_evaluation(
    csv_path: Path = DEFAULT_CSV_PATH,
    *,
    run_id: str | None = None,          # default: UTC timestamp + short hash
    golden_only: bool = False,          # restrict to evaluation/golden/golden_manifest.json
    stages: tuple[str, ...] | None = None,  # default: all eight; can restrict e.g. ("module7",)
) -> PipelineRunReport: ...

# evaluation/batch/batch_executor.py
def run_batch_evaluation(
    csv_path: Path,
    *,
    max_concurrency: int = 1,           # RTX 4060 / 16GB RAM default: serialize GPU stages
) -> PipelineRunReport: ...

# evaluation/benchmarking/regression_detector.py
def detect_regressions(
    current: PipelineRunReport,
    baseline_run_id: str | None = None, # default: most recent prior run in history
) -> list[RegressionFinding]: ...

# evaluation/benchmarking/golden_sample_manager.py
def run_golden_regression_suite() -> PipelineRunReport: ...

# evaluation/reporting/report_renderer.py
def render_report(
    report: PipelineRunReport,
    fmt: Literal["json", "markdown", "html"] = "markdown",
) -> str: ...

# evaluation/cli.py
def main(argv: list[str] | None = None) -> int: ...
    # subcommands: run | golden | batch | compare | report
```

`run_full_evaluation` is the one function an engineer or a CI job calls for
"does the pipeline still work end to end." Everything else is composable
around it.

---

## 5. Internal APIs (component contracts)

Following the exact `ABC`-interface-per-component-family pattern used in
`vre_components/interfaces.py`, `generation_components/interfaces.py`,
`decision_components/interfaces.py`, and `composition_components/interfaces.py`:

```python
# evaluation/module_validators/interfaces.py
class IModuleValidator(ABC):
    @abstractmethod
    def validate(self, video_id: str, artifact_path: Path) -> ModuleValidationResult:
        """Check schema conformance + module-specific invariants for one persisted artifact."""

# evaluation/quality/interfaces.py
class IQualityScorer(ABC):
    @property
    @abstractmethod
    def dimension(self) -> str:
        """One of the fourteen quality dimensions this scorer produces (Section 7)."""

    @abstractmethod
    def score(self, context: QualityScoringContext) -> DimensionScore:
        """Compute one dimension's score for one generated thumbnail."""

# evaluation/benchmarking/interfaces.py
class IHistoricalStore(ABC):
    @abstractmethod
    def append(self, record: BenchmarkRecord) -> None: ...
    @abstractmethod
    def load_recent(self, n: int) -> list[BenchmarkRecord]: ...

class IRegressionRule(ABC):
    @abstractmethod
    def check(self, current: BenchmarkRecord, baseline: BenchmarkRecord) -> RegressionFinding | None:
        """One statistically-defined regression check (Section 11)."""

# evaluation/reporting/interfaces.py
class IReportRenderer(ABC):
    @abstractmethod
    def render(self, report: PipelineRunReport) -> str: ...
```

`QualityScoringContext` (an internal, non-persisted dataclass/model passed
between scorers) bundles everything a scorer might need without each scorer
re-loading it: the `ImageGenerationResult`, the `PromptPackage`, the
`ThumbnailIntelligence` (source), the `RedesignSpecification`, the
`GenerationBundle`, and file handles to both the source thumbnail and the
generated image. This is assembled once per video by `aggregator.py` and
handed to every registered `IQualityScorer`, so fourteen scorers do not
each independently re-open the same two image files.

---

## 6. Class Responsibilities at a Glance

| Component | Responsibility | Depends on (existing repo code) |
|---|---|---|
| `PipelineRunner` | Drives stages 1–7 (+10) via existing public functions; captures per-stage timing, exceptions, artifact paths | `main.py` functions, all `*_engine.py` public functions |
| `ModuleValidator` (×8) | Schema + invariant checks per module's persisted JSON | `models.py` classes for parsing |
| `QualityScorer` (×14, one per dimension) | Compute one Section-7 dimension | `vision_stack/*`, `modules/models.py` |
| `Aggregator` | Merges 14 `DimensionScore`s into one `QualityEvaluationReport` | scorer outputs only |
| `HistoricalStore` | Append-only JSONL persistence of `BenchmarkRecord`s | mirrors `MetricsCollector` in `image_generator.py` |
| `RegressionDetector` | Statistical current-vs-baseline comparison | `HistoricalStore` |
| `GoldenSampleManager` | Loads/validates the pinned golden creator set + last-known-good baselines | `csv_reader.py` schema |
| `ReportBuilder` | Assembles the final `PipelineRunReport` | all of the above |
| `ReportRenderer` (×3) | JSON / Markdown / HTML views of one report | `ReportBuilder` output only |
| `BatchExecutor` | Concurrency-bounded multi-creator orchestration | `PipelineRunner` |

---

## 7. Quality Evaluation Pipeline

Evaluation runs **after** a video's `ImageGenerationResult` (Module 7) and,
if present, `CompositionWorkspace`/`GenerationBundle` (Module 10) are on
disk. It never re-triggers generation itself except for the determinism
check (7.11), which is explicitly opt-in and expensive.

### 7.1 Face preservation
Reuses `InsightFaceMultiWrapper` (already in `vision_stack/insightface_multi.py`).
Computes cosine similarity between the source thumbnail's face embedding
(the same crop Module 6.5's `_ComfyUIWebSocketTransport`/VRE pipeline already
produced at `ReferenceAssets.face_crop_path`) and the corresponding region of
the generated image. This is presentation-layer reuse of the **same**
model instance Module 7's inline `identity_score` already uses — see 7.2's
note on not duplicating that computation when a source face is absent
(`face_detected=False`, `skipped=True` on the existing `FaceMatchResult`).

### 7.2 Reused Module 7 signals (not recomputed)
Rather than recompute what Module 7 already computed per-candidate,
`aggregator.py` reads `ImageGenerationResult.candidate_scores` (specifically
the entry where `selected=True`) and lifts `identity_score`,
`composition_score`, `text_safe_zone_score`, `object_preservation_score`,
`color_compliance_score`, and `overall_score` directly out of the existing
`QualityAssuranceReport`/`CandidateScore` models — read-only, zero
duplication. PVQEF's own scorers for these same-named dimensions (7.1, 7.3,
7.5, 7.6, 7.9) exist to provide an **independent, out-of-band cross-check**
using different tooling/thresholds than Module 7's inline gate, specifically
so that a bug in Module 7's own scorer cannot silently mark its own homework.
Both numbers are reported side by side in `QualityEvaluationReport`
(`inline_score` vs. `independent_score`); a persistent divergence between
them is itself a `RegressionFinding` (Section 11).

### 7.3 Object preservation
Reuses `GroundingDINOWrapper` to detect the same labeled objects
`ThumbnailIntelligence.objects` (Module 4) found in the source, confirming
their presence/absence/displacement in the generated image.

### 7.4 Background quality
Reuses `BiRefNetWrapper`/`BiSeNetWrapper` foreground/background segmentation
(the same masks VRE already computes) to measure background-region
sharpness, artifact density (checkerboard/seam detection at composite
boundaries), and absence of hallucinated foreground objects.

### 7.5 Composition quality
Reuses `DepthAnythingWrapper`'s depth map plus deterministic rule-of-thirds /
visual-weight heuristics already established in Module 4's
`CompositionAnalysis` model — same scoring vocabulary, applied to the
generated image instead of the source.

### 7.6 Text readability
Reuses the existing OCR path (Module 4's `OCRResult`/`TextRegion` models)
against the generated image: contrast ratio against local background,
minimum stroke width relative to canvas size, and safe-zone containment
(reusing `PromptPackage.generation_parameters` safe-zone constants from
`config.py`'s "Zone-label thresholds" section).

### 7.7 Color harmony
Reuses Module 4's `ColorProfile` extraction logic against the generated
image, compared against `RedesignSpecification.color_direction` — is the
generated palette actually closer to the *intended* direction than the
source was.

### 7.8 Visual consistency
Cross-checks lighting direction/color temperature between composited
real-asset regions (from `CompositionWorkspace.layers`) and the AI-generated
background, flagging seams a human reviewer would notice (a shadow falling
the wrong way, mismatched white balance).

### 7.9 Similarity to redesign intent (prompt adherence)
Uses the new `OpenCLIPWrapper` (Section 0, item 3) to compute text-image
similarity between `PromptPackage.positive_prompt` and the generated image,
and separately between the *redesign spec's* structured directives
(`color_direction`, `subject_treatment`, `layout_direction`) and the image —
giving both a "did it follow the prompt text" and a "did it follow the
structured design intent" score.

### 7.10 Thumbnail attractiveness
Uses the same `OpenCLIPWrapper` embedding against a small, versioned set of
aesthetic-anchor prompts/exemplars (not a live external "aesthetics API" —
consistent with the fully-local-stack requirement); this is explicitly
labeled a **proxy score**, not a ground-truth CTR predictor, and documented
as such in every report (Section 10) to avoid over-trusting it.

### 7.11 Generation determinism
Opt-in, expensive: re-runs `run_image_generation_pipeline` for the same
`video_id`/seed N times (config-driven, default N=3) and diffs
`generation_hash`, `GeneratedAsset.sha256`, and pixel-level SSIM between
runs. Confirms the seed-from-`video_id`-hash design (established during
Module 6 work) actually produces bit-identical or perceptually-identical
output across repeated runs on the same hardware — the actual meaning of
"generation determinism" in the objectives list. This is the only scorer
that triggers new Module 7 executions; every other scorer is read-only.

### 7.12 Runtime performance
`performance_profiler.py` wraps each pipeline stage call in `pipeline_runner.py`
with wall-clock timing (mirrors `ImageGenerationResult.stage_durations_seconds`,
extended to cover Modules 1–6/10, which today do not record per-stage timing
at all — this is additive instrumentation in the runner, not an edit to
those modules' own code).

### 7.13 Memory usage
Peak RSS (via `resource.getrusage` / `psutil`, already an implicit dependency
given `vision_stack`'s GPU/CPU model loading) and peak VRAM (via the same
mechanism `MODULE7_CAPABILITY_PROBE_ENABLED`/`capability_probe.py` already
uses to query the ComfyUI server) sampled once per stage.

### 7.14 Failure rate
Not a per-video scorer — a batch-level aggregate: `succeeded / total` and a
per-stage failure breakdown, computed by `ReportBuilder` from every
creator's `ModuleValidationResult` list, mirroring the `succeeded`/`skipped`
counters `main.run_pipeline` already prints, just persisted and compared
over time instead of only printed to stdout.

---

## 8. Scoring Architecture

Each of the fourteen dimensions produces one `DimensionScore`:

```python
class DimensionScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    dimension: str                       # one of the 14 names in Section 7
    score: float                         # normalized [0.0, 1.0]
    passed: bool                         # score >= configured threshold for this dimension
    threshold: float
    detail: dict[str, Any] = {}          # dimension-specific breakdown, e.g. per-object IoU
    scorer_version: str                  # for reproducibility across scorer code changes
    duration_seconds: float = 0.0
    status: Literal["success", "partial", "error", "skipped"] = "success"
    error_message: Optional[str] = None
```

`aggregator.py` collects all fourteen into:

```python
class QualityEvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    generated_asset_sha256: str          # ties this report to one exact GeneratedAsset
    dimension_scores: list[DimensionScore] = []
    inline_scores: dict[str, float] = {} # lifted from Module 7's own QualityAssuranceReport (7.2)
    weighted_overall_score: float = 0.0  # weights configurable, see EVAL_QUALITY_WEIGHTS
    hard_gate_passed: bool               # AND of every dimension's hard-gate-eligible `passed`
    status: Literal["success", "partial", "error"] = "success"
    partial_failure_reasons: list[str] = []
    error_message: Optional[str] = None
    total_duration_seconds: float = 0.0
    evaluated_at: str
```

Weighting mirrors `MODULE7_QA_WEIGHTS`'s existing dict-of-floats convention
in `config.py` exactly (`EVAL_QUALITY_WEIGHTS: dict[str, float]`), so
re-weighting dimensions is a config change, never a code change. Dimensions
that are structurally incapable of a pass/fail hard gate (e.g. attractiveness,
7.10 — a proxy score by design) are explicitly excluded from
`hard_gate_passed` and only contribute to `weighted_overall_score`.

---

## 9. Reporting Architecture

```python
class ModuleValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    module_name: str                     # "module1_csv_reader" ... "module10_asset_composer"
    artifact_path: Optional[str] = None
    schema_valid: bool
    invariants_checked: list[str] = []
    invariants_failed: list[str] = []
    status: Literal["success", "partial", "error", "skipped"] = "success"
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    validated_at: str

class PipelineRunReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    csv_path: str
    golden_only: bool = False
    total_creators: int
    succeeded: int
    skipped: int
    module_results: dict[str, list[ModuleValidationResult]] = {}   # video_id -> results
    quality_reports: dict[str, QualityEvaluationReport] = {}        # video_id -> report
    regressions: list[RegressionFinding] = []
    stage_failure_counts: dict[str, int] = {}   # Section 7.14
    aggregate_performance: dict[str, float] = {}  # p50/p95 durations per stage
    status: Literal["success", "partial", "error"] = "success"
    started_at: str
    completed_at: str
    total_duration_seconds: float = 0.0
```

`report_builder.py` assembles this once, atomically writes it (temp file +
`Path.replace()`, exactly the pattern `main.py::_persist_generated_thumbnail`
already uses) to `data/evaluation/runs/{run_id}/run_manifest.json`, and the
three `IReportRenderer` implementations (`json` — the canonical form, is the
manifest verbatim; `markdown` — human-readable summary with a module-by-module
table and a before/after thumbnail-path table; `html` — same content,
styled, suitable for opening directly in a browser) derive from it without
re-computing anything.

Before/after comparisons (`comparisons/{video_id}_before_after.json`)
reference `ThumbnailData.thumbnail_path` (source, Module 3) and
`GeneratedAsset.path` (final, Module 7) side by side with the
`QualityEvaluationReport` for that video — the report never embeds image
bytes, following the `GeneratedAsset` convention that "image bytes are never
embedded" in any persisted model.

---

## 10. Benchmarking

`benchmark_history.jsonl` is append-only, one JSON object per run, mirroring
`MODULE7_METRICS_PATH`'s existing `module7_metrics.jsonl` convention exactly
(same append-only, one-line-per-record, `MetricsCollector`-style writer
pattern already implemented in `image_generator.py`):

```python
class BenchmarkRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    recorded_at: str
    total_creators: int
    succeeded: int
    skipped: int
    mean_weighted_overall_score: float
    per_dimension_mean_scores: dict[str, float] = {}
    mean_stage_durations_seconds: dict[str, float] = {}
    peak_vram_mb: Optional[float] = None
```

`historical_store.py::load_recent(n)` reads the tail of this JSONL (no
database, consistent with the project's fully-local, no-external-service
stack) for trend charts in the Markdown/HTML report (a simple ASCII sparkline
in the Markdown form; an inline `<canvas>`/SVG mini-chart in HTML, no
external JS charting library — consistent with the repo having no frontend
dependency today).

Future model comparisons (the brief's "supports future model comparisons")
are supported by tagging every `BenchmarkRecord` and `QualityEvaluationReport`
with the `profile_name` and `workflow_version` already present on
`ImageGenerationResult` — comparing two `BenchmarkRecord`s with different
`profile_name` values (e.g. a future SDXL-Turbo profile vs. the current
default) is a filter over the same history file, not new schema.

---

## 11. Regression Framework

`regression_detector.py` compares a `BenchmarkRecord` (current) against a
rolling baseline (default: mean of the last 5 `BenchmarkRecord`s, config-
driven `EVAL_REGRESSION_WINDOW`). Each `IRegressionRule` is independent and
declares its own sensitivity — new rules are added by registering a new
class, never by editing `regression_detector.py`'s core loop:

| Rule | Trigger |
|---|---|
| `OverallScoreDropRule` | `mean_weighted_overall_score` drops more than `EVAL_REGRESSION_SCORE_DELTA` (default 0.05) vs. baseline |
| `DimensionRegressionRule` | any single dimension's mean score drops more than its own configured delta |
| `FailureRateSpikeRule` | `skipped/total` increases beyond `EVAL_REGRESSION_FAILURE_DELTA` |
| `PerformanceRegressionRule` | any stage's mean duration increases beyond `EVAL_REGRESSION_LATENCY_MULTIPLIER` (default 1.5×) baseline |
| `InlineVsIndependentDivergenceRule` | Module 7's own `overall_score` and PVQEF's independent recomputation (Section 7.2) diverge beyond `EVAL_QA_DIVERGENCE_THRESHOLD` for more than one run in a row |
| `DeterminismDriftRule` | (only when 7.11 was run) SSIM between repeated generations for the same seed drops below `EVAL_DETERMINISM_SSIM_THRESHOLD` |

Each produces a `RegressionFinding`:

```python
class RegressionFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_name: str
    severity: Literal["info", "warning", "critical"]
    dimension_or_stage: Optional[str] = None
    current_value: float
    baseline_value: float
    delta: float
    message: str
```

Golden-sample regression (`run_golden_regression_suite`) is the same
machinery pointed at a pinned, version-controlled creator set (Section 12)
compared against a checked-in "last known good" baseline rather than a
rolling window — this is the test an engineer runs before merging a change
to any of Modules 1–10, analogous to a snapshot test.

---

## 12. Batch Execution

`batch_executor.py` wraps `pipeline_runner.py` with bounded concurrency.
Given Afsar's hardware constraint (RTX 4060 laptop GPU, 16 GB RAM, i9
13th-gen — the same constraint already driving `MODULE7_MAX_CONCURRENT_GENERATIONS: int = 1`
in `config.py`), the default `EVAL_MAX_CONCURRENCY` is **1** for any creator
whose evaluation touches Module 7 or a `vision_stack` GPU model (Sections
7.1, 7.3–7.5, 7.9–7.11), and only CPU-only validators (Modules 1–3,
schema-only checks) are safely parallelizable — `EVAL_CPU_ONLY_CONCURRENCY`
defaults to `os.cpu_count() - 2` matching the "leave headroom for Ollama"
caution already implicit in Module 4's design discussions. Batch runs process
creators sequentially through the GPU-bound stages and only fan out for the
cheap validators, avoiding the VRAM-exhaustion class of failure
`VRAMExhaustedError` already anticipates in `module7_exceptions.py`.

---

## 13. Configuration

New additive section in `modules/config.py` (directory paths only, following
`DEFAULT_{X}_DIR` naming):

```python
# ---------------------------------------------------------------------------
# Evaluation Framework
# ---------------------------------------------------------------------------

EVAL_LOG_PATH: Path = LOG_DIR / "evaluation.log"
EVAL_RUNS_DIR: Path = PROJECT_ROOT / "data" / "evaluation" / "runs"
EVAL_GOLDEN_DIR: Path = PROJECT_ROOT / "data" / "evaluation" / "golden"
EVAL_HISTORY_PATH: Path = PROJECT_ROOT / "data" / "evaluation" / "history" / "benchmark_history.jsonl"
```

All scoring weights, thresholds, and concurrency settings live in
`evaluation/config.py` (kept separate — see Section 3's rationale):
`EVAL_QUALITY_WEIGHTS`, `EVAL_REGRESSION_WINDOW`,
`EVAL_REGRESSION_SCORE_DELTA`, `EVAL_REGRESSION_FAILURE_DELTA`,
`EVAL_REGRESSION_LATENCY_MULTIPLIER`, `EVAL_QA_DIVERGENCE_THRESHOLD`,
`EVAL_DETERMINISM_SSIM_THRESHOLD`, `EVAL_MAX_CONCURRENCY`,
`EVAL_CPU_ONLY_CONCURRENCY`, `EVAL_DETERMINISM_REPEAT_COUNT` (default 3),
a `lightweight` vs. `full` scorer-profile switch (the former skips 7.9/7.10/
7.11 — the three most VRAM/time-expensive dimensions — for fast local
iteration on the RTX 4060).

---

## 14. Logging

One dedicated logger sink, exactly matching every existing module's pattern
(`logger.add(str(EVAL_LOG_PATH), rotation="10 MB", retention="30 days", format=_LOG_FORMAT, level="DEBUG", enqueue=True)`),
configured once in `evaluation/cli.py`'s entry point (not at import time in
every submodule, avoiding duplicate sinks — the same discipline
`image_generator.py`/`thumbnail_intelligence.py` already follow). Every
validator and scorer logs `video_id`, `module_name`/`dimension`, and
`status` as structured `loguru` kwargs, not f-string interpolation, matching
the `"...{email}..."`, `email=creator.email` style used throughout `main.py`.

---

## 15. Caching

Two independent caches:

1. **`QualityScoringContext` cache** — within a single run, the bundle
   assembled in Section 5 is built once per video and shared across all
   fourteen scorers (avoids 14× redundant file reads/model calls).
2. **Vision-stack model instance cache** — `evaluation/quality/` scorers
   request wrapper instances (`InsightFaceMultiWrapper`, `GroundingDINOWrapper`,
   etc.) through the *existing* `vision_stack.loader.ModelLoader` /
   `vision_stack.registry.ModelRegistry`, never instantiating a second copy
   of a model already loaded and registered for Module 4/6.5 in the same
   process — this is the load-bearing reason PVQEF must run in-process
   alongside (or immediately after) a pipeline invocation rather than as a
   fully detached process when GPU memory is scarce, consistent with the
   registry's own `lifecycle.py` transition guarantees.

No caching of `QualityEvaluationReport`/`PipelineRunReport` results
themselves across runs — every run is evaluated fresh; only the *history* of
past runs' summaries is retained (Section 10), never their full detail,
keeping `data/evaluation/history/` small regardless of run count.

---

## 16. Dependency Injection

Every orchestrator (`PipelineRunner`, `Aggregator`, `ReportBuilder`,
`BatchExecutor`) accepts its collaborators as constructor parameters typed
against the `ABC` interfaces in Section 5, with production defaults —
identical to the pattern already used throughout, e.g.
`MetricsCollector(metrics_path: Path = MODULE7_METRICS_PATH)` in
`image_generator.py`, or `ProfileSelector()`'s default construction in
`main.py::_select_module7_profile`. This is what makes
`test_pipeline_runner.py` able to inject fake `IModuleValidator`s that
return canned `ModuleValidationResult`s without invoking any real module,
and what makes `test_quality_components/` able to test `Aggregator` against
hand-built `DimensionScore` lists without loading any vision-stack model —
mirroring exactly how `test_main_pipeline.py` already mocks `main`'s
module-level functions via `monkeypatch`/`SimpleNamespace` rather than
running real network/GPU calls.

---

## 17. Error Handling

New, additive exception module `evaluation/evaluation_exceptions.py`,
following the `Module7Error`/`VREBaseError` base-plus-specific-subclasses
convention exactly:

```python
class PVQEFError(Exception):
    """Base exception for every PVQEF failure."""

class PipelineStageInvocationError(PVQEFError):
    """Raised when a pipeline stage cannot be invoked at all (import/signature mismatch)."""

class ModuleArtifactMissingError(PVQEFError):
    """Raised when a module's expected persisted artifact is absent."""

class ModuleArtifactSchemaError(PVQEFError):
    """Raised when a persisted artifact fails Pydantic validation against models.py."""

class QualityScorerError(PVQEFError):
    """Base class for scorer-specific failures; caught per-scorer, never fatal to the batch."""

class QualityScorerModelUnavailableError(QualityScorerError):
    """Raised when a required vision_stack wrapper (e.g. OpenCLIPWrapper) cannot be loaded."""

class DeterminismCheckError(QualityScorerError):
    """Raised when a repeated-generation determinism check itself fails to execute."""

class ReportPersistError(PVQEFError):
    """Raised when a PipelineRunReport cannot be atomically written."""

class GoldenSampleInvalidError(PVQEFError):
    """Raised when the pinned golden manifest fails validation."""

class RegressionRuleError(PVQEFError):
    """Raised when a regression rule cannot be evaluated (e.g. missing baseline)."""
```

Every one of these is caught at the narrowest possible scope (per-validator,
per-scorer, per-creator) and recorded as a `status="error"` /
`error_message=...` field on the relevant result model — never allowed to
propagate up and abort a batch run, mirroring `main.run_pipeline`'s own
"log it, count it, continue" discipline for every one of its eight stages.

---

## 18. Testing Strategy

Matches `pytest.ini`'s existing marker conventions exactly — no new markers
needed, the existing `integration` and `gpu` markers already cover this
framework's needs:

- **Unit tests** (`tests/test_evaluation/test_module_validators/`,
  `test_quality_components/`): every validator and scorer tested against
  hand-built `models.py` instances (e.g. a `ThumbnailIntelligence` with a
  deliberately malformed `FaceAnalysis`), no real files, no real models,
  fully offline — same style as `tests/test_decision_components/`.
- **Integration tests** (`@pytest.mark.integration`): `PipelineRunner`
  driving real Modules 1–6/10 against a tiny fixture CSV (no network — same
  `enable_oembed_fallback=False`-by-default discipline `main.py` documents
  for its own test-suite offline requirement).
- **GPU tests** (`@pytest.mark.gpu`): `QualityScorer` implementations that
  load real `vision_stack` wrappers and a real Module 7/ComfyUI-generated
  image; skipped by default per `pytest.ini`'s `addopts`.
- **Golden/regression tests** (`test_e2e_golden.py`, both markers): the
  full `run_golden_regression_suite()` against the pinned creator set —
  the closest thing to a true end-to-end smoke test, run explicitly
  (`pytest -m "integration and gpu"`) before merging changes to any
  existing module, exactly the scenario the task brief's "detects
  regressions" objective describes.
- Every test file follows `test_main_pipeline.py`'s `sys.path` bootstrap
  pattern for project-root and `modules/` importability.

---

## 19. Implementation Phases

**Phase 0 — Prerequisite (blocking):**
Build `OpenCLIPWrapper` in `modules/vision_stack/openclip.py` and
`openclip_exceptions.py`, strictly following `birefnet.py`'s existing shape
(`ModelLoader`-backed load, `VisionStackRuntimeError`/`OutOfMemoryError`
subclasses, lifecycle registration). This unblocks Sections 7.9 and 7.10 only
— every other scorer has zero new-model dependencies and can proceed in
parallel.

**Phase 1 — Skeleton & module validation:**
`evaluation/` package scaffold, `evaluation_exceptions.py`,
`evaluation/config.py`, `modules/config.py` additive section,
`PipelineRunner`, all eight `ModuleValidator`s, `ModuleValidationResult`,
basic `PipelineRunReport` (module-validation fields only, no quality
scoring yet). Deliverable: `run_full_evaluation(stages=(...))` can already
answer "did every module produce a schema-valid artifact."

**Phase 2 — Quality scoring (reused-signal dimensions first):**
`QualityScoringContext`, `Aggregator`, and the scorers with no new-model
dependency: 7.1 (face, InsightFace), 7.2 (reused Module 7 signals — no new
code, pure field-lifting), 7.3 (objects, GroundingDINO), 7.4 (background,
BiRefNet/BiSeNet), 7.5 (composition, DepthAnything), 7.6 (text, existing
OCR), 7.7 (color, existing `ColorProfile` logic), 7.8 (visual consistency).

**Phase 3 — CLIP-dependent scoring:**
7.9 (prompt adherence) and 7.10 (attractiveness), gated on Phase 0.

**Phase 4 — Performance/determinism:**
7.12 (runtime), 7.13 (memory), 7.14 (failure rate — batch-level, in
`ReportBuilder`), then 7.11 (determinism — the only scorer that re-triggers
generation; built last since it is the most expensive to test).

**Phase 5 — Reporting & benchmarking:**
`ReportBuilder`, all three `IReportRenderer`s, `HistoricalStore`,
`BenchmarkRecord`.

**Phase 6 — Regression & golden suite:**
All `IRegressionRule`s, `GoldenSampleManager`, `run_golden_regression_suite`,
the pinned golden manifest (a small, deliberately fixed subset of
`data/creators.csv`, checked into the repo).

**Phase 7 — Batch execution & CLI:**
`BatchExecutor`, `evaluation/cli.py` subcommands, hardware-aware
concurrency defaults (Section 12).

Each phase is independently mergeable and independently testable per
Section 18 — later phases never require reopening earlier ones, since every
new dimension/rule/renderer is a new class behind an existing interface.

---

## 20. Integration Strategy

- **No edits to `main.py`.** PVQEF calls the same functions `main.py` calls
  (`process_video`, `analyze_thumbnail`, `build_redesign_specification`,
  `compile_prompt_package`, `AssetComposer().prepare_generation_workspace`,
  `run_image_generation_pipeline`) directly from `pipeline_runner.py`,
  duplicating only the *sequencing*, not the logic — if `main.py`'s
  sequencing changes, `pipeline_runner.py` is the one place to update, and
  a unit test (`test_pipeline_runner.py`) asserting the stage order matches
  `main.run_pipeline`'s own order is the guard against drift.
- **No edits to `models.py`'s existing classes** — only new, additive model
  classes appended under a new `# Evaluation Framework` section, exactly as
  Module 9's own document proposed for its new models.
- **No edits to `vision_stack/`'s existing wrappers** — only one new
  wrapper file for an already-provisioned, currently-unbuilt model slot
  (Phase 0).
- **Invocation is entirely separate from production.** `python main.py`
  (production) and `python -m evaluation.cli run` (validation) are two
  independent entry points; running one never requires the other to be
  running, and PVQEF is expected to run in CI, in a scheduled batch job, or
  manually before/after a change to any module — never inline with a live
  creator-outreach run.
- **Handoff artifacts** for this framework should be authored and reviewed
  exactly like Modules 6.5/7/8/9/10 were: this document goes to Codex as
  the Phase 1 handoff; subsequent phases each get their own granular,
  repository-state-verified design document once the previous phase's code
  exists to verify against — the same strict phase separation already
  established as non-negotiable for this project.

---

## 21. Risks and Open Questions

1. **OpenCLIP model choice/checkpoint is unspecified.** `vision_stack.yaml`
   reserves the `openclip` slot but does not pin a specific checkpoint
   (e.g. ViT-B/32 vs. ViT-L/14) — this is a decision for whoever implements
   Phase 0, likely biased toward the smallest checkpoint that fits
   comfortably beside Module 7's already-loaded SDXL pipeline on 8 GB VRAM.
2. **Attractiveness (7.10) is a proxy, not ground truth.** No amount of
   local scoring substitutes for real CTR data from actual thumbnail
   performance; this is stated explicitly in every report and should not be
   used as a hard gate.
3. **Determinism checking (7.11) cost.** Running generation 3× per golden
   creator is the single most expensive thing in this framework on an
   RTX 4060; the default golden set should stay small (single digits) or
   this check should run on a longer cadence (weekly) rather than every
   commit.
4. **No Module 8/9/10 generated-image asset-extraction re-use yet regarding
   "object preservation."** Module 9's document already flagged that
   Module 8 (`AssetExtractionManifest`) does not exist in the repository.
   Section 7.3's object-preservation scorer works around this by calling
   `GroundingDINOWrapper` directly rather than depending on Module 8's
   still-hypothetical output — if Module 8 ships later with the contract
   Module 9's document proposed, Section 7.3 should be revisited to consume
   it instead, avoiding a second independent object-detection pass.
5. **Golden manifest curation is a manual, ongoing responsibility** — someone
   (Afsar) must periodically confirm the pinned creators' "last known good"
   baselines are still actually good, or regressions will be measured
   against a stale, possibly already-degraded baseline.
