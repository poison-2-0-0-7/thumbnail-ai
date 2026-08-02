# MODULE8_THUMBNAIL_QUALITY_OPTIMIZATION_ARCHITECTURE.md

**Status:** Architecture only. No implementation code, no tests, no repository files modified.
**Source of truth:** `poison-2-0-0-7/thumbnail-ai`, freshly cloned/pulled and read in full for this task, including every doc in `docs/`, the `evaluation/` package, `observability/`, `modules/decision_components/`, `modules/design_blueprint_components/`, and `modules/generation_components/`.
**Author role:** Lead Software Architect.
**Consumer:** Gemini CLI will implement this later, from this document alone.

---

## 0. Naming Conflict — flagged, not resolved here

The brief asks for a document named `MODULE8_THUMBNAIL_QUALITY_OPTIMIZATION_ARCHITECTURE.md` and for the new system to be "Module 8." **`Module 8` already exists in this repository** — `docs/MODULE8_ASSET_EXTRACTION_ENGINE_ARCHITECTURE.md`, `modules/asset_extraction_engine.py`, `modules/asset_extraction_components/` — and does something unrelated (extracting reusable visual assets from source thumbnails). This document uses the requested filename verbatim, as instructed, and refers to the new system by role ("the Optimization Layer") rather than by a module number, leaving renumbering to you, consistent with how the Module 10/Asset-Composer naming collision was handled previously in this project.

---

## 1. Executive Summary

The infrastructure work is genuinely done: staged editing activates correctly, ControlNet/IPAdapter conditioning reaches the sampler, PORCE observability and root-cause diagnostics exist and run automatically. **What's missing is not another rendering system — it's the decision-making layer that targets "beat the original" instead of "render without error."** This document's central finding, from a full audit, is that almost every component the brief's eight phases ask for **already exists**, distributed across four subsystems that were built for other purposes and never wired into a single optimization loop:

| Brief phase | Already exists as | Location |
|---|---|---|
| Phase 1 — Quality Analysis Engine | 14-dimension quality scorer, already scores generated **and loads the source** for comparison | `evaluation/quality/` (PVQEF) |
| Phase 2 — Intelligent Edit Planner | keep/remove/replace/enhance/add arbitration engine, deterministic-first, LLM-assisted | `modules/decision_engine.py` (Module 9) |
| Phase 3 — Prompt Optimization | headline/hook authoring + layout/strategy planning, deterministic | `modules/design_blueprint_components/` (Module 5.5) |
| Phase 4 — Multi-Candidate Generation | strategy-bounded (not just seed-bounded) per-candidate prompt perturbation + ranking | `modules/generation_components/candidate_strategy_planner.py`, Module 7 Phase 4 |
| Phase 5 — Quality Scoring / winner selection | per-candidate `QualityAssuranceReport`/`CandidateScore` (real-time, in Module 7) **and** the richer PVQEF `Aggregator` (offline) | `modules/models.py`, `evaluation/quality/aggregator.py` |
| Phase 8 — Validation gates | `hard_gate_passed`, identity/face/composition/text-safe-zone/object-preservation scores | `modules/models.py::QualityAssuranceReport` |

**What genuinely does not exist**, verified by direct inspection, and what this document actually designs:

1. **No comparative "does the generated thumbnail beat the original" verdict anywhere.** Every existing scorer evaluates the generated image (optionally against the source as a *fidelity reference*, e.g. face/object preservation) — none produces a head-to-head "original vs. generated, which wins, and by how much" judgment. This is the single most direct gap relative to the brief's actual objective.
2. **No feedback loop.** `evaluation/benchmarking/historical_store.py` and `regression_detector.py` persist and compare scores over time, but nothing reads that history back into `modules/decision_engine.py`'s rule engine or `design_blueprint_components/strategy_engine.py` to change future decisions. Scores are recorded, never learned from.
3. **`GenerationTraceRecord`/`PipelineTrace` (PORCE) capture zero quality, candidate-ranking, or planner-reasoning fields today** — confirmed by reading the full model in `observability/models.py`. Phase 7 is a real, additive gap.
4. **No "over-editing" / structural-divergence-from-source scorer** among PVQEF's 14 dimensions — `VisualConsistencyScorer` checks lighting/color consistency *between generated regions*, not overall edit magnitude versus the original composition.
5. **Prompt optimization doesn't yet fold in transcript, captions, or current trends** — `design_blueprint_components/copywriter.py` uses `ThumbnailIntelligence`, `RedesignSpecification`, and `VideoMetadata`, with no transcript/caption/trend input path found anywhere in the repo.

This document's Phases 1–8 (§8–15) are therefore written almost entirely as **integration and gap-closing specs**, not new subsystems, per the brief's own strict rule to reuse existing infrastructure.

---

## 2. Root Cause Analysis (Phase 0)

**Why are thumbnails still worse than the original despite successful, correctly-conditioned rendering?**

Investigated across every decision point in the pipeline:

- **Module 9 (Decision Engine) decides what to keep/remove/replace/enhance/add, but has no visibility into quality outcomes.** Its rule engine (`modules/decision_components/rule_engine.py`, `rules/{keep,remove,replace,enhance,add}_rules.py`) is deterministic-first with LLM assistance for ambiguous cases — but it makes every decision from Module 4/5/6/8 artifacts alone. It has never seen a `QualityAssuranceReport` or a PVQEF score. It cannot know that, say, its `enhance_rules.py` heuristics have historically produced lower `face_quality_score` outcomes than `keep_rules.py` ones for a given niche — that data exists (in Module 7's per-run reports and PVQEF's historical store) but is never read by Module 9.
- **Module 7 Phase 4 selects the best of N candidates using an absolute quality bar (`hard_gate_passed`, `overall_score`), not a comparative bar against the original.** A generation batch where every candidate scores below the original's own (never-computed) quality baseline will still "successfully" select a winner and ship it — because nothing ever computes the original's baseline to compare against. This is the most direct mechanical explanation for "technically correct but visually poor": the system optimizes within the candidate set, never against the actual competitor (the original thumbnail).
- **PVQEF exists but runs offline, after the fact, for human review — not in the generation hot path**, by its own explicit design (`docs/PIPELINE_VALIDATION_QUALITY_EVALUATION_FRAMEWORK_ARCHITECTURE.md` §0: *"a read-only observer that sits beside the pipeline... adds nothing to the hot path"*). Its 14-dimension richness (attractiveness, color harmony, whitespace-adjacent composition scoring, prompt adherence) is therefore never available at generation-decision time — only Module 7's narrower 6-field inline `QualityAssuranceReport` is.
- **Prompt optimization (Module 5.5 + Module 6) is deterministic and template/lexicon-driven, with no mechanism to learn which hook types, headline templates, or layout strategies have historically scored well.** `copywriter.py`'s `MODULE55_HEADLINE_SCORE_WEIGHTS` are static constants in `config.py`, not derived from outcome data.
- **No component anywhere asks "did we do better than the input?"** This is not a bug in any single module — it is the absence of a question no existing module was ever scoped to ask. Every module answers "is this correct/consistent/gated," never "is this better."

**Conclusion:** the poor-quality outcome is not caused by any single defect. It's the predictable result of four mature, individually-correct subsystems (planning, prompting, generation/ranking, scoring) operating with no shared comparative objective and no feedback path between them. The fix is an integration layer with exactly one job the rest of the system doesn't have: hold the original thumbnail's own score as the target to beat, and close the loop so every other subsystem can act on that target and on what happened last time.

---

## 3. Current Pipeline Review

```
Module 1-4   ingestion + CV intelligence
Module 5     Redesign Specification (deterministic keep/remove/preserve directives)
Module 5.5   Copywriter + Layout Planner + Strategy Engine  →  DesignBlueprint
Module 6     Prompt Compiler                                 →  PromptPackage
Module 6.5   Visual Reference Engine                         →  face/bg/fg/masks/depth/canny
Module 8     Asset Extraction Engine  (existing; unrelated to this brief — see §0)
Module 9     AI Decision Engine  (keep/remove/replace/enhance/add arbitration) → DecisionManifest
Module 10    Asset Composer                                  →  CompositionWorkspace, GenerationBundle
Module 10.5  Thumbnail Planner & Conditioning Pipeline        →  GenerationPlan
Module 7 V2  ImageGeneratorPipeline.run() — staged edit, ControlNet/IPAdapter,
             multi-candidate (CandidateStrategyPlanner + ranking), PORCE trace write
                    │
                    ▼
        data/generated_thumbnails/{video_id}/...  +  ImageGenerationResult
                    │
        ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄  (nothing runs here today, in production)  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
                    │
                    ▼ (only when a human/CI explicitly invokes `evaluation/cli.py`)
        PVQEF (evaluation/) — 14-dimension scoring, benchmarking, reporting
```

The gap the brief is pointing at is precisely the dotted line: nothing connects the bottom of the generation pipeline back to a comparative decision, and nothing connects PVQEF's evaluation back up into Modules 5.5/9/7's decision-making for the *next* run.

---

## 4. Proposed Architecture

Introduce a single new package, **`optimization/`**, sitting beside `evaluation/` and `observability/` at the repository root (matching their existing top-level-package pattern, not nested under `modules/`, since like PVQEF it is cross-cutting rather than pipeline-stage-specific). It has exactly four responsibilities, each mapped to a real gap from §1:

1. **Comparative Scoring** (§8) — wraps PVQEF's existing scorers, adds the one new dimension they lack (original-vs-generated head-to-head verdict), and adds the over-editing/structural-divergence scorer.
2. **Optimization Orchestration** (§6) — the closed loop: plan → prompt → generate candidates → score comparatively → if no candidate beats the original, re-plan with an adjusted strategy (bounded retry) → select winner.
3. **Feedback Store & Application** (§12) — persists outcome-linked decisions (which Module 9 rule fired, which Module 5.5 hook type, which candidate strategy) against their comparative scores, and exposes a read API that Module 9's `RuleEngine` and Module 5.5's `strategy_engine.py` can optionally consult — additive, opt-in, never required for either module to function standalone.
4. **Trace/PORCE Extension** (§13–14) — additive fields only, on the existing `GenerationTraceRecord`/`PipelineTrace` models, plus new PORCE diagnostic rules that reuse the existing `RuleExecutionEngine`.

**Explicitly not built:** a second decision engine, a second prompt compiler, a second candidate generator, a second scoring framework, a second trace/report system. Every one of those already exists and is reused as-is.

---

## 5. Component Design

```
optimization/
  __init__.py
  config.py                       # OPTIMIZATION_* constants, additive to global config pattern
  exceptions.py                   # OptimizationBaseError hierarchy
  comparative/
    __init__.py
    interfaces.py                 # IComparativeScorer, matches evaluation/quality/interfaces.py's IQualityScorer shape
    baseline_scorer.py            # scores the ORIGINAL thumbnail using PVQEF's existing scorers, unmodified
    beats_original_scorer.py      # NEW dimension: head-to-head verdict + margin (§8.2)
    edit_magnitude_scorer.py      # NEW dimension: over-editing / structural divergence (§8.3)
  orchestration/
    __init__.py
    interfaces.py
    optimization_loop.py          # OptimizationLoop — the closed-loop orchestrator (§6)
    retry_strategy.py             # bounded re-plan/re-prompt/re-generate policy (§6.3)
    winner_selector.py            # wraps Module 7's CandidateScore + comparative scores into one decision (§10)
  feedback/
    __init__.py
    interfaces.py
    outcome_recorder.py           # persists (decision, strategy, score) tuples (§12.1)
    outcome_store.py              # read/query API over persisted outcomes (§12.2)
    prior_provider.py             # optional read-only hook consumed by Module 9 / Module 5.5 (§12.3)
  trace/
    __init__.py
    trace_extension.py            # builds the additive PORCE fields (§13)
  validation/
    __init__.py
    acceptance_gate.py            # Phase 8 — final accept/reject before shipping (§15)
```

Every component under `comparative/` and `winner_selector.py` depends on `evaluation.quality.*` and `modules.models.*` only — never modifies them. Every component under `feedback/` depends on `evaluation.benchmarking.historical_store` for persistence conventions but writes to its own namespace (§12.1), not into PVQEF's existing store, to avoid conflating "CI regression tracking" (PVQEF's actual purpose) with "generation-time decision feedback" (this system's purpose) — these are different audiences and different write cadences and should not share a table.

---

## 6. Runtime Flow

```
1. Module 10.5 GenerationPlan ready
        │
2. optimization.comparative.baseline_scorer.score(source_thumbnail)
   → BaselineScore  (reuses evaluation/quality/*, scored ONCE per video_id, cached)
        │
3. optimization.orchestration.optimization_loop.run(video_id, plan, baseline)
        │
        ├─ 3a. Module 7 ImageGeneratorPipeline.run(...)  [UNMODIFIED CALL]
        │        → N candidates, each with its existing inline QualityAssuranceReport
        │
        ├─ 3b. For each candidate: optimization.comparative.beats_original_scorer.score(
        │         source, candidate, baseline_score)
        │        → BeatsOriginalVerdict per candidate (§8.2)
        │
        ├─ 3c. optimization.orchestration.winner_selector.select(
        │         candidates, quality_reports, beats_original_verdicts)
        │        → best candidate, or NONE if no candidate beats baseline (§10)
        │
        ├─ 3d. IF winner found → proceed to step 4
        │      IF no winner AND retry budget remains →
        │           optimization.orchestration.retry_strategy.next_attempt(
        │               prior_attempts, decision_manifest, design_blueprint)
        │           → adjusted DecisionManifest/DesignBlueprint hint (bounded,
        │             e.g. "prefer keep over enhance for face region next attempt")
        │           → loop back to 3a with the adjusted inputs
        │      IF no winner AND retry budget exhausted →
        │           ship best-available candidate, flag it in the trace as
        │           "did not beat original" (never silently claim success)
        │
4. optimization.validation.acceptance_gate.evaluate(winner)  → accept / reject (§15)
5. optimization.feedback.outcome_recorder.record(decision_manifest, strategy,
   scores, winner)   [async-safe, never blocks shipping the thumbnail]
6. optimization.trace.trace_extension.build(...)  → additive GenerationTraceRecord fields
```

Step 3a is the **only** call into Module 7, made through its existing, unmodified public `run()` signature. Step 3d's "adjusted DecisionManifest/DesignBlueprint hint" is a bounded parameter nudge (e.g. re-weighting which `decision_components/rules/*.py` rule wins a conflict via `ConflictResolver`'s existing confidence mechanism), never a rewrite of Module 9's logic itself — the retry strategy calls Module 9/5.5's existing public entry points again with a different bias input, exactly as Module 7 Phase 4 already re-invokes `CandidateStrategyPlanner` per candidate index today.

---

## 7. Data Flow

```
ThumbnailIntelligence, RedesignSpecification, DecisionManifest, DesignBlueprint,
PromptPackage, GenerationBundle, CompositionWorkspace, GenerationPlan
        │                                            (all existing, read-only)
        ▼
BaselineScore (new)  ──────────────────────┐
        │                                  │
        ▼                                  │
ImageGenerationResult × N candidates       │
        │                                  │
        ▼                                  │
QualityAssuranceReport × N (existing)      │
        │                                  ▼
        └──────────────► BeatsOriginalVerdict × N (new)
                                  │
                                  ▼
                          WinnerSelection (new)
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
          AcceptanceResult   TraceExtension   OutcomeRecord
             (new)              (new)            (new)
                    │             │             │
                    ▼             ▼             ▼
          shipped thumbnail  PipelineTrace   OutcomeStore
                              (additive)     (new, queryable
                                              by future runs)
```

---

## 8. Planner Design — Phase 1 + Phase 2 Integration

### 8.1 Quality Analysis Engine — reuse, not rebuild

All fourteen dimensions the brief's Phase 1 lists (composition, readability, facial visibility, visual hierarchy, subject prominence, clutter, background distraction, lighting, contrast, saturation, color harmony, whitespace, emotional impact, CTR potential) map onto PVQEF's existing scorers with the following correspondence — verified against `evaluation/quality/*.py`:

| Brief metric | Existing scorer |
|---|---|
| composition, visual hierarchy, subject prominence | `composition_scorer.py` |
| readability, text obstruction | `text_readability_scorer.py` |
| facial visibility, identity | `face_preservation_scorer.py` |
| clutter, background distraction | `background_quality_scorer.py` |
| lighting, contrast, saturation | folded into `attractiveness_scorer.py` + `visual_consistency_scorer.py` |
| color harmony | `color_harmony_scorer.py` |
| object fidelity | `object_preservation_scorer.py` |
| prompt intent match | `prompt_adherence_scorer.py` |
| emotional impact, CTR potential, whitespace | **no existing scorer — genuine gap** |

The Optimization Layer does not re-implement any row with an existing scorer. `baseline_scorer.py` (§5) is a thin wrapper that runs the *existing* `Aggregator` (`evaluation/quality/aggregator.py`) against the **original** thumbnail instead of a generated one — `QualityScoringContext` already supports this since it loads `source_thumbnail_path` unconditionally (verified in `evaluation/quality/scoring_context.py`).

**Genuinely new scorers needed** (added to `evaluation/quality/` itself, as two more `IQualityScorer` implementations, since they belong with the other thirteen and PVQEF's `Aggregator`/`report_builder.py` already iterate the scorer registry generically — this is additive registration, not a new framework):
- `emotional_ctr_scorer.py` — proxies emotional impact/CTR potential from measurable correlates already available elsewhere in the repo: face expression/size from `thumbnail_intelligence.py`'s existing InsightFace output, headline hook type and score already computed by `copywriter.py` (`HeadlineCandidate.score`), and color saturation/contrast (already computed inline by `attractiveness_scorer.py`, reused not recomputed). This is explicitly a **proxy/heuristic score, not a true CTR prediction model** — no ground-truth CTR data exists anywhere in this repository (a finding already surfaced in the Module 9 Phase 2 CTR engine design), and this document does not claim otherwise.
- `whitespace_scorer.py` — measures negative-space ratio directly from the generated image's saliency/segmentation output already produced by the vision stack (`bisenet`/`sam2` wrappers already implemented under `modules/vision_stack/`), compared against `RedesignSpecification.layout_direction.target_negative_space_ratio` (already an existing field).

### 8.2 The Comparative Verdict — the actual new capability

```python
class BeatsOriginalVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)
    video_id: str
    candidate_index: int
    baseline_overall_score: float      # original thumbnail's own Aggregator score
    candidate_overall_score: float     # existing QualityAssuranceReport.overall_score, or
                                        # a full PVQEF Aggregator run if OPTIMIZATION_DEEP_SCORE=True
    delta: float                       # candidate - baseline
    beats_original: bool               # delta > OPTIMIZATION_MIN_WIN_MARGIN
    per_dimension_delta: dict[str, float]  # which specific dimensions improved/regressed
```

`OPTIMIZATION_MIN_WIN_MARGIN` (new config constant, default e.g. `0.05` on PVQEF's existing 0–1 scale) exists so "beats original" requires a meaningful margin, not noise — this directly prevents shipping thumbnails that are merely statistically indistinguishable from the source, which is a weaker bar than the brief's "consistently outperform."

### 8.3 Edit Magnitude / Over-Editing Scorer (feeds Phase 8)

```python
class EditMagnitudeScore(BaseModel):
    model_config = ConfigDict(frozen=True)
    structural_similarity: float   # SSIM between source and generated, standard, no new dependency
    identity_drift: float          # reads existing identity_score from QualityAssuranceReport, inverted
    over_edited: bool              # structural_similarity below OPTIMIZATION_MIN_STRUCTURAL_SIMILARITY
                                    # AND identity_drift above threshold — i.e. changed too much AND
                                    # lost the subject, not just "changed a lot" (a strong edit that
                                    # preserves identity is not penalized)
```

### 8.4 Intelligent Edit Planning — Phase 2 is Module 9, used as-is

The brief's Phase 2 list — preserve face, preserve branding, preserve product, replace background, enlarge subject, remove clutter, improve lighting, improve framing, improve typography, preserve composition — is not a new taxonomy to design. It is Module 9's existing `keep`/`remove`/`replace`/`enhance`/`add` decision space, already produced per-element by `modules/decision_components/rules/{keep,remove,replace,enhance,add}_rules.py`, arbitrated by `ConflictResolver`, and already resumable/cacheable/LLM-assisted for ambiguous cases. The Optimization Layer's only addition here is §12.3's optional feedback hook — it does not touch the rule files themselves.

---

## 9. Prompt Optimization — Phase 3 Integration

Module 5.5 (`copywriter.py`, `layout_planner.py`, `strategy_engine.py`) and Module 6 (`prompt_compiler.py`) already combine YouTube title, `ThumbnailIntelligence`, and `RedesignSpecification` into `HeadlineCandidate`s and a `PromptPackage`, deterministically. Per the repo audit, **captions, transcript, and "current trends" are not currently ingested anywhere** — there is no transcript-fetching code, no caption parser, and no trend-data source in the repository at all.

This document does not invent a trends data source (that would violate "no fabrication" and there is nothing in the repo to ground it in). It specifies the **integration point** only: `design_blueprint_components/copywriter.py`'s `HeadlineCandidate` generation already accepts `ThumbnailIntelligence`/`RedesignSpecification`/`VideoMetadata` as typed inputs; a transcript/caption field would be added the same way (additive optional field on `VideoMetadata` or a new sibling model), consumed by a new template/keyword rule in `copywriter.py`'s existing template-library pattern. **This is flagged as a prerequisite data-acquisition task outside this document's scope** (no transcript/caption source currently exists to wire in) rather than designed as if the data already existed.

What the Optimization Layer *can* do today, without new data sources: feed `outcome_store` (§12) history back into `MODULE55_HEADLINE_SCORE_WEIGHTS`-style weighting so that hook types (`curiosity`, `shock`, `controversy`, etc.) which have historically produced higher `BeatsOriginalVerdict.delta` are preferred — this is the prompt-optimization "learning" the brief's Phase 6 actually asks for, applied to the one prompt-adjacent subsystem where it's currently groundable.

---

## 10. Candidate Generation — Phase 4 Integration

Module 7 Phase 4's `CandidateStrategyPlanner` already derives bounded, strategy-driven (not just seed-driven) per-candidate `PromptPackage`s from a `DesignBlueprint`, and Module 7 already ranks candidates via `CandidateScore` (`rank`, `selected`, `hard_gate_passed`). The Optimization Layer's `winner_selector.py` **wraps** this existing ranking rather than replacing it:

```python
class OptimizedSelection(BaseModel):
    model_config = ConfigDict(frozen=True)
    video_id: str
    module7_selected_index: Optional[int]     # what Module 7's own CandidateRanker picked
    optimization_selected_index: Optional[int]  # what beats_original scoring picked; may differ
    selection_agrees: bool
    reason: str                                # human-readable: why optimization overrode
                                                # Module 7's pick, if it did (e.g. Module 7's
                                                # top-ranked candidate didn't beat the original,
                                                # but a lower-ranked one did)
```

This design deliberately keeps Module 7's own selection visible and auditable rather than silently discarding it — `selection_agrees=False` cases are exactly the evidence needed to later tune Module 7's own `MODULE7_QA_WEIGHTS` (a change out of scope for this document, per "do not redesign Module 7," but the data to justify that future change is what this field produces).

---

## 11. Quality Scoring — Phase 5

Two scoring cadences already exist and remain separate, by design (this mirrors PVQEF's own explicit real-time-vs-offline distinction from §0 of its architecture doc):

- **Real-time (in the generation loop, step 3b of §6):** `QualityAssuranceReport` (existing, Module 7 inline) + `BeatsOriginalVerdict` (new, §8.2) — must be fast enough to run per-candidate without materially extending the 25-55s generation times already observed. `emotional_ctr_scorer`/`whitespace_scorer` (§8.1) are deliberately cheap heuristics for this reason, not new ML models.
- **Offline/batch (PVQEF, unchanged):** full 14+2-dimension `Aggregator` run, used for `baseline_scorer.py` (once per video, not per candidate — the original doesn't change between candidates, so this cost is paid once) and for periodic regression benchmarking via the existing `evaluation/benchmarking/` tooling, unmodified.

The "automatic winning candidate" determination the brief's Phase 5 asks for is `winner_selector.py` (§10) — it is automatic, deterministic given its inputs, and fully auditable via `OptimizedSelection.reason`.

---

## 12. Feedback System — Phase 6

### 12.1 Outcome Recording
```python
class OptimizationOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    video_id: str
    niche: str
    decisions_applied: list[str]       # element_key -> LayerDecision summary, from Module 9's DecisionManifest
    hook_type_used: Optional[str]      # from HeadlineCandidate, if applicable
    candidate_strategy_name: str       # from CandidateStrategy.name (Module 7 Phase 4, existing)
    beats_original: bool
    delta: float
    per_dimension_delta: dict[str, float]
    recorded_at: str
```
Persisted to `data/optimization/outcomes/{video_id}/outcome.json` — sharded like every other module's data directory, atomic writes, matching the established convention throughout this codebase.

### 12.2 Outcome Store — read API
`outcome_store.py` exposes aggregate, read-only queries: `mean_delta_by_hook_type(niche)`, `mean_delta_by_decision_rule(rule_id)`, `mean_delta_by_candidate_strategy(strategy_name)`. Pure aggregation over persisted `OptimizationOutcome` records — no ML, no external calls, deterministic given the same outcome history.

### 12.3 Prior Provider — the actual "improve without architectural changes" mechanism
```python
class IPriorProvider(ABC):
    def rule_confidence_prior(self, rule_id: str) -> float: ...      # consumed by
    def hook_type_prior(self, hook_type: str) -> float: ...          # ConflictResolver's
    def candidate_strategy_prior(self, strategy_name: str) -> float: # existing confidence
                                                                       # mechanism, additively
```
Module 9's `ConflictResolver` and Module 5.5's `strategy_engine.py` already resolve competing options via confidence scores (`decision_components/confidence.py` already exists). `prior_provider.py` is an **optional** additional confidence input those existing mechanisms can blend in — a config flag (`OPTIMIZATION_FEEDBACK_ENABLED`, default `False` until validated) controls whether it's consulted at all. This satisfies the brief's "improve future generations without requiring architectural changes" literally: no new decision logic is added to Module 9/5.5, only a new confidence input those modules already know how to blend.

---

## 13. GenerationTrace Integration — Phase 7

Additive fields only, on `observability/models.py::GenerationTraceRecord` (never modifying existing fields, matching the convention every other module in this codebase has followed):

```python
# GenerationTraceRecord additions
baseline_score: Optional[float] = None
candidate_scores: list[float] = Field(default_factory=list)
beats_original: Optional[bool] = None
winning_candidate_index: Optional[int] = None
module7_selected_index: Optional[int] = None
selection_agreed: Optional[bool] = None
edit_magnitude: Optional[float] = None
over_edited: Optional[bool] = None
optimization_strategy_used: Optional[str] = None
retry_attempt_count: int = 0
```

Populated by `optimization/trace/trace_extension.py`, called at the same point in `main.py`'s flow where `GenerationTraceRecord` is already written today — one additive call, no existing write path altered.

---

## 14. PORCE Integration — Phase 7 (continued)

New diagnostic rules, added the same way `edit_mode_resolution_rules.py` and `controlnet_capability_rules.py` were added in the immediately preceding architecture pass — new files under `observability/diagnostics/rules/`, registered with the existing `RuleRegistry`, executed by the existing `RuleExecutionEngine`:

```python
class GeneratedThumbnailDidNotBeatOriginalRule(IDiagnosticRule):
    """RULE-OPT-01: Flags when GenerationTraceRecord.beats_original is False,
    surfacing it through the existing RootCauseReport pipeline exactly like
    any other finding."""

class OverEditedAcceptedRule(IDiagnosticRule):
    """RULE-OPT-02: Flags when over_edited=True was recorded but the thumbnail
    still shipped — checks acceptance_gate.py's decision against the recorded
    edit magnitude."""

class OptimizationSelectionDisagreementRule(IDiagnosticRule):
    """RULE-OPT-03: Flags when selection_agreed=False, surfacing cases worth
    reviewing for a future Module 7 CandidateRanker weight tuning pass
    (out of scope here, per 'do not redesign Module 7')."""
```

No change to `RuleExecutionEngine`, `RootCauseAssembler`, or any existing rule file. `PipelineTrace` requires no structural change — `GenerationTraceRecord`'s additive fields (§13) are already inside its existing `generation_trace` field.

---

## 15. Validation — Phase 8

`acceptance_gate.py` is the final gate before a thumbnail ships, and is explicitly additive to, not a replacement for, Module 7's own `hard_gate_passed`:

```python
class AcceptanceResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    accepted: bool
    reasons_rejected: list[str] = []   # e.g. "over_edited", "identity_drift_exceeded",
                                        # "text_obstruction_detected", "did_not_beat_original"
```

Checks, each reusing an existing or §8-defined signal — nothing here recomputes anything:

| Brief's Phase 8 check | Signal reused |
|---|---|
| over-editing | `EditMagnitudeScore.over_edited` (§8.3) |
| identity loss | existing `QualityAssuranceReport.identity_score` |
| poor readability | existing `text_readability_scorer.py` output |
| text obstruction | existing `text_readability_scorer.py` + `TextPlacement` overlap check (Module 10's existing model) |
| low contrast | existing `attractiveness_scorer.py` output |
| incorrect composition | existing `composition_scorer.py` output |
| missing branding | existing `object_preservation_scorer.py`, if branding is modeled as a preserved object (already the case per Module 9's `preserve` directive semantics) |

If `AcceptanceResult.accepted=False` and retry budget remains, control returns to §6 step 3d. If budget is exhausted, the best-available candidate ships with `AcceptanceResult` persisted alongside it — the pipeline never silently claims success on a rejected thumbnail; the rejection is data, not a hidden failure, continuing this project's established norm that a defect should always leave deterministic evidence (as with the PORCE investigation architecture immediately preceding this document).

---

## 16. Testing Strategy (design only)

- Unit tests per new scorer (`beats_original_scorer`, `edit_magnitude_scorer`, `emotional_ctr_scorer`, `whitespace_scorer`) against synthetic `QualityScoringContext` fixtures, following `evaluation/`'s existing scorer test conventions.
- `winner_selector.py` tests using real historical `ImageGenerationResult`/`QualityAssuranceReport` fixtures already available under `data/generated_thumbnails/`, asserting `selection_agrees` correctly flags known-divergent cases.
- `outcome_store.py` aggregation tests with synthetic `OptimizationOutcome` history — pure data-in/data-out, no ML, fully deterministic.
- Golden-file regression tests for `acceptance_gate.py`'s decision table, one fixture per rejection reason in §15's table.
- PORCE rule tests (`RULE-OPT-01..03`) following the exact pattern already established for `RULE-EDIT-01/02`.
- No `gpu`/`integration` markers needed for anything under `optimization/` itself — all of it is pure Python/Pydantic over already-computed scores and already-persisted artifacts, same simplification Module 10 and the PORCE extension both already benefited from.

---

## 17. Migration Plan

1. Ship `comparative/` scorers first, wired only into PVQEF's offline path (`evaluation/cli.py`) — zero production risk, since PVQEF already runs outside the hot path.
2. Validate `baseline_scorer.py`/`beats_original_scorer.py` against the real historical thumbnails already in `data/generated_thumbnails/` before touching the live generation loop — confirm the comparative verdicts match human judgment on a sample before trusting them to gate anything.
3. Introduce `optimization_loop.py` behind a config flag (`OPTIMIZATION_LOOP_ENABLED`, default `False`), calling Module 7 exactly as `main.py` does today, with retry disabled (`OPTIMIZATION_MAX_RETRIES=0`) initially — this validates the wrapping/selection logic without changing generation volume or cost.
4. Enable retries (`OPTIMIZATION_MAX_RETRIES>0`) only after step 3 is validated, and monitor `retry_attempt_count`/generation cost via the new trace fields (§13) before raising the retry budget.
5. Enable `OPTIMIZATION_FEEDBACK_ENABLED` last, and only after enough `OptimizationOutcome` history exists to make priors meaningful (a minimum sample size threshold, e.g. per-niche, should gate this — an empty or tiny outcome store must not silently bias `ConflictResolver` on noise).
6. `acceptance_gate.py` should initially run in **report-only mode** (log `AcceptanceResult` without blocking shipment) before being allowed to trigger retries or withhold a thumbnail, so its false-positive rate can be measured against real output first.

---

## 18. Risks

| Risk | Mitigation |
|---|---|
| Real-time comparative scoring adds latency per candidate, multiplying Module 7's already-measured 25-55s generation time | Keep real-time scorers (§8.1's two new ones) cheap/heuristic by design; reserve full PVQEF `Aggregator` runs for the once-per-video baseline only |
| Retry loop could multiply GPU/ComfyUI load unpredictably on the RTX 4060 laptop's limited VRAM budget | Hard `OPTIMIZATION_MAX_RETRIES` ceiling (§17 step 4), never unbounded; retries reuse Module 7's existing VRAM-aware `ProfileSelector`, not a separate resource path |
| Feedback priors could reinforce an early bad pattern before enough data exists (cold-start bias) | Minimum-sample-size gate before priors are non-zero (§17 step 5); `prior_provider.py` defaults to neutral (no adjustment) below threshold |
| `emotional_ctr_scorer`/`whitespace_scorer` could be mistaken for validated CTR prediction rather than heuristic proxies | Explicitly documented as proxies in §8.1 and in the model docstrings; no ground-truth CTR data exists in this repo to validate against, and this document does not claim otherwise |
| Adding fields to `GenerationTraceRecord`/rules to PORCE could be perceived as "redesigning" observability, which the brief forbids | All changes are strictly additive (new optional fields, new rule files); zero existing field, rule, or class is modified — verified against the exact pattern the brief's own "already completed" PORCE extensions used |
| `Module 8` naming collision (§0) causes confusion if implemented literally under that name | Flagged prominently; implementation should use a distinct package name (`optimization/`, as specified in §5) regardless of what the document is titled |

---

## 19. Future Work

- Once a real transcript/caption ingestion path exists (a prerequisite not currently in this repo, per §9), extend `copywriter.py`'s template system with transcript-derived keyword extraction.
- Once `selection_agrees=False` data (§10) accumulates, consider a scoped, separate architecture pass to tune Module 7's own `MODULE7_QA_WEIGHTS` — explicitly out of scope here per "do not redesign Module 7."
- A true CTR model (as opposed to the heuristic proxy in §8.1) would require an actual outcome data source (real click-through data per thumbnail) that does not exist anywhere in this project today — noted as a standing gap, not something this document can design around.
- Golden-sample-based A/B evaluation (comparing `OPTIMIZATION_FEEDBACK_ENABLED=True` vs `False` cohorts over time) using PVQEF's existing `evaluation/benchmarking/regression_detector.py`, unmodified, once enough production volume exists to make the comparison statistically meaningful.
