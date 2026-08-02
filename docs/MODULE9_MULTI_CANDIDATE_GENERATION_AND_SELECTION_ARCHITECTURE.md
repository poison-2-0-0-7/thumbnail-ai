# Module 9 — Multi-Candidate Generation and Selection Architecture

**thumbnail-ai**
**Status:** Architecture only. Zero implementation code, zero tests, zero repository modification.
**Deliverable filename honored exactly as specified:** `MODULE9_MULTI_CANDIDATE_GENERATION_AND_SELECTION_ARCHITECTURE.md`. **Its content, per §0, positions this work as an extension of Module 7's existing candidate-generation subsystem, not as new code between the real Module 8 and Module 10, and not as a redesign of the real Module 9** — the filename is a deliverable label per instruction, not a claim about where in the pipeline this logic executes.

---

## 0. Grounding note — reconciling the brief's module numbering against `main`

The brief's Background section describes "Module 8" as already providing "Rendering, Optimization, Comparative scoring, Acceptance gate, Feedback recording, GenerationTrace, PORCE diagnostics" and asks this document to design "Module 9" as the layer above it. Checked directly against the repository rather than assumed:

- **The real Module 8 is the Asset Extraction Engine** (`docs/MODULE8_ASSET_EXTRACTION_ENGINE_ARCHITECTURE.md`, `modules/asset_extraction_engine.py`) — person/object/scene/typography/composition extraction from the *original* thumbnail, upstream of generation entirely. It has no relationship to rendering, candidate scoring, or acceptance gates.
- **The real Module 9 is the AI Decision Engine** (`docs/MODULE9_AI_DECISION_ENGINE_ARCHITECTURE.md`, `modules/decision_engine.py`) — per-element `KEEP`/`REPLACE`/`ENHANCE`/`REMOVE`/`ADD` resolution, feeding Module 10's composition workspace. It also has no relationship to candidate generation or ranking.
- **What the brief's Background actually describes — rendering, optimization, comparative scoring, acceptance gate, feedback recording, GenerationTrace, PORCE diagnostics — is Module 7**: `ImageGeneratorPipeline`'s per-candidate loop (rendering), `WorkflowGraphCache`/strategy-pack machinery (optimization, per `MODULE7_PHASE4_MULTI_CANDIDATE_GENERATION_ARCHITECTURE.md`, real and already implemented), `CandidateRanker` (comparative scoring), `QualityAssuranceStage`'s hard-gate logic (acceptance gate), `GenerationTraceRecord` (GenerationTrace), and PORCE (diagnostics). Every one of these already exists, inside Module 7, not as a numbered stage between real-Module-8 and real-Module-10.
- **The brief's stated problem — "the pipeline still assumes a small number of candidates" — is real and independently confirmed**, but not because multi-candidate infrastructure is missing: `modules/config.py` line 569 reads `MODULE7_MAX_CANDIDATES: int = 1`. The entire Phase 4 multi-candidate system (`StrategyPackResolver`, `CandidateStrategyPlanner`, `WorkflowGraphCache`, a real, well-built five-strategy pack at `data/strategy_packs/default_five.json`) is implemented and wired — it is capped at exactly one candidate by a single config constant, the same "infrastructure built, unreachable via a config value" shape found repeatedly across this codebase's history (`MODULE7_PROFILE_PREFERENCE` excluding the edit-capable profile; hardcoded `denoise: 1.0`; hardcoded `latent_source="noise"` in trace capture — each documented in its own prior architecture document under `docs/`).

**This document's scope, accordingly:** it does not propose a new numbered pipeline stage. It extends Module 7's existing, real, already-designed candidate-generation subsystem — reusing `CandidateStrategy`, `StrategyPackResolver`, `CandidateStrategyPlanner`, `WorkflowGraphCache`, `CandidateRanker`, `CandidateScore`, `GenerationTraceRecord`, and PORCE's rule-engine pattern exactly as they exist today — with the new capabilities the brief's seven phases ask for: richer diversity dimensions, near-duplicate clustering, multi-factor ranking, selection explainability, human review, a learning feedback loop, and PORCE rules for this specific failure class. It does not touch the real Module 8 or the real Module 9 anywhere in its design, honoring "do not redesign Modules 1–8" under the repository's actual numbering.

---

## 1. Executive Summary

Module 7's Phase 4 multi-candidate architecture (`docs/MODULE7_PHASE4_MULTI_CANDIDATE_GENERATION_ARCHITECTURE.md`) is real, implemented, and well-designed — a `CandidateStrategy` model with six bias dimensions, a five-strategy default pack, a pure `CandidateStrategyPlanner.derive_package()` function, a per-run `WorkflowGraphCache`, and a `CandidateRanker`. What it does not yet do, verified directly against the code: it never varies `lighting_instructions` (`CandidateStrategyPlanner`'s own documented step 6 explicitly leaves it untouched); it has no mechanism to detect that a batch of "diverse" candidates rendered to near-identical images (no clustering exists anywhere in `modules/`); its ranking is two-dimensional (`overall_score` then `identity_similarity`, `image_generator.py:1230`) rather than the CTR/readability/branding/originality/diversity-weighted ranking professional creative review requires; it records no reasoning for why a candidate won beyond its raw scores; and it has no human-override or feedback-learning path at all. This document designs all five gaps as additive extensions to the existing, real Phase 4 subsystem — plus the specific PORCE rules needed to detect when any of them silently regresses.

---

## 2. Root Cause Analysis (Phase 0 audit)

**Where candidates are generated:** `ImageGeneratorPipeline.run()`'s per-candidate loop (`image_generator.py`), iterating `strategies = self.strategy_pack_resolver.resolve(...)`. Confirmed correctly wired end-to-end per `MODULE7_PHASE4_MULTI_CANDIDATE_GENERATION_ARCHITECTURE.md` §2/§7 — this is not the gap.

**Where the candidate count is actually constrained:** `MODULE7_MAX_CANDIDATES: int = 1` (`config.py:569`). `StrategyPackResolver.resolve(requested_pack, max_candidates)` truncates any resolved pack to this ceiling regardless of how many strategies the pack itself defines — `default_five.json` defines five strategies, but only the first (`"faithful"`, by file order) is ever actually used in production today. This is the primary, single, root cause of "the pipeline still assumes a small number of candidates," and it is a one-line config value, not a missing architecture — this document does not need to design new generation machinery to fix the *count*; it needs to design what happens once the count is legitimately raised (diversity quality, dedup, ranking, review), which is where genuine gaps exist.

**Where diversity is lost:** two confirmed, specific mechanisms, not a vague "needs more variety":
1. `CandidateStrategyPlanner.derive_package()`'s own documented behavior (§4.3 of `MODULE7_PHASE4_MULTI_CANDIDATE_GENERATION_ARCHITECTURE.md`, step 6, verified against `generation_components/candidate_strategy_planner.py`) explicitly leaves `lighting_instructions`' core content untouched across every strategy — **lighting variation, one of the seven dimensions this brief's Phase 1 explicitly asks for, is structurally absent today**, not merely under-tuned.
2. `default_five.json`'s bias magnitudes are modest by design (e.g. `color_grade_bias: 0.25`, `object_emphasis_bias: -0.15`) — appropriate for staying within `IdentityPreservationStage`/QA gates (§3 of the Phase 4 doc's gap analysis explicitly notes strategies must operate *within* existing gates, not around them), but with no clustering step to verify the resulting *images* actually diverge, there is no closed-loop guarantee that five distinct instruction sets produce five visually distinct outputs — two strategies could legitimately converge to visually similar results (e.g. `faithful` and a low-magnitude variant, under a seed/prompt combination where the bias has little visible effect) and nothing in the current pipeline would notice.

**Where duplicate candidates occur:** nowhere is currently checked. `grep -rln "cluster|near.duplicate|perceptual_hash|phash|dedup" modules/` returns zero matches — there is no mechanism anywhere in this repository, today, that compares two generated candidate images to each other. Every candidate that passes `QualityAssuranceStage`'s hard gate is ranked and one is selected; two visually-identical passing candidates are ranked as if they were meaningfully different options.

**Where candidate comparison is insufficient:** `CandidateRanker.rank()` (`image_generator.py:1217-1248`) sorts strictly by `(-overall_score, -identity_similarity, candidate_index)` — `overall_score` is itself `MODULE7_QA_WEIGHTS`'s existing six-dimension composite (`identity_score`, `face_quality_score`, `composition_score`, `text_safe_zone_score`, `object_preservation_score`, `color_compliance_score`, `config.py:541-545`), which measures **fidelity to the source and internal image quality** — it has no dimension for CTR potential, readability *of the redesigned headline specifically*, brand-asset presence, originality relative to the batch, or a diversity bonus. This is architecturally sound for what it was built to do (a QA/acceptance gate) but is not, and was never designed to be, a creative-comparison ranking across a diverse candidate set — the brief's Phase 3 requirement is a genuinely new capability layered on top, not a fix to a broken existing one.

---

## 3. Current Candidate Pipeline

Reproduced from `MODULE7_PHASE4_MULTI_CANDIDATE_GENERATION_ARCHITECTURE.md` §2/§4.1, verified current on `main`, and used as this document's unmodified foundation:

```
ConditioningAssetResolver.resolve()                          [once per run]
StrategyPackResolver.resolve(requested_pack, MODULE7_MAX_CANDIDATES)   [→ len 1, today]
for cand_idx, strategy in enumerate(strategies):
    CandidateStrategyPlanner.derive_package(base_package, blueprint, strategy, cand_idx)
    WorkflowBuilder.build(cand_package, ..., cache=WorkflowGraphCache)
    ComfyUIClient.generate()
    IdentityPreservationStage.verify()  (+ bounded retries)
    FaceRestorationStage.restore()
    BackgroundCompositor.composite()
    UpscaleStage.upscale()
    QualityAssuranceStage.evaluate()    → CandidateScore(overall_score, identity_similarity, hard_gate_passed)
CandidateRanker.rank()                  → sort by (-overall_score, -identity_similarity, cand_idx); mark selected
ArtifactWriter.write_manifest()
MetricsCollector.append()
GenerationTraceFactory.create() (per candidate)                        [PORCE, per prior documents]
```

Nothing in §3 is redesigned by this document (per the strict rules and per §0's scope statement) — every new component in §5–§10 attaches to this sequence at named, additive insertion points.

---

## 4. Proposed Architecture

```
StrategyPackResolver.resolve(...)              [existing — extended: richer packs, §5]
    │
    ▼
CandidateStrategyPlanner.derive_package(...)    [existing — extended: lighting/framing dims, §5]
    │
    ▼  (per-candidate generation loop, unchanged, §3)
    │
    ▼
CandidateClusteringEngine.cluster(candidates)   [NEW, §7]
    │  drops/flags near-duplicates before ranking
    ▼
CandidateRankingEngine.rank(surviving_candidates)  [NEW — supersedes CandidateRanker's sort key,
    │                                                  reuses CandidateRanker's hard-gate/tie-break shape, §6]
    ▼
SelectionExplainer.explain(ranked, winner)      [NEW, §8/§9]
    │
    ├──▶ GenerationTraceRecord extension (§9)
    ├──▶ PORCE rules (§10)
    └──▶ HumanReviewWorkspace (optional, §8)
              │
              ▼
         ManualSelectionRecord (if overridden) ──▶ LearningFeedbackStore (§6.4)
```

Every "NEW" box is additive — none replaces `CandidateRanker`'s existing hard-gate/tie-break contract; `CandidateRankingEngine` is designed in §6 as a strict superset that falls back to today's exact ordering when the new scoring inputs are unavailable (§11's backward-compatibility requirement).

---

## 5. Diversity Engine (Phase 1)

**Extends, does not replace, `CandidateStrategy`/`CandidateStrategyPlanner`** (§0 — reuse mandate). Two additive changes:

### 5.1 New bias dimensions on `CandidateStrategy`

Two new optional, defaulted fields (`lighting_bias: float = 0.0`, `framing_bias: float = 0.0`) — additive to the frozen Pydantic model, matching the exact pattern every other bias field already uses (bounded float, applied via `CandidateStrategyPlanner`'s existing "small, declarative phrase-append rules" mechanism, §7.2 step 4 of the Phase 4 document, reused verbatim rather than inventing a new instruction-modification mechanism):

- `lighting_bias` → applied to `lighting_instructions` (currently untouched, §2's confirmed gap) via the same bounded phrase-append pattern already used for `background_instructions`/`color_instructions` (e.g. `", dramatic rim lighting and deeper shadow contrast"` at positive bias, `", softer even lighting"` at negative bias — one phrase per dimension, capped, deterministic, auditable — no free-text generation).
- `framing_bias` → applied to `object_placement`/composition instructions as a **distinct** dimension from the existing `camera_distance_shift` (which governs zoom/distance): `framing_bias` governs off-center/rule-of-thirds placement bias versus centered framing, addressing "framing variation" as the brief names it, separately from "composition variation" (which `object_emphasis_bias`/`camera_distance_shift` already cover).

### 5.2 New strategy packs (additive files, not a change to the resolver)

`StrategyPackResolver`'s existing `resolve()` contract (§7.1 of the Phase 4 document) already supports named packs beyond `default_five` with zero code change — a new pack is a new JSON file under `data/strategy_packs/`. This document proposes (as data, not code) two additional packs following `default_five.json`'s exact schema:

- `lighting_and_framing_focus.json` — strategies isolating `lighting_bias`/`framing_bias` at meaningful magnitude while holding other dimensions near zero, so a batch using this pack cleanly demonstrates the two dimensions §5.1 adds, useful for validating §5.1 lands correctly (§12) independent of ranking/clustering changes.
- `full_spectrum_eight.json` — an eight-strategy pack (up to a raised `MODULE7_MAX_CANDIDATES`, §11) spanning all eight bias dimensions in combination, the pack this document expects production use to converge on once §7's clustering guarantees the larger candidate count doesn't waste generation budget on near-duplicates.

No change to `StrategyPackResolver.resolve()`'s signature or fallback behavior (`requested_pack is None` still returns the single in-code `faithful_default()`, byte-identical to today, per §7.1 of the Phase 4 document — untouched).

---

## 6. Ranking Engine (Phase 3)

### 6.1 New dimensions, each reusing an existing signal where one exists

| Dimension | Source | Reuse, or genuinely new? |
|---|---|---|
| Module 7 QA (existing) | `CandidateScore.overall_score` | Reused unchanged — remains the acceptance-gate signal, not replaced |
| CTR score | New, deterministic — **not** a new ML model (project-wide deterministic-first convention, honored here). Computed from already-available signals: strategy metadata (`aggressive_ctr`-family strategies get a documented base contribution), `QualityAssuranceStage`'s existing `composition_score`/`color_compliance_score` sub-scores, and — reusing an established precedent from a different module rather than inventing scoring logic from scratch — the same weighted-composite *pattern* Module 5's `HeadlineCandidate.ctr_potential_score`/`.composite_score` (`modules/models.py:670-684`) already uses for headline-text CTR scoring. This document proposes the *image-level* CTR score follow that already-proven pattern's shape (small, named, weighted sub-signal table), not that it reuse the headline-specific computation itself. | New, but pattern-reused |
| Readability | Reuses `_calculate_text_safe_zone_score` if/when it reflects real OCR-based measurement (flagged as a separately-tracked, already-documented gap in `PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md` §3/§11 — not redesigned here) | Reused as-is, whatever its current fidelity |
| Branding consistency | Reuses `_calculate_color_compliance_score` plus a new, additive check: does the candidate still contain every `AssetExtractionManifest` element flagged as a brand asset (logo/watermark) with a `KEEP` decision from the real Module 9 (Decision Engine) — i.e., cross-referencing already-existing `DecisionManifest` data, not new asset detection | Reused signal + one new cross-reference, no new detection model |
| Originality | New — a per-batch measure: perceptual distance (§7.1) from every *other* candidate in the same batch, aggregated (mean pairwise distance) — directly composed with the clustering engine's own distance computation, not a separate computation | New, but computed from §7's clustering machinery, not a parallel one |
| Diversity bonus | New — a small, capped bonus applied to a candidate whose strategy occupies an underrepresented region of the pack's bias-vector space relative to the rest of the surviving (post-clustering) batch — rewards genuine creative-direction spread, not just per-candidate quality | New |

### 6.2 `CandidateRankingEngine` (new, `modules/generation_components/candidate_ranking_engine.py`)

A strict superset of `CandidateRanker`, not a replacement class — `CandidateRanker` remains, unmodified, as the QA hard-gate/tie-break authority (§11's backward-compatibility requirement depends on this); `CandidateRankingEngine` wraps it: candidates that fail `CandidateRanker`'s existing hard gate are excluded before any new dimension is even computed (the acceptance gate remains absolute, exactly as today — no new dimension can rescue a hard-gate failure, matching the brief's own "acceptance gate" framing as a prior, separate concept from ranking).

**Weighted composite**, following the exact `MODULE7_QA_WEIGHTS`-style config-table convention (`config.py`, new `MODULE7_CANDIDATE_RANKING_WEIGHTS: dict[str, float]`, validated-sums-to-1.0 via the same `validate_qa_weights()`-shaped function already established):

```
MODULE7_CANDIDATE_RANKING_WEIGHTS = {
  "qa_overall_score": 0.35, "ctr_score": 0.20, "readability_score": 0.15,
  "branding_consistency_score": 0.15, "originality_score": 0.10, "diversity_bonus": 0.05,
}
```

Tie-break: identical to `CandidateRanker`'s existing rule (`-identity_similarity`, then `candidate_index`) — reused verbatim, not redesigned.

### 6.3 Backward compatibility (Phase 6's "extend existing optimization" rule, made concrete)

When `MODULE7_MAX_CANDIDATES == 1` (today's default, §2) — or more generally, whenever a batch has exactly one surviving candidate after §7's clustering — `CandidateRankingEngine` short-circuits to `CandidateRanker`'s exact existing behavior (a single candidate, ranked and selected trivially) with **zero new computation performed**, since CTR/originality/diversity-bonus are only meaningful relative to a batch of two or more. This guarantees the new engine is a no-op, byte-for-byte, on every production run until `MODULE7_MAX_CANDIDATES` is deliberately raised — the same "additive, inert until configured" guarantee every prior document in this repository's `docs/` tree has required of its own changes.

### 6.4 Learning (Phase 6)

`LearningFeedbackStore` (new, `data/candidate_feedback/`, append-only JSONL — mirroring `module7_metrics.jsonl`'s existing append-only convention rather than inventing a new persistence shape) records, per completed run with a human override (§8): the algorithmic winner, the human-selected winner (if different), and the full per-candidate dimension breakdown for both. **This document deliberately does not propose an online/automatic reinforcement-learning loop** — consistent with the project's deterministic-first character established across every prior architecture document — instead, `MODULE7_CANDIDATE_RANKING_WEIGHTS` (§6.2) is designed to be periodically, manually recalibrated by a developer reviewing `LearningFeedbackStore`'s aggregated override patterns (e.g. "when humans override, they favor `branding_consistency_score` more than the current weight reflects" → a reviewed, committed config change) — the same "data, not code, reviewable, no silent behavior drift" posture every ranking/QA weight table in this repository already follows.

---

## 7. Candidate Clustering (Phase 2)

### 7.1 `CandidateClusteringEngine` (new, `modules/generation_components/candidate_clustering_engine.py`)

Perceptual-hash-based (pHash/dHash via Pillow, already a vendored dependency — no new heavy dependency, consistent with §16's "reuse existing infrastructure" mandate), not embedding/ML-based, for the same deterministic-first reason every scoring component in this repository is deterministic-first: `cluster(candidate_paths: list[Path]) -> ClusteringResult`, computing pairwise Hamming distance between each candidate's perceptual hash, grouping candidates within `MODULE7_CLUSTERING_DISTANCE_THRESHOLD` (new config constant, small closed-table convention) of each other into a cluster.

**Within a cluster:** only the highest-`overall_score` (§3's existing QA composite, computed before clustering, since QA is orthogonal to visual similarity) member survives to ranking (§6); the rest are marked `excluded_reason="near_duplicate_of_{winning_candidate_id}"` in `CandidateScore` (an additive, optional field) rather than deleted — their generation cost was already spent and their data remains available for §9's trace/§10's PORCE diagnostics, just excluded from §6's ranking pool.

**Whole-batch diversity check:** if, after clustering, fewer than a configured minimum fraction of the original batch survives (`MODULE7_MIN_SURVIVING_CANDIDATE_FRACTION`, new config constant), this is recorded as a `weak_diversity` fact (§10) — the strategy pack in use produced insufficient real visual variation for this particular source thumbnail/prompt combination, a diagnosable, PORCE-visible condition rather than a silent quality loss.

### 7.2 Ordering relative to §6

Clustering runs **before** ranking (§4's diagram), not after — ranking a near-duplicate cluster as if each member were an independent creative option would inflate the diversity-bonus dimension incoherently (§6.1's diversity bonus explicitly depends on the *post-clustering* surviving set, not the raw batch) and would let two visually-identical candidates occupy both top ranking slots, defeating the purpose of the brief's own "guarantee diversity" requirement.

---

## 8. Winner Selection (Phases 4–5 combined, per the brief's own section list)

### 8.1 Algorithmic selection (default path, unchanged in character from today)

`CandidateRankingEngine.rank()` (§6) produces an ordered list; the top entry is `selected=True`, exactly matching `CandidateScore.selected`'s existing field and `CandidateRanker`'s existing selection semantics (§3) — no new selection *mechanism*, only a richer *ranking* feeding the same mechanism.

### 8.2 `SelectionExplainer` (new, Phase 4 — Selection Explainability)

Produces a `SelectionExplanation` (new, frozen model): `winner_candidate_index`, `winning_margin` (winner's composite score minus runner-up's), `dominant_dimensions` (which 1–2 weighted dimensions contributed most to the winner's lead — a simple weighted-contribution decomposition of §6.2's composite, not a new model, fully deterministic and reproducible from `MODULE7_CANDIDATE_RANKING_WEIGHTS` and each candidate's per-dimension scores), `excluded_candidates_summary` (how many were dropped by clustering and why, §7.1), `plain_language_summary: str` (a template-filled sentence, e.g. *"Candidate 3 selected: highest CTR score (0.82) and strongest branding consistency (0.91); candidates 1 and 4 were excluded as near-duplicates of candidate 3."* — templated from the structured fields above, not free-text-generated, matching the project's consistent "deterministic template, not LLM narration" convention for any human-facing explanation text).

### 8.3 Human Review Mode (Phase 5)

`MODULE7_HUMAN_REVIEW_ENABLED: bool = False` (new, default off — additive, non-breaking per §11). When enabled, `ArtifactWriter` (existing, extended additively) persists all surviving candidates + `SelectionExplanation` to a `HumanReviewWorkspace` directory (`data/candidate_review/{video_id}/`, mirroring the project's existing `data/{module}/{video_id}/` convention) instead of immediately finalizing the algorithmic winner. A reviewer's selection (via whatever interface consumes this workspace — out of scope for this document to design, per "do not implement," but the workspace's file contract — a simple `candidates/`, `explanation.json`, `selection.json`-to-be-written layout — is the integration point) is captured as a `ManualSelectionRecord` (new, frozen): `video_id`, `algorithmic_winner_index`, `human_winner_index`, `reviewer_note: str | None`, `timestamp`. If `human_winner_index != algorithmic_winner_index`, the human selection becomes the pipeline's final output — an override, not a suggestion — and the record feeds `LearningFeedbackStore` (§6.4). If human review is enabled but no selection is recorded within a configurable timeout, the algorithmic winner stands by default (graceful degradation, matching every other optional-stage precedent in this repository, e.g. `IdentityPreservationStage.skipped`).

---

## 9. GenerationTrace Integration

`GenerationTraceRecord`/`FragmentAttachmentRecord`'s established additive-extension pattern (already used twice — `PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md`'s original design, and `MODULE7_CONTROLNET_COMPATIBILITY_ARCHITECTURE.md`'s capability-resolution fields) is reused a third time, not redesigned. New, optional, additive fields on `GenerationTraceRecord` (old records unaffected, exactly per precedent):

```
strategy_name: str | None                    # which CandidateStrategy produced this candidate
cluster_id: str | None                        # which near-duplicate cluster this candidate belonged to
excluded_reason: str | None                   # e.g. "near_duplicate_of_candidate_2", mirrors CandidateScore's new field (§7.1)
ranking_dimension_scores: dict[str, float] | None   # per-dimension scores feeding §6.2's composite
selection_explanation: SelectionExplanation | None  # attached only to the winning candidate's record
manual_override: ManualSelectionRecord | None       # attached only when human review (§8.3) overrode the winner
```

This is the mechanism that satisfies the brief's Phase 4 requirement ("record why one candidate wins... include detailed reasoning in GenerationTrace") using the exact, already-established extension point rather than a new artifact type.

---

## 10. PORCE Integration (Phase 7)

Four new rules, following `RULE-EDIT-02`/`RULE-EDIT-04`'s exact class/interface shape (`observability/diagnostics/rules/`, new sibling file `candidate_selection_rules.py`), reusing `IDiagnosticRule`/`Finding`/`RuleContext` unchanged:

```
RULE-CAND-01 — Duplicate Candidate Detection
  Fires WARNING when a batch's cluster count is materially lower than its requested
  candidate count (§7.1's weak_diversity fact) — surfaces that generation budget was
  spent on visually redundant candidates.

RULE-CAND-02 — Weak Diversity
  Fires WARNING when mean pairwise perceptual distance across a surviving (post-cluster)
  batch falls below a configured floor even without triggering RULE-CAND-01's cluster-count
  threshold — catches "technically distinct, practically similar" batches, a softer signal
  than outright duplication.

RULE-CAND-03 — Inconsistent Ranking
  Fires WARNING when the algorithmic winner does not have the highest qa_overall_score
  (§6.1) among surviving candidates, i.e. a non-QA dimension outweighed QA in the final
  decision — not necessarily wrong (that's the point of a multi-factor ranking), but
  worth surfacing explicitly so a developer reviewing output quality understands *why*
  the top-QA candidate wasn't chosen, using SelectionExplanation's dominant_dimensions
  (§8.2) directly as the finding's supporting evidence.

RULE-CAND-04 — Poor Winner Selection
  Fires FAIL when the selected winner (algorithmic or human-overridden) failed any
  Tier-1 QA hard gate that a non-selected surviving candidate passed — a correctness
  check on the acceptance-gate/selection boundary itself (§6.2's stated invariant that
  hard-gate-failed candidates are excluded before ranking); firing at all indicates that
  invariant was violated somewhere in the implementation, not a creative-quality judgment.
```

Each rule operates over `PipelineTrace`/`TraceFacts` exactly as `PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md` §9 already established — no new fact-derivation mechanism, only new fact fields (§9's `GenerationTraceRecord` extensions) for existing rule-registration machinery to consume.

---

## 11. Validation

- **Backward compatibility (the strict-rules mandate, made concrete and checkable):** `MODULE7_MAX_CANDIDATES=1` (today's value) must produce a pipeline run with zero calls into `CandidateClusteringEngine`'s pairwise-distance logic beyond a trivial single-element no-op, zero new `MODULE7_CANDIDATE_RANKING_WEIGHTS`-dimension computation (§6.3), and a `GenerationTraceRecord` whose new fields (§9) are all `None` except `strategy_name` — verifiable as a literal diff-based test (§12), not a design assertion.
- **Startup validation, following the `validate_module7_edit_reachability()`/`validate_controlnet_capability_availability()` precedent (both already established in this repository's `docs/` tree):** a new `validate_candidate_ranking_weights()` asserting `MODULE7_CANDIDATE_RANKING_WEIGHTS` sums to 1.0, at the same startup checkpoint as its two precedents.
- **Cluster-threshold sanity:** `MODULE7_CLUSTERING_DISTANCE_THRESHOLD` and `MODULE7_MIN_SURVIVING_CANDIDATE_FRACTION` should be validated at startup to be within sane bounds (e.g. threshold > 0, fraction ∈ (0, 1]) — a simple range check, not a new validation paradigm.

---

## 12. Testing Strategy

Mirrors this repository's established 1:1 module-to-test-file convention:

- `test_candidate_strategy_planner.py` (extended, not replaced): new cases for `lighting_bias`/`framing_bias` (§5.1), asserting `lighting_instructions` and `object_placement` actually change while every other field is untouched, per the pure-function contract §7.2 of the Phase 4 document already established.
- `test_candidate_clustering_engine.py` (new): fixture pairs of genuinely-identical and genuinely-different images (synthetic, checked-in), asserting correct cluster grouping and correct `overall_score`-based survivor selection within a cluster.
- `test_candidate_ranking_engine.py` (new): table-driven per dimension, plus the §6.3 backward-compatibility regression test (single-candidate batch produces `CandidateRanker`-identical output).
- `test_selection_explainer.py` (new): asserts `dominant_dimensions`/`plain_language_summary` are deterministic and reproducible from a fixed input, and — critically — that the summary sentence's stated facts (scores, exclusion reasons) match the structured fields exactly (a template-consistency test, not a prose-quality test).
- `test_candidate_selection_rules.py` (new): one fixture per rule (§10), including a fixture directly reproducing §2's confirmed "lighting never varies" gap as a `RULE-CAND-02`-triggering case, following this repository's now-established practice (per `MODULE7_RENDER_EXECUTION_ARCHITECTURE.md` §20, `MODULE7_CONTROLNET_COMPATIBILITY_ARCHITECTURE.md` §14) of anchoring at least one regression test per rule to a real, confirmed finding rather than only a synthetic case.
- Backward-compatibility test (repository-wide precedent, reused verbatim): given `MODULE7_MAX_CANDIDATES=1`, assert the full pipeline's output is byte-for-byte identical to a pre-this-document baseline.

---

## 13. Migration Plan

Fully additive, following the exact phasing discipline every prior document in this repository's `docs/` tree has used — no phase requires an existing test to change, and `MODULE7_MAX_CANDIDATES` is **not** raised as part of this document's own migration (raising it is an operational/config decision for whoever deploys this work, made safe only once Phases 1–4 below have shipped and been validated at the current cap of 1, consistent with §11's backward-compatibility validation running first):

| Phase | Change | Breaking? |
|---|---|---|
| 1 | `CandidateStrategy` new fields (§5.1), new strategy pack files (§5.2) — inert until a pack referencing them is requested | No |
| 2 | `CandidateClusteringEngine` (§7) — new, unreachable until wired into the loop | No |
| 3 | `CandidateRankingEngine` + `MODULE7_CANDIDATE_RANKING_WEIGHTS` (§6) — new, `CandidateRanker` remains the active path until wired | No |
| 4 | `SelectionExplainer` + `GenerationTraceRecord` extensions (§8.2, §9) | No |
| 5 | PORCE rules (§10) registered | No |
| 6 | `HumanReviewMode`/`ManualSelectionRecord`/`LearningFeedbackStore` (§8.3, §6.4), flag-gated off by default | No |
| 7 (operational, not architectural) | `MODULE7_MAX_CANDIDATES` raised beyond 1, informed by real §5–§7 output review | The point at which new behavior first becomes visible in production — deliberately the last step, not the first |

---

## 14. Risks

| Risk | Detail | Mitigation |
|---|---|---|
| Raising `MODULE7_MAX_CANDIDATES` multiplies ComfyUI generation time/VRAM cost linearly | A genuinely real cost, not a design defect | `WorkflowGraphCache` (existing, §3) already amortizes the template/fragment-assembly portion; §7's clustering ensures the marginal candidates are not wasted on redundant output; this document does not itself decide the new cap value — that is an operational trade-off for whoever deploys Phase 7 of §13 |
| Perceptual hashing (§7.1) may not reliably distinguish semantically-different-but-visually-similar candidates (e.g. same composition, different headline text only) | pHash/dHash are pixel-structure-based, not semantic | Documented limitation, not silently assumed away; `MODULE7_CLUSTERING_DISTANCE_THRESHOLD` should be tuned conservatively (favor under-clustering over over-clustering) since a missed near-duplicate costs only ranking-list clutter, while an incorrectly-merged genuine variant costs a real creative option |
| CTR score (§6.1) is a new, un-validated-against-real-outcomes heuristic | No live CTR feedback loop exists in this project (§6.4 deliberately does not propose building one) | Weighted low relative to QA (§6.2's proposed 0.20 vs. 0.35) until `LearningFeedbackStore` (§6.4) accumulates enough human-override data to justify recalibration |
| `MODULE7_CANDIDATE_RANKING_WEIGHTS` could drift out of sync with `MODULE7_QA_WEIGHTS` if the two are tuned independently without a shared review process | Two adjacent-but-distinct weight tables, both human-tunable | Documented explicitly here as a coordination risk; both tables' `validate_*_weights()` functions (§11) should be reviewed together whenever either changes, a process note for implementers rather than a code-level coupling (deliberately not coupling the two tables' values together, since QA and creative ranking are legitimately different concerns, §6.1) |

---

## 15. Future Work

- **Real CTR outcome data.** If this project ever gains a live feedback channel (e.g. creators reporting which delivered thumbnail they used, or downstream performance data), `LearningFeedbackStore` (§6.4) is the natural ingestion point for recalibrating `ctr_score`'s weighted sub-signals against ground truth, rather than the deterministic heuristic proposed here — explicitly deferred, since no such channel currently exists in this repository.
- **Cross-video diversity tracking.** `PIPELINE_OBSERVABILITY_ROOT_CAUSE_ENGINE_ARCHITECTURE.md` §24's "cross-video pattern mining" future extension is the natural home for a batch-level version of this document's originality dimension (§6.1) — has this creator's last N thumbnails all converged on the same strategy, suggesting the pack itself needs more range for their niche — not designed here.
- **Automatic strategy-pack authoring.** `full_spectrum_eight.json` (§5.2) is hand-authored; a future extension could propose new bias combinations automatically from `LearningFeedbackStore`'s override patterns (§6.4) — explicitly deferred as a second-order extension of a system (§6.4) this document already scopes conservatively.
