# MODULE10_CREATOR_STYLE_LEARNING_ARCHITECTURE.md

**Status:** Architecture only. No implementation code, no tests, no repository files modified.
**Source of truth:** `poison-2-0-0-7/thumbnail-ai`, freshly pulled (no new commits since the previous architecture pass — the Optimization Layer design has not yet been implemented) and re-read for this task.
**Author role:** Lead Software Architect.

---

## 0. Naming Conflict — flagged, not resolved here

`Module 10` already exists (`docs/MODULE10_ASSET_COMPOSER_ARCHITECTURE.md`, `modules/composition_engine.py`) and does something unrelated (pixel-space layer/mask composition for a single video's generation). This document uses the requested filename verbatim and refers to the new system by role ("the Style Layer") internally, exactly as the two prior naming collisions in this project (Module 10/Asset Composer vs. Email Assembler; Module 8/Asset Extraction vs. Quality Optimization) were handled — flagged, renumbering left to you.

---

## 1. Executive Summary

The pipeline already extracts almost every raw signal this brief's Phase 1 asks for — color, composition, faces, text regions — **per video**, via Module 4's `ThumbnailIntelligence`. It already has an image-embedding model (`OpenCLIPWrapper.encode_image()` / `compute_similarity()`) sitting unused in `modules/vision_stack/` outside two PVQEF scorers. It already extracts `channel_id` for every video (Module 2). **None of it is ever grouped by creator.** Every artifact in this codebase — from `data/thumbnails/{video_id}.jpg` to the newest PORCE trace — is sharded by `video_id` alone. `main.py`'s outer loop (`for creator in creators:`) processes each CSV row as a fully independent unit with zero state carried across rows, even when the same `email`/`channel_id` appears twice (confirmed: it already does, in the real `data/creators.csv`). That is the entire root cause, and it is structural, not a missing algorithm: **the data to build a creator style profile already exists; there is no place it is ever accumulated.**

This document designs the accumulation layer (a new `channel_id`-sharded store, built entirely from existing per-video artifacts) plus the specific new capabilities that don't yet exist anywhere: a compact style representation, a similarity/drift check reusing the existing CLIP wrapper, and the prompt/ranking/trace/PORCE integration points needed to act on it — all additive, none of it touching Modules 1–9's existing logic.

---

## 2. Root Cause Analysis (Phase 0)

Audited directly against the running code, not the docs:

1. **`channel_id` is extracted and then discarded.** `VideoMetadata.channel_id: str` (`modules/models.py`) is populated by Module 2 (`youtube_metadata.py::extract_metadata`) for every video. It is never read again by any downstream module — confirmed by tracing every `channel_id` reference in the codebase (`models.py`, `thumbnail_intelligence.py`, `youtube_metadata.py` — the field is defined and set, never joined against anything).
2. **Every persisted artifact in the entire repository is sharded by `video_id`, with no exceptions.** `data/thumbnails/`, `data/redesign_specs/`, `data/prompt_packages/`, `data/visual_references/`, `data/composition_workspaces/`, `data/generated_thumbnails/`, `data/observability/generation_traces/`, `data/optimization/outcomes/` (per the previous architecture pass) — all keyed by `video_id`. There is no `data/creator_profiles/` or equivalent anywhere.
3. **`main.py`'s orchestration loop has no cross-row memory.** `for creator in creators:` (line 238) processes `load_all_creators(csv_path)` sequentially with no accumulator, no lookup, no state object that survives past a single iteration. Two rows in the real `data/creators.csv` already share the same `email` (`Test@gmail.com`, two different `video_url`s) — today, the pipeline generates two completely independent thumbnails for that creator with zero awareness they belong to the same person.
4. **The one embedding model in the repo that could support style similarity (`OpenCLIPWrapper`) is wired into exactly two consumers — both inside PVQEF's `evaluation/quality/` (attractiveness/prompt-adherence scoring) — never into any creator-identity or style-matching path.**
5. **There is no mechanism to fetch a creator's other videos from YouTube at all.** `youtube_metadata.py` calls yt-dlp for exactly one `video_url` at a time (`_fetch_yt_dlp_info`); there is no channel-listing/playlist call anywhere in the codebase. This means the *only* source of "historical thumbnails" available to this system, without adding a new external-API capability (out of scope for an architecture built on "reuse existing systems"), is **the creator's own accumulated footprint across multiple pipeline runs over time** — i.e., every time the same `channel_id` is processed again, that's one more data point. This is a real, load-bearing constraint on the whole design, not a detail: style learning here is necessarily incremental and slow-starting, not a one-shot backfill from a creator's full YouTube history.
6. **No dedicated logo/watermark or typography-style (font family) detector exists.** `objects: list[DetectedObject]` (Module 4, YOLO-based) can incidentally catch a logo if YOLO's generic classes happen to include it, and `branding_constraints: list[str]` (Module 5.5's `derive_branding_constraints()`) is a short rule-derived text list, not a learned per-creator logo template or placement model. "Typography" in this repo today means `OCRResult.text_regions` (where text sits, how much of the frame it covers) — not font family/weight, which no component detects. Both are flagged as genuine gaps in §4, not assumed solved.

**Conclusion:** "every creator is treated almost the same" is literally true by construction — the pipeline has no concept of "creator" as a persistent entity anywhere in its data model or storage layout. This is not a modeling failure to fix inside Modules 1–9 (forbidden anyway, per the brief); it's a missing layer above them.

---

## 3. Current Pipeline Review

```
data/creators.csv  (email, video_url — no explicit style/channel grouping used downstream)
        │
        ▼  per-row, independently, no cross-row state
Module 2   VideoMetadata.channel_id extracted, never reused
Module 3   Thumbnail Downloader          → data/thumbnails/{video_id}.jpg
Module 4   Thumbnail Intelligence         → ThumbnailIntelligence{ocr, faces, objects, colors, composition}
                                            per video_id, never aggregated
Module 5   Redesign Spec  →  Module 5.5 Copywriter/Layout  →  Module 6 Prompt Compiler
Module 6.5 Visual Reference Engine
Module 8   Asset Extraction
Module 9   Decision Engine
Module 10  Asset Composer  (existing — see §0)
Module 10.5 Thumbnail Planner
Module 7   Generation + Optimization (per the prior architecture pass) + PORCE trace
        │
        ▼
data/generated_thumbnails/{video_id}/...   — no creator-level record anywhere
```

Everything needed for Phase 1 extraction already flows through Module 4's output per video. The gap is entirely at the top (no grouping key used) and the bottom (no persistent creator-level store).

---

## 4. Style Extraction (Phase 1)

**Design principle: extract nothing new that Module 4 already computes; aggregate what exists, and add only the two genuinely missing signals (face scale as a normalized ratio, and the two flagged gaps below).**

| Brief dimension | Source | Status |
|---|---|---|
| colors | `ThumbnailIntelligence.colors` (`ColorProfile`: `dominant_colors`, `brightness`, `contrast`, `saturation`, `warm_or_cool`, `harmony_score`) | Fully covered, reused as-is |
| contrast | `ColorProfile.contrast` | Fully covered |
| lighting | `ColorProfile.brightness` + `warm_or_cool` (proxy — no dedicated lighting-direction model exists in the repo; same honest proxy framing used for "lighting" throughout this project's prior documents) | Covered as proxy |
| layout | `ThumbnailIntelligence.composition` (`CompositionAnalysis`: `subject_placement`, `negative_space_ratio`, `balance_score`, `symmetry_score`, `rule_of_thirds_score`) | Fully covered |
| object placement | `ThumbnailIntelligence.objects: list[DetectedObject]` (existing bbox + class per object) | Fully covered |
| clutter | `CompositionAnalysis.clutter_score` | Fully covered |
| face scale | `ThumbnailIntelligence.faces.faces` (per-face bbox) — normalized as `face_bbox_area / frame_area`, a derived ratio, not a new detector | Fully covered, one small derived field |
| background style | `ColorProfile` computed over VRE's existing `background.png` region (Module 6.5) rather than the whole frame — reuses VRE's existing background/foreground segmentation, does not re-segment | Fully covered via existing VRE asset |
| typography (text placement/coverage) | `ThumbnailIntelligence.ocr` (`OCRResult`: `text_regions`, `text_coverage_ratio`, `word_count`) | Placement/coverage covered; **font family/weight — genuine gap, no detector exists in repo, not designed around here** |
| logo placement | Best-effort via `DetectedObject` entries whose YOLO class is logo-adjacent, when present | **Partial — no dedicated logo detector; flagged, not invented** |

### 4.1 Extraction Pipeline

```
StyleExtractor.extract_signature(video_id) -> ThumbnailStyleSignature
    reads (all existing, read-only):
      ThumbnailIntelligence   (Module 4, from data/redesign_specs-adjacent cache or re-load)
      VisualReferenceManifest (Module 6.5, for background-region color)
    produces:
      ThumbnailStyleSignature  (§5) — one per video_id, same cadence Module 4 already runs at
```

This is a **read-only derivation step**, not a new CV stage — it runs after Module 4 (and optionally after Module 6.5, if background-only color is desired) and requires no new model, no GPU beyond what's already loaded for those modules.

---

## 5. Style Representation (Phase 2)

Two representations, serving different consumers, both derived from the same extraction step — deliberately not conflated, matching the "real-time vs. offline, different audiences" split already established for quality scoring in the prior architecture pass:

### 5.1 Structured Signature — auditable, drives prompting (§7) directly

```python
class ThumbnailStyleSignature(BaseModel):
    model_config = ConfigDict(frozen=True)
    video_id: str
    channel_id: str
    dominant_colors: list[str]
    brightness: float
    contrast: float
    saturation: float
    warm_or_cool: Literal["warm", "cool", "neutral"]
    color_harmony_score: float
    subject_placement: str          # reused verbatim from CompositionAnalysis
    negative_space_ratio: float
    balance_score: float
    symmetry_score: float
    face_scale_ratio: Optional[float] = None   # None when has_face=False
    text_coverage_ratio: float
    text_region_count: int
    object_classes_present: list[str]
    extracted_at: str
```

### 5.2 Embedding Vector — drives similarity/drift (§6) efficiently

```python
class CreatorStyleEmbedding(BaseModel):
    model_config = ConfigDict(frozen=True)
    channel_id: str
    embedding: list[float]      # OpenCLIPWrapper.encode_image() output, reused as-is
    embedding_model: str        # from RegisteredVisionModel, existing convention
    source_video_ids: list[str] # which thumbnails contributed to this centroid
    sample_count: int
    computed_at: str
```

**Aggregation across a creator's accumulated thumbnails** (§4's constraint: this grows one video at a time): the embedding is the **running centroid** (mean vector) of `OpenCLIPWrapper.encode_image()` outputs across every thumbnail seen so far for that `channel_id` — an incremental mean update (`new_centroid = old_centroid + (new_vec - old_centroid) / (n+1)`), so no historical embeddings need to be re-loaded or re-computed on every new video, keeping this cheap regardless of how large a creator's history grows.

### 5.3 Persistent Store

```
data/creator_style_profiles/{channel_id}/
  signatures/{video_id}.json          # ThumbnailStyleSignature, one per contributing video
  style_embedding.json                # CreatorStyleEmbedding, the running centroid
  profile_manifest.json               # sample_count, first_seen_at, last_updated_at, schema_version
```

Sharded by `channel_id` — the one new sharding key this document introduces, deliberately distinct from every existing `video_id`-sharded directory. Atomic temp-file-then-replace writes, matching every other module's persistence convention. A profile with `sample_count < OPTIMIZATION_STYLE_MIN_SAMPLES` (new config constant, e.g. `3`) is considered **not yet established** — every consumer below (§6–8) must check this and fall back to no-style-influence behavior below the threshold, exactly matching the cold-start handling already specified for the feedback system in the prior architecture pass.

---

## 6. Style Similarity (Phase 3)

```python
class StyleSimilarityResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    video_id: str
    channel_id: str
    similarity_score: float          # OpenCLIPWrapper.compute_similarity(new_embedding, profile.embedding)
    belongs_to_identity: bool        # similarity_score >= OPTIMIZATION_STYLE_SIMILARITY_THRESHOLD
    profile_established: bool        # sample_count >= OPTIMIZATION_STYLE_MIN_SAMPLES
```

Computed by calling `OpenCLIPWrapper.compute_similarity()` directly — the exact same method PVQEF's scorers already call — between a candidate thumbnail's embedding and the creator's stored centroid (§5.2). No new similarity math is introduced; this is pure reuse.

When `profile_established=False`, `belongs_to_identity` is reported as `True` with a `not yet established` annotation (never blocks generation on a creator the system hasn't seen enough of yet) — this is the same cold-start posture as the feedback system's prior provider.

---

## 7. Prompt Integration (Phase 4)

Style-aware prompting is an **additive instruction block**, appended to the existing `PromptPackage` fields Module 6 already produces — never a rewrite of `prompt_compiler.py`'s logic:

```python
class StylePromptGuidance(BaseModel):
    model_config = ConfigDict(frozen=True)
    channel_id: str
    color_guidance: str      # derived from ThumbnailStyleSignature's dominant_colors/warm_or_cool,
                              # templated the same way prompt_compiler.py's existing per-field
                              # instruction compilers work (deterministic, no LLM)
    composition_guidance: str # derived from subject_placement/balance_score
    face_scale_guidance: Optional[str] = None
```

`design_blueprint_components/strategy_engine.py` (Module 5.5, existing) already resolves competing constraints (e.g. CTR-driven layout suggestions vs. preservation directives) via its existing confidence/priority mechanism — `StylePromptGuidance` is submitted to that **same existing resolution path** as one more weighted input, phrased identically to how the prior architecture pass wired `prior_provider.py` into `ConflictResolver` as an additive, optional confidence signal. If `strategy_engine.py`'s existing logic already prioritizes CTR improvement over strict preservation (per the brief's own Phase 4 wording — "preserve creator identity while still improving CTR" is a balance, not an override), that balance is expressed by weighting `StylePromptGuidance` at a bounded, configurable strength (`OPTIMIZATION_STYLE_PROMPT_WEIGHT`, new config constant) rather than by hard-coding a rule inside Module 5.5 itself — keeping the actual constraint-resolution logic untouched, per "do not redesign Modules 1–9."

---

## 8. Ranking Integration (Phase 5)

Extends `winner_selector.py` from the prior Optimization Layer architecture (§10 of that document) with one additive scoring input — this section assumes that layer exists, since style-aware ranking is naturally a candidate-ranking concern and that's precisely what `winner_selector.py` already does for `BeatsOriginalVerdict`:

```python
class StyleAwareScore(BaseModel):
    model_config = ConfigDict(frozen=True)
    candidate_index: int
    style_similarity: float          # from §6, computed on the CANDIDATE, not the original
    style_bonus: float               # bounded reward term, e.g. max(0, similarity - threshold) * weight
```

`winner_selector.select()` gains one additive input list (`style_scores: list[StyleAwareScore] | None = None`) — when `None` (a creator with no established profile, or the Style Layer disabled entirely), behavior is byte-identical to the existing selector, preserving backward compatibility exactly as the brief requires. When present, `style_bonus` is added to the existing comparative-quality ranking score with a bounded, configurable weight — never allowed to override a `hard_gate_passed=False` rejection or an `over_edited=True` acceptance-gate rejection from Module 7 / the Optimization Layer's own validation (§15 of that document) — style preference never bypasses a correctness gate.

---

## 9. Drift Detection (Phase 6)

"Detect when a creator intentionally changes their style" is fundamentally a **statistical change-point problem over the similarity signal (§6), computed on the creator's own real thumbnails over time** — not a new visual model.

```python
class StyleDriftAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)
    channel_id: str
    recent_similarity_scores: list[float]   # last N videos' similarity to the *pre-drift* centroid
    drift_detected: bool
    drift_confidence: float
    recommended_action: Literal["none", "monitor", "update_centroid"]
```

**Detection rule (simple, deterministic, auditable — no ML classifier introduced):** if the last `OPTIMIZATION_STYLE_DRIFT_WINDOW` (new config constant, e.g. `3`) consecutive videos for a `channel_id` all score below `OPTIMIZATION_STYLE_SIMILARITY_THRESHOLD` against the *current* stored centroid, **and** those same videos are mutually similar to *each other* (their own pairwise `compute_similarity` is high) — i.e. the creator has consistently moved to a new, internally-consistent look, not just produced one noisy outlier thumbnail — `drift_detected=True`.

**"Update stored style safely"** means: drift detection never silently overwrites `style_embedding.json`. It flags `recommended_action="update_centroid"`, and the actual centroid reset is a **separate, explicit, logged operation** (`StyleProfileStore.reset_centroid(channel_id, from_video_ids=[...])`) — requiring the drift window's videos as its explicit new seed set. This mirrors the prior architecture pass's insistence that acceptance/rejection decisions always leave deterministic evidence rather than silently changing system behavior. A single outlier thumbnail (one unusual video, not a sustained shift) never triggers a reset — the mutual-similarity check in the rule above specifically guards against that.

---

## 10. GenerationTrace Integration (Phase 7)

Additive fields only, on `observability/models.py::GenerationTraceRecord` — same convention as the prior two architecture passes, no existing field touched:

```python
# GenerationTraceRecord additions
creator_channel_id: Optional[str] = None
style_signature_reference: Optional[str] = None    # path/hash of the ThumbnailStyleSignature used
style_embedding_similarity: Optional[float] = None
style_profile_established: Optional[bool] = None
style_bonus_applied: Optional[float] = None
drift_detected: Optional[bool] = None
drift_confidence: Optional[float] = None
style_prompt_guidance_applied: Optional[bool] = None
```

Populated at the same call site the prior architecture pass's `trace_extension.py` already writes to — one more additive builder function, not a new write path.

---

## 11. PORCE Integration (Phase 8)

New diagnostic rules, added exactly as `edit_mode_resolution_rules.py` and the Optimization Layer's `RULE-OPT-*` rules were added — new files under `observability/diagnostics/rules/`, registered with the existing `RuleRegistry`, executed by the existing `RuleExecutionEngine`, zero changes to the engine itself:

```python
class StyleViolationRule(IDiagnosticRule):
    """RULE-STYLE-01: Flags when style_embedding_similarity is present but below
    OPTIMIZATION_STYLE_SIMILARITY_THRESHOLD for a channel with an established
    profile (style_profile_established=True) — i.e., the shipped candidate
    diverged from the creator's identity without a corresponding drift_detected."""

class BrandingInconsistencyRule(IDiagnosticRule):
    """RULE-STYLE-02: Cross-checks branding_constraints (Module 5.5, existing)
    against object-preservation evidence already captured by the Optimization
    Layer's validation gate (§15 of the prior architecture) — flags when a
    branding constraint existed but the corresponding object/region was not
    preserved."""

class IdentityLossWithoutDriftRule(IDiagnosticRule):
    """RULE-STYLE-03: Flags low style_embedding_similarity when
    drift_detected=False — distinguishes 'the creator changed styles'
    (expected, §9) from 'the renderer lost the creator's identity'
    (a defect), which is exactly the distinction the brief's Phase 8 asks
    PORCE to make."""
```

---

## 12. Validation

A style-specific extension of the Optimization Layer's existing `acceptance_gate.py` (§15 of the prior architecture pass) — one additive check, not a new gate framework:

```python
# AcceptanceResult.reasons_rejected gains one more possible value:
"style_identity_lost"   # style_embedding_similarity < threshold AND drift_detected == False
                          # AND style_profile_established == True
```

Consistent with the existing gate's posture: this is advisory/blocking exactly like every other rejection reason already in that table, never a separate mechanism, and — per §8 — style considerations never override a hard correctness gate; they only ever add an additional, optional rejection reason on top of the existing ones.

---

## 13. Testing Strategy (design only)

- `StyleExtractor` unit tests against real `ThumbnailIntelligence`/`VisualReferenceManifest` fixtures already present in `data/` (`tests/test_visual_reference_engine.py`'s fixture style), asserting derived fields (`face_scale_ratio`, background-region color) compute correctly.
- Incremental-centroid update tests: verify the running-mean update produces the same result as a batch mean over N embeddings, for N up to a reasonably large synthetic sample — a pure-math property test, no GPU required (mock `OpenCLIPWrapper.encode_image()` outputs).
- `StyleSimilarityResult` tests using two clearly-different synthetic embeddings (should reject) and two near-identical ones (should accept), plus the `profile_established=False` cold-start path.
- Drift-detection tests: three scenarios — (a) one noisy outlier thumbnail (should NOT trigger drift), (b) three consecutive, mutually-similar off-centroid thumbnails (should trigger), (c) steady-state matching thumbnails (should never trigger) — golden-file style, matching the deterministic-regression pattern used throughout this project.
- PORCE rule tests (`RULE-STYLE-01..03`) following the exact pattern already established for `RULE-EDIT-*`/`RULE-OPT-*`.
- Backward-compatibility test: `winner_selector.select()` called with `style_scores=None` must produce byte-identical output to a pre-Style-Layer run, given identical other inputs — this is the explicit regression guard for the brief's "maintain backward compatibility" rule.
- No `gpu`/`integration` markers needed for the storage/aggregation/drift logic itself; `OpenCLIPWrapper.encode_image()` calls (already GPU-capable, already has a deterministic hash-based fallback per the wrapper's existing `_hash_image_to_embedding` method) are the only GPU-adjacent dependency, and it's entirely inherited/reused, not new.

---

## 14. Migration Plan

1. Ship `StyleExtractor` + `data/creator_style_profiles/` storage first, running read-only alongside the existing pipeline (`main.py` gains one additive, non-blocking call per creator, writing signatures/embeddings) — zero behavior change to what gets generated or shipped.
2. Let profiles accumulate for a period before enabling any consumer (§6–8) — `OPTIMIZATION_STYLE_MIN_SAMPLES` (default `3`) already gates this structurally, but an explicit soak period (e.g. process at least one full `creators.csv` run) is still recommended before trusting early centroids.
3. Enable §7 (style-aware prompting) behind `OPTIMIZATION_STYLE_PROMPT_ENABLED` (default `False`), at a conservative `OPTIMIZATION_STYLE_PROMPT_WEIGHT`, and compare against a control group of runs with it disabled using the Optimization Layer's existing `BeatsOriginalVerdict` — style guidance should not measurably reduce the "beats original" win rate; if it does, the weight is too high.
4. Enable §8 (style-aware ranking) only after step 3 is validated — it directly affects which candidate ships, so it carries more risk than prompting alone.
5. Enable §9 (drift detection) last, since it's the only component that can change stored state (`reset_centroid`) — validate its three test scenarios (§13) against real accumulated data before allowing `recommended_action="update_centroid"` to ever be acted on automatically; consider requiring manual confirmation for the first several drift events in production.
6. §10–12 (trace/PORCE/validation) can ship alongside step 1 with no risk — they're purely additive recording, gated on the same `Optional` fields being `None` until the corresponding consumer is enabled.

---

## 15. Risks

| Risk | Mitigation |
|---|---|
| Cold-start: most creators in `data/creators.csv` will have `sample_count < OPTIMIZATION_STYLE_MIN_SAMPLES` for a long time, since there's no channel-history backfill capability (§2.5) | Every consumer explicitly treats `profile_established=False` as "no influence," never as "reject" — designed in from the start, not a follow-on patch |
| Running centroid could be permanently skewed by an early bad thumbnail (e.g. a mis-processed video) before enough samples exist to dilute it | `reset_centroid` (§9) provides a recovery path; consider also exposing a manual "discard profile and restart" operation for a given `channel_id` as an operational safeguard (implementation detail, not designed in depth here since it's a small addition to `StyleProfileStore`) |
| Style-aware ranking (§8) could systematically favor "safe"/similar candidates over ones that would have won on pure CTR-improvement grounds, undermining the Optimization Layer's actual objective | Bounded, configurable weight (§7, §8) plus the explicit backward-compatibility default of `style_scores=None`; migration plan (§14 step 3) requires measuring against `BeatsOriginalVerdict` win rate before enabling ranking influence |
| Drift detection could misfire on genuinely sparse creators (few videos, naturally more variance) mistaking noise for an intentional style change | The mutual-similarity check (§9) specifically guards against single-outlier false positives; `OPTIMIZATION_STYLE_DRIFT_WINDOW` should scale conservatively for low-`sample_count` channels — a documented follow-on tuning task, not solved definitively here since it depends on production data this system doesn't have yet |
| `channel_id`-sharded storage is a genuinely new data-partitioning axis in a codebase that has used `video_id` exclusively everywhere else | Fully isolated under `data/creator_style_profiles/`; no existing directory structure, model, or module is touched — this is additive infrastructure, not a migration of existing data |

---

## 16. Future Work

- A real channel-history backfill (fetching a creator's other public video thumbnails via the YouTube Data API, not yt-dlp's single-video path) would remove the cold-start constraint from §2.5/§15 entirely — explicitly out of scope here since no such capability exists in the repo today and this document does not design around infrastructure that isn't there.
- A dedicated logo/watermark detector (§4) would close the one clearly-flagged extraction gap; likely a fine-tuned or template-matching addition to the existing YOLO-based object detection stage in Module 4, scoped as its own small architecture pass.
- Typography style (font family/weight, not just placement) would require a font-classification model with no current counterpart in `modules/vision_stack/` — flagged, not designed.
- Once enough `channel_id` history accumulates across many creators, the Optimization Layer's feedback store (§12.2 of the prior architecture pass) could be extended to answer "which style-preservation strength produces the best `BeatsOriginalVerdict` outcomes, per niche" — connecting this document's Style Layer to that document's feedback loop, a natural second-order integration deliberately left for after both systems have independently accumulated enough production data to make it meaningful.
