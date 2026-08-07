# Thumbnail Intelligence Architecture

**thumbnail-ai**
**Status:** Architecture only. Zero implementation code.
**Author role:** Principal AI Systems Architect.
**Relationship to other documents:** Subordinate to `docs/thumbnail-renderer-v2-architecture.md` (Renderer V2 — production, not redesigned here) and additive to `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` (already implemented — see §0.2). Verified against the live repository: `modules/thumbnail_understanding/` (`schemas.py`, `understanding_engine.py`, `director_engine.py`, `psychology_assessor.py`, `weakness_analyzer.py`, `scene_grounding.py`, `hierarchy_calculator.py`, `scene_decomposer.py`), `modules/creator_style/` (`profile_store.py`, `style_extractor.py`, `style_similarity.py`, `drift_detector.py`, `style_prompt_guidance.py`, `style_aware_ranking.py`), `modules/decision_components/`, `modules/design_blueprint_components/`, `modules/planner_components/`, `modules/vision_stack/openclip.py`, `optimization/feedback/` (`outcome_recorder.py`, `outcome_store.py`, `prior_provider.py`), `evaluation/benchmarking/` (`historical_store.py`, `golden_sample_manager.py`, `regression_detector.py`), `observability/`, and `main.py`'s actual pipeline ordering.

---

## 0. Grounding Note — what already exists, and what this document adds

This section exists because the brief asks for an architecture that "builds on the existing implementation, not replace it," and that instruction is only honorable if the existing implementation is stated correctly first.

### 0.1 What the brief calls "the Intelligence Engine" is partially real today

The repository already contains a working per-thumbnail reasoning stack, built exactly along the lines `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` proposed and evidently since implemented (its Phase 1–16 numbering matches `schemas.py`'s docstring: "Phase 2, 4, 5 ... Phase 15, 16"). Concretely, `modules/thumbnail_understanding/understanding_engine.py` already produces a `ThumbnailUnderstanding` object containing:

- A grounded `SceneGraph` (`SceneElement` + `ElementRelationship`, joined to real detections — not invented).
- A deterministic `VisualHierarchy` (reading order, focal strength, clutter, subject separation).
- `CompositionIntelligence` and `WeaknessAnalysis`.
- A `ThumbnailPsychologyAssessment` with per-driver `PsychologyDriver` entries (already structured, already grounded in `associated_element_ids`, not free text).
- A `DecomposedScene` (editable layers, depth-ordered).
- An `AIThumbnailDirectorPlan` and a `ProfessionalImprovementPlan` — produced by `director_engine.py`'s `AIThumbnailDirector.generate_plan()`, which already emits `ImprovementAction` entries carrying `expected_ctr_gain`, `identity_risk`, `visual_risk`, `dependencies`, and `fallback_action`.

This is a real, working **Perceptual Reasoning Layer**: it looks at one thumbnail, in isolation, and correctly answers "what is in this frame, what does it mean, what is wrong with it, what should change." It is grounded well — every field above traces to a detection, a formula, or a narrowly-scoped model call, per the discipline `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §4.3 established. **This document does not redesign it, extend its schemas destructively, or duplicate its responsibility.**

### 0.2 What is genuinely missing — and what this document is

Every field in `ThumbnailUnderstanding` is computed from exactly one input: the single thumbnail (plus whatever `ThumbnailIntelligence` and `VideoMetadata` accompany it) being analyzed *right now*. The Perceptual Reasoning Layer has no memory of any other thumbnail, this creator's own history, this niche's competitors, or which design decisions actually improved outcomes in the past. Concretely, verified absent from the repository:

- No cross-video knowledge base of thumbnail designs, archetypes, or patterns (`data/design_blueprints/*.json` and `data/redesign_specs/*.json` exist, one file per video, but nothing indexes or queries across them).
- No competitor-channel ingestion or comparison anywhere in `modules/` or `data/`.
- No formal archetype taxonomy — `AIThumbnailDirectorPlan.creative_direction` is free text with no reference to a named, reusable design template.
- No retrieval or embedding index over historical designs — `modules/vision_stack/openclip.py`'s `OpenCLIPWrapper` (512-dim) is real and loaded, but its only current consumer is `creator_style/style_similarity.py`, computing one single-vector centroid distance per channel, not a retrieval corpus.
- `optimization/feedback/outcome_store.py` and `prior_provider.py` are real and already close a learning loop — but only at the granularity of a single decision rule (`rule_confidence_prior(rule_id)`), not at the granularity of a whole creative direction or archetype choice.

**This document's job is to design the missing layer**: a **Strategic Reasoning Layer** that sits immediately downstream of the existing `ThumbnailUnderstanding` and answers the cross-video, cross-creator, cross-niche questions the brief lists — *why* this direction, grounded in what has actually worked, for this creator, in this niche, against these competitors — and packages the answer into one new, additive artifact: **`DesignBrief`**. Everything downstream (Module 5 `RedesignSpecification`, Module 5.5 `DesignBlueprint`, Module 9 `DecisionManifest`) already has a proven additive-input migration precedent to follow — `main.py`'s Module 5 call already accepts an optional `understanding=understanding` keyword alongside the required `intelligence` argument. `DesignBrief` is designed to enter the pipeline the same way.

---

## 1. System Philosophy

Three governing principles, none of them new inventions — each is an existing convention in this repository, made explicit and applied consistently to the new layer.

**1. Interpretation, not invention (`thumbnail-renderer-v2-architecture-v2.md` §0).** Renderer V2 already states this for pixels: never generate detail that isn't grounded in something real. This document is the creative-reasoning analog: the Intelligence Engine must never assert a design recommendation, a competitor comparison, or a CTR claim that cannot be traced to an `EvidenceReference` (§19.2). A recommendation with no evidence is not a lower-confidence recommendation — it is a defect, rejected at the schema boundary the same way `ThumbnailUnderstanding`'s Pydantic validation rejects a relationship referencing a nonexistent `element_id` (`docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §12).

**2. Deterministic where possible, narrow-model-call where necessary.** `hierarchy_calculator.py` computes `VisualHierarchy` with a documented formula, not a model call, precisely because reading-order approximation from geometric facts doesn't need one. This document follows the same test at every subsystem boundary: retrieval, ranking-by-similarity, frequency counting, and rule-based archetype matching are deterministic; only genuinely open-ended judgment (does this competitor's thumbnail read as *aggressive* or *calm*; does this headline's phrasing match this creator's voice) goes to a constrained model call, and even then the call is shown pre-retrieved, pre-structured evidence rather than being asked to recall or invent facts from its own parametric memory.

**3. Additive, never destructive.** Every existing schema (`ThumbnailUnderstanding`, `RedesignSpecification`, `DesignBlueprint`, `ResolvedDecision`, `CreatorStyleEmbedding`) keeps every field it has. New capability is new, optional, additively-consumed data — new files, new optional constructor arguments, new optional schema fields with safe defaults — following the exact precedent `main.py` already establishes for `understanding=understanding`. Nothing in `renderer_v2/` is touched; nothing in `modules/thumbnail_understanding/` is rewritten.

**Rejected alternative — a single larger LLM call that "knows" the creator's history and competitors from a long prompt.** This is the shape of system `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §2.1 already diagnosed as this project's core historical failure mode (`GeminiReasoning`'s one-call, ungrounded, un-decomposed reasoning). Stuffing competitor thumbnails, archetype descriptions, and historical outcomes into one long prompt reintroduces exactly that failure at a larger scale — the model would be asked to both *recall* facts (which thumbnails exist, what their outcomes were) and *reason* over them in the same undifferentiated pass, with no verifiable grounding and no bound on hallucinated detail. This document instead separates retrieval (deterministic, verifiable, §16) from reasoning (narrow, evidence-scoped, §19), the same separation `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §4.3 already applied to relationship/psychology reasoning.

---

## 2. Responsibilities

| In scope | Out of scope (explicitly, and why) |
|---|---|
| Cross-video, cross-creator, cross-niche knowledge retrieval and storage | Per-thumbnail scene/psychology/weakness analysis — real, owned by `thumbnail_understanding/` |
| Archetype classification and matching | Pixel generation, compositing, relighting — owned by Renderer V2 |
| Competitor-channel visual/structural comparison | Downloading/scraping competitor content — a data-acquisition concern belonging to a Module-1/2-style ingestion component (§9.4), reusing this project's existing extraction stack, not reasoning |
| Creator brand-rule extraction and long-horizon drift tracking | Per-candidate style similarity scoring at generation time — real, owned by `creator_style/style_aware_ranking.py` |
| CTR/emotion/audience-psychology reasoning *grounded in retrieved evidence* | CTR *prediction* as a trained regression/classification model — no such model exists in this repository and none is proposed here (§20, §23) |
| `DesignBrief` synthesis — the single new artifact this layer produces | `RedesignSpecification`/`DesignBlueprint`/`DecisionManifest` construction — those remain Module 5/5.5/9's responsibility; `DesignBrief` is an additional *input* to them |
| Benchmarking and evaluation of this layer's own output quality | Module 7 generation-candidate quality scoring — real, owned by `evaluation/quality/` |
| Feedback-loop design for this layer's knowledge (archetype/competitor weighting) | Feedback-loop mechanics at the decision-rule level — real, owned by `optimization/feedback/` |

---

## 3. Subsystem Overview

```
                         ┌───────────────────────────────────────────┐
                         │           KNOWLEDGE BASE (§5)              │
                         │  ┌───────────────┐ ┌─────────────────────┐│
                         │  │ Archetype      │ │ Historical Thumbnail││
                         │  │ Library (§7)   │ │ Database (§8)       ││
                         │  └───────────────┘ └─────────────────────┘│
                         │  ┌───────────────┐ ┌─────────────────────┐│
                         │  │ Competitor     │ │ Design Pattern      ││
                         │  │ Intelligence(9)│ │ Library (§15)       ││
                         │  └───────────────┘ └─────────────────────┘│
                         └───────────────┬───────────────────────────┘
                                         │  KnowledgeEntry (embedding + structured facets)
                                         ▼
┌────────────────────┐        ┌───────────────────────┐        ┌──────────────────────┐
│ Creator Profile     │───────▶│ Retrieval Architecture │◀───────│ Embedding Architecture│
│ System (§6)          │       │ (§16)                  │        │ (§17)                 │
│ (extends creator_style)│      └──────────┬────────────┘        └──────────────────────┘
└────────────────────┘                    │  retrieved, filtered evidence set
                                            ▼
                         ┌───────────────────────────────────────────┐
                         │        STRATEGIC REASONING LAYER            │
                         │  ┌───────────────────────┐                 │
ThumbnailUnderstanding   │  │ Visual Storytelling     │                 │
(existing, §0.1) ───────▶│  │ Engine (§11)            │                 │
                         │  └───────────────────────┘                 │
                         │  ┌───────────────────────┐                 │
                         │  │ CTR Reasoning Engine    │                 │
                         │  │ (§12)                   │                 │
                         │  └───────────────────────┘                 │
                         │  ┌───────────────────────┐                 │
                         │  │ Emotion Reasoning (§13) │                 │
                         │  └───────────────────────┘                 │
                         │  ┌───────────────────────┐                 │
                         │  │ Audience Psychology     │                 │
                         │  │ (§14)                   │                 │
                         │  └───────────────────────┘                 │
                         └───────────────┬───────────────────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────────────────┐
                         │      DesignBrief Generator (§19)            │
                         │  every field → EvidenceReference             │
                         │  full trace  → ReasoningTrace                │
                         └───────────────┬───────────────────────────┘
                                         │  DesignBrief (new, additive input)
                                         ▼
                    Module 5 (RedesignSpecification) / Module 5.5 (DesignBlueprint) / Module 9 (DecisionManifest)
                                    — unchanged responsibility, richer input —
                                         │
                                         ▼
                              Renderer V2 (unchanged, out of scope)
                                         │
                                         ▼
                    Memory Architecture (§18) / Benchmarking (§21) / Evaluation (§22) / Future Learning (§23)
                              — closes the loop back into the Knowledge Base —
```

Each box is specified fully in its own numbered section below, with Purpose / Inputs / Outputs / Owner / Lifecycle / Dependencies / Failure Modes / Computational Cost / Future Extensibility, per the brief's requirement.

---

## 4. Pipeline

Exact insertion point, verified against `main.py`'s real stage order (`Module 4 → Module 10 [creator style] → Module 8 [asset extraction] → thumbnail_understanding → Module 5 → Module 5.5 → Module 6 → Module 9 → Module 10 [composition] → Module 10.5 → Module 7`):

```
Module 4 (Thumbnail Intelligence, unchanged)
   │
   ▼
Creator Style extraction (modules/creator_style, unchanged — StyleExtractor.extract_signature)
   │
   ▼
Module 8 (Asset Extraction, unchanged)
   │
   ▼
ThumbnailUnderstandingEngine.understand(...) → ThumbnailUnderstanding   [EXISTING, §0.1]
   │
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NEW: Intelligence Engine (this document)          │
│                                                                        │
│  1. Retrieval Architecture (§16) queries Knowledge Base (§5) using    │
│     ThumbnailUnderstanding + ChannelProfile as the query key           │
│         → RetrievedEvidenceSet (bounded, typed, scored)               │
│                                                                        │
│  2. Strategic Reasoning Layer (§11–§14) consumes                       │
│     ThumbnailUnderstanding + RetrievedEvidenceSet                      │
│         → StoryFrame, CTRHypothesis[], EmotionProfile, AudiencePattern[]│
│                                                                        │
│  3. DesignBrief Generator (§19) assembles all of the above into one    │
│     validated, fully-evidenced DesignBrief                             │
└──────────────────────────────┬────────────────────────────────────────┘
                                │  DesignBrief (new, optional keyword input)
                                ▼
Module 5  build_redesign_specification(intelligence, understanding=..., design_brief=...)   [additive param]
   │
   ▼
Module 5.5  build_design_blueprint(intelligence, redesign_spec, metadata, design_brief=...)  [additive param]
   │
   ▼
Module 6 (Prompt Compiler, unchanged — richer template inputs only, same pattern §8.3 of the prior doc used)
   │
   ▼
Module 9  run_decision_engine(..., design_brief=...)   [additive param; ResolvedDecision.rationale
                                                          may now cite DesignBrief.design_reasons]
   │
   ▼
Module 10 / 10.5 / Module 7 (unchanged, out of scope)
   │
   ▼
Optimization / Evaluation (existing outcome + benchmark stores) → feeds back into §18/§23
```

No existing call signature loses a required argument; every new parameter is optional with a safe default (`design_brief: Optional[DesignBrief] = None`), so the pipeline runs identically, with a strictly narrower `DesignBrief`-shaped fallback, on a cold-start creator with an empty Knowledge Base (§20).

---

## 5. Knowledge Base

**Purpose.** A single logical container — physically four sharded stores — holding everything the Strategic Reasoning Layer can cite as evidence beyond the current video: Archetype Library, Historical Thumbnail Database, Competitor Intelligence, Design Pattern Library.

**Inputs.** `DesignBlueprint`/`RedesignSpecification` records already produced by every pipeline run (`data/design_blueprints/`, `data/redesign_specs/` — real, currently un-indexed); `optimization/feedback` outcome records (real); competitor-channel ingestion runs (§9.4, new); curated seed data for archetypes and audience-psychology patterns (§7, §14 — hand-authored, versioned, not learned, at least initially).

**Outputs.** `KnowledgeEntry` records (§5.1), each carrying one embedding vector and a structured facet set, queryable by the Retrieval Architecture (§16).

**Owner.** A new top-level package, `intelligence_kb/` (§9.3 folder layout), sibling to `modules/`, `optimization/`, `evaluation/` — not nested inside `modules/thumbnail_understanding/`, because its lifecycle (append-only accumulation across the whole project's history) is fundamentally different from a per-video analysis package's lifecycle.

**Lifecycle.** Append-mostly. A `KnowledgeEntry` is written once when its source event occurs (a pipeline run completes, a competitor snapshot is ingested) and is never mutated in place — corrections are new entries superseding old ones by `superseded_by` reference, matching this repository's existing atomic-write-then-replace convention (`profile_store.py`'s `_atomic_write_json`) applied to *files*, extended here to *logical* entries because embeddings, once computed, should not be silently rewritten under a retrieval system that may have cached them.

**Dependencies.** `OpenCLIPWrapper` (embedding, §17), `evaluation/benchmarking/historical_store.py` (outcome linkage, §21), `creator_style/profile_store.py` (channel identity, §6).

**Failure modes.** See consolidated table, §20 (rows: cold-start sparsity, stale competitor snapshots, embedding-index/store desync).

**Computational cost.** Write path: one embedding call per new entry (amortized into existing per-video pipeline cost — no new per-video latency beyond §17's cost). Read path: bounded by Retrieval Architecture's top-k design (§16) — never a full-corpus scan at query time.

**Future extensibility.** New facet types (e.g., a "seasonal trend" facet) are additive fields on `KnowledgeEntry.facets`, not new tables — see §24.

### 5.1 `KnowledgeEntry` — the unifying data contract

Every one of the four Knowledge Base subsystems below produces `KnowledgeEntry` records, differentiated by `entry_type`, so the Retrieval Architecture (§16) can query across all four with one interface instead of four bespoke ones — the same "one structural contract, many producers" pattern `SceneElement` already uses across OCR/face/object detectors.

```python
class KnowledgeEntryType(str, Enum):
    ARCHETYPE_EXAMPLE = "archetype_example"
    HISTORICAL_THUMBNAIL = "historical_thumbnail"
    COMPETITOR_THUMBNAIL = "competitor_thumbnail"
    DESIGN_PATTERN = "design_pattern"

class KnowledgeEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry_id: str                              # deterministic hash, this repo's existing ID convention
    entry_type: KnowledgeEntryType
    embedding: list[float]                     # OpenCLIP 512-dim, §17
    embedding_model: str = "OpenCLIP-ViT-B-32"  # matches CreatorStyleEmbedding.embedding_model convention
    source_video_id: Optional[str] = None       # for HISTORICAL_THUMBNAIL entries
    source_channel_id: Optional[str] = None
    source_competitor_id: Optional[str] = None  # for COMPETITOR_THUMBNAIL entries
    archetype_id: Optional[str] = None          # for ARCHETYPE_EXAMPLE / classified entries
    pattern_id: Optional[str] = None            # for DESIGN_PATTERN entries
    niche: str = "general"
    facets: dict[str, Any] = Field(default_factory=dict)   # structured, typed filters (§16.2)
    outcome_ref: Optional[str] = None           # joins to optimization OutcomeStore record, when known
    superseded_by: Optional[str] = None
    created_at: str
```

---

## 6. Creator Profile System

**Purpose.** Answer "what fits *this* creator" — extending, not duplicating, `modules/creator_style/`'s already-real per-channel style signature and centroid embedding.

**Inputs.** `CreatorStyleEmbedding`, `StyleProfileManifest`, `ThumbnailStyleSignature` (all real, `modules/models.py:2050-2110`); brand rules from Brand Learning (§10).

**Outputs.** `ChannelProfile` (§6.2, wraps and extends `StyleProfileManifest` without modifying it) and `CreatorProfile` (§6.1, a new higher-level identity that can span multiple channels for the same creator — e.g., a main channel and a Shorts/second channel, which this repository's current one-`channel_id`-per-profile model cannot represent).

**Owner.** `intelligence_kb/creator_profiles/`, reading `data/creator_style_profiles/{channel_id}/` (unchanged directory) and writing a new sibling `data/creator_profiles/{creator_id}.json`.

**Lifecycle.** Updated once per pipeline run per channel, same cadence as `StyleProfileStore.get_manifest()`'s existing update — this is a read-mostly aggregation over existing per-channel state, not a new heavy write path.

**Dependencies.** `creator_style/profile_store.py` (read-only), `creator_style/drift_detector.py` (read-only, feeds `brand_stability_score`).

**Failure modes.** A creator with `sample_count` below `MODULE10_STYLE_MIN_SAMPLES` has `profile_established=False` (already a real, checked field) — `ChannelProfile` propagates that flag rather than fabricating a profile; the Strategic Reasoning Layer treats an unestablished profile as "no creator-specific evidence available" (§20), not as license to guess.

**Computational cost.** Negligible — aggregation over already-computed, already-persisted fields.

**Future extensibility.** Multi-channel `CreatorProfile` is designed in from the start (§6.1) specifically so a future "creator has 3 channels" case doesn't require a schema migration — see §24.

### 6.1 `CreatorProfile`

```python
class CreatorProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    creator_id: str                             # stable identity, independent of any one channel_id
    display_name: str
    channel_ids: list[str] = Field(default_factory=list)   # 1..n ChannelProfile references
    primary_niche: str = "general"
    brand_rules: list["DesignReason"] = Field(default_factory=list)   # from Brand Learning, §10
    cross_channel_consistency_score: Optional[float] = None  # None until 2+ channels established
    created_at: str
    updated_at: str
```

### 6.2 `ChannelProfile`

```python
class ChannelProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    channel_id: str
    creator_id: Optional[str] = None            # None until linked to a CreatorProfile
    niche: str = "general"
    style_embedding_ref: str                    # points to existing CreatorStyleEmbedding, not duplicated
    profile_established: bool                   # mirrors StyleProfileManifest.profile_established exactly
    sample_count: int                            # mirrors StyleProfileManifest.sample_count exactly
    archetype_affinity: dict[str, float] = Field(default_factory=dict)   # archetype_id -> usage frequency
    dominant_hook_types: list[str] = Field(default_factory=list)         # from DesignBlueprint.hook_type history
    brand_stability_score: Optional[float] = None   # from StyleDriftDetector, real component
    last_updated_at: str
```

**Rejected alternative — replacing `StyleProfileManifest`/`CreatorStyleEmbedding` with a new unified schema.** These are real, working, and consumed by `style_aware_ranking.py` at generation time — a component this document does not touch. `ChannelProfile` wraps them by reference (`style_embedding_ref`) instead of re-storing the vector, avoiding data duplication and the drift-between-two-copies failure mode that duplication would introduce.

---

## 7. Thumbnail Archetype Library

**Purpose.** A holistic, named vocabulary of recognizable thumbnail *templates* — the layer `AIThumbnailDirectorPlan.creative_direction` currently lacks entirely (it is free text with no controlled vocabulary, verified in `schemas.py`). Answers "what kind of thumbnail is this, or should this become."

**Inputs.** A curated seed set (hand-authored initially — see rejected alternative below), `SceneGraph`/`CompositionIntelligence` from every processed video (real, from `ThumbnailUnderstanding`), and archetype-example `KnowledgeEntry` records accumulated over time.

**Outputs.** `Archetype` definitions (§7.1) plus, per analyzed thumbnail, an `ArchetypeMatch` (§7.2) identifying the closest archetype(s) and confidence.

**Owner.** `intelligence_kb/archetypes/`.

**Lifecycle.** The Archetype *definitions* are versioned and change rarely (curated, reviewed additions — see §24). Archetype *matches* against individual videos are computed once per pipeline run and cached as `KnowledgeEntry(entry_type=ARCHETYPE_EXAMPLE)`.

**Dependencies.** Embedding Architecture (§17, for similarity-based candidate shortlisting) plus a small deterministic rule layer (for the final assignment — see below).

**Failure modes.** A thumbnail matching no archetype well (all similarity scores below threshold) is labeled `archetype_id=None, match_confidence=0.0` and reported as a `WeaknessFinding`-style gap ("no recognizable archetype — composition may read as generic"), not forced into the nearest one — a specific instance of the "no decision without evidence" mandate (§1).

**Computational cost.** One embedding lookup (already-computed image embedding, §17) plus a fixed small set of rule evaluations per archetype — O(archetype count), not O(corpus size), because matching is against ~15–30 curated archetype centroids, not the full Historical Thumbnail Database.

**Future extensibility.** New archetypes are added as new `Archetype` records with their own seed examples; no schema change required (§24).

### 7.1 `Archetype`

```python
class Archetype(BaseModel):
    model_config = ConfigDict(frozen=True)

    archetype_id: str                            # e.g. "big_face_reaction", "before_after_split"
    name: str
    description: str
    defining_scene_graph_pattern: dict[str, Any]  # structured, checkable predicate — e.g.
                                                    # {"hero_role": "hero", "hero_bbox_area_min": 0.35,
                                                    #  "text_element_count_max": 1}
    typical_hook_types: list[str] = Field(default_factory=list)   # references DesignBlueprint.hook_type values
    typical_emotion: Optional[str] = None
    niches_observed_in: list[str] = Field(default_factory=list)
    centroid_embedding: list[float]               # mean OpenCLIP embedding over curated + accumulated examples
    example_count: int = 0
    version: str = "1.0.0"
    created_at: str
```

### 7.2 `ArchetypeMatch`

```python
class ArchetypeMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    archetype_id: Optional[str] = None
    match_confidence: float = 0.0
    matched_via: Literal["embedding_similarity", "structural_predicate", "both", "none"] = "none"
    runner_up_archetype_ids: list[str] = Field(default_factory=list)
```

**Design decision — matching is embedding-shortlist + deterministic-predicate-confirm, not a single similarity threshold.** A pure nearest-centroid match (as `style_similarity.py` does for creator identity) is appropriate for "does this look like this creator's own past work" — a fuzzy, continuous question. Archetype assignment is a *categorical* claim ("this is a Big-Face-Reaction thumbnail") that downstream reasoning will cite as evidence, so it needs the same auditability `hierarchy_calculator.py`'s deterministic formula gives `VisualHierarchy`: embedding similarity produces a shortlist of 2–3 candidate archetypes (cheap, approximate), and `defining_scene_graph_pattern`'s structured predicate (checked against the real `SceneGraph`/`CompositionIntelligence` already computed upstream) confirms or rejects each candidate — an explainable, two-stage design, not a black-box nearest-neighbor call.

**Rejected alternative — learn archetypes unsupervised via clustering over the Historical Thumbnail Database.** Rejected for the same reason `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §8's OpenCLIP-vs-SigLIP discussion rejected an unproven migration: unsupervised clusters are not guaranteed to align with human-recognizable, nameable design categories a copywriter or creative director would actually use, and they are not auditable ("archetype #7" means nothing to a human reviewer). A small, curated, named seed set — extended by human review as new patterns are confirmed — is both cheaper to build correctly and directly usable in `DesignBrief.creative_direction` prose. Clustering remains a valid *future* input to proposing new candidate archetypes for human review (§24), never to auto-publishing them.

---

## 8. Historical Thumbnail Database

**Purpose.** Turn the already-real but currently un-indexed `data/design_blueprints/*.json` / `data/redesign_specs/*.json` (verified: one flat JSON file per `video_id`, no cross-file index, no query interface) into a genuinely queryable corpus — answering "what has this creator, or creators in this niche, actually done before, and did it work."

**Inputs.** Every completed pipeline run's `ThumbnailUnderstanding`, `RedesignSpecification`, `DesignBlueprint`, and — when available — a linked `optimization` outcome record.

**Outputs.** `KnowledgeEntry(entry_type=HISTORICAL_THUMBNAIL)` records, embedding-indexed and facet-tagged.

**Owner.** `intelligence_kb/historical/`, reading the existing `data/design_blueprints/`, `data/redesign_specs/`, `data/thumbnail_understanding/` directories (unchanged) and writing a new index alongside them, not into them.

**Lifecycle.** Append-on-pipeline-completion — one new `KnowledgeEntry` per video, permanently retained (this is a corpus, not a cache; unlike `evaluation/benchmarking/historical_store.py`'s JSONL, which this component reuses as its outcome-linkage source, not replaces).

**Dependencies.** Embedding Architecture (§17); `evaluation/benchmarking/historical_store.py` (`HistoricalStore.load_recent`, reused read-only for outcome linkage — see rejected alternative below).

**Failure modes.** A brand-new project or brand-new niche has an empty or near-empty Historical Thumbnail Database — the Retrieval Architecture (§16) must return an explicit empty `RetrievedEvidenceSet` rather than a low-confidence guess, and the Strategic Reasoning Layer must degrade gracefully (§20) rather than fabricate historical claims.

**Computational cost.** One embedding computation + one index write per pipeline run (same cost class as the existing per-video style-signature extraction it runs alongside). Read cost is bounded by the Retrieval Architecture's top-k design, never a full scan.

**Future extensibility.** Additional facets (e.g., a `seasonal_period` tag) are additive on `KnowledgeEntry.facets`.

**Design decision — reuse `evaluation/benchmarking/historical_store.py`'s `BenchmarkRecord` linkage instead of building a parallel outcome-tracking mechanism.** That component already appends `PipelineRunReport`-linked records in JSONL. The Historical Thumbnail Database's job is narrower and different: it indexes *designs* for retrieval by content similarity, not *pipeline runs* for regression detection. The two are linked (`KnowledgeEntry.outcome_ref` → a `BenchmarkRecord`/`OptimizationOutcome` id) rather than merged, because their lifecycles differ — `HistoricalStore` is append-only for regression comparison across code versions; the Historical Thumbnail Database is append-only for content retrieval and is agnostic to which pipeline version produced an entry.

**Rejected alternative — retroactively backfill this index from a full re-scan of every historical `data/design_blueprints/*.json` file at query time.** Rejected on cost grounds: real-time full-corpus scanning grows unboundedly with project history and defeats the point of an index. The correct migration path (§24) is a one-time offline backfill job that reads every existing file once, computes embeddings, and populates the index — after which the index is maintained incrementally, per-run, going forward.

---

## 9. Competitor Intelligence

**Purpose.** The one subsystem with zero existing precedent anywhere in this repository — answers "what differs from competitors," which the brief lists explicitly and which is genuinely absent from every module verified in §0.2.

**Inputs.** A configured list of competitor channel identifiers per niche (operator-provided, not auto-discovered — see rejected alternative below); competitor thumbnails, titles, and (where available) public engagement signals, ingested through the same extraction stack already used for the project's own creators.

**Outputs.** `CompetitorProfile` (§9.1) per competitor channel, plus `KnowledgeEntry(entry_type=COMPETITOR_THUMBNAIL)` records feeding retrieval, plus a `DifferentiationSummary` (§9.2) — the structured answer to "what does this creator do differently from the competitive set."

**Owner.** `intelligence_kb/competitor_intelligence/`.

**Lifecycle.** Competitor snapshots are refreshed on an operator-configured cadence (e.g., weekly), not per-pipeline-run — competitor channels change far less often than this project processes its own creators' videos, and re-ingesting a competitor's full thumbnail history on every run of every one of *our* creators would be wasted, unbounded work. Snapshot staleness is a tracked field (`CompetitorProfile.last_ingested_at`), not silently ignored.

**Dependencies.** §9.4's ingestion component (new, thin), which reuses — not reimplements — the existing `modules/vision_stack/` extraction pipeline (OCR, face, object, color, composition) and `creator_style/style_extractor.py`'s signature extraction logic, applied to competitor thumbnails exactly as it is applied to this project's own creators' thumbnails.

**Failure modes.** No configured competitors for a niche → `DifferentiationSummary` is `None`/absent, not fabricated (§20). A competitor channel becomes unavailable (deleted, private) → its `CompetitorProfile.status` is marked `unavailable`, existing embedded entries are retained (they remain valid historical evidence) but flagged as non-refreshable.

**Computational cost.** Bounded by ingestion cadence, not by this project's own per-video volume — the expensive part (running the vision stack over competitor thumbnails) happens on a schedule, not inline in the hot path of processing this project's own videos.

**Future extensibility.** New competitors are added by appending to the configured list — no schema change (§24).

### 9.1 `CompetitorProfile`

```python
class CompetitorProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    competitor_id: str
    channel_name: str
    niche: str
    style_embedding: list[float]                  # same OpenCLIP backbone, §17 — directly comparable
                                                    # to ChannelProfile's style_embedding_ref vector
    dominant_archetypes: list[str] = Field(default_factory=list)   # archetype_id references, §7
    dominant_hook_types: list[str] = Field(default_factory=list)
    color_palette_signature: list[str] = Field(default_factory=list)   # reuses ThumbnailStyleSignature
                                                                          # field convention exactly
    text_density_avg: float = 0.0
    sample_count: int = 0
    status: Literal["active", "unavailable", "stale"] = "active"
    last_ingested_at: str
    created_at: str
```

### 9.2 `DifferentiationSummary`

```python
class DifferentiationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    channel_id: str
    niche: str
    competitor_ids_considered: list[str]
    shared_conventions: list[str] = Field(default_factory=list)    # e.g. "large face, left-third placement —
                                                                       # used by 4/5 competitors and this creator"
    differentiating_factors: list["DesignReason"] = Field(default_factory=list)   # §19.1 — each grounded
    convergence_risk: Literal["low", "medium", "high"] = "low"     # high = this creator's thumbnails are
                                                                       # visually indistinguishable from the
                                                                       # competitive set, a real, citable risk
    computed_at: str
```

### 9.3 Folder structure

```
intelligence_kb/
  __init__.py
  creator_profiles/
    __init__.py
    profile_aggregator.py          # builds ChannelProfile/CreatorProfile from existing creator_style state
  archetypes/
    __init__.py
    archetype_definitions.py       # curated Archetype seed set, versioned
    archetype_matcher.py           # embedding shortlist + structural predicate confirm, §7.2
  historical/
    __init__.py
    historical_indexer.py          # writes KnowledgeEntry(HISTORICAL_THUMBNAIL) per completed run
    backfill_job.py                # one-time offline index population, §8's rejected-alternative note
  competitor_intelligence/
    __init__.py
    competitor_ingestion.py        # thin reuse-wrapper over vision_stack + creator_style extraction, §9.4
    differentiation_engine.py      # produces DifferentiationSummary, deterministic comparison logic
  design_patterns/
    __init__.py
    pattern_library.py             # curated + frequency-mined DesignPattern seed set, §15
  retrieval/
    __init__.py
    retrieval_engine.py            # §16
    query_builder.py
  embedding/
    __init__.py
    embedding_service.py           # thin wrapper over existing OpenCLIPWrapper, §17
    text_embedding_backend.py      # new capability, §17.2
  reasoning/
    __init__.py
    storytelling_engine.py         # §11
    ctr_reasoning_engine.py        # §12
    emotion_reasoning_engine.py    # §13
    audience_psychology_engine.py  # §14
  brief/
    __init__.py
    design_brief_generator.py      # §19
    evidence_validator.py          # rejects any DesignBrief field without a resolvable EvidenceReference
data/
  intelligence_kb/
    archetypes/{archetype_id}.json
    historical_index/{video_id}.json
    competitors/{competitor_id}.json
    design_patterns/{pattern_id}.json
    creator_profiles/{creator_id}.json
    design_briefs/{video_id}.json
```

No existing directory (`modules/`, `optimization/`, `evaluation/`, `observability/`, `renderer_v2/`, `data/design_blueprints/`, `data/creator_style_profiles/`) is modified or moved — fully additive, matching the migration discipline `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §9 already established for its own new package.

### 9.4 Ingestion component — explicitly thin

`competitor_ingestion.py` is a deliberately small wrapper, not a new extraction pipeline: it calls the same `modules/vision_stack/` wrappers and `creator_style/style_extractor.py` logic already proven against this project's own creators, pointed at competitor-provided thumbnail images instead. **Rejected alternative — build a bespoke, lighter-weight competitor analysis pipeline "since we don't need full understanding depth for competitors."** Rejected because it would create two independently-maintained extraction paths producing differently-shaped output for what must ultimately be one comparable embedding space (`CompetitorProfile.style_embedding` vs `ChannelProfile`'s embedding must be computed identically to be comparable at all) — a correctness requirement, not an optimization.

---

## 10. Brand Learning

**Purpose.** Close the gap `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §2.2 explicitly flagged and left unresolved: *"this system cannot currently assert 'this creator always keeps their face large and centered because that's their recognition anchor' — only 'this candidate's face-scale-ratio is similar to past ones.'"* Brand Learning is the component that produces the former from the latter.

**Inputs.** `ThumbnailStyleSignature` history (real, per-video, already in `data/creator_style_profiles/{channel_id}/signatures/`), `StyleDriftDetector` output (real), `AIThumbnailDirectorPlan.elements_to_keep` history across a creator's processed videos (real, per-video, from `director_engine.py`).

**Outputs.** `DesignReason` records (§19.1) tagged as brand rules — e.g. "face_scale_ratio consistently 0.30–0.38 across 22/24 analyzed thumbnails → treat as a brand recognition anchor, high preserve-priority," each with the underlying `EvidenceReference` pointing at the specific `ThumbnailStyleSignature` records that support it.

**Owner.** Extends `intelligence_kb/creator_profiles/profile_aggregator.py` (§9.3) — not a new top-level package, since this is fundamentally an aggregation over already-collected `creator_style/` data, not a new data-collection concern.

**Lifecycle.** Recomputed once a `ChannelProfile.sample_count` crosses each of a small set of confidence thresholds (e.g., re-derive brand rules at 10, 25, 50 samples) rather than on every single run — brand rules should be stable, not jittering on every new video, which is itself consistent with `MODULE10_STYLE_MIN_SAMPLES`'s existing "don't assert a profile before enough evidence" convention.

**Dependencies.** `creator_style/drift_detector.py` (a rule asserted as a brand consistency shouldn't be one that `StyleDriftDetector` has just flagged as actively drifting — the two signals must agree before a rule is asserted, an internal consistency check, not two independent claims).

**Failure modes.** A creator with high `StyleDriftDetector`-reported variance across signatures produces zero or low-confidence brand rules, correctly — Brand Learning must not invent stability that isn't there (§20).

**Computational cost.** A statistics pass (mean/variance per numeric `ThumbnailStyleSignature` field, mode over categorical fields like `subject_placement`) over an already-small per-channel sample set (bounded by `sample_count`, typically tens to low hundreds) — negligible.

**Future extensibility.** New candidate brand-rule dimensions are new fields checked in the same aggregation pass — additive, no schema change to `DesignReason` itself (§24).

**Design decision — a brand rule requires both (a) low variance across ≥ `MODULE10_STYLE_MIN_SAMPLES` and (b) `StyleDriftDetector` reporting no active drift, before being asserted with `confidence > 0.5`.** This two-condition gate is the concrete mechanism that prevents the failure mode named in §1's philosophy — a brand rule is exactly the kind of claim ("this is fundamentally how this creator brands themselves") that is expensive to get wrong downstream (a false brand rule could suppress a genuinely good creative change), so its evidence bar is set higher than a one-off `PsychologyDriver` claim.

---

## 11. Visual Storytelling Engine

**Purpose.** Answer "what story is being told" — grounded, per §1's mandate, in the real `SceneGraph.relationships` (`ElementRelationship`, `SpatialRelation` — already real: `holding`, `wearing`, `looking_at`, `interacting_with`) plus `VideoMetadata` (title, description) and transcript, not invented from the image alone.

**Inputs.** `ThumbnailUnderstanding.scene_graph` (existing), `VideoMetadata` (existing, per Module 2), `AIThumbnailDirectorPlan.story_analysis` (existing free-text field — this engine's job is to make that field's claims checkable, not to replace it).

**Outputs.** `StoryFrame` (§11.1) — a structured claim about what narrative the thumbnail currently communicates, and whether that narrative matches the video's actual content (title/transcript), each field tied to a specific `ElementRelationship` or metadata field.

**Owner.** `intelligence_kb/reasoning/storytelling_engine.py`.

**Lifecycle.** Computed once per pipeline run, per video — no cross-video state (unlike §6–§10's persistent stores).

**Dependencies.** None beyond already-existing `ThumbnailUnderstanding` fields — this engine adds no new detection capability, only structures an existing free-text claim (`story_analysis`) into a checkable one, the same "structure what already exists" principle `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §4.1 applied to `GeminiReasoning`.

**Failure modes.** `SceneGraph.relationships` is empty (e.g., a static object-only thumbnail with no detected interactions) → `StoryFrame.narrative_grounded` is `False` and `narrative_claim` falls back to a minimal, low-confidence description rather than an invented action ("person demonstrating a product" when no `holding`/`interacting_with` relationship exists is exactly the kind of invented detail §1 forbids).

**Computational cost.** Negligible — pure structuring of already-computed fields, no new model call in the common case; a narrow VLM call only when `content_mismatch_detected` (already a real `ThumbnailPsychologyAssessment` field) is `True` and a specific mismatch explanation is needed, reusing the existing `psychology_assessor.py` narrow-call pattern rather than adding a new one.

**Future extensibility.** None required near-term — this is a thin structuring layer by design.

### 11.1 `StoryFrame`

```python
class StoryFrame(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    narrative_claim: str                          # e.g. "creator reacting with shock to an object being shown"
    narrative_grounded: bool                       # True only if backed by ≥1 real ElementRelationship
    supporting_relationship_ids: list[str] = Field(default_factory=list)  # indices into scene_graph.relationships
    title_content_alignment_score: float = 0.5     # 1.0 = thumbnail narrative matches title's implied content
    mismatch_flagged: bool = False                  # mirrors psychology.content_mismatch_detected
    what_should_be_preserved_narratively: list[str] = Field(default_factory=list)   # element_ids
    what_should_change_narratively: list[str] = Field(default_factory=list)         # element_ids
```

---

## 12. CTR Reasoning Engine

**Purpose.** Extend `ThumbnailPsychologyAssessment.drivers` (real, structured `PsychologyDriver` entries — already better than the brief assumes) with evidence from *outside* the current thumbnail: does this driver's claimed effect hold up against the Historical Thumbnail Database and, where linked, real outcome deltas — closing the gap between an isolated per-video claim and a corpus-checked one.

**Inputs.** `ThumbnailPsychologyAssessment` (existing), `ArchetypeMatch` (§7.2), `RetrievedEvidenceSet` (§16) of similar historical/competitor entries with linked outcomes, `optimization/feedback/prior_provider.py`'s existing `rule_confidence_prior` mechanism (reused, not reimplemented).

**Outputs.** `CTRHypothesis` (§12.1) records — one per candidate design lever (e.g., "increasing hero face scale," "adding a text-based curiosity hook") — each carrying an evidence-graded confidence, distinct from `ImprovementAction.expected_ctr_gain` (which is a single-video, un-corpus-checked estimate today).

**Owner.** `intelligence_kb/reasoning/ctr_reasoning_engine.py`.

**Lifecycle.** Computed once per pipeline run.

**Dependencies.** Retrieval Architecture (§16); `optimization/feedback/outcome_store.py` (`mean_delta_by_decision_rule`, real, reused read-only).

**Failure modes.** Sparse or absent linked-outcome evidence for a niche → `CTRHypothesis.evidence_grade` is explicitly `"weak"` or `"pattern_only"` (not silently defaulted to a numeric confidence that looks stronger than it is) — see §20's cold-start row.

**Computational cost.** Bounded by the Retrieval Architecture's top-k (§16) — a handful of similarity lookups plus existing `PriorProvider` calls per hypothesis, not a new model call per hypothesis.

**Future extensibility.** New lever types are new `CTRHypothesis.lever_type` values, no schema change.

### 12.1 `CTRHypothesis`

```python
class CTRHypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis_id: str
    video_id: str
    lever_type: str                                # e.g. "hero_face_scale_increase", "background_declutter"
    target_element_id: Optional[str] = None
    claimed_direction: Literal["increase", "decrease", "neutral"]
    single_video_estimate: Optional[float] = None   # from ThumbnailPsychologyAssessment / ImprovementAction,
                                                       # carried through unchanged, not recomputed
    corpus_supported_estimate: Optional[float] = None   # from linked historical/competitor outcome deltas
    evidence_grade: Literal["strong", "moderate", "weak", "pattern_only", "none"]
    evidence_refs: list["EvidenceReference"] = Field(default_factory=list)
    sample_size: int = 0                             # number of corpus entries the corpus_supported_estimate
                                                       # is derived from — always disclosed, never hidden
```

**Design decision — two separate estimate fields, never merged into one number.** Silently blending a single-video LLM estimate with a corpus-derived one would hide exactly the distinction `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §12's risk-analysis table already warned about for `expected_ctr_gain`/`risk`: *"could give false confidence if not clearly labeled as estimates."* `evidence_grade` and `sample_size` make the evidentiary strength of every number legible to whatever consumes it downstream (a human reviewer, or `Module 9`'s `rationale` field), consistent with that prior document's explicit decision to keep such fields `Optional` and disclosed rather than hard gates.

---

## 13. Emotion Reasoning

**Purpose.** Distinguish two axes this repository's real schema already separates correctly at the detection level but does not yet reason about jointly: **expressed emotion** (`SceneElement.emotion`/`emotion_confidence` — real, per `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §8's InsightFace-attribute-head recommendation, now implemented per `schemas.py`'s `emotion`/`emotion_confidence`/`expression_intensity` fields on `SceneElement`) versus **intended viewer-elicited emotion** (curiosity, surprise, urgency, comfort — a claim about the *audience's* reaction, which is a fundamentally different, and currently entirely unmodeled, quantity).

**Inputs.** `SceneElement.emotion`/`emotion_confidence`/`expression_intensity` for hero/primary elements (existing, real detections — never re-derived by a model call, per §1's determinism-where-possible principle), `ThumbnailPsychologyAssessment.curiosity_gap_score` (existing), `Archetype.typical_emotion` (§7.1, when matched).

**Outputs.** `EmotionProfile` (§13.1) — separating the creator's detected expression from the modeled audience-elicited emotional target, with an explicit, checkable relationship between the two ("does the hero's expressed shock plausibly elicit viewer curiosity, or does it read as generic/mismatched").

**Owner.** `intelligence_kb/reasoning/emotion_reasoning_engine.py`.

**Lifecycle.** Computed once per pipeline run.

**Dependencies.** None new — reuses existing `SceneElement` fields exclusively for the "expressed" half; the "audience-elicited" half draws on Audience Psychology's pattern library (§14) rather than a new detection model.

**Failure modes.** No detected face / no hero element → `EmotionProfile.expressed_emotion` is `None`, and `audience_target_emotion` is derived only from text/hook signals (`DesignBlueprint.hook_type`, when available) — never a fabricated facial-emotion claim for a thumbnail with no face.

**Computational cost.** Negligible — pure aggregation of existing fields plus one lookup against Audience Psychology's pattern library (§14, deterministic, not a model call).

**Future extensibility.** Additional audience-emotion categories are additive enum values.

### 13.1 `EmotionProfile`

```python
class EmotionProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    expressed_emotion: Optional[str] = None          # from SceneElement.emotion, hero only, carried through
    expressed_emotion_confidence: Optional[float] = None   # from SceneElement.emotion_confidence, unchanged
    audience_target_emotion: Optional[str] = None    # e.g. "curiosity", "urgency" — from AudiencePattern match
    alignment_assessment: Literal["aligned", "mismatched", "unclear", "not_applicable"] = "not_applicable"
    alignment_rationale: str = ""
    amplification_recommended: bool = False           # should expressed intensity be increased
    amplification_reason: Optional[str] = None
```

---

## 14. Audience Psychology

**Purpose.** A curated, niche-aware library of named, evidence-checkable persuasion mechanisms (curiosity gap, pattern interrupt, social proof, before/after contrast, numbered-list specificity, FOMO/urgency) — giving §12's `CTRHypothesis.lever_type` and §13's `audience_target_emotion` a controlled vocabulary instead of free text, the same role the Archetype Library (§7) plays for holistic design templates.

**Inputs.** Curated seed set (hand-authored, citing established, publicly-documented persuasion/attention mechanisms — not invented by a model call); frequency evidence from the Historical Thumbnail Database and Competitor Intelligence about which mechanisms actually recur in which niches.

**Outputs.** `AudiencePattern` records (§14.1).

**Owner.** Folded into the Design Pattern Library's storage (§15) as a specialization — `AudiencePattern` and `DesignPattern` (§15.1) share the same underlying `KnowledgeEntry(entry_type=DESIGN_PATTERN)` storage, differentiated by a `pattern_scope` field, rather than as two separate stores, because they are queried together in practice (a copywriting decision usually needs both "what visual pattern" and "what psychological mechanism" evidence at once).

**Lifecycle.** Definitions are curated and versioned, like Archetypes (§7); niche-applicability weighting is refreshed as Historical/Competitor evidence accumulates.

**Dependencies.** §15's shared storage; §16's retrieval.

**Failure modes.** A pattern applied outside the niches it has evidence for (`AudiencePattern.evidenced_niches` empty for the current video's niche) is surfaced with `evidence_grade="pattern_only"` (mirrors §12.1's field) rather than asserted with false specificity.

**Computational cost.** Lookup-only at reasoning time; curation is an offline, human-reviewed process (§24).

**Future extensibility.** New mechanisms are new curated `AudiencePattern` entries.

### 14.1 `AudiencePattern`

```python
class AudiencePattern(BaseModel):
    model_config = ConfigDict(frozen=True)

    pattern_id: str
    pattern_scope: Literal["audience_psychology", "visual_design"]   # differentiates from DesignPattern, §15
    name: str                                       # e.g. "curiosity_gap_partial_reveal"
    description: str
    mechanism_explanation: str                      # why this works, in general persuasion-theory terms
    typical_signals: list[str] = Field(default_factory=list)   # what to look for in SceneGraph/text to detect it
    evidenced_niches: list[str] = Field(default_factory=list)
    curated: bool = True                             # True = hand-authored seed; False = frequency-mined candidate
    version: str = "1.0.0"
```

**Rejected alternative — let the psychology-reasoning model call invent persuasion mechanisms freely per video.** This is exactly the shape of unconstrained reasoning §1 and §3.1's "interpretation, not invention" principle rules out — a model asked to name "the psychological mechanism at play" with no controlled vocabulary will produce plausible-sounding but unverifiable prose, indistinguishable in form from `GeminiReasoning`'s original flat free text that this whole project's history has been moving away from.

---

## 15. Design Pattern Library

**Purpose.** Reusable, granular design *techniques* — distinct from Archetypes (§7), which are holistic templates. A pattern is a technique like "arrow-plus-circle callout," "oversized bold numeral," "left-third face placement with right-side negative space for text," "high-saturation rim light on subject edge." One archetype typically composes several patterns.

**Inputs.** Curated seed set; frequency analysis over the Historical Thumbnail Database and Competitor Intelligence (a legitimate use of unsupervised frequency mining here, unlike §7's rejection of it for archetypes — see distinction below).

**Outputs.** `DesignPattern` records (§15.1), stored alongside `AudiencePattern` (§14) under the same `KnowledgeEntry(entry_type=DESIGN_PATTERN)` type.

**Owner.** `intelligence_kb/design_patterns/pattern_library.py`.

**Lifecycle.** Curated seed patterns are stable; frequency-mined candidate patterns are proposed continuously but held in a `curated=False` state pending human review before being cited with full confidence in a `DesignBrief` (§19) — mirrors §7's rejected-alternative reasoning but applied as an *accepted*, gated pathway here rather than rejected outright.

**Dependencies.** §16 retrieval; §17 embedding (for visual-region-level pattern matching, a finer grain than whole-thumbnail archetype matching).

**Failure modes.** Frequency-mined patterns with low `evidence_grade` are excluded from `DesignBrief` generation entirely until reviewed (§19's evidence gate), preventing an unreviewed statistical artifact from being cited as design guidance.

**Computational cost.** Curation offline; frequency mining runs as a scheduled batch job (same cadence class as Competitor Intelligence refresh, §9), not inline per-video.

**Future extensibility.** New pattern categories are additive.

### 15.1 `DesignPattern`

```python
class DesignPattern(BaseModel):
    model_config = ConfigDict(frozen=True)

    pattern_id: str
    pattern_scope: Literal["audience_psychology", "visual_design"] = "visual_design"
    name: str
    description: str
    applicable_element_types: list["ElementType"] = Field(default_factory=list)   # reuses existing enum
    frequency_in_niche: dict[str, float] = Field(default_factory=dict)   # niche -> observed frequency
    curated: bool = True
    proposed_from_entry_ids: list[str] = Field(default_factory=list)   # provenance, when curated=False
    version: str = "1.0.0"
```

**Why frequency-mining is acceptable here but was rejected for Archetypes (§7).** A `DesignPattern` is a narrow, low-stakes, easily human-verifiable claim ("this specific visual technique recurs at X% frequency in this niche") — reviewing a proposed pattern candidate is a quick visual check. An `Archetype` is a holistic categorical claim that becomes load-bearing evidence across many downstream `DesignBrief` fields at once; the cost of a wrong archetype taxonomy is much higher than the cost of a wrong pattern-frequency estimate, which is why §7 keeps archetype *definition* fully curated while this section allows pattern *candidates* to be statistically proposed, gated by the same `curated`/review flag either way.

---

## 16. Retrieval Architecture

**Purpose.** The single interface the Strategic Reasoning Layer and `DesignBrief` Generator use to pull evidence from the Knowledge Base (§5) — deliberately hybrid, not pure vector search, so every retrieval is auditable.

**Inputs.** A `RetrievalQuery` (§16.1) built from the current `ThumbnailUnderstanding`, `ChannelProfile`, and target `KnowledgeEntryType`(s).

**Outputs.** `RetrievedEvidenceSet` (§16.2) — a bounded, scored, typed list of `KnowledgeEntry` references, never raw unbounded corpus access.

**Owner.** `intelligence_kb/retrieval/retrieval_engine.py`.

**Lifecycle.** Stateless, called once per reasoning-engine invocation per query type — no persistence of its own.

**Dependencies.** Embedding Architecture (§17) for the similarity leg; the four Knowledge Base stores (§5, §7, §8, §9, §15) for the structured-filter leg.

**Failure modes.** See §20's consolidated table (empty result set, over-broad match, stale index).

**Computational cost.** Bounded, always: `top_k` similarity search over a per-facet-filtered subset (facet filters — niche, entry_type, archetype_id — applied *before* the similarity search narrows further, not after, keeping the vector comparison count small and predictable rather than a full-corpus scan followed by filtering).

**Future extensibility.** New facet filter types are additive query fields (§24).

### 16.1 Hybrid retrieval, not pure vector search

**Two-stage design, always:**

1. **Structured filter stage (deterministic).** Apply hard filters first — `niche`, `entry_type`, `channel_id` (for "this creator's own history" queries), `archetype_id` (when already matched, §7.2). This is a simple index lookup, not a model call, and it is what keeps retrieval auditable: a human reviewer can always answer "why was this evidence retrieved" with "same niche, same archetype" before any similarity math enters the picture.
2. **Similarity ranking stage.** Only within the filtered subset, rank by OpenCLIP cosine similarity (§17) and return `top_k` (a small, fixed, configured number — not "everything above a threshold," which can silently return zero or thousands of results depending on corpus density).

**Rejected alternative — a single unified vector database with metadata filtering bolted on (the common "just use a vector DB" pattern).** Rejected for three concrete reasons specific to this project: (a) it would introduce a new infrastructure dependency (a vector DB service) into a project whose entire existing persistence layer is sharded JSON on local disk (`profile_store.py`, `historical_store.py`'s JSONL, every `data/` subdirectory) — a real operational-complexity cost with no demonstrated corpus-scale justification yet (this project's per-creator video counts are, per `data/creator_style_profiles/` sampling above, in the tens-to-low-hundreds range, not the millions a vector DB is built for); (b) filter-after-similarity search (the common vector-DB usage pattern) computes similarity across the *whole* index before filtering, which is both wasteful and, worse, means the top-k results can be dominated by irrelevant-niche entries that merely happen to be visually similar; (c) it obscures the auditability property stage 1 above gives for free. This document instead recommends a linear/IVF-free in-process similarity search (numpy cosine similarity over a filtered, small, per-niche array) — reusing the same computational shape `style_similarity.py` already implements for one-vector-vs-centroid comparison, extended to one-vector-vs-filtered-set.

### 16.2 `RetrievalQuery` / `RetrievedEvidenceSet`

```python
class RetrievalQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_embedding: list[float]                    # from the current thumbnail, §17
    entry_types: list["KnowledgeEntryType"]
    niche: str
    channel_id: Optional[str] = None                 # set for "this creator's own history only" queries
    archetype_id: Optional[str] = None
    top_k: int = 8
    min_similarity: float = 0.0                       # a floor, not a substitute for top_k bounding

class RetrievedEvidenceSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: RetrievalQuery
    entries: list["KnowledgeEntry"] = Field(default_factory=list)   # already sorted, already bounded
    similarity_scores: list[float] = Field(default_factory=list)     # parallel to entries
    total_candidates_before_topk: int = 0             # disclosed, so "8 results out of 8 available" is
                                                          # distinguishable from "8 results out of 4,000"
    retrieved_at: str
```

---

## 17. Embedding Architecture

**Purpose.** One consistent visual embedding backbone across every subsystem that needs similarity (§6 creator style — already real, §7 archetype matching, §8 historical retrieval, §9 competitor comparison, §16 retrieval), plus the one genuinely new capability this document requires: text embedding for title/hook/transcript-level retrieval, which nothing in the current `vision_stack/` provides.

**Inputs.** Images (thumbnails, competitor thumbnails) for the visual backbone; titles/headlines/hooks/transcript excerpts for the text backbone.

**Outputs.** Fixed-dimension float vectors, stored as `KnowledgeEntry.embedding` / `CompetitorProfile.style_embedding` / `ChannelProfile`'s referenced `CreatorStyleEmbedding.embedding`.

**Owner.** `intelligence_kb/embedding/embedding_service.py`.

**Lifecycle.** Stateless service, called on entry-write (§5–§9) and on query construction (§16).

**Dependencies.** `modules/vision_stack/openclip.py`'s `OpenCLIPWrapper` (visual, reused directly); a new, separate text embedding backend (§17.2).

**Failure modes.** Embedding-model version drift (if `OpenCLIPWrapper`'s underlying weights are ever upgraded) would make old and new vectors non-comparable — mitigated by `KnowledgeEntry.embedding_model` being a required, checked field; retrieval must never compare vectors produced by different `embedding_model` values (§20).

**Computational cost.** One forward pass per new entry (write path) — same cost class already paid by `creator_style/style_extractor.py`'s existing embedding calls, extended to a larger set of write sites, not a new cost category.

**Future extensibility.** A backbone upgrade (e.g., a future SigLIP evaluation, explicitly deferred as future work in `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §8) requires a full re-embedding migration, tracked the same way that prior document already tracked it — not undertaken here without demonstrated quality gap (§24).

### 17.1 Visual backbone — reuse `OpenCLIPWrapper`, do not add a second visual embedding model

`OpenCLIPWrapper` is already loaded, already the backbone of the real `creator_style` similarity system, and already produces 512-dim vectors (verified: `self._embedding_dim: int = 512`). Every visual embedding need in this document — archetype matching, historical retrieval, competitor comparison — uses this same instance and the same vector space, which is what makes cross-subsystem comparison ("is this closer to this creator's own history or to a competitor's archetype") mathematically meaningful in the first place. **Rejected alternative — a specialized "design similarity" embedding model, fine-tuned or otherwise distinct from the creator-style embedding.** Rejected because it would fragment the vector space: a `ChannelProfile` embedding and an `Archetype` centroid embedding must live in the same space to be directly compared (§9.1's `CompetitorProfile.style_embedding` is explicitly noted as "directly comparable to `ChannelProfile`'s vector" for exactly this reason) — introducing a second backbone for a subset of comparisons would silently break that comparability everywhere it isn't used.

### 17.2 Text backbone — a genuinely new, small capability

Nothing in `vision_stack/` embeds text (OpenCLIP's `encode_text` exists and is used for zero-shot-style classification internally per its wrapper, but is not currently exposed as a general-purpose text-embedding utility for retrieval, verified against `openclip.py`'s consumer list). Retrieval over titles/hooks/headlines (needed for §12's CTR-lever retrieval and §14's audience-pattern matching against text signals) needs a text vector space. **Recommendation: reuse `OpenCLIPWrapper.encode_text` directly** (already loaded, already in the same vector space as the visual encoder — CLIP's joint embedding space is exactly designed for cross-modal comparison) rather than adding a new, separate text-only embedding model. This closes the one real gap §0.2 identified without introducing a new model dependency, and keeps text and image evidence comparable in the same retrieval call.

**Rejected alternative — add a dedicated sentence-embedding model (e.g., a compact sentence-transformer) for higher text-similarity quality.** A plausible future upgrade, explicitly deferred: CLIP's text encoder is tuned for short, image-descriptive captions, not long-form transcript similarity, so a dedicated text model would likely out-perform it for transcript-level retrieval specifically. This document does not recommend adding it now, following the same "don't add a new model without a demonstrated, specific gap" discipline `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §8 already applied to Florence-2/YOLO-World/MediaPipe — title/hook-length text (a few words, close to CLIP's training distribution) is the primary near-term retrieval need, and full-transcript retrieval is listed as Future Work (§24) pending a demonstrated quality shortfall.

---

## 18. Memory Architecture

**Purpose.** Distinguish, explicitly, the different memory tiers this document and the existing repository together maintain — because conflating them is a real risk (e.g., treating a single decision-rule's outcome delta as if it validated an entire creative direction).

**Inputs/Outputs/Owner.** Not a single subsystem — this section is the map across §6, §8, §9, and the existing `optimization/feedback/` package, clarifying what each tier is for for the reader implementing against this document.

| Tier | Scope | Owner (existing or new) | Update cadence | What it must never be used for |
|---|---|---|---|---|
| Per-decision-rule outcome memory | One rule_id's historical delta | `optimization/feedback/outcome_store.py`, `prior_provider.py` — **real, unchanged** | Per pipeline run | Explaining a whole creative direction's success — too narrow a grain |
| Per-channel style memory | One creator's own visual/compositional tendencies | `creator_style/profile_store.py` — **real, unchanged**; extended (read-only) by `ChannelProfile` (§6.2) | Per pipeline run | Cross-creator or cross-niche claims |
| Global Knowledge Base | Archetypes, historical corpus, competitor set, design/audience patterns | §5–§9, §14–§15 — **new** | Per pipeline run (historical/creator) or scheduled (competitor, pattern mining) | Overriding a creator's own established brand rules without disclosure (§10's gate) |
| **New: DesignBrief-level outcome memory** | Whether a *whole* creative direction (an archetype choice + a set of `DesignReason`s) improved outcomes, not just one decision rule | **New**, §18.1 | Per pipeline run, linked at `DesignBrief` completion | Per-rule confidence tuning — too coarse a grain, `PriorProvider` already owns that |

### 18.1 The one new memory component this document adds: `DesignBriefOutcomeLink`

`optimization/feedback/outcome_recorder.py`'s real `OptimizationOutcome` already links `decisions_applied` to a `delta`. This document adds one new, additive field enabling the missing tier above:

```python
# OptimizationOutcome (optimization/feedback/outcome_recorder.py) — additive field only
design_brief_id: Optional[str] = None   # links this outcome to the DesignBrief (§19) that produced
                                          # the decisions_applied list, when one was used
```

This single additive field is what makes §23's Future Learning loop possible without touching `OutcomeRecorder`'s existing logic, `OutcomeStore`'s existing query methods, or `PriorProvider`'s existing per-rule confidence computation — all three keep working unmodified for the tiers they already own.

---

## 19. DesignBrief Generation

**Purpose.** The central deliverable of this entire document — one new, additive artifact that packages everything from §5–§18 into a form Module 5/5.5/9 can consume, every field traceable to evidence, per §1.

**Inputs.** `ThumbnailUnderstanding` (existing), `ChannelProfile`/`CreatorProfile` (§6), `ArchetypeMatch` (§7.2), `RetrievedEvidenceSet` (§16), `StoryFrame` (§11.1), `CTRHypothesis[]` (§12.1), `EmotionProfile` (§13.1), matched `AudiencePattern`/`DesignPattern` (§14–§15), `DifferentiationSummary` (§9.2, when competitors are configured for the niche).

**Outputs.** One `DesignBrief` (§19.3) per video, persisted at `data/intelligence_kb/design_briefs/{video_id}.json`.

**Owner.** `intelligence_kb/brief/design_brief_generator.py`, with `evidence_validator.py` enforcing the grounding gate below at construction time — not as an optional lint step.

**Lifecycle.** Computed once per pipeline run, immediately after the Strategic Reasoning Layer (§4's pipeline diagram), before Module 5.

**Dependencies.** Everything above — this is intentionally the layer with the most dependencies, since consolidation is its entire purpose; every dependency is read-only from this component's perspective.

**Failure modes.** See §20 — most centrally, "insufficient evidence for a confident brief" must produce a *smaller, honestly-scoped* `DesignBrief` (fewer `design_reasons`, more fields `None`), never a fabricated complete one.

**Computational cost.** Assembly-only — no new model call beyond what §11–§14 already made; this is a structuring and validation step, the same shape as `ThumbnailUnderstanding`'s own top-level assembly in `understanding_engine.py`.

**Future extensibility.** New evidence source types plug in as new optional constructor inputs; `DesignBrief`'s own schema grows additively (§24).

### 19.1 `EvidenceReference` and `DesignReason`

```python
class EvidenceReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_type: Literal[
        "scene_graph_element", "scene_graph_relationship", "psychology_driver",
        "knowledge_entry", "creator_profile_field", "competitor_profile_field",
        "archetype_match", "audience_pattern", "design_pattern", "outcome_record",
    ]
    source_id: str                                   # element_id / entry_id / competitor_id / etc.
    source_field: Optional[str] = None                # which specific field on the source is being cited
    excerpt_or_value: str = ""                          # a short, literal quotation of the cited fact —
                                                          # never a paraphrase that could drift from the source

class DesignReason(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason_id: str
    claim: str                                        # e.g. "Increase hero face scale — this creator's own
                                                          # historical average face_scale_ratio (0.34) exceeds
                                                          # the current thumbnail's 0.21"
    reason_type: Literal[
        "brand_consistency", "ctr_evidence", "competitor_differentiation",
        "archetype_alignment", "audience_psychology", "narrative_grounding",
    ]
    confidence: float
    evidence: list[EvidenceReference] = Field(default_factory=list, min_length=1)   # non-empty is enforced —
                                                                                        # see §19.2
    target_element_id: Optional[str] = None
```

### 19.2 The grounding gate — enforced, not advisory

`evidence_validator.py` rejects, at construction time, any `DesignReason` whose `evidence` list is empty — a structural guarantee, the same class of guarantee `ThumbnailUnderstanding`'s Pydantic validation already gives against dangling `element_id` references (`docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §12's risk table). This is the concrete mechanism realizing §1's "no decision without evidence" mandate — not a style guideline for whoever writes the reasoning-engine prompts, but a schema-level constraint every `DesignReason`, from every one of §11–§15's engines, must pass through before it can appear in a `DesignBrief` at all.

### 19.3 `DesignBrief`

```python
class ReasoningTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    steps: list[str]                                  # ordered, human-readable log of retrieval + reasoning
                                                          # calls made while assembling this brief — auditability,
                                                          # not re-derivable logic
    retrieval_queries_used: list["RetrievalQuery"] = Field(default_factory=list)
    generated_at: str

class DesignBrief(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    channel_id: str
    niche: str
    matched_archetype: Optional["ArchetypeMatch"] = None
    story_frame: "StoryFrame"
    emotion_profile: "EmotionProfile"
    ctr_hypotheses: list["CTRHypothesis"] = Field(default_factory=list)
    differentiation: Optional["DifferentiationSummary"] = None
    design_reasons: list[DesignReason] = Field(default_factory=list)   # every entry evidence-gated, §19.2
    creative_direction_summary: str = ""                # short prose synthesis, itself required to cite
                                                            # ≥1 design_reasons entry per claim it makes —
                                                            # checked by evidence_validator.py, not free-form
    redesign_aggressiveness_recommendation: Literal["conservative", "moderate", "aggressive"] = "moderate"
    reasoning_trace: ReasoningTrace
    brief_confidence: float                              # aggregate, computed deterministically from the
                                                            # confidence/evidence_grade distribution of its
                                                            # constituent parts — never a separately-asserted number
    status: Literal["complete", "partial_cold_start", "error"] = "complete"
    generated_at: str
```

### 19.4 How this reaches Module 5 / 5.5 / 9 without changing their responsibility

Following the exact precedent `main.py` already establishes for `understanding=understanding`:

```python
# Module 5 (redesign_spec_engine.build_redesign_specification) — additive parameter only
def build_redesign_specification(intelligence, understanding=None, design_brief=None): ...

# Module 5.5 (design_blueprint_engine.build_design_blueprint) — additive parameter only
def build_design_blueprint(intelligence, redesign_spec, metadata, design_brief=None): ...

# Module 9 (decision_engine.run_decision_engine) — additive parameter only
def run_decision_engine(..., design_brief=None): ...
```

When `design_brief` is provided, Module 5/5.5's existing deterministic-template logic (`strategy_engine.py`, `copywriter.py`, `layout_planner.py` — all real, all unchanged) gains richer, more specific inputs to compile from — `DesignBrief.creative_direction_summary` and `design_reasons` feeding `RedesignSpecification.overall_rationale` and `elements_to_preserve`, exactly the "same deterministic compiler, richer facts" pattern `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §8.3 already established for Module 6. Module 9's `ResolvedDecision.rationale` may now cite `DesignReason.reason_id` values directly. **When `design_brief` is `None` — a cold-start creator with no Knowledge Base evidence yet, or the parameter simply omitted — every consumer falls back to its current, already-real, `ThumbnailUnderstanding`-only behavior exactly as it works today.** No existing behavior is contingent on this new artifact existing.

---

## 20. Failure Modes

Consolidated across every section above, because several recur in the same shape and are best understood together.

| Failure mode | Where it occurs | Effect if unmitigated | Mitigation |
|---|---|---|---|
| Cold-start sparsity — new creator, new niche, empty Knowledge Base | §6, §8, §9, §12 | Fabricated-sounding confident claims from too little evidence | `profile_established=False` / `evidence_grade="none"`/`"pattern_only"` propagate up to `DesignBrief.status="partial_cold_start"`; every numeric confidence is computed from actual sample size, never asserted independently |
| Embedding-model version drift | §17 | Silently incomparable vectors, corrupted similarity rankings | `KnowledgeEntry.embedding_model` required and checked before any comparison; re-embedding is a tracked migration (§24), not silent |
| Stale competitor snapshot | §9 | Differentiation claims based on outdated competitor designs | `CompetitorProfile.last_ingested_at` + `status` tracked and surfaced; `DifferentiationSummary` can flag staleness in its own `EvidenceReference` excerpts |
| Retrieval over-broad match (structured filters too loose) | §16 | Evidence retrieved from the wrong niche/context cited as if directly relevant | Two-stage hybrid retrieval (§16.1) — hard filters always applied before similarity ranking, never the reverse |
| Retrieval empty result | §16 | A reasoning engine has nothing to cite | `RetrievedEvidenceSet.total_candidates_before_topk = 0` is a valid, disclosed result — downstream engines must handle it as "no evidence," not retry with loosened filters silently (loosening filters silently reintroduces the over-broad-match risk above) |
| Unreviewed frequency-mined pattern treated as established fact | §15 | A statistical artifact cited with false confidence | `curated=False` patterns excluded from `DesignBrief` generation until human-reviewed (§19.2's gate applies transitively) |
| Brand rule asserted from too little or unstable data | §10 | Suppressing a genuinely good creative change because it "violates the brand" | Two-condition gate: sample-size threshold *and* `StyleDriftDetector` agreement, both required before `confidence > 0.5` |
| Grounding gate bypassed by a future engineer adding a new reasoning engine that writes `DesignReason` directly | §19.2 | Reintroduces exactly the ungrounded-claim failure mode this whole document exists to prevent | `evidence_validator.py` enforced at `DesignBrief` construction, not at each individual engine — a single, unavoidable choke point, the same "validate at the model boundary" pattern `ThumbnailUnderstanding` already uses |

---

## 21. Benchmarking

**Purpose.** Extend, not duplicate, the already-real `evaluation/benchmarking/` package (`historical_store.py`, `golden_sample_manager.py`, `regression_detector.py`) to cover this document's new artifact.

**Inputs.** `DesignBrief` records produced across pipeline runs; the existing `BenchmarkRecord`/`PipelineRunReport` schema.

**Outputs.** An additive `BenchmarkRecord` field and a new, narrow regression check.

**Owner.** Extends `evaluation/benchmarking/`, does not fork it.

**Lifecycle.** Runs on the same cadence as the existing golden-sample regression suite (`golden_sample_manager.py`, real, pinned golden creator set).

**Dependencies.** `regression_detector.py` (real, reused).

**Failure modes.** A schema or reasoning-logic change to the Intelligence Engine that silently reduces evidence density (more `DesignBrief`s landing at `status="partial_cold_start"` than the golden baseline) is exactly the kind of regression this section's addition is meant to catch, mirroring `regression_detector.py`'s existing purpose for generation-quality regressions.

**Computational cost.** Runs within the existing golden-sample suite's cost envelope — no new infrastructure, one new metric.

**Future extensibility.** Additional `DesignBrief`-specific benchmark metrics are additive fields on the extended record.

```python
# BenchmarkRecord (modules/models.py) — additive fields only
design_brief_evidence_density: Optional[float] = None   # fraction of design_reasons with evidence_grade
                                                            # in {"strong","moderate"} vs {"weak","pattern_only","none"}
design_brief_status_distribution: Optional[dict[str, int]] = None   # counts of complete/partial_cold_start/error
                                                                       # across the golden run, tracked over time
```

**Design decision — measure evidence density, not brief "quality" via a subjective score.** A subjective 1–10 "brief quality" score would reintroduce the exact ungrounded-scalar problem §3.1's root-cause table already diagnosed in `GeminiReasoning`'s original scores. Evidence density is a purely structural, computable metric (fraction of claims with strong/moderate evidence) — it measures the one property this document is actually designed to guarantee, and regressions in it are directly actionable (which engine, which retrieval query, stopped finding evidence it used to find).

---

## 22. Evaluation

**Purpose.** Distinguish from Benchmarking (§21, which tracks regression over time against a fixed golden set) — Evaluation here means per-run quality assessment of a single `DesignBrief`, feeding both human review and the downstream acceptance gate.

**Inputs.** A single `DesignBrief`; `optimization/validation/acceptance_gate.py` (real, existing gate for generation candidates — reused as the integration point, not duplicated).

**Outputs.** An evidence-density score (same metric as §21, computed per-run rather than aggregated) and a `downstream_acceptance_correlation` — did decisions grounded in this brief's `design_reasons` end up in the final accepted candidate (via `optimization/validation/acceptance_gate.py`'s existing accept/reject record) more or less often than ungrounded decisions.

**Owner.** `evaluation/quality/` gains one new scorer module, following the existing pattern (`attractiveness_scorer.py`, `composition_scorer.py`, etc. — all real, all single-responsibility scorer modules feeding `aggregator.py`).

**Lifecycle.** Runs once per pipeline run, alongside the existing quality scorers.

**Dependencies.** `evaluation/quality/aggregator.py` (real, reused as the composition point — this document adds one more scorer input to an already-real aggregation mechanism, not a new aggregation mechanism).

**Failure modes.** None beyond what §20 already covers — this section is purely a read/measure layer over already-validated `DesignBrief` data.

**Computational cost.** Negligible — reads already-computed fields.

**Future extensibility.** Correlating brief-grounded decisions against Module 7's real, existing `CandidateRankingEngine` acceptance outcomes (mentioned in `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §8.5) is a natural, low-cost extension once enough `design_brief_id`-linked outcomes (§18.1) accumulate.

---

## 23. Future Learning

**Purpose.** Design the feedback loop that lets the Knowledge Base's *weighting* — not its curated taxonomy — improve from real outcomes, using §18.1's `design_brief_id` link.

**Inputs.** `OptimizationOutcome` records with a populated `design_brief_id` (§18.1); the `DesignBrief`s they reference.

**Outputs.** Adjusted `Archetype.example_count`/implicit weighting in retrieval ranking, and `AudiencePattern`/`DesignPattern` niche-applicability weighting — never a change to curated `Archetype`/`AudiencePattern` *definitions* themselves.

**Owner.** A thin new consumer inside `intelligence_kb/`, reading `optimization/feedback/outcome_store.py` (read-only, unchanged) and writing weighting adjustments back into the Knowledge Base stores.

**Lifecycle.** Runs on the same batch cadence as Competitor Intelligence refresh and pattern-frequency mining (§9, §15) — not inline per-video, since a single video's outcome should not visibly swing a shared, cross-creator archetype weighting.

**Dependencies.** §18.1's additive field; `optimization/feedback/prior_provider.py`'s existing bounded-adjustment convention (`max(-0.2, min(0.2, mean_delta))`) — reused as the pattern for bounding this document's own weighting adjustments, for the same reason: an unbounded adjustment from a small, noisy sample is a real risk this repository's existing feedback code already guards against.

**Failure modes.** Overfitting archetype/pattern weighting to a small number of early outcomes — mitigated by the same `min_sample_size` gate `PriorProvider` already applies (`OPTIMIZATION_FEEDBACK_MIN_SAMPLE_SIZE`, real, reused directly rather than reinvented with a different threshold).

**Computational cost.** Batch job, bounded by outcome-record volume since last run — same cost class as existing scheduled work in this design (§9, §15).

**Future extensibility.** This is explicitly the seam where a future, more ambitious learning mechanism (e.g., learned archetype-embedding fine-tuning) could attach — not designed here, flagged as long-horizon future work below, consistent with this document's own evidence-gated, incremental-extension discipline.

**Explicitly rejected — an end-to-end trained model (e.g., a learned CTR predictor or a fine-tuned ranking model) replacing this rule-based weighting loop.** Consistent with `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §8's Kontext-evaluation decision ("no decision made without benchmark evidence") and this repository's demonstrated preference throughout for deterministic, auditable mechanisms over learned black boxes wherever the deterministic option is viable (`hierarchy_calculator.py`, `PriorProvider`'s bounded-adjustment formula) — a trained model here would trade away exactly the evidence-traceability property (§1, §19.2) this entire document is built to guarantee, for a capability improvement that is unproven at this project's current data scale (tens-to-low-hundreds of samples per creator, per §10's observation). Long-horizon future work, gated on a demonstrated data-scale and quality justification neither exists nor is assumed here.

---

## 24. Extensibility

**Versioning.** Every curated definition (`Archetype`, `AudiencePattern`, `DesignPattern`) carries a `version` field already (§7.1, §14.1, §15.1) — a new version supersedes the old via the same `superseded_by` convention `KnowledgeEntry` uses (§5.1), never an in-place mutation, preserving the ability to reproduce a past `DesignBrief`'s reasoning exactly against the definitions that existed when it was generated.

**Adding a new archetype.** Author a new `Archetype` record (curated, human-reviewed per §7's rejected-alternative discussion) with its `defining_scene_graph_pattern` predicate and a small set of seed `KnowledgeEntry(entry_type=ARCHETYPE_EXAMPLE)` examples — no code change to `archetype_matcher.py` required, since matching is data-driven against the `Archetype` store.

**Adding a new competitor.** Append to the operator-configured competitor list per niche (§9) — no schema change.

**Adding a new niche.** Every schema in this document keys on `niche: str` as a plain string, not a closed enum, specifically so new niches require zero schema migration — a deliberate, low-cost extensibility choice made throughout §6–§19.

**Multi-channel creators.** `CreatorProfile.channel_ids` (§6.1) is a list from the start, anticipating the "one creator, several channels" case §6 identified as unrepresentable in the current single-`channel_id`-keyed `StyleProfileManifest` — the extension point already exists in the schema, not deferred to a future migration.

**Embedding backbone upgrade.** Tracked explicitly as a versioned, disclosed migration (`KnowledgeEntry.embedding_model`, §17, §20) rather than assumed away — any future backbone change (e.g., a SigLIP evaluation, per `docs/NEXT_GEN_THUMBNAIL_INTELLIGENCE_ARCHITECTURE.md` §8's own deferred item) requires a full re-embedding pass across every store in §5–§9, planned as a batch migration job, not a silent in-place swap.

**Long-horizon future work (explicitly deferred, not designed here, consistent with this document's evidence-gated discipline):** dedicated sentence-embedding backend for full-transcript retrieval (§17.2); unsupervised clustering as a *candidate-proposal* input to human-reviewed archetype curation (§7's rejected-alternative note); learned CTR/ranking models once sufficient linked-outcome data volume exists (§23); cross-channel `CreatorProfile.cross_channel_consistency_score` computation logic, once multi-channel data exists to compute it against (§6.1).

---

## 25. Summary — what this document changes, and what it explicitly does not

| Changed / added (this document) | Unchanged (verified real, not touched) |
|---|---|
| New `intelligence_kb/` package (§9.3) | `modules/thumbnail_understanding/` — Perceptual Reasoning Layer, real, complete |
| New `DesignBrief` artifact and its constituent schemas (§7, §9, §11–§19) | `renderer_v2/` — out of scope entirely, per the brief's own top priority-order instruction |
| New optional `design_brief=None` parameter on Module 5 / 5.5 / 9 entry points | Every existing required parameter and existing behavior when `design_brief` is omitted |
| One additive field on `OptimizationOutcome` (§18.1) | `optimization/feedback/outcome_recorder.py`/`outcome_store.py`/`prior_provider.py` internal logic |
| One additive field pair on `BenchmarkRecord` (§21) | `evaluation/benchmarking/historical_store.py`/`golden_sample_manager.py`/`regression_detector.py` internal logic |
| One new `evaluation/quality/` scorer module (§22) | `evaluation/quality/aggregator.py`'s existing composition mechanism |
| Read-only extension of `creator_style/` via `ChannelProfile` (§6.2) | `modules/creator_style/` — Creator Style Learning, real, untouched internally |

This document's entire contribution is the layer between "the pipeline understands this one thumbnail correctly" (real, already true) and "the pipeline's recommendation for this one thumbnail is grounded in more than this one thumbnail" (not yet true, and the reason the brief was written) — delivered as one new artifact, `DesignBrief`, built from four new knowledge stores and four new reasoning engines, every field of it evidence-gated by construction, and consumed by the existing pipeline exactly the way `understanding=understanding` already showed this project how to extend a stage without breaking the ones around it.
