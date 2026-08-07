# Next-Generation Thumbnail Intelligence Architecture

**thumbnail-ai**
**Status:** Architecture only. Zero implementation code.
**Author role:** Principal AI Architect.
**Source of truth:** `poison-2-0-0-7/thumbnail-ai` @ `main`, re-verified for this document (`modules/models.py`, `modules/config.py`, `modules/vision_stack/`, `modules/decision_engine.py`/`decision_components/`, `modules/thumbnail_intelligence.py`, `modules/creator_style/`, `modules/image_generator.py`, `workflows/`, and every architecture doc under `docs/` produced in this project's history — 27 documents at time of review).

---

## 0. Grounding note — module numbering, verified before anything else

Per this brief's own "trust the repository" instruction, the numbering in its Audit section does not match `main`:

| Brief calls it | Real repository identity | Real location |
|---|---|---|
| "Module 4 — Thumbnail Intelligence" | **Matches.** Module 4, `modules/thumbnail_intelligence.py` | Correct as named |
| "Module 5 — Creator Style Learning" | Real Module 5 is the **Redesign Spec engine** (`modules/redesign_spec_engine.py`). Creator Style Learning is real, implemented, and lives at `modules/creator_style/` — committed under the filename `docs/MODULE10_CREATOR_STYLE_LEARNING_ARCHITECTURE.md`, itself already flagging a collision with the *real* Module 10 (Asset Composer) | `modules/creator_style/` |
| "Module 6 — Redesign Specification" | Real Module 6 is the **Prompt Compiler** (`modules/prompt_compiler.py`). Redesign Specification is real Module 5 | `modules/redesign_spec_engine.py` |
| "Module 7 — Image Generation Interface" | **Matches** (`modules/image_generator.py`) | Correct as named |
| "Module 8 — Decision Engine" | Real Module 8 is **Asset Extraction** (`modules/asset_extraction_engine.py`). The Decision Engine is real Module 9 (`modules/decision_engine.py`) | `modules/decision_engine.py` |

This document audits the **five real components the brief is actually pointing at** — Thumbnail Intelligence, Creator Style Learning, Redesign Spec, Prompt Compiler, Decision Engine, plus Image Generation as context for Part 9 — using their correct names throughout, and does not introduce a sixth numbering scheme.

---

## 1. Executive Summary

The repository's own accumulated diagnostic history (`docs/MODULE7_RENDER_EXECUTION_ARCHITECTURE.md`, `docs/MODULE7_EDIT_MODE_ACTIVATION_FIX_ARCHITECTURE.md`) already proved the rendering *infrastructure* now works: source images are loaded, VAE-encoded, staged through ControlNet/IPAdapter-conditioned inpainting. What those documents did not — and could not — diagnose is the finding this document confirms directly against the models: **the pipeline's "understanding" of a thumbnail is a single flat signal, not a structured one.** `GeminiReasoning` (Module 4's own AI-reasoning stage, `modules/models.py:385-410`) already asks a single Gemini call to estimate CTR potential, curiosity gap, emotional impact, strengths, weaknesses, and elements to preserve — in one undifferentiated pass, with no scene graph, no subject-importance ranking, no relationship model, and — critically — **no connection back to the deterministic CV signals** (`ocr`, `faces`, `objects`, `colors`, `composition`) sitting right next to it in the same `ThumbnailIntelligence` object. Module 9's `ResolvedDecision` (`modules/models.py:1440-1453`) already carries `confidence`, `rationale`, and `priority_rank` — genuinely more than the brief assumes — but has no `expected_ctr_gain`, no `risk`, and no `dependencies` field; a decision to replace a background and a decision to change a headline are resolved with no model of how they interact.

**The generated thumbnails looking "almost identical to the original" is therefore not a rendering-conservativeness problem — it is what a correctly-conservative renderer produces when fed decisions that were never confidently, specifically, or hierarchically reasoned about in the first place.** Module 9 today resolves conflicts between what different upstream modules *assert*; it does not resolve conflicts between what a thumbnail *needs* — because nothing upstream of it ever built a ranked, causally-connected model of what the thumbnail needs. This document designs that layer.

---

## 2. Current Architecture Review (Part 1)

### 2.1 Module 4 — Thumbnail Intelligence

- **Purpose/Responsibilities:** Produce `ThumbnailIntelligence` — OCR (`OCRResult`), face analysis (`FaceAnalysis`), object detection (`list[DetectedObject]`), color profile (`ColorProfile`), composition analysis (`CompositionAnalysis`), and one `GeminiReasoning` pass, per video.
- **Inputs:** the downloaded thumbnail (Module 3), video title/description/transcript context (Module 2).
- **Outputs:** `data/analysis/{video_id}.json`.
- **Dependencies:** `modules/vision_stack/` (GroundingDINO, SAM2, BiRefNet, BiSeNet, InsightFace-multi, Depth Anything, TEED, OpenCLIP — nine wrapped models, per `docs/AI_Vision_Stack_V2.1_Architecture.md`), plus one Gemini API call.
- **Weaknesses (verified, not assumed):**
  - `objects: list[DetectedObject]` is a **flat list** — no parent/child, no "the mug is on the desk," no "the person is holding the microphone." There is no relationship model anywhere in this schema.
  - `reasoning: Optional[GeminiReasoning]` is **one LLM call reasoning over everything at once** — CTR, emotion, story, strengths/weaknesses, all produced in a single pass with no intermediate structure the deterministic CV outputs could ground or constrain. The LLM is not shown a scene graph; it is shown (per `thumbnail_intelligence.py`'s existing prompt-assembly pattern) a text summary of the other stages' outputs.
  - No subject-importance ranking exists — `faces: FaceAnalysis` lists faces; nothing declares which face is "the hero" when more than one is present.
  - No depth-based layering exists in Module 4 itself, despite `DepthAnythingWrapper` already being a registered, loaded model in `vision_stack/` — it is currently consumed only by Module 8 (Asset Extraction)'s composition asset, not by Module 4's own reasoning.
- **Scalability/Maintainability:** the nine-model vision stack's lazy-load/VRAM-lifecycle design (`vision_stack/lifecycle.py`, `resources.py`) is genuinely well-built and scales fine to more models of the same shape — this is not a weakness.
- **Technical debt:** the single-LLM-call reasoning pattern is the load-bearing debt — every new "understanding" requirement anyone has asked this pipeline for so far (this document included) has to either cram another instruction into that one Gemini prompt or bypass Module 4 entirely, because there is no structured intermediate representation to extend.
- **Performance bottleneck:** none specific to Module 4 beyond the already-documented general VRAM/model-loading costs shared by every vision-stack consumer.
- **AI limitation:** a single, un-decomposed LLM judgment call is exactly the shape of system most prone to shallow, generic output ("good lighting, clear subject") rather than thumbnail-specific, decision-actionable findings — there is no mechanism forcing the model to ground a claim like "curiosity gap: high" in a specific visual fact.

### 2.2 Creator Style Learning (`modules/creator_style/`)

- **Purpose:** per-channel style signature/embedding accumulation, similarity/drift detection, style-aware prompt guidance and ranking bonus.
- **Verified real and implemented** (`style_extractor.py`, `profile_store.py`, `style_similarity.py`, `drift_detector.py`, `style_prompt_guidance.py`, `style_aware_ranking.py`, `data/creator_style_profiles/`), following its own architecture doc closely, with `MODULE10_STYLE_*` config constants aliased to the doc's originally specified `OPTIMIZATION_STYLE_*` names.
- **Weakness directly relevant to this brief:** `StyleExtractor.extract_signature()` derives color/composition signature fields from the **whole-frame** `ThumbnailIntelligence.colors`, not from a VRE background-only region as its own architecture document specified — meaning "creator style" today conflates background style with whatever the dominant subject happens to be wearing/holding in each thumbnail, diluting the actual channel identity signal. Flagged as a factual deviation from that component's own governing document, not invented here.
- **Weakness relevant to this document's Part 6 (AI Director):** style guidance (§7 of that architecture) only ever appends bounded phrases to existing prompt fields — it has no representation of *why* a color or composition choice is "on-brand" beyond raw visual similarity, so it cannot explain a preservation decision the way a human creative director would ("this creator always keeps their face large and centered because that's their recognition anchor" is not a fact this system can currently assert — only "this candidate's face-scale-ratio is similar to past ones" is).

### 2.3 Module 5 — Redesign Specification

- **Purpose:** turns `ThumbnailIntelligence` + `GeminiReasoning` into a `RedesignSpecification` — a first-pass, still fairly coarse, "what to change" document, upstream of Module 5.5 (Copywriter/Layout) and Module 9 (Decision Engine).
- **Weakness:** because Module 4's reasoning is flat (§2.1), Module 5's redesign spec inherits the same flatness — it can say "improve background" but has no causal chain back to a specific composition-analysis fact or a specific CTR-potential sub-score explaining *why* that recommendation exists, beyond `GeminiReasoning.redesign_recommendations`'s free text.

### 2.4 Module 6 — Prompt Compiler

- **Purpose:** deterministic compilation of `PromptPackage` from upstream structured fields — genuinely one of the strongest-engineered components in this repository (per `docs/IMAGE_GENERATION_ARCHITECTURE.md`'s own framing: "Module 6 compiles, it does not reason").
- **Weakness relevant to this brief:** because it is *deliberately* non-reasoning (a correct design choice, not a defect — see §13's "Keep" list), it cannot compensate for upstream reasoning gaps. A better Module 6 is not the fix; a better upstream signal is.

### 2.5 Module 9 — Decision Engine

- **Purpose:** per-element `KEEP`/`REPLACE`/`ENHANCE`/`REMOVE`/`ADD` resolution (`decision_engine.py`, `decision_components/conflict_resolver.py`), producing `DecisionManifest`.
- **Verified, real strength:** `ResolvedDecision` already carries `confidence: float`, `rationale: str`, `priority_rank: int`, `source: DecisionSource`, and `machine_reasoning: dict[str, Any]` (`models.py:1440-1453`) — genuinely closer to "professional redesign plan" territory than the brief assumes.
- **Verified, real gap:** no `expected_ctr_gain` field, no `risk` field, no `dependencies` field (i.e., no representation that "replacing the background" and "changing the headline color" might interact, or that one decision's risk is conditional on another decision also landing correctly). `priority_rank` orders decisions but does not explain *why* one outranks another beyond whatever `rationale`'s free text says.
- **Weakness this document treats as central:** Module 9 resolves conflicts between what upstream modules *assert*, via `ConflictResolver`'s existing confidence/priority mechanism (reused, per `docs/MODULE9_AI_DECISION_ENGINE_ARCHITECTURE.md`, by the Creator Style Learning integration already) — but nothing upstream of Module 9 ever asserts a *ranked model of what actually drives this specific thumbnail's performance*. Module 9 is a correctly-built arbitration layer sitting downstream of an under-specified opinion.

### 2.6 Module 7 — Image Generation Interface (context for Part 9, §9 below)

Already exhaustively audited across four prior documents in this repository (`MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md`, `MODULE7_RENDER_EXECUTION_ARCHITECTURE.md`, `MODULE7_EDIT_MODE_ACTIVATION_FIX_ARCHITECTURE.md`, `MODULE7_CONTROLNET_COMPATIBILITY_ARCHITECTURE.md`) — this document does not re-audit it in depth, and treats its infrastructure (staged inpainting, ControlNet/IPAdapter capability resolution, multi-candidate generation, ranking) as real and load-bearing, per the brief's own framing that this infrastructure now functions. §9 evaluates only whether that infrastructure's *organization* (not its correctness) is sufficient for the richer decision model this document proposes.

---

## 3. Root Cause Analysis (Part 2 — does the pipeline truly understand a thumbnail?)

Checked against `GeminiReasoning` and every model that feeds it, question by question:

| Question | Can the pipeline answer this today? | Evidence |
|---|---|---|
| What is happening? | Partially — `visual_storytelling_notes` (free text) | No structured scene description; no scene graph |
| Why would people click? | Partially — `ctr_potential_score` + `curiosity_gap_score` (two scalars) | No decomposition into *which* visual elements drive the score |
| Who is the hero? | **No** | `FaceAnalysis`/`DetectedObject` have no importance-rank or "primary subject" field |
| What emotion exists? | Partially — `emotional_impact: str`, a single label | No per-face emotion, no confidence, no basis in facial-expression detection (no emotion model in `vision_stack/`) |
| What visual hierarchy exists? | **No** | `CompositionAnalysis` has `balance_score`/`symmetry_score`/`rule_of_thirds_score` — aggregate composition quality metrics, not a hierarchy (ranked list of "what draws the eye first, second, third") |
| What should be preserved? | Partially — `elements_to_preserve: list[str]` (free text) | Not connected to any specific `DetectedObject`/`FaceAnalysis` entry — a free-text list, not a set of element references Module 9 can act on programmatically without an LLM re-parsing it |
| What should be replaced? | Partially — `weaknesses` + `redesign_recommendations` (free text), further processed by Module 5 | Same disconnection — recommendations are prose, not structured references to specific detected elements |
| What should be emphasized? | **No** | No emphasis/importance field anywhere in the schema |
| What should be removed? | Partially, via Module 9's `REMOVE` decision — but Module 9 decides this from Module 8's asset manifest + Module 5's spec, not from any direct "this element should be removed" signal in Module 4's own output | Confirms the reasoning gap is upstream of Module 9, not inside it |

**Conclusion:** the pipeline has real, working pieces of understanding, but they are unconnected scalars and free text, not a structured model. This is why generated output looks "almost identical" — every downstream module (5, 5.5, 9) is working from the same thin signal, however well each of them individually processes it.

---

## 4. Next Generation Architecture (Part 3 — Thumbnail Intelligence Engine v2)

### 4.1 Design principle

**Everything Module 4 currently computes is reused, not replaced.** `OCRResult`, `FaceAnalysis`, `DetectedObject`, `ColorProfile`, `CompositionAnalysis` remain exactly as they are — they become **inputs to a new second stage**, not outputs to discard. The new stage's job is exclusively to *structure and connect* what already exists, plus fill the specific, narrow set of gaps §3 identified (hierarchy, relationships, importance ranking, per-element preserve/change/emphasize/remove tagging) — not to re-invent object/face/OCR detection, which already works.

### 4.2 New component: `ThumbnailUnderstandingEngine` (new stage, sits after today's Module 4, before Module 5)

```
ThumbnailUnderstandingEngine.understand(intelligence: ThumbnailIntelligence) -> ThumbnailUnderstanding
```

Produces a single new, structured artifact, `ThumbnailUnderstanding`, replacing `GeminiReasoning`'s role as "the" reasoning output (kept as a legacy-compatible field within the new model for one migration cycle — see §13) with a genuinely structured representation:

```python
class SceneElement(BaseModel):
    element_id: str                      # stable reference, joins to DetectedObject/FaceAnalysis entries
    element_type: Literal["person","object","text","logo","background"]
    source_ref: str                      # which upstream detection this element_id maps to
    importance_rank: int                 # 1 = most important
    role: Literal["hero","supporting","prop","background","distraction"]
    emotion: Optional[str] = None        # per-person, when element_type == "person"
    emotion_confidence: Optional[float] = None
    relationships: list["ElementRelationship"] = []
    preserve: bool
    preserve_reason: Optional[str] = None
    change_recommendation: Optional[str] = None
    emphasize: bool = False
    remove_candidate: bool = False

class ElementRelationship(BaseModel):
    subject_element_id: str
    relation: Literal["holding","wearing","standing_on","looking_at","occluding","adjacent_to","part_of"]
    object_element_id: str
    confidence: float

class SceneGraph(BaseModel):
    elements: list[SceneElement]
    relationships: list[ElementRelationship]
    hero_element_id: Optional[str]        # None only when no person/primary subject detected

class VisualHierarchy(BaseModel):
    reading_order: list[str]              # element_ids in the order a viewer's eye is modeled to visit them
    dominant_element_id: str
    hierarchy_basis: Literal["size","contrast","position","face_priority","text_priority"]

class ThumbnailPsychologyAssessment(BaseModel):
    ctr_potential_score: float            # reused field name/semantics from GeminiReasoning — not renamed
    ctr_drivers: list[str]                # element_ids + short reason, replacing free-text-only strengths
    curiosity_gap_score: float
    curiosity_gap_mechanism: Optional[str] # e.g. "partial reveal", "incongruity", "question implied by text"
    content_mismatch_detected: bool
    mismatch_explanation: Optional[str]

class ThumbnailUnderstanding(BaseModel):
    video_id: str
    scene_graph: SceneGraph
    hierarchy: VisualHierarchy
    psychology: ThumbnailPsychologyAssessment
    weaknesses: list[str]                 # kept as free text — not every finding needs full structure
    legacy_reasoning: Optional[GeminiReasoning] = None   # migration bridge, §13
    analyzed_at: str
```

### 4.3 How this is actually produced — multi-model, not one LLM call (Part 4 detail in §10)

The `element_id`-referenced structure above is only trustworthy if it's grounded in the deterministic detections, not re-hallucinated by an LLM. Pipeline, in order:

1. **Deterministic grounding pass** — every `SceneElement` is created *from* an existing `DetectedObject`/`FaceAnalysis` entry first (a 1:1 mapping, zero new detection), with `element_id` deterministically derived (hash of bbox + class label, matching this repo's existing hash-based ID conventions elsewhere).
2. **Relationship pass** — a single, constrained VLM call (Gemini Vision, or a locally-hosted VLM per §10) is shown the image **plus the already-extracted element list with bboxes overlaid** (a grounding technique, not asking the model to detect from scratch) and asked only to (a) assign `role`/`importance_rank`/`relationships` between the *given* elements, and (b) identify the `hero_element_id` — a narrower, more constrained, more gradeable task than today's single open-ended reasoning call, and one that produces structured output the schema above can validate (reject a relationship referencing an `element_id` that doesn't exist, reject an `importance_rank` collision, etc.) — the model's freedom is deliberately reduced, which is what makes structured, decision-actionable output achievable instead of generic prose.
3. **Hierarchy pass** — computed **deterministically**, not by a model call: `VisualHierarchy.reading_order` derived from a documented, auditable formula combining `CompositionAnalysis`'s existing `subject_placement`/`balance_score`, each element's bbox area (from step 1, already normalized per `models.py`'s `BoundingBox` convention), and face-priority weighting (faces generally read before objects, a documented, overridable heuristic, not a hidden one) — kept out of the LLM entirely, since eye-tracking-order approximation from geometric facts is exactly the kind of task this project's "deterministic where possible" convention (established across every prior architecture document) says should not be delegated to a model call.
4. **Psychology pass** — the one component that remains closest to today's Gemini call, but now *grounded*: the model is shown the structured scene graph + hierarchy from steps 1–3 (not the raw pixels alone) and asked to produce `ctr_drivers`/`curiosity_gap_mechanism` **as references into the already-built element list**, not free-standing prose — the model's job narrows from "understand and describe the whole thumbnail" to "explain the CTR/curiosity signal in terms of the elements we've already identified," which is both a more constrained (more reliable) task and one whose output Module 9 can consume programmatically.

This is the direct fix for §3's findings: every "partially" in that table becomes a structured, element-referenced field, and the "no" answers (hero, hierarchy, emphasis) get dedicated fields for the first time — without discarding a single existing detection stage.

---

## 5. Module Responsibilities (post-redesign)

| Stage | Responsibility | Change from today |
|---|---|---|
| Module 4 (unchanged) | Deterministic CV extraction: OCR, faces, objects, colors, composition | None |
| **`ThumbnailUnderstandingEngine`** (new) | Scene graph, hierarchy, psychology — §4 | New stage, inserted after Module 4 |
| Module 5 (Redesign Spec) | Consumes `ThumbnailUnderstanding` instead of raw `GeminiReasoning` | Input contract change only — internal logic re-targets its existing free-text-parsing paths onto now-structured fields, a narrowing of what it has to infer, not a rewrite of its responsibility |
| Creator Style Learning | Unchanged responsibility; gains one new consumer relationship — `preserve_reason` on hero elements can now cite style-profile similarity directly (§4.2's `preserve_reason: str`), closing part of §2.2's "cannot explain a preservation decision" gap | Additive |
| Module 6 (Prompt Compiler) | Unchanged — still deliberately non-reasoning, per §2.4 | None |
| Module 9 (Decision Engine) | Gains three new `ResolvedDecision` fields (§7) and consumes `ThumbnailUnderstanding` as an additional input alongside its existing ones | Additive schema extension, existing conflict-resolution logic reused |
| Module 7 | Consumes the same `DecisionManifest`/`GenerationPlan` contract it does today — §9 (Part 9) evaluates whether its internal organization needs to change, concludes largely no | Minimal |

---

## 6. Complete Data Flow

```
Module 3 (thumbnail) ──▶ Module 4 (CV extraction, unchanged)
                                │
                                ▼
                    ThumbnailUnderstandingEngine  ◀── overlaid-bbox VLM call (§4.3 step 2)
                                │
                    ThumbnailUnderstanding (scene graph + hierarchy + psychology)
                                │
                                ▼
Module 5 (Redesign Spec) ──▶ Module 5.5 (Copywriter/Layout) ──▶ Module 6 (Prompt Compiler)
                                │
                                ▼
Module 8 (Asset Extraction) ──▶ Module 9 (Decision Engine)  ◀── ThumbnailUnderstanding (element references)
                                │                              ◀── Creator Style profile (existing)
                    DecisionManifest (extended, §7)
                                │
                                ▼
Module 10 (Composition) ──▶ Module 10.5 (Planner) ──▶ Module 7 (Generation, unchanged internals)
```

---

## 7. Decision Engine Extension & JSON Schemas (Parts 7 and 9 combined)

`ResolvedDecision`'s three new fields directly satisfy the brief's Part 7 request. The already-real fields are unchanged:

- **Why** → already real (`rationale: str`) — unchanged.
- **Priority** → already real (`priority_rank: int`) — unchanged.
- **Confidence** → already real (`confidence: float`) — unchanged.
- **Expected CTR gain** → new, `expected_ctr_gain: Optional[float]`, populated from `ThumbnailUnderstanding.psychology.ctr_drivers`'s per-element attribution when a decision's target matches a scored element — `None` when no such attribution exists (e.g. a purely stylistic decision with no claimed CTR effect), never a fabricated number.
- **Risk** → new, `risk: Literal["low","medium","high"]`, a small deterministic rule (not a model call): `high` when a decision's target is the `hero_element_id` or has `depends_on_decision_ids` non-empty, `medium` when it touches a `role="supporting"` element, `low` otherwise — auditable, not learned.
- **Dependencies** → new, `depends_on_decision_ids: list[str]`, populated by `ConflictResolver`'s existing logic (already resolving cross-decision interactions, per §2.5) — this is a new *output* field surfacing a computation the resolver's existing confidence/priority mechanism already implicitly performs, not new resolution logic.

```python
# ResolvedDecision (modules/models.py) — additive fields, existing fields unchanged
expected_ctr_gain: Optional[float] = None     # signed estimate, from Understanding's psychology pass
risk: Literal["low","medium","high"] = "low"
depends_on_decision_ids: list[str] = Field(default_factory=list)
```

```python
# GenerationPlan (Module 10.5, existing) — one additive field
scene_graph_reference: Optional[str] = None    # path to the ThumbnailUnderstanding used, for PORCE traceability
```

---

## 8. Model Responsibilities (Part 4 — multi-model stack evaluation)

**Principle carried over from this repository's entire existing vision-stack design (`docs/AI_Vision_Stack_V2.1_Architecture.md`): specialist models cooperate; no single model is asked to do a job a purpose-built model does better.** Evaluated against what's already loaded:

| Capability | Already in repo | Recommendation | Why |
|---|---|---|---|
| Open-vocab object detection | GroundingDINO | **Keep.** Evaluate GroundingDINO 1.5 as a drop-in upgrade only if its license/weight availability is confirmed compatible with this project's fully-local, offline requirement (unverified as of this document; flagged, not assumed) | Already correctly chosen; 1.5 is a quality upgrade to the same job, not a different capability |
| Segmentation | SAM2 | **Keep.** SAM-HQ is a plausible upgrade for fine boundary quality — evaluate only if Module 8's asset-extraction mask quality is empirically found insufficient (no such finding exists yet) | Don't replace a working component on a hypothetical quality gap |
| Background matting | BiRefNet | **Keep** | No brief-driven reason to change |
| Face parsing | BiSeNet | **Keep** | No brief-driven reason to change |
| Face recognition/analysis | InsightFace-multi | **Keep for identity.** Add a dedicated **facial expression/emotion model** — InsightFace's own attribute models (age/gender/emotion heads already exist in the InsightFace model zoo, a same-vendor addition, not a new integration surface) — since §3 confirmed `emotion: str` is currently a single LLM label with no CV grounding at all | Closes the one clearly-verified emotion gap with the smallest possible new integration surface |
| Depth | Depth Anything | **Keep; extend consumption.** Already loaded, already used by Module 8 — §4.3's hierarchy calculator should also consume it (closer-to-camera elements read as more prominent), an additive consumer, not a new model | Reuse, don't duplicate |
| Edge detection | TEED | **Keep** — feeds ControlNet canny conditioning (Module 7), unrelated to this document's scope |
| Embeddings/similarity | OpenCLIP | **Keep** — already the backbone of Creator Style Learning; **evaluate SigLIP as an alternative embedding backbone only as a Future Work item**, since switching would require re-embedding every existing `data/creator_style_profiles/` centroid — a real migration cost this document does not recommend incurring without a demonstrated quality gap |
| Scene/relationship reasoning (§4.3 steps 2/4) | **Gap — currently a single flat Gemini call** | **Gemini Vision remains the default backend** (already integrated) but the new `vlm_backends/` abstraction (§10) should support a **local alternative — Qwen2.5-VL** as the strongest currently-available open-weight option for grounded, bbox-referenced visual reasoning (its documented strength, relative to Molmo/InternVL, at exactly the "reason over provided regions" task §4.3 needs) | Keeps the fully-local option viable without forcing an immediate migration off a working, already-integrated Gemini path |
| Florence-2, YOLO-World, MediaPipe | Not in repo | **Not recommended for addition.** Florence-2/YOLO-World's core capability (open-vocab detection + grounding) is already covered by GroundingDINO; MediaPipe's face-mesh/landmark capability is already covered by InsightFace+BiSeNet for this project's actual needs | Avoid capability-duplicating additions — every new model is a new VRAM/maintenance cost this project's own architecture discipline (established across 27 prior documents) consistently avoids paying without a specific, named gap |

### 8.1 Layer Generation (Part 5)

Module 8 (Asset Extraction, real, existing) already produces `PersonAsset`/`ObjectAsset`/`SceneAsset`/`TypographyAsset`/`CompositionAsset` — a real layer system, organized by *asset category*, not by the richer per-purpose taxonomy the brief's Part 5 asks for. **This document recommends against building furniture/food/electronics/logos/emoji as literal distinct layer types** — they are all just `ObjectAsset` with a class label; a new layer *type* per object category multiplies Module 8's schema without adding capability, since `ObjectAsset.class_label` already differentiates them. What genuinely doesn't exist and is worth adding, additively:

```python
# ObjectAsset (Module 8, existing) — additive
depth_layer: Optional[float] = None        # from Depth Anything, already loaded — enables proper
                                             # occlusion-aware compositing, not currently present
priority: Optional[int] = None             # reuses SceneElement.importance_rank (§4.2) by reference
scene_element_ref: Optional[str] = None    # joins to ThumbnailUnderstanding.scene_graph
```

Lighting/shadow/reflection/atmosphere as **separate learned layers** is explicitly **not recommended** as a near-term addition — no model in this repository's stack (or evaluated above) produces these as distinct, editable layers, and inventing that capability is a multi-month research effort disproportionate to this document's diagnosed root cause (§1). Flagged in §16 as long-horizon future work.

### 8.2 AI Director (Part 6)

**Recommendation: do not build a sixth new module called "AI Director."** Its described responsibilities — what should change, what should never change, how to increase CTR, how to preserve identity, how to improve composition/storytelling/psychology — are exactly Module 9's existing, real responsibility (§2.5), extended by this document's §4/§7 inputs. A separate "Director" module sitting between Understanding and Decision Engine would duplicate Module 9's conflict-resolution machinery (`ConflictResolver`, already real, already reused three times across this project's history) for no added capability. **The "AI Director" is Module 9, once it has §4's structured Understanding and §7's risk/dependency/CTR-gain fields to reason over.**

### 8.3 Prompt Generation (Part 8)

**Module 6 remains deliberately non-reasoning (§2.4) — this document does not add reasoning to the Prompt Compiler.** What it adds is richer *input*: `PromptCompiler`'s existing per-field deterministic-template pattern (already established, reused by Creator Style Learning's `StylePromptGuidance`) gains new template inputs sourced from `ThumbnailUnderstanding`: `hierarchy.dominant_element_id` → an explicit "keep X as the visual focal point" instruction; `scene_graph.relationships` → explicit relational instructions ("the person should still be holding the microphone") that today's flat object list cannot express; `psychology.ctr_drivers` → phrased CTR-preserving instructions, templated exactly like every other Module 6 field (deterministic phrase selection, not LLM-generated prose). **"Professional creative direction" is achieved by giving the same deterministic compiler richer, more specific facts to compile from — not by making the compiler itself a reasoning system.**

### 8.4 Image Generation Reorganization (Part 9)

**Finding: the current ComfyUI workflow architecture's *organization* does not need to change; its *inputs* need to be richer, which this document already provides via Module 9's extensions.**

- **ControlNet/IPAdapter/regional prompting/masked diffusion** — already real, already staged, per `MODULE7_V2_EDITING_ENGINE_ARCHITECTURE.md`'s design and `MODULE7_RENDER_EXECUTION_ARCHITECTURE.md`'s confirmation that the staging mechanics work. **Not redesigned here.**
- **Layered inpainting** — Module 7's existing per-region staged-edit design (`BackgroundEditStage`/`ObjectEditStage`) already anticipates exactly this; this document's `scene_element_ref`/`depth_layer` additions (§8.1) give those stages a richer per-region signal (occlusion-aware ordering) than they have today, without requiring new node types in the ComfyUI graph itself.
- **Flux Kontext** — genuinely relevant for an *editing-first* workflow, since Kontext-family models are specifically trained for instruction-based image editing rather than generation-from-noise. **Recommended as a Future Work evaluation, not a decision made here** — no verified benchmark exists comparing Kontext against this repository's current SDXL/FLUX-schnell staged-inpainting approach, and recommending a checkpoint swap without that evidence would violate this document's own "trust the repository, don't invent" mandate.
- **Should these be reorganized? No** — the organization is sound. The fix is upstream (§4).

### 8.5 Quality Control (Part 10)

The already-real `CandidateRankingEngine`/`CandidateClusteringEngine`/QA hard-gate machinery is extended with one new rejection class and one new retry strategy, both additive:

```python
# New AcceptanceResult rejection reason, alongside existing ones (style_identity_lost, over_edited, etc.)
"scene_graph_violated"   # a KEEP-decided hero_element_id's region changed beyond a configured
                          # perceptual-distance threshold — reuses CandidateClusteringEngine's existing
                          # perceptual-hash machinery applied to one element's region instead of the whole frame
```

**Automatic retry strategy:** when `scene_graph_violated` fires, retry with the *same* seed/strategy but an increased `MODULE7_V2_DENOISE_BY_DECISION["keep"]` protection (lower denoise / stronger mask enforcement for that specific element) rather than a blind full re-roll — a targeted retry, reusing the existing per-decision denoise table.

---

## 9. Folder Structure

```
modules/
  thumbnail_understanding/                  # new package, sibling to thumbnail_intelligence.py
    __init__.py
    understanding_engine.py                 # ThumbnailUnderstandingEngine
    scene_grounding.py                      # step 1, deterministic element grounding
    relationship_reasoner.py                # step 2, constrained VLM call
    hierarchy_calculator.py                 # step 3, deterministic reading-order formula
    psychology_assessor.py                  # step 4, grounded VLM call
    vlm_backends/                           # pluggable model backends, §8
      __init__.py
      gemini_vision_backend.py
      local_vlm_backend.py                  # Qwen2.5-VL / InternVL / Molmo
data/
  thumbnail_understanding/{video_id}.json   # ThumbnailUnderstanding, new, video_id-sharded (consistent with every other module's convention)
```

No existing directory is touched — fully additive, matching this repository's established migration discipline.

---

## 10. Sequence Diagram (single video, happy path)

```
Client/main.py
   │ for creator in creators:
   ▼
Module 4.extract() ─────────────────▶ ThumbnailIntelligence
   │
   ▼
ThumbnailUnderstandingEngine.understand(intelligence)
   │  1. ground elements from intelligence.objects/faces  (deterministic)
   │  2. VLM call: assign role/importance/relationships/hero over the grounded elements
   │  3. compute VisualHierarchy.reading_order  (deterministic formula)
   │  4. VLM call: psychology assessment, referencing the built scene graph
   ▼
ThumbnailUnderstanding ─────────────▶ Module 5.derive_spec(..., understanding=...)
   │
   ▼
... (Module 5.5, Module 6 unchanged) ...
   │
   ▼
Module 9.resolve(..., understanding=..., style_profile=...) ─▶ DecisionManifest (extended)
   │
   ▼
... (Module 10, 10.5, Module 7 unchanged) ...
```

---

## 11. Performance Analysis

- **New cost:** two additional model calls per video (§4.3 steps 2 and 4) versus today's one Gemini call — roughly a 2x increase in LLM-call latency/cost for the reasoning stage specifically, not for the whole pipeline (CV extraction, generation, etc. are unchanged). Deterministic steps 1 and 3 add negligible cost (pure Python over already-computed data).
- **Mitigation:** steps 2 and 4 are independent and can be parallelized (no data dependency between "assign relationships" and "assess psychology" until both feed the final `ThumbnailUnderstanding` object) — a straightforward `asyncio.gather`-shaped optimization, not a new architecture.
- **VRAM:** the one new recommended model (§8, InsightFace's existing emotion-attribute head) is same-vendor, likely already loadable within InsightFace's existing loaded-model footprint (unverified without direct benchmarking against this project's target hardware, flagged not assumed) — a materially smaller cost than any of the "add a new foundation model" options this document explicitly recommends against.

---

## 12. Risk Analysis

| Risk | Mitigation |
|---|---|
| Step 2/4 VLM calls introduce new failure modes (malformed structured output, hallucinated `element_id` references) | `ThumbnailUnderstanding`'s Pydantic validation rejects any relationship/hero reference to a nonexistent `element_id` at the model-construction boundary — a structural, not best-effort, guarantee, consistent with every other module's frozen-model validation convention in this repository |
| Two-call reasoning pipeline could be slower/costlier at scale than today's single call | §11's parallelization mitigates latency; cost is a real, disclosed trade-off, not hidden |
| `expected_ctr_gain`/`risk` (§7) could give false confidence if not clearly labeled as estimates | Field naming/typing makes them `Optional`, and this document does not propose using them as hard gates anywhere — only as `ResolvedDecision` metadata for `rationale`/PORCE explainability, never as an automatic accept/reject threshold |
| Local VLM backend (§8, Qwen2.5-VL) may not match Gemini's quality for the relationship/psychology tasks | `vlm_backends/` abstraction makes Gemini the default; local backend is opt-in, evaluated, not forced |

---

## 13. Migration Plan (Part 11)

| Component | Decision |
|---|---|
| Module 4's five CV stages (OCR/faces/objects/colors/composition) | **Keep**, unmodified |
| Nine-model vision stack (`vision_stack/`) | **Keep**, extend with one InsightFace attribute head (§8) |
| `GeminiReasoning` | **Refactor** — retained as `ThumbnailUnderstanding.legacy_reasoning` for one migration cycle (any code still reading the old field keeps working), but no longer the primary reasoning artifact; deprecated for removal once Module 5/5.5/9 are confirmed migrated to reading `ThumbnailUnderstanding` directly |
| Module 5, 5.5, 6 | **Refactor** input contracts only — internal logic largely unchanged, now consuming structured fields instead of parsing free text where such parsing existed |
| Module 9 (`ResolvedDecision`, `ConflictResolver`) | **Keep + additive extension** (§7) — no rewrite |
| Module 7, ComfyUI workflows | **Keep**, unmodified (§8.4's finding) |
| Creator Style Learning | **Keep + one bug fix** (§2.2's VRE-region deviation — flagged as a factual finding for whoever owns that component, not redesigned here) |
| "AI Director" as a literal new module | **Delete from the plan** — explicitly not built (§8.2); its responsibility is Module 9's, extended |
| Per-object-category layer types (furniture/food/electronics as distinct types) | **Delete from the plan** — `ObjectAsset.class_label` already differentiates them (§8.1) |
| Lighting/shadow/reflection/atmosphere as distinct learned layers | **Future Work** — not designed in depth here (§8.1) |
| Flux Kontext evaluation | **Future Work** — no decision made without benchmark evidence (§8.4) |

---

## 14. Implementation Roadmap

1. `ThumbnailUnderstanding` schema + `scene_grounding.py` (deterministic element grounding, step 1) — inert until wired, zero risk.
2. `hierarchy_calculator.py` (deterministic, step 3) — inert until wired, zero risk, independently testable against `CompositionAnalysis` fixtures already present in this repo's test data.
3. `relationship_reasoner.py`/`psychology_assessor.py` (VLM calls, steps 2/4) behind the `vlm_backends/` abstraction, Gemini backend first (reuses existing integration).
4. `ResolvedDecision` additive fields (§7) + `ConflictResolver` populating `depends_on_decision_ids` from its existing resolution trace.
5. Module 5/5.5/6 input-contract migration to `ThumbnailUnderstanding` (with `legacy_reasoning` fallback live throughout, §13).
6. Quality-control extension (§8.5, `scene_graph_violated` + targeted retry).
7. Local VLM backend (Qwen2.5-VL) evaluation — opt-in, after step 3 has production data to compare against.
8. Creator Style Learning VRE-region fix (§2.2) — independent, can land any time.

## 15. Priority Order

**1 → 2 → 3 → 4** is the critical path — everything downstream (§3's understanding gap) is fixed only once steps 1–4 land; **5** is what actually changes generated-thumbnail behavior in production (the input contract switch); **6** hardens it; **7–8** are independent, lower-priority hardening/evaluation work that can proceed in parallel with 5–6 once 1–4 are stable.
