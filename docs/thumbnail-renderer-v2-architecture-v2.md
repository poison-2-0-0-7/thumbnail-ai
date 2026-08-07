# Renderer V2 — A Reconstruction Engine, Not a Generator
### Architecture for a fully-local, layer-based YouTube thumbnail redesign system (RTX 4060, 8GB VRAM)
### v2 — Intelligence-First Pipeline

---

## 0. The one architectural decision everything else follows from

The brief already names it: **reconstruction, not regeneration.** But that phrase has a sharper technical consequence than "use img2img with low denoise." It means:

> **The face, hands, and product should never pass through a diffusion model at all, in most edits.**

Every VRAM-hungry, identity-drifting, "almost right" thumbnail problem you'll hit traces back to routing pixels that don't need to change through a generative model anyway. InstantID/PuLID/IP-Adapter exist to *approximate* an identity a diffusion model is about to destroy. That's solving a problem your architecture shouldn't have. If you segment the creator out as a locked raster layer and only regenerate what's *behind* and *around* them, you don't need identity-preservation adapters for 80% of edits — you need them only for the harder case (re-posing, hand fixing, relighting the face itself).

This reframes the whole system from "one big edit model with adapters bolted on" to:

- **Pixels that are already good → segment, lock, relight in-place, never regenerate.**
- **Pixels that are bad or missing (background, glow, depth) → generate/inpaint.**
- **Pixels that are structured, not photographic (text, arrows, icons, shapes) → do NOT use diffusion at all. Render them procedurally.**

That third bullet is the biggest departure from what most "AI thumbnail" tools do, and it's the biggest quality win. AnyText, GlyphControl, and TextDiffuser exist because generic diffusion models can't spell — but *you don't need a diffusion model to draw text on a thumbnail.* Pillow/Cairo/Skia render arbitrary fonts, outlines, drop shadows, and gradients with zero VRAM cost and perfect legibility. The "looks AI generated, not professional" complaint your business exists to fix is disproportionately caused by melted diffusion-rendered text and mushy diffusion-rendered icons. Real typography engines solve that instantly. Same logic applies to arrows, circles, numbered badges, logos — these are vector/raster compositing problems, not generative ones. Reserve diffusion for what actually requires it: background synthesis, relighting, and inpainting.

This single reframing cuts your VRAM budget, your latency, your failure surface, and your "looks AI" problem simultaneously. Keep it as the governing rule for every layer below.

**v2 extends the same principle upstream, into reasoning.** Everything below §2 was already disciplined about not inventing pixels that didn't need inventing. v2 applies that discipline to *creative decisions*, not just pixels: the system should never invent a narrative, an emotion, or a design direction that isn't grounded in something real — the video's actual title, its actual transcript, the creator's actual brand history, what actually performs in this channel's niche. **Interpretation, not invention** is the creative-reasoning analog of "reconstruction, not regeneration," and it's the reason v2 introduces a dedicated reasoning stage (§3) instead of letting the planner both invent a creative direction and execute it in the same step, from thin context, as v1 did.

---

## 1. Pipeline

```
Original Thumbnail ──┐
Video Metadata ───────┤
Thumbnail OCR ─────────┼──▶  Input Bundle
Video Title ────────────┤
Transcript ──────────────┘
          │
          │ (Original Thumbnail specifically feeds the decomposer below;
          │  the full Input Bundle feeds the Intelligence Engine in §3)
          ▼
┌─────────────────────┐
│ 1. Scene Decomposer  │  SAM2 + GroundingDINO + BiRefNet matting + Depth-Anything V2
│    (segmentation,    │  → per-instance masks, alpha mattes, depth map, camera/pose estimate
│     depth, matting)  │
└─────────┬────────────┘
          │ SceneGraph
          ▼
┌───────────────────────────────┐
│ 2. Thumbnail Intelligence       │  reasons over SceneGraph + full Input Bundle +
│    Engine                       │  Style Learning + historical thumbnails + channel
│    (reasoning, not rendering)   │  style + competitor set + archetype library
└─────────┬───────────────────────┘
          │ DesignBrief
          ▼
┌─────────────────────┐
│ 3. Execution Planner  │  SceneGraph + DesignBrief → ExecutionPlan (deterministic
│    (ExecutionPlan)    │  translation only — no creative decisions made here)
└─────────┬────────────┘
          ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Layer Engines (each independent, each optional)        │
│                                                             │
│  Locked layers (never regenerated, only transformed):      │
│    • Creator / Face / Hands / Product   → matte + relight  │
│    • Logos / Branding                   → matte + preserve │
│                                                             │
│  Generative layers (diffusion, only these touch a model):  │
│    • Background          → FLUX.1-Fill-dev / BrushNet      │
│    • Relighting           → IC-Light V2 (on locked subject)│
│    • Depth-aware shadow synthesis                          │
│                                                             │
│  Procedural layers (zero diffusion, classical graphics):   │
│    • Typography          → Pillow/Skia + font/kerning rules│
│    • Arrows / Icons / Badges / Shapes → SVG asset library  │
│    • Glow / Particles / Vignette / Grain → shader-style FX │
│    • Color grade / LUT   → OpenCV/Pillow curve ops          │
└─────────┬───────────────────────────────────────────────┘
          ▼
┌─────────────────────┐
│ 5. Compositor         │  Depth-ordered alpha compositing, edge feathering, harmonization
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ 6. Quality Engine     │  Identity/brand/composition preservation scores + CTR/aesthetic scores
└─────────┬────────────┘
          ▼
     score ≥ threshold? ──No──▶ Targeted Critique Loop ──▶ back to Execution Planner
          │Yes                   (NOT back to the Intelligence Engine — see §3.7)
          ▼
     7. Exporter (PNG/JPG, multiple aspect variants, email/DM-ready sizes)
```

The load-bearing change from v1 is the insertion of §3 between the Scene Decomposer and the planner, and the corresponding narrowing of the planner's job (§4). Everything from §4 onward — Layer Engines, Compositor, Quality Engine — is architecturally unchanged from v1; those subsystems don't know or care that a new reasoning stage was added upstream of them, because the contract they consume (ExecutionPlan) hasn't changed shape, only where its content originates.

---

## 2. Scene Decomposer

**Job:** turn one flat thumbnail into a layered scene graph before anything else touches it.

| Component | Model | Why | RTX 4060 8GB fit |
|---|---|---|---|
| Open-vocabulary detection | GroundingDINO (tiny/base) | Finds "person", "hand", "logo", "product", "phone", etc. from text prompts, feeds boxes to SAM2 | ~1.5GB, fine |
| Instance segmentation | SAM2 (small/base+) | Turns boxes into pixel masks; also does video-quality mask propagation if you ever do batches | ~1–2GB fp16 |
| Matting refinement | BiRefNet or MODNet | SAM2 masks are good but hard-edged; hair/fingers need soft alpha mattes for clean compositing | <1GB |
| Depth | Depth-Anything V2 (small/base) | Gives you a depth map for shadow placement, DoF blur, "background" vs "foreground" separation without manual masking | <1GB |
| Face landmarks/keypoints | InsightFace (buffalo_l) | Cheap, CPU-capable, used by Quality Engine for face-similarity scoring and by the Relight engine for face-region targeting | negligible |

All five run sequentially, not concurrently — load, infer, unload. On 8GB you don't need concurrent residency; this stage is <5 seconds total and none of these models are the bottleneck.

**Output:** a JSON scene graph — list of instances (class, mask, alpha matte, bbox, depth layer, is_locked) plus a global depth map. This is the substrate every downstream layer reads from — now including the Intelligence Engine (§3), which reasons *about* the scene graph's contents but never modifies it. This is also the artifact your existing Scene Understanding / Asset Extraction modules should be upgraded to emit, so you're not duplicating work you already built.

This subsystem is unchanged in v2. It was already correctly scoped: decomposition is perception, not reasoning, and it stays that way.

---

## 3. Thumbnail Intelligence Engine

### 3.1 What this subsystem is, and isn't

The Thumbnail Intelligence Engine is not a renderer. It is not a diffusion model. It does not touch a pixel. Its entire output is a single structured document — the DesignBrief (§3.4) — and its entire job is producing that document well. Everything downstream of it in the pipeline is execution; everything at or upstream of it is perception and reasoning. That boundary is the architectural point of this subsystem: v1 asked one planner to both *decide what the thumbnail should say and feel like* and *translate that into per-layer parameters* in a single step, from a context window that only contained the scene graph and some style signals. That's asking a single reasoning pass to do editorial thinking and mechanical translation at once, and it's why v1's planner was the hardest subsystem to improve without regressing something else — a change to make it better at composition reasoning risked making its JSON output less reliable, and vice versa. Splitting those two jobs across two subsystems, connected by a well-defined document, means each one gets to be good at one thing.

### 3.2 Responsibilities

The Intelligence Engine's job is to answer, in writing, the questions a competent creative director would ask before touching a single layer:

- What kind of thumbnail is this, structurally? (archetype)
- What is this video actually about, and what's the one thing the thumbnail needs to communicate? (narrative, primary story)
- What should someone feel in the half-second they see this in a feed? (target emotion)
- What should the eye see first, second, third? (visual hierarchy)
- What can't change, and why? (brand constraints, identity constraints)
- What's our theory for why this redesign will out-perform the original? (CTR hypotheses)
- What does "done" look like for composition, lighting, typography, and background, specifically enough that a downstream engine doesn't have to guess? (the four *_goals fields)
- What are we deliberately not attempting, and what could go wrong? (rendering constraints, risk analysis)
- How will we know if it worked? (success metrics)

It answers these by reasoning, not by rendering — it has no access to any generative model and no ability to produce or modify pixels. This is enforced architecturally, not just by convention: the Intelligence Engine's only interface to the rest of the system is the DesignBrief document it emits.

### 3.3 Inputs

| Input | Source | Why it matters here |
|---|---|---|
| SceneGraph | Scene Decomposer (§2) | What's actually in the frame — instances, composition, depth structure |
| Video title | Input Bundle | Primary signal for narrative and genre; often the single strongest CTR-psychology signal available |
| Video transcript | Input Bundle | Grounds the "primary story" field in what the video actually delivers, not just what the title implies — prevents the brief from promising something the content doesn't back up |
| Thumbnail OCR | Input Bundle | What text, if any, the original already carries — informs typography goals and prevents redundant re-statement |
| Creator branding | Style Learning (existing module) | Palette, font families, recurring visual motifs — feeds brand_constraints |
| Historical thumbnails | Style Learning (existing module) | What this channel has actually posted; grounds archetype selection in precedent rather than a generic taxonomy |
| Channel style | Style Learning / Blueprint (existing modules) | Recurring composition and tone patterns specific to this creator |
| Competitor thumbnails | Existing Thumbnail Intelligence / Decision Engine modules, extended | Differentiation signal — informs CTR hypotheses and what visual hierarchy will actually stand out in this niche's feed |
| Thumbnail archetype library | New — a curated, versioned taxonomy (e.g. "reaction/shock," "before-after," "expert-authority," "curiosity-gap," "tutorial-result") | Constrains archetype selection to a known, testable vocabulary instead of open-ended free text |

### 3.4 Output: the DesignBrief

The DesignBrief is the only creative input the Execution Planner is allowed to consume. It is deliberately more expressive and more opinionated than the old EditPlan was — because it's the last stage where creative judgment is exercised at all. Everything after it is translation and execution.

```json
{
  "brief_id": "brief_0091",
  "scene_ref": "scene_0091",
  "archetype": "curiosity_gap",
  "narrative": {
    "primary_story": "Creator attempts a claim from the video and the thumbnail should tease the outcome without revealing it.",
    "target_emotion": "surprise",
    "secondary_emotion": "curiosity"
  },
  "visual_hierarchy": [
    {"order": 1, "element": "creator_0", "reason": "expression carries the emotional read"},
    {"order": 2, "element": "typography.headline", "reason": "reinforces the curiosity gap in 2-3 words"},
    {"order": 3, "element": "product_0", "reason": "grounds the claim in something concrete"}
  ],
  "brand_constraints": {
    "palette_ref": "creator_brand_palette",
    "font_ref": "creator_brand_font",
    "logo_placement": "bottom_right",
    "tone": "high-energy, not corporate"
  },
  "identity_constraints": {
    "locked_instances": ["creator_0", "logo_0"],
    "pose_change_allowed": false
  },
  "ctr_hypotheses": [
    "Original thumbnail's flat lighting reads as low-effort against this niche's competitor set; correcting lighting alone should improve perceived production value.",
    "Original text restates the title verbatim; replacing it with a shorter curiosity-gap phrase should reduce redundancy and increase click intent."
  ],
  "composition_goals": {
    "focal_point": "creator_0.face",
    "safe_zones": ["top_left_text_block"],
    "depth_treatment": "shallow, subject-forward"
  },
  "lighting_goals": {
    "mood": "high-key, punchy",
    "direction_hint": "top_left key, soft rim"
  },
  "typography_goals": {
    "max_word_count": 3,
    "role": "reinforce, not restate, the title",
    "placement_constraint": "must not overlap creator_0 face region"
  },
  "background_goals": {
    "style_direction": "modern neon studio",
    "must_not_compete_with_subject": true
  },
  "rendering_constraints": {
    "no_pose_change": true,
    "no_new_props": true
  },
  "success_metrics": {
    "min_identity_similarity": 0.90,
    "min_composition_preservation": 0.85,
    "min_brand_preservation": 0.90,
    "min_readability_score": 0.80
  },
  "risk_analysis": [
    {"risk": "curiosity-gap text can read as clickbait if overstated", "mitigation": "typography_goals caps word count and requires transcript-grounded phrasing"}
  ]
}
```

Note what's *not* in this document: no font file paths, no diffusion model names, no per-pixel parameters, no engine selection. Those belong to the Execution Planner (§4), which owns the "how." The DesignBrief owns the "what" and "why," and every field in it should be answerable by a human creative director without knowing anything about the rendering stack.

### 3.5 Internal reasoning

The Intelligence Engine is implemented as a local VLM/LLM (the same model family as v1's planner is a reasonable starting point — Qwen2-VL / InternVL-class, local, 4-bit) reasoning over a substantially larger context than v1's planner had: title, transcript (or a summarized/chunked version of it for long videos), OCR, the scene graph, and retrieved historical/competitor thumbnails. The historical and competitor retrieval is a lightweight, CPU-side embedding-similarity lookup against a vector store your existing Style Learning module should maintain — this is not a GPU cost and should not be budgeted against VRAM.

Two design choices matter here:

- **Grounding before generation.** The prompt structure should force the model to cite which input (title, transcript, OCR, or history) supports each major field before it commits to a value — a "grounding pass" ahead of the "commit to brief" pass, even if only the final structured output is kept. This is the direct implementation of §0's "interpretation, not invention" principle, and it's the single biggest lever against the failure mode in §3.6.
- **Archetype-constrained, not archetype-free.** The `archetype` field is selected from the versioned taxonomy in §3.3, not generated as free text. This keeps the Execution Planner's downstream logic (and the Quality Engine's later evaluation) able to reason about "what does a curiosity-gap thumbnail's typography usually need" as a lookup against known patterns, rather than needing to parse an arbitrary description every time.

### 3.6 Failure modes

- **Ungrounded narrative:** the brief invents a story or emotional hook not actually supported by the transcript or title. Mitigated by the grounding-pass prompt structure above; caught downstream by the Quality Engine's readability/brand checks only indirectly, so this is worth its own lightweight automated check — a citation-coverage check that every narrative claim in the brief traces to a specific input span — before the brief is accepted.
- **Archetype misclassification:** picking the wrong structural pattern for the content (e.g. treating a tutorial as a reaction video). Mitigated by keeping the taxonomy small and well-differentiated rather than large and overlapping, and by feeding historical thumbnails from the *same channel* as a strong prior — most channels are archetype-consistent.
- **Brand constraint drift:** the brief's stylistic goals quietly contradict the creator's actual brand history (e.g. proposing a palette the channel has never used). Mitigated by sourcing `brand_constraints` directly from the Style Learning module's extracted signals rather than letting the model infer brand from general impressions of the scene graph.
- **Over-specification:** a brief so prescriptive that the Execution Planner and Layer Engines have no room to make sound mechanical decisions (e.g. dictating an exact hex color instead of a palette reference). Mitigated by keeping the DesignBrief's vocabulary intentionally at the *goal* level (§3.4's `*_goals` fields), not the *parameter* level — that boundary is what keeps the Execution Planner's job meaningfully different from the Intelligence Engine's.
- **Latency/cost creep from transcript length:** long-form video transcripts can be large; naive full-context reasoning over them doesn't scale well on an 8GB card's practical context-length ceiling. Mitigated by a chunked-summarization pre-pass (cheap, can run on the same local LLM or a smaller one) that produces a condensed transcript digest before the main reasoning pass, rather than feeding raw transcripts directly.

### 3.7 Extensibility

The archetype taxonomy, the competitor/historical retrieval store, and the grounding-pass prompt structure are all designed to be extended independently of the rest of the pipeline: adding a new archetype, improving retrieval quality, or swapping the underlying reasoning model are all changes contained entirely within this subsystem, because nothing downstream depends on *how* a DesignBrief was produced — only on it conforming to the schema in §3.4. This mirrors the same swappability discipline already applied to model choices in the Layer Engines (§4 of the original spec) and the Scene Decomposer's component table (§2).

### 3.8 Interaction with other subsystems

- **Style Learning:** primary upstream dependency. The Intelligence Engine should never infer brand/palette/font facts on its own when Style Learning has already extracted them — it consumes those as structured facts, not raw pixels.
- **Scene Graph:** read-only input. The Intelligence Engine reasons about what's already been decomposed; it has no path to request re-decomposition or modify the graph.
- **Execution Planner:** sole downstream consumer. The relationship is strictly one-directional and one-shot per render — the Execution Planner does not send requests back upstream to the Intelligence Engine mid-render (see §3.9 below on why the critique loop doesn't either).
- **Quality Engine:** indirect relationship only. The Quality Engine scores the *rendered output* against the DesignBrief's `success_metrics`, but it does not communicate directly with the Intelligence Engine — see §3.9.

### 3.9 Why the critique loop targets the Execution Planner, not this engine

This is worth stating explicitly because it's a deliberate constraint, not an oversight. The DesignBrief is meant to be a stable creative artifact for the lifetime of one render job — re-running the Intelligence Engine on every failed quality check would mean the *creative direction itself* could drift between iterations, which defeats the purpose of the Quality Engine's targeted-critique loop (§6): that loop exists to make *mechanical* corrections ("relight strength was too aggressive," "text overlapped the safe zone") converge quickly, not to re-litigate what the thumbnail is supposed to be about every time a score comes in low. If a render repeatedly fails quality checks in a way that looks like a brief problem rather than an execution problem (e.g. `success_metrics` were unreachable given what's actually in the scene graph), that's a signal for human review or a new top-level render attempt with a fresh Intelligence Engine pass — not for silently mutating the brief inside the automated loop.

---

## 4. Execution Planner — ExecutionPlan v2 schema

The Edit Planner from v1 evolves into the **Execution Planner**. Its responsibility has narrowed, deliberately: it no longer makes creative decisions — those now live entirely in the DesignBrief (§3.4). Its job is a pure translation:

```
SceneGraph + DesignBrief → ExecutionPlan
```

Where the DesignBrief speaks in goals ("high-key, punchy" lighting mood; "modern neon studio" background style direction), the ExecutionPlan speaks in engine parameters (which relighting engine, what key-light strength, which inpainting model, what LUT). The schema itself is structurally similar to v1's EditPlan — the same per-layer parameter shape still applies to the same Layer Engines — but every value in it should now be traceable to a specific DesignBrief field rather than invented at planning time.

```json
{
  "scene_ref": "scene_0091",
  "brief_ref": "brief_0091",
  "locked_instances": ["creator_0", "logo_0"],
  "background": {
    "action": "replace",
    "generator": "flux_fill",
    "target_style": {
      "prompt": "modern neon studio, soft rim light, shallow depth",
      "palette_ref": "creator_brand_palette",
      "depth_target": "shallow"
    },
    "sourced_from": "background_goals"
  },
  "relight": {
    "instance": "creator_0",
    "engine": "ic_light_v2",
    "key_light": {"direction": "top_left", "strength": 0.7, "color_temp": 5600},
    "rim_light": {"enabled": true, "strength": 0.8},
    "preserve_identity": true,
    "sourced_from": "lighting_goals"
  },
  "shadow_sync": {
    "enabled": true,
    "cast_by": ["creator_0"],
    "light_source": "relight.key_light"
  },
  "typography": [
    {
      "text": "I TRIED THIS",
      "engine": "procedural",
      "role": "headline",
      "position": {"anchor": "top_left", "safe_margin_pct": 6},
      "style_ref": "creator_brand_font",
      "effects": ["drop_shadow", "outline"],
      "max_lines": 2,
      "contrast_target_instance": "background",
      "sourced_from": "typography_goals"
    }
  ],
  "graphics": [
    {"type": "arrow", "from": "text_0", "to": "product_0", "style": "bold_red_curved"}
  ],
  "color_grade": {"lut": "warm_punchy", "saturation_boost": 0.15},
  "effects": ["vignette_soft", "grain_subtle"],
  "quality_targets": {
    "min_identity_similarity": 0.90,
    "min_composition_preservation": 0.85,
    "min_brand_preservation": 0.90,
    "min_readability_score": 0.80
  }
}
```

What changed from v1's EditPlan, and why:

- **`brief_ref`** — every ExecutionPlan is traceable to exactly one DesignBrief. This is what makes the §3.9 boundary enforceable in practice, not just in principle: an ExecutionPlan without a valid `brief_ref` is a bug, not an edge case.
- **`sourced_from` annotations** on the major creative-adjacent fields — a lightweight but deliberate audit trail tying each execution parameter back to the DesignBrief goal that motivated it. This is what makes the Execution Planner's "translation only, no invention" constraint checkable rather than aspirational: a code review or automated test can assert that `relight.key_light.direction` traces back to `lighting_goals.direction_hint`, not to a value the planner introduced unprompted.
- **`quality_targets`** now sourced directly from the DesignBrief's `success_metrics` (§3.4) rather than being decided at planning time — the Execution Planner copies these through, it doesn't set them, since "what counts as good enough for this thumbnail" is a creative judgment that belongs upstream.
- Everything else — `locked_instances`, the per-layer parameter shapes, `shadow_sync`, `graphics`, `color_grade`, `effects` — is structurally unchanged from v1, because the Layer Engines that consume them are unchanged.

**Planner model:** unchanged from v1 — a local VLM (Qwen2-VL / InternVL, 4-bit, ~5-6GB) is still the right size class for this job, and arguably now an easier job than v1's, since it's no longer also responsible for creative reasoning. Load, translate, unload, same as before.

---

## 5. Layer Engines

Unchanged from v1. Included here for completeness since the pipeline around it changed.

### 5.1 Locked layers (creator, hands, product, logo)

Never regenerated. The matte from §2 is composited back in at full resolution after every other layer is done, *except* where `relight` targets that instance — and even then, IC-Light-style relighting models take an image and a target lighting condition and re-render *illumination only*, not identity, structure, or texture, which is exactly the constrained edit you want. This is the correct tool for "make the lighting match the new background" without the identity drift that a full img2img pass would cause.

### 5.2 Background — inpainting/outpainting, not restyling

Recommended: **FLUX.1-Fill-dev** (or BrushNet on SDXL if VRAM-constrained) for the actual background synthesis, run against the *inverse* of the locked-instance mask. FLUX-Fill at fp8/GGUF-Q8 fits in ~8-10GB; on an 8GB card you'll want the Q4/Q5 GGUF quantization or SDXL+BrushNet instead, which comfortably fits at fp16 in 6-7GB. Given your hardware ceiling, **default to SDXL+BrushNet or IC-Light's SDXL base, not FLUX**, and treat FLUX support as an optional high-VRAM path rather than the baseline.

### 5.3 Relighting — IC-Light V2

IC-Light (and V2) is purpose-built for exactly this: relight a foreground subject to match a new background/lighting condition while preserving identity, because it conditions on the *existing* subject rather than generating one. This directly replaces any need for InstantID/PuLID in the common case. Reserve InstantID/PuLID for the narrower case where the plan calls for a full pose change — that's a real generative identity problem and does need those adapters, but it should be the exception path, not the default.

### 5.4 Procedural layers — the quality lever most tools skip

- **Typography:** Pillow + HarfBuzz (via `uharfbuzz`) for proper kerning/shaping, a curated font library, drop shadow/outline/gradient-fill rendering, and automatic contrast-aware placement using the depth map + a saliency map.
- **Arrows/icons/badges:** an SVG asset library composited with procedural color-matching to the palette, not diffusion-generated per request.
- **Glow/particles/vignette/grain/lens-flare:** classical compositing — radial gradients, blend modes, procedural particle placement — all trivial CPU/OpenCV work, zero VRAM.

This tier remains the single biggest differentiator for "looks professionally designed, not AI generated," because it's the tier with zero stochastic artifacts.

### 5.5 Color grading

LUT-based (3D LUT application via OpenCV or `colour`) plus curve/vibrance/contrast adjustment — zero diffusion, sourced from Style Learning's extracted palette/grade signatures.

---

## 6. Compositor

Unchanged from v1. Depth-ordered alpha composite of: background → shadows → locked/relit instances → graphics/typography → color grade → effects. Deliberately kept deterministic and non-generative: this stage should never introduce new artifacts, only assemble ones already validated.

---

## 7. Quality Engine + Targeted Critique Loop

| Score | Method |
|---|---|
| Identity similarity | InsightFace embedding cosine similarity, original crop vs. final crop of `creator_0` |
| Composition preservation | IoU / centroid drift of locked-instance bounding boxes, original vs. final |
| Brand/logo preservation | Template match + embedding similarity on locked logo instance |
| Lighting consistency | Estimated light direction (from shading gradients) compared across subject and background regions |
| Readability | OCR confidence + contrast ratio (WCAG-style) of rendered text against its local background |
| Aesthetic/CTR proxy | A small local aesthetic scorer, trained/calibrated on your own historical thumbnail-CTR data if you have it |

**Loop:** Plan → Execute → Score, evaluated against the `quality_targets` carried through from the DesignBrief's `success_metrics` (§3.4, §4). If any score is below threshold, the Quality Engine emits a **CritiqueReport** (§9.5) — a specific failing dimension plus a targeted correction hypothesis ("identity_similarity 0.82 < 0.90, relight strength likely too aggressive") — and sends it to the **Execution Planner**, which produces a revised ExecutionPlan against the *same* DesignBrief. This makes each iteration a directed mechanical correction, not a creative do-over, per the reasoning in §3.9. Cap iterations at N=4; a thumbnail that still fails after 4 targeted corrections flags for human review rather than looping indefinitely or falling back to a fresh Intelligence Engine pass.

---

## 8. VRAM budget reality check (RTX 4060 Laptop, 8GB)

Nothing below needs to be resident simultaneously — this remains a **sequential pipeline with model swapping**:

| Stage | Peak VRAM | Notes |
|---|---|---|
| Scene Decomposer (SAM2+GroundingDINO+matting+depth) | ~3-4GB | Sequential sub-loads, fp16 |
| **Thumbnail Intelligence Engine (VLM/LLM, 4-bit, 7-8B)** | **~5-7GB** | **New in v2. Context includes title/transcript digest/OCR/scene graph/retrieved history — larger context than the planner alone, but same model size class. Unload before Execution Planner stage; never concurrent with it.** |
| Execution Planner VLM (4-bit, 7-8B) | ~5-6GB | Unchanged from v1. Can reuse the same loaded weights as the Intelligence Engine if both are the same base model — worth exploiting in implementation to avoid a redundant load/unload cycle, though the two must still run as logically separate reasoning passes with separate context windows |
| Background inpaint (SDXL+BrushNet) | ~6-7GB | fp16, tiled if needed |
| Relight (IC-Light, SDXL-based) | ~5-6GB | |
| Identity-preserving regen (InstantID/PuLID, exception path only) | ~7-8GB | Rare path, budget it as the ceiling case |
| Retrieval (historical/competitor thumbnail embedding lookup) | negligible, CPU-side | Vector similarity search, not a GPU cost |
| Procedural layers + compositor + quality scorers | <2GB | CPU-heavy, GPU-light |

The one implementation note worth flagging: if the Intelligence Engine and Execution Planner share a base model (likely, since both are reasoning tasks of similar complexity on similar input types), the practical VRAM cost of adding §3 to the pipeline may be close to zero beyond a longer context window and a second inference pass — not a second model's worth of weights. This should be validated empirically once Phase 3 (§11) is underway rather than assumed.

---

## 9. Data Contracts

Every artifact exchanged between phases is formalized here with its owner, producer, consumer, and lifecycle. This section is new in v2 — v1 defined its one contract (EditPlan) inline in §3; as the pipeline has grown to six inter-stage artifacts, they're worth documenting as a set.

### 9.1 SceneGraph

- **Producer:** Scene Decomposer (§2)
- **Consumers:** Thumbnail Intelligence Engine (§3, read-only), Execution Planner (§4), Quality Engine (§7, for composition-preservation scoring against the original)
- **Owner:** Scene Decomposer
- **Lifecycle:** created once per render job, immutable thereafter. Never mutated by any downstream subsystem.

### 9.2 DesignBrief

- **Producer:** Thumbnail Intelligence Engine (§3)
- **Consumers:** Execution Planner (§4, sole consumer)
- **Owner:** Thumbnail Intelligence Engine
- **Lifecycle:** created once per render job, immutable for the lifetime of that job's critique loop (§3.9, §7). A new render attempt (as opposed to a critique-loop iteration) may produce a new DesignBrief; a critique-loop iteration never does.

### 9.3 ExecutionPlan

- **Producer:** Execution Planner (§4)
- **Consumers:** Layer Engines (§5)
- **Owner:** Execution Planner
- **Lifecycle:** created on the initial planning pass and re-created (against the same DesignBrief and CritiqueReport) on every critique-loop iteration. Versioned per iteration so a failed render's history is inspectable.

### 9.4 RenderWorkspace

- **Producer:** initialized by the Execution Planner at the start of a render job; populated incrementally by each Layer Engine and the Compositor
- **Consumers:** Compositor (§6, reads all layers), Quality Engine (§7, reads the final composite plus intermediate layers for diagnosis)
- **Owner:** shared across Layer Engines and Compositor for the duration of one render job — this is the working state that makes it possible for the Quality Engine's CritiqueReport to point at a specific layer ("the relight layer, not the background layer") rather than only the flattened output
- **Lifecycle:** created per render job, updated in place across critique-loop iterations (so re-execution can selectively re-run only the layers a CritiqueReport implicates, rather than the whole plan, once that optimization is worth building), discarded or archived after export or after the iteration cap is hit

### 9.5 QualityReport

- **Producer:** Quality Engine (§7)
- **Consumers:** Exporter (§1, gates export on threshold), CritiqueReport generation (§9.6)
- **Owner:** Quality Engine
- **Lifecycle:** one per render attempt/iteration, retained for audit and for training future aesthetic/CTR scorers (§8 of v1's self-critique still applies)

### 9.6 CritiqueReport

- **Producer:** Quality Engine (§7), derived from a QualityReport that failed one or more thresholds
- **Consumers:** Execution Planner (§4) only — never the Thumbnail Intelligence Engine (§3.9)
- **Owner:** Quality Engine
- **Lifecycle:** one per failed iteration; consumed immediately by the next Execution Planner pass and retained alongside its QualityReport for audit

---

## 10. What NOT to build (self-critique)

- **Don't build a general-purpose "edit anything" diffusion inpainter as the typography/graphics solution.** AnyText and TextDiffuser are real, respectable research, but they're solving "make a diffusion model spell," which is the wrong problem when Pillow already spells perfectly. Skip them.
- **Don't default to FLUX for background generation on an 8GB card.** Build against SDXL+BrushNet as the baseline and treat FLUX as an optional high-VRAM path.
- **Don't use InstantID/PuLID as the default identity strategy.** They're the right tool only for the pose-change exception path.
- **Don't let the Quality Engine's aesthetic/CTR score be a generic imported model long-term.** Worth investing in a small custom head trained on your own data once you have export volume.
- **Watch licensing on the SVG asset library and fonts.** Every font and vector asset needs a clear commercial license.
- **Don't let the Intelligence Engine and Execution Planner blur back into one subsystem for the sake of implementation convenience.** It will be tempting, once both are local VLMs of similar size, to merge them into a single prompt/pass "since it's the same model anyway." Resist this — the value of the split isn't the model boundary, it's the *reasoning* boundary (§3.1): one pass answers "what should this be," the other answers "how do I make that happen with the engines I have." Collapsing them silently reintroduces v1's original problem of one reasoning pass doing two jobs, even if it happens to run on the same weights.
- **Don't let the critique loop mutate the DesignBrief.** Covered in depth in §3.9, but worth repeating here as a standing constraint: a CritiqueReport is scoped to execution-level corrections. If a render genuinely needs a new creative direction, that's a new top-level render attempt with a fresh Intelligence Engine pass, not a loop iteration.
- **Don't let historical/competitor retrieval become a black box the Intelligence Engine can't explain its use of.** The grounding-pass requirement in §3.5 exists specifically so retrieved evidence shows up traceably in the DesignBrief's `ctr_hypotheses` and `risk_analysis`, not just as an unexplained influence on archetype selection.

---

## 11. Phased roadmap

| Phase | Deliverable | Status |
|---|---|---|
| **1. Proof of Concept** | Scene Decomposer + naive background inpaint (SDXL+BrushNet) + straight recomposite, no relight/typography/quality loop | **Implemented** |
| **2. Planner (v1)** | VLM-based Edit Planner producing EditPlan JSON directly from SceneGraph + Style Learning signals | **Implemented** — superseded in role by Phase 3+4 below; the same underlying model and JSON-emission approach carries forward into the Execution Planner (§4), which now consumes a DesignBrief instead of reasoning from scratch |
| **3. Thumbnail Intelligence Engine** | New subsystem (§3): Input Bundle ingestion (title/transcript/OCR), archetype taxonomy, historical/competitor retrieval, grounding-pass reasoning, DesignBrief emission | Next up |
| **4. Execution Planner hardening** | Narrow the existing Phase 2 planner's scope to pure DesignBrief→ExecutionPlan translation (§4); add `sourced_from` traceability; wire `quality_targets` pass-through from `success_metrics` | Follows Phase 3 |
| **5. Identity/Locked-Layer Engine** | Matting refinement, locked-instance compositing pipeline hardened, exception-path InstantID/PuLID wired in for pose-change requests only | Planned |
| **6. Background Engine** | Full ExecutionPlan-driven background generation, palette/style conditioning from Style Learning | Planned |
| **7. Relighting Engine** | IC-Light V2 integration, shadow_sync, light-direction consistency scoring | Planned |
| **8. Typography + Graphics Engine** | Procedural text/arrow/icon/badge system, SVG asset library, contrast-aware placement | Planned — this is where "looks professional" first becomes true |
| **9. Quality Engine + Targeted Critique Loop** | All scorers, CritiqueReport generation, Execution-Planner-targeted feedback loop, threshold gating | Planned |
| **10. Integration** | Wire into existing CSV/metadata/OCR/email pipeline, batch mode, export presets per channel (email/DM/portfolio) | Planned |

The renumbering here reflects a genuine shift, not just cosmetics: **Renderer V2 now evolves intelligence-first rather than renderer-first.** Phases 1 and 2 proved the mechanical pipeline works end to end. Phases 3 and 4 are where the system stops being "a planner that both invents and executes a creative idea" and becomes "a reasoning stage that decides, and an execution stage that builds" — which is the precondition for every subsequent phase (5 through 9) to be evaluated against a stable, explainable creative brief instead of an opaque one-shot plan. Each phase still ends with something you can point at a real thumbnail and get a real output — that discipline from v1's roadmap carries forward unchanged.
