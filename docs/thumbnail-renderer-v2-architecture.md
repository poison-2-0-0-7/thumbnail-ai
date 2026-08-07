# Renderer V2 — A Reconstruction Engine, Not a Generator
### Architecture for a fully-local, layer-based YouTube thumbnail redesign system (RTX 4060, 8GB VRAM)

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

---

## 1. Pipeline

```
Original Thumbnail
      │
      ▼
┌─────────────────────┐
│ 1. Scene Decomposer  │  SAM2 + GroundingDINO + BiRefNet matting + Depth-Anything V2
│    (segmentation,    │  → per-instance masks, alpha mattes, depth map, camera/pose estimate
│     depth, matting)  │
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ 2. Edit Planner       │  VLM (Qwen2-VL / InternVL local) reasons over the decomposed scene
│    (EditPlan JSON)    │  + your existing Style Learning / Blueprint modules
└─────────┬────────────┘
          ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Layer Engines (each independent, each optional)        │
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
│ 4. Compositor         │  Depth-ordered alpha compositing, edge feathering, harmonization
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ 5. Quality Engine     │  Identity/brand/composition preservation scores + CTR/aesthetic scores
└─────────┬────────────┘
          ▼
     score ≥ threshold? ──No──▶ back to Planner with critique (iterate, capped at N rounds)
          │Yes
          ▼
     Exporter (PNG/JPG, multiple aspect variants, email/DM-ready sizes)
```

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

**Output:** a JSON scene graph — list of instances (class, mask, alpha matte, bbox, depth layer, is_locked) plus a global depth map. This is the substrate every downstream layer reads from. This is also the artifact your existing Scene Understanding / Asset Extraction modules should be upgraded to emit, so you're not duplicating work you already built.

---

## 3. Edit Planner — EditPlan v2 schema

The brief's example schema is a good instinct but too generation-flavored ("replace", "style": free text). A reconstruction system needs a schema that references the scene graph's actual instance IDs and separates "how" from "what," because each layer engine below consumes different parameters.

```json
{
  "scene_ref": "scene_0091",
  "locked_instances": ["creator_0", "logo_0"],
  "background": {
    "action": "replace",
    "generator": "flux_fill",
    "target_style": {
      "prompt": "modern neon studio, soft rim light, shallow depth",
      "palette_ref": "creator_brand_palette",
      "depth_target": "shallow"
    }
  },
  "relight": {
    "instance": "creator_0",
    "engine": "ic_light_v2",
    "key_light": {"direction": "top_left", "strength": 0.7, "color_temp": 5600},
    "rim_light": {"enabled": true, "strength": 0.8},
    "preserve_identity": true
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
      "contrast_target_instance": "background"
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

Key differences from the brief's version, and why:

- **`scene_ref` + `locked_instances`** instead of a bare `"preserve": true` — the planner must be explicit about *which segmented instance* is locked, because downstream engines operate per-instance, not per-concept.
- **Typography is its own top-level array with `engine: "procedural"`**, not a diffusion parameter. This is the change that most affects output quality (see §0).
- **`quality_targets` lives in the plan itself**, not bolted on afterward — so the Quality Engine and the iterative loop (§6) know what "good enough" means for *this specific thumbnail*, not a global constant. A close-up talking-head thumbnail has a much higher identity-similarity bar than a wide product-shot thumbnail.
- **`shadow_sync`** ties shadow generation to the actual chosen light direction, so background and subject don't end up lit from different directions — the single most common tell of "this was pasted together."

**Planner model:** a local VLM (Qwen2-VL-7B or InternVL2-8B, 4-bit) reasoning over the scene graph + your existing Style Learning/Blueprint/Thumbnail Intelligence outputs, fine-tuned or few-shot prompted to emit this JSON. At 4-bit this is ~5-6GB, so it should not be resident at the same time as the generative layer engines — load, plan, unload, then load FLUX-Fill/IC-Light for execution.

---

## 4. Layer Engines

### 4.1 Locked layers (creator, hands, product, logo)

Never regenerated. The matte from §2 is composited back in at full resolution after every other layer is done, *except* where `relight` targets that instance — and even then, IC-Light-style relighting models take an image and a target lighting condition and re-render *illumination only*, not identity, structure, or texture, which is exactly the constrained edit you want. This is the correct tool for "make the lighting match the new background" without the identity drift that a full img2img pass would cause.

### 4.2 Background — inpainting/outpainting, not restyling

Recommended: **FLUX.1-Fill-dev** (or BrushNet on SDXL if VRAM-constrained) for the actual background synthesis, run against the *inverse* of the locked-instance mask. This is a solved, mature problem — the research questions here are less open than the brief implies. FLUX-Fill at fp8/GGUF-Q8 fits in ~8-10GB; on an 8GB card you'll want the Q4/Q5 GGUF quantization or SDXL+BrushNet instead, which comfortably fits at fp16 in 6-7GB. Given your hardware ceiling, **default to SDXL+BrushNet or IC-Light's SDXL base, not FLUX**, and treat FLUX support as a "phase 2, if a user has more VRAM" option rather than the baseline — this is a case where "more well-known" loses to "actually fits your stated hardware."

### 4.3 Relighting — IC-Light V2

IC-Light (and V2) is purpose-built for exactly this: relight a foreground subject to match a new background/lighting condition while preserving identity, because it conditions on the *existing* subject rather than generating one. This directly replaces any need for InstantID/PuLID in the common case — you're not asking a model to *recreate* a face under new lighting, you're asking it to *relight* the actual face pixels. Reserve InstantID/PuLID for the narrower case where the planner decides a full pose change is needed (e.g., turning a straight-on shot into a 3/4 angle) — that's a real generative identity problem and does need those adapters, but it should be the exception path, not the default.

### 4.4 Procedural layers — the quality lever most tools skip

- **Typography:** Pillow + HarfBuzz (via `uharfbuzz`) for proper kerning/shaping, a curated font library (~15-20 bold, high-CTR display fonts covering the styles your Style Learning module has already identified as high-performing), drop shadow/outline/gradient-fill rendering, and automatic contrast-aware placement using the depth map + a saliency map so text never lands on a busy area or the creator's face.
- **Arrows/icons/badges:** an SVG asset library (a few hundred hand-picked, license-clear vector assets covering "circle highlight," "arrow pointing," "X-out," "numbered badge," "before/after split" — the actual recurring visual vocabulary of high-CTR thumbnails) composited with procedural color-matching to the palette, not diffusion-generated per request.
- **Glow/particles/vignette/grain/lens-flare:** classical compositing — radial gradients, blend modes (screen/overlay), procedural particle placement — all trivial CPU/OpenCV work, zero VRAM.

This tier is the one most "AI thumbnail generator" competitors skip because it's less exciting to build than diffusion, and it is the single biggest differentiator for "looks professionally designed, not AI generated," because it's the tier with zero stochastic artifacts.

### 4.5 Color grading

LUT-based (3D LUT application via OpenCV or `colour`) plus curve/vibrance/contrast adjustment — again, zero diffusion. Your Style Learning module should already be extracting palette/grade signatures from high-performing thumbnails; feed that directly into a LUT selection or generation step rather than describing it in a text prompt to a diffusion model.

---

## 5. Compositor

Depth-ordered alpha composite of: background → shadows → locked/relit instances → graphics/typography → color grade → effects. The two things that make composites look fake are (a) edge halos from hard mattes and (b) inconsistent light direction — both are already handled upstream (§2 matting, §4.4 shadow_sync), so the compositor itself is a straightforward, deterministic operation. Deliberately keep it deterministic and non-generative: this stage should never introduce new artifacts, only assemble ones already validated.

---

## 6. Quality Engine + iterative loop

| Score | Method |
|---|---|
| Identity similarity | InsightFace embedding cosine similarity, original crop vs. final crop of `creator_0` |
| Composition preservation | IoU / centroid drift of locked-instance bounding boxes, original vs. final |
| Brand/logo preservation | Template match + embedding similarity on locked logo instance |
| Lighting consistency | Estimated light direction (from shading gradients) compared across subject and background regions |
| Readability | OCR confidence + contrast ratio (WCAG-style) of rendered text against its local background |
| Aesthetic/CTR proxy | A small local aesthetic scorer (e.g., a fine-tuned CLIP-based predictor, or LAION-Aesthetics-style head) trained/calibrated on your own historical thumbnail-CTR data if you have it — this is the one score worth training yourself rather than importing, since generic aesthetic scorers don't know what drives clicks in your niches |

**Loop:** Plan → Execute → Score. If any score is below its `quality_targets` threshold, feed the specific failing dimension back to the Planner as a targeted critique ("identity_similarity 0.82 < 0.90, relight strength likely too aggressive") rather than a generic "try again" — this makes each iteration a directed correction instead of a random resample, and lets you cap iterations at 2-3 in practice instead of needing many. Cap hard at N=4 regardless; a thumbnail that still fails after 4 targeted corrections should flag for human review, not loop indefinitely.

---

## 7. VRAM budget reality check (RTX 4060 Laptop, 8GB)

Nothing above needs to be resident simultaneously — this is a **sequential pipeline with model swapping**, not a single always-loaded stack, and that's the right design for your hardware regardless of what a cloud-scale competitor would do:

| Stage | Peak VRAM | Notes |
|---|---|---|
| Scene Decomposer (SAM2+GroundingDINO+matting+depth) | ~3-4GB | Sequential sub-loads, fp16 |
| Planner VLM (4-bit, 7-8B) | ~5-6GB | Unload before generative stage |
| Background inpaint (SDXL+BrushNet) | ~6-7GB | fp16, tiled if needed |
| Relight (IC-Light, SDXL-based) | ~5-6GB | |
| Identity-preserving regen (InstantID/PuLID, exception path only) | ~7-8GB | Rare path, budget it as the ceiling case |
| Procedural layers + compositor + quality scorers | <2GB | CPU-heavy, GPU-light |

Model swapping costs seconds, not minutes, with weights cached on fast local SSD — accept that latency cost deliberately rather than trying to keep everything resident and fighting OOMs.

---

## 8. What NOT to build (self-critique)

- **Don't build a general-purpose "edit anything" diffusion inpainter as the typography/graphics solution.** AnyText and TextDiffuser are real, respectable research, but they're solving "make a diffusion model spell," which is the wrong problem when Pillow already spells perfectly. Importing them adds VRAM, latency, and a new class of failure (wrong text, bad kerning) for a problem that doesn't need generative modeling. Skip them.
- **Don't default to FLUX for background generation on an 8GB card.** It's the better model on paper; it's the wrong choice under this hardware constraint. Build against SDXL+BrushNet as the baseline and treat FLUX as an optional high-VRAM path.
- **Don't use InstantID/PuLID as the default identity strategy.** They're the right tool only for the pose-change exception path (§4.3). Using them as the default couples every edit to a generative face model you don't actually need most of the time, and reintroduces the identity-drift problem the whole "reconstruction not regeneration" philosophy exists to avoid.
- **Don't let the Quality Engine's aesthetic/CTR score be a generic imported model long-term.** It's fine as a placeholder, but a scorer that doesn't know your actual niches' historical CTR patterns will plateau in usefulness. This is worth investing in a small custom head trained on your own data once you have export volume.
- **Watch licensing on the SVG asset library and fonts.** "Professional, not AI generated, used commercially for client acquisition" means every font and vector asset needs a clear commercial license — this is a real operational risk, not a technical one, and worth an explicit review pass before Phase 1 ships.

---

## 9. Phased roadmap

| Phase | Deliverable | Working renderer at end of phase? |
|---|---|---|
| **1. Proof of Concept** | Scene Decomposer + naive background inpaint (SDXL+BrushNet) + straight recomposite, no relight/typography/quality loop yet | Yes — crude but end-to-end |
| **2. Identity/Locked-Layer Engine** | Matting refinement, locked-instance compositing pipeline hardened, exception-path InstantID/PuLID wired in for pose-change requests only | Yes |
| **3. Background Engine** | Full EditPlan-driven background generation, palette/style conditioning from existing Style Learning module | Yes |
| **4. Relighting Engine** | IC-Light V2 integration, shadow_sync, light-direction consistency scoring | Yes |
| **5. Typography + Graphics Engine** | Procedural text/arrow/icon/badge system, SVG asset library, contrast-aware placement | Yes — this is where "looks professional" first becomes true |
| **6. Quality Engine + Iteration Loop** | All scorers, targeted-critique feedback loop, threshold gating | Yes — first version that self-rejects bad output |
| **7. Integration** | Wire into existing CSV/metadata/OCR/email pipeline, batch mode, export presets per channel (email/DM/portfolio) | Yes — production |

Each phase ends with something you can point at a real thumbnail and get a real (if incomplete) output — no phase is "just infrastructure with nothing to look at."
