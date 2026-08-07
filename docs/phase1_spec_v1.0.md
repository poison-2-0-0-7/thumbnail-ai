# Phase 1 Spec — Scene Decomposer + Background Inpaint + Recompositor
### Scope: exactly what the roadmap says and nothing else

Phase 1 per the architecture doc: **Scene Decomposer + naive background inpaint (SDXL+BrushNet) + straight recomposite. No relight, no typography, no quality loop, no EditPlan-driven styling.** This document covers that scope only. No implementation has started — this is the design for approval.

One deviation from the architecture doc is flagged below (§1.1) based on research done for this phase; everything else follows the approved doc as written.

---

## 1. Research & Benchmarking

### 1.1 Detection + Segmentation — flagging a change from the architecture doc

The architecture doc specified GroundingDINO (text→boxes) + SAM2 (boxes→masks) as two separate models. Researching Phase 1 specifically surfaced **SAM3** (Meta, released Nov 2025), which merges both jobs into one model: it takes noun-phrase text prompts directly and returns per-instance masks for every matching object in a single forward pass, with no separate detector stage. Reported footprint is ~3.4GB of weights and sub-4GB VRAM in independent local tests. This is a straightforward win for Phase 1 — it removes an entire model (GroundingDINO), removes the box→mask handoff (a common source of segmentation errors when boxes are loose), and uses less VRAM than the two-model combo it replaces.

**Recommendation: replace GroundingDINO+SAM2 with SAM3 for detection+segmentation.** This is a scope-preserving substitution (same job: text-prompted instance masks), not scope creep, so I'm flagging it rather than silently deviating, and holding off on locking it in until you confirm.

| Option | VRAM | Speed | Notes |
|---|---|---|---|
| **SAM3** (recommended) | ~3.5-4GB | Single pass, no separate detector | Unifies detection+segmentation; text-prompted; newest, so smaller community/tooling base than SAM2 — flagged as the main risk |
| GroundingDINO(tiny) + SAM2.1(base+) | ~4-5GB combined | Two sequential passes | Mature, huge ComfyUI/HF ecosystem, well-documented failure modes | 
| GroundingDINO + MobileSAM | ~2-3GB | Fastest | MobileSAM trades mask quality (mIoU ~0.74 vs SAM's ~0.77+) for speed we don't need in an offline batch pipeline — Phase 1 isn't latency-constrained, so this tradeoff isn't worth taking |
| SAM2.1 alone (visual prompts only) | ~3-8GB depending on checkpoint | Fast | No text prompting — would need a separate open-vocab detector anyway, so this doesn't actually reduce complexity vs. the DINO+SAM2 combo |

Decision needed from you: **SAM3 (new, fewer moving parts) vs. GroundingDINO+SAM2.1 (mature, matches original doc exactly).** My recommendation is SAM3, with GroundingDINO+SAM2.1 kept as a documented fallback interface (see §3 — the interface is model-agnostic specifically so this swap is cheap either way).

### 1.2 Matting refinement

| Option | VRAM | Notes |
|---|---|---|
| **BiRefNet** (recommended) | ~1-1.5GB (fp16, `BiRefNet-lite` variant) | Best published matte quality for hair/fine detail among open local models; `-lite` variant exists specifically for VRAM-constrained cases |
| MODNet | <500MB | Faster, older, noticeably worse on hair/fur edges — acceptable fallback if BiRefNet-lite still doesn't fit alongside other resident models |

Recommendation: **BiRefNet-lite**, MODNet as fallback if VRAM budget gets tight once all Phase 1 models are accounted for.

### 1.3 Depth

**Depth-Anything V2 (small)** — <1GB VRAM, fp16. This is not a contested choice; it's the de facto standard for local monocular depth and there's no serious competitor at this size/quality point right now. Used in Phase 1 only to separate foreground/background for masking, not yet for shadow synthesis (that's Phase 4).

### 1.4 Background inpainting

Per the architecture doc's own reasoning (§4.2 of the architecture doc), FLUX.1-Fill was already ruled out as the Phase 1 default because it doesn't comfortably fit 8GB except at aggressive GGUF quantization. Research for this phase confirms that reasoning still holds — FLUX-Fill quality is higher, but SDXL+BrushNet is the correct fit for the actual hardware constraint, and Phase 1 is explicitly the "prove the pipeline end-to-end" phase, not the "maximize quality" phase.

| Option | VRAM (8GB card) | Notes |
|---|---|---|
| **SDXL + BrushNet** (recommended) | ~6-7GB fp16 | Purpose-built dual-branch inpainting; strong masked-region preservation scores in published comparisons; mature ComfyUI/diffusers support |
| SDXL native inpainting (diffusers `StableDiffusionXLInpaintPipeline`) | ~6-7GB | Simpler integration, but published comparisons consistently show it leaking content across mask boundaries and misreading prompts more often than BrushNet |
| PowerPaint v2 | ~5-6GB (SD1.5-based) | No SDXL version exists, meaning lower base resolution/quality than the SDXL options — not worth the tradeoff here |
| FLUX.1-Fill-dev (GGUF Q4/Q5) | ~7-8.5GB | Best quality per published benchmarks, but leaves almost no VRAM headroom for the rest of the pipeline (decomposer + inpaint can't be concurrently resident, and even sequential swap gets tight) — deferred, matches original doc's call |

**Recommendation: SDXL + BrushNet (`segmentation_mask_brushnet_ckpt_sdxl_v0`)** — confirms the architecture doc's choice, no deviation here.

### 1.5 Summary of Phase 1 model stack

| Stage | Model | VRAM | Resident concurrently with |
|---|---|---|---|
| Detection+segmentation | SAM3 (pending your confirmation; else GroundingDINO+SAM2.1) | ~3.5-5GB | Nothing else — unloaded before next stage |
| Matting | BiRefNet-lite | ~1-1.5GB | Nothing else |
| Depth | Depth-Anything V2 small | <1GB | Nothing else |
| Background inpaint | SDXL + BrushNet | ~6-7GB | Nothing else |

All four stages run **sequentially with model unload/reload between them** — confirmed compatible with the 8GB ceiling with margin in every stage, matching the architecture doc's "sequential pipeline with model swapping" design (§7 of that doc).

---

## 2. Architecture (Phase 1 only)

```
input_image.png
      │
      ▼
┌───────────────────────────┐
│ SceneDecomposer             │
│  - detect+segment (SAM3)    │
│  - matte refine (BiRefNet)  │
│  - depth (Depth-Anything V2)│
└──────────┬─────────────────┘
           │  SceneGraph (instances, mattes, depth map)
           ▼
┌───────────────────────────┐
│ locked-region mask builder  │  union of "creator"/"logo"/"product" instance mattes
└──────────┬─────────────────┘
           │  inverse mask = background region to inpaint
           ▼
┌───────────────────────────┐
│ BackgroundInpainter          │
│  SDXL + BrushNet             │
│  prompt: static/config for   │
│  Phase 1 (no Planner yet)    │
└──────────┬─────────────────┘
           │  inpainted background image
           ▼
┌───────────────────────────┐
│ Recompositor                 │  alpha-composite locked instances back over
│  (straight composite, no    │  inpainted background, feathered edge blend
│   relight/typography/FX)    │
└──────────┬─────────────────┘
           ▼
   output_image.png + debug artifacts (masks, mattes, depth viz)
```

No Planner/EditPlan in Phase 1 — the architecture doc's roadmap explicitly defers EditPlan-driven behavior to Phase 3. The inpainting prompt/style for Phase 1 is a fixed config value (or a small fixed set of styles), not planner-generated. This keeps Phase 1 honestly scoped to "prove decomposition→inpaint→recomposite works end to end," which is what the roadmap calls for.

---

## 3. Folder structure

```
renderer_v2/
├── phase1/
│   ├── __init__.py
│   ├── config.py                    # paths, model IDs, VRAM/device settings, fixed inpaint prompts
│   ├── scene_decomposer/
│   │   ├── __init__.py
│   │   ├── base.py                  # abstract Detector, Matter, DepthEstimator interfaces
│   │   ├── sam3_detector.py         # SAM3 implementation of Detector
│   │   ├── groundingdino_sam2_detector.py  # fallback implementation, same interface
│   │   ├── birefnet_matter.py       # Matter implementation
│   │   ├── depth_anything.py        # DepthEstimator implementation
│   │   └── decomposer.py            # SceneDecomposer orchestrator, sequential load/unload
│   ├── inpaint/
│   │   ├── __init__.py
│   │   ├── base.py                  # abstract BackgroundInpainter interface
│   │   ├── sdxl_brushnet.py         # chosen implementation
│   │   └── mask_utils.py            # locked-region union, inverse mask, dilation/feathering
│   ├── compositor/
│   │   ├── __init__.py
│   │   └── recompositor.py          # straight alpha composite, edge feather
│   ├── pipeline.py                  # wires decomposer → inpaint → recompositor
│   ├── schemas.py                   # dataclasses: Instance, SceneGraph, PipelineResult
│   └── model_registry.py            # load/unload lifecycle management, VRAM guard
├── tests/
│   ├── phase1/
│   │   ├── fixtures/                # small set of real+synthetic sample thumbnails, hand-labeled masks
│   │   ├── test_scene_decomposer.py
│   │   ├── test_inpaint.py
│   │   ├── test_compositor.py
│   │   ├── test_pipeline_integration.py
│   │   └── test_vram_budget.py      # asserts peak VRAM per stage stays under budget
│   └── conftest.py
├── benchmarks/
│   ├── run_benchmark.py             # sweeps model alternatives, logs VRAM/latency/quality proxy
│   ├── results/                     # benchmark run outputs (csv/json), not committed model weights
│   └── report_template.md
├── scripts/
│   └── run_phase1_on_image.py       # CLI entry point for manual testing
└── models_cache/                    # gitignored, local weight cache
```

Rationale: every model-dependent stage (`Detector`, `Matter`, `DepthEstimator`, `BackgroundInpainter`) is defined as an abstract interface with a concrete implementation file, specifically so the SAM3-vs-GroundingDINO+SAM2 decision (and any future swap) is a one-file change, not a pipeline rewrite. `model_registry.py` centralizes load/unload so the sequential-swap VRAM discipline from §1.5 is enforced in one place instead of scattered across stages.

---

## 4. Interfaces

```python
# schemas.py
from dataclasses import dataclass
from typing import Literal
import numpy as np

InstanceClass = Literal["creator", "logo", "product", "other"]

@dataclass
class Instance:
    instance_id: str
    cls: InstanceClass
    mask: np.ndarray          # hard binary mask, HxW
    alpha_matte: np.ndarray   # soft matte, HxW float32 [0,1]
    bbox: tuple[int, int, int, int]
    depth_layer: float        # mean depth value within mask
    locked: bool              # True for creator/logo/product

@dataclass
class SceneGraph:
    source_image: np.ndarray
    instances: list[Instance]
    depth_map: np.ndarray     # HxW float32
    width: int
    height: int

@dataclass
class PipelineResult:
    output_image: np.ndarray
    scene_graph: SceneGraph
    inpainted_background: np.ndarray
    locked_region_mask: np.ndarray
    debug_artifacts: dict[str, np.ndarray]
```

```python
# scene_decomposer/base.py
from abc import ABC, abstractmethod

class Detector(ABC):
    """Text-prompted instance detection+segmentation. One implementation per model choice."""
    @abstractmethod
    def detect(self, image: np.ndarray, class_prompts: list[str]) -> list[Instance]: ...

class Matter(ABC):
    @abstractmethod
    def refine(self, image: np.ndarray, instance: Instance) -> np.ndarray:  # returns alpha matte
        ...

class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray) -> np.ndarray:  # HxW depth map
        ...
```

```python
# inpaint/base.py
class BackgroundInpainter(ABC):
    @abstractmethod
    def inpaint(self, image: np.ndarray, inverse_mask: np.ndarray, prompt: str) -> np.ndarray: ...
```

```python
# pipeline.py
class Phase1Pipeline:
    def __init__(self, detector: Detector, matter: Matter, depth: DepthEstimator,
                 inpainter: BackgroundInpainter, registry: ModelRegistry): ...

    def run(self, image_path: str, class_prompts: list[str], inpaint_prompt: str) -> PipelineResult: ...
```

Every concrete model implementation (`sam3_detector.py`, `sdxl_brushnet.py`, etc.) satisfies these interfaces and nothing else imports the underlying model library directly outside its own file — that boundary is what makes §1.1's decision reversible without touching `pipeline.py` or the tests.

---

## 5. Data flow

1. **Input:** raw thumbnail image (PNG/JPG), a fixed list of class prompts (`["person", "logo", "product"]` for Phase 1 — no free-text planner yet), a fixed inpaint style prompt from config.
2. **Decompose:** `Detector.detect()` → list of `Instance` (mask + bbox per class match) → `Matter.refine()` per instance → soft alpha mattes → `DepthEstimator.estimate()` → depth map → assembled into `SceneGraph`.
3. **Lock:** union all `locked=True` instance mattes into one `locked_region_mask`; invert it (dilated slightly, ~8-16px, to avoid inpaint bleeding right up to the subject edge) → `inverse_mask` passed to inpainting.
4. **Inpaint:** `BackgroundInpainter.inpaint(image, inverse_mask, prompt)` → full-frame background-replaced image (locked regions will look wrong/regenerated inside them at this point — that's expected and discarded next step).
5. **Recomposite:** alpha-composite the *original* locked-instance pixels (via their soft mattes, not the inpainted version) back over the inpainted background, feathered at the matte edge → final output.
6. **Output:** final image + all intermediates (masks, mattes, depth viz, raw inpaint output) written to a debug directory — required for testing (§6), not just convenience.

No feedback loop, no scoring, no iteration in Phase 1 — output is produced in exactly one pass, per scope.

---

## 6. Testing strategy

| Level | What | How |
|---|---|---|
| **Unit — Detector** | Mask IoU against a small (~20-30 image) hand-labeled fixture set covering talking-head, product-shot, and multi-person thumbnails | Fixture masks drawn once, stored in `tests/phase1/fixtures/`; regression-tested on every model swap so §1.1's decision is empirically checked, not just VRAM-reasoned |
| **Unit — Matter** | Alpha matte quality at hair/finger boundaries vs. a small set of matting-benchmark-style images with known ground truth | SAD/MSE against ground-truth alpha where available; visual diff otherwise |
| **Unit — Depth** | Sanity checks only (foreground instances have lower depth-layer value than background regions) | No ground-truth depth needed for Phase 1's actual use (fg/bg separation, not shadow placement yet) |
| **Unit — Inpainter** | Masked-region preservation: unmasked pixels must be ~unchanged (PSNR/LPIPS against input outside the mask) | Catches the "inpainter leaked outside the mask" failure mode specifically flagged in the BrushNet vs. SDXL-native research |
| **Unit — Compositor** | No hard edges/halos at matte boundaries; locked-region pixels in output match locked-region pixels in input exactly (not just "close") | Pixel-diff assertion inside the eroded matte core; feather-only diff allowed at the boundary band |
| **Integration** | End-to-end run on the full fixture set, output inspected against a checklist (identity/logo intact, no visible seam, background actually changed) | Semi-automated: automated checks above + a manual pass/fail checklist per image, logged |
| **VRAM/regression guard** | Peak VRAM per stage stays under the budget in §1.5's table, on the actual RTX 4060 8GB target (or closest available CI GPU) | `torch.cuda.max_memory_allocated()` captured per stage, asserted in `test_vram_budget.py`, tracked over time to catch creep |
| **Golden-output regression** | Once Phase 1 output quality is approved, freeze a small set of golden outputs; future changes to Phase 1 code must not silently change them (SSIM threshold) | Prevents an unrelated refactor from quietly degrading a working pipeline |

Fixture set composition matters more than fixture count here: talking-head close-up, product-in-hand, multi-person, and a logo-heavy thumbnail each stress a different part of the decomposer, so the ~20-30 image set should be deliberately picked to cover those, not randomly sampled.

---

## 7. Benchmark plan

`benchmarks/run_benchmark.py` runs every candidate from §1 (SAM3 vs. GroundingDINO+SAM2.1; BiRefNet vs. MODNet; BrushNet vs. native SDXL inpaint) against the same fixture set and logs:

- Peak VRAM per model (measured, not estimated from docs — the numbers in §1's tables are from published sources and need confirming on the actual target card)
- Wall-clock latency per stage
- Mask IoU / matte SAD against the hand-labeled fixtures
- Masked-region preservation (PSNR/LPIPS outside mask) for inpainting candidates

Output: a CSV/JSON per run plus a filled-in `report_template.md` — this becomes the evidence behind the §1.1 SAM3 decision instead of leaving it as a documentation-only claim. This benchmark should run **before** implementation is finalized on the detector choice specifically, since that's the one open decision in this doc.

---

## 8. Open decision before implementation starts

**SAM3 vs. GroundingDINO+SAM2.1 for detection+segmentation (§1.1).** Everything else in this document follows the architecture doc directly. Recommend confirming this one point, then running the §7 benchmark to validate it empirically, then beginning implementation.
