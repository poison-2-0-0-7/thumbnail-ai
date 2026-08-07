# Renderer V2 Phase 2 Spec — Edit Planner Architecture

### Local, Deterministic Intelligence Layer for YouTube Thumbnail Redesign
**Platform:** Windows / Python 3.13 / RTX 4060 (8GB VRAM) / Commercial Use  
**Status:** Production Implementation Complete

---

## 1. Executive Summary & Purpose

The **Edit Planner** is the central intelligence layer of Renderer V2. Positioned strictly between **Phase 1 (Scene Decomposition)** and downstream rendering execution, the Edit Planner decides **WHAT** must be modified to produce high-CTR, visually dominant YouTube thumbnails that outperform the original while preserving creator identity, key products, and branding.

### Core Architectural Axiom
> **The Edit Planner is a deterministic reasoning layer. It does NOT render pixels, does NOT generate images, and does NOT pass locked subjects through diffusion models.**

Given identical input scenes, the Edit Planner produces bit-for-bit identical, auditable edit plans containing structured geometric, lighting, spatial, and layer modification directives.

---

## 2. End-to-End Pipeline & Data Flow

```
Original Thumbnail
       │
       ▼
┌───────────────────────────────────────────────────────────┐
│ 1. Phase 1: Scene Decomposer & Workspace                   │
│    • Open-vocabulary instance detection & segmentation     │
│    • Soft alpha matting (BiRefNet)                        │
│    • Metric monocular depth estimation (Depth-Anything V2) │
│    • Initial background inpainting & clean workspace      │
└─────────────────────────────┬─────────────────────────────┘
                              │ SceneGraph + DepthMap + Instances
                              ▼
┌───────────────────────────────────────────────────────────┐
│ 2. Phase 2: Edit Planner (INTELLIGENCE LAYER)             │
│    • Multi-scale spectral & contrast saliency estimation   │
│    • Rule-of-thirds & negative space geometric analysis    │
│    • Objective 10-dimension scoring (0-100)               │
│    • Deterministic rule engine (object & layout decisions) │
│    • Output: Structured Deterministic Edit Plan JSON       │
└─────────────────────────────┬─────────────────────────────┘
                              │ Structured EditPlan JSON
                              ▼
┌───────────────────────────────────────────────────────────┐
│ 3. Phase 3: Identity Engine                               │
│    • Locked raster layer protection                        │
│    • Exception-path pose changes & face embedding gating  │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ 4. Phase 4+: Layer Engines & Compositor                   │
│    • Depth-aware relighting (IC-Light V2)                  │
│    • Background synthesis / stylized inpainting            │
│    • Procedural typography & vector graphics              │
│    • Quality validation & iterative loop                  │
└───────────────────────────────────────────────────────────┘
```

---

## 3. The 11 Deterministic Object Decisions

For every detected entity, region, and background in the scene graph, the Edit Planner assigns an explicit deterministic action with a clear, auditable reason:

| Action | Meaning & Execution Context | Example Reason |
|---|---|---|
| `KEEP` | Lock instance pixels; composite directly back at full resolution without diffusion drift. | *"Preserve creator identity and facial features as locked raster layer without generative diffusion drift"* |
| `REMOVE` | Inpaint and eliminate unwanted background clutter, stray people, or competing distractions. | *"Remove distracting secondary clutter to clarify focus hierarchy and open text safe zones"* |
| `MOVE` | Reposition instance to rule-of-thirds power points or away from safe zone hazards. | *"Relocate brand logo from bottom-right to top-left to avoid occlusion by YouTube video duration badge"* |
| `RESIZE` | Scale subject up or down to achieve optimal thumbnail canvas area (30–45%). | *"Scale creator by 1.25x to achieve dominant thumbnail subject prominence (target 30-40% canvas)"* |
| `RECOLOR` | Adjust chromatic tone or align instance colors with target brand palette. | *"Adjust brand accent color to match high-CTR complementary palette"* |
| `RELIGHT` | Direct key and rim lighting adjustment on the subject to match background illumination. | *"Relight foreground creator with directional key and rim lighting for high-contrast subject separation"* |
| `REPLACE` | Replace cluttered, low-contrast, or drab background with high-depth stylized backdrop. | *"Replace low-contrast, cluttered background with a depth-layered stylized studio backdrop to boost subject contrast"* |
| `REGENERATE`| Exception-path generative reconstruction (only for damaged or requested pose alterations). | *"Regenerate occluded subject arm via conditioned inpainting"* |
| `BLUR` | Apply depth-of-field Gaussian blur to background to enhance foreground pop. | *"Apply subtle depth-of-field Gaussian blur to background to enhance focal separation"* |
| `DESATURATE` | Reduce background chroma to increase relative vibrancy of foreground subject. | *"Desaturate background by 15% to increase chromatic contrast against vibrant foreground subject"* |
| `ENHANCE` | Apply micro-contrast, high-pass sharpening, or specular highlight boost. | *"Apply micro-contrast and edge clarity enhancement to creator eyes and facial expression"* |

---

## 4. Overall Composition Analysis & Directives

The planner evaluates and outputs structured parameters for every dimension of thumbnail composition:

1. **Subject Scale**: Evaluates foreground subject area ratio against the total 1280x720 canvas (optimal: 28% to 45%).
2. **Subject Position**: Normalizes centroid coordinates $(x, y) \in [0.0, 1.0]$.
3. **Rule of Thirds**: Measures Euclidean distance to normalized power points $(1/3, 1/3)$, $(2/3, 1/3)$, $(1/3, 2/3)$, $(2/3, 2/3)$ and scores alignment $[0.0, 1.0]$.
4. **Negative Space**: Evaluates fraction of canvas free from subject occupancy and high saliency for headline copy.
5. **Text Safe Zones**: Identifies candidate uncluttered bounding boxes $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ avoiding subjects and the bottom-right YouTube duration badge ($x > 75\%, y > 78\%$).
6. **Focus Hierarchy & Clarity**: Measures visual weight separation between primary focal elements and background.
7. **Contrast Ratio**: Computes WCAG relative luminance contrast ratio $(L_1 + 0.05) / (L_2 + 0.05)$ between foreground and local background.
8. **Visual Balance**: Evaluates horizontal and vertical visual mass distribution.
9. **Attention Direction**: Computes vector flow from focal point to safe zones (e.g., `"left_to_right"`, `"center_outward"`).
10. **Color Harmony**: Classifies palette into `"complementary"`, `"split_complementary"`, `"triadic"`, `"analogous"`, or `"monochromatic"`.
11. **CTR Improvement Potential**: Quantifies optimization headroom across subject scale, contrast, and layout.

---

## 5. Objective Scoring Framework (0-100)

Every dimension is objectively calculated using deterministic mathematical formulas:

$$\text{Overall Score} = \sum_{i=1}^{10} w_i \cdot S_i$$

```
┌───────────────────────────┬────────┬────────────────────────────────────────────────────────┐
│ Dimension                 │ Weight │ Objective Evaluation Method                            │
├───────────────────────────┼────────┼────────────────────────────────────────────────────────┤
│ 1. Composition            │  0.20  │ 40% Rule of Thirds + 35% Balance + 25% Scale Optimality│
│ 2. Contrast               │  0.15  │ WCAG Luminance Contrast Ratio against 4.5:1 AA target  │
│ 3. Subject Prominence     │  0.15  │ 50% Canvas Area Ratio + 50% Saliency Dominance         │
│ 4. Readability            │  0.10  │ 60% Safe Zone Presence + 40% Negative Space Fraction   │
│ 5. Visual Clutter         │  0.10  │ Inverse of Background Laplacian Edge Energy & Entropy  │
│ 6. Background Quality     │  0.10  │ 60% Cleanliness + 40% Depth Map Layering Variance      │
│ 7. Identity Preservation  │  0.10  │ 100.0 if locked raster layers are untouched            │
│ 8. Text Placement         │  0.05  │ Normalized Area & Aspect of Primary Safe Zone          │
│ 9. Depth Usage            │  0.05  │ Monocular Depth Variance between FG and BG planes      │
│ 10. Focus Hierarchy       │  0.05  │ Saliency Differential between Subject and Secondary    │
└───────────────────────────┴────────┴────────────────────────────────────────────────────────┘
```

---

## 6. Output JSON Schema & Example

### 6.1 JSON Contract

```json
{
  "summary": "Thumbnail Optimization Plan (Baseline Score: 78.4/100 -> Target: 92.5/100)...",
  "composition_score": 78.4,
  "target_composition_score": 92.5,
  "changes": [
    {
      "target": "creator_0",
      "action": "keep",
      "reason": "Preserve creator identity and facial features as locked raster layer without generative diffusion drift",
      "target_category": "creator_face",
      "parameters": {
        "locked": true,
        "instance_id": "creator_0",
        "bbox": [720, 140, 1160, 620]
      },
      "confidence": 1.0,
      "priority": 1
    },
    {
      "target": "creator_0",
      "action": "relight",
      "reason": "Relight foreground creator with directional key and rim lighting for high-contrast subject separation",
      "target_category": "creator_body",
      "parameters": {
        "key_light_direction": "top_left",
        "key_light_angle_deg": 135,
        "rim_light_enabled": true,
        "rim_light_strength": 0.75,
        "color_temp_k": 5600
      },
      "confidence": 0.95,
      "priority": 2
    },
    {
      "target": "background",
      "action": "replace",
      "reason": "Replace low-contrast, cluttered background with a depth-layered stylized studio backdrop to boost subject contrast",
      "target_category": "background",
      "parameters": {
        "depth_style": "shallow_dof",
        "palette_ref": "brand_contrast_palette",
        "lighting_sync": "top_left"
      },
      "confidence": 0.95,
      "priority": 1
    }
  ],
  "scoring_breakdown": {
    "composition": 82.5,
    "contrast": 68.0,
    "subject_prominence": 85.0,
    "readability": 88.0,
    "visual_clutter": 72.0,
    "background_quality": 65.0,
    "identity_preservation": 100.0,
    "text_placement": 90.0,
    "depth_usage": 70.0,
    "focus_hierarchy": 85.0,
    "overall": 78.4
  },
  "composition_analysis": {
    "subject_scale": 0.36,
    "subject_position": [0.73, 0.52],
    "rule_of_thirds_alignment": 0.88,
    "negative_space_ratio": 0.48,
    "text_safe_zone_available": true,
    "text_safe_zones": [[48, 48, 650, 360]],
    "hierarchy_clarity": 0.82,
    "contrast_ratio": 3.8,
    "visual_balance": 0.84,
    "focus_score": 0.89,
    "attention_direction": "left_to_right",
    "color_harmony": "complementary",
    "ctr_improvement_potential": 0.22
  },
  "composition_directives": {
    "target_subject_scale": 0.36,
    "target_subject_position": [0.67, 0.5],
    "rule_of_thirds_target": [0.67, 0.5],
    "recommended_text_zone": [48, 48, 650, 360],
    "depth_layering_order": [
      "background",
      "shadow_sync",
      "locked_instances",
      "typography",
      "graphic_overlays",
      "color_grade"
    ],
    "lighting_direction": "top_left",
    "color_palette_target": ["#FF2E63", "#08D9D6", "#0F172A", "#FFFFFF"],
    "contrast_boost_factor": 1.15
  },
  "locked_instances": ["creator_0"],
  "quality_targets": {
    "min_identity_similarity": 0.9,
    "min_composition_preservation": 0.85,
    "min_brand_preservation": 0.9,
    "min_readability_score": 0.8,
    "target_contrast_ratio": 4.5
  },
  "metadata": {
    "image_dimensions": "1280x720",
    "instance_count": 2,
    "locked_count": 1,
    "archetype": "single_creator_face",
    "channel_id": "creator_studio"
  }
}
```

---

## 7. Folder Layout & Module Architecture

```
renderer_v2/
├── __init__.py
├── phase1/                                # Phase 1 Scene Decomposer & Workspace (Production)
│   ├── config.py
│   ├── schemas.py                         # Instance, SceneGraph, PipelineResult
│   ├── model_registry.py
│   ├── pipeline.py
│   ├── scene_decomposer/
│   ├── inpaint/
│   └── compositor/
├── planning/                              # Phase 2 Edit Planner (Intelligence Layer)
│   ├── __init__.py                        # Clean public exports
│   ├── planner_types.py                   # Pydantic schemas, enums, serialization contracts
│   ├── saliency.py                        # Deterministic spectral & depth saliency engine
│   ├── composition.py                     # Rule-of-thirds, safe zone & spatial geometry engine
│   ├── scoring.py                         # Objective 10-dimension 0-100 scoring engine
│   ├── planner_rules.py                   # Deterministic decision rules for objects & layout
│   └── planner.py                         # Main EditPlanner orchestrator
└── tests/
    ├── phase1/                            # Phase 1 unit & integration test suite
    └── phase2/                            # Phase 2 unit, integration & determinism test suite
        ├── test_planner_types.py
        ├── test_saliency.py
        ├── test_composition.py
        ├── test_scoring.py
        ├── test_planner_rules.py
        ├── test_planner_deterministic.py
        └── test_planner_integration.py
```

---

## 8. Verification & Determinism Proof

1. **Pure Mathematical Determinism**: No random seeds, no diffusion iterations, and no non-deterministic CUDA operations.
2. **Bit-for-Bit Consistency**: Tested across repeated invocations on identical `SceneGraph` inputs in `tests/phase2/test_planner_deterministic.py` (all JSON outputs, scores, and change lists match identically).
3. **Comprehensive Test Suite**: 23 unit and integration tests passing in <2 seconds.
