# Module 8 — Asset Extraction Engine

**Design document — architecture only. No implementation code.**
**Source of truth verified against:** `poison-2-0-0-7/thumbnail-ai` @ `main` (cloned and reviewed in full prior to writing this document — `modules/`, `docs/`, `tests/`, `config.py`, `models.py`, `vision_stack/`, `vre_components/`, `main.py`, `pytest.ini`).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Findings That Shape This Design](#2-repository-findings-that-shape-this-design)
3. [Design Goals / Non-Goals](#3-design-goals--non-goals)
4. [Pipeline Position & Data Flow](#4-pipeline-position--data-flow)
5. [High-Level Architecture](#5-high-level-architecture)
6. [Folder Structure](#6-folder-structure)
7. [Python Module Layout](#7-python-module-layout)
8. [Data Models](#8-data-models-modelspy-additions)
9. [Public API](#9-public-api)
10. [Internal APIs / Component Contracts](#10-internal-apis--component-contracts)
11. [Responsibilities of Every Class](#11-responsibilities-of-every-class)
12. [Asset Lifecycle](#12-asset-lifecycle)
13. [Vision-Stack Integration (Model Execution)](#13-vision-stack-integration-model-execution)
14. [Outputs](#14-outputs)
15. [Manifest Schema](#15-manifest-schema-asset_manifestjson)
16. [Caching Strategy](#16-caching-strategy)
17. [Error Recovery / Resume-After-Interruption](#17-error-recovery--resume-after-interruption)
18. [Logging Strategy](#18-logging-strategy)
19. [Error Handling / Exception Hierarchy](#19-error-handling--exception-hierarchy)
20. [Configuration (`config.py` additions)](#20-configuration-configpy-additions)
21. [Performance Design (RTX 4060, 16GB RAM)](#21-performance-design-rtx-4060-16gb-ram)
22. [Testing Strategy](#22-testing-strategy)
23. [Integration with Modules 1–7](#23-integration-with-modules-17)
24. [Integration with Modules 9–11](#24-integration-with-modules-911)
25. [Implementation Roadmap — Phases for Autonomous Coding Agents](#25-implementation-roadmap--phases-for-autonomous-coding-agents)
26. [Risks & Open Questions](#26-risks--open-questions)

---

## 1. Overview

Module 8, the **Asset Extraction Engine (AEE)**, is the bridge between *understanding* a thumbnail (Modules 4–5) and *rebuilding* one (Modules 9–11). It takes a source thumbnail plus its already-computed `ThumbnailIntelligence` report and produces a complete, disk-persisted, reusable set of pixel-level and structured assets — crops, masks, embeddings, conditioning maps, and metadata — covering every element named in the brief: people, scene, objects, typography, visual properties, composition, and effects.

Module 8 does **not** decide what to keep, remove, replace, enhance, or add. That is Module 9's job. Module 8's job is to make Module 9's job possible without ever re-opening the source image.

Module 8 supersedes and generalizes Module 6.5 (Visual Reference Engine). VRE proved the pattern — orchestrator + swappable processors + immutable manifest + atomic writes + SHA-256 cache — for one face and one generic foreground object. Module 8 is that same pattern scaled to every extractable asset family, and it deliberately reuses VRE's own components where they already generalize (`AssetWriter`, the atomic-write discipline, the manifest-cache-verification algorithm).

---

## 2. Repository Findings That Shape This Design

These are load-bearing facts discovered while reviewing the repository. Every one of them changed a decision below.

1. **`ThumbnailIntelligence` (Module 4) already computes, per thumbnail:** OCR `TextRegion`s (text + confidence + bbox), `FaceDetail`s (bbox, confidence, emotion, gaze, head pose — *plural*, not just one face), `DetectedObject`s (label, confidence, bbox), a full `ColorProfile`, and a full `CompositionAnalysis` (rule-of-thirds, negative space, clutter, balance, symmetry). **Module 8 must not recompute any of this.** It consumes these results and turns the *regions they describe* into *pixel assets* (crops, masks) plus the handful of properties Module 4 does not compute (embeddings, landmarks, segmentation masks, depth, gradients, per-region font/style estimates, effects).
2. **Module 6.5 (`visual_reference_engine.py` + `vre_components/`) already exists** as a smaller, single-face/single-object version of exactly this problem. Its `AssetWriter` (atomic temp-file + `Path.replace()`, OpenCV encode, non-empty-array guard) is asset-format-agnostic and is reused directly by Module 8 rather than reimplemented. Its manifest-cache-verification algorithm (hash match + per-file existence + per-file non-zero-size check) is the pattern Module 8's cache layer extends.
3. **A `vision_stack` V2.1 package already exists** (`modules/vision_stack/`) with a YAML-driven `VisionStackConfig` declaring exactly the model families Module 8 needs: `sam2`, `birefnet`, `bisenet`, `depth_anything`, `teed`, `insightface`, `grounding_dino`, `florence2`, `openclip`, `paddleocr`. It provides `ModelRegistry` (lifecycle tracking), `GPUResourceManager` (single-active-model reservation via `reserve(name)` context manager, enforcing sequential GPU use — precisely the RTX 4060 constraint), and a `ModelLoader` that resolves and validates checkpoint files. **Today this package is boot-time metadata and lifecycle plumbing only** — the one real inference wrapper implemented is `grounding_dino.py` (`_GroundingDINOOutputParser` + wrapper class, following a load → infer → parse-to-Pydantic → release pattern). SAM2, BiRefNet, BiSeNet, DepthAnything, TEED, and multi-purpose InsightFace usage do **not** yet have inference wrappers. **Module 8 must be designed to consume the vision-stack registry/GPU-lock contract, and must specify (but not implement) new wrapper modules for these five model families, following the `grounding_dino.py` pattern exactly.** This is the single largest piece of net-new engineering Module 8 implies outside its own package.
4. **Module 4's OCR stage uses EasyOCR, not PaddleOCR.** The vision-stack YAML lists `paddleocr` as a V2.1 stack member, but Module 4 was built before V2.1 and was never migrated. Module 8 does not need PaddleOCR at all — it reuses Module 4's `TextRegion`s. This is called out explicitly so no implementer "fixes" this by wiring PaddleOCR into Module 8; that would be scope creep and a duplicate OCR pass.
5. **Every module's public save/cache function follows one shape:** `build_X(...) -> XModel`, `save_X(model, dir=DEFAULT_X_DIR) -> Path`, atomic JSON write via temp file + `Path.replace()`, and a `Cache Error` exception subclassing that module's base error. Module 8's public API mirrors this exactly for consistency with `redesign_spec_engine.py`, `prompt_compiler.py`, and `thumbnail_intelligence.py`.
6. **`main.py` does not currently wire Module 6 or Module 6.5 into the running pipeline** — both exist as complete, tested, standalone modules that a future integration step will splice in. Per Afsar's established pattern (confirmed by the state of Modules 6 and 6.5), **Module 8 is designed but not wired into `main.py`** in this document. Wiring is explicitly listed as a Phase 8 (post-implementation, non-design) step, matching how Module 6 and 6.5 were left.
7. **Every existing module's Pydantic models are `frozen=True`**, validate non-empty strings by stripping, and every module gets exactly one log file under `logs/` via a `_configure_logger()` free function called once at import time with `enqueue=True`, 10 MB rotation, 30-day retention. Module 8 follows this exactly — one log file (`module8.log`), one `_configure_logger()`.
8. **`data/` subdirectories are named `data/<noun>/`, sharded by `video_id` when the module produces more than one file per creator** (`data/visual_references/<video_id>/...`). Module 8 follows this: `data/asset_extraction/<video_id>/...`.
9. **`config.py` has zero cross-module runtime imports except `models.py` and `vision_stack.config`** — it is pure constants plus one `GenerationProfile`-typed dict. Module 8's additions follow this: constants only, appended in a clearly delimited section, no new top-level imports beyond what's already there.

---

## 3. Design Goals / Non-Goals

### Goals
- Extract **every** reusable visual asset named in the brief, once, and never require another module to re-open the source pixels.
- Be **resumable**: a killed process (OOM, laptop sleep, Ctrl-C) must be able to pick up mid-extraction without redoing completed work.
- Be **incremental**: if only the object detections change (e.g. Module 4 was re-run), only object-family assets should recompute.
- Run entirely within an 8 GB VRAM / 16 GB system RAM budget, one model resident on GPU at a time.
- Produce a single, versioned, immutable, machine-readable contract (`asset_manifest.json`) that Module 9 can consume without any image library.
- Degrade gracefully: a failed model (e.g. SAM2 OOM) should not fail the whole run — it should mark that asset family absent/low-confidence and continue, exactly as Module 4 already does with `IntelligenceStatus.PARTIAL`.

### Non-Goals
- Module 8 does **not** decide keep/remove/replace/enhance/add — that is Module 9.
- Module 8 does **not** run OCR, face detection, object detection, color analysis, or composition scoring — those are Module 4's job and are consumed as input, not recomputed.
- Module 8 does **not** composite, inpaint, or generate any new pixels — it only extracts/derives assets from the source image (crops, masks, maps). Any generative work belongs to Module 10/11.
- Module 8 does **not** implement the five missing vision-stack inference wrappers as part of *this* module's package — those are specified here as a dependency and belong in `modules/vision_stack/`, following the existing `grounding_dino.py` precedent, implemented in the phase plan below but as vision-stack deliverables, not `asset_extraction_engine.py` internals.

---

## 4. Pipeline Position & Data Flow

```
Module 3 (ThumbnailData)  ──┐
                             │
Module 4 (ThumbnailIntelligence) ──► Module 8: Asset Extraction Engine ──► AssetExtractionManifest
                             │                                                     │
                             │                                                     ▼
                             │                                          data/asset_extraction/<video_id>/
                             │                                            ├── asset_manifest.json
                             │                                            ├── people/, objects/, scene/,
                             │                                            │   typography/, effects/
                             │                                            └── *.png, *.json sidecars
                             │
                             └── (thumbnail_path, video_id — Module 8 never re-reads Module 3's raw
                                  bytes independently of what Module 4 already validated)
                                                     │
                                                     ▼
                                        Module 9: AI Decision Engine
                                    (keep.json / remove.json / replace.json /
                                     enhance.json / add.json — reads the manifest only)
                                                     │
                                                     ▼
                                     Module 10: Asset Composer  (reads manifest + decisions)
                                                     │
                                                     ▼
                                  Module 11: Generation Pipeline (ComfyUI, reuses preserved assets)
```

Module 8's only inputs are `ThumbnailData` (Module 3 — for the validated source image path) and `ThumbnailIntelligence` (Module 4 — for every already-detected region and score). This mirrors Module 5's and Module 6's inputs exactly (both take Module 4's report, nothing else), keeping Module 8 a peer of Modules 5/6 rather than a parallel branch that reinvents Module 4.

---

## 5. High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                      AssetExtractionEngine (orchestrator)             │
│  modules/asset_extraction_engine.py                                   │
│                                                                         │
│  extract(video_id, source_image_path, intelligence, options)          │
│    1. validate video_id / source path                                 │
│    2. compute cache key = sha256(source_bytes) + sha256(intelligence  │
│       .model_dump_json())                                             │
│    3. load partial-progress manifest if present (resume)              │
│    4. for each asset family NOT already complete & cache-valid:       │
│         dispatch to the family's processor under the GPU lock         │
│    5. merge results into AssetExtractionManifest                      │
│    6. write manifest atomically (survives interruption: written       │
│       incrementally per family, see §17)                              │
│    7. return manifest                                                  │
└───────────────────────────────────────────────────────────────────────┘
        │            │            │            │           │          │
        ▼            ▼            ▼            ▼           ▼          ▼
   PersonProcessor SceneProcessor ObjectProcessor TypographyProcessor
   VisualProperties CompositionAssetProcessor  EffectsProcessor
        │            │            │
        └──────┬─────┴─────┬──────┘
               ▼            ▼
      AssetWriter (reused    ModelBridge
      from vre_components,   (thin adapter over
      extended for JSON      vision_stack.RuntimeManager /
      sidecars)              GPUResourceManager)
               │                    │
               ▼                    ▼
      ManifestBuilder        vision_stack model wrappers
      (asset_manifest.json)  (grounding_dino.py exists;
                              sam2.py, birefnet.py, bisenet.py,
                              depth_anything.py, teed.py,
                              insightface_multi.py — specified
                              here, implemented as vision_stack
                              deliverables)
```

Every processor implements a narrow `IXProcessor` ABC (mirroring `vre_components/interfaces.py`) and returns plain `numpy` arrays / dicts — never Pydantic models. Only `ManifestBuilder` touches Pydantic. This is the same separation of concerns VRE already uses and it is what makes each processor independently unit-testable with fabricated arrays and no GPU.

---

## 6. Folder Structure

```
thumbnail-ai/
├── modules/
│   ├── asset_extraction_engine.py          # orchestrator (peer of visual_reference_engine.py)
│   ├── asset_extraction_exceptions.py       # module8 exception hierarchy
│   ├── asset_extraction_components/
│   │   ├── __init__.py
│   │   ├── interfaces.py                    # ABCs for every processor + writer + builder
│   │   ├── person_processor.py              # faces(plural), embeddings, landmarks, body,
│   │   │                                     # pose, clothing, hair, accessories
│   │   ├── scene_processor.py                # background/foreground/depth/segmentation/sky/ground
│   │   ├── object_processor.py               # per-object crop/mask/bbox/hierarchy
│   │   ├── typography_processor.py           # per-text-region crop/font-estimate/alignment/style
│   │   ├── visual_properties_processor.py    # palette/gradients/lighting/shadow/highlight/blur/focus
│   │   ├── composition_processor.py          # eye-flow map, negative-space mask, hierarchy overlay
│   │   ├── effects_processor.py              # glow/outline/drop-shadow/motion-blur/particles (heuristic)
│   │   ├── model_bridge.py                   # thin adapter over vision_stack RuntimeManager
│   │   ├── asset_writer.py                   # extends vre_components.AssetWriter pattern (PNG+JSON)
│   │   └── manifest_builder.py               # builds/validates AssetExtractionManifest
│   ├── vision_stack/
│   │   ├── sam2.py                           # NEW — follows grounding_dino.py pattern
│   │   ├── sam2_exceptions.py                # NEW
│   │   ├── birefnet.py                       # NEW
│   │   ├── birefnet_exceptions.py            # NEW
│   │   ├── bisenet.py                        # NEW
│   │   ├── bisenet_exceptions.py             # NEW
│   │   ├── depth_anything.py                 # NEW
│   │   ├── depth_anything_exceptions.py      # NEW
│   │   ├── teed.py                           # NEW
│   │   ├── teed_exceptions.py                # NEW
│   │   └── insightface_multi.py              # NEW — multi-face variant of Module 4's single-pass usage
│   ├── models.py                             # + Module 8 model block (§8), appended, never edited in place
│   └── config.py                             # + Module 8 constants block (§20), appended
├── data/
│   └── asset_extraction/
│       └── <video_id>/
│           ├── asset_manifest.json
│           ├── people/
│           │   ├── face_01.png  face_01_mask.png  face_02.png ...
│           │   ├── body_mask.png  hair_mask.png
│           │   ├── clothing_mask.png  accessories_mask.png
│           │   ├── landmarks.json  embeddings.json  pose.json
│           ├── scene/
│           │   ├── background.png  foreground.png
│           │   ├── depth.png  segmentation.png
│           │   ├── sky_mask.png  ground_mask.png
│           ├── objects/
│           │   ├── object_01.png  object_01_mask.png  object_02.png ...
│           │   ├── object_masks.json  object_hierarchy.json
│           ├── typography/
│           │   ├── text_region_01.png  text_region_02.png ...
│           │   ├── text_boxes.json
│           ├── visual/
│           │   ├── colors.json  gradients.json  lighting.json
│           ├── composition/
│           │   ├── eye_flow.png  negative_space_mask.png
│           │   ├── composition.json
│           └── effects/
│               └── effects.json
└── tests/
    ├── test_asset_extraction_engine.py
    └── asset_extraction_components/
        ├── test_person_processor.py
        ├── test_scene_processor.py
        ├── test_object_processor.py
        ├── test_typography_processor.py
        ├── test_visual_properties_processor.py
        ├── test_composition_processor.py
        ├── test_effects_processor.py
        ├── test_model_bridge.py
        ├── test_asset_writer.py
        └── test_manifest_builder.py
```

`asset_manifest.json` is the **only** artifact Module 9 is contractually allowed to depend on; every PNG/JSON path inside it is an implementation detail Module 9 reaches through the manifest, not by guessing filenames.

---

## 7. Python Module Layout

| Module | Responsibility | Depends on |
|---|---|---|
| `asset_extraction_engine.py` | Orchestration, caching, resume, dispatch, top-level public API | `models`, `config`, `asset_extraction_exceptions`, `asset_extraction_components.*` |
| `asset_extraction_exceptions.py` | Typed exception hierarchy | none (leaf module, like `vre_exceptions.py`) |
| `asset_extraction_components/interfaces.py` | ABCs only | `models`, `numpy` |
| `asset_extraction_components/person_processor.py` | People family | `vision_stack` (insightface_multi, bisenet), `interfaces` |
| `asset_extraction_components/scene_processor.py` | Scene family | `vision_stack` (birefnet, sam2, depth_anything, grounding_dino), `interfaces` |
| `asset_extraction_components/object_processor.py` | Objects family | `vision_stack` (sam2), `interfaces` |
| `asset_extraction_components/typography_processor.py` | Typography family | pure OpenCV/PIL, no vision_stack |
| `asset_extraction_components/visual_properties_processor.py` | Visual family | pure OpenCV/numpy, no vision_stack |
| `asset_extraction_components/composition_processor.py` | Composition family | pure OpenCV/numpy, no vision_stack |
| `asset_extraction_components/effects_processor.py` | Effects family | pure OpenCV/numpy, no vision_stack |
| `asset_extraction_components/model_bridge.py` | Adapter to `vision_stack.RuntimeManager` | `vision_stack.runtime`, `vision_stack.resources` |
| `asset_extraction_components/asset_writer.py` | Atomic PNG + JSON persistence | `vre_components.asset_writer` (composition, not inheritance — see §11) |
| `asset_extraction_components/manifest_builder.py` | Build/validate `AssetExtractionManifest` | `models` |

This table is also the dependency-direction contract: nothing under `asset_extraction_components/` imports `asset_extraction_engine.py`, and no processor imports another processor. All cross-processor coordination happens in the orchestrator only.

---

## 8. Data Models (`models.py` additions)

All new models are appended to the existing `Module 8 — Asset Extraction Engine` section at the end of `models.py`, `frozen=True`, following the exact validator idioms already used by `AssetMetadata` / `VisualReferenceManifest`. No existing model is modified.

```
BoundingBox                — REUSED from Module 4 (normalized [0,1] fractions); do not redefine.
VisualBoundingBox           — REUSED from Module 6.5 (absolute pixel box); do not redefine.

AssetFileRef                        # replaces ad-hoc "file_path: str" scattered per family
    asset_type: str
    file_path: str
    checksum: str                   # sha256 hex, reuses AssetMetadata's validator pattern
    resolution: tuple[int, int]
    confidence_score: Optional[float] = None
    source: Literal["module4_reuse", "extracted", "derived"]
        # module4_reuse = crop of a Module-4-detected region (e.g. a face crop from FaceDetail.bbox)
        # extracted     = produced by a vision-stack model this module invoked (e.g. SAM2 mask)
        # derived       = computed analytically from pixels with no ML model (e.g. dominant gradient)

PersonAsset
    person_index: int
    face: Optional[AssetFileRef]
    face_mask: Optional[AssetFileRef]
    face_embedding: Optional[list[float]]      # InsightFace 512-d embedding, stored inline (small)
    facial_landmarks: Optional[list[tuple[float, float]]]
    body_mask: Optional[AssetFileRef]
    pose_keypoints: Optional[list[tuple[float, float, float]]]   # (x, y, confidence)
    clothing_mask: Optional[AssetFileRef]
    hair_mask: Optional[AssetFileRef]
    accessories_masks: list[AssetFileRef] = []
    source_face_detail_index: int              # index back into ThumbnailIntelligence.faces.faces
    extraction_status: Literal["success", "partial", "skipped"]
    extraction_notes: list[str] = []

SceneAsset
    background: Optional[AssetFileRef]
    foreground: Optional[AssetFileRef]
    depth_map: Optional[AssetFileRef]
    segmentation_map: Optional[AssetFileRef]
    sky_mask: Optional[AssetFileRef]
    ground_mask: Optional[AssetFileRef]
    extraction_status: Literal["success", "partial", "skipped"]
    extraction_notes: list[str] = []

ObjectAsset
    object_index: int
    label: str
    crop: Optional[AssetFileRef]
    mask: Optional[AssetFileRef]
    bbox: BoundingBox
    confidence: float
    parent_object_index: Optional[int]         # object hierarchy: None = top-level
    child_object_indices: list[int] = []
    source_detected_object_index: int          # index back into ThumbnailIntelligence.objects

TypographyAsset
    text_region_index: int
    crop: Optional[AssetFileRef]
    text: str                                  # copied verbatim from Module 4's TextRegion.text
    bbox: BoundingBox
    estimated_font_family_guess: Optional[str]  # best-effort heuristic label, e.g. "sans-serif-bold"
    estimated_font_size_px: Optional[float]
    alignment: Literal["left", "center", "right", "unknown"]
    dominant_text_color: Optional[str]          # #rrggbb
    has_stroke_or_outline: bool
    source_text_region_index: int               # index back into ThumbnailIntelligence.ocr.text_regions

VisualPropertiesAsset
    dominant_colors: list[str]                  # REUSED verbatim from ColorProfile.dominant_colors
    palette_extended: list[str]                 # AEE-derived, larger k-means palette (k=8 vs Module 4's k=5)
    gradients_detected: list[str]                # coarse labels, e.g. "top-to-bottom-dark-to-light"
    lighting_direction: Optional[str]             # e.g. "top-left", "flat", "backlit"
    shadow_regions: list[BoundingBox] = []
    highlight_regions: list[BoundingBox] = []
    blur_map_summary: Literal["sharp", "mixed", "soft"]
    focus_bbox: Optional[BoundingBox]             # estimated sharpest region

CompositionAsset
    eye_flow_map: Optional[AssetFileRef]
    negative_space_mask: Optional[AssetFileRef]
    visual_hierarchy_overlay: Optional[AssetFileRef]
    source_composition_analysis: CompositionAnalysis   # REUSED, embedded verbatim from Module 4

EffectsAsset
    glow_detected: bool
    outline_detected: bool
    drop_shadow_detected: bool
    motion_blur_detected: bool
    particles_detected: bool
    confidence: float                            # heuristic-only; deliberately low-weighted by Module 9
    notes: list[str] = []

AssetExtractionStatus(str, Enum)
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"

AssetExtractionManifest
    video_id: str
    source_thumbnail_path: str
    source_hash: str                             # sha256 of the source image bytes
    intelligence_hash: str                        # sha256 of Module 4's report — cache invalidation input
    engine_version: str
    people: list[PersonAsset] = []
    scene: Optional[SceneAsset]
    objects: list[ObjectAsset] = []
    typography: list[TypographyAsset] = []
    visual_properties: Optional[VisualPropertiesAsset]
    composition: Optional[CompositionAsset]
    effects: Optional[EffectsAsset]
    status: AssetExtractionStatus
    partial_failure_reasons: list[str] = []
    completed_families: list[str] = []            # resume bookkeeping, see §17
    total_duration_seconds: float
    extracted_at: str                              # ISO-8601 UTC
```

Validators mirror existing conventions exactly: every `video_id`, `source_thumbnail_path`, checksum, and confidence field reuses the same `@field_validator` idioms already present in `AssetMetadata` and `ThumbnailIntelligence` (strip-and-reject-empty for strings, `sha256` hex-digest length/charset check for checksums, `[0.0, 1.0]` range check for confidences). These are not re-specified per model above to avoid repetition; implementers copy the validator bodies verbatim from `models.py` lines covering `AssetMetadata`.

---

## 9. Public API

Mirrors `redesign_spec_engine.py` / `prompt_compiler.py` / `thumbnail_intelligence.py` exactly, so `main.py` integration (when it happens, per §23) is a drop-in following the pattern already used for every prior module.

```python
# asset_extraction_engine.py

def extract_assets(
    video_id: str,
    source_image_path: str,
    intelligence: ThumbnailIntelligence,
    *,
    storage_root: Path = DEFAULT_ASSET_EXTRACTION_DIR,
    options: Optional[dict] = None,
) -> AssetExtractionManifest:
    """Extract every asset family for one thumbnail. Cache-aware, resumable."""

def save_asset_manifest(
    manifest: AssetExtractionManifest,
    *,
    storage_root: Path = DEFAULT_ASSET_EXTRACTION_DIR,
) -> Path:
    """Atomically persist a manifest. (extract_assets calls this internally;
    exposed separately only for symmetry with every other module's save_X."""

def load_asset_manifest(
    video_id: str,
    *,
    storage_root: Path = DEFAULT_ASSET_EXTRACTION_DIR,
) -> AssetExtractionManifest:
    """Load and validate a persisted manifest for a video_id. Raises
    ManifestNotFoundError / ManifestValidationError."""


class AssetExtractionEngine:
    """Stateful orchestrator, analogous to VisualReferenceEngine, for callers
    that want to inject fake processors/writers (tests) or reuse a warm
    ModelBridge across many creators in one run."""

    def __init__(
        self,
        storage_root: Path = DEFAULT_ASSET_EXTRACTION_DIR,
        person_processor: IPersonProcessor | None = None,
        scene_processor: ISceneProcessor | None = None,
        object_processor: IObjectProcessor | None = None,
        typography_processor: ITypographyProcessor | None = None,
        visual_properties_processor: IVisualPropertiesProcessor | None = None,
        composition_processor: ICompositionProcessor | None = None,
        effects_processor: IEffectsProcessor | None = None,
        asset_writer: IAssetExtractionWriter | None = None,
        manifest_builder: IAssetManifestBuilder | None = None,
        model_bridge: IModelBridge | None = None,
        cache_enabled: bool = ASSET_EXTRACTION_CACHE_ENABLED,
    ) -> None: ...

    def extract(
        self,
        video_id: str,
        source_image_path: str,
        intelligence: ThumbnailIntelligence,
        options: Optional[dict] = None,
    ) -> AssetExtractionManifest: ...

    def clean_assets(self, video_id: str) -> bool:
        """Remove the generated shard for one video_id."""
```

`extract_assets()` (module-level function) is a thin convenience wrapper that constructs one `AssetExtractionEngine` with defaults and calls `.extract()` — same relationship `analyze_thumbnail()` has to Module 4's internal class, and the same relationship `build_redesign_specification()` has to Module 5's internals.

---

## 10. Internal APIs / Component Contracts

`asset_extraction_components/interfaces.py`, one ABC per processor, deliberately narrow — each takes the source image plus only the slice of `ThumbnailIntelligence` it needs, and returns plain data (arrays, dicts), never Pydantic models or file paths. This keeps every processor GPU/model-agnostic in its type signature and trivially fakeable in tests, exactly like `vre_components/interfaces.py`.

```python
class IPersonProcessor(ABC):
    @abstractmethod
    def process(
        self, image: np.ndarray, faces: FaceAnalysis
    ) -> list[dict[str, Any]]:
        """One dict per detected face: crops/masks as np.ndarray, embeddings
        as list[float], landmarks/pose as coordinate lists, all keyed by
        the same field names used in PersonAsset."""

class ISceneProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Keys: background, foreground, depth_map, segmentation_map,
        sky_mask, ground_mask. Missing keys mean that sub-asset could not
        be produced (model skipped/failed) — not an exception."""

class IObjectProcessor(ABC):
    @abstractmethod
    def process(
        self, image: np.ndarray, objects: list[DetectedObject]
    ) -> list[dict[str, Any]]:
        """One dict per input DetectedObject: crop, mask (np.ndarray),
        plus parent_index/child_indices computed from bbox containment."""

class ITypographyProcessor(ABC):
    @abstractmethod
    def process(
        self, image: np.ndarray, text_regions: list[TextRegion]
    ) -> list[dict[str, Any]]:
        """One dict per input TextRegion: crop, font/alignment/color
        estimates. Pure OpenCV — no model dependency."""

class IVisualPropertiesProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray, colors: ColorProfile) -> dict[str, Any]:
        """Palette/gradient/lighting/blur/focus, seeded by Module 4's
        already-computed ColorProfile so brightness/contrast/saturation
        are never recomputed."""

class ICompositionAssetProcessor(ABC):
    @abstractmethod
    def process(
        self, image: np.ndarray, composition: CompositionAnalysis
    ) -> dict[str, Any]:
        """Renders visual overlays (eye-flow map, negative-space mask)
        from Module 4's already-computed scores — never recomputes them."""

class IEffectsProcessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> dict[str, Any]:
        """Heuristic-only glow/outline/shadow/motion-blur/particle flags."""

class IModelBridge(ABC):
    @abstractmethod
    def run(self, model_name: str, operation: Callable[[Any], Any]) -> Any:
        """Reserve the shared GPU slot for model_name via vision_stack's
        GPUResourceManager, execute operation(model), release. Raises
        ModelUnavailableError if the checkpoint is missing/invalid and
        falls back per the model's configured VisionModelFallback policy."""

class IAssetExtractionWriter(ABC):
    @abstractmethod
    def write_image(self, array: np.ndarray, destination_path: Path) -> bool: ...
    @abstractmethod
    def write_json_sidecar(self, data: dict, destination_path: Path) -> bool: ...
    @abstractmethod
    def purge_directory(self, target_dir: Path) -> bool: ...

class IAssetManifestBuilder(ABC):
    @abstractmethod
    def build(self, **family_results) -> AssetExtractionManifest: ...
    @abstractmethod
    def serialize_to_disk(self, manifest: AssetExtractionManifest, path: Path) -> None: ...
```

---

## 11. Responsibilities of Every Class

| Class | Single responsibility |
|---|---|
| `AssetExtractionEngine` | Validate inputs, compute cache key, decide which families to (re)run, hold the family-dispatch loop, merge results, delegate persistence. Contains **no** image-processing or model code. |
| `PersonProcessor` | For every `FaceDetail` Module 4 already found, produce a face crop + mask, then invoke `ModelBridge` for embedding (InsightFace) and, where confidence allows, body/hair/clothing/accessory masks (BiSeNet parsing) and pose (a lightweight pose head or BiSeNet-derived proxy — see Open Questions). Never re-runs face *detection*. |
| `SceneProcessor` | Foreground/background split (BiRefNet), depth (DepthAnything), sky/ground semantic masks (GroundingDINO open-vocabulary query "sky . ground" then SAM2 mask refinement), full-frame segmentation map (SAM2 automatic mask generation, downselected). |
| `ObjectProcessor` | For every `DetectedObject` Module 4 already found, refine the coarse bbox into a precise mask via SAM2 (box-prompted), crop it, and compute a containment-based hierarchy (an object whose bbox is ≥90% inside another's is a child). Never re-runs object *detection*. |
| `TypographyProcessor` | For every `TextRegion` Module 4 already found, crop the region, then run cheap classical-CV heuristics (stroke width transform for font-weight guess, connected-component alignment inference, k-means-1 for dominant text color, edge-ring detection for stroke/outline). No ML model. |
| `VisualPropertiesProcessor` | Extend Module 4's `ColorProfile` with a larger palette, coarse gradient direction labels (Sobel-based), a lighting-direction estimate (luminance-gradient centroid), and a Laplacian-variance blur map summarized to one label + one focus bbox. No ML model. |
| `CompositionAssetProcessor` | Render Module 4's already-computed composition *scores* into visual artifacts Module 10 can literally use in a composite (an eye-flow arrow map, a negative-space binary mask, a hierarchy heat overlay). No ML model, no re-scoring. |
| `EffectsProcessor` | Best-effort classical-CV heuristics for glow (halo luminance ring around high-contrast edges), outline (Canny + morphological ring near subject bbox), drop-shadow (offset dark blob near subject), motion blur (directional Laplacian), particles (small isolated high-contrast blob count). Every flag ships with a confidence Module 9 is expected to weight low. |
| `ModelBridge` | The **only** class in Module 8 that touches `vision_stack`. Wraps `RuntimeManager.reserve_model()` / `GPUResourceManager.reserve()`, translates vision-stack lifecycle/resource exceptions into Module 8's own exception hierarchy, and applies each model's configured `VisionModelFallback` (skip_stage / cpu_fallback / cpu_tiled_processing / retry_once) exactly as declared in `vision_stack.yaml` — Module 8 never invents its own fallback policy. |
| `AssetWriter` (Module 8's) | Atomic PNG writes, reusing `vre_components.asset_writer.AssetWriter`'s exact temp-file-then-`Path.replace()` algorithm (composed, not subclassed — see rationale below), plus one new method, `write_json_sidecar`, for the family-level JSON files (`landmarks.json`, `object_masks.json`, `text_boxes.json`, etc.), using the same atomic-temp-file discipline. |
| `ManifestBuilder` | Assembles `AssetExtractionManifest` from the per-family raw results, computes `source_hash`/`intelligence_hash`, sets `status`/`partial_failure_reasons` from which families degraded, and performs the final Pydantic validation pass (the single point where a malformed asset set becomes a hard `ManifestValidationError` rather than a silently wrong file on disk). |

**Why `AssetWriter` is composed, not subclassed:** `vre_components.asset_writer.AssetWriter` is scoped to `VRE_STORAGE_ROOT`-shaped single-image writes and has no JSON-sidecar method. Module 8's writer holds an internal `vre_components.asset_writer.AssetWriter()` instance and delegates `write_image` to it, adding `write_json_sidecar` as new surface. This avoids modifying `vre_components/` (which the brief's constraints — and Afsar's standing instruction to never modify existing files additively — both forbid) while still eliminating duplicate atomic-write logic.

---

## 12. Asset Lifecycle

```
 [Module 4 report ready]
          │
          ▼
  video_id + source path + intelligence handed to AssetExtractionEngine.extract()
          │
          ▼
  ┌─────────────────────────┐
  │ REGISTERED               │  cache key computed, target shard dir resolved
  └───────────┬──────────────┘
              ▼
  ┌─────────────────────────┐
  │ CACHE_CHECK               │  existing manifest (if any) hash-compared;
  └───────────┬──────────────┘  per-family completeness checked against
              │                  completed_families + per-file existence
      cache hit, all families        cache miss / partial / new video_id
      complete & valid                          │
              │                                  ▼
              │                     ┌─────────────────────────┐
              │                     │ EXTRACTING (per family)  │  families run in a fixed
              │                     └───────────┬──────────────┘  deterministic order (§21);
              │                                  │                 each family's manifest
              │                                  ▼                 fragment is written the
              │                     ┌─────────────────────────┐    moment it completes
              │                     │ FAMILY_PERSISTED         │
              │                     └───────────┬──────────────┘
              │                                  │  (loop until all families attempted)
              │                                  ▼
              │                     ┌─────────────────────────┐
              │                     │ FINALIZING                │  status computed
              │                     └───────────┬──────────────┘  (success/partial/error)
              │                                  │
              └──────────────►  ┌─────────────────────────┐
                                 │ COMPLETE                  │  full manifest returned
                                 └───────────┬──────────────┘
                                             │
                                     video_id explicitly cleaned
                                             ▼
                                 ┌─────────────────────────┐
                                 │ PURGED                    │  clean_assets(video_id)
                                 └─────────────────────────┘  removes the whole shard
```

A family that fails all its retries lands in `FAMILY_PERSISTED` with its manifest fragment set to `None`/empty and a `partial_failure_reasons` entry — it is never left "in progress" on disk; every write is atomic and every family is terminal (success, partial, or explicitly absent) by the time the loop moves to the next family. This is what makes resume safe: on restart, the engine only ever sees fully-terminal family states in the on-disk manifest fragment.

---

## 13. Vision-Stack Integration (Model Execution)

Module 8 is a **consumer** of `vision_stack`, never a re-implementer of it. Concretely:

1. On first use in a process, `AssetExtractionEngine` (via `ModelBridge`) calls `RuntimeManager.bootstrap()` once (idempotent — guarded by the same pattern `RuntimeManager` already uses for `self.runtime`), which loads `vision_stack.yaml`, validates checkpoint presence, and registers all ten models.
2. For each model a processor needs (e.g. `SceneProcessor` needs `birefnet`, `depth_anything`, `sam2`, `grounding_dino`), `ModelBridge.run(model_name, operation)` calls `gpu_resources.reserve(model_name)`, which — per the existing `resources.py` contract — raises `VisionStackResourceError` if the GPU is already held (should never happen given Module 8's sequential dispatch, but is a hard safety net) and otherwise transitions the model `REGISTERED → CPU_CACHED → GPU_ACTIVE`, yields it, and on `__exit__` demotes it back to `CPU_CACHED`.
3. **The five missing inference wrappers** (`sam2.py`, `birefnet.py`, `bisenet.py`, `depth_anything.py`, `teed.py`) and the multi-face variant of InsightFace usage are specified here as required dependencies but are **`vision_stack` package deliverables**, not `asset_extraction_engine.py` internals — exactly as `grounding_dino.py` lives in `vision_stack/`, not in whatever module first needed GroundingDINO. Each new wrapper must follow the `grounding_dino.py` precedent:
   - a private `_<Model>OutputParser` class converting raw tensor/numpy output into a typed, immutable Pydantic result (added to `vision_stack/models.py`),
   - a public wrapper class taking a `RegisteredVisionModel` and raw `np.ndarray` input, returning the parsed result,
   - a dedicated `*_exceptions.py` (e.g. `SAM2InferenceError`, `SAM2LoadError`, `SAM2OutOfMemoryError`) subclassing `vision_stack.exceptions.VisionStackError` — never Module 8's own exception base, since these failures happen *inside* vision_stack, before Module 8's own error handling wraps them,
   - its own `_configure_logger()` writing to `logs/vision_stack_<model>.log`, matching `grounding_dino.py`'s per-wrapper log file.
4. `ModelBridge` translates any `VisionStack*Error` it catches into Module 8's own `AssetFamilyModelError` (see §19), attaching the family name and model name, so `AssetExtractionEngine`'s per-family error handling never needs to know vision-stack's exception types directly — one adapter boundary, one place to update if vision_stack's exception hierarchy changes.
5. **Fallback policy is never decided by Module 8.** `vision_stack.yaml` already declares, per model, whether a failure should `skip_stage`, `cpu_fallback`, `cpu_tiled_processing`, or `retry_once`. `ModelBridge.run()` reads `VisionModelConfig.fallback` off the registered model and applies it before raising anything to the processor — a processor only ever sees a clean result or a terminal `AssetFamilyModelError` after the configured fallback has already been exhausted.

---

## 14. Outputs

Per `<video_id>` shard (paths relative to `data/asset_extraction/<video_id>/`):

```
asset_manifest.json                 # THE contract — everything else is reached through it

people/face_01.png                  people/face_01_mask.png
people/face_02.png                  people/face_02_mask.png   (N = FaceAnalysis.face_count)
people/body_mask.png                people/hair_mask.png
people/clothing_mask.png            people/accessories_01_mask.png ...
people/landmarks.json               people/embeddings.json     people/pose.json

scene/background.png                scene/foreground.png
scene/depth.png                     scene/segmentation.png
scene/sky_mask.png                  scene/ground_mask.png

objects/object_01.png               objects/object_01_mask.png  (N = len(objects))
objects/object_masks.json           objects/object_hierarchy.json

typography/text_region_01.png ...   (N = len(ocr.text_regions))
typography/text_boxes.json

visual/colors.json                  visual/gradients.json       visual/lighting.json

composition/eye_flow.png            composition/negative_space_mask.png
composition/composition.json

effects/effects.json
```

This directly satisfies the brief's example output list while sharding by family (matching the repository's existing convention of grouping related artifacts rather than a single flat directory — see how `data/analysis/`, `data/redesign_specs/`, `data/prompt_packages/` are each single-file-per-video, and `data/visual_references/<video_id>/` is the one existing precedent for a multi-file shard, which Module 8 extends).

---

## 15. Manifest Schema (`asset_manifest.json`)

The on-disk JSON is exactly `AssetExtractionManifest.model_dump_json(indent=2)` — no hand-written serialization layer, matching how every other module persists its Pydantic model (`ThumbnailIntelligence`, `RedesignSpecification`, `PromptPackage`, `VisualReferenceManifest` are all persisted this same way). `ManifestBuilder.serialize_to_disk()` writes to a `.tmp` sibling and `Path.replace()`s it into place, matching the atomic-write discipline already used everywhere paths are persisted in this repository (`_persist_generated_thumbnail` in `main.py`, `AssetWriter._atomic_write` in `vre_components`).

---

## 16. Caching Strategy

Cache key = `sha256(source_image_bytes) || sha256(intelligence.model_dump_json())`, stored as two separate fields (`source_hash`, `intelligence_hash`) rather than one combined hash, so a cache-miss diagnostic can say *which* upstream input changed. This directly extends VRE's single-hash cache (source-image-only) to also invalidate when Module 4's report changes — necessary because unlike VRE (which only ever needed the raw pixels), most of Module 8's families are seeded by Module 4's detections, so a re-run of Module 4 (e.g. after a model upgrade) must invalidate Module 8's cache even if the image bytes are unchanged.

Cache verification (`_verify_cache`, modeled directly on `VisualReferenceEngine._verify_cache`):
1. Load the persisted manifest if `asset_manifest.json` exists.
2. Compare both hashes; any mismatch → full cache miss, re-extract every family.
3. If hashes match, walk every `AssetFileRef` in every family and confirm the file exists and has non-zero size (identical check to VRE's `path.is_file() or path.stat().st_size <= 0`).
4. Any missing/empty file → that specific **family** (not the whole manifest) is marked stale and re-run; families not touched by the missing file are left cached. This is the incremental-extraction requirement from the brief, implemented as fine-grained as the family level (not per-asset, to keep the cache-check cost bounded — see Open Questions for finer granularity tradeoffs).
5. A fully-valid cache hit returns the manifest with a `processing_metadata`-style flag (`cached_hit: true` per family, mirroring VRE's `processing_metadata.cached_hit` field) — Module 8 encodes this as an addition to each family's raw-result dict before it reaches `ManifestBuilder`, not as a manifest-level field, since a manifest can be a mix of cached and freshly-computed families.

---

## 17. Error Recovery / Resume-After-Interruption

Two mechanisms work together:

1. **Family-level atomicity.** Exactly as described in §12, the engine never holds a family "half-written." Either a family's assets and its manifest fragment are both fully persisted, or neither is. This is enforced by writing every family's PNGs first, then that family's contribution to the manifest fragment last, using the same temp-file-then-replace pattern throughout — so a process kill mid-family leaves, at worst, orphaned PNGs from an incomplete family and *no* corresponding manifest entry, which the next run's cache-verification step (§16 step 3–4) will detect as "family not in `completed_families`" and safely re-run, overwriting the orphans.
2. **`completed_families` bookkeeping.** `AssetExtractionManifest.completed_families` is updated and the manifest re-serialized to disk **after every single family**, not just at the very end. This means a full `extract()` call is really N small transactions (N = number of families = 7), and a resumed run reads the on-disk manifest, sees which families are already in `completed_families` with all their files verified present (§16), and skips straight to the first incomplete family — never restarting the whole extraction from zero.

This satisfies the brief's explicit "resume after interruption" and "incremental extraction" requirements without needing a separate journal/WAL file — the manifest itself, written incrementally, *is* the resume checkpoint.

---

## 18. Logging Strategy

One log file, `logs/module8.log`, configured once via a module-level `_configure_logger()` in `asset_extraction_engine.py`, called at import time — identical to every other module's pattern (`MODULE65_LOG_PATH`, `MODULE7_LOG_PATH`, etc.):

```python
logger.add(str(MODULE8_LOG_PATH), rotation="10 MB", retention="30 days",
           format=_LOG_FORMAT, level="DEBUG", enqueue=True)
```

`enqueue=True` is mandatory here — Afsar's Module 2 debugging history in memory shows this project already hit a Loguru serialization crash from logging raw exception objects with unpicklable tracebacks under `enqueue=True`; Module 8's exception logging must always pass `exc=str(exc)` (or `repr(exc)`), never the raw exception object, into any `logger.error(...)` call, exactly as the Module 2 fix established.

Per-family log lines follow the existing `logger.info("... video_id={id} ...", id=video_id)` structured-field style throughout (`family=`, `duration_ms=`, `cached=`, `status=`). The five new `vision_stack` wrappers each get their **own** log file (`logs/vision_stack_sam2.log`, etc.), matching `grounding_dino.py`'s precedent of one log file per wrapper rather than funneling model-level logs through Module 8's log.

---

## 19. Error Handling / Exception Hierarchy

`modules/asset_extraction_exceptions.py` — a leaf module with zero project-internal imports, following `vre_exceptions.py` exactly:

```python
class AssetExtractionError(Exception):
    """Base exception for every recoverable Module 8 failure."""

class SourceImageNotFoundError(AssetExtractionError):
    """Raised when a source image path is missing or unreadable."""

class IntelligenceReportInvalidError(AssetExtractionError):
    """Raised when the supplied ThumbnailIntelligence cannot seed extraction
    (e.g. status == 'error', or a referenced bbox is out of range)."""

class AssetFamilyModelError(AssetExtractionError):
    """Raised when a vision-stack-backed family exhausts its configured
    fallback policy. Carries family_name and model_name for logging."""

    def __init__(self, message: str, *, family_name: str, model_name: str | None = None) -> None:
        super().__init__(message)
        self.family_name = family_name
        self.model_name = model_name

class AssetFamilyDegradedWarning(Warning):
    """Signals one family fell back to a lower-fidelity result (e.g.
    cpu_fallback) but still produced usable output."""

class AssetWriteError(AssetExtractionError):
    """Raised when generated assets cannot be atomically persisted."""

class ManifestValidationError(AssetExtractionError):
    """Raised when the assembled manifest fails Pydantic validation."""

class ManifestNotFoundError(AssetExtractionError):
    """Raised by load_asset_manifest() when no manifest exists for a video_id."""

class CacheCorruptError(AssetExtractionError):
    """Raised (caught internally, never surfaced) when a cached manifest or
    asset file is unreadable; triggers a full or partial recompute."""
```

**Handling policy**, matching Module 4's `status: success | partial | error` precedent exactly:
- A single family's `AssetFamilyModelError` is **caught inside the engine's dispatch loop**, logged, recorded in `partial_failure_reasons`, and that family's slot in the manifest is left `None`/empty — extraction continues to the next family. The overall manifest `status` becomes `"partial"`.
- `SourceImageNotFoundError` and `IntelligenceReportInvalidError` are **not** caught inside the loop — they fail fast before any family runs, since nothing can be extracted without a valid source and a valid upstream report (mirrors `analyze_thumbnail`'s `InvalidMetadataError` fail-fast).
- `AssetWriteError` / `ManifestValidationError` at the final persistence step are **not** swallowed — a manifest that cannot be written or does not validate must fail the whole call, since a phantom in-memory-only manifest would break the resume contract in §17.
- Callers (eventually `main.py`) catch `AssetExtractionError` broadly at the pipeline level, exactly as `main.py` already does for `Module7Error`/`InvalidMetadataError`/etc. — one `except` clause, log, `skipped += 1`, `continue`.

---

## 20. Configuration (`config.py` additions)

Appended as a new delimited section, after the existing Module 6.5 section, before the Vision Stack V2.1 import block — constants only, no new imports beyond `Path` (already imported):

```python
# ---------------------------------------------------------------------------
# Module 8 — Asset Extraction Engine
# ---------------------------------------------------------------------------

MODULE8_LOG_PATH: Path = LOG_DIR / "module8.log"

DEFAULT_ASSET_EXTRACTION_DIR: Path = PROJECT_ROOT / "data" / "asset_extraction"
ASSET_MANIFEST_FILENAME: str = "asset_manifest.json"
ASSET_EXTRACTION_ENGINE_VERSION: str = "1.0.0"
ASSET_EXTRACTION_CACHE_ENABLED: bool = True

# --- Family execution order (fixed, deterministic; drives sequential GPU use) ---
ASSET_EXTRACTION_FAMILY_ORDER: tuple[str, ...] = (
    "typography",       # cheapest, no model — run first for fast partial results
    "visual_properties", # cheapest, no model
    "composition",        # cheapest, no model
    "objects",             # SAM2
    "people",               # InsightFace + BiSeNet
    "scene",                 # BiRefNet + DepthAnything + SAM2 + GroundingDINO
    "effects",                # cheapest, no model — run last, lowest priority
)

# --- Per-family thresholds ---
ASSET_MIN_FACE_CROP_CONFIDENCE: float = 0.5          # reuse Module 4's FACE_MIN_CONFIDENCE value
ASSET_MIN_OBJECT_MASK_CONFIDENCE: float = 0.35        # SAM2 box-prompted mask acceptance floor
ASSET_OBJECT_HIERARCHY_CONTAINMENT_RATIO: float = 0.9  # child-bbox-inside-parent threshold
ASSET_SKY_GROUND_PROMPT: str = "sky . ground . horizon"  # GroundingDINO open-vocab query
ASSET_EXTENDED_PALETTE_K: int = 8                       # vs Module 4's k=5 in ColorProfile
ASSET_BLUR_LAPLACIAN_SHARP_THRESHOLD: float = 100.0
ASSET_BLUR_LAPLACIAN_SOFT_THRESHOLD: float = 30.0
ASSET_EFFECTS_MIN_CONFIDENCE_TO_FLAG: float = 0.4

# --- Resource budget (RTX 4060 laptop, 8GB VRAM / 16GB system RAM) ---
ASSET_EXTRACTION_MAX_IMAGE_DIMENSION_PX: int = 2048     # downscale ceiling before any model call
ASSET_EXTRACTION_SAM2_TILE_SIZE_PX: int = 1024           # cpu_tiled_processing tile size
ASSET_EXTRACTION_MODEL_TIMEOUT_SECONDS: float = 30.0
ASSET_EXTRACTION_MAX_RETRY_ATTEMPTS: int = 2
```

`ASSET_EXTRACTION_FAMILY_ORDER` is deliberately ordered cheapest/no-model-first: if the process is interrupted early, the resumable manifest (§17) already contains the three zero-GPU families, and the most expensive family (`scene`, which needs four model reservations) runs last, when interruption has already claimed the most value.

---

## 21. Performance Design (RTX 4060, 16GB RAM)

- **Sequential single-model GPU use is enforced structurally**, not by convention: every model call goes through `ModelBridge.run()` → `GPUResourceManager.reserve()`, which raises rather than allows two models active at once (§13 point 2). Module 8 adds no parallelism inside a single video_id's extraction.
- **Batch processing across creators**: `AssetExtractionEngine` is safe to reuse across many `extract()` calls in one process (one `RuntimeManager.bootstrap()`, many `.extract()` calls) — this amortizes the one-time checkpoint-validation cost across a whole `creators.csv` run, matching how `ComfyUIClient` is constructed once in `main.py`'s `run_pipeline()` today... except Module 8's engine should be constructed once outside the per-creator loop, not once per creator (a difference implementers should note explicitly when eventually wiring `main.py`).
- **Memory efficiency**: every processor receives already-decoded `np.ndarray` images at `ASSET_EXTRACTION_MAX_IMAGE_DIMENSION_PX` ceiling (downscaled once by the orchestrator before dispatch, upscaled back only for the final crop/mask coordinates) so no processor holds more than one full-resolution copy in memory at a time; `cv2.imwrite`'s existing atomic-write path already streams to disk rather than buffering multiple outputs in RAM.
- **Model eviction**: `ModelBridge.run()`'s `finally` block always releases the GPU reservation (`GPUResourceManager.reserve()` already guarantees this via its own `try/finally`), returning the model to `CPU_CACHED` — Module 8 additionally calls `registry.transition(model_name, EVICTED)` after the **last** family that uses a given model in a given `extract()` call (tracked via a small "models remaining to use this run" set the orchestrator computes from `ASSET_EXTRACTION_FAMILY_ORDER` up front), so VRAM for e.g. SAM2 (used by both `objects` and `scene`) is only fully freed once, after `scene` completes — not evicted-and-reloaded between `objects` and `scene`.
- **CPU fallback / tiled processing**: honored per §13 point 5 without exception — SAM2's `cpu_tiled_processing` fallback tiles at `ASSET_EXTRACTION_SAM2_TILE_SIZE_PX`, keeping peak RAM bounded even on the 16GB system budget when VRAM is exhausted and a CPU path is taken.
- **Parallel execution where possible**: the three zero-model families (`typography`, `visual_properties`, `composition`) do not touch the GPU lock at all and are the one place true parallelism is safe — the orchestrator may run them concurrently via a small thread pool (they are CPU/OpenCV-bound, release the GIL during OpenCV calls) while a model-backed family is queued, but must never let a zero-model family and a model-backed family's *post-processing* (e.g. writing PNGs) race on the same file-system shard directory; `AssetWriter`'s atomic per-file writes make this safe by construction (no partial-file window for another thread to observe).

---

## 22. Testing Strategy

Mirrors `tests/test_visual_reference_engine.py` and `tests/test_thumbnail_intelligence.py` conventions: `pytest`, fabricated `np.ndarray` fixtures (no real images required for unit tests), `unittest.mock.MagicMock` for fake processors/model bridges, and the two existing `pytest.ini` markers (`integration`, `gpu`) reused — any test that needs a real vision-stack checkpoint loaded is marked `gpu` and skipped by default (`addopts = -m "not integration and not gpu"` already enforces this project-wide).

### Unit tests (`tests/asset_extraction_components/test_*.py`)
- One file per processor. Each processor is tested against synthetic `np.ndarray` inputs and, for model-backed processors, a `MagicMock` standing in for `IModelBridge` — verifying the processor calls `model_bridge.run("sam2", ...)` with the right arguments and correctly maps the mocked return value into its output dict shape. No real model weights are loaded in these tests.
- `test_asset_writer.py`: atomic-write success, corrupted/empty-array rejection (matches VRE's `AssetWriteError` test), JSON sidecar round-trip.
- `test_manifest_builder.py`: schema assembly from fabricated per-family dicts, `status` computation (`success`/`partial`/`error`) under every combination of family success/failure, and Pydantic validation-failure paths.
- `test_model_bridge.py`: reservation success, `VisionStackResourceError` → `AssetFamilyModelError` translation, fallback-policy application per `VisionModelFallback` value (mock the registry's config to each fallback enum value and assert the bridge's behavior differs correctly).

### Integration tests (`tests/test_asset_extraction_engine.py`)
- End-to-end `extract()` against a small real fixture image (following the existing `smoke_test_vre.py` precedent — Module 8 gets an analogous `smoke_test_aee.py`) with all processors faked except one representative model-backed processor, verifying the full manifest round-trips through `save_asset_manifest`/`load_asset_manifest`.
- Cache-hit / cache-miss / partial-cache-miss (one family's file deleted) scenarios, directly modeled on VRE's `_verify_cache` test suite.
- Resume test: simulate an interrupted run (call the orchestrator's internal per-family dispatch for only the first three `ASSET_EXTRACTION_FAMILY_ORDER` entries, kill it, then call `extract()` again) and assert only the remaining four families execute.
- Failure-injection tests: force one family's mocked processor to raise `AssetFamilyModelError` and assert `status == "partial"`, the failing family is `None` in the manifest, and every other family is still populated.

### Contract tests
- Validate `AssetExtractionManifest.model_dump_json()` against a JSON-schema-equivalent fixture, and validate that every `AssetFileRef.checksum` in a freshly-built manifest actually matches `sha256` of the file at `file_path` — this is the single test most likely to catch a writer/builder desync bug.

### GPU-marked tests (`-m gpu`, opt-in only, run on Afsar's RTX 4060)
- Real checkpoint load + one real inference call per new `vision_stack` wrapper (`sam2.py`, `birefnet.py`, `bisenet.py`, `depth_anything.py`, `teed.py`), verifying VRAM is released after the `reserve()` context exits (`torch.cuda.memory_allocated()` before/after comparison).
- A full real end-to-end `extract()` call against a real downloaded thumbnail, timed, to validate the performance budget in §21.

### What is explicitly NOT tested here
- Module 9's decision logic — out of scope by design (§3 Non-Goals).
- OCR/face-detection/object-detection/color/composition *accuracy* — that is Module 4's test suite's job; Module 8's tests assume `ThumbnailIntelligence` inputs are already correct and only test what Module 8 does with them.

---

## 23. Integration with Modules 1–7

- **Module 3 (`ThumbnailData`)**: Module 8 takes `thumbnail.thumbnail_path` as its `source_image_path` — the same validated, already-downloaded file Module 4 analyzed. Module 8 never re-downloads or re-validates thumbnail bytes; it trusts Module 3/4's prior validation (matches how Module 5 and Module 6 already trust Module 4's output without re-validating the source image).
- **Module 4 (`ThumbnailIntelligence`)**: the primary input. Every bbox, region, and score Module 8 needs to seed its processors comes from here — `intelligence.faces.faces`, `intelligence.objects`, `intelligence.ocr.text_regions`, `intelligence.colors`, `intelligence.composition`. If `intelligence.status == "error"`, `extract_assets()` raises `IntelligenceReportInvalidError` immediately (nothing meaningful can be extracted from a report that failed outright); `status == "partial"` is accepted with a logged warning, since a partially-degraded report (e.g. OCR failed but faces succeeded) still lets most families proceed — the `typography` family simply produces zero `TypographyAsset`s if `ocr.text_regions` is empty, which is a legitimate empty result, not an error.
- **Modules 5/6 (`RedesignSpecification`, `PromptPackage`)**: no direct dependency. Module 8 is a peer branch off Module 4, not downstream of Module 5/6 — this matches the brief's stated goal ("bridge between analysis and generation") and avoids coupling asset extraction to the deterministic redesign-direction logic, which may change independently.
- **Module 7 (Image Generation Engine)**: no direct dependency *today*. Module 7's own `ReferenceAssetResolver` (in `image_generator.py`) currently resolves reference assets via Module 6.5's `VisualReferenceManifest` contract. Once Module 8 exists, `ReferenceAssetResolver` becomes a natural future integration point to also consult `AssetExtractionManifest` — but that is a Module 7 change, out of scope for this document, and explicitly not made here per the standing instruction to never modify existing files additively without an explicit, justified reason. This document only notes the seam.
- **Module 6.5 (Visual Reference Engine)**: not deprecated or modified by this document. Module 8 reuses its `AssetWriter`'s atomic-write algorithm by composition (§11) but does not import, wrap, or replace `VisualReferenceEngine` itself — the two modules coexist; a future cleanup phase (explicitly out of scope here) could decide whether Module 6.5 is subsumed by Module 8's `scene`/`people` families, but that decision belongs to Afsar, not to this design.
- **`vision_stack` (V2.1)**: consumed as described in §13; the five new wrapper modules are the one piece of net-new work this document specifies outside Module 8's own package.

---

## 24. Integration with Modules 9–11

- **Module 9 (AI Decision Engine)**: reads `asset_manifest.json` only — never opens image pixels directly. Every `AssetFileRef.confidence_score` and every family's `extraction_status`/`extraction_notes` exists specifically so Module 9 can weight a KEEP/REMOVE/REPLACE/ENHANCE/ADD decision by extraction confidence (e.g. a `people[0].face` with high confidence → strong KEEP candidate; an `effects.glow_detected` with low confidence → Module 9 should not weight it heavily). Module 8 does not need to change if Module 9's decision schema (`keep.json`/etc.) changes, since Module 8 exposes raw material, not decisions.
- **Module 10 (Asset Composer)**: consumes the manifest's `AssetFileRef.file_path` values directly as compositing inputs (face crops, masks, background/foreground layers). Module 8's guarantee to Module 10 is that every path in the manifest is verified to exist and be non-empty at manifest-write time (§16) — Module 10 does not need its own existence checks.
- **Module 11 (Generation Pipeline / ComfyUI)**: consumes preserved assets (face crops, masks, depth/segmentation conditioning maps) as ControlNet/IPAdapter inputs, the same way Module 7's `ReferenceAssetResolver` already consumes Module 6.5's outputs today. Module 8's `scene.depth_map` and `scene.segmentation_map` are produced in exactly the pixel format (`cv2`-writable single/three-channel PNG) Module 7's `WorkflowBuilder` already expects for ControlNet inputs, so no format-adaptation layer is needed between Module 8 and Module 11.

---

## 25. Implementation Roadmap — Phases for Autonomous Coding Agents

Each phase is independently testable, produces code that imports and runs standalone, and does **not** require any later phase to exist or compile. Phases are ordered so that every dependency a phase needs already exists by the time that phase starts. Each phase ends with its own passing `pytest` suite before the next begins.

### Phase 1 — Foundations (no ML, no vision_stack dependency)
Scope: `asset_extraction_exceptions.py`; the full Module 8 block appended to `models.py` (§8); the full Module 8 block appended to `config.py` (§20); `asset_extraction_components/__init__.py` and `interfaces.py` (all ABCs, §10); folder scaffolding under `data/asset_extraction/`.
Testable via: model round-trip tests (construct every new Pydantic model with fabricated data, assert validators fire correctly on bad input), config constant sanity tests. No image processing code exists yet — this phase is pure contract-definition, exactly like how Module 6.5's models were defined before its processors.

### Phase 2 — Zero-model families (Typography, Visual Properties, Composition)
Scope: `typography_processor.py`, `visual_properties_processor.py`, `composition_processor.py`, plus Module 8's `asset_writer.py` (composed over `vre_components.asset_writer`, §11) and `manifest_builder.py` restricted to just these three families.
Testable via: fabricated `np.ndarray` + fabricated `TextRegion`/`ColorProfile`/`CompositionAnalysis` inputs (all constructible standalone from `models.py`, no Module 4 run required), full unit coverage, no GPU. This phase alone already delivers real, useful output (typography crops, extended palette, composition overlays) and can ship independently.

### Phase 3 — Object family (first vision_stack consumer)
Scope: `object_processor.py` + the **new** `vision_stack/sam2.py` inference wrapper (and `sam2_exceptions.py`), following the `grounding_dino.py` pattern exactly (§13 point 3). Requires Phase 1 (models/config) and reads but does not modify `vision_stack/registry.py`/`resources.py`.
Testable via: unit tests with a mocked `IModelBridge` (no real SAM2 needed) for `object_processor.py`'s hierarchy/crop/mask-assembly logic; separately, `gpu`-marked tests for the real `sam2.py` wrapper against a real checkpoint, run manually on Afsar's RTX 4060. The two test suites do not depend on each other.

### Phase 4 — Person family
Scope: `person_processor.py` + **new** `vision_stack/bisenet.py` (parsing: hair/clothing/accessories masks) + **new** `vision_stack/insightface_multi.py` (multi-face embeddings/landmarks, generalizing Module 4's existing single-pass InsightFace usage in `thumbnail_intelligence.py` without modifying that file).
Testable via: same split as Phase 3 — mocked-bridge unit tests for assembly/indexing logic against Module 4's `FaceAnalysis`, `gpu`-marked wrapper tests separately.
Depends on: Phase 1 only (does not depend on Phase 3's SAM2 wrapper).

### Phase 5 — Scene family (heaviest phase — four models)
Scope: `scene_processor.py` + **new** `vision_stack/birefnet.py`, `vision_stack/depth_anything.py`, `vision_stack/teed.py` (TEED reserved for a possible future edge-map asset, see Open Questions) + reuse of Phase 3's `grounding_dino.py`/`sam2.py` for the sky/ground open-vocabulary query.
Testable via: mocked-bridge unit tests for the assembly logic; `gpu`-marked wrapper tests per new model.
Depends on: Phase 3 (reuses its SAM2 wrapper) and Phase 1.

### Phase 6 — Effects family (lowest priority, no model)
Scope: `effects_processor.py` only — pure OpenCV heuristics.
Testable via: fabricated synthetic images with known injected glow/blur/shadow patterns, asserting the heuristic flags fire at expected confidence bands.
Depends on: Phase 1 only. Can be built in parallel with any other phase.

### Phase 7 — Orchestration
Scope: `asset_extraction_engine.py` (the `AssetExtractionEngine` class and module-level `extract_assets`/`save_asset_manifest`/`load_asset_manifest` functions), `model_bridge.py` (the real `IModelBridge` implementation over `vision_stack.RuntimeManager`), full caching (§16) and resume (§17) logic.
Testable via: the full integration/contract/resume/failure-injection suite from §22, using **all real processors from Phases 2–6** but a mocked `IModelBridge` for speed (no GPU needed for the default test run); a separate `gpu`-marked end-to-end smoke test (mirroring `smoke_test_vre.py`) exercises the real stack.
Depends on: Phases 1–6 all complete (this is the only phase that needs everything else, by design — it is the integration point, not a building block).

### Phase 8 — Pipeline wiring (explicitly out of scope for this design; listed for completeness)
Scope: adding a `# ── Module 8: extract reusable assets ──` block to `main.py`'s `run_pipeline()`, following the exact try/except/log/skip pattern already used for Modules 5–7 (§19 handling policy). Not designed further here, matching how Module 6 and Module 6.5 were both left unwired after their own design/implementation phases — wiring is a deliberate, separate decision Afsar makes per the project's established phase-separation discipline.

---

## 26. Risks & Open Questions

- **Pose estimation** is listed in the brief but no dedicated pose model exists in `vision_stack.yaml`. Options: (a) treat BiSeNet's body-part segmentation as a coarse pose proxy (centroid-per-part), (b) add a dedicated lightweight pose model (e.g. a MediaPipe- or RTMPose-class model) to `vision_stack.yaml` as an eleventh entry. This document assumes (a) for Phase 4 to avoid growing the vision-stack model roster, but flags (b) as the correct long-term answer if pose accuracy proves insufficient — a decision for Afsar, not made unilaterally here.
- **TEED** (edge detection) has no assigned asset family above. It is a legitimate candidate for a future `structural_edge_map` addition to `SceneAsset` (useful as a ControlNet Canny-alternative for Module 11) but was left out of the family list in this document because the brief's asset list does not name it explicitly; Phase 5 registers a wrapper for it regardless, since it's already declared in `vision_stack.yaml`, so it costs nothing to have ready.
- **Cache granularity below the family level** (per-asset rather than per-family, e.g. re-running only `face_02` if only that one crop's file went missing) was considered and rejected for v1 — the added bookkeeping complexity was judged not worth it against family-level granularity, which already bounds re-work to at most 1/7th of a full extraction. Revisit if profiling on real creator batches shows family-level re-work is a bottleneck.
- **Downscaling ceiling (`ASSET_EXTRACTION_MAX_IMAGE_DIMENSION_PX = 2048`)** trades a small amount of mask/crop fidelity for VRAM headroom on the RTX 4060. Source YouTube thumbnails are typically 1280×720, so this ceiling only bites on unusually large source images and is not expected to visibly degrade normal runs — but should be revisited if Module 3 starts fetching `maxresdefault` (1920×1080 or larger) thumbnails as a matter of course.
- **`data/asset_extraction/` disk footprint**: a fully-extracted shard (7 families × multiple PNGs each, at up to 2048px) is meaningfully larger per-creator than any prior module's output. No retention/pruning policy is specified here; this is a reasonable follow-up config addition (`ASSET_EXTRACTION_RETENTION_DAYS`) once real disk usage is measured, deliberately left out of v1 to avoid speculative complexity.
