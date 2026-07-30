# MODULE 10 — ASSET COMPOSER
## Architecture Design Document
### `thumbnail-ai` — Composition Workspace Preparation Stage

**Status:** Design only. Zero implementation code. Handoff artifact for Codex.
**Source of truth:** `github.com/poison-2-0-0-7/thumbnail-ai` (cloned and reviewed in full before writing this document).
**Naming note:** This document uses "Module 10 — Asset Composer" as requested. See §0 for a repo-verified numbering conflict that you should resolve on your own terms before handoff.

---

## §0. Repository Findings That Shape This Design (read first)

Before any architecture decision below, these are the load-bearing facts verified directly in the repository, not assumptions:

1. **Image generation is not a future stage — it's Module 7, and it's live in `main.py`.** The pipeline today is CSV Reader (1) → YouTube Metadata (2) → Thumbnail Downloader (3) → Thumbnail Intelligence (4) → Redesign Spec (5) → Prompt Compiler (6) → **Image Generation (7, `image_generator.py` + `comfyui_client.py`)**. Module 10 sits *before* Module 7, not before "future image generation."

2. **The repo's own roadmap already reserves "Module 10" for something else.** `docs/module5_architecture_v2.md` explicitly assigns Module 8 = Redesign QA, Module 9 = Outreach Copywriter, **Module 10 = Email Assembler**, Module 11 = Gmail Sender — the post-generation outreach half of the pipeline. You've chosen to handle this numbering collision yourself, so this document does not renumber anything; it just names the new component "Asset Composer" and leaves slotting it into your roadmap to you.

3. **Module 6.5 (Visual Reference Engine) already extracts almost every raw asset the brief asks for — but is not wired into `main.py`.** `VisualReferenceEngine.prepare_assets()` (in `modules/visual_reference_engine.py`) already produces, per `video_id`, under `data/visual_references/{video_id}/`: `creator_face.png` + `face_mask.png`, `foreground.png` + `background.png`, `object_crop.png` + `object_mask.png`, `depth_map.png`, `canny_map.png`, plus a validated `reference_manifest.json` (`VisualReferenceManifest`). This is the majority of the "background/foreground/people/objects/masks/depth" folder structure the brief describes — it already exists as a sibling module. **`main.py` never calls `VisualReferenceEngine`** — there is no wiring from Module 6 → 6.5 → 7 today. Asset Composer must call VRE directly (reusing it, not duplicating its CV logic) rather than re-deriving these assets.

4. **Module 7's own `ReferenceAssetResolver` is primitive and does not consume the VRE manifest at all.** It resolves only `source_thumbnail_path` and an optional Module 4 `analysis_path`. Its `ReferenceAssets.face_crop_path` field is declared but never populated — this is the exact gap Module 6.5 was designed to close, and it is still open. `WorkflowBuilder._slots()` only ever forwards `source_thumbnail_path` as a single string into the ComfyUI graph — no mask, depth, canny, or per-layer path is passed anywhere today.

5. **The ComfyUI workflow templates (`workflows/*.json`) have no nodes to consume compositing assets even if they were supplied.** Every niche template (`gaming.json`, `finance.json`, etc.) is a plain checkpoint → CLIPTextEncode(x2) → KSampler → VAEDecode → SaveImage graph. There is no `ControlNetLoader`, no `IPAdapter` node, no image-loader/mask node, despite `GenerationProfile.controlnet_enabled` / `ipadapter_enabled` existing as flags. **This is a real gap outside Module 10's scope** — Asset Composer can produce a fully correct `GenerationBundle`, but nothing downstream can consume it as ControlNet/IPAdapter conditioning until the workflow graphs and `WorkflowBuilder` are extended (a Module 7 change, not a Module 10 change; flagged here so it isn't silently assumed to already work).

6. **`RedesignSpecification.ObjectDirective.action` supports exactly three values: `include`, `remove`, `preserve`.** There is no `replace`, `enhance`, or `add` concept anywhere upstream. The brief's five-way KEEP/REMOVE/REPLACE/ENHANCE/ADD taxonomy does **not** have a 1:1 upstream data source. §11 below defines a grounded mapping instead of inventing new Module 5 fields (which would violate "do not modify previous modules").

7. **Everything upstream of Module 7 is explicitly, deliberately deterministic** — Module 5 and Module 6's docstrings both say "fully deterministic — no AI/LLM dependency, no network calls." Module 6.5 uses real CV models but its manifest-building is a deterministic function of those models' outputs plus caching by source-file hash. **Asset Composer must preserve this property**: no new ML inference, no randomness, no invented content — pure geometry/layout resolution over already-computed upstream data. This is also what "reuse existing functionality, never duplicate" cashes out to concretely: VRE already does segmentation/depth/face detection; Composer must not re-implement any of it.

8. **Canvas size is fixed today.** `GenerationParameters` defaults to `1280×720`, `16:9` (`DEFAULT_GENERATION_WIDTH`/`HEIGHT` in `config.py`). There is no per-video canvas override anywhere in the repo. Composer's canvas resolution is inherited from `PromptPackage.generation_parameters`, not invented independently.

9. **No shadow-synthesis, relighting, or inpainting/replacement generative model exists anywhere in the repo.** "Shadow consistency" and "REPLACE" in the brief's sense of swapping in a different generated object are not achievable by any existing component. Where the brief asks for these, this document either maps them to something the repo already does (see §11) or explicitly marks them **NOT SUPPORTED — no upstream capability**.

---

## §1. Overall Architecture

Asset Composer is a **deterministic, non-generative geometry/layout resolver**. It performs zero net-new CV or ML inference. It:

- Invokes the existing `VisualReferenceEngine` (Module 6.5) to obtain/verify the raw conditioning assets for a `video_id` (reused as-is, cache-aware, via its existing public API).
- Reads the existing `RedesignSpecification` (Module 5) for semantic decisions (subject treatment, object directives, text overlay, layout targets, color direction).
- Reads the existing `PromptPackage` (Module 6) for the canvas size, seed, and textual instructions the layers must stay consistent with.
- Resolves all of the above into **pixel-space, deterministic, hash-addressable layer geometry**: what each layer is, where it sits on the canvas, what mask conditions it, what z-order it occupies, and what decision (KEEP/REMOVE/REPLACE/ENHANCE/ADD, per the grounded mapping in §11) governs it.
- Persists this as a versioned **Composition Workspace** on disk (mirroring the `data/visual_references/{video_id}/` sharding convention already used by VRE) and a compact **Generation Bundle** — the flat, ComfyUI-consumable summary that a future-enhanced `WorkflowBuilder` will read.
- Never touches ComfyUI, never generates pixels, never calls Ollama/Gemini, never invents copy or new visual content beyond what Module 5 already decided.

```
Module 6.5 (VRE)            Module 5 (RedesignSpec)      Module 6 (PromptPackage)
 reference_manifest.json  +  object_directives,         +  canvas size, seed,
 creator_face/face_mask      subject_treatment,             textual instructions
 foreground/background       text_overlay,
 object_crop/object_mask     layout_direction
 depth_map/canny_map
        \                          |                          /
         \                         |                         /
          -------------  ASSET COMPOSER (Module 10)  --------
                                    |
                    CompositionWorkspace (workspace/ + composition.json)
                                    |
                          GenerationBundle (generation_bundle.json)
                                    |
                                    v
                Module 7 (image_generator.py) — ReferenceAssetResolver /
                WorkflowBuilder consume the bundle (additive change, §19)
```

---

## §2. Responsibilities

**In scope:**
- Determine, per detected element, which of the five decision buckets (§11) applies, using only existing upstream fields.
- Convert every normalized (`0.0–1.0`) `BoundingBox` from `RedesignSpecification` into absolute pixel coordinates against the canvas resolved from `PromptPackage.generation_parameters`.
- Build an ordered, z-indexed layer stack (background → foreground/person → objects → text → effects).
- Attach the correct VRE mask/depth/canny reference to each layer where one exists.
- Compute safe margins, negative-space compliance, and text-avoid-zone geometry in pixel space.
- Validate the resulting workspace (structural + referential integrity) before it is considered usable.
- Persist workspace + bundle atomically, support resume/cache-reuse keyed on upstream hashes.

**Explicitly out of scope (do not invent):**
- Any new image generation, inpainting, or relighting.
- Any new object detection/segmentation/depth estimation (all reused from VRE).
- Any new textual content ("suggested_text", new copy) — none of that exists in scope for this module per `docs/module5_architecture_v2.md`'s own assignment to a copywriting module.
- Modifying `RedesignSpecification`, `PromptPackage`, or VRE's internals.
- Building or modifying ComfyUI graph JSON (that remains `WorkflowLibrary`/`WorkflowBuilder`'s job; §19 covers the minimal additive touchpoint).

---

## §3. Folder Structure

```
modules/
  composition_engine.py                 # AssetComposer orchestrator (public API surface)
  composition_components/
    __init__.py
    interfaces.py                       # ABCs, mirrors vre_components/interfaces.py pattern
    decision_resolver.py                 # KEEP/REMOVE/REPLACE/ENHANCE/ADD mapping (§11)
    placement_engine.py                  # normalized bbox -> pixel geometry, alignment, safe margins
    transform_engine.py                  # scale/crop/translate resolution per layer
    layer_manager.py                     # z-ordering, layer hierarchy, grouping
    mask_manager.py                      # binds VRE mask/depth/canny paths to layers, feathering config
    asset_registry.py                    # indexes VRE assets + checksums, resolves by role
    workspace_manager.py                 # directory sharding, atomic persistence, resume
    composition_validator.py             # structural + referential validation, hard-gate checks
    generation_bundle_builder.py         # flattens workspace -> GenerationBundle
    metadata_builder.py                  # WorkspaceMetadata + WorkspaceStatistics assembly
  composition_exceptions.py              # typed exception hierarchy (mirrors module7_exceptions.py)

tests/
  test_composition_engine.py
  test_composition_components/
    test_decision_resolver.py
    test_placement_engine.py
    test_transform_engine.py
    test_layer_manager.py
    test_mask_manager.py
    test_asset_registry.py
    test_workspace_manager.py
    test_composition_validator.py
    test_generation_bundle_builder.py
    test_metadata_builder.py
```

This mirrors the VRE precedent exactly (`visual_reference_engine.py` orchestrator + `vre_components/` package + `vre_exceptions.py` + one flat top-level test file plus, here, a component-test subfolder since Module 10 has more components than VRE's four).

---

## §4. Data Flow

```
video_id
  │
  ├─► PromptPackageLoader-style loader reads data/prompt_packages/{video_id}.json  (Module 6 output, reused loader pattern)
  ├─► redesign spec loader reads data/redesign_specs/{video_id}.json                (Module 5 output)
  ├─► VisualReferenceEngine(storage_root=VRE_STORAGE_ROOT).prepare_assets(video_id, thumbnail_path)
  │        → VisualReferenceManifest (cache-aware; VRE's own hash check applies unchanged)
  │
  ▼
AssetComposer.compose_workspace(video_id)
  │
  ├─ 1. AssetRegistry.index(manifest)              → asset_id -> AssetMetadata lookup
  ├─ 2. DecisionResolver.resolve(redesign_spec)     → per-element LayerDecision (§11)
  ├─ 3. PlacementEngine.place(redesign_spec, canvas)→ pixel-space AssetPlacement per layer
  ├─ 4. TransformEngine.resolve(placement, decision)→ LayerTransform (scale/crop/translate)
  ├─ 5. MaskManager.bind(asset_registry, layer)     → MaskReference per layer where applicable
  ├─ 6. LayerManager.order(layers)                  → z-indexed CompositionLayer list + LayerGroups
  ├─ 7. MetadataBuilder.build(...)                  → WorkspaceMetadata + WorkspaceStatistics
  ├─ 8. CompositionValidator.validate(workspace)     → structural/referential pass or typed error
  └─ 9. WorkspaceManager.persist(workspace)          → atomic write of workspace/ + composition.json
       │
       ▼
GenerationBundleBuilder.build_generation_bundle(workspace) → generation_bundle.json
```

Every step above is a pure function of its inputs plus the constants in `config.py` — no hidden state, no network calls, no ML inference performed by Module 10 itself.

---

## §5. Component Diagram (textual)

```
                         ┌────────────────────────┐
                         │   AssetComposer         │  (composition_engine.py)
                         │   public API surface    │
                         └───────────┬─────────────┘
             ┌────────────┬──────────┼──────────┬─────────────┬───────────────┐
             ▼            ▼          ▼          ▼             ▼               ▼
     AssetRegistry  DecisionResolver PlacementEngine TransformEngine  MaskManager  LayerManager
             │            │          │          │             │               │
             └─────┬──────┴────┬─────┴────┬─────┴──────┬──────┴───────┬───────┘
                    ▼           ▼          ▼            ▼              ▼
              MetadataBuilder  CompositionValidator  WorkspaceManager  GenerationBundleBuilder
                                                          │
                                                 data/composition_workspaces/{video_id}/
```

Each component implements one ABC from `composition_components/interfaces.py`, following the exact DI pattern already used by `vre_components/interfaces.py` (`IFaceProcessor`, `ISegmentationProcessor`, etc.) — every collaborator is injectable into `AssetComposer.__init__` with a concrete default, enabling the same fake-object test strategy already used in `tests/test_visual_reference_engine.py`.

---

## §6. Public APIs

Mirroring the verb/noun convention already established by `visual_reference_engine.py` (`prepare_assets`, `clean_assets`) and `redesign_spec_engine.py`/`prompt_compiler.py` (`build_*`, `save_*`):

```python
class AssetComposer:
    def compose_workspace(self, video_id: str, options: dict | None = None) -> CompositionWorkspace: ...
    def save_workspace(self, workspace: CompositionWorkspace) -> Path: ...
    def load_workspace(self, video_id: str) -> CompositionWorkspace: ...
    def validate_workspace(self, workspace: CompositionWorkspace) -> CompositionWorkspace: ...
    def resume_workspace(self, video_id: str) -> CompositionWorkspace | None: ...
    def build_generation_bundle(self, workspace: CompositionWorkspace) -> GenerationBundle: ...
    def prepare_generation_workspace(self, video_id: str, options: dict | None = None) -> GenerationBundle:
        """Convenience: compose_workspace -> validate_workspace -> save_workspace -> build_generation_bundle."""
    def clean_workspace(self, video_id: str) -> bool:
        """Mirrors VisualReferenceEngine.clean_assets for symmetry."""
```

---

## §7. Internal APIs (per component)

```python
# asset_registry.py
class AssetRegistry:
    def index(self, manifest: VisualReferenceManifest) -> dict[str, AssetMetadata]: ...
    def resolve(self, role: str) -> AssetMetadata | None: ...
    def verify_integrity(self) -> list[str]:  # returns list of missing/corrupt asset_ids

# decision_resolver.py
class DecisionResolver:
    def resolve(self, spec: RedesignSpecification) -> list[tuple[str, LayerRole, LayerDecision, str]]:
        """Returns (element_key, role, decision, rationale) tuples per §11 mapping."""

# placement_engine.py
class PlacementEngine:
    def place(self, spec: RedesignSpecification, canvas: CanvasTransform) -> dict[str, VisualBoundingBox]:
        """Converts every normalized BoundingBox on the spec into absolute pixel VisualBoundingBox."""
    def resolve_focal_zone(self, spec: RedesignSpecification, canvas: CanvasTransform) -> VisualBoundingBox | None: ...
    def resolve_text_zones(self, spec: RedesignSpecification, canvas: CanvasTransform) -> TextPlacement: ...

# transform_engine.py
class TransformEngine:
    def resolve(self, pixel_bbox: VisualBoundingBox, decision: LayerDecision, crop_tighter: bool) -> LayerTransform: ...

# mask_manager.py
class MaskManager:
    def bind(self, registry: AssetRegistry, role: LayerRole) -> MaskReference | None: ...
    def feather(self, mask_ref: MaskReference, feather_px: int) -> MaskReference: ...

# layer_manager.py
class LayerManager:
    def order(self, layers: list[CompositionLayer]) -> list[CompositionLayer]:
        """Deterministic z-order: background(0) < foreground/person(10) < objects(20) < text(30) < effects(40)."""
    def group(self, layers: list[CompositionLayer]) -> list[LayerGroup]: ...

# composition_validator.py
class CompositionValidator:
    def validate(self, workspace: CompositionWorkspace) -> list[str]:  # empty list == valid
        """Checks: every referenced asset path exists + checksum matches, z-order has no
        collisions, canvas bounds respected, text zone does not overlap kept subject bbox
        (already guaranteed upstream by Module 5 but re-verified defensively), statistics sum
        correctly, at minimum one KEEP layer exists."""

# generation_bundle_builder.py
class GenerationBundleBuilder:
    def build_generation_bundle(self, workspace: CompositionWorkspace) -> GenerationBundle: ...

# metadata_builder.py
class MetadataBuilder:
    def build(self, video_id, manifest, spec, package) -> WorkspaceMetadata: ...
    def statistics(self, layers: list[CompositionLayer]) -> WorkspaceStatistics: ...

# workspace_manager.py
class WorkspaceManager:
    def target_dir(self, video_id: str) -> Path: ...
    def persist(self, workspace: CompositionWorkspace) -> Path:  # temp-file-then-replace, like ArtifactWriter
    def load(self, video_id: str) -> CompositionWorkspace: ...
    def resume(self, video_id: str, expected_hashes: dict[str, str]) -> CompositionWorkspace | None: ...
    def purge(self, video_id: str) -> bool: ...
```

---

## §8. Data Models

All new models are **appended** to `modules/models.py` (never modifying existing classes), following the exact precedent set when Module 6's models were added — `frozen=True` Pydantic models, `field_validator`s for non-empty text/hash fields matching the SHA-256 validators already used by `AssetMetadata`/`VisualReferenceManifest`.

```python
# ---------------------------------------------------------------------------
# Module 10 - Asset Composer
# ---------------------------------------------------------------------------

class LayerDecision(str, Enum):
    KEEP = "keep"
    REMOVE = "remove"
    REPLACE = "replace"
    ENHANCE = "enhance"
    ADD = "add"


class LayerRole(str, Enum):
    BACKGROUND = "background"
    FOREGROUND = "foreground"
    PERSON = "person"
    OBJECT = "object"
    TEXT = "text"
    EFFECT = "effect"


class CanvasTransform(BaseModel):
    model_config = ConfigDict(frozen=True)
    width: int
    height: int
    aspect_ratio: str


class LayerTransform(BaseModel):
    model_config = ConfigDict(frozen=True)
    translate_x: int = 0
    translate_y: int = 0
    scale_x: float = 1.0
    scale_y: float = 1.0
    crop_box: Optional[VisualBoundingBox] = None


class MaskReference(BaseModel):
    model_config = ConfigDict(frozen=True)
    mask_path: str
    mask_checksum: str
    feather_px: int = 0
    source: Literal["vre"] = "vre"   # only source that exists today; see §0.3


class PlacementConstraints(BaseModel):
    model_config = ConfigDict(frozen=True)
    safe_margin_px: int = 0
    avoid_zones_px: list[VisualBoundingBox] = []
    focal_zone_px: Optional[VisualBoundingBox] = None


class TextPlacement(BaseModel):
    model_config = ConfigDict(frozen=True)
    include_text: bool = False
    placement_zone_px: Optional[VisualBoundingBox] = None
    avoid_zones_px: list[VisualBoundingBox] = []


class LightingAdjustment(BaseModel):
    model_config = ConfigDict(frozen=True)
    target_brightness: float
    target_contrast: float
    target_saturation: float
    warm_or_cool: Literal["warm", "cool", "neutral"]


class AssetPlacement(BaseModel):
    model_config = ConfigDict(frozen=True)
    asset_id: str
    role: LayerRole
    decision: LayerDecision
    source_path: Optional[str] = None
    mask: Optional[MaskReference] = None
    transform: LayerTransform
    z_index: int
    rationale: str = ""


class CompositionLayer(BaseModel):
    model_config = ConfigDict(frozen=True)
    layer_id: str
    placement: AssetPlacement
    depth_hint_path: Optional[str] = None   # points at VRE depth_map.png when relevant


class LayerGroup(BaseModel):
    model_config = ConfigDict(frozen=True)
    group_id: str
    role: LayerRole
    layer_ids: list[str]


class WorkspaceStatistics(BaseModel):
    model_config = ConfigDict(frozen=True)
    total_layers: int
    kept: int = 0
    removed: int = 0
    replaced: int = 0
    enhanced: int = 0
    added: int = 0


class WorkspaceMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    video_id: str
    created_at: str
    vre_source_hash: str
    redesign_spec_hash: str
    prompt_package_hash: str
    engine_version: str


class CompositionWorkspace(BaseModel):
    model_config = ConfigDict(frozen=True)
    video_id: str
    canvas: CanvasTransform
    layers: list[CompositionLayer]
    groups: list[LayerGroup]
    text_placement: TextPlacement
    lighting: LightingAdjustment
    constraints: PlacementConstraints
    statistics: WorkspaceStatistics
    metadata: WorkspaceMetadata
    status: Literal["success", "partial", "error"] = "success"
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


class GenerationBundle(BaseModel):
    model_config = ConfigDict(frozen=True)
    video_id: str
    canvas: CanvasTransform
    reference_image_paths: dict[str, str] = {}   # role/asset_id -> file path
    mask_paths: dict[str, str] = {}
    depth_path: Optional[str] = None
    canny_path: Optional[str] = None
    layer_order: list[str] = []                  # z-ordered layer_ids
    workspace_hash: str
    prompt_package_hash: str
    generated_at: str
```

`BoundingBox` (normalized, Module 4/5) and `VisualBoundingBox` (absolute pixel, Module 6.5) both already exist — Composer reuses `VisualBoundingBox` for all pixel-space geometry rather than introducing a third bounding-box type.

---

## §9. Workspace Layout

Sharded exactly like VRE's `data/visual_references/{video_id}/`, under a sibling root:

```
data/composition_workspaces/{video_id}/
  background/            # symlink-or-copy pointer to VRE background.png (see note below)
  foreground/            # -> VRE foreground.png
  people/                # -> VRE creator_face.png
  objects/               # -> VRE object_crop.png (repeated per kept object if VRE is extended to multi-object later — today VRE emits one object_crop)
  text/                  # empty placeholder dir unless text_placement.include_text
  effects/               # empty today — no effects pipeline exists in the repo (flagged §0.9)
  masks/                 # -> VRE face_mask.png, object_mask.png
  depth/                 # -> VRE depth_map.png
  lighting/              # metadata-only; no separate lighting asset exists
  layers/                # per-layer JSON snippets (one file per CompositionLayer, for human inspection)
  composition.json        # CompositionWorkspace, serialized
  generation_bundle.json  # GenerationBundle, serialized
  metadata.json           # WorkspaceMetadata + WorkspaceStatistics, serialized
  workspace_manifest.json # top-level index: hashes, versions, completeness flags (used for resume)
```

**Important, repo-grounded caveat:** VRE currently produces **one** file per asset type (one `object_crop.png`, one `object_mask.png`) — not one per detected object. If your `RedesignSpecification` has multiple `object_directives`, they currently share the same single VRE object crop/mask. Composer's `AssetRegistry` must not assume per-object VRE assets exist; it can only bind what VRE actually emits today. Multi-object per-asset extraction is a VRE-level gap (Module 6.5), not something Module 10 can retroactively invent.

To avoid duplicating multi-megabyte PNGs, `background/`, `foreground/`, `people/`, `objects/`, `masks/`, `depth/` should contain **path references recorded in `composition.json`/`generation_bundle.json` pointing directly at the existing `data/visual_references/{video_id}/...` files**, not physical copies — this satisfies "reuse, never duplicate" at the filesystem level too. The subfolders exist for human/Codex readability and for the `layers/` per-layer JSON breakdown, not as a second copy of binary assets.

---

## §10. Layer Management

- **Hierarchy:** `LayerGroup` collects layers by `LayerRole`; a workspace has at most one `background` group, one `foreground`/`person` group, N `object` groups (bounded by however many VRE emits, per §9's caveat), one `text` group when `include_text`, and an empty `effects` group (reserved, unused — §0.9).
- **Z-ordering:** fixed, deterministic bands — `background=0`, `foreground/person=10`, `object=20`, `text=30`, `effect=40`. Within a band, layers are ordered by ascending `asset_id` for full determinism (no floating z-values, no randomness).
- **Depth ordering:** `depth_hint_path` on each layer points at the existing VRE `depth_map.png` for informational/consistency purposes (e.g., a future ControlNet-depth node could read it) — Composer does not interpret the depth map's pixel values itself; that would be new CV inference, out of scope.

---

## §11. Composition Pipeline — the KEEP/REMOVE/REPLACE/ENHANCE/ADD Mapping

This is the section where the brief's requested taxonomy is reconciled against §0.6's finding. Every rule below cites the exact upstream field it derives from — nothing here is invented.

| Decision | Grounded in | Rule |
|---|---|---|
| **KEEP** | `RedesignSpecification.subject_treatment.has_subject`, `ObjectDirective.action in {"preserve", "include"}` | The subject/person layer (VRE `creator_face`+`face_mask`) is KEPT whenever `has_subject=True`. Each object directive with `action="preserve"` or `"include"` becomes a KEPT object layer. |
| **REMOVE** | `ObjectDirective.action == "remove"` | The corresponding object layer is excluded from the generation bundle entirely — its VRE asset is indexed but never emitted into `reference_image_paths`. |
| **REPLACE** | Project's own documented architecture (`docs/IMAGE_GENERATION_ARCHITECTURE.md`'s "extract real elements + AI-generated background only" approach; `BackgroundCompositor` in `image_generator.py`) | The **background layer is always REPLACE**, by design — real background pixels are discarded and only VRE's `background.png`/`depth_map`/`canny_map` are kept as *structural* conditioning references for an AI-generated replacement. No other element is ever marked REPLACE; there is no upstream signal for "replace this specific object with a different one." |
| **ENHANCE** | Existing Module 7 stages `FaceRestorationStage`, `UpscaleStage` | The KEPT subject/person layer is flagged `enhancement_requested=True` (carried as part of its `rationale`/statistics, not a new pixel operation) so Module 7's *already-existing* restoration/upscale stages know to run on it. Composer does not perform enhancement itself — it only flags intent for Module 7 to act on with capability it already has. |
| **ADD** | `RedesignSpecification.text_overlay.include_text` | The **only** ADD-able content the deterministic pipeline permits is the text-placement layer, and even then it is placement-only geometry (`TextOverlaySpec` "never contains new copy" per its own docstring). No new visual object can be ADD-ed — there is no generative or asset-sourcing capability in the repo for that. |

**Canvas positioning, transforms, scaling, cropping, alignment, safe margins:**
- Canvas = `PromptPackage.generation_parameters.{width,height}` (currently always 1280×720).
- Every normalized `BoundingBox` (subject `target_bbox`, `focal_zone`, text `placement_zone`/`avoid_zones`) is converted to pixel space: `x_px = x_min * width`, etc. — deterministic arithmetic, no rounding ambiguity beyond standard `int()` truncation (documented, testable).
- `subject_treatment.crop_tighter=True` maps to a non-unit `scale_x/scale_y` in the subject's `LayerTransform`, per `TransformEngine`.
- Safe margins: new configurable constant (`COMPOSITION_SAFE_MARGIN_PX`, §16) applied uniformly around the canvas edge when placing object/text layers — no upstream field defines this today, so it is introduced as a new, clearly-labeled, deterministic config constant (consistent with how every prior module has added its own thresholds to `config.py`), not as invented *architecture*.
- Negative space / clutter compliance: **not re-derived** — `layout_direction.target_negative_space_ratio`/`target_clutter_score` are carried through as metadata for validation only; Composer does not attempt to compute actual achieved negative space (that requires pixel analysis of a not-yet-generated image, which is Module 7/8's job, not Module 10's).

**Blending, color harmony, lighting/perspective consistency, shadow consistency:**
- `LightingAdjustment` carries `color_direction` verbatim from Module 5 into the workspace for downstream consistency — Composer performs no color-space computation itself.
- Perspective consistency is addressed only by passing through VRE's existing `depth_map`/`canny_map` references (§10) — Composer does not compute perspective itself.
- **Shadow consistency: NOT SUPPORTED.** No shadow-synthesis capability exists anywhere in the repo. This document does not invent one; it is flagged as an open gap for a future module if pursued.

**Determinism:** the entire pipeline is a pure function of `(VisualReferenceManifest, RedesignSpecification, PromptPackage)`. Given identical inputs, `compose_workspace()` produces byte-identical JSON (verified the same way Module 7 verifies reproducibility — via `canonical_json_hash`, reused, not reimplemented).

---

## §12. Asset Management

- **Asset Registry:** indexes every `AssetMetadata` entry from the VRE manifest (`asset_type`, `file_path`, `checksum`, `resolution`, `confidence_score` — already-existing fields), keyed by role.
- **Asset Validation:** `AssetRegistry.verify_integrity()` re-checks file existence + re-hashes each referenced asset (SHA-256, matching `AssetMetadata.checksum`'s existing validator convention) before a workspace is considered valid — defends against a VRE shard being partially deleted between VRE's run and Composer's run.
- **Versioning:** workspace `metadata.json` stores `vre_source_hash` (VRE's own `source_hash`), `redesign_spec_hash`, and `prompt_package_hash` (reusing `canonical_json_hash` exactly as Module 7 does for its own hash). A workspace is only considered current if all three still match on resume.

---

## §13. Caching

Reuses VRE's exact cache-verification pattern (`_verify_cache` in `visual_reference_engine.py`):
1. On `compose_workspace(video_id)`, check for an existing `workspace_manifest.json`.
2. If present, compare its stored `vre_source_hash`/`redesign_spec_hash`/`prompt_package_hash` against freshly computed hashes of the current upstream artifacts.
3. If all three match **and** `AssetRegistry.verify_integrity()` passes, return the cached workspace unchanged (mark `cached_hit=True` in metadata, same convention VRE uses).
4. Otherwise, recompute from scratch — no partial/incremental recomposition, matching VRE's own current all-or-nothing rebuild strategy (VRE doesn't do incremental recomputation either; Composer shouldn't pretend to do better than its own upstream dependency).

`COMPOSITION_CACHE_ENABLED: bool = True` (config default), overridable per-call via the same `options: dict` pattern VRE already uses.

---

## §14. Resume Strategy

- `resume_workspace(video_id)` reads `workspace_manifest.json` only (cheap, no full workspace deserialization) to check completeness flags (`layers_written`, `bundle_written`, `validated`) before deciding whether a full reload is needed.
- If `workspace_manifest.json` exists but `composition.json` is missing/corrupt (interrupted write), `resume_workspace` returns `None` and the caller (`prepare_generation_workspace`) transparently falls back to a full `compose_workspace` rebuild — no manual repair step required, matching the "Resume Strategy" pattern already established by VRE's cache-or-recompute logic and Module 7's temp-file-then-replace writes.

---

## §15. Logging

New Loguru sink following the exact one-file-per-module convention already used by every existing module (`MODULE1_LOG_PATH` through `MODULE65_LOG_PATH`):

```python
# config.py additions (append only)
MODULE10_LOG_PATH: Path = LOG_DIR / "module10.log"   # rename freely once you settle numbering
```

`composition_engine.py` calls a `_configure_logger()` identical in shape to the one in `visual_reference_engine.py` (`rotation="10 MB"`, `retention="30 days"`, `enqueue=True`, same `_LOG_FORMAT` string literal already duplicated per-module in this codebase). Every component logs at `INFO` for lifecycle events (workspace composed/cached/persisted) and `DEBUG` for per-layer decisions, matching VRE's granularity.

---

## §16. Error Handling

New file `modules/composition_exceptions.py`, mirroring `vre_exceptions.py`/`module7_exceptions.py`'s flat-hierarchy-off-one-base convention:

```python
class CompositionBaseError(Exception):
    """Base exception for every Asset Composer failure."""

class CompositionInputInvalidError(CompositionBaseError):
    """Raised when the upstream RedesignSpecification or PromptPackage is unusable."""
    # named to avoid collision with prompt_compiler.py's existing InvalidRedesignSpecError

class AssetRegistryError(CompositionBaseError):
    """Raised when a referenced VRE asset is missing, unreadable, or checksum-mismatched."""

class LayerPlacementError(CompositionBaseError):
    """Raised when geometry resolution produces an invalid or out-of-canvas placement."""

class MaskResolutionError(CompositionBaseError):
    """Raised when a required mask cannot be bound to its layer."""

class WorkspaceValidationError(CompositionBaseError):
    """Raised when CompositionValidator finds structural or referential defects."""

class WorkspacePersistenceError(CompositionBaseError):
    """Raised when the workspace cannot be atomically written to disk."""

class GenerationBundleError(CompositionBaseError):
    """Raised when a validated workspace cannot be flattened into a GenerationBundle."""
```

New config constants needed (append-only, `config.py`):
```python
MODULE10_LOG_PATH: Path = LOG_DIR / "module10.log"
COMPOSITION_WORKSPACE_ROOT: Path = PROJECT_ROOT / "data" / "composition_workspaces"
COMPOSITION_ENGINE_VERSION: str = "1.0.0"
COMPOSITION_CACHE_ENABLED: bool = True
COMPOSITION_SAFE_MARGIN_PX: int = 24
COMPOSITION_TEXT_FEATHER_PX: int = 6
COMPOSITION_MANIFEST_FILENAME: str = "workspace_manifest.json"
```

---

## §17. Testing Strategy

Follows the existing `pytest.ini` markers exactly (`integration`, `gpu`, both excluded by default via `addopts = -m "not integration and not gpu"`). Because Composer performs **no ML inference and no ComfyUI I/O**, essentially all of its tests run unmarked (fast, fully offline) — this is a meaningful simplification versus Module 7/VRE's test suites, which must fake CV models.

- **Unit tests:** one file per component (per §3), following the fake-collaborator pattern already used in `tests/test_visual_reference_engine.py` (`FakeFaceProcessor`, etc.) — here, fakes are trivial since Composer's collaborators are pure functions over Pydantic models, not CV models.
- **Integration tests:** `test_composition_engine.py` exercises the full `compose_workspace` → `validate_workspace` → `build_generation_bundle` chain against real `RedesignSpecification`/`PromptPackage`/`VisualReferenceManifest` fixtures built from the actual sample JSON already checked into `data/redesign_specs/`, `data/prompt_packages/`, and `data/visual_references/smoke_test/` — no `integration` marker needed since none of this touches a live external service.
- **Regression tests:** hash-stability tests asserting `compose_workspace(video_id)` called twice on unchanged inputs produces byte-identical `composition.json` (determinism guarantee from §11).
- **Validation tests:** deliberately corrupt a VRE asset checksum / delete a referenced file / feed an out-of-canvas bbox, assert the correct typed exception from §16 is raised.
- **Performance tests:** marked `gpu`-adjacent only if you choose to benchmark against very large source thumbnails; otherwise these are plain wall-clock assertions since no GPU is involved (§18).
- **Mock strategy:** no ComfyUI, no InsightFace/segmentation models required — Composer's tests never need `-m gpu`. This is a genuine architectural win: Module 10 can be fully covered by CI without any GPU present, unlike Modules 4, 6.5, and 7.

---

## §18. Performance Optimizations

Because Asset Composer performs no ML inference:
- **No GPU requirement at all** — this differs materially from Modules 4/6.5/7, all of which need CUDA/models. Composer is pure Python + Pydantic + filesystem I/O.
- **Large thumbnails/masks:** Composer never decodes pixel buffers itself (no `cv2.imread`/`PIL.Image.open` in the hot path) — it only reads file paths, sizes (from `AssetMetadata.resolution`, already computed by VRE), and checksums. This avoids the memory pressure of loading multi-megapixel arrays a second time.
- **Batch composition:** `compose_workspace` is naturally parallelizable across `video_id`s (no shared mutable state beyond the filesystem target directory, same threading-safety property VRE already provides via its `self._lock`) — a caller can run one `AssetComposer` per worker thread/process safely.
- **Memory reuse:** by referencing VRE's existing files instead of copying (§9), Composer avoids doubling on-disk storage for large PNGs, which matters directly for your RTX 4060 laptop's limited local disk/VRAM budget.
- **Streaming/incremental composition:** not applicable — workspaces are small JSON + path references, not multi-gigabyte payloads; there's nothing to stream.

---

## §19. Integration Strategy

**`main.py` change (additive, one new step):** insert Asset Composer between the existing Module 6 block and the existing `_run_module7_generation` call:

```python
# after: prompt_package = compile_prompt_package(redesign_spec); save_prompt_package(...)
generation_bundle = AssetComposer().prepare_generation_workspace(prompt_package.video_id)
# then: _run_module7_generation(prompt_package, ..., generation_bundle=generation_bundle)
```

**Module 7 touchpoints (minimal, additive — flagged, not performed by this document):**
- `ReferenceAssets` dataclass gains new optional fields (`background_path`, `foreground_path`, `mask_path`, `depth_path`, `canny_path`) populated from the `GenerationBundle` instead of left `None` forever, closing the "declared but never filled" gap from §0.4.
- `WorkflowBuilder._slots()` gains new optional placeholder keys mirroring those fields, **but** — per §0.5 — actually consuming them requires the ComfyUI workflow JSON templates to gain ControlNet/IPAdapter/image-loader nodes that do not exist today. Wiring the bundle's paths into `WorkflowBuilder` without those graph nodes is a no-op at generation time; both changes must land together, and both are explicitly outside Module 10's own scope. This document surfaces the dependency; it does not design the ComfyUI graph changes.
- No existing public API is modified — both changes are additive optional fields with safe `None` defaults, preserving every existing caller and test.

**Module 6.5 touchpoint:** since VRE is not currently called from `main.py` (§0.3), Asset Composer becomes VRE's first real caller in the production pipeline. No VRE code changes are required — `AssetComposer` simply constructs a `VisualReferenceEngine()` and calls `prepare_assets()`, exactly as `smoke_test_vre.py` already demonstrates is a valid call pattern.

**No other module is modified or redesigned.**

---

## §20. Phase-by-Phase Implementation Roadmap

### Phase 1 — Data Models & Exceptions
- **Purpose:** Establish the typed contract before any logic exists.
- **Files to create:** `modules/composition_exceptions.py`.
- **Files to modify:** `modules/models.py` (append §8 models only), `modules/config.py` (append §16 constants only).
- **Dependencies:** none.
- **Expected outputs:** importable models/exceptions, zero behavior.
- **Tests:** Pydantic validation tests only (bad hash, negative coordinates, etc., matching existing `AssetMetadata`/`VisualBoundingBox` validator test style).
- **Completion criteria:** `python -c "from models import CompositionWorkspace, GenerationBundle"` succeeds; full existing test suite still passes with zero regressions (same bar Module 6 held: 0 regressions).

### Phase 2 — Asset Registry & Decision Resolver
- **Purpose:** Wire read-only access to VRE + Module 5 outputs and implement the §11 mapping.
- **Files to create:** `modules/composition_components/{__init__.py,interfaces.py,asset_registry.py,decision_resolver.py}`.
- **Dependencies:** Phase 1.
- **Expected outputs:** given a `VisualReferenceManifest` + `RedesignSpecification` fixture, produce correct `(element_key, role, decision, rationale)` tuples for every case in the §11 table.
- **Tests:** `test_asset_registry.py`, `test_decision_resolver.py` — including the REMOVE and REPLACE edge cases and the "no multi-object VRE asset" caveat from §9.
- **Validation:** exhaustive test of all three `ObjectDirective.action` values plus `has_subject=False` and `include_text=False` paths.
- **Completion criteria:** 100% branch coverage on the decision table.

### Phase 3 — Placement, Transform, Mask, Layer Management
- **Purpose:** Geometry resolution.
- **Files to create:** `modules/composition_components/{placement_engine.py,transform_engine.py,mask_manager.py,layer_manager.py}`.
- **Dependencies:** Phase 2.
- **Expected outputs:** pixel-space `AssetPlacement`/`LayerTransform`/`MaskReference`/ordered `CompositionLayer` list for a fixture spec at 1280×720.
- **Tests:** bbox-conversion arithmetic (normalized → pixel, verified against hand-computed expected values from the real sample in `data/redesign_specs/I-bnBd5lCew.json`), z-order determinism test (same input twice → identical order).
- **Completion criteria:** deterministic, reproducible layer stack for the real sample fixture.

### Phase 4 — Metadata, Validation, Workspace Persistence
- **Purpose:** Make the workspace real, cached, and resumable.
- **Files to create:** `modules/composition_components/{metadata_builder.py,composition_validator.py,workspace_manager.py}`.
- **Dependencies:** Phase 3.
- **Expected outputs:** `data/composition_workspaces/{video_id}/` populated on disk with all files from §9; second call on unchanged inputs hits cache.
- **Tests:** cache-hit/cache-miss tests (mirroring `test_visual_reference_engine.py`'s cache tests), atomic-write interruption simulation (temp-file left behind → resume correctly falls back), validator rejects a deliberately corrupted fixture.
- **Completion criteria:** resume/cache behavior matches VRE's own test coverage pattern 1:1.

### Phase 5 — Generation Bundle & Orchestrator
- **Purpose:** Ship the public API.
- **Files to create:** `modules/composition_engine.py`, `modules/composition_components/generation_bundle_builder.py`.
- **Dependencies:** Phases 1–4.
- **Expected outputs:** `AssetComposer().prepare_generation_workspace(video_id)` returns a valid `GenerationBundle` end-to-end against real fixtures.
- **Tests:** `tests/test_composition_engine.py` full-pipeline test; regression hash-stability test (§17).
- **Completion criteria:** full pipeline runs offline, no GPU/ComfyUI marker needed, 0 regressions in the existing suite.

### Phase 6 — `main.py` Wiring (additive only)
- **Purpose:** Make Module 10 reachable from the real pipeline.
- **Files to modify:** `main.py` only (insert the one call shown in §19; no existing lines removed).
- **Dependencies:** Phase 5.
- **Expected outputs:** running `python main.py` against `data/creators.csv` produces a composition workspace per creator before Module 7 runs.
- **Tests:** extend `tests/test_main_pipeline.py` with one new assertion that the composition workspace directory exists post-run.
- **Completion criteria:** existing `test_main_pipeline.py` assertions still pass unchanged; new assertion passes.

**Explicitly deferred, not part of this roadmap (flagged, not silently dropped):** extending `ReferenceAssets`/`WorkflowBuilder`/the ComfyUI workflow JSON templates to actually *consume* the `GenerationBundle` for ControlNet/IPAdapter conditioning. Per §0.5 and §19, that is a Module 7 change with its own design/testing surface (new graph nodes, new `GenerationProfile` behavior verification) and should be scoped as its own follow-on design document once you're ready for it.
