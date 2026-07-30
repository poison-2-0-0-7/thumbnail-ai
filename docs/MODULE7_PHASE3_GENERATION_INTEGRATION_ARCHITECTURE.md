# Module 7 — Phase 3: Generation Integration Architecture

**Enhancement to the existing Module 7 (Local Image Generation Engine).**
**Repository:** `poison-2-0-0-7/thumbnail-ai`
**Scope:** Module 7 only. Module 10 (Asset Composer), the main pipeline, and ComfyUI Phase 2 transport/queue/output components are treated as fixed, reused contracts.

---

## 0. Repository Grounding — What Actually Exists Today

This section states, factually, what the current repository does and does not support, so the design below is an extension of real code rather than an invented one.

**Confirmed, as of the current `main` branch:**

- `modules/composition_engine.py` (Module 10 / `AssetComposer`) produces a `CompositionWorkspace` and can flatten it into a `GenerationBundle` via `GenerationBundleBuilder.build_generation_bundle()`. `AssetComposer.prepare_generation_workspace(video_id)` is the one-call convenience entry point that returns a `GenerationBundle`.
- `GenerationBundle` (in `modules/models.py`) currently exposes exactly: `video_id`, `canvas`, `reference_image_paths: dict[str, str]` (keyed by `LayerRole` value, e.g. `"background"`, `"foreground"`, `"person"`, `"object"`), `mask_paths: dict[str, str]` (same keying), `depth_path: Optional[str]`, `canny_path: Optional[str]`, `layer_order: list[str]`, `workspace_hash`, `prompt_package_hash`, `generated_at`.
- `GenerationBundleBuilder.build_generation_bundle()` **never populates `canny_path`** — it is always `None` today, because no layer currently carries a canny hint path (only `depth_hint_path` exists on `CompositionLayer`). This is an existing gap in Module 10's builder, not something Phase 3 should paper over silently; it is called out in §22 (Risks) and left to Module 10's owners.
- `CompositionWorkspace` carries richer per-layer detail than the flat bundle: each `CompositionLayer` has an `AssetPlacement` with `role` (`LayerRole`), `decision` (`LayerDecision`: keep/remove/replace/enhance/add), `source_path`, an optional `MaskReference` (`mask_path`, `mask_checksum`, `feather_px`), a `LayerTransform`, and `z_index`. `CompositionWorkspace` also carries `text_placement` (avoid-zones and, when present, an `include_text` + `placement_zone_px`) and `constraints` (safe margins, avoid zones, focal zone).
- The Visual Reference Engine (Module 6.5, `visual_reference_engine.py` / `vre_components/manifest_builder.py`) is the **only** current producer feeding Module 10's `AssetRegistry`, and it emits exactly these asset keys: `creator_face`, `face_mask`, `object_crop`, `object_mask`, `foreground`, `background`, `depth_map`, `canny_map`. There is **no segmentation asset, no IP-Adapter reference asset, and no text-exclusion mask asset** anywhere in this pipeline.
- Module 8 (`asset_extraction_engine.py`, `SceneAsset.segmentation_map`) **does** produce a segmentation map, via SAM2/BiSeNet wrappers — but this is a separate manifest (`AssetExtractionManifest`) that Module 10's `AssetRegistry` does not currently index. Module 8's segmentation output is not wired into `CompositionWorkspace` or `GenerationBundle` today.
- Module 7's current `ImageGeneratorPipeline.run()` (Phase 3 foundation, `image_generator.py`) does **not** accept a `GenerationBundle` or `CompositionWorkspace` at all. It loads only a `PromptPackage` (via `PromptPackageLoader`) and resolves only a raw source thumbnail + optional Module 4 analysis JSON (via `ReferenceAssetResolver`). Module 7 is, today, fully disconnected from Module 10's output.
- `WorkflowBuilder._slots()` builds a flat placeholder dictionary from `PromptPackage`, `GenerationProfile`, and `ReferenceAssets` (source thumbnail path only), and `WorkflowBuilder._substitute()` performs pure `{{placeholder}}` string substitution over a static template graph loaded from `workflows/*.json` by `WorkflowLibrary`.
- Every shipped template (`workflows/general.json`, `gaming.json`, etc.) contains exactly seven nodes: `CheckpointLoaderSimple → CLIPTextEncode(+/-) → EmptyLatentImage → KSampler → VAEDecode → SaveImage`. **No template contains a ControlNet or IP-Adapter node today**, even though `GenerationProfile.controlnet_enabled` and `GenerationProfile.ipadapter_enabled` already exist as configured flags (`modules/config.py`, `MODULE7_GENERATION_PROFILES`) and are `True` for several profiles. These flags are therefore currently inert.
- `WorkflowLibrary.validate()` enforces only a minimal, generic contract (`_meta.{name,niche,workflow_version}` + a non-empty `graph` of `{class_type, inputs}` nodes) — it does not hardcode node topology, which is favorable for extension.
- `ComfyUIClient.generate(built_workflow, ...)` (Phase 2) is content-agnostic: it submits `built_workflow.graph` as-is and does not need to know about ControlNet/IP-Adapter/masks. **No changes to `comfyui_client.py`'s public surface are required**, with one narrow, additive exception proposed in §12.3 (an optional `/object_info` capability probe).

**Conclusion driving this design:** Phase 3's job is almost entirely inside `modules/image_generator.py` (+ new sibling components) and `workflows/*.json`. It must (a) start consuming the `GenerationBundle`/`CompositionWorkspace` contracts that already exist, (b) extend `WorkflowBuilder` to conditionally graft ControlNet/IP-Adapter/mask-conditioning fragments onto the existing static templates, and (c) be defensively written so that asset kinds not yet produced anywhere in the repository (segmentation maps, IP-Adapter references, text-exclusion masks, a second/third ControlNet) become "ready but dormant" — fully wired on the Module 7 side, safely inert until an upstream producer exists. Where those upstream producers are missing, this document says so explicitly rather than inventing a Module 10/Module 8 redesign.

---

## 1. Purpose

Module 7 Phase 3 — Generation Integration — extends the existing local Image Generation Engine so that it can consume the full richness of the Composition Workspace / Generation Bundle contract produced by Module 10, instead of generating from `PromptPackage` text alone plus a single raw source thumbnail. Concretely, Phase 3 gives Module 7 the ability to:

- Load a `GenerationBundle` (and, when needed, the underlying `CompositionWorkspace`) alongside the existing `PromptPackage`.
- Feed foreground/background source images, per-role masks, depth maps, and (when available) canny maps, segmentation maps, IP-Adapter reference images, and text-exclusion masks into the ComfyUI graph as real conditioning inputs.
- Support multiple simultaneous ControlNet inputs (e.g., depth + canny) rather than a single hardcoded conditioning path.
- Do all of this **without breaking any existing Phase 1/Phase 2/Phase 3 behavior** when a `GenerationBundle` is absent — the current text-only generation path must remain byte-for-byte reproducible.

This is explicitly *not* a new module, a Module 10 redesign, or a pipeline redesign. It is the smallest coherent set of additions to Module 7 that lets it act on data Module 10 already produces (and to be forward-compatible with data Module 10/Module 8 do not yet produce).

---

## 2. Design Goals

1. **Additive, not disruptive.** Every new input to `ImageGeneratorPipeline.run()`, `WorkflowBuilder.build()`, and the workflow template placeholder set is optional with a safe default. No existing method signature loses a parameter; no existing required behavior changes.
2. **Deterministic.** Given the same `PromptPackage`, `GenerationProfile`, `GenerationBundle`/`CompositionWorkspace`, and seed, the materialized ComfyUI graph — and therefore `workflow_hash` and `generation_hash` — must be byte-identical across runs. Conditional fragment injection must be a pure function of its inputs, exactly like today's `_substitute()`.
3. **Graceful degradation over hard failure.** Missing conditioning assets (no depth map, no IP-Adapter reference, bundle absent entirely) must never abort generation. They must cause the corresponding graph fragment to be skipped, with a logged reason, and generation proceeds on whatever conditioning is available — down to the current Phase 2 baseline graph.
4. **Reuse existing collaborators.** `WorkflowLibrary`, `ComfyUIClient`, `PromptPackageLoader`, `ReferenceAssetResolver`, and all Phase 3 pipeline stages (identity, restoration, background composition, upscale, QA, ranking) are reused unmodified or extended in place — never duplicated.
5. **No new responsibility duplication.** Module 7 does not re-implement layer placement, decision resolution, mask generation, or workspace validation — all of that stays owned by Module 10 (`composition_components/*`) and Module 6.5 (VRE). Module 7 only *consumes* their already-validated outputs.
6. **Explicit about repository gaps.** Where an objective (segmentation maps, IP-Adapter references, text-exclusion masks) has no current upstream producer, Module 7's contract is defined to accept it opportunistically (defensive attribute access, tolerant of an unextended `GenerationBundle`) rather than assuming it will always be present.
7. **Testable in isolation.** Every new collaborator is a small, single-responsibility class with constructor-injected dependencies, following the exact pattern already used by `ImageGeneratorPipeline` and Module 10's `composition_components` (interfaces + concrete default implementations).

---

## 3. Folder Structure

Phase 3 introduces one new package, `modules/generation_components/`, mirroring the existing `composition_components/` convention, plus a workflow-template fragment library. No existing file is moved or renamed.

```
modules/
  image_generator.py                 # EXTENDED: WorkflowBuilder, ReferenceAssets,
                                      #   ImageGeneratorPipeline (new optional params/collaborators)
  workflow_library.py                # UNCHANGED (already generic enough)
  comfyui_client.py                  # EXTENDED (additive only): object_info() capability probe
  module7_exceptions.py              # EXTENDED: new exception subclasses (§14)
  config.py                          # EXTENDED: new MODULE7_* settings (§15)
  models.py                          # UNCHANGED contract; see §7.3 for the one proposed,
                                      #   backward-compatible, opt-in extension to GenerationBundle
  generation_components/             # NEW package (Module 7 Phase 3 only)
    __init__.py
    interfaces.py                    # ABCs: IGenerationBundleLoader, ICompositionWorkspaceLoader,
                                      #   IConditioningAssetResolver, INodeFragmentLibrary,
                                      #   IWorkflowGraphAssembler, IComfyUICapabilityProbe
    generation_bundle_loader.py      # Loads a persisted GenerationBundle by video_id (read-only)
    workspace_loader.py              # Loads a persisted CompositionWorkspace by video_id (read-only)
    conditioning_asset_resolver.py   # Normalizes Bundle + Workspace + legacy ReferenceAssets into
                                      #   one GenerationConditioningContext value object
    node_fragment_library.py         # Loads/validates declarative graph fragments (§9, §10)
    workflow_graph_assembler.py      # Merges fragments into a materialized template graph (§9)
    capability_probe.py              # Optional: queries ComfyUI /object_info to skip fragments
                                      #   whose custom nodes are not installed (§12.3)
workflows/
  general.json ... documentary.json  # UNCHANGED (existing 7-node templates keep working as-is)
  fragments/                         # NEW: declarative, reusable graph fragments (§9.2)
    controlnet_depth.json
    controlnet_canny.json
    controlnet_segmentation.json     # dormant until an upstream segmentation producer exists
    ipadapter_reference.json         # dormant until an upstream IP-Adapter producer exists
    text_exclusion_mask.json         # dormant until an upstream exclusion-mask producer exists
    regional_mask_conditioning.json  # per-layer mask-gated conditioning (uses CompositionWorkspace)
tests/
  test_generation_components/        # NEW, mirrors tests/test_composition_components/
    test_generation_bundle_loader.py
    test_workspace_loader.py
    test_conditioning_asset_resolver.py
    test_node_fragment_library.py
    test_workflow_graph_assembler.py
    test_capability_probe.py
  test_image_generator.py            # EXTENDED with new bundle-aware pipeline tests
```

---

## 4. Public APIs

Public surfaces are additive extensions of existing signatures; nothing already public is removed.

### 4.1 `ImageGeneratorPipeline.run()` (extended)

```
run(
    video_id: str,
    niche: str = "general",
    available_vram_gb: float = float("inf"),
    prompt_package: PromptPackage | None = None,
    generation_bundle: GenerationBundle | None = None,        # NEW
    composition_workspace: CompositionWorkspace | None = None, # NEW
) -> ImageGenerationResult
```

`generation_bundle` and `composition_workspace` are both optional and independent:
- Neither supplied → identical behavior to current Phase 3 (text + raw thumbnail only).
- `generation_bundle` supplied → layer-flattened conditioning (masks, depth/canny, per-role source images) is used.
- `composition_workspace` supplied (with or without an explicit bundle) → per-layer detail (feathered masks, per-layer transforms, text avoid-zones) becomes available for regional conditioning; if `generation_bundle` is omitted but `composition_workspace` is given, Module 7 derives an equivalent bundle in-process via the same `GenerationBundleBuilder` Module 10 already uses (imported, not re-implemented — see §7.2).

### 4.2 `run_image_generation_pipeline()` (extended)

Adds the same two optional parameters, forwarded unchanged to `ImageGeneratorPipeline.run()`. This preserves the existing top-level helper contract used by `main.py`.

### 4.3 `WorkflowBuilder.build()` (extended)

```
build(
    package: PromptPackage,
    profile: GenerationProfile,
    workflow_ref: WorkflowTemplateRef,
    reference_assets: ReferenceAssets | None = None,
    library: WorkflowLibrary | None = None,
    conditioning: GenerationConditioningContext | None = None,  # NEW
) -> BuiltWorkflow
```

`conditioning` is optional; `None` reproduces the exact current template-substitution behavior.

### 4.4 New public types (in `generation_components`)

- `GenerationBundleLoader.load(video_id: str) -> GenerationBundle`
- `CompositionWorkspaceLoader.load(video_id: str) -> CompositionWorkspace`
- `ConditioningAssetResolver.resolve(bundle, workspace, reference_assets, profile) -> GenerationConditioningContext`
- `GenerationConditioningContext` (frozen dataclass, see §7.1) — the single normalized object the rest of Module 7 depends on.
- `NodeFragmentLibrary.discover() -> list[Path]`, `.load(fragment_id) -> WorkflowFragment`
- `WorkflowGraphAssembler.assemble(base_graph, fragments, conditioning, profile) -> dict[str, Any]`

---

## 5. Internal APIs

These are private/internal collaborators, not intended for cross-module import.

- `WorkflowBuilder._slots()` — extended to add a fixed, always-present set of new placeholder keys sourced from `GenerationConditioningContext` (empty-string/sentinel defaults when data is absent), exactly mirroring the existing `"source_thumbnail_path": ... if references else ""` pattern.
- `WorkflowBuilder._select_fragments(profile, conditioning) -> list[str]` — pure function returning the deterministic, ordered list of fragment IDs to inject, based on profile flags and conditioning-context presence checks (§9.1).
- `WorkflowGraphAssembler._namespace_fragment(fragment, prefix) -> dict` — rewrites a fragment's internal node IDs with a stable prefix (e.g. `cn_depth::5`) to avoid collisions with base-template node IDs or other fragments.
- `WorkflowGraphAssembler._rewire_attachment_point(graph, attachment_point, new_output_ref)` — rewrites the single declared downstream input reference (e.g. `KSampler.inputs.positive`) to point at a fragment's output node instead of the base node it previously pointed to, threading fragments in a defined, deterministic order when more than one attaches to the same point (§9.3).
- `CapabilityProbe._installed_node_types() -> frozenset[str]` — cached, single-call-per-run wrapper around the new `ComfyUIClient.object_info()`/`_ComfyUIHTTPTransport.object_info()` method.

---

## 6. Data Flow

```
                     ┌─────────────────────────┐
                     │   Module 10 (fixed)      │
                     │   AssetComposer          │
                     │  .prepare_generation_     │
                     │   workspace(video_id)     │
                     └───────────┬──────────────┘
                                 │ GenerationBundle (in-memory, or persisted JSON)
                                 │ + CompositionWorkspace (persisted JSON, via WorkspaceManager)
                                 ▼
     ┌──────────────────────────────────────────────────────────────┐
     │                Module 7 Phase 3 — Generation Integration       │
     │                                                                  │
     │  PromptPackageLoader ──► PromptPackage                          │
     │  GenerationBundleLoader ──► GenerationBundle  (optional)        │
     │  CompositionWorkspaceLoader ──► CompositionWorkspace (optional) │
     │  ReferenceAssetResolver ──► ReferenceAssets (legacy, always run)│
     │                     │                                          │
     │                     ▼                                          │
     │        ConditioningAssetResolver.resolve(...)                  │
     │                     │                                          │
     │                     ▼                                          │
     │          GenerationConditioningContext (frozen)                │
     │                     │                                          │
     │  ProfileSelector ───┤                                          │
     │  WorkflowLibrary ───┤                                          │
     │                     ▼                                          │
     │   WorkflowBuilder.build(package, profile, ref, conditioning)   │
     │      ├─ base graph = template substitution (unchanged path)    │
     │      ├─ fragment selection (pure function of profile+context)  │
     │      └─ WorkflowGraphAssembler merges selected fragments        │
     │                     │                                          │
     │                     ▼                                          │
     │              BuiltWorkflow (graph + hash)                       │
     │                     │                                          │
     │                     ▼                                          │
     │        ComfyUIClient.generate(built_workflow, ...)  (unchanged) │
     │                     │                                          │
     │        …existing Phase 3 stages: identity → restoration →      │
     │        background composition → upscale → QA → ranking…        │
     └──────────────────────────────────────────────────────────────┘
```

Key property: `ConditioningAssetResolver` is the single seam where Module 10's contract enters Module 7. Everything downstream of `GenerationConditioningContext` — including `WorkflowBuilder` — has no direct dependency on `CompositionWorkspace`/`GenerationBundle` types, which keeps Module 7 loosely coupled to Module 10's schema.

---

## 7. GenerationBundle Consumption

### 7.1 `GenerationConditioningContext`

A new frozen, dataclass-like value object (constructed with plain Python types and `Path`s, not a Pydantic model, to keep it an internal Module 7 concern rather than a cross-module contract) with the following shape:

- `source_thumbnail_path: Path | None` — legacy fallback, from `ReferenceAssets` (always populated when a source thumbnail exists, bundle or not).
- `canvas_width`, `canvas_height`, `aspect_ratio` — from `GenerationBundle.canvas` when present, else from `PromptPackage.generation_parameters`.
- `role_image_paths: dict[str, Path]` — from `GenerationBundle.reference_image_paths`, keyed by `LayerRole` value (`background`, `foreground`, `person`, `object`; `text`/`effect` layers rarely carry a source image but the mapping is generic).
- `role_mask_paths: dict[str, Path]` — from `GenerationBundle.mask_paths`, same keying.
- `depth_path: Path | None` — from `GenerationBundle.depth_path`.
- `canny_path: Path | None` — from `GenerationBundle.canny_path` (today always `None`; see §0 and §22).
- `segmentation_path: Path | None` — read **defensively** via `getattr(bundle, "segmentation_path", None)`; always `None` against today's `GenerationBundle` schema, non-`None` only if/when Module 10 is separately extended (§7.3).
- `ip_adapter_reference_paths: dict[str, Path]` — read defensively via `getattr(bundle, "ip_adapter_reference_paths", {})`; always empty today.
- `text_exclusion_mask_path: Path | None` — read defensively via `getattr(bundle, "text_exclusion_mask_path", None)`; always `None` today, but also derivable, when a `CompositionWorkspace` is supplied, from `workspace.text_placement.avoid_zones_px` (§7.2) even without any bundle-schema change.
- `layer_order: tuple[str, ...]` — from `GenerationBundle.layer_order`.
- `per_layer: dict[str, LayerConditioning] | None` — populated only when a `CompositionWorkspace` is supplied (§7.2); each `LayerConditioning` carries `role`, `decision`, `mask_path`, `feather_px`, `z_index`, `crop_box` for regional/masked conditioning use cases.

All fields default to `None`/empty. Every field on `GenerationConditioningContext` is independently optional — the resolver never raises for a missing asset; it only raises (`ConditioningResolutionError`, §14) for structurally invalid input it was explicitly given (e.g., a `mask_paths` entry pointing at a file that does not exist on disk, matching the existing "resolve local paths, verify nothing" philosophy already used by `ReferenceAssetResolver` and `AssetRegistry.verify_integrity`).

### 7.2 Deriving from `CompositionWorkspace` when only a workspace is supplied

If the caller supplies `composition_workspace` but not `generation_bundle`, `ConditioningAssetResolver` calls the **existing, imported** `GenerationBundleBuilder.build_generation_bundle(workspace)` (from `composition_components.generation_bundle_builder`) to obtain an equivalent flat bundle, then proceeds exactly as in §7.1. This reuses Module 10's own flattening logic instead of re-implementing it in Module 7 — satisfying "reuse existing components" and "avoid duplicate responsibilities."

In addition, when a `CompositionWorkspace` is available, the resolver builds the `per_layer` map directly from `workspace.layers`, giving Module 7 access to per-layer `MaskReference.feather_px` and per-layer `z_index`/`decision` — detail the flat bundle intentionally drops. This is what enables regional/masked ControlNet conditioning (§12.2) and is the concrete reason Objective "Read CompositionWorkspace" exists as distinct from "Consume GenerationBundle."

`workspace.text_placement.avoid_zones_px` (already present today) is also used, opportunistically, to synthesize a **derived** text-exclusion mask region (a pixel rectangle set, not a full raster mask) for consumers that only need coarse exclusion. A raster text-exclusion mask *image* (the objective's literal ask) is not currently produced anywhere upstream; §7.3 addresses that gap explicitly rather than fabricating a rasterizer inside Module 7.

### 7.3 Proposed, backward-compatible `GenerationBundle` extension (not implemented by this document)

To fully satisfy the objectives that have no current upstream producer — segmentation maps, IP-Adapter reference images, a raster text-exclusion mask, and a generalized multi-ControlNet map — the smallest compatible change would be **purely additive** new `Optional`/default-empty fields on `GenerationBundle` (all `None`/`{}` by default, so existing consumers and existing tests are unaffected):

- `segmentation_path: Optional[str] = None`
- `ip_adapter_reference_paths: dict[str, str] = {}`
- `text_exclusion_mask_path: Optional[str] = None`
- `controlnet_inputs: dict[str, str] = {}` — a generalized, keyed replacement path for today's single `depth_path`/`canny_path` fields, so an arbitrary number of named ControlNet conditioning images (e.g. `"depth"`, `"canny"`, `"segmentation"`, `"lineart"`) can travel through the bundle without repeatedly adding new top-level fields. `depth_path`/`canny_path` would be kept, unmodified, for backward compatibility.

This is explicitly **out of scope for this document** — it is a Module 10 model/builder change, and the instructions for this task prohibit redesigning Module 10. It is recorded here so that (a) Module 7's `ConditioningAssetResolver` is written defensively enough to pick these fields up automatically the day they exist, with zero Module 7 code changes, and (b) the reader is not left thinking segmentation/IP-Adapter/exclusion-mask support was silently dropped — it is architected for, and blocked only on an upstream field that does not exist yet.

---

## 8. Workspace Loading

- `CompositionWorkspaceLoader` reads a persisted `CompositionWorkspace` JSON file directly with `CompositionWorkspace.model_validate_json(...)`, the same direct-file-plus-Pydantic-validation pattern `PromptPackageLoader` already uses for `PromptPackage`. It does **not** invoke `AssetComposer`/`composition_engine.py` at runtime (no VRE re-run, no recomposition) — it is a pure read of an already-composed, already-validated artifact.
- The storage location is resolved through the **same** `COMPOSITION_WORKSPACE_ROOT` / `WorkspaceManager` path convention Module 10 already defines, imported from `config.py`, so there is exactly one source of truth for "where workspaces live" (no path duplication between Module 7 and Module 10).
- If the file is missing, `CompositionWorkspaceLoader.load()` raises `WorkspaceNotFoundError` (new, §14) — but this loader is only ever invoked by `ImageGeneratorPipeline.run()` when the caller explicitly opted in by passing `composition_workspace=None` *and* requesting workspace-based loading via an explicit `video_id`-only call path (i.e., loading is caller-initiated, not automatic/implicit — see §20). Silent, implicit disk probing for a workspace that may or may not exist would violate the backward-compatibility goal (a Phase 1/2-only caller must not incur new failure modes it never asked for).
- `GenerationBundleLoader` follows the identical pattern for a persisted `GenerationBundle`, if/when Module 10's `WorkspaceManager`-equivalent persistence for bundles is used; if a caller already has a bundle in memory (the common case — `AssetComposer.prepare_generation_workspace()` returns one directly), the loader is bypassed entirely by passing `generation_bundle=...` to `run()`.

---

## 9. WorkflowBuilder Extensions

### 9.1 Fragment selection (pure, deterministic)

`WorkflowBuilder._select_fragments(profile, conditioning)` returns an **ordered** list of fragment IDs, computed purely from:

| Condition | Fragment selected |
|---|---|
| `profile.controlnet_enabled` and `conditioning.depth_path is not None` | `controlnet_depth` |
| `profile.controlnet_enabled` and `conditioning.canny_path is not None` | `controlnet_canny` |
| `profile.controlnet_enabled` and `conditioning.segmentation_path is not None` | `controlnet_segmentation` (dormant today — §7.3) |
| `profile.ipadapter_enabled` and `conditioning.ip_adapter_reference_paths` non-empty | `ipadapter_reference` (dormant today — §7.3) |
| `conditioning.text_exclusion_mask_path is not None` or derivable exclusion rectangles present | `text_exclusion_mask` (partially live via §7.2 rectangle derivation) |
| `conditioning.per_layer` present and any layer carries a mask | `regional_mask_conditioning` |

The ordering itself is fixed (declared as a constant tuple, not dict-iteration order), so two builds with identical inputs always select fragments in the identical order — required for the graph-hash determinism goal (§2.2).

Zero fragments selected ⇒ `WorkflowGraphAssembler` returns the base template graph completely untouched, which is what guarantees byte-identical output for every existing Phase 1/2/3 test and caller.

### 9.2 Fragment structure

Each file under `workflows/fragments/*.json` is a small, self-contained, placeholder-parameterized graph patch with the same `{class_type, inputs}` node shape `WorkflowLibrary.validate()` already accepts, plus one new declarative block, `_attach`, describing how the fragment splices into a host graph:

- `_attach.point`: a symbolic name for the single input reference the fragment overrides (e.g. `positive_conditioning`, `model`, `latent`). Symbolic names are mapped to concrete `(base_node_id, input_key)` pairs by a small, versioned lookup table owned by the base template's `_meta` block (`_meta.attachment_points`), so fragments never hardcode a specific template's numeric node IDs.
- `_attach.output_node` / `_attach.output_slot`: which node+slot inside the fragment becomes the new value at that attachment point.
- `_attach.requires`: the list of `GenerationConditioningContext` fields this fragment needs populated (used by fragment-level validation, independent of `_select_fragments`'s own selection logic, as a defense-in-depth check).

This keeps fragments declarative JSON — consistent with "NO code" for this document and with the existing template format — while making them genuinely reusable across templates/niches without per-niche duplication.

### 9.3 Multi-fragment attachment ordering

When more than one fragment targets the same attachment point (e.g. two ControlNets both needing to modify `positive_conditioning`), `WorkflowGraphAssembler` chains them: the first selected fragment's output becomes the base graph's new input at that point, and each subsequent fragment targeting the same point is rewired to consume the *previous* fragment's output rather than the original base node, producing a linear conditioning chain (`base_positive → ControlNetApply(depth) → ControlNetApply(canny) → KSampler.positive`). This is the standard ComfyUI multi-ControlNet composition pattern and requires no new node types beyond what a single-ControlNet fragment already needs, generalized.

### 9.4 `_slots()` extension

`WorkflowBuilder._slots()` gains a fixed set of additional keys, always present with safe defaults, exactly following the existing `"source_thumbnail_path": str(references.source_thumbnail_path) if references else ""` idiom:

`foreground_image_path`, `background_image_path`, `person_mask_path`, `object_mask_path`, `depth_map_path`, `canny_map_path`, `segmentation_map_path`, `text_exclusion_mask_path`, and one indirection, `ip_adapter_reference_paths` (a list, substituted only inside the `ipadapter_reference` fragment, which is the only template context that ever iterates over it).

Because template substitution (`_substitute`) already raises `WorkflowBuildError` on an unknown placeholder key, and fragments are the only place new placeholder names are introduced, existing base templates remain valid without modification — they simply never reference the new keys.

---

## 10. ComfyUI Workflow Architecture

- **Base graph** (per niche/profile, unchanged): `CheckpointLoaderSimple → CLIPTextEncode(±) → EmptyLatentImage → KSampler → VAEDecode → SaveImage`.
- **Depth/Canny ControlNet fragment shape:** `LoadImage(conditioning image path) → ControlNetLoader → ControlNetApply(image, positive_conditioning, strength) → (new positive_conditioning)`. Strength is sourced from a new per-profile/per-fragment configuration value (§15), not hardcoded in the fragment.
- **IP-Adapter fragment shape:** `LoadImage(reference image path) → IPAdapterModelLoader / CLIPVisionLoader → IPAdapterApply(model, image, weight) → (new model)`. Attaches at the `model` point rather than `positive_conditioning`.
- **Regional mask conditioning fragment shape:** `LoadImageMask(per-layer mask path) → ConditioningSetMask(positive_conditioning, mask, strength, set_cond_area) → (new positive_conditioning)`, one instance per masked layer in `conditioning.per_layer`, chained per §9.3.
- **Text-exclusion fragment shape:** functionally a `ConditioningSetMask`/negative-region variant that either (a) consumes a real raster mask path when `conditioning.text_exclusion_mask_path` is populated (post §7.3 extension), or (b) is skipped today, with the coarse rectangle data from `workspace.text_placement.avoid_zones_px` instead surfaced as an additional negative-prompt clause appended in `_slots()`'s existing `negative_prompt` assembly (a text-level fallback that requires no new node type and works with today's schema).
- **Version compatibility:** fragments declare a `_meta.min_comfyui_version` / `_meta.required_node_types` block. `CapabilityProbe` (§12.3) checks required node types against the live ComfyUI instance before a fragment is selected; if unavailable, the fragment is dropped with a `logger.warning`, and generation proceeds without it (never a hard failure) — directly satisfying the "Fallback logic" and "Version compatibility" objectives.

---

## 11. Asset Loading Strategy

- All asset paths that reach the ComfyUI graph are **local filesystem paths already resolved and integrity-checked upstream** (`AssetRegistry.verify_integrity()` in Module 10, `MaskReference.mask_checksum` validation in `models.py`). Module 7 does not re-validate checksums; it re-validates only **existence** at resolve time (`ConditioningAssetResolver`), because a workspace could be stale relative to a since-cleaned `data/` directory (`AssetComposer.clean_workspace` / `VisualReferenceEngine.clean_assets` both exist and can run independently of Module 7's schedule).
- Loading is entirely **lazy and reference-based**: Module 7 never reads image bytes for conditioning assets into Python memory. Paths are substituted into `LoadImage`/`LoadImageMask` nodes, and ComfyUI itself performs the file I/O — consistent with the existing `GeneratedAsset`/`ReferenceAssets` philosophy of "never embed image bytes in a manifest or in-process object."
- `ConditioningAssetResolver` is the single choke point for path resolution, so a future storage-backend change (e.g., asset paths becoming URIs instead of local paths) touches one class, not the `WorkflowBuilder` or any fragment.

---

## 12. ControlNet Architecture

### 12.1 Single-input (today, live)

Depth and canny each map 1:1 to a `controlnet_depth`/`controlnet_canny` fragment, gated on `profile.controlnet_enabled` and asset presence (§9.1). This is fully supported by the existing `GenerationBundle.depth_path` field; `canny_path` is wired identically and will activate automatically once the existing Module 10 gap (§0, always-`None` `canny_path`) is fixed upstream — no Module 7 change required at that point.

### 12.2 Regional/multi-ControlNet (via `CompositionWorkspace`)

When a `CompositionWorkspace` is supplied, each layer's own mask (already present today, per-layer, with feathering) can gate its own conditioning region via the `regional_mask_conditioning` fragment (§10), letting depth conditioning apply broadly while, e.g., a person layer's mask restricts an IP-Adapter or a second ControlNet to just that region. This is additive to, not a replacement for, the flat-bundle path.

### 12.3 Capability probing (new, additive)

`_ComfyUIHTTPTransport` gains one new read-only method, `object_info() -> dict[str, Any]`, wrapping ComfyUI's existing `/object_info` HTTP endpoint (the same request/response pattern already used by `system_stats()`), and `ComfyUIClient` exposes a thin `object_info()` passthrough. `CapabilityProbe` calls this once per pipeline run (cached), and `WorkflowBuilder`/`WorkflowGraphAssembler` consult it before finalizing fragment selection, dropping any fragment whose `_meta.required_node_types` are not present in the response. This is additive to Phase 2's `comfyui_client.py` public surface (`SystemStats`, `ComfyUIClient` in `__all__` — `object_info` is simply exported alongside them) and does not alter `generate()`'s existing contract or the metrics-recording path.

---

## 13. IP-Adapter Architecture

Structurally identical to §12's ControlNet pattern, attaching at the `model` point instead of `positive_conditioning` (§10). `conditioning.ip_adapter_reference_paths` is a `dict[str, Path]` (not a single path) so multiple reference images (e.g., a face reference and a style reference) can be supplied and iterated by the `ipadapter_reference` fragment, each producing a chained `IPAdapterApply` node (same chaining mechanism as §9.3). As documented in §0/§7.3, **no upstream producer currently populates IP-Adapter reference images**, so this path is fully built but will select zero fragments in every run against the current repository state — verified by a dedicated test (§19) asserting the fragment is never selected when `ip_adapter_reference_paths` is empty, which it always is today.

---

## 14. Error Handling

New exceptions, added to `module7_exceptions.py`, following the existing `Module7Error` hierarchy and existing granularity conventions (one class per distinct failure a caller might want to catch separately):

- `ConditioningResolutionError(Module7Error)` — a bundle/workspace was supplied but references a path that does not exist on disk, or a `mask_paths`/`reference_image_paths` entry is structurally malformed.
- `WorkspaceNotFoundError(Module7Error)` — `CompositionWorkspaceLoader.load()` was explicitly invoked and found nothing.
- `GenerationBundleInvalidError(Module7Error)` — a supplied `GenerationBundle`'s `video_id` doesn't match the requested `video_id`, or `status` indicates a failed Module 10 run (mirrors `PromptPackageLoader`'s existing `video_id` mismatch / error-status checks).
- `FragmentAttachmentError(WorkflowBuildError)` — a fragment declares an `_attach.point` the base template's `_meta.attachment_points` doesn't define (a template/fragment authoring mistake, caught at build time, not at ComfyUI submission time).
- `UnsupportedNodeTypeWarning(Warning)` — non-fatal; logged when `CapabilityProbe` drops a fragment for a missing node type (mirrors the existing `ProfileDowngradedWarning` pattern of a `Warning` subclass used for logged-not-raised conditions).

**Failure policy:** everything in §7–§13 that depends on *optional* conditioning data fails soft (skip + log). Everything that depends on data the caller *explicitly and deliberately supplied* (a bundle, a workspace) but which turns out to be internally inconsistent fails hard, exactly like today's `PromptPackageInvalidError`/`ReferenceAssetError` — because a caller who passed a bundle expects it to be honored, and a silent downgrade there would be a correctness bug, not a graceful degradation.

---

## 15. Configuration

New settings added to `config.py`, alongside the existing `MODULE7_*` block, following existing naming/typing conventions:

- `MODULE7_FRAGMENT_LIBRARY_DIR: Path = PROJECT_ROOT / "workflows" / "fragments"`
- `MODULE7_CONTROLNET_STRENGTH_DEFAULTS: dict[str, float] = {"depth": 0.55, "canny": 0.45, "segmentation": 0.5}`
- `MODULE7_IPADAPTER_WEIGHT_DEFAULT: float = 0.6`
- `MODULE7_CAPABILITY_PROBE_ENABLED: bool = True` (an explicit escape hatch to disable the `/object_info` round-trip entirely, e.g. for offline template validation or CI without a live ComfyUI instance)
- `MODULE7_CAPABILITY_PROBE_CACHE_SECONDS: float = 300.0`
- `MODULE7_COMPOSITION_WORKSPACE_ROOT` — **reused, not duplicated**: imported directly from the existing `COMPOSITION_WORKSPACE_ROOT` Module 10 setting, so Module 7 never hardcodes a second copy of that path.

All new settings have safe, working defaults; no `.env`/deployment change is required for Phase 3 to ship — profiles that already set `controlnet_enabled=True`/`ipadapter_enabled=True` will simply start doing something the first time a `GenerationBundle` with real assets is supplied.

---

## 16. Logging

All new components attach to the **existing** `MODULE7_LOG_PATH` Loguru sink via the existing `_configure_logger()` pattern already duplicated (deliberately, per current repo convention) across `image_generator.py`, `workflow_library.py`, and `comfyui_client.py` — Phase 3's new files (`generation_components/*.py`) follow the identical per-file `_configure_logger()` + module-level call convention, with the same `_LOG_FORMAT`. New log events, matching the existing structured-message style (`logger.info("... {field}", field=value)`):

- `"Resolved GenerationConditioningContext for video_id={video_id}: roles={roles}, masks={masks}, depth={has_depth}, canny={has_canny}, ip_adapter_refs={n_refs}"`
- `"Fragment selection for video_id={video_id}: selected={fragment_ids}"`
- `"Fragment {fragment_id} dropped: required node type {node_type} not available in ComfyUI /object_info"`
- `"Assembled workflow graph with {n_fragments} fragment(s) attached; final node count={count}"`

---

## 17. Caching

- **Capability probe cache:** `CapabilityProbe` caches the `/object_info` result in-process for `MODULE7_CAPABILITY_PROBE_CACHE_SECONDS`, avoiding a network round-trip per candidate within one multi-candidate `run()` call (today's pipeline already loops `num_candidates` times per video).
- **No new persistent cache is introduced.** `CompositionWorkspaceLoader`/`GenerationBundleLoader` read straight from disk each call; Module 10's own `WorkspaceManager.resume()` cache (keyed by content hash) already prevents redundant recomposition upstream, and duplicating that cache inside Module 7 would violate the "avoid duplicate responsibilities" design goal.
- **Graph assembly is cheap and pure** (dict merging over a handful of small JSON fragments), so no fragment-assembly cache is warranted; `workflow_hash` itself already serves as the natural memoization key if a future phase wants one.

---

## 18. Dependency Injection

Every new collaborator follows the exact constructor-injection pattern already used throughout Module 7 (`ImageGeneratorPipeline.__init__`) and Module 10 (`AssetComposer.__init__` + `composition_components/interfaces.py` ABCs):

- `generation_components/interfaces.py` defines `ABC`s: `IGenerationBundleLoader`, `ICompositionWorkspaceLoader`, `IConditioningAssetResolver`, `INodeFragmentLibrary`, `IWorkflowGraphAssembler`, `IComfyUICapabilityProbe`.
- Concrete default implementations (`GenerationBundleLoader`, `CompositionWorkspaceLoader`, `ConditioningAssetResolver`, `NodeFragmentLibrary`, `WorkflowGraphAssembler`, `CapabilityProbe`) implement these interfaces.
- `ImageGeneratorPipeline.__init__` gains matching `Optional[I...] = None` constructor parameters for each, defaulting to the concrete implementation when unset — identical to how `workflow_library`, `identity_stage`, `background_compositor`, etc. are already wired today. No global singletons, no service locator, no module-level mutable state beyond the existing Loguru sink pattern.
- `WorkflowBuilder` itself gains optional constructor parameters for `NodeFragmentLibrary`/`WorkflowGraphAssembler`/`CapabilityProbe` (defaulting the same way), so unit tests can substitute fakes exactly as `tests/test_image_generator.py` already substitutes a mock `client`.

---

## 19. Testing Strategy

Mirrors the existing convention: one test module per component, `pytest`, `tmp_path` fixtures, no network access, no real ComfyUI instance, and reuse of the existing model-construction helper style seen in `tests/test_image_generator.py` and `tests/test_composition_engine.py`.

- **`test_generation_bundle_loader.py` / `test_workspace_loader.py`:** load success, missing-file → correct new exception, mismatched `video_id` → `GenerationBundleInvalidError`.
- **`test_conditioning_asset_resolver.py`:** (a) bundle=None, workspace=None → context is all-empty, downstream builder output unchanged; (b) full bundle with all current fields populated → context fields map 1:1; (c) bundle missing new-schema attributes entirely (simulating today's actual `GenerationBundle`) → defensive `getattr` paths return `None`/`{}` without raising; (d) workspace-only input → bundle is derived via the real `GenerationBundleBuilder` and matches a directly-built bundle for the same workspace (equality/hash check); (e) a referenced path that doesn't exist on disk → `ConditioningResolutionError`.
- **`test_node_fragment_library.py` / `test_workflow_graph_assembler.py`:** fragment discovery is deterministic/sorted (mirrors `WorkflowLibrary.discover()`'s existing sorted-glob test); assembling zero fragments returns the base graph unchanged (object equality with the pre-Phase-3 golden graph fixture — the critical regression guard); assembling two fragments at the same attachment point produces the documented chained wiring (§9.3); an unknown `_attach.point` raises `FragmentAttachmentError`.
- **`test_capability_probe.py`:** required node types present → fragment kept; absent → fragment dropped + `UnsupportedNodeTypeWarning` logged; `MODULE7_CAPABILITY_PROBE_ENABLED=False` → probe never called (verified via a call-counting fake HTTP transport, same style as existing `test_comfyui_client.py` fakes).
- **`test_image_generator.py` additions:**
  - **Golden regression test:** building a workflow with `conditioning=None` (or omitted) produces an identical `workflow_hash` to the pre-Phase-3 baseline fixture, for every shipped template in `workflows/*.json` — this is the single most important test in this plan, guaranteeing §2's backward-compatibility goal empirically, not just by design.
  - Full end-to-end `ImageGeneratorPipeline.run(..., generation_bundle=<fixture with depth+mask+two role images>)` against a mock `ComfyUIClient`, asserting the submitted graph contains the expected injected nodes and that `ImageGenerationResult` still validates and persists identically to today's shape (no new required fields on `ImageGenerationResult` — see §21).
  - Property-style determinism test: building the same `(package, profile, workflow_ref, conditioning)` tuple twice yields identical `workflow_hash` (extends the existing `test_workflow_builder_is_pure_and_hashes_resolved_graph`).
- **Fixtures:** a small `tests/fixtures/generation_bundles/` and `tests/fixtures/composition_workspaces/` directory of minimal valid JSON, constructed the same way `tests/test_composition_models.py` already builds `CompositionWorkspace` instances in Python (preferred over hand-authored JSON, to stay in sync with the Pydantic schema automatically).

---

## 20. Integration Strategy

- **`main.py` (fixed pipeline) integration point:** after Module 10's `AssetComposer.prepare_generation_workspace(video_id)` returns a `GenerationBundle` (already the case today, per the confirmed Module 10 contract — see §0), the orchestrator passes it straight into `run_image_generation_pipeline(video_id, ..., prompt_package=..., generation_bundle=bundle)`. No new orchestration step is introduced; this is a one-argument addition to an existing call site. `composition_workspace` is passed only when the orchestrator has one in hand (e.g., it can call `AssetComposer.load_workspace(video_id)` if regional conditioning is desired) — entirely optional and independently toggle-able.
- **Rollout is per-call, not global.** Because both new parameters default to `None`, existing callers (including any external scripts, `smoke_test_vre.py`-style manual tools, or not-yet-updated tests) continue to work unmodified. There is no feature flag needed at the config level beyond `MODULE7_CAPABILITY_PROBE_ENABLED`; the presence/absence of a bundle *is* the feature flag.
- **Template rollout is independent of code rollout.** Shipping the fragment files under `workflows/fragments/` does not, by itself, change any existing template's behavior, because base templates are loaded and substituted exactly as before; fragments only attach when `_select_fragments` chooses them, which requires both a profile flag and real conditioning data.

---

## 21. Backward Compatibility

- `ImageGenerationResult`, `GenerationMetrics`, and the on-disk manifest (`{video_id}_manifest.json`) schemas are **unchanged** — no new required fields. Generation runs that use the new conditioning path are indistinguishable, at the manifest level, from ones that don't, other than (optionally, non-breaking) `workflow_hash` differing because the graph itself differs when fragments are attached — which is the correct and expected behavior of a hash over the actual graph content.
- Every existing public function/method signature retains its exact current parameter list; only new optional, defaulted parameters are appended at the end, which is non-breaking for both positional- and keyword-style existing call sites (all current call sites in this repository use keyword arguments for these calls, consistent with the existing code style, further reducing positional-argument risk).
- Every existing shipped workflow template (`workflows/*.json`) is loaded, validated, and substituted by the exact unmodified `WorkflowLibrary`/`WorkflowBuilder._substitute()` code path when no fragments are selected — verified by the golden regression test in §19.
- `WorkflowLibrary.validate()`'s existing generic node-schema check is untouched, so fragments do not require any change to template validation rules.

---

## 22. Migration Strategy

No data migration is required — this is a purely additive code and template change.

1. **Land Module 7 Phase 3 code** (`generation_components/`, `WorkflowBuilder`/`ImageGeneratorPipeline` extensions, new config, new exceptions) with the golden regression test in place first, as a safety net.
2. **Land fragment files** for `controlnet_depth` and `controlnet_canny` (the two conditioning kinds Module 10 already has a real field for, even though `canny_path` currently always resolves empty — see §0). These are the only fragments capable of actually activating against the current repository state.
3. **Wire `main.py`'s call site** to pass `generation_bundle=...` from the already-existing `AssetComposer.prepare_generation_workspace()` call.
4. **Leave `regional_mask_conditioning`, `ipadapter_reference`, `controlnet_segmentation`, and `text_exclusion_mask` fragments in place but dormant**, documented as such, pending the upstream schema/producer work called out in §7.3 and §0 — which is intentionally out of scope here and owned by whoever next touches Module 10/Module 8.
5. No backfill of historical `data/generated_thumbnails/` output is needed or attempted; Phase 3 only affects generation from this point forward.

---

## 23. Risks

- **Module 10's `canny_path` is currently always `None`** (`GenerationBundleBuilder` never sets it — confirmed in §0). The `controlnet_canny` fragment is fully built but will never fire against the current repository until that Module 10 defect is fixed independently. Risk: stakeholders may expect canny conditioning to "just work" once Phase 3 ships; it will not, and this document flags why up front.
- **Fragment/attachment-point coupling to template `_meta`.** Requiring each base template to declare `_meta.attachment_points` means the eleven existing `workflows/*.json` files each need one small, additive `_meta` block update (not a graph change) before fragments can attach to them at all. This is a small, mechanical, low-risk change, but it is a required precondition, not automatic.
- **ComfyUI custom-node availability varies by operator install.** `CapabilityProbe` mitigates but cannot eliminate the risk of a fragment referencing a node type (e.g., a specific `IPAdapterApply` implementation) whose exact name differs across ComfyUI custom-node package versions; fragment authors must keep `_meta.required_node_types` accurate as the operator's environment evolves.
- **Silent under-conditioning.** Because every new capability fails soft by design (§14), a misconfigured environment (e.g., ComfyUI custom nodes missing) degrades generation quality without erroring. This is the correct behavior per the design goals, but it means monitoring (§16 logging, plus `GenerationMetrics`) is the only signal an operator has that fragments are being silently dropped — dashboards/alerting on the new log lines is recommended but out of scope for this document.
- **Determinism under fragment chaining order.** If a future contributor adds a new fragment type without extending the fixed ordering constant in §9.1, two builds could nondeterministically differ in attachment order if that constant is ever accidentally replaced with dict/set iteration. This is called out explicitly as an implementation constraint, not left implicit.

---

## 24. Implementation Phases

**Phase 3.1 — Foundation (no behavior change):**
`generation_components/interfaces.py`, `GenerationBundleLoader`, `CompositionWorkspaceLoader`, `ConditioningAssetResolver` (with all-empty-context behavior verified), new exceptions, new config keys, golden regression test suite for existing templates. Ships dark — nothing calls these yet from `ImageGeneratorPipeline`.

**Phase 3.2 — WorkflowBuilder + Assembler wiring:**
`NodeFragmentLibrary`, `WorkflowGraphAssembler`, `WorkflowBuilder` extension (`_select_fragments`, `_slots()` additions, `conditioning` parameter), `_meta.attachment_points` added to all eleven existing templates. `ImageGeneratorPipeline`/`run_image_generation_pipeline` gain the two new optional parameters, still unused by `main.py`.

**Phase 3.3 — ControlNet fragments (live capability):**
`controlnet_depth.json`, `controlnet_canny.json`, `CapabilityProbe` + `object_info()` addition to `comfyui_client.py`. `main.py` updated to pass `generation_bundle` through. This is the first phase with an observable behavior change in production, gated entirely by `profile.controlnet_enabled` + asset presence.

**Phase 3.4 — Regional/workspace-aware conditioning:**
`regional_mask_conditioning.json`, `per_layer` population in `ConditioningAssetResolver`, `main.py` opt-in `composition_workspace` wiring.

**Phase 3.5 — Dormant capabilities, ready for activation:**
`ipadapter_reference.json`, `controlnet_segmentation.json`, `text_exclusion_mask.json`, plus the defensive `getattr`-based reads for the not-yet-existing `GenerationBundle` fields from §7.3. Shipped and tested (asserting they correctly select zero fragments against today's schema) so that the day Module 10/Module 8 are separately extended to populate these fields, Module 7 requires no further changes.
