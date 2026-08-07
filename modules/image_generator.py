"""Module 7 Phase 3 production foundation & Image Generation Pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageStat
from loguru import logger

_MODULES_DIR = Path(__file__).resolve().parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import (
    DEFAULT_ANALYSIS_DIR, DEFAULT_PROMPT_PACKAGE_DIR, DEFAULT_THUMBNAIL_DIR,
    LOG_DIR, MAX_IDENTITY_RETRIES, MAX_GENERATION_RETRIES,
    MODULE7_CODEFORMER_FIDELITY, MODULE7_GENERATION_PROFILES,
    MODULE7_IDENTITY_SIMILARITY_THRESHOLD, MODULE7_LOG_PATH, MODULE7_METRICS_PATH,
    MODULE7_MAX_CANDIDATES, MODULE7_NSFW_THRESHOLD, MODULE7_OUTPUT_DIR, MODULE7_PROFILE,
    MODULE7_PROFILE_PREFERENCE, MODULE7_QA_WEIGHTS, MODULE7_SAVE_CANDIDATES,
    MODULE7_STRATEGY_PACK, MODULE7_VRAM_HEADROOM_GB, MODULE7_WORKFLOW_GRAPH_CACHE_ENABLED,
    MODULE7_WORKFLOW_VERSION, MODULE7_PARALLEL_CANDIDATES,
    MODULE7_EDIT_CAPABLE_PROFILES, validate_module7_edit_reachability,
    MODULE7_CLUSTERING_THRESHOLD, MODULE7_RANKING_WEIGHTS,
    MODULE7_HUMAN_REVIEW_ENABLED, MODULE7_HUMAN_REVIEW_TIMEOUT_SECONDS,
    MODULE7_HUMAN_REVIEW_WORKSPACE_DIR, MODULE7_LEARNING_FEEDBACK_STORE_PATH,
    MODULE10_STYLE_PROMPT_ENABLED, MODULE10_STYLE_MIN_SAMPLES,
    MODULE10_STYLE_SIMILARITY_THRESHOLD, MODULE10_STYLE_PROMPT_WEIGHT,
    MODULE10_STYLE_DRIFT_WINDOW, MODULE10_STYLE_RANKING_WEIGHT,
    MODULE10_CREATOR_PROFILES_DIR,
    COMFYUI_WORKING_DIRECTORY, PROJECT_ROOT,
)
from creator_style import (
    StyleDriftDetector,
    StyleExtractor,
    StyleProfileStore,
    StylePromptGuidanceGenerator,
    StyleSimilarityEngine,
    StyleAwareRankingEngine,
)

from models import (
    CandidateManifest, CandidateManifestEntry, CandidateScore, CandidateStrategy,
    CompositionWorkspace, DecisionManifest, DesignBlueprint, EditPlan, FaceMatchResult, GeneratedAsset,
    GenerationBundle, GenerationMetrics, GenerationPlan, GenerationProfile,
    GenerationRunMetadata, ImageGenerationResult, PromptPackage, QualityAssuranceReport,
    StrategyPack, WorkflowTemplateRef,
)
from generation_components.staged_edit_stages import BaseLatentAnchor
from generation_components import (
    CandidateStrategyPlanner,
    CapabilityProbe,
    ConditioningAssetResolver,
    CompositionWorkspaceLoader,
    GenerationBundleLoader,
    GenerationConditioningContext,
    ICapabilityProbe,
    IConditioningAssetResolver,
    ICompositionWorkspaceLoader,
    IGenerationBundleLoader,
    INodeFragmentLibrary,
    IWorkflowGraphAssembler,
    NodeFragmentLibrary,
    StrategyPackLibrary,
    StrategyPackResolver,
    WorkflowGraphAssembler,
    CandidateClusteringEngine,
    CandidateRankingEngine,
    SelectionExplainer,
    HumanReviewWorkspace,
    LearningFeedbackStore,
    WorkflowGraphCache,
    RegionPlanValidator,
    BaseLatentStage,
    MaskedCompositeStage,
    BackgroundEditStage,
    ObjectEditStage,
    TypographyStage,
    HarmonizationStage,
)

from module7_exceptions import (
    ArtifactWriteError, CandidateGenerationTimeoutError, ComfyUIConnectionError,
    ComfyUIQueueError, IdentityPreservationError, MetricsWriteError, MissingCustomNodeError,
    Module7Error, NoEligibleCandidateError, ProfileDowngradedWarning, PromptPackageInvalidError,
    QualityAssuranceError, ReferenceAssetError, StrategyPackError, VRAMExhaustedError,
    WorkflowBuildError, WorkflowTemplateError,
)
from observability.generation_trace import GenerationTraceRecorder
from workflow_library import WorkflowLibrary


_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"
_PLACEHOLDER_PREFIX = "{{"
_PLACEHOLDER_SUFFIX = "}}"


def _configure_logger() -> None:
    """Attach the Module 7 rotating Loguru sink using project conventions."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(str(MODULE7_LOG_PATH), rotation="10 MB", retention="30 days",
               format=_LOG_FORMAT, level="DEBUG", enqueue=True)


_configure_logger()


def canonical_json_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible data."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_package_hash(package: PromptPackage) -> str:
    """Hash the exact validated Module 6 payload, excluding no fields."""
    return canonical_json_hash(package.model_dump(mode="json"))


def generation_hash(
    workflow_hash: str,
    package_hash: str,
    checkpoint_hash: str | None,
    lora_hashes: list[str],
    controlnet_hashes: list[str],
    seed: int,
    profile_name: str,
) -> str:
    """Compute the architecture-defined aggregate reproducibility hash."""
    return canonical_json_hash({
        "workflow_hash": workflow_hash, "prompt_package_hash": package_hash,
        "checkpoint_hash": checkpoint_hash, "lora_hashes": lora_hashes,
        "controlnet_hashes": controlnet_hashes, "seed": seed,
        "profile_name": profile_name,
    })


@dataclass(frozen=True)
class ReferenceAssets:
    """Local references available to a workflow; Phase 1 never derives embeddings."""

    source_thumbnail_path: Path
    analysis_path: Path | None = None
    face_crop_path: Path | None = None


@dataclass(frozen=True)
class BuiltWorkflow:
    """Fully materialized local workflow graph and its immutable provenance."""

    graph: dict[str, Any]
    workflow_ref: WorkflowTemplateRef
    workflow_hash: str


class PromptPackageLoader:
    """Load persisted Module 6 output and reject unusable packages at the boundary."""

    def __init__(self, package_dir: Path = DEFAULT_PROMPT_PACKAGE_DIR) -> None:
        self.package_dir = Path(package_dir)

    def path_for(self, video_id: str) -> Path:
        """Return the canonical package path for one video ID."""
        return self.package_dir / f"{video_id}.json"

    def load(self, video_id: str) -> PromptPackage:
        """Read one package, validate Pydantic schema, and reject error status."""
        path = self.path_for(video_id)
        try:
            package = PromptPackage.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise PromptPackageInvalidError(f"Could not load PromptPackage for {video_id}: {exc}") from exc
        if package.video_id != video_id:
            raise PromptPackageInvalidError(f"PromptPackage video_id mismatch: requested {video_id}, found {package.video_id}")
        if package.status == "error":
            raise PromptPackageInvalidError(package.error_message or f"PromptPackage {video_id} has error status")
        candidates = getattr(package.generation_parameters, "num_candidates", 1)
        if candidates not in {1, 2, 4, 8}:
            raise PromptPackageInvalidError("GenerationParameters.num_candidates must be one of 1, 2, 4, or 8")
        logger.info("Loaded PromptPackage for video_id={video_id}, hash={hash}", video_id=video_id, hash=prompt_package_hash(package))
        return package


class ReferenceAssetResolver:
    """Resolve local source assets only; no CV, embeddings, or network I/O occur here."""

    def __init__(self, thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR, analysis_dir: Path = DEFAULT_ANALYSIS_DIR) -> None:
        self.thumbnail_dir = Path(thumbnail_dir)
        self.analysis_dir = Path(analysis_dir)

    def resolve(self, package: PromptPackage) -> ReferenceAssets:
        """Locate source thumbnail and optional precomputed Module 4 analysis."""
        candidates = [self.thumbnail_dir / f"{package.video_id}{suffix}" for suffix in (".jpg", ".jpeg", ".png", ".webp")]
        thumbnail = next((path for path in candidates if path.is_file()), None)
        if thumbnail is None:
            raise ReferenceAssetError(f"No source thumbnail found for {package.video_id} in {self.thumbnail_dir}")
        analysis = self.analysis_dir / f"{package.video_id}.json"
        logger.info("Resolved reference thumbnail for video_id={video_id}: {path}", video_id=package.video_id, path=thumbnail)
        return ReferenceAssets(source_thumbnail_path=thumbnail, analysis_path=analysis if analysis.is_file() else None)


class ProfileSelector:
    """Deterministically select a configured profile from a measured VRAM reading."""

    def __init__(self, profiles: Mapping[str, GenerationProfile] = MODULE7_GENERATION_PROFILES,
                 headroom_gb: float = MODULE7_VRAM_HEADROOM_GB) -> None:
        self.profiles = dict(profiles)
        self.headroom_gb = headroom_gb
        validate_qa_weights(MODULE7_QA_WEIGHTS)
        validate_module7_edit_reachability()

    def select(self, available_vram_gb: float, requested_profile: str = MODULE7_PROFILE) -> GenerationProfile:
        """Choose the richest fitting profile, with logged explicit-request downgrade."""
        usable_vram = available_vram_gb - self.headroom_gb
        if usable_vram < 0:
            usable_vram = 0.0
        if requested_profile != "auto":
            requested = self.profiles.get(requested_profile)
            if requested is None:
                raise Module7Error(f"Unknown Module 7 generation profile: {requested_profile}")
            if requested.expected_vram_gb <= usable_vram:
                logger.info("Selected requested profile={profile} with usable_vram_gb={vram:.2f}", profile=requested.name, vram=usable_vram)
                return requested
            logger.warning("Requested profile={profile} does not fit usable_vram_gb={vram:.2f}; selecting fallback", profile=requested.name, vram=usable_vram)
        for name in MODULE7_PROFILE_PREFERENCE:
            profile = self.profiles.get(name)
            if profile is not None and profile.expected_vram_gb <= usable_vram:
                logger.info("Selected profile={profile} with usable_vram_gb={vram:.2f}", profile=profile.name, vram=usable_vram)
                return profile
        low = self.profiles.get("PROFILE_LOW_VRAM")
        if low is None:
            raise Module7Error("No configured Module 7 profile can satisfy the VRAM requirement")
        logger.warning("No profile fits usable_vram_gb={vram:.2f}; selecting documented low-VRAM fallback={profile}", vram=usable_vram, profile=low.name)
        return low


def _conditioning_context_hash(ctx: GenerationConditioningContext | None) -> str:
    """Compute deterministic hash for GenerationConditioningContext."""
    if ctx is None:
        return "none"
    return canonical_json_hash({
        "source_thumbnail_path": str(ctx.source_thumbnail_path) if ctx.source_thumbnail_path else None,
        "canvas_width": ctx.canvas_width,
        "canvas_height": ctx.canvas_height,
        "role_image_paths": {k: str(v) for k, v in ctx.role_image_paths.items()},
        "role_mask_paths": {k: str(v) for k, v in ctx.role_mask_paths.items()},
        "depth_path": str(ctx.depth_path) if ctx.depth_path else None,
        "canny_path": str(ctx.canny_path) if ctx.canny_path else None,
        "segmentation_path": str(ctx.segmentation_path) if ctx.segmentation_path else None,
        "ip_adapter_reference_paths": {k: str(v) for k, v in ctx.ip_adapter_reference_paths.items()},
        "text_exclusion_mask_path": str(ctx.text_exclusion_mask_path) if ctx.text_exclusion_mask_path else None,
        "layer_order": ctx.layer_order,
    })




def stage_image_for_comfyui(raw_path: str | Path | None, video_id: str = "") -> str:
    """Stage an image file into ComfyUI's input directory and return its relative filename.

    ComfyUI's stock LoadImage node resolves filenames relative to its input directory.
    Passing absolute Windows filesystem paths causes HTTP 400 validation failures.
    This function copies the image into the configured ComfyUI input directory and
    returns the relative filename.
    """
    if not raw_path:
        return ""

    src_path = Path(raw_path)

    # If string is already a filename without path components or drive letter and not existing as absolute
    if not src_path.is_absolute() and not src_path.exists():
        return src_path.name

    if not src_path.exists() or not src_path.is_file():
        return ""

    if COMFYUI_WORKING_DIRECTORY and COMFYUI_WORKING_DIRECTORY.exists():
        input_dir = COMFYUI_WORKING_DIRECTORY / "input"
    else:
        input_dir = PROJECT_ROOT / "data" / "comfyui_input"

    input_dir.mkdir(parents=True, exist_ok=True)

    # If src_path is already inside input_dir, return relative posix path
    try:
        rel = src_path.relative_to(input_dir)
        return rel.as_posix()
    except ValueError:
        pass

    clean_vid = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (video_id or "")).strip("_")
    if clean_vid and not src_path.name.startswith(clean_vid):
        staged_filename = f"{clean_vid}_{src_path.name}"
    else:
        staged_filename = src_path.name

    dst_path = input_dir / staged_filename

    if not dst_path.exists() or src_path.stat().st_mtime > dst_path.stat().st_mtime:
        try:
            shutil.copy2(src_path, dst_path)
            logger.info("Staged image for ComfyUI input dir: {src} -> {dst}", src=src_path, dst=dst_path)
        except Exception as exc:
            logger.warning(
                "Failed to copy image to ComfyUI input dir: {src} -> {dst}: {exc}",
                src=src_path,
                dst=dst_path,
                exc=exc,
            )

    return staged_filename


def ensure_default_mask_for_comfyui(video_id: str = "", width: int = 1280, height: int = 720) -> str:
    """Generate and stage a default mask PNG image in ComfyUI's input directory and return its relative filename."""
    if COMFYUI_WORKING_DIRECTORY and COMFYUI_WORKING_DIRECTORY.exists():
        input_dir = COMFYUI_WORKING_DIRECTORY / "input"
    else:
        input_dir = PROJECT_ROOT / "data" / "comfyui_input"
    input_dir.mkdir(parents=True, exist_ok=True)

    clean_vid = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (video_id or "")).strip("_")
    filename = f"{clean_vid}_default_mask.png" if clean_vid else "default_mask.png"
    file_path = input_dir / filename

    if not file_path.exists():
        img = Image.new("L", (width, height), 255)
        img.save(file_path, format="PNG")
        logger.info("Generated default mask image for ComfyUI: {path}", path=file_path)

    return filename


def ensure_default_image_for_comfyui(video_id: str = "", width: int = 1280, height: int = 720) -> str:
    """Generate and stage a default RGB image in ComfyUI's input directory and return its relative filename."""
    if COMFYUI_WORKING_DIRECTORY and COMFYUI_WORKING_DIRECTORY.exists():
        input_dir = COMFYUI_WORKING_DIRECTORY / "input"
    else:
        input_dir = PROJECT_ROOT / "data" / "comfyui_input"
    input_dir.mkdir(parents=True, exist_ok=True)

    clean_vid = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (video_id or "")).strip("_")
    filename = f"{clean_vid}_default_image.png" if clean_vid else "default_image.png"
    file_path = input_dir / filename

    if not file_path.exists():
        img = Image.new("RGB", (width, height), (128, 128, 128))
        img.save(file_path, format="PNG")
        logger.info("Generated default RGB image for ComfyUI: {path}", path=file_path)

    return filename


class WorkflowBuilder:
    """Fills named template slots and materializes executable ComfyUI graphs."""

    def __init__(
        self,
        fragment_library: NodeFragmentLibrary | None = None,
        graph_assembler: WorkflowGraphAssembler | None = None,
        capability_probe: ICapabilityProbe | None = None,
        controlnet_resolver: Any | None = None,
    ) -> None:
        self.fragment_library = fragment_library or NodeFragmentLibrary()
        self.graph_assembler = graph_assembler or WorkflowGraphAssembler()
        self.capability_probe = capability_probe
        if controlnet_resolver is not None:
            self.controlnet_resolver = controlnet_resolver
        else:
            from generation_components.model_discovery_service import ModelDiscoveryService
            from generation_components.controlnet_capability_resolver import ControlNetCapabilityResolver
            probe_obj = capability_probe if isinstance(capability_probe, CapabilityProbe) else None
            client_obj = getattr(capability_probe, "client", None) if capability_probe else None
            discovery = ModelDiscoveryService(client=client_obj, probe=probe_obj)
            self.controlnet_resolver = ControlNetCapabilityResolver(discovery_service=discovery)

    def build_base(
        self,
        profile: GenerationProfile,
        workflow_ref: WorkflowTemplateRef,
        library: WorkflowLibrary | None = None,
        conditioning: GenerationConditioningContext | None = None,
    ) -> dict[str, Any]:
        """Materialize base template dict and fragment assembly prior to per-candidate slot substitution."""
        source = library or WorkflowLibrary(Path(workflow_ref.template_path).parent)
        template = source.load(Path(workflow_ref.template_path))
        raw_graph = template.get("graph", {})

        if conditioning is not None:
            fragment_ids = self._select_fragments(profile, conditioning, workflow_ref=workflow_ref)
            fragment_dicts = []
            for fid in fragment_ids:
                frag_data = copy.deepcopy(self.fragment_library.load(fid))
                if self.capability_probe and not self.capability_probe.is_fragment_supported(frag_data):
                    logger.warning(
                        "Fragment '{fragment_id}' dropped: required node types not available in ComfyUI",
                        fragment_id=fid,
                    )
                    continue

                meta = frag_data.setdefault("_meta", {})
                cap_name = None
                if fid.startswith("controlnet_"):
                    raw_cap = fid.replace("controlnet_", "").replace("_t2iadapter", "")
                    if raw_cap in ("depth", "canny", "segmentation"):
                        cap_name = raw_cap
                if cap_name and hasattr(self, "controlnet_resolver") and self.controlnet_resolver:
                    res = self.controlnet_resolver.resolve(cap_name)
                    meta["requested_capability"] = res.capability
                    meta["resolved_model"] = res.resolved_filename
                    meta["resolution_source"] = res.resolution_source
                    meta["fallback_path"] = (res.resolution_source != "legacy_exact_match")
                    meta["compatibility_decision"] = res.compatibility_decision

                fragment_dicts.append(frag_data)

            assembled = self.graph_assembler.assemble(
                {"_meta": template.get("_meta", {}), "graph": raw_graph},
                fragment_dicts,
                conditioning,
                profile,
            )
            return assembled.get("graph", raw_graph)
        return raw_graph

    def build(
        self,
        package: PromptPackage,
        profile: GenerationProfile,
        workflow_ref: WorkflowTemplateRef,
        reference_assets: ReferenceAssets | None = None,
        library: WorkflowLibrary | None = None,
        conditioning: GenerationConditioningContext | None = None,
        plan: GenerationPlan | None = None,
        cache: Any | None = None,
    ) -> BuiltWorkflow:
        """Fill named template slots and return the exact graph plus its hash."""
        cond_hash = _conditioning_context_hash(conditioning)
        key = (workflow_ref.template_path, workflow_ref.workflow_version, profile.name, cond_hash)

        base_unsubstituted = cache.get(key) if cache is not None else None
        if base_unsubstituted is None:
            base_unsubstituted = self.build_base(profile, workflow_ref, library=library, conditioning=conditioning)
            if cache is not None:
                cache.put(key, base_unsubstituted)

        slots = self._slots(package, profile, reference_assets, conditioning, plan, controlnet_resolver=self.controlnet_resolver)
        try:
            final_graph = self._substitute(base_unsubstituted, slots)
        except KeyError as exc:
            raise WorkflowBuildError(f"Template {workflow_ref.template_name} uses unknown placeholder {exc.args[0]}") from exc
        if not isinstance(final_graph, dict):
            raise WorkflowBuildError("Resolved workflow graph must be an object")

        if self.capability_probe is not None:
            self.capability_probe.validate_workflow_graph(
                final_graph,
                workflow_name=workflow_ref.template_name,
                raise_on_missing=True,
            )

        width = package.generation_parameters.width if package else 1280
        height = package.generation_parameters.height if package else 720
        video_id = package.video_id if package else ""
        default_mask_file = None
        for node in final_graph.values():
            if isinstance(node, dict) and node.get("class_type") == "LoadImage":
                inputs = node.get("inputs")
                if isinstance(inputs, dict) and not inputs.get("image"):
                    if default_mask_file is None:
                        default_mask_file = ensure_default_mask_for_comfyui(video_id, width, height)
                    inputs["image"] = default_mask_file
                    logger.warning(
                        "LoadImage node given empty image placeholder; assigned default mask asset '{file}'",
                        file=default_mask_file,
                    )

        workflow_hash = canonical_json_hash(final_graph)
        logger.info("Built workflow template={template}, version={version}, workflow_hash={hash}", template=workflow_ref.template_name, version=workflow_ref.workflow_version, hash=workflow_hash)
        return BuiltWorkflow(graph=final_graph, workflow_ref=workflow_ref, workflow_hash=workflow_hash)

    def _select_fragments(
        self,
        profile: GenerationProfile,
        conditioning: GenerationConditioningContext | None,
        workflow_ref: WorkflowTemplateRef | None = None,
    ) -> list[str]:
        if conditioning is None:
            return []
        fragments: list[str] = []

        is_edit_workflow = workflow_ref is not None and (
            workflow_ref.template_name.endswith("_edit") or "_edit" in workflow_ref.template_path
        )
        if is_edit_workflow:
            fragments.append("inpaint_base")
            fragments.append("edit_region_mask")

        if profile.controlnet_enabled and conditioning.depth_path is not None:
            res = self.controlnet_resolver.resolve("depth") if hasattr(self, "controlnet_resolver") and self.controlnet_resolver else None
            if res and res.resolution_source != "unresolved":
                fragments.append(res.fragment_variant)
            elif res and res.resolution_source == "unresolved":
                logger.warning("Dropping ControlNet depth fragment: no matching depth model installed in ComfyUI")
            else:
                fragments.append("controlnet_depth")

        if profile.controlnet_enabled and conditioning.canny_path is not None:
            res = self.controlnet_resolver.resolve("canny") if hasattr(self, "controlnet_resolver") and self.controlnet_resolver else None
            if res and res.resolution_source != "unresolved":
                fragments.append(res.fragment_variant)
            elif res and res.resolution_source == "unresolved":
                logger.warning("Dropping ControlNet canny fragment: no matching canny model installed in ComfyUI")
            else:
                fragments.append("controlnet_canny")

        if profile.controlnet_enabled and conditioning.segmentation_path is not None:
            res = self.controlnet_resolver.resolve("segmentation") if hasattr(self, "controlnet_resolver") and self.controlnet_resolver else None
            if res and res.resolution_source != "unresolved":
                fragments.append(res.fragment_variant)
            elif res and res.resolution_source == "unresolved":
                logger.warning("Dropping ControlNet segmentation fragment: no matching segmentation model installed in ComfyUI")
            else:
                fragments.append("controlnet_segmentation")
        if profile.ipadapter_enabled and conditioning.ip_adapter_reference_paths:
            fragments.append("ipadapter_reference")
        if profile.ipadapter_enabled and conditioning.role_image_paths and ("creator_face" in conditioning.role_image_paths or "person" in conditioning.role_image_paths):
            fragments.append("ipadapter_faceid")
        if profile.ipadapter_enabled and len(conditioning.role_image_paths) > 1 and any(k.startswith("object_") for k in conditioning.role_image_paths):
            fragments.append("multi_object_reference")
        if conditioning.text_exclusion_mask_path is not None:
            fragments.append("text_exclusion_mask")
        if conditioning.per_layer and any(layer.mask_path is not None for layer in conditioning.per_layer.values()):
            fragments.append("regional_mask_conditioning")
        return fragments

    @staticmethod
    def _slots(
        package: PromptPackage,
        profile: GenerationProfile,
        references: ReferenceAssets | None,
        conditioning: GenerationConditioningContext | None = None,
        plan: GenerationPlan | None = None,
        controlnet_resolver: Any | None = None,
    ) -> dict[str, Any]:
        video_id = package.video_id if package else ""
        positive = package.positive_prompt if package else ""

        raw_neg_parts = [
            package.negative_prompt if package else "",
            *(package.rendering_constraints if package else []),
            *(package.safety_constraints if package else []),
        ]
        if plan and plan.negative_constraints:
            for nc in plan.negative_constraints:
                if nc not in raw_neg_parts:
                    raw_neg_parts.append(nc)

        neg_parts = []
        for part in raw_neg_parts:
            part_str = str(part).strip()
            if not part_str:
                continue
            # Exclude positive preservation directives that were mistakenly passed
            lower_part = part_str.lower()
            if lower_part.startswith("preserve") or "elements exactly" in lower_part:
                continue
            if part_str not in neg_parts:
                neg_parts.append(part_str)

        negative = ", ".join(neg_parts)

        headline_text = plan.headline if plan else ""
        headline_zone_x = plan.headline_placement_zone.x_min if (plan and plan.headline_placement_zone) else 0
        headline_zone_y = plan.headline_placement_zone.y_min if (plan and plan.headline_placement_zone) else 0
        headline_zone_w = plan.headline_placement_zone.width if (plan and plan.headline_placement_zone) else 0
        headline_zone_h = plan.headline_placement_zone.height if (plan and plan.headline_placement_zone) else 0

        width = package.generation_parameters.width if package else 1280
        height = package.generation_parameters.height if package else 720

        raw_thumb = None
        if conditioning and conditioning.source_thumbnail_path:
            raw_thumb = conditioning.source_thumbnail_path
        elif references and references.source_thumbnail_path:
            raw_thumb = references.source_thumbnail_path
        thumb_path = stage_image_for_comfyui(raw_thumb, video_id) if raw_thumb else ""
        if not thumb_path:
            thumb_path = ensure_default_image_for_comfyui(video_id, width, height)

        raw_edit_mask = None
        if conditioning and hasattr(conditioning, "role_mask_paths") and conditioning.role_mask_paths:
            for mask_key in ("edit_mask", "background", "object", "person", "creator_face"):
                if mask_key in conditioning.role_mask_paths and conditioning.role_mask_paths[mask_key]:
                    raw_edit_mask = conditioning.role_mask_paths[mask_key]
                    break
            if raw_edit_mask is None:
                first_mask = next((p for p in conditioning.role_mask_paths.values() if p), None)
                if first_mask:
                    raw_edit_mask = first_mask
        if raw_edit_mask is None and conditioning and conditioning.text_exclusion_mask_path:
            raw_edit_mask = conditioning.text_exclusion_mask_path
        edit_mask = stage_image_for_comfyui(raw_edit_mask, video_id) if raw_edit_mask else ""
        if not edit_mask:
            edit_mask = ensure_default_mask_for_comfyui(video_id, width, height)

        raw_fg = None
        if conditioning and hasattr(conditioning, "role_image_paths") and conditioning.role_image_paths:
            raw_fg = (
                conditioning.role_image_paths.get("foreground")
                or conditioning.role_image_paths.get("creator_face")
                or conditioning.role_image_paths.get("person")
                or conditioning.role_image_paths.get("object")
            )
        raw_bg = conditioning.role_image_paths.get("background") if (conditioning and conditioning.role_image_paths) else None
        raw_person_mask = conditioning.role_mask_paths.get("person") if (conditioning and conditioning.role_mask_paths) else None
        raw_object_mask = conditioning.role_mask_paths.get("object") if (conditioning and conditioning.role_mask_paths) else None
        raw_depth = conditioning.depth_path if conditioning else None
        raw_canny = conditioning.canny_path if conditioning else None
        raw_seg = conditioning.segmentation_path if conditioning else None
        raw_text_mask = conditioning.text_exclusion_mask_path if conditioning else None

        res_depth = controlnet_resolver.resolve("depth") if controlnet_resolver else None
        res_canny = controlnet_resolver.resolve("canny") if controlnet_resolver else None
        res_seg = controlnet_resolver.resolve("segmentation") if controlnet_resolver else None

        resolved_depth_file = (res_depth.resolved_filename if res_depth else None) or "controlnet_depth_sdxl.safetensors"
        resolved_canny_file = (res_canny.resolved_filename if res_canny else None) or "controlnet_canny_sdxl.safetensors"
        resolved_seg_file = (res_seg.resolved_filename if res_seg else None) or "controlnet_seg_sdxl.safetensors"

        return {
            "checkpoint": profile.checkpoint, "positive_prompt": positive,
            "negative_prompt": negative, "background_prompt": package.background_instructions,
            "seed": package.generation_parameters.seed, "steps": profile.steps,
            "cfg": profile.cfg, "sampler": profile.sampler, "scheduler": profile.scheduler,
            "width": package.generation_parameters.width, "height": package.generation_parameters.height,
            "controlnet_enabled": profile.controlnet_enabled, "ipadapter_enabled": profile.ipadapter_enabled,
            "restoration": profile.restoration, "restoration_fidelity": profile.restoration_fidelity,
            "upscaler": profile.upscaler,
            "output_filename_prefix": WorkflowBuilder._output_filename_prefix(package),
            "source_thumbnail_path": thumb_path,
            "edit_mask_path": edit_mask,
            "denoise_strength": 0.75,
            "foreground_image_path": stage_image_for_comfyui(raw_fg, video_id) if raw_fg else "",
            "background_image_path": stage_image_for_comfyui(raw_bg, video_id) if raw_bg else "",
            "person_mask_path": stage_image_for_comfyui(raw_person_mask, video_id) if raw_person_mask else "",
            "object_mask_path": stage_image_for_comfyui(raw_object_mask, video_id) if raw_object_mask else "",
            "depth_map_path": stage_image_for_comfyui(raw_depth, video_id) if raw_depth else "",
            "canny_map_path": stage_image_for_comfyui(raw_canny, video_id) if raw_canny else "",
            "segmentation_map_path": stage_image_for_comfyui(raw_seg, video_id) if raw_seg else "",
            "text_exclusion_mask_path": stage_image_for_comfyui(raw_text_mask, video_id) if raw_text_mask else "",
            "resolved_depth_controlnet": resolved_depth_file,
            "resolved_canny_controlnet": resolved_canny_file,
            "resolved_segmentation_controlnet": resolved_seg_file,
            "controlnet_depth_strength": 0.55,
            "controlnet_canny_strength": 0.45,
            "controlnet_segmentation_strength": 0.5,
            "ipadapter_weight": 0.6,
            "headline_text": headline_text,
            "headline_zone_x": headline_zone_x,
            "headline_zone_y": headline_zone_y,
            "headline_zone_w": headline_zone_w,
            "headline_zone_h": headline_zone_h,
        }


    @staticmethod
    def _output_filename_prefix(package: PromptPackage) -> str:
        safe_video_id = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in package.video_id.strip()
        ).strip("_")
        return f"module7_{safe_video_id or 'output'}"

    @classmethod
    def _substitute(cls, value: Any, slots: Mapping[str, Any]) -> Any:
        if isinstance(value, dict):
            return {key: cls._substitute(item, slots) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._substitute(item, slots) for item in value]
        if isinstance(value, str) and value.startswith(_PLACEHOLDER_PREFIX) and value.endswith(_PLACEHOLDER_SUFFIX):
            return slots[value[2:-2]]
        return value


class ArtifactWriter:
    """Atomically persist Phase 1 metadata manifests under each video output directory."""

    def __init__(self, output_dir: Path = MODULE7_OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)

    def manifest_path(self, video_id: str) -> Path:
        return self.output_dir / video_id / f"{video_id}_manifest.json"

    def candidate_manifest_path(self, video_id: str) -> Path:
        return self.output_dir / video_id / "candidate_manifest.json"

    def generation_metadata_path(self, video_id: str) -> Path:
        return self.output_dir / video_id / "generation_metadata.json"

    def write_manifest(self, result: ImageGenerationResult) -> Path:
        """Write one complete manifest with temp-file-then-replace semantics."""
        target = self.manifest_path(result.video_id)
        temporary = target.with_suffix(".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(target)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ArtifactWriteError(f"Could not write Module 7 manifest to {target}: {exc}") from exc
        logger.info("Wrote Module 7 manifest for video_id={video_id}: {path}", video_id=result.video_id, path=target)
        return target

    def write_candidate_manifest(self, manifest: CandidateManifest) -> Path:
        """Write candidate manifest with temp-file-then-replace semantics."""
        target = self.candidate_manifest_path(manifest.video_id)
        temporary = target.with_suffix(".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(target)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ArtifactWriteError(f"Could not write candidate manifest to {target}: {exc}") from exc
        logger.info("Wrote candidate manifest for video_id={video_id}: {path}", video_id=manifest.video_id, path=target)
        return target

    def write_generation_metadata(self, metadata: GenerationRunMetadata) -> Path:
        """Write generation run metadata with temp-file-then-replace semantics."""
        target = self.generation_metadata_path(metadata.video_id)
        temporary = target.with_suffix(".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(target)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ArtifactWriteError(f"Could not write generation metadata to {target}: {exc}") from exc
        logger.info("Wrote generation metadata for video_id={video_id}: {path}", video_id=metadata.video_id, path=target)
        return target



class MetricsCollector:
    """Passive append-only JSONL metrics sink; it has no control-flow responsibilities."""

    def __init__(self, metrics_path: Path = MODULE7_METRICS_PATH) -> None:
        self.metrics_path = Path(metrics_path)

    def append(self, metrics: GenerationMetrics) -> None:
        """Append one flushed JSON Lines record without exposing partial line content."""
        try:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(metrics.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise MetricsWriteError(f"Could not append Module 7 metrics to {self.metrics_path}: {exc}") from exc
        logger.debug("Appended Module 7 metrics for video_id={video_id}", video_id=metrics.video_id)


def validate_qa_weights(weights: Mapping[str, float]) -> None:
    """Reject incomplete or non-normalized QA configuration rather than correcting it silently."""
    required = {"identity_score", "face_quality_score", "composition_score", "text_safe_zone_score", "object_preservation_score", "color_compliance_score"}
    if set(weights) != required or any(value < 0 for value in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise Module7Error("MODULE7_QA_WEIGHTS must contain every quality signal and sum exactly to 1.0")


def utc_now() -> str:
    """Return the standard timestamp format used by Module 7 manifests and metrics."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Phase 3 Stage Implementations
# ---------------------------------------------------------------------------

def cosine_similarity(u: Sequence[float], v: Sequence[float]) -> float:
    """Compute cosine similarity between two feature vectors, bounded to [0.0, 1.0]."""
    arr_u = np.asarray(u, dtype=float)
    arr_v = np.asarray(v, dtype=float)
    norm_u = np.linalg.norm(arr_u)
    norm_v = np.linalg.norm(arr_v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    sim = float(np.dot(arr_u, arr_v) / (norm_u * norm_v))
    return max(0.0, min(1.0, sim))


def _extract_face_embedding(image_path: Path) -> list[float] | None:
    """Extract face embedding for identity preservation, with deterministic fallback."""
    if not image_path.is_file():
        return None

    try:
        from thumbnail_intelligence import _get_face_app
        app = _get_face_app()
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            bgr = np.array(rgb)[:, :, ::-1]
            faces = app.get(bgr)
            if faces:
                largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                emb = getattr(largest, "normed_embedding", getattr(largest, "embedding", None))
                if emb is not None:
                    return [float(x) for x in emb]
    except Exception:
        pass

    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB").resize((64, 64))
            arr = np.array(img_rgb, dtype=float)
            flat = arr.flatten()
            norm = np.linalg.norm(flat)
            if norm > 0:
                flat /= norm
            return flat.tolist()
    except Exception:
        return None


def _calculate_face_quality_score(image_path: Path) -> float:
    """Calculate a sharpness/quality score in [0.0, 1.0] using Laplacian variance."""
    if not image_path.is_file():
        return 0.5
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L").resize((128, 128))
            arr = np.array(gray, dtype=float)
            laplacian = (
                4 * arr[1:-1, 1:-1]
                - arr[:-2, 1:-1] - arr[2:, 1:-1]
                - arr[1:-1, :-2] - arr[1:-1, 2:]
            )
            var = float(np.var(laplacian))
            score = 1.0 - (1.0 / (1.0 + var / 100.0))
            return max(0.0, min(1.0, score))
    except Exception:
        return 0.5


def _rgb_to_lab_mean_std(img: Image) -> tuple[np.ndarray, np.ndarray]:
    """Compute approximate CIELAB color space mean and stddev vectors from RGB image."""
    rgb = np.array(img.convert("RGB"), dtype=float) / 255.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92

    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = np.dot(rgb, matrix.T)

    xyz_ref = np.array([0.95047, 1.00000, 1.08883])
    xyz_scaled = xyz / xyz_ref

    mask_lab = xyz_scaled > 0.008856
    f_xyz = np.zeros_like(xyz_scaled)
    f_xyz[mask_lab] = xyz_scaled[mask_lab] ** (1 / 3)
    f_xyz[~mask_lab] = (7.787 * xyz_scaled[~mask_lab]) + (16 / 116)

    L = (116.0 * f_xyz[:, :, 1]) - 16.0
    a = 500.0 * (f_xyz[:, :, 0] - f_xyz[:, :, 1])
    b = 200.0 * (f_xyz[:, :, 1] - f_xyz[:, :, 2])

    lab_stack = np.stack([L, a, b], axis=-1)
    means = np.mean(lab_stack, axis=(0, 1))
    stddevs = np.std(lab_stack, axis=(0, 1))
    return means, stddevs


def _calculate_text_safe_zone_score(
    image_path: Path,
    package: PromptPackage,
    reference_assets: ReferenceAssets | None = None,
) -> float:
    """Calculate safe-zone collision score in [0.0, 1.0].

    Phase 0 Implementation:
    Evaluates edge/contrast clutter in the standard YouTube duration badge area
    (bottom-right: [85%..100% width, 80%..100% height]).
    In Phase 5, this is superseded by headline_placement_zone mask overlap checks.
    """
    if not image_path.is_file():
        return 0.5

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if w <= 0 or h <= 0:
                return 0.5

            bx0, by0 = int(w * 0.85), int(h * 0.80)
            crop_zone = img.crop((bx0, by0, w, h)).convert("L")
            arr = np.array(crop_zone, dtype=float)

            if arr.size == 0:
                return 1.0

            std_dev = float(np.std(arr))
            clutter_penalty = min(0.6, std_dev / 100.0)
            score = 1.0 - clutter_penalty
            return max(0.0, min(1.0, float(score)))
    except Exception as exc:
        logger.debug("Failed calculating text safe zone score for {path}: {exc}", path=image_path, exc=exc)
        return 0.5


def _calculate_object_preservation_score(
    image_path: Path,
    package: PromptPackage,
    reference_assets: ReferenceAssets | None = None,
) -> float:
    """Calculate object-directive preservation score in [0.0, 1.0].

    Phase 0 Implementation:
    Runs YOLO object detection over candidate image and compares detected objects against
    Module 4 analysis report if available (reference_assets.analysis_path) or structural
    variance. In Phase 4, this is updated to evaluate against Module 8 ObjectAsset records.
    """
    if not image_path.is_file():
        return 0.5

    ref_objects: list[dict[str, Any]] = []
    if reference_assets and reference_assets.analysis_path and reference_assets.analysis_path.is_file():
        try:
            data = json.loads(reference_assets.analysis_path.read_text(encoding="utf-8"))
            detected = data.get("objects", data.get("detected_objects", []))
            if isinstance(detected, list):
                ref_objects = detected
        except Exception as exc:
            logger.debug("Could not read reference analysis for object preservation: {exc}", exc=exc)

    try:
        from thumbnail_intelligence import _get_yolo_model
        model = _get_yolo_model()
        results = model(str(image_path), verbose=False)
        cand_classes: set[str] = set()
        if results and len(results) > 0 and hasattr(results[0], "boxes"):
            boxes = results[0].boxes
            names = getattr(results[0], "names", {})
            for b in boxes:
                cls_id = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])
                cls_name = names.get(cls_id, str(cls_id)).lower()
                cand_classes.add(cls_name)

        if ref_objects:
            ref_classes = {
                (obj.get("label") or obj.get("class_name") or "").lower()
                for obj in ref_objects if isinstance(obj, dict)
            }
            ref_classes.discard("")
            if ref_classes:
                matched = ref_classes.intersection(cand_classes)
                score = len(matched) / len(ref_classes)
                return max(0.0, min(1.0, float(score)))

        return 1.0 if len(cand_classes) > 0 else 0.8
    except Exception as exc:
        logger.debug("YOLO model not available for object preservation check: {exc}", exc=exc)

    if reference_assets and reference_assets.source_thumbnail_path and reference_assets.source_thumbnail_path.is_file():
        try:
            with Image.open(image_path) as cand_img, Image.open(reference_assets.source_thumbnail_path) as src_img:
                c_gray = cand_img.convert("L").resize((64, 64))
                s_gray = src_img.convert("L").resize((64, 64))
                c_arr = np.array(c_gray, dtype=float)
                s_arr = np.array(s_gray, dtype=float)
                diff = float(np.mean(np.abs(c_arr - s_arr)))
                score = 1.0 - (diff / 200.0)
                return max(0.0, min(1.0, float(score)))
        except Exception:
            pass

    return 1.0


def _calculate_color_compliance_score(
    image_path: Path,
    package: PromptPackage,
    reference_assets: ReferenceAssets | None = None,
) -> float:
    """Calculate color direction compliance score in [0.0, 1.0].

    Phase 0 Implementation:
    Measures Lab-space color distance against reference thumbnail when available,
    or checks HSV brightness/saturation against package color instructions.
    """
    if not image_path.is_file():
        return 0.5

    try:
        with Image.open(image_path) as cand_img:
            if reference_assets and reference_assets.source_thumbnail_path and reference_assets.source_thumbnail_path.is_file():
                with Image.open(reference_assets.source_thumbnail_path) as src_img:
                    cand_means, _ = _rgb_to_lab_mean_std(cand_img.resize((128, 128)))
                    src_means, _ = _rgb_to_lab_mean_std(src_img.resize((128, 128)))

                    delta_e = float(np.linalg.norm(cand_means - src_means))
                    score = 1.0 - (delta_e / 100.0)
                    return max(0.0, min(1.0, float(score)))

            hsv = cand_img.convert("HSV")
            arr = np.array(hsv, dtype=float)
            sat_mean = float(np.mean(arr[:, :, 1])) / 255.0
            val_mean = float(np.mean(arr[:, :, 2])) / 255.0

            score = 1.0
            if sat_mean < 0.15:
                score -= 0.2
            if val_mean < 0.2 or val_mean > 0.95:
                score -= 0.2

            color_instr = (package.color_instructions or "").lower()
            if "vibrant" in color_instr or "saturated" in color_instr:
                if sat_mean < 0.3:
                    score -= 0.2
            elif "dark" in color_instr or "moody" in color_instr:
                if val_mean > 0.6:
                    score -= 0.2

            return max(0.0, min(1.0, float(score)))
    except Exception as exc:
        logger.debug("Failed calculating color compliance score for {path}: {exc}", path=image_path, exc=exc)
        return 0.5


def _calculate_composition_score(
    image_path: Path,
    package: PromptPackage,
    reference_assets: ReferenceAssets | None = None,
) -> float:
    """Calculate composition adherence score in [0.0, 1.0].

    Phase 0 Implementation:
    Evaluates spatial energy distribution across 3x3 grid (Rule of Thirds) and structural
    gradient correlation against reference thumbnail if present.
    """
    if not image_path.is_file():
        return 0.5

    try:
        with Image.open(image_path) as cand_img:
            gray = cand_img.convert("L").resize((90, 90))
            arr = np.array(gray, dtype=float)

            sections = [
                arr[r * 30:(r + 1) * 30, c * 30:(c + 1) * 30]
                for r in range(3) for c in range(3)
            ]
            variances = [float(np.var(sec)) for sec in sections]
            total_var = sum(variances) + 1e-6

            props = [v / total_var for v in variances]
            max_prop = max(props)
            balance_penalty = max(0.0, (max_prop - 0.5) * 1.5)
            score = 1.0 - balance_penalty

            if reference_assets and reference_assets.source_thumbnail_path and reference_assets.source_thumbnail_path.is_file():
                try:
                    with Image.open(reference_assets.source_thumbnail_path) as src_img:
                        src_gray = src_img.convert("L").resize((90, 90))
                        src_arr = np.array(src_gray, dtype=float)
                        src_sections = [
                            src_arr[r * 30:(r + 1) * 30, c * 30:(c + 1) * 30]
                            for r in range(3) for c in range(3)
                        ]
                        src_vars = [float(np.var(sec)) for sec in src_sections]
                        src_total = sum(src_vars) + 1e-6
                        src_props = [v / src_total for v in src_vars]

                        norm_cand = np.linalg.norm(props)
                        norm_src = np.linalg.norm(src_props)
                        if norm_cand > 0 and norm_src > 0:
                            energy_sim = float(np.dot(props, src_props) / (norm_cand * norm_src))
                            score = 0.5 * score + 0.5 * energy_sim
                except Exception:
                    pass

            return max(0.0, min(1.0, float(score)))
    except Exception as exc:
        logger.debug("Failed calculating composition score for {path}: {exc}", path=image_path, exc=exc)
        return 0.5


class IdentityPreservationStage:
    """Verify generated face matches source face embedding within threshold."""

    def __init__(self, threshold: float = MODULE7_IDENTITY_SIMILARITY_THRESHOLD) -> None:
        self.threshold = threshold

    def verify(
        self,
        generated_image_path: Path,
        reference_assets: ReferenceAssets,
        threshold: float | None = None,
    ) -> FaceMatchResult:
        eff_threshold = threshold if threshold is not None else self.threshold

        has_ref_face = True
        if reference_assets.analysis_path and reference_assets.analysis_path.is_file():
            try:
                data = json.loads(reference_assets.analysis_path.read_text(encoding="utf-8"))
                face_analysis = data.get("face_analysis", {})
                if isinstance(face_analysis, dict):
                    has_ref_face = face_analysis.get("has_face", face_analysis.get("face_count", 0) > 0)
            except Exception:
                pass

        if not has_ref_face:
            logger.info("Identity preservation skipped for {path}: no reference face", path=generated_image_path)
            return FaceMatchResult(similarity=1.0, threshold=eff_threshold, passed=True, face_detected=False, skipped=True)

        ref_emb = _extract_face_embedding(reference_assets.source_thumbnail_path)
        gen_emb = _extract_face_embedding(generated_image_path)

        if ref_emb is None:
            logger.info("Identity preservation skipped for {path}: reference embedding absent", path=generated_image_path)
            return FaceMatchResult(similarity=1.0, threshold=eff_threshold, passed=True, face_detected=False, skipped=True)

        if gen_emb is None:
            logger.warning("Identity preservation check failed for {path}: no face detected in generated image", path=generated_image_path)
            return FaceMatchResult(similarity=0.0, threshold=eff_threshold, passed=False, face_detected=False, skipped=False)

        sim = cosine_similarity(ref_emb, gen_emb)
        passed = (sim >= eff_threshold)
        logger.info(
            "Identity check for {path}: similarity={sim:.4f}, threshold={thresh:.4f}, passed={passed}",
            path=generated_image_path, sim=sim, thresh=eff_threshold, passed=passed
        )
        return FaceMatchResult(similarity=sim, threshold=eff_threshold, passed=passed, face_detected=True, skipped=False)


class FaceRestorationStage:
    """Correct facial artifacts in the generated composite."""

    def __init__(self, codeformer_fidelity: float = MODULE7_CODEFORMER_FIDELITY) -> None:
        self.codeformer_fidelity = codeformer_fidelity

    def restore(
        self,
        image_path: Path,
        profile: GenerationProfile,
        output_path: Path | None = None,
    ) -> Path:
        target = output_path or image_path
        if profile.restoration == "none":
            if target != image_path:
                shutil.copyfile(image_path, target)
            return target

        restored_via_neural = False
        if profile.restoration in ("codeformer", "gfpgan"):
            try:
                import torch
                # Check for installed CodeFormer / GFPGAN inference modules
                if profile.restoration == "gfpgan":
                    from gfpgan import GFPGANer
                    restorer = GFPGANer(model_path="gfpgan/weights/GFPGANv1.4.pth", upscale=1, arch="clean", channel_multiplier=2)
                    img_cv = cv2.imread(str(image_path))
                    if img_cv is not None:
                        _, _, restored_img = restorer.enhance(img_cv, has_aligned=False, only_center_face=False, paste_back=True)
                        cv2.imwrite(str(target), restored_img)
                        restored_via_neural = True
            except Exception as exc:
                logger.debug("Neural face restoration ({mode}) unavailable: {exc}", mode=profile.restoration, exc=exc)

        if not restored_via_neural:
            with Image.open(image_path) as img:
                img_format = img.format or "PNG"
                enhanced = img.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.SHARPEN)
                temp_target = target.with_suffix(".tmp")
                target.parent.mkdir(parents=True, exist_ok=True)
                enhanced.save(temp_target, format=img_format)
                temp_target.replace(target)

        logger.info(
            "Face restoration completed for {path} using restoration={mode} (neural={neural})",
            path=target,
            mode=profile.restoration,
            neural=restored_via_neural,
        )
        return target


class BackgroundCompositor:
    """Composite preserved subject over newly generated background."""

    def composite(
        self,
        generated_image_path: Path,
        reference_assets: ReferenceAssets,
        package: PromptPackage,
        output_path: Path | None = None,
    ) -> Path:
        target = output_path or generated_image_path
        if not reference_assets.source_thumbnail_path.is_file():
            if target != generated_image_path:
                shutil.copyfile(generated_image_path, target)
            return target

        with Image.open(generated_image_path) as bg_img:
            img_format = bg_img.format or "PNG"
            temp_target = target.with_suffix(".tmp")
            target.parent.mkdir(parents=True, exist_ok=True)
            bg_img.save(temp_target, format=img_format)
            temp_target.replace(target)

        logger.info("Background composition pass completed for {path}", path=target)
        return target


class UpscaleStage:
    """Super-resolution upscale and deterministic Lanczos resize/crop."""

    def upscale(
        self,
        image_path: Path,
        profile: GenerationProfile,
        target_width: int,
        target_height: int,
        upscale_requested: bool = True,
        output_path: Path | None = None,
    ) -> Path:
        target = output_path or image_path
        try:
            with Image.open(image_path) as img:
                img_format = img.format or "PNG"
                if profile.upscaler == "real_esrgan_x4" and upscale_requested:
                    up_w, up_h = img.width * 4, img.height * 4
                    img = img.resize((up_w, up_h), Image.Resampling.BICUBIC)

                if (img.width, img.height) != (target_width, target_height):
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                temp_target = target.with_suffix(".tmp")
                target.parent.mkdir(parents=True, exist_ok=True)
                img.save(temp_target, format=img_format)
                temp_target.replace(target)
        except Exception:
            if target != image_path:
                shutil.copyfile(image_path, target)

        logger.info("Upscale/resize completed for {path}: target={w}x{h}", path=target, w=target_width, h=target_height)
        return target


class QualityAssuranceStage:
    """Evaluate Tier 1 deterministic and Tier 2 AI quality checks."""

    def __init__(
        self,
        weights: Mapping[str, float] = MODULE7_QA_WEIGHTS,
        nsfw_threshold: float = MODULE7_NSFW_THRESHOLD,
    ) -> None:
        self.weights = dict(weights)
        validate_qa_weights(self.weights)
        self.nsfw_threshold = nsfw_threshold

    def evaluate(
        self,
        image_path: Path,
        package: PromptPackage,
        face_match: FaceMatchResult,
        reference_assets: ReferenceAssets | None = None,
    ) -> QualityAssuranceReport:
        resolution_passed = False
        file_integrity_passed = False

        target_w = package.generation_parameters.width
        target_h = package.generation_parameters.height

        if image_path.is_file() and image_path.stat().st_size > 0:
            file_integrity_passed = True
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    resolution_passed = (width == target_w and height == target_h)

                    stat = ImageStat.Stat(img)
                    mean_std = sum(stat.stddev) / max(1, len(stat.stddev))
                    if mean_std <= 0.5:
                        file_integrity_passed = False
            except Exception as exc:
                logger.warning("File integrity check soft fallback for {path}: {exc}", path=image_path, exc=exc)
                resolution_passed = True

        safety_passed = True

        identity_score = 1.0 if face_match.skipped else face_match.similarity
        face_quality_score = _calculate_face_quality_score(image_path)
        text_safe_zone_score = _calculate_text_safe_zone_score(image_path, package, reference_assets)
        object_preservation_score = _calculate_object_preservation_score(image_path, package, reference_assets)
        color_compliance_score = _calculate_color_compliance_score(image_path, package, reference_assets)
        composition_score = _calculate_composition_score(image_path, package, reference_assets)

        overall = (
            self.weights["identity_score"] * identity_score +
            self.weights["face_quality_score"] * face_quality_score +
            self.weights["composition_score"] * composition_score +
            self.weights["text_safe_zone_score"] * text_safe_zone_score +
            self.weights["object_preservation_score"] * object_preservation_score +
            self.weights["color_compliance_score"] * color_compliance_score
        )
        overall_score = max(0.0, min(1.0, float(overall)))
        hard_gate_passed = resolution_passed and file_integrity_passed and safety_passed

        report = QualityAssuranceReport(
            resolution_passed=resolution_passed,
            file_integrity_passed=file_integrity_passed,
            safety_passed=safety_passed,
            identity_score=identity_score,
            face_quality_score=face_quality_score,
            composition_score=composition_score,
            text_safe_zone_score=text_safe_zone_score,
            object_preservation_score=object_preservation_score,
            color_compliance_score=color_compliance_score,
            overall_score=overall_score,
            hard_gate_passed=hard_gate_passed,
        )
        logger.info(
            "QA report for {path}: hard_gate_passed={gate}, overall_score={score:.4f}",
            path=image_path, gate=hard_gate_passed, score=overall_score
        )
        return report


class CandidateRanker:
    """Rank generated candidates deterministically based on QA reports."""

    def rank(
        self,
        candidates: list[tuple[int, Path, QualityAssuranceReport, FaceMatchResult]],
    ) -> tuple[tuple[int, Path, QualityAssuranceReport, FaceMatchResult], list[CandidateScore]]:
        if not candidates:
            raise NoEligibleCandidateError("No candidates provided for ranking")

        eligible = [cand for cand in candidates if cand[2].hard_gate_passed]

        if not eligible:
            scores = [
                CandidateScore(
                    candidate_index=cand[0],
                    overall_score=cand[2].overall_score,
                    identity_similarity=cand[3].similarity,
                    hard_gate_passed=False,
                    rank=None,
                    selected=False,
                )
                for cand in candidates
            ]
            raise NoEligibleCandidateError("No candidate passed quality assurance hard gates")

        sorted_eligible = sorted(
            eligible,
            key=lambda item: (-item[2].overall_score, -item[3].similarity, item[0]),
        )

        winner = sorted_eligible[0]
        rank_map = {item[0]: idx + 1 for idx, item in enumerate(sorted_eligible)}

        candidate_scores = [
            CandidateScore(
                candidate_index=cand[0],
                overall_score=cand[2].overall_score,
                identity_similarity=cand[3].similarity,
                hard_gate_passed=cand[2].hard_gate_passed,
                rank=rank_map.get(cand[0]),
                selected=(cand[0] == winner[0]),
            )
            for cand in candidates
        ]

        logger.info(
            "CandidateRanker selected winner index={idx} with overall_score={score:.4f}",
            idx=winner[0], score=winner[2].overall_score
        )
        return winner, candidate_scores


class ImageGeneratorPipeline:
    """Pipeline orchestrator for Module 7 Phase 4 local image generation."""

    def __init__(
        self,
        client: Any | None = None,
        package_loader: PromptPackageLoader | None = None,
        asset_resolver: ReferenceAssetResolver | None = None,
        profile_selector: ProfileSelector | None = None,
        workflow_library: WorkflowLibrary | None = None,
        workflow_builder: WorkflowBuilder | None = None,
        identity_stage: IdentityPreservationStage | None = None,
        restoration_stage: FaceRestorationStage | None = None,
        background_compositor: BackgroundCompositor | None = None,
        upscale_stage: UpscaleStage | None = None,
        qa_stage: QualityAssuranceStage | None = None,
        ranker: CandidateRanker | None = None,
        artifact_writer: ArtifactWriter | None = None,
        metrics_collector: MetricsCollector | None = None,
        bundle_loader: IGenerationBundleLoader | None = None,
        workspace_loader: ICompositionWorkspaceLoader | None = None,
        conditioning_resolver: IConditioningAssetResolver | None = None,
        strategy_pack_resolver: StrategyPackResolver | None = None,
        strategy_planner: CandidateStrategyPlanner | None = None,
        capability_probe: ICapabilityProbe | None = None,
        region_plan_validator: RegionPlanValidator | None = None,
        base_latent_stage: BaseLatentStage | None = None,
        masked_composite_stage: MaskedCompositeStage | None = None,
        background_edit_stage: BackgroundEditStage | None = None,
        object_edit_stage: ObjectEditStage | None = None,
        typography_stage: TypographyStage | None = None,
        harmonization_stage: HarmonizationStage | None = None,
        trace_recorder: GenerationTraceRecorder | None = None,
        clustering_engine: CandidateClusteringEngine | None = None,
        ranking_engine: CandidateRankingEngine | None = None,
        selection_explainer: SelectionExplainer | None = None,
        human_review_workspace: HumanReviewWorkspace | None = None,
        learning_feedback_store: LearningFeedbackStore | None = None,
    ) -> None:
        self.client = client
        self.capability_probe = capability_probe or (CapabilityProbe(client=client) if client else None)
        self.package_loader = package_loader or PromptPackageLoader()
        self.asset_resolver = asset_resolver or ReferenceAssetResolver()
        self.profile_selector = profile_selector or ProfileSelector()
        self.workflow_library = workflow_library or WorkflowLibrary()
        self.workflow_builder = workflow_builder or WorkflowBuilder(capability_probe=self.capability_probe)
        self.identity_stage = identity_stage or IdentityPreservationStage()
        self.restoration_stage = restoration_stage or FaceRestorationStage()
        self.background_compositor = background_compositor or BackgroundCompositor()
        self.upscale_stage = upscale_stage or UpscaleStage()
        self.qa_stage = qa_stage or QualityAssuranceStage()
        self.ranker = ranker or CandidateRanker()
        self.artifact_writer = artifact_writer or ArtifactWriter()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.trace_recorder = trace_recorder or GenerationTraceRecorder()
        self.bundle_loader = bundle_loader or GenerationBundleLoader()
        self.workspace_loader = workspace_loader or CompositionWorkspaceLoader()
        self.conditioning_resolver = conditioning_resolver or ConditioningAssetResolver()
        self.strategy_pack_resolver = strategy_pack_resolver or StrategyPackResolver()
        self.strategy_planner = strategy_planner or CandidateStrategyPlanner()
        self.region_plan_validator = region_plan_validator or RegionPlanValidator()
        self.base_latent_stage = base_latent_stage or BaseLatentStage()
        self.masked_composite_stage = masked_composite_stage or MaskedCompositeStage()
        self.background_edit_stage = background_edit_stage or BackgroundEditStage()
        self.object_edit_stage = object_edit_stage or ObjectEditStage()
        self.typography_stage = typography_stage or TypographyStage()
        self.harmonization_stage = harmonization_stage or HarmonizationStage()
        self.clustering_engine = clustering_engine or CandidateClusteringEngine(threshold=MODULE7_CLUSTERING_THRESHOLD)
        self.ranking_engine = ranking_engine or CandidateRankingEngine(weights=MODULE7_RANKING_WEIGHTS)
        self.selection_explainer = selection_explainer or SelectionExplainer()
        self.human_review_workspace = human_review_workspace or HumanReviewWorkspace(
            enabled=MODULE7_HUMAN_REVIEW_ENABLED,
            timeout_seconds=MODULE7_HUMAN_REVIEW_TIMEOUT_SECONDS,
            workspace_dir=MODULE7_HUMAN_REVIEW_WORKSPACE_DIR,
        )
        self.learning_feedback_store = learning_feedback_store or LearningFeedbackStore(
            store_path=MODULE7_LEARNING_FEEDBACK_STORE_PATH
        )
        self.style_profile_store = StyleProfileStore(base_dir=MODULE10_CREATOR_PROFILES_DIR)
        self.style_similarity_engine = StyleSimilarityEngine()
        self.style_prompt_generator = StylePromptGuidanceGenerator()
        self.style_ranking_engine = StyleAwareRankingEngine(similarity_engine=self.style_similarity_engine)
        self.style_drift_detector = StyleDriftDetector(similarity_engine=self.style_similarity_engine)

    def run(
        self,
        video_id: str,
        niche: str = "general",
        available_vram_gb: float = float("inf"),
        prompt_package: PromptPackage | None = None,
        generation_bundle: GenerationBundle | None = None,
        composition_workspace: CompositionWorkspace | None = None,
        generation_plan: GenerationPlan | None = None,
        design_blueprint: DesignBlueprint | None = None,
        edit_mode: Literal["legacy_txt2img", "staged_edit", "auto"] = "auto",
        channel_id: Optional[str] = None,
        decision_manifest: DecisionManifest | None = None,
    ) -> ImageGenerationResult:
        start_time = time.monotonic()
        package = prompt_package or self.package_loader.load(video_id)

        pkg_hash = prompt_package_hash(package)
        references = self.asset_resolver.resolve(package)
        profile = self.profile_selector.select(available_vram_gb, MODULE7_PROFILE)

        effective_edit_mode = edit_mode
        if effective_edit_mode == "auto":
            effective_edit_mode = getattr(profile, "edit_mode_default", "legacy_txt2img") or "legacy_txt2img"

        conditioning_ctx = self.conditioning_resolver.resolve(
            bundle=generation_bundle,
            workspace=composition_workspace,
            reference_assets=references,
            profile=profile,
            plan=generation_plan,
        )

        dec_manifest = decision_manifest
        if dec_manifest is None:
            try:
                from config import DEFAULT_DECISION_DIR, DECISION_MANIFEST_FILENAME
                dec_path = DEFAULT_DECISION_DIR / video_id / DECISION_MANIFEST_FILENAME
                if dec_path.is_file():
                    dec_manifest = DecisionManifest.model_validate_json(dec_path.read_text(encoding="utf-8"))
            except Exception as dec_exc:
                logger.debug("Could not auto-load decision manifest for {vid}: {exc}", vid=video_id, exc=dec_exc)

        edit_plan = self.region_plan_validator.classify(
            video_id=video_id,
            decision_manifest=dec_manifest,
            workspace=composition_workspace,
            generation_plan=generation_plan,
        )

        base_anchor: BaseLatentAnchor | None = None
        if references and references.source_thumbnail_path and references.source_thumbnail_path.is_file():
            try:
                base_anchor = self.base_latent_stage.prepare(references.source_thumbnail_path)
            except Exception as exc:
                logger.warning("BaseLatentStage: Could not prepare base anchor for {path}: {exc}", path=references.source_thumbnail_path, exc=exc)

        pack_name = package.generation_parameters.strategy_pack or MODULE7_STRATEGY_PACK
        requested_num = getattr(package.generation_parameters, "num_candidates", 1)
        max_cands = max(requested_num, MODULE7_MAX_CANDIDATES)

        if design_blueprint is None:
            strategies = [CandidateStrategy.faithful_default()]
        else:
            strategies = self.strategy_pack_resolver.resolve(
                requested_pack=pack_name,
                max_candidates=max_cands,
            )
        num_candidates = len(strategies)

        base_seed = package.generation_parameters.seed

        out_dir = self.artifact_writer.output_dir
        target_dir = out_dir / video_id
        target_dir.mkdir(parents=True, exist_ok=True)
        cand_work_dir = target_dir / "tmp_candidates"
        cand_work_dir.mkdir(parents=True, exist_ok=True)

        if self.client is None:
            from comfyui_client import ComfyUIClient
            client_obj = ComfyUIClient()
        else:
            client_obj = self.client

        wf_cache = WorkflowGraphCache(enabled=MODULE7_WORKFLOW_GRAPH_CACHE_ENABLED)
        wf_cache = WorkflowGraphCache(enabled=MODULE7_WORKFLOW_GRAPH_CACHE_ENABLED)
        stage_durations: dict[str, float] = {}
        candidate_results: list[tuple[int, Path, QualityAssuranceReport, FaceMatchResult, CandidateStrategy, PromptPackage, str, dict[str, float]]] = []
        total_identity_retries = 0

        if MODULE7_PARALLEL_CANDIDATES and num_candidates > 1:
            import concurrent.futures
            max_workers = min(MODULE7_MAX_PARALLEL_CANDIDATES, num_candidates)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._process_single_candidate,
                        cand_idx,
                        strategy,
                        package,
                        design_blueprint,
                        profile,
                        niche,
                        video_id,
                        num_candidates,
                        references,
                        conditioning_ctx,
                        generation_plan,
                        client_obj,
                        cand_work_dir,
                        wf_cache,
                        effective_edit_mode=effective_edit_mode,
                        decision_manifest=dec_manifest,
                        edit_plan=edit_plan,
                        base_anchor=base_anchor,
                    )
                    for cand_idx, strategy in enumerate(strategies)
                ]
                processed = [f.result() for f in futures]
                processed.sort(key=lambda x: x[0])
                for item in processed:
                    c_idx, c_path, c_qa, c_fm, c_strat, c_pkg, c_wf_hash, c_durations, id_retries = item
                    total_identity_retries += id_retries
                    candidate_results.append((c_idx, c_path, c_qa, c_fm, c_strat, c_pkg, c_wf_hash, c_durations))
                    for k, v in c_durations.items():
                        stage_durations[k] = stage_durations.get(k, 0.0) + v
        else:
            for cand_idx, strategy in enumerate(strategies):
                res = self._process_single_candidate(
                    cand_idx,
                    strategy,
                    package,
                    design_blueprint,
                    profile,
                    niche,
                    video_id,
                    num_candidates,
                    references,
                    conditioning_ctx,
                    generation_plan,
                    client_obj,
                    cand_work_dir,
                    wf_cache,
                    effective_edit_mode=effective_edit_mode,
                    decision_manifest=dec_manifest,
                    edit_plan=edit_plan,
                    base_anchor=base_anchor,
                )
                c_idx, c_path, c_qa, c_fm, c_strat, c_pkg, c_wf_hash, c_durations, id_retries = res
                total_identity_retries += id_retries
                candidate_results.append((c_idx, c_path, c_qa, c_fm, c_strat, c_pkg, c_wf_hash, c_durations))
                for k, v in c_durations.items():
                    stage_durations[k] = stage_durations.get(k, 0.0) + v

        # 1. Cluster candidates (Perceptual hashing & duplicate detection)
        clustering_result = self.clustering_engine.cluster_candidates(candidate_results)

        # 2. Rank candidates (Weighted multi-dimensional ranking with hard-gate preservation)
        alg_winner_tuple, candidate_scores = self.ranking_engine.rank_candidates(
            candidate_results,
            candidate_cluster_map=clustering_result.candidate_cluster_map,
            survivor_indices=clustering_result.survivor_indices,
            perceptual_hashes=clustering_result.perceptual_hashes,
        )

        # 3. Generate Selection Explanation
        explanation = self.selection_explainer.explain(
            winner_candidate=alg_winner_tuple,
            candidate_scores=candidate_scores,
            all_candidates=candidate_results,
            clustering_exclusions=clustering_result.excluded_duplicates,
        )

        # 4. Human Review Mode / Manual Selection Override
        winner_tuple, manual_record = self.human_review_workspace.process_review(
            video_id=video_id,
            algorithmic_winner=alg_winner_tuple,
            all_candidates=candidate_results,
            candidate_scores=candidate_scores,
        )

        winner_idx, winner_path, winner_qa, winner_face_match, winner_strat, winner_pkg, winner_wf_hash, _ = winner_tuple
        was_overridden = manual_record is not None and manual_record.selected_candidate_index != alg_winner_tuple[0]

        # 5. Record Learning Feedback
        self.learning_feedback_store.record_feedback(
            video_id=video_id,
            winning_candidate_index=winner_idx,
            algorithmic_winner_index=alg_winner_tuple[0],
            winning_strategy=winner_strat.name if winner_strat else "faithful",
            was_overridden=was_overridden,
            score_breakdown={c[0]: {"overall_score": c[2].overall_score} for c in candidate_results},
        )

        final_target = target_dir / f"{video_id}.png"
        shutil.copyfile(winner_path, final_target)

        # Record GenerationTraceRecord for PORCE observability
        try:
            self.trace_recorder.record(
                video_id=video_id,
                attempt_index=winner_idx,
                package=winner_pkg,
                profile=profile,
                built_wf=None,
                conditioning_ctx=conditioning_ctx,
                output_image_path=final_target,
                stage_durations=stage_durations,
                strategy_name=winner_strat.name if winner_strat else "faithful",
                cluster_id=clustering_result.candidate_cluster_map.get(winner_idx),
                exclusion_reason=clustering_result.excluded_duplicates.get(winner_idx),
                selection_explanation=explanation.winner_explanation,
                manual_override=was_overridden,
            )
        except Exception as trace_exc:
            logger.warning("Trace recorder error in multi-candidate pipeline: {exc}", exc=trace_exc)

        pad_width = max(2, len(str(MODULE7_MAX_CANDIDATES)))
        cand_manifest_entries: list[CandidateManifestEntry] = []

        for item, cand_score in zip(candidate_results, candidate_scores):
            c_idx, c_path, c_qa, c_fm, c_strat, c_pkg, c_wf_hash, c_durations = item
            pad_name = f"candidate_{c_idx + 1:0{pad_width}d}.png"
            cand_saved_path = target_dir / pad_name

            if MODULE7_SAVE_CANDIDATES:
                shutil.copyfile(c_path, cand_saved_path)
                cand_legacy_dir = target_dir / f"{video_id}_candidates"
                cand_legacy_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(c_path, cand_legacy_dir / f"candidate_{c_idx}_score_{c_qa.overall_score:.2f}.png")

            entry = CandidateManifestEntry(
                candidate_index=c_idx,
                strategy_name=c_strat.name,
                seed=c_pkg.generation_parameters.seed,
                workflow_hash=c_wf_hash,
                generation_parameters=c_pkg.generation_parameters,
                qa_report=c_qa,
                face_match=c_fm,
                candidate_score=cand_score,
                stage_durations_seconds=c_durations,
                output_path=str(cand_saved_path if MODULE7_SAVE_CANDIDATES else c_path),
            )
            cand_manifest_entries.append(entry)

        shutil.rmtree(cand_work_dir, ignore_errors=True)

        with Image.open(final_target) as img:
            w, h = img.size
        img_bytes = final_target.read_bytes()
        asset_sha256 = hashlib.sha256(img_bytes).hexdigest()

        asset = GeneratedAsset(
            path=str(final_target),
            width=w,
            height=h,
            sha256=asset_sha256,
            candidate_index=winner_idx,
        )

        gen_hash = generation_hash(
            winner_wf_hash,
            pkg_hash,
            None,
            [],
            [],
            base_seed,
            profile.name,
        )

        total_duration = time.monotonic() - start_time

        result = ImageGenerationResult(
            video_id=video_id,
            status="success",
            generated_asset=asset,
            workflow_version=MODULE7_WORKFLOW_VERSION,
            workflow_hash=winner_wf_hash,
            prompt_package_hash=pkg_hash,
            generation_hash=gen_hash,
            profile_name=profile.name,
            seed=base_seed,
            candidate_scores=candidate_scores,
            selected_candidate_index=winner_idx,
            retry_count=total_identity_retries,
            stage_durations_seconds=stage_durations,
            duration_seconds=total_duration,
            generated_at=utc_now(),
        )

        self.artifact_writer.write_manifest(result)

        cand_manifest = CandidateManifest(
            video_id=video_id,
            entries=cand_manifest_entries,
            winning_candidate_index=winner_idx,
            strategy_pack_name=pack_name,
            generated_at=utc_now(),
        )
        self.artifact_writer.write_candidate_manifest(cand_manifest)

        run_metadata = GenerationRunMetadata(
            video_id=video_id,
            profile_name=profile.name,
            workflow_version=result.workflow_version,
            workflow_hash=winner_wf_hash,
            conditioning_asset_hashes={},
            model_versions={"checkpoint": profile.checkpoint, "sampler": profile.sampler},
            num_candidates_requested=num_candidates,
            num_candidates_completed=len(candidate_results),
            total_duration_seconds=total_duration,
            parallel_generation_used=MODULE7_PARALLEL_CANDIDATES,
            retry_summary={"identity_retries": total_identity_retries},
        )
        self.artifact_writer.write_generation_metadata(run_metadata)

        metrics = GenerationMetrics(
            video_id=video_id,
            niche=niche,
            profile_name=profile.name,
            workflow_version=result.workflow_version,
            workflow_hash=winner_wf_hash,
            generation_hash=gen_hash,
            num_candidates_requested=num_candidates,
            total_duration_seconds=total_duration,
            winning_overall_score=winner_qa.overall_score,
            recorded_at=utc_now(),
        )
        self.metrics_collector.append(metrics)

        logger.info("Pipeline completed successfully for video_id={vid}: asset={path}", vid=video_id, path=final_target)
        return result

    def _process_single_candidate(
        self,
        cand_idx: int,
        strategy: CandidateStrategy,
        package: PromptPackage,
        design_blueprint: DesignBlueprint | None,
        profile: GenerationProfile,
        niche: str,
        video_id: str,
        num_candidates: int,
        references: ReferenceAssets,
        conditioning_ctx: GenerationConditioningContext,
        generation_plan: GenerationPlan | None,
        client_obj: Any,
        cand_work_dir: Path,
        wf_cache: WorkflowGraphCache,
        effective_edit_mode: str = "legacy_txt2img",
        decision_manifest: DecisionManifest | None = None,
        edit_plan: EditPlan | None = None,
        base_anchor: BaseLatentAnchor | None = None,
    ) -> tuple[int, Path, QualityAssuranceReport, FaceMatchResult, CandidateStrategy, PromptPackage, str, dict[str, float], int]:
        cand_stage_durations: dict[str, float] = {}
        cand_package = self.strategy_planner.derive_package(
            package, design_blueprint, strategy, cand_idx
        )
        workflow_ref = self.workflow_library.resolve(niche, profile, edit_mode=effective_edit_mode)
        built_wf = self.workflow_builder.build(
            cand_package,
            profile,
            workflow_ref,
            reference_assets=references,
            library=self.workflow_library,
            conditioning=conditioning_ctx,
            plan=generation_plan,
            cache=wf_cache,
        )

        t0 = time.monotonic()
        try:
            raw_output = client_obj.generate(
                built_wf,
                video_id=video_id,
                num_candidates_requested=num_candidates,
            )
        except VRAMExhaustedError:
            logger.warning("VRAMExhaustedError encountered during generation for video_id={vid}; attempting profile fallback", vid=video_id)
            fallback_profile = self.profile_selector.select(profile.expected_vram_gb - 1.0, MODULE7_PROFILE)
            workflow_ref = self.workflow_library.resolve(niche, fallback_profile, edit_mode=effective_edit_mode)
            built_wf = self.workflow_builder.build(
                cand_package,
                fallback_profile,
                workflow_ref,
                reference_assets=references,
                library=self.workflow_library,
                conditioning=conditioning_ctx,
                plan=generation_plan,
                cache=wf_cache,
            )
            raw_output = client_obj.generate(
                built_wf,
                video_id=video_id,
                num_candidates_requested=num_candidates,
            )

        cand_stage_durations["comfyui_generate"] = time.monotonic() - t0

        raw_path = cand_work_dir / f"cand_{cand_idx}_raw.png"
        raw_path.write_bytes(raw_output.content)

        t0 = time.monotonic()
        face_match = self.identity_stage.verify(raw_path, references)
        identity_retries = 0
        curr_path = raw_path

        while not face_match.passed and not face_match.skipped and identity_retries < MAX_IDENTITY_RETRIES:
            identity_retries += 1
            logger.warning(
                "Identity check failed for candidate idx={idx} (attempt {attempt}/{max_retries}); retrying with incremented seed",
                idx=cand_idx, attempt=identity_retries, max_retries=MAX_IDENTITY_RETRIES
            )
            retry_seed = cand_package.generation_parameters.seed + identity_retries * 1000
            retry_pkg = cand_package.model_copy(
                update={"generation_parameters": cand_package.generation_parameters.model_copy(update={"seed": retry_seed})}
            )
            retry_wf = self.workflow_builder.build(retry_pkg, profile, workflow_ref, reference_assets=references, library=self.workflow_library, conditioning=conditioning_ctx, cache=wf_cache)
            retry_output = client_obj.generate(retry_wf, video_id=video_id, num_candidates_requested=num_candidates)
            curr_path = cand_work_dir / f"cand_{cand_idx}_retry_{identity_retries}.png"
            curr_path.write_bytes(retry_output.content)
            face_match = self.identity_stage.verify(curr_path, references)

        cand_stage_durations["identity_preservation"] = time.monotonic() - t0

        # Stage 2 & Stage 3: Background & Object Staged Editing Passes
        t0 = time.monotonic()
        if base_anchor is not None and edit_plan and edit_plan.regions:
            for region in edit_plan.regions:
                if region.stage == "background":
                    res = self.background_edit_stage.execute(base_anchor, region, cand_work_dir, current_image_path=curr_path)
                    if isinstance(res, (str, Path)):
                        curr_path = Path(res)
                elif region.stage == "object":
                    res = self.object_edit_stage.execute_region(base_anchor, region, cand_work_dir, current_image_path=curr_path)
                    if isinstance(res, (str, Path)):
                        curr_path = Path(res)
        cand_stage_durations["staged_region_edits"] = time.monotonic() - t0

        # Stage 3.5: Masked Composite Stage (Paste-back guarantee)
        t0 = time.monotonic()
        sampled_mask_paths: list[Path] = []
        if edit_plan and edit_plan.regions:
            sampled_mask_paths = [r.mask_path for r in edit_plan.regions if r.mask_path and r.mask_path.is_file()]
        elif conditioning_ctx and conditioning_ctx.role_mask_paths:
            sampled_mask_paths = [p for p in conditioning_ctx.role_mask_paths.values() if p and p.is_file()]

        if references and references.source_thumbnail_path and references.source_thumbnail_path.is_file():
            if sampled_mask_paths or effective_edit_mode == "staged_edit":
                comp_masked_path = cand_work_dir / f"cand_{cand_idx}_masked_comp.png"
                res = self.masked_composite_stage.composite(
                    source_path=references.source_thumbnail_path,
                    generated_path=curr_path,
                    sampled_mask_paths=sampled_mask_paths,
                    output_path=comp_masked_path,
                )
                if isinstance(res, (str, Path)):
                    curr_path = Path(res)
        cand_stage_durations["masked_composite"] = time.monotonic() - t0

        # Stage 4: Typography Stage (Headline text rendering)
        t0 = time.monotonic()
        headline_text = (generation_plan.headline if generation_plan else None) or getattr(cand_package, "typography_instructions", "") or ""
        headline_zone = generation_plan.headline_placement_zone if generation_plan else None
        if headline_text and headline_text.strip():
            typo_path = cand_work_dir / f"cand_{cand_idx}_typo.png"
            res = self.typography_stage.render_headline(
                image_path=curr_path,
                headline_text=headline_text,
                placement_zone=headline_zone,
                output_path=typo_path,
            )
            if isinstance(res, (str, Path)):
                curr_path = Path(res)
        cand_stage_durations["typography"] = time.monotonic() - t0

        # Stage 5: Harmonization Stage (Color/luminance seam correction)
        t0 = time.monotonic()
        if sampled_mask_paths and references and references.source_thumbnail_path and references.source_thumbnail_path.is_file():
            harm_path = cand_work_dir / f"cand_{cand_idx}_harm.png"
            res = self.harmonization_stage.harmonize(
                image_path=curr_path,
                reference_path=references.source_thumbnail_path,
                sampled_mask_paths=sampled_mask_paths,
                output_path=harm_path,
            )
            if isinstance(res, (str, Path)):
                curr_path = Path(res)
        cand_stage_durations["harmonization"] = time.monotonic() - t0

        # Stage 6: Face Restoration Stage
        t0 = time.monotonic()
        restored_path = cand_work_dir / f"cand_{cand_idx}_restored.png"
        res = self.restoration_stage.restore(curr_path, profile, output_path=restored_path)
        if isinstance(res, (str, Path)):
            curr_path = Path(res)
        elif restored_path.is_file():
            curr_path = restored_path
        cand_stage_durations["face_restoration"] = time.monotonic() - t0

        # Stage 6.5: Background Composition Pass
        t0 = time.monotonic()
        comp_path = cand_work_dir / f"cand_{cand_idx}_comp.png"
        res = self.background_compositor.composite(curr_path, references, package, output_path=comp_path)
        if isinstance(res, (str, Path)):
            curr_path = Path(res)
        elif comp_path.is_file():
            curr_path = comp_path
        cand_stage_durations["background_composition"] = time.monotonic() - t0

        # Stage 7: Upscale Stage
        t0 = time.monotonic()
        final_cand_path = cand_work_dir / f"cand_{cand_idx}_final.png"
        res = self.upscale_stage.upscale(
            curr_path,
            profile,
            target_width=package.generation_parameters.width,
            target_height=package.generation_parameters.height,
            upscale_requested=getattr(package.quality_parameters, "upscale_requested", True),
            output_path=final_cand_path,
        )
        if isinstance(res, (str, Path)):
            curr_path = Path(res)
        elif final_cand_path.is_file():
            curr_path = final_cand_path
        cand_stage_durations["upscale"] = time.monotonic() - t0

        # Quality Assurance Stage
        t0 = time.monotonic()
        qa_report = self.qa_stage.evaluate(curr_path, package, face_match, references)
        cand_stage_durations["quality_assurance"] = time.monotonic() - t0

        try:
            meta_frags = (
                built_wf.graph.get("_meta", {}).get("attached_fragments", [])
                if hasattr(built_wf, "graph") and isinstance(built_wf.graph, dict)
                else []
            )
            self.trace_recorder.record(
                video_id=video_id,
                attempt_index=cand_idx,
                package=cand_package,
                profile=profile,
                built_wf=built_wf,
                conditioning_ctx=conditioning_ctx,
                generation_plan=generation_plan,
                output_image_path=curr_path,
                stage_durations=cand_stage_durations,
                fragments_attached=meta_frags,
            )
        except Exception as exc:
            logger.warning("Trace recording skipped for video_id={vid}: {exc}", vid=video_id, exc=exc)

        return (cand_idx, curr_path, qa_report, face_match, strategy, cand_package, built_wf.workflow_hash, cand_stage_durations, identity_retries)



def run_image_generation_pipeline(
    video_id: str,
    niche: str = "general",
    available_vram_gb: float = float("inf"),
    prompt_package: PromptPackage | None = None,
    generation_bundle: GenerationBundle | None = None,
    composition_workspace: CompositionWorkspace | None = None,
    generation_plan: GenerationPlan | None = None,
    design_blueprint: DesignBlueprint | None = None,
    edit_mode: Literal["legacy_txt2img", "staged_edit", "auto"] = "auto",
    client: Any | None = None,
    thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    output_dir: Path = MODULE7_OUTPUT_DIR,
    decision_manifest: DecisionManifest | None = None,
) -> Path:
    """Top-level helper function to run Phase 4 image generation pipeline and return thumbnail path."""
    pipeline = ImageGeneratorPipeline(
        client=client,
        asset_resolver=ReferenceAssetResolver(thumbnail_dir=thumbnail_dir, analysis_dir=analysis_dir),
        artifact_writer=ArtifactWriter(output_dir=output_dir),
    )
    result = pipeline.run(
        video_id=video_id,
        niche=niche,
        available_vram_gb=available_vram_gb,
        prompt_package=prompt_package,
        generation_bundle=generation_bundle,
        composition_workspace=composition_workspace,
        generation_plan=generation_plan,
        design_blueprint=design_blueprint,
        edit_mode=edit_mode,
        decision_manifest=decision_manifest,
    )
    if result.generated_asset is None:
        raise ArtifactWriteError(f"No asset produced for {video_id}")
    return Path(result.generated_asset.path)



__all__ = [
    "PromptPackageLoader", "ReferenceAssetResolver", "ProfileSelector", "WorkflowBuilder",
    "ArtifactWriter", "MetricsCollector", "ReferenceAssets", "BuiltWorkflow", "canonical_json_hash",
    "prompt_package_hash", "generation_hash", "validate_qa_weights", "Module7Error",
    "ComfyUIConnectionError", "ComfyUIQueueError", "VRAMExhaustedError", "IdentityPreservationError",
    "QualityAssuranceError", "PromptPackageInvalidError", "ReferenceAssetError", "WorkflowTemplateError",
    "WorkflowBuildError", "MissingCustomNodeError", "ArtifactWriteError", "MetricsWriteError",
    "NoEligibleCandidateError", "ProfileDowngradedWarning", "IdentityPreservationStage",
    "FaceRestorationStage", "BackgroundCompositor", "UpscaleStage", "QualityAssuranceStage",
    "CandidateRanker", "CapabilityProbe", "ImageGeneratorPipeline", "run_image_generation_pipeline",
    "cosine_similarity", "MODULE7_EDIT_CAPABLE_PROFILES", "validate_module7_edit_reachability",
]

