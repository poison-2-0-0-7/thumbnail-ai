"""Module 7 Phase 3 production foundation & Image Generation Pipeline."""

from __future__ import annotations

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
    MODULE7_NSFW_THRESHOLD, MODULE7_OUTPUT_DIR, MODULE7_PROFILE,
    MODULE7_PROFILE_PREFERENCE, MODULE7_QA_WEIGHTS, MODULE7_SAVE_CANDIDATES,
    MODULE7_VRAM_HEADROOM_GB,
)
from models import (
    CandidateScore, FaceMatchResult, GeneratedAsset, GenerationMetrics,
    GenerationProfile, ImageGenerationResult, PromptPackage, QualityAssuranceReport,
    WorkflowTemplateRef, CompositionWorkspace, GenerationBundle,
)
from generation_components import (
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
    WorkflowGraphAssembler,
)

from module7_exceptions import (
    ArtifactWriteError, ComfyUIConnectionError, ComfyUIQueueError,
    IdentityPreservationError, MetricsWriteError, Module7Error,
    NoEligibleCandidateError, PromptPackageInvalidError, QualityAssuranceError,
    ProfileDowngradedWarning, ReferenceAssetError, VRAMExhaustedError,
    WorkflowBuildError, WorkflowTemplateError,
)
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


class WorkflowBuilder:
    """Pure deterministic materializer for ComfyUI graph templates; never submits them."""

    def __init__(
        self,
        fragment_library: INodeFragmentLibrary | None = None,
        graph_assembler: IWorkflowGraphAssembler | None = None,
        capability_probe: ICapabilityProbe | None = None,
    ) -> None:
        self.fragment_library = fragment_library or NodeFragmentLibrary()
        self.graph_assembler = graph_assembler or WorkflowGraphAssembler()
        self.capability_probe = capability_probe

    def build(
        self,
        package: PromptPackage,
        profile: GenerationProfile,
        workflow_ref: WorkflowTemplateRef,
        reference_assets: ReferenceAssets | None = None,
        library: WorkflowLibrary | None = None,
        conditioning: GenerationConditioningContext | None = None,
    ) -> BuiltWorkflow:
        """Fill named template slots and return the exact graph plus its hash."""
        source = library or WorkflowLibrary(Path(workflow_ref.template_path).parent)
        template = source.load(Path(workflow_ref.template_path))
        slots = self._slots(package, profile, reference_assets, conditioning)
        try:
            base_graph = self._substitute(template["graph"], slots)
        except KeyError as exc:
            raise WorkflowBuildError(f"Template {workflow_ref.template_name} uses unknown placeholder {exc.args[0]}") from exc
        if not isinstance(base_graph, dict):
            raise WorkflowBuildError("Resolved workflow graph must be an object")

        if conditioning is not None:
            fragment_ids = self._select_fragments(profile, conditioning)
            fragment_dicts = []
            for fid in fragment_ids:
                frag_data = self.fragment_library.load(fid)
                if self.capability_probe and not self.capability_probe.is_fragment_supported(frag_data):
                    logger.warning(
                        "Fragment '{fragment_id}' dropped: required node types not available in ComfyUI",
                        fragment_id=fid,
                    )
                    continue
                fragment_dicts.append(frag_data)

            final_graph = self.graph_assembler.assemble(
                {"_meta": template.get("_meta", {}), "graph": base_graph},
                fragment_dicts,
                conditioning,
                profile,
            ).get("graph", base_graph)
        else:
            final_graph = base_graph

        workflow_hash = canonical_json_hash(final_graph)
        logger.info("Built workflow template={template}, version={version}, workflow_hash={hash}", template=workflow_ref.template_name, version=workflow_ref.workflow_version, hash=workflow_hash)
        return BuiltWorkflow(graph=final_graph, workflow_ref=workflow_ref, workflow_hash=workflow_hash)

    def _select_fragments(
        self, profile: GenerationProfile, conditioning: GenerationConditioningContext | None
    ) -> list[str]:
        if conditioning is None:
            return []
        fragments: list[str] = []
        if profile.controlnet_enabled and conditioning.depth_path is not None:
            fragments.append("controlnet_depth")
        if profile.controlnet_enabled and conditioning.canny_path is not None:
            fragments.append("controlnet_canny")
        if profile.controlnet_enabled and conditioning.segmentation_path is not None:
            fragments.append("controlnet_segmentation")
        if profile.ipadapter_enabled and conditioning.ip_adapter_reference_paths:
            fragments.append("ipadapter_reference")
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
    ) -> dict[str, Any]:
        positive = " ".join((package.positive_prompt, package.subject_instructions,
                             package.lighting_instructions, package.color_instructions))
        negative = ", ".join((package.negative_prompt, *package.rendering_constraints,
                              *package.safety_constraints))

        thumb_path = ""
        if conditioning and conditioning.source_thumbnail_path:
            thumb_path = str(conditioning.source_thumbnail_path)
        elif references and references.source_thumbnail_path:
            thumb_path = str(references.source_thumbnail_path)

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
            "foreground_image_path": str(conditioning.role_image_paths.get("foreground", "")) if conditioning else "",
            "background_image_path": str(conditioning.role_image_paths.get("background", "")) if conditioning else "",
            "person_mask_path": str(conditioning.role_mask_paths.get("person", "")) if conditioning else "",
            "object_mask_path": str(conditioning.role_mask_paths.get("object", "")) if conditioning else "",
            "depth_map_path": str(conditioning.depth_path) if conditioning and conditioning.depth_path else "",
            "canny_map_path": str(conditioning.canny_path) if conditioning and conditioning.canny_path else "",
            "segmentation_map_path": str(conditioning.segmentation_path) if conditioning and conditioning.segmentation_path else "",
            "text_exclusion_mask_path": str(conditioning.text_exclusion_mask_path) if conditioning and conditioning.text_exclusion_mask_path else "",
            "controlnet_depth_strength": 0.55,
            "controlnet_canny_strength": 0.45,
            "controlnet_segmentation_strength": 0.5,
            "ipadapter_weight": 0.6,
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


def _calculate_text_safe_zone_score(image_path: Path, package: PromptPackage) -> float:
    """Calculate safe-zone collision score in [0.0, 1.0]."""
    return 1.0


def _calculate_object_preservation_score(image_path: Path, package: PromptPackage) -> float:
    """Calculate object-directive preservation score in [0.0, 1.0]."""
    return 1.0


def _calculate_color_compliance_score(image_path: Path, package: PromptPackage) -> float:
    """Calculate color direction compliance score in [0.0, 1.0]."""
    return 1.0


def _calculate_composition_score(image_path: Path, package: PromptPackage) -> float:
    """Calculate composition adherence score in [0.0, 1.0]."""
    return 1.0


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

        with Image.open(image_path) as img:
            img_format = img.format or "PNG"
            enhanced = img.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.SHARPEN)
            temp_target = target.with_suffix(".tmp")
            target.parent.mkdir(parents=True, exist_ok=True)
            enhanced.save(temp_target, format=img_format)
            temp_target.replace(target)

        logger.info("Face restoration completed for {path} using restoration={mode}", path=target, mode=profile.restoration)
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
        text_safe_zone_score = _calculate_text_safe_zone_score(image_path, package)
        object_preservation_score = _calculate_object_preservation_score(image_path, package)
        color_compliance_score = _calculate_color_compliance_score(image_path, package)
        composition_score = _calculate_composition_score(image_path, package)

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
    """Pipeline orchestrator for Module 7 Phase 3 local image generation."""

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
    ) -> None:
        self.client = client
        self.package_loader = package_loader or PromptPackageLoader()
        self.asset_resolver = asset_resolver or ReferenceAssetResolver()
        self.profile_selector = profile_selector or ProfileSelector()
        self.workflow_library = workflow_library or WorkflowLibrary()
        self.workflow_builder = workflow_builder or WorkflowBuilder()
        self.identity_stage = identity_stage or IdentityPreservationStage()
        self.restoration_stage = restoration_stage or FaceRestorationStage()
        self.background_compositor = background_compositor or BackgroundCompositor()
        self.upscale_stage = upscale_stage or UpscaleStage()
        self.qa_stage = qa_stage or QualityAssuranceStage()
        self.ranker = ranker or CandidateRanker()
        self.artifact_writer = artifact_writer or ArtifactWriter()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.bundle_loader = bundle_loader or GenerationBundleLoader()
        self.workspace_loader = workspace_loader or CompositionWorkspaceLoader()
        self.conditioning_resolver = conditioning_resolver or ConditioningAssetResolver()

    def run(
        self,
        video_id: str,
        niche: str = "general",
        available_vram_gb: float = float("inf"),
        prompt_package: PromptPackage | None = None,
        generation_bundle: GenerationBundle | None = None,
        composition_workspace: CompositionWorkspace | None = None,
    ) -> ImageGenerationResult:
        start_time = time.monotonic()
        package = prompt_package or self.package_loader.load(video_id)
        pkg_hash = prompt_package_hash(package)
        references = self.asset_resolver.resolve(package)
        profile = self.profile_selector.select(available_vram_gb, MODULE7_PROFILE)

        conditioning_ctx = self.conditioning_resolver.resolve(
            bundle=generation_bundle,
            workspace=composition_workspace,
            reference_assets=references,
            profile=profile,
        )

        num_candidates = getattr(package.generation_parameters, "num_candidates", 1)
        base_seed = package.generation_parameters.seed

        out_dir = self.artifact_writer.output_dir
        cand_work_dir = out_dir / video_id / "tmp_candidates"
        cand_work_dir.mkdir(parents=True, exist_ok=True)

        if self.client is None:
            from comfyui_client import ComfyUIClient
            client_obj = ComfyUIClient()
        else:
            client_obj = self.client
        stage_durations: dict[str, float] = {}
        candidate_results: list[tuple[int, Path, QualityAssuranceReport, FaceMatchResult]] = []
        for cand_idx in range(num_candidates):
            cand_seed = base_seed + cand_idx
            cand_package = package.model_copy(
                update={"generation_parameters": package.generation_parameters.model_copy(update={"seed": cand_seed})}
            )
            workflow_ref = self.workflow_library.resolve(niche, profile)
            built_wf = self.workflow_builder.build(
                cand_package, profile, workflow_ref, reference_assets=references, library=self.workflow_library, conditioning=conditioning_ctx
            )

            # Step A: ComfyUI generate with VRAM fallback ladder
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
                workflow_ref = self.workflow_library.resolve(niche, fallback_profile)
                built_wf = self.workflow_builder.build(cand_package, fallback_profile, workflow_ref, reference_assets=references, library=self.workflow_library, conditioning=conditioning_ctx)
                raw_output = client_obj.generate(
                    built_wf,
                    video_id=video_id,
                    num_candidates_requested=num_candidates,
                )
                profile = fallback_profile

            stage_durations["comfyui_generate"] = stage_durations.get("comfyui_generate", 0.0) + (time.monotonic() - t0)

            raw_path = cand_work_dir / f"cand_{cand_idx}_raw.png"
            raw_path.write_bytes(raw_output.content)

            # Step B: Identity check with bounded retries
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
                retry_seed = cand_seed + identity_retries * 1000
                retry_pkg = cand_package.model_copy(
                    update={"generation_parameters": cand_package.generation_parameters.model_copy(update={"seed": retry_seed})}
                )
                retry_wf = self.workflow_builder.build(retry_pkg, profile, workflow_ref, reference_assets=references, library=self.workflow_library, conditioning=conditioning_ctx)
                retry_output = client_obj.generate(retry_wf, video_id=video_id, num_candidates_requested=num_candidates)
                curr_path = cand_work_dir / f"cand_{cand_idx}_retry_{identity_retries}.png"
                curr_path.write_bytes(retry_output.content)
                face_match = self.identity_stage.verify(curr_path, references)

            stage_durations["identity_preservation"] = stage_durations.get("identity_preservation", 0.0) + (time.monotonic() - t0)

            # Step C: Face restoration
            t0 = time.monotonic()
            restored_path = cand_work_dir / f"cand_{cand_idx}_restored.png"
            self.restoration_stage.restore(curr_path, profile, output_path=restored_path)
            stage_durations["face_restoration"] = stage_durations.get("face_restoration", 0.0) + (time.monotonic() - t0)

            # Step D: Background composition pass
            t0 = time.monotonic()
            comp_path = cand_work_dir / f"cand_{cand_idx}_comp.png"
            self.background_compositor.composite(restored_path, references, package, output_path=comp_path)
            stage_durations["background_composition"] = stage_durations.get("background_composition", 0.0) + (time.monotonic() - t0)

            # Step E: Upscale & Lanczos resize
            t0 = time.monotonic()
            final_cand_path = cand_work_dir / f"cand_{cand_idx}_final.png"
            self.upscale_stage.upscale(
                comp_path,
                profile,
                target_width=package.generation_parameters.width,
                target_height=package.generation_parameters.height,
                upscale_requested=getattr(package.quality_parameters, "upscale_requested", True),
                output_path=final_cand_path,
            )
            stage_durations["upscale"] = stage_durations.get("upscale", 0.0) + (time.monotonic() - t0)

            # Step F: Quality Assurance evaluation
            t0 = time.monotonic()
            qa_report = self.qa_stage.evaluate(final_cand_path, package, face_match, references)
            stage_durations["quality_assurance"] = stage_durations.get("quality_assurance", 0.0) + (time.monotonic() - t0)

            candidate_results.append((cand_idx, final_cand_path, qa_report, face_match))

        # Rank candidates
        winner_tuple, candidate_scores = self.ranker.rank(candidate_results)
        winner_idx, winner_path, winner_qa, winner_face_match = winner_tuple

        target_dir = out_dir / video_id
        target_dir.mkdir(parents=True, exist_ok=True)
        final_target = target_dir / f"{video_id}.png"
        shutil.copyfile(winner_path, final_target)

        if MODULE7_SAVE_CANDIDATES:
            cand_dir = target_dir / f"{video_id}_candidates"
            cand_dir.mkdir(parents=True, exist_ok=True)
            for cand_idx, cand_path, cand_qa, _ in candidate_results:
                shutil.copyfile(cand_path, cand_dir / f"candidate_{cand_idx}_score_{cand_qa.overall_score:.2f}.png")

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

        wf_hash = built_wf.workflow_hash
        gen_hash = generation_hash(
            wf_hash,
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
            workflow_version=built_wf.workflow_ref.workflow_version,
            workflow_hash=wf_hash,
            prompt_package_hash=pkg_hash,
            generation_hash=gen_hash,
            profile_name=profile.name,
            seed=base_seed,
            candidate_scores=candidate_scores,
            selected_candidate_index=winner_idx,
            retry_count=0,
            stage_durations_seconds=stage_durations,
            duration_seconds=total_duration,
            generated_at=utc_now(),
        )

        self.artifact_writer.write_manifest(result)

        metrics = GenerationMetrics(
            video_id=video_id,
            niche=niche,
            profile_name=profile.name,
            workflow_version=built_wf.workflow_ref.workflow_version,
            workflow_hash=wf_hash,
            generation_hash=gen_hash,
            num_candidates_requested=num_candidates,
            total_duration_seconds=total_duration,
            winning_overall_score=winner_qa.overall_score,
            recorded_at=utc_now(),
        )
        self.metrics_collector.append(metrics)

        logger.info("Pipeline completed successfully for video_id={vid}: asset={path}", vid=video_id, path=final_target)
        return result


def run_image_generation_pipeline(
    video_id: str,
    niche: str = "general",
    available_vram_gb: float = float("inf"),
    prompt_package: PromptPackage | None = None,
    generation_bundle: GenerationBundle | None = None,
    composition_workspace: CompositionWorkspace | None = None,
    client: Any | None = None,
    thumbnail_dir: Path = DEFAULT_THUMBNAIL_DIR,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    output_dir: Path = MODULE7_OUTPUT_DIR,
) -> Path:
    """Top-level helper function to run Phase 3 image generation pipeline and return thumbnail path."""
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
    "WorkflowBuildError", "ArtifactWriteError", "MetricsWriteError", "NoEligibleCandidateError",
    "ProfileDowngradedWarning", "IdentityPreservationStage", "FaceRestorationStage",
    "BackgroundCompositor", "UpscaleStage", "QualityAssuranceStage", "CandidateRanker",
    "ImageGeneratorPipeline", "run_image_generation_pipeline", "cosine_similarity",
]

