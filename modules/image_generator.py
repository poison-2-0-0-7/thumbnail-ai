"""Module 7 Phase 1 production foundation; it intentionally performs no generation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_MODULES_DIR = Path(__file__).resolve().parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from loguru import logger

from config import (
    DEFAULT_ANALYSIS_DIR, DEFAULT_PROMPT_PACKAGE_DIR, DEFAULT_THUMBNAIL_DIR,
    LOG_DIR, MODULE7_GENERATION_PROFILES, MODULE7_LOG_PATH, MODULE7_METRICS_PATH,
    MODULE7_OUTPUT_DIR, MODULE7_PROFILE, MODULE7_PROFILE_PREFERENCE,
    MODULE7_QA_WEIGHTS, MODULE7_VRAM_HEADROOM_GB,
)
from models import (
    GenerationMetrics, GenerationProfile, ImageGenerationResult, PromptPackage,
    WorkflowTemplateRef,
)
from module7_exceptions import (
    ArtifactWriteError, ComfyUIConnectionError, ComfyUIQueueError,
    IdentityPreservationError, MetricsWriteError, Module7Error,
    NoEligibleCandidateError, PromptPackageInvalidError, QualityAssuranceError,
    ProfileDowngradedWarning, ReferenceAssetError, VRAMExhaustedError, WorkflowBuildError, WorkflowTemplateError,
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

    def build(self, package: PromptPackage, profile: GenerationProfile,
              workflow_ref: WorkflowTemplateRef, reference_assets: ReferenceAssets | None = None,
              library: WorkflowLibrary | None = None) -> BuiltWorkflow:
        """Fill named template slots and return the exact graph plus its hash."""
        source = library or WorkflowLibrary(Path(workflow_ref.template_path).parent)
        template = source.load(Path(workflow_ref.template_path))
        slots = self._slots(package, profile, reference_assets)
        try:
            graph = self._substitute(template["graph"], slots)
        except KeyError as exc:
            raise WorkflowBuildError(f"Template {workflow_ref.template_name} uses unknown placeholder {exc.args[0]}") from exc
        if not isinstance(graph, dict):  # defensive; validation established this already
            raise WorkflowBuildError("Resolved workflow graph must be an object")
        workflow_hash = canonical_json_hash(graph)
        logger.info("Built workflow template={template}, version={version}, workflow_hash={hash}", template=workflow_ref.template_name, version=workflow_ref.workflow_version, hash=workflow_hash)
        return BuiltWorkflow(graph=graph, workflow_ref=workflow_ref, workflow_hash=workflow_hash)

    @staticmethod
    def _slots(package: PromptPackage, profile: GenerationProfile,
               references: ReferenceAssets | None) -> dict[str, Any]:
        positive = " ".join((package.positive_prompt, package.subject_instructions,
                             package.lighting_instructions, package.color_instructions))
        negative = ", ".join((package.negative_prompt, *package.rendering_constraints,
                              *package.safety_constraints))
        return {
            "checkpoint": profile.checkpoint, "positive_prompt": positive,
            "negative_prompt": negative, "background_prompt": package.background_instructions,
            "seed": package.generation_parameters.seed, "steps": profile.steps,
            "cfg": profile.cfg, "sampler": profile.sampler, "scheduler": profile.scheduler,
            "width": package.generation_parameters.width, "height": package.generation_parameters.height,
            "controlnet_enabled": profile.controlnet_enabled, "ipadapter_enabled": profile.ipadapter_enabled,
            "restoration": profile.restoration, "restoration_fidelity": profile.restoration_fidelity,
            "upscaler": profile.upscaler,
            "source_thumbnail_path": str(references.source_thumbnail_path) if references else "",
        }

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


__all__ = [
    "PromptPackageLoader", "ReferenceAssetResolver", "ProfileSelector", "WorkflowBuilder",
    "ArtifactWriter", "MetricsCollector", "ReferenceAssets", "BuiltWorkflow", "canonical_json_hash",
    "prompt_package_hash", "generation_hash", "validate_qa_weights", "Module7Error",
    "ComfyUIConnectionError", "ComfyUIQueueError", "VRAMExhaustedError", "IdentityPreservationError",
    "QualityAssuranceError", "PromptPackageInvalidError", "ReferenceAssetError", "WorkflowTemplateError",
    "WorkflowBuildError", "ArtifactWriteError", "MetricsWriteError", "NoEligibleCandidateError", "ProfileDowngradedWarning",
]
