"""
observability/generation_trace.py
=================================

Factory, Persistence, and Recorder for Module 7 GenerationTraceRecords.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import ValidationError

from observability.config import OBS_GENERATION_TRACES_DIR
from observability.models import (
    FragmentAttachmentRecord,
    GenerationTraceRecord,
)


class GenerationTraceFactory:
    """
    Factory for creating GenerationTraceRecord instances from Module 7 execution context.
    """

    @staticmethod
    def create(
        video_id: str,
        attempt_index: int = 0,
        package: Any | None = None,
        profile: Any | None = None,
        built_wf: Any | None = None,
        conditioning_ctx: Any | None = None,
        generation_plan: Any | None = None,
        output_image_path: Optional[Path | str] = None,
        stage_durations: Optional[dict[str, float]] = None,
        fragments_attached: Optional[list[FragmentAttachmentRecord | dict[str, Any]]] = None,
        generation_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
        cluster_id: Optional[str] = None,
        exclusion_reason: Optional[str] = None,
        ranking_dimensions: Optional[dict[str, float]] = None,
        selection_explanation: Optional[str] = None,
        manual_override: bool = False,
        **kwargs: Any,
    ) -> GenerationTraceRecord:
        """
        Construct a strongly-typed GenerationTraceRecord from Module 7 execution context.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        gen_id = generation_id or f"{video_id}_cand_{attempt_index}"

        workflow_template = ""
        workflow_hash = ""
        if built_wf is not None:
            workflow_hash = getattr(built_wf, "workflow_hash", "")
            wf_ref = getattr(built_wf, "workflow_ref", None)
            if wf_ref:
                workflow_template = getattr(wf_ref, "template_name", str(wf_ref))

        # Attached fragments
        parsed_frags: list[FragmentAttachmentRecord] = []
        if fragments_attached:
            for item in fragments_attached:
                if isinstance(item, FragmentAttachmentRecord):
                    parsed_frags.append(item)
                elif isinstance(item, dict):
                    parsed_frags.append(
                        FragmentAttachmentRecord(
                            fragment_name=item.get("fragment_name", "unknown"),
                            attach_point=item.get("attach_point", "unknown"),
                            strength_or_weight=item.get("strength_or_weight"),
                            requested_capability=item.get("requested_capability"),
                            resolved_model=item.get("resolved_model"),
                            resolution_source=item.get("resolution_source"),
                            fallback_path=item.get("fallback_path", False),
                            compatibility_decision=item.get("compatibility_decision"),
                        )
                    )

        # Profile parameters
        profile_name = getattr(profile, "name", None) if profile else None
        controlnet_enabled = bool(getattr(profile, "controlnet_enabled", False)) if profile else False
        ipadapter_enabled = bool(getattr(profile, "ipadapter_enabled", False)) if profile else False
        seed = int(package.generation_parameters.seed) if (package and hasattr(package, "generation_parameters")) else 0
        scheduler = getattr(profile, "scheduler", None) if profile else None
        sampler = getattr(profile, "sampler", None) if profile else None
        steps = int(getattr(profile, "steps", 0)) if (profile and hasattr(profile, "steps")) else None
        cfg = float(getattr(profile, "cfg", 0.0)) if (profile and hasattr(profile, "cfg")) else None

        # Conditioning paths
        cond_assets: list[str] = []
        asset_refs: list[str] = []
        mask_refs: list[str] = []
        src_thumb: Optional[str] = None

        if conditioning_ctx is not None:
            if getattr(conditioning_ctx, "depth_path", None):
                cond_assets.append(str(conditioning_ctx.depth_path))
            if getattr(conditioning_ctx, "canny_path", None):
                cond_assets.append(str(conditioning_ctx.canny_path))
            if getattr(conditioning_ctx, "segmentation_path", None):
                cond_assets.append(str(conditioning_ctx.segmentation_path))
            if getattr(conditioning_ctx, "source_thumbnail_path", None):
                src_thumb = str(conditioning_ctx.source_thumbnail_path)

            role_imgs = getattr(conditioning_ctx, "role_image_paths", {}) or {}
            for path in role_imgs.values():
                if path:
                    asset_refs.append(str(path))

            role_masks = getattr(conditioning_ctx, "role_mask_paths", {}) or {}
            for path in role_masks.values():
                if path:
                    mask_refs.append(str(path))

        out_img_str = str(output_image_path) if output_image_path else None

        timestamps = {"recorded_at": now_str}
        if stage_durations:
            for k, v in stage_durations.items():
                timestamps[f"duration_{k}"] = f"{v:.4f}s"

        # Determine latent_source, denoise, and edit_mode truthfully from built_wf.graph
        latent_source = "noise"
        denoise = 1.0
        edit_mode = "txt2img"

        if built_wf is not None:
            graph = getattr(built_wf, "graph", None)
            if isinstance(graph, dict):
                ksampler_node = None
                for node in graph.values():
                    if isinstance(node, dict) and node.get("class_type") == "KSampler":
                        ksampler_node = node
                        break
                if ksampler_node is None and "5" in graph and isinstance(graph["5"], dict):
                    ksampler_node = graph["5"]

                if ksampler_node is not None:
                    inputs = ksampler_node.get("inputs", {})
                    if "denoise" in inputs:
                        try:
                            denoise = float(inputs["denoise"])
                        except (ValueError, TypeError):
                            pass

                has_inpaint = any(
                    isinstance(node, dict) and node.get("class_type") in ("VAEEncodeForInpaint", "LoadImage")
                    for node in graph.values()
                )
                is_edit_wf = bool(workflow_template and (workflow_template.endswith("_edit") or "_edit" in str(workflow_template)))

                if has_inpaint or is_edit_wf or (denoise < 1.0):
                    edit_mode = "staged_edit"
                    if has_inpaint or is_edit_wf:
                        latent_source = "vae_encoded_source"

        return GenerationTraceRecord(
            video_id=video_id,
            attempt_index=attempt_index,
            generation_id=gen_id,
            workflow_template=workflow_template,
            workflow_hash=workflow_hash,
            workflow_fragments=[f.fragment_name for f in parsed_frags],
            fragments_attached=parsed_frags,
            latent_source=latent_source,
            denoise=denoise,
            seed=seed,
            scheduler=scheduler,
            sampler=sampler,
            steps=steps,
            cfg=cfg,
            controlnet_enabled=controlnet_enabled,
            ipadapter_enabled=ipadapter_enabled,
            edit_mode=edit_mode,
            generation_profile=profile_name,
            controlnet_config={"enabled": controlnet_enabled},
            ipadapter_config={"enabled": ipadapter_enabled},
            conditioning_assets=cond_assets,
            asset_references=asset_refs,
            mask_references=mask_refs,
            source_thumbnail_path=src_thumb,
            output_image_path=out_img_str,
            execution_timestamps=timestamps,
            recorded_at=now_str,
            strategy_name=strategy_name,
            cluster_id=cluster_id,
            exclusion_reason=exclusion_reason,
            ranking_dimensions=ranking_dimensions,
            selection_explanation=selection_explanation,
            manual_override=manual_override,
            creator_channel_id=kwargs.get("creator_channel_id"),
            style_signature_reference=kwargs.get("style_signature_reference"),
            style_embedding_similarity=kwargs.get("style_embedding_similarity"),
            style_profile_established=kwargs.get("style_profile_established"),
            style_bonus_applied=kwargs.get("style_bonus_applied"),
            drift_detected=kwargs.get("drift_detected"),
            drift_confidence=kwargs.get("drift_confidence"),
            style_prompt_guidance_applied=kwargs.get("style_prompt_guidance_applied"),
        )


class GenerationTracePersistence:
    """
    Handles atomic persistence, loading, and validation of GenerationTraceRecord.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or OBS_GENERATION_TRACES_DIR

    def get_target_path(self, video_id: str, attempt_index: int = 0) -> Path:
        """Return the target file path for a generation trace record."""
        target_dir = self.output_dir / video_id
        if attempt_index == 0:
            return target_dir / "generation_trace_record.json"
        return target_dir / f"generation_trace_{attempt_index}.json"

    def save(self, record: GenerationTraceRecord) -> Path:
        """
        Atomically write a GenerationTraceRecord to disk.
        """
        target_path = self.get_target_path(record.video_id, record.attempt_index)
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix=".gentrace_tmp_", suffix=".json")
        try:
            with open(tmp_fd, "w", encoding="utf-8") as f:
                f.write(record.model_dump_json(indent=2))
            Path(tmp_path).replace(target_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        logger.debug("Saved GenerationTraceRecord for video_id={vid} to {path}", vid=record.video_id, path=target_path)
        return target_path

    def load_file(self, path: Path) -> GenerationTraceRecord:
        """
        Load and validate a GenerationTraceRecord from a specific file.
        """
        if not path.is_file():
            raise FileNotFoundError(f"GenerationTraceRecord file not found: {path}")

        content = path.read_text(encoding="utf-8")
        return GenerationTraceRecord.model_validate_json(content)

    def load(self, video_id: str, attempt_index: int = 0) -> Optional[GenerationTraceRecord]:
        """
        Load the GenerationTraceRecord for video_id if present.
        """
        target_path = self.get_target_path(video_id, attempt_index)
        if not target_path.is_file():
            return None
        try:
            return self.load_file(target_path)
        except Exception as exc:
            logger.warning("Could not load GenerationTraceRecord at {path}: {exc}", path=target_path, exc=exc)
            return None

    def validate(self, data: Any) -> bool:
        """
        Validate data dictionary or model against GenerationTraceRecord schema.
        """
        if isinstance(data, GenerationTraceRecord):
            return True
        if isinstance(data, dict):
            try:
                GenerationTraceRecord.model_validate(data)
                return True
            except ValidationError:
                return False
        return False


class GenerationTraceRecorder:
    """
    High-level recorder for Module 7 generation attempts.
    All record operations are non-fatal side-effects wrapped in exception handlers.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.persistence = GenerationTracePersistence(output_dir=output_dir)

    def record(
        self,
        video_id: str,
        attempt_index: int = 0,
        package: Any | None = None,
        profile: Any | None = None,
        built_wf: Any | None = None,
        conditioning_ctx: Any | None = None,
        generation_plan: Any | None = None,
        output_image_path: Optional[Path | str] = None,
        stage_durations: Optional[dict[str, float]] = None,
        fragments_attached: Optional[list[FragmentAttachmentRecord | dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[Path]:
        """
        Construct and persist a GenerationTraceRecord for a generation attempt.
        Never raises exceptions to caller.
        """
        try:
            record = GenerationTraceFactory.create(
                video_id=video_id,
                attempt_index=attempt_index,
                package=package,
                profile=profile,
                built_wf=built_wf,
                conditioning_ctx=conditioning_ctx,
                generation_plan=generation_plan,
                output_image_path=output_image_path,
                stage_durations=stage_durations,
                fragments_attached=fragments_attached,
                **kwargs,
            )
            return self.persistence.save(record)

        except Exception as exc:
            logger.warning(
                "GenerationTraceRecorder failed to record trace for video_id={vid}: {exc}",
                vid=video_id,
                exc=exc,
            )
            return None
