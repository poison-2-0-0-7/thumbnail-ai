"""
observability/facts/extractor.py
=================================

FactExtractor converts a PipelineTrace into deterministic structured facts.
Facts are observations, NOT conclusions, diagnostics, or recommendations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from observability.config import OBS_FACTS_VERSION
from observability.facts.interfaces import IFactExtractor
from observability.facts.models import FactCollection, FactModel, TraceFacts
from observability.facts.registry import FactRegistry
from observability.models import GenerationTraceRecord, PipelineTrace


class FactExtractor(IFactExtractor):
    """
    Extracts deterministic, reproducible observation facts from a PipelineTrace.
    """

    def __init__(self, registry: Optional[FactRegistry] = None) -> None:
        self.registry = registry or FactRegistry()

    def extract(self, trace: PipelineTrace) -> FactCollection:
        """
        Extract facts from PipelineTrace and return a FactCollection.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        vid = trace.video_id

        # 1. Inspect Module Entries for timing, status, outputs, config
        module_status: dict[str, str] = {}
        timing: dict[str, Optional[float]] = {}
        outputs: dict[str, Optional[str]] = {}
        availability: dict[str, bool] = {}

        asset_extraction_enabled = False
        decision_engine_enabled = False
        thumbnail_planner_enabled = False

        total_duration = 0.0
        has_timing = False

        for entry in trace.modules:
            mod = entry.module
            module_status[mod] = entry.status
            timing[mod] = entry.duration_seconds
            if entry.duration_seconds is not None:
                total_duration += entry.duration_seconds
                has_timing = True

            # Config snapshot checks
            cfg_snap = entry.config_snapshot or {}
            if "ASSET_EXTRACTION_ENABLED" in cfg_snap:
                asset_extraction_enabled = bool(cfg_snap["ASSET_EXTRACTION_ENABLED"])
            if "DECISION_ENGINE_ENABLED" in cfg_snap:
                decision_engine_enabled = bool(cfg_snap["DECISION_ENGINE_ENABLED"])
            if "THUMBNAIL_PLANNER_ENABLED" in cfg_snap:
                thumbnail_planner_enabled = bool(cfg_snap["THUMBNAIL_PLANNER_ENABLED"])

            # Outputs & Availability
            for ref in entry.outputs:
                key = f"{mod}_{ref.artifact_type}"
                availability[key] = ref.exists
                if ref.exists and ref.path:
                    outputs[mod] = ref.path

        # 2. Inspect Artifact Index directly
        for ref in trace.artifact_index.refs:
            key = f"{ref.module}_{ref.artifact_type}"
            availability[key] = ref.exists
            if ref.exists and ref.path and ref.module not in outputs:
                outputs[ref.module] = ref.path

        # 3. Source & Generated thumbnail checks
        m3_outputs = [r for r in trace.artifact_index.refs if r.module == "module3"]
        source_thumb_exists = any(r.exists for r in m3_outputs)

        m7_outputs = [r for r in trace.artifact_index.refs if r.module == "module7"]
        gen_thumb_exists = any(r.exists for r in m7_outputs)

        # 4. Composition workspace check
        m10_refs = [r for r in trace.artifact_index.refs if r.module == "module10"]
        has_comp_ws = any(r.exists for r in m10_refs)
        comp_ws_path = next((r.path for r in m10_refs if r.exists and r.path), None)

        # 5. Inspect GenerationTraceRecord if present
        gen_rec: Optional[GenerationTraceRecord] = None
        if isinstance(trace.generation_trace, GenerationTraceRecord):
            gen_rec = trace.generation_trace
        elif isinstance(trace.generation_trace, dict):
            try:
                gen_rec = GenerationTraceRecord.model_validate(trace.generation_trace)
            except Exception:
                gen_rec = None

        workflow_selected: Optional[str] = None
        edit_mode: Optional[str] = None
        generation_profile: Optional[str] = None
        model_used: Optional[str] = None
        sampler: Optional[str] = None
        scheduler: Optional[str] = None
        seed: Optional[int] = None
        cfg: Optional[float] = None
        steps: Optional[int] = None
        denoise: Optional[float] = None
        latent_initialization_mode: Optional[str] = None

        controlnet_count = 0
        controlnet_configuration: dict[str, Any] = {}
        controlnet_enabled = False
        ipadapter_count = 0
        ipadapter_configuration: dict[str, Any] = {}
        ipadapter_enabled = False

        mask_count = 0
        edit_mask_paths: list[str] = []
        conditioning_assets: list[str] = []
        background_assets: list[str] = []
        foreground_assets: list[str] = []

        gen_plan_ref: Optional[str] = None
        prompt_ref: Optional[str] = None
        neg_prompt_ref: Optional[str] = None
        pos_prompt: Optional[str] = None
        neg_prompt: Optional[str] = None

        renderer_version: Optional[str] = None
        attached_fragment_count = 0
        attached_fragment_names: list[str] = []

        if gen_rec is not None:
            workflow_selected = gen_rec.workflow_template or None
            edit_mode = gen_rec.edit_mode or None
            generation_profile = gen_rec.generation_profile or None
            model_used = gen_rec.model_version or None
            sampler = gen_rec.sampler or None
            scheduler = gen_rec.scheduler or None
            seed = gen_rec.seed
            cfg = gen_rec.cfg
            steps = gen_rec.steps
            denoise = gen_rec.denoise

            # Map latent source to latent initialization mode
            if gen_rec.latent_source == "noise":
                latent_initialization_mode = "EmptyLatentImage"
            else:
                latent_initialization_mode = gen_rec.latent_source

            controlnet_enabled = gen_rec.controlnet_enabled
            controlnet_count = 1 if controlnet_enabled else 0
            controlnet_configuration = gen_rec.controlnet_config or {}

            ipadapter_enabled = gen_rec.ipadapter_enabled
            ipadapter_count = 1 if ipadapter_enabled else 0
            ipadapter_configuration = gen_rec.ipadapter_config or {}

            edit_mask_paths = list(gen_rec.edit_mask_paths or [])
            mask_refs = list(gen_rec.mask_references or [])
            combined_masks = list(set(edit_mask_paths + mask_refs))
            mask_count = len(combined_masks)

            conditioning_assets = list(gen_rec.conditioning_assets or [])
            asset_refs = list(gen_rec.asset_references or [])

            # Derive foreground / background asset references if present
            for path_str in asset_refs:
                lower = path_str.lower()
                if "bg" in lower or "background" in lower:
                    background_assets.append(path_str)
                else:
                    foreground_assets.append(path_str)

            gen_plan_ref = gen_rec.generation_plan_reference or outputs.get("module10.5")
            prompt_ref = gen_rec.prompt_reference or outputs.get("module6")
            neg_prompt_ref = gen_rec.negative_prompt_reference

            renderer_version = gen_rec.renderer_version or "1.0.0"

            attached_fragment_count = len(gen_rec.fragments_attached or [])
            attached_fragment_names = [f.fragment_name for f in (gen_rec.fragments_attached or [])]

            beats_original = getattr(gen_rec, "beats_original", None)
            over_edited = getattr(gen_rec, "over_edited", None)
            selection_agreed = getattr(gen_rec, "selection_agreed", None)
            baseline_score = getattr(gen_rec, "baseline_score", None)
            winning_candidate_index = getattr(gen_rec, "winning_candidate_index", None)
            module7_selected_index = getattr(gen_rec, "module7_selected_index", None)
            edit_magnitude = getattr(gen_rec, "edit_magnitude", None)
        else:
            # Fallback to module outputs if generation_rec is missing
            gen_plan_ref = outputs.get("module10.5")
            prompt_ref = outputs.get("module6")
            beats_original = None
            over_edited = None
            selection_agreed = None
            baseline_score = None
            winning_candidate_index = None
            module7_selected_index = None
            edit_magnitude = None

        # Build TraceFacts model
        trace_facts = TraceFacts(
            video_id=vid,
            extracted_at=now_str,
            fact_version=OBS_FACTS_VERSION,
            workflow_selected=workflow_selected,
            edit_mode=edit_mode,
            generation_profile=generation_profile,
            model_used=model_used,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            cfg=cfg,
            steps=steps,
            denoise=denoise,
            latent_initialization_mode=latent_initialization_mode,
            controlnet_count=controlnet_count,
            controlnet_configuration=controlnet_configuration,
            controlnet_enabled=controlnet_enabled,
            ipadapter_count=ipadapter_count,
            ipadapter_configuration=ipadapter_configuration,
            ipadapter_enabled=ipadapter_enabled,
            mask_count=mask_count,
            edit_mask_paths=edit_mask_paths,
            conditioning_assets=conditioning_assets,
            background_assets=background_assets,
            foreground_assets=foreground_assets,
            composition_workspace=comp_ws_path,
            has_composition_workspace=has_comp_ws,
            generation_plan_reference=gen_plan_ref,
            prompt_reference=prompt_ref,
            negative_prompt_reference=neg_prompt_ref,
            positive_prompt=pos_prompt,
            negative_prompt=neg_prompt,
            renderer_version=renderer_version,
            execution_timing=timing,
            total_execution_time_seconds=total_duration if has_timing else None,
            artifact_availability=availability,
            persisted_outputs=outputs,
            module_completion_status=module_status,
            attached_fragment_count=attached_fragment_count,
            attached_fragment_names=attached_fragment_names,
            source_thumbnail_exists=source_thumb_exists,
            generated_thumbnail_exists=gen_thumb_exists,
            asset_extraction_enabled=asset_extraction_enabled,
            decision_engine_enabled=decision_engine_enabled,
            thumbnail_planner_enabled=thumbnail_planner_enabled,
            beats_original=beats_original,
            over_edited=over_edited,
            selection_agreed=selection_agreed,
            baseline_score=baseline_score,
            winning_candidate_index=winning_candidate_index,
            module7_selected_index=module7_selected_index,
            edit_magnitude=edit_magnitude,
        )

        # Build individual FactModel list
        atomic_facts: list[FactModel] = [
            FactModel(
                fact_key="workflow_selected",
                category="generation",
                name="Workflow Selected",
                value=workflow_selected,
                data_type="string",
                source_module="module7",
            ),
            FactModel(
                fact_key="edit_mode",
                category="generation",
                name="Edit Mode",
                value=edit_mode,
                data_type="string",
                source_module="module7",
            ),
            FactModel(
                fact_key="generation_profile",
                category="generation",
                name="Generation Profile",
                value=generation_profile,
                data_type="string",
                source_module="module7",
            ),
            FactModel(
                fact_key="latent_initialization_mode",
                category="generation",
                name="Latent Initialization Mode",
                value=latent_initialization_mode,
                data_type="string",
                source_module="module7",
            ),
            FactModel(
                fact_key="controlnet_count",
                category="generation",
                name="ControlNet Count",
                value=controlnet_count,
                data_type="int",
                source_module="module7",
            ),
            FactModel(
                fact_key="ipadapter_count",
                category="generation",
                name="IPAdapter Count",
                value=ipadapter_count,
                data_type="int",
                source_module="module7",
            ),
            FactModel(
                fact_key="mask_count",
                category="generation",
                name="Mask Count",
                value=mask_count,
                data_type="int",
                source_module="module7",
            ),
            FactModel(
                fact_key="has_composition_workspace",
                category="composition",
                name="Has Composition Workspace",
                value=has_comp_ws,
                data_type="bool",
                source_module="module10",
            ),
            FactModel(
                fact_key="source_thumbnail_exists",
                category="artifact",
                name="Source Thumbnail Exists",
                value=source_thumb_exists,
                data_type="bool",
                source_module="module3",
            ),
            FactModel(
                fact_key="generated_thumbnail_exists",
                category="artifact",
                name="Generated Thumbnail Exists",
                value=gen_thumb_exists,
                data_type="bool",
                source_module="module7",
            ),
        ]

        # Add custom facts from registry if any custom handlers exist
        custom_facts = self.registry.extract_all_custom_facts(trace)
        for cat, facts_dict in custom_facts.items():
            for key, val in facts_dict.items():
                atomic_facts.append(
                    FactModel(
                        fact_key=key,
                        category=cat,
                        name=key.replace("_", " ").title(),
                        value=val,
                        data_type=type(val).__name__,
                        source_module="custom",
                    )
                )

        return FactCollection(
            video_id=vid,
            trace_facts=trace_facts,
            atomic_facts=atomic_facts,
            extracted_at=now_str,
            fact_version=OBS_FACTS_VERSION,
        )
