"""
observability/trace/trace_assembler.py
======================================

Assembles a PipelineTrace for a given video_id from an ArtifactIndex and correlated LogLineRefs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from modules import config
from observability.interfaces import ITraceAssembler
from observability.models import (
    ArtifactIndex,
    ArtifactRef,
    LogLineRef,
    ModuleTraceEntry,
    PipelineTrace,
)


def _parse_iso(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse timestamp string to datetime."""
    if not ts_str:
        return None
    ts_str = ts_str.replace("T", " ").strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class TraceAssembler(ITraceAssembler):
    """
    Assembles a full PipelineTrace for a single video_id based on ArtifactIndex and LogLineRefs.
    """

    def __init__(self, metrics_path: Optional[Path] = None) -> None:
        self.metrics_path = metrics_path or getattr(config, "MODULE7_METRICS_PATH", config.LOG_DIR / "module7_metrics.jsonl")

    def _get_module7_exact_duration(self, video_id: str) -> Optional[float]:
        """Attempt to read exact duration from module7_metrics.jsonl."""
        if not self.metrics_path.exists():
            return None
        try:
            with open(self.metrics_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if video_id in line:
                        try:
                            data = json.loads(line)
                            if data.get("video_id") == video_id and "total_duration_seconds" in data:
                                return float(data["total_duration_seconds"])
                        except Exception:
                            continue
        except Exception:
            pass
        return None

    def assemble_from_parts(
        self,
        video_id: str,
        artifact_index: ArtifactIndex,
        log_lines: list[LogLineRef],
    ) -> PipelineTrace:
        """
        Build a PipelineTrace using pre-collected ArtifactIndex and LogLineRefs.
        """
        # Map artifact refs by artifact_type or module
        ref_map: dict[str, list[ArtifactRef]] = {}
        for ref in artifact_index.refs:
            ref_map.setdefault(ref.module, []).append(ref)

        # Helper to retrieve refs by module name
        def get_refs(mod: str) -> list[ArtifactRef]:
            return ref_map.get(mod, [])

        # Define module stage configurations in pipeline order
        stages_definition = [
            {
                "module": "module1",
                "stage_order": 1,
                "flag": None,
                "input_modules": [],
                "config": {"CSV_ENCODING": getattr(config, "CSV_ENCODING", "utf-8")},
            },
            {
                "module": "module2",
                "stage_order": 2,
                "flag": None,
                "input_modules": ["module1"],
                "config": {},
            },
            {
                "module": "module3",
                "stage_order": 3,
                "flag": None,
                "input_modules": ["module2"],
                "config": {
                    "TIMEOUT": getattr(config, "THUMBNAIL_REQUEST_TIMEOUT_SECONDS", 30.0),
                    "ACCEPTED_FORMATS": list(getattr(config, "THUMBNAIL_ACCEPTED_IMAGE_FORMATS", [])),
                },
            },
            {
                "module": "module4",
                "stage_order": 4,
                "flag": None,
                "input_modules": ["module3"],
                "config": {
                    "YOLO_MODEL": getattr(config, "YOLO_MODEL_NAME", "yolo11n.pt"),
                    "FACE_MODEL": getattr(config, "FACE_MODEL_NAME", "buffalo_l"),
                    "OLLAMA_MODEL": getattr(config, "OLLAMA_MODEL", "qwen3:8b"),
                },
            },
            {
                "module": "module8",
                "stage_order": 5,
                "flag": getattr(config, "ASSET_EXTRACTION_ENABLED", False),
                "input_modules": ["module3", "module4"],
                "config": {
                    "ASSET_EXTRACTION_ENABLED": getattr(config, "ASSET_EXTRACTION_ENABLED", False)
                },
            },
            {
                "module": "module5",
                "stage_order": 6,
                "flag": None,
                "input_modules": ["module4"],
                "config": {
                    "CLUTTER_HIGH_THRESHOLD": getattr(config, "CLUTTER_HIGH_THRESHOLD", 0.6)
                },
            },
            {
                "module": "module5.5",
                "stage_order": 7,
                "flag": None,
                "input_modules": ["module5", "module4"],
                "config": {},
            },
            {
                "module": "module6",
                "stage_order": 8,
                "flag": None,
                "input_modules": ["module5.5", "module5"],
                "config": {},
            },
            {
                "module": "module9",
                "stage_order": 9,
                "flag": getattr(config, "DECISION_ENGINE_ENABLED", False),
                "input_modules": ["module4", "module8", "module5"],
                "config": {
                    "DECISION_ENGINE_ENABLED": getattr(config, "DECISION_ENGINE_ENABLED", False)
                },
            },
            {
                "module": "module10",
                "stage_order": 10,
                "flag": None,
                "input_modules": ["module6", "module9", "module5.5"],
                "config": {},
            },
            {
                "module": "module10.5",
                "stage_order": 11,
                "flag": getattr(config, "THUMBNAIL_PLANNER_ENABLED", False),
                "input_modules": ["module10", "module9"],
                "config": {
                    "THUMBNAIL_PLANNER_ENABLED": getattr(config, "THUMBNAIL_PLANNER_ENABLED", False)
                },
            },
            {
                "module": "module7",
                "stage_order": 12,
                "flag": None,
                "input_modules": ["module6", "module10", "module10.5"],
                "config": {
                    "COMFYUI_ENABLED": getattr(config, "COMFYUI_ENABLED", True)
                },
            },
        ]

        module_entries: list[ModuleTraceEntry] = []

        for stage in stages_definition:
            mod_name = stage["module"]
            order = stage["stage_order"]
            flag = stage["flag"]
            input_mods = stage["input_modules"]
            cfg_snap = stage["config"]

            outputs = get_refs(mod_name)
            inputs: list[ArtifactRef] = []
            for in_mod in input_mods:
                inputs.extend(get_refs(in_mod))

            # Filter log lines relevant to this module
            mod_logs = [
                line
                for line in log_lines
                if line.module == mod_name or line.module.startswith(f"{mod_name}.")
            ]

            errors = [l.message for l in mod_logs if l.level in ("ERROR", "CRITICAL")]
            warnings = [l.message for l in mod_logs if l.level == "WARNING"]

            started_at = mod_logs[0].timestamp if mod_logs else None
            completed_at = mod_logs[-1].timestamp if mod_logs else None

            # Calculate duration
            duration_seconds: Optional[float] = None
            duration_source: str = "unavailable"

            if mod_name == "module7":
                exact_dur = self._get_module7_exact_duration(video_id)
                if exact_dur is not None:
                    duration_seconds = exact_dur
                    duration_source = "exact"

            if duration_seconds is None and started_at and completed_at:
                dt_start = _parse_iso(started_at)
                dt_end = _parse_iso(completed_at)
                if dt_start and dt_end:
                    delta = (dt_end - dt_start).total_seconds()
                    if delta >= 0:
                        duration_seconds = delta
                        duration_source = "log_derived"

            # Determine status
            if flag is False:
                status = "not_run"
            else:
                has_existing_output = any(r.exists for r in outputs)
                if has_existing_output:
                    status = "partial" if errors else "success"
                elif mod_name in ("module1", "module2") and not errors:
                    # In-memory or shared file modules
                    status = "success"
                elif outputs:
                    status = "error"
                else:
                    status = "success" if not errors else "error"

            entry = ModuleTraceEntry(
                module=mod_name,
                stage_order=order,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration_seconds,
                duration_source=duration_source,
                inputs=inputs,
                outputs=outputs,
                config_snapshot=cfg_snap,
                log_lines=mod_logs,
                errors=errors,
                warnings=warnings,
            )
            module_entries.append(entry)

        # Compute overall status
        non_disabled = [m for m in module_entries if m.status != "not_run"]
        if all(m.status == "success" for m in non_disabled):
            overall_status = "success"
        elif any(m.status == "error" for m in non_disabled):
            overall_status = "error"
        else:
            overall_status = "partial"

        assembled_at = datetime.now(timezone.utc).isoformat()

        # Load generation trace record if present
        from observability.generation_trace import GenerationTracePersistence
        gen_trace = GenerationTracePersistence().load(video_id)

        return PipelineTrace(
            video_id=video_id,
            modules=module_entries,
            artifact_index=artifact_index,
            generation_trace=gen_trace,
            overall_status=overall_status,
            assembled_at=assembled_at,
        )

    def assemble(self, video_id: str) -> PipelineTrace:
        """
        Primary interface method; requires pre-building index and log correlation if called directly.
        Use PipelineTraceBuilder for end-to-end convenience.
        """
        from observability.trace.artifact_index_builder import ArtifactIndexBuilder
        from observability.trace.log_correlator import LogCorrelator

        index_builder = ArtifactIndexBuilder()
        log_correlator = LogCorrelator()

        artifact_index = index_builder.collect(video_id)
        log_lines = log_correlator.correlate(video_id)
        return self.assemble_from_parts(video_id, artifact_index, log_lines)
