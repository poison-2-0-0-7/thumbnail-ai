"""
redesign_spec_validator.py
===========================

Validator for Module 5 — Redesign Specification Engine output.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from modules.models import ModuleValidationResult, RedesignSpecification
from .interfaces import IModuleValidator


class RedesignSpecValidator(IModuleValidator):
    @property
    def module_name(self) -> str:
        return "module5_redesign_spec"

    def validate(self, video_id: str, artifact_path: Path) -> ModuleValidationResult:
        start_time = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat()
        invariants_checked = [
            "artifact_file_exists",
            "schema_conformance",
            "video_id_match",
            "source_thumbnail_path_non_empty",
            "color_direction_valid",
            "subject_treatment_valid",
        ]
        invariants_failed = []

        if not artifact_path.exists():
            return ModuleValidationResult(
                video_id=video_id,
                module_name=self.module_name,
                artifact_path=str(artifact_path),
                schema_valid=False,
                invariants_checked=invariants_checked,
                invariants_failed=["artifact_file_exists"],
                status="error",
                error_message=f"Artifact missing: {artifact_path}",
                duration_seconds=time.monotonic() - start_time,
                validated_at=now_iso,
            )

        try:
            raw_data = json.loads(artifact_path.read_text(encoding="utf-8"))
            spec = RedesignSpecification.model_validate(raw_data)

            if spec.video_id != video_id:
                invariants_failed.append("video_id_match")
            if not spec.source_thumbnail_path or not spec.source_thumbnail_path.strip():
                invariants_failed.append("source_thumbnail_path_non_empty")
            if spec.color_direction is None:
                invariants_failed.append("color_direction_valid")
            if spec.subject_treatment is None:
                invariants_failed.append("subject_treatment_valid")

            status = "success" if not invariants_failed else "partial"
            return ModuleValidationResult(
                video_id=video_id,
                module_name=self.module_name,
                artifact_path=str(artifact_path),
                schema_valid=True,
                invariants_checked=invariants_checked,
                invariants_failed=invariants_failed,
                status=status,
                error_message=None if status == "success" else f"Failed invariants: {invariants_failed}",
                duration_seconds=time.monotonic() - start_time,
                validated_at=now_iso,
            )

        except Exception as exc:
            return ModuleValidationResult(
                video_id=video_id,
                module_name=self.module_name,
                artifact_path=str(artifact_path),
                schema_valid=False,
                invariants_checked=invariants_checked,
                invariants_failed=["schema_conformance"],
                status="error",
                error_message=str(exc),
                duration_seconds=time.monotonic() - start_time,
                validated_at=now_iso,
            )
