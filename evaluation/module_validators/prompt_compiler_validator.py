"""
prompt_compiler_validator.py
=============================

Validator for Module 6 — Prompt Compiler output.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from modules.models import ModuleValidationResult, PromptPackage
from .interfaces import IModuleValidator


class PromptCompilerValidator(IModuleValidator):
    @property
    def module_name(self) -> str:
        return "module6_prompt_compiler"

    def validate(self, video_id: str, artifact_path: Path) -> ModuleValidationResult:
        start_time = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat()
        invariants_checked = [
            "artifact_file_exists",
            "schema_conformance",
            "video_id_match",
            "positive_prompt_non_empty",
            "generation_parameters_present",
            "seed_non_negative",
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
            package = PromptPackage.model_validate(raw_data)

            if package.video_id != video_id:
                invariants_failed.append("video_id_match")
            if not package.positive_prompt or not package.positive_prompt.strip():
                invariants_failed.append("positive_prompt_non_empty")
            if package.generation_parameters is None:
                invariants_failed.append("generation_parameters_present")
            elif package.generation_parameters.seed < 0:
                invariants_failed.append("seed_non_negative")

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
