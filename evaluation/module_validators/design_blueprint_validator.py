"""design_blueprint_validator.py

Validator for Module 5.5 — Thumbnail Copywriter & Layout Planner output.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from modules.models import DesignBlueprint, ModuleValidationResult
from .interfaces import IModuleValidator


class DesignBlueprintValidator(IModuleValidator):
    @property
    def module_name(self) -> str:
        return "module5_5_design_blueprint"

    def validate(self, video_id: str, artifact_path: Path) -> ModuleValidationResult:
        start_time = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat()
        invariants_checked = [
            "artifact_file_exists",
            "schema_conformance",
            "video_id_match",
            "headline_non_empty",
            "text_position_valid",
            "camera_distance_valid",
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
            blueprint = DesignBlueprint.model_validate(raw_data)

            if blueprint.video_id != video_id:
                invariants_failed.append("video_id_match")
            if not blueprint.headline or not blueprint.headline.strip():
                invariants_failed.append("headline_non_empty")
            if blueprint.text_position is None:
                invariants_failed.append("text_position_valid")
            if not blueprint.camera_distance:
                invariants_failed.append("camera_distance_valid")

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
