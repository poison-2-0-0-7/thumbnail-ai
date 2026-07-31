"""
module7_validator.py
====================

Validator for Module 7 — Image Generation Engine output.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from modules.models import ImageGenerationResult, ModuleValidationResult
from .interfaces import IModuleValidator


class Module7Validator(IModuleValidator):
    @property
    def module_name(self) -> str:
        return "module7_image_generator"

    def validate(self, video_id: str, artifact_path: Path) -> ModuleValidationResult:
        start_time = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat()
        invariants_checked = [
            "artifact_file_exists",
            "schema_conformance",
            "video_id_match",
            "generated_asset_present",
            "generated_image_file_exists",
            "generated_image_non_empty",
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
            res = ImageGenerationResult.model_validate(raw_data)

            if res.video_id != video_id:
                invariants_failed.append("video_id_match")

            if res.status != "success":
                invariants_failed.append("generated_asset_present")
            elif res.generated_asset is None:
                invariants_failed.append("generated_asset_present")
            else:
                img_path = Path(res.generated_asset.path)
                if not img_path.is_absolute():
                    img_path = artifact_path.parent / img_path
                if not img_path.exists():
                    invariants_failed.append("generated_image_file_exists")
                elif img_path.stat().st_size == 0:
                    invariants_failed.append("generated_image_non_empty")

            cand_manifest_path = artifact_path.parent / "candidate_manifest.json"
            if cand_manifest_path.is_file():
                invariants_checked.append("candidate_manifest_valid")
                try:
                    from modules.models import CandidateManifest
                    CandidateManifest.model_validate_json(cand_manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    invariants_failed.append("candidate_manifest_valid")

            gen_meta_path = artifact_path.parent / "generation_metadata.json"
            if gen_meta_path.is_file():
                invariants_checked.append("generation_metadata_valid")
                try:
                    from modules.models import GenerationRunMetadata
                    GenerationRunMetadata.model_validate_json(gen_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    invariants_failed.append("generation_metadata_valid")


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
