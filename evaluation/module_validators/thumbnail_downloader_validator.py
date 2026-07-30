"""
thumbnail_downloader_validator.py
==================================

Validator for Module 3 — Thumbnail Downloader output.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from modules.models import ModuleValidationResult, ThumbnailData
from .interfaces import IModuleValidator


class ThumbnailDownloaderValidator(IModuleValidator):
    @property
    def module_name(self) -> str:
        return "module3_thumbnail_downloader"

    def validate(self, video_id: str, artifact_path: Path) -> ModuleValidationResult:
        start_time = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat()
        invariants_checked = [
            "artifact_file_exists",
            "schema_conformance",
            "video_id_match",
            "image_file_exists",
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
            data = ThumbnailData.model_validate(raw_data)

            if data.metadata.video_id != video_id:
                invariants_failed.append("video_id_match")
            img_p = Path(data.thumbnail_path)
            if not img_p.is_absolute():
                img_p = artifact_path.parent / img_p
            if not img_p.exists() or img_p.stat().st_size == 0:
                invariants_failed.append("image_file_exists")

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
