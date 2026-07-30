"""
csv_reader_validator.py
========================

Validator for Module 1 — CSV Reader output.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from csv_reader import Creator
from modules.models import ModuleValidationResult
from .interfaces import IModuleValidator


class CSVReaderValidator(IModuleValidator):
    @property
    def module_name(self) -> str:
        return "module1_csv_reader"

    def validate(self, video_id: str, artifact_path: Path) -> ModuleValidationResult:
        start_time = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat()
        invariants_checked = [
            "artifact_file_exists",
            "schema_conformance",
            "email_non_empty",
            "video_url_non_empty",
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
            if artifact_path.suffix.lower() == ".csv":
                from csv_reader import load_all_creators
                creators = load_all_creators(artifact_path)
                if not creators:
                    invariants_failed.append("schema_conformance")
                else:
                    c = creators[0]
                    if not c.email:
                        invariants_failed.append("email_non_empty")
                    if not c.video_url:
                        invariants_failed.append("video_url_non_empty")
            else:
                raw_data = json.loads(artifact_path.read_text(encoding="utf-8"))
                item_data = raw_data[0] if isinstance(raw_data, list) and raw_data else raw_data
                c = Creator(email=str(item_data.get("email", "")), video_url=str(item_data.get("video_url", "")))
                if not c.email:
                    invariants_failed.append("email_non_empty")
                if not c.video_url:
                    invariants_failed.append("video_url_non_empty")

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
