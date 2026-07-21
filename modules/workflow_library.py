"""Versioned, local ComfyUI workflow-template discovery and validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_MODULES_DIR = Path(__file__).resolve().parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from loguru import logger

from config import (
    LOG_DIR, MODULE7_LOG_PATH, MODULE7_NICHE_WORKFLOW_MAP,
    MODULE7_WORKFLOW_LIBRARY_DIR, MODULE7_WORKFLOW_VERSION,
)
from models import GenerationProfile, WorkflowTemplateRef
from module7_exceptions import WorkflowTemplateError

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}"


def _configure_logger() -> None:
    """Attach the Module 7 rotating Loguru sink."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(str(MODULE7_LOG_PATH), rotation="10 MB", retention="30 days",
               format=_LOG_FORMAT, level="DEBUG", enqueue=True)


_configure_logger()


class WorkflowLibrary:
    """Resolve only validated, version-controlled templates below one root."""

    def __init__(self, library_dir: Path = MODULE7_WORKFLOW_LIBRARY_DIR) -> None:
        self.library_dir = Path(library_dir).resolve()

    def discover(self) -> list[Path]:
        """Return template files in deterministic filename order."""
        if not self.library_dir.is_dir():
            raise WorkflowTemplateError(f"Workflow library directory does not exist: {self.library_dir}")
        return sorted(self.library_dir.glob("*.json"), key=lambda path: path.name)

    def load(self, template_path: Path) -> dict[str, Any]:
        """Read and validate a workflow template constrained to the library root."""
        path = Path(template_path).resolve()
        try:
            path.relative_to(self.library_dir)
        except ValueError as exc:
            raise WorkflowTemplateError(f"Workflow template escapes library directory: {path}") from exc
        try:
            template = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowTemplateError(f"Could not read workflow template {path}: {exc}") from exc
        self.validate(template, path)
        return template

    def validate(self, template: object, source: Path | None = None) -> None:
        """Validate the small, stable template contract required by Phase 1."""
        label = str(source) if source is not None else "workflow template"
        if not isinstance(template, dict):
            raise WorkflowTemplateError(f"{label} must be a JSON object")
        meta = template.get("_meta")
        graph = template.get("graph")
        if not isinstance(meta, dict) or not isinstance(graph, dict):
            raise WorkflowTemplateError(f"{label} must contain object-valued '_meta' and 'graph' keys")
        for field in ("name", "niche", "workflow_version"):
            if not isinstance(meta.get(field), str) or not meta[field].strip():
                raise WorkflowTemplateError(f"{label} has invalid _meta.{field}")
        if not graph:
            raise WorkflowTemplateError(f"{label} graph must contain at least one node")
        for node_id, node in graph.items():
            if not isinstance(node_id, str) or not isinstance(node, dict):
                raise WorkflowTemplateError(f"{label} contains an invalid graph node")
            if not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), dict):
                raise WorkflowTemplateError(f"{label} node '{node_id}' requires class_type and inputs")

    def resolve(self, niche: str, profile: GenerationProfile) -> WorkflowTemplateRef:
        """Resolve ``(niche, profile)`` deterministically, falling back to general."""
        normalized = niche.strip().lower() if niche else ""
        filename = MODULE7_NICHE_WORKFLOW_MAP.get(normalized, "general.json")
        path = self.library_dir / filename
        if filename != "general.json" and not path.exists():
            logger.warning("Workflow template missing for niche={niche}; falling back to general.json", niche=normalized)
            path = self.library_dir / "general.json"
        if filename == "general.json" and normalized not in MODULE7_NICHE_WORKFLOW_MAP:
            logger.info("Workflow fallback selected for unmapped niche={niche}", niche=normalized or "<empty>")
        template = self.load(path)
        meta = template["_meta"]
        ref = WorkflowTemplateRef(niche=normalized or "general", profile_name=profile.name,
                                  template_path=str(path), workflow_version=meta["workflow_version"],
                                  template_name=meta["name"])
        logger.info("Workflow resolved: niche={niche}, profile={profile}, template={template}",
                    niche=ref.niche, profile=profile.name, template=path.name)
        return ref


__all__ = ["WorkflowLibrary", "WorkflowTemplateRef", "WorkflowTemplateError"]
