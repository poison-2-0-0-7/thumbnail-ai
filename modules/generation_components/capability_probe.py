"""
capability_probe.py
====================

Probes live ComfyUI server via /object_info to verify installed node types,
detect node requirements across workflow templates/fragments, generate clear
missing custom node reports, and prevent execution of invalid workflows.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, ClassVar

from config import (
    MODULE7_CAPABILITY_PROBE_CACHE_SECONDS,
    MODULE7_CAPABILITY_PROBE_ENABLED,
    MODULE7_LOG_PATH,
    MODULE7_WORKFLOW_LIBRARY_DIR,
)
from generation_components.interfaces import ICapabilityProbe
from module7_exceptions import MissingCustomNodeError, UnsupportedNodeTypeWarning
from loguru import logger


RECOMMENDED_CUSTOM_NODE_PACKAGES: dict[str, str] = {
    "ControlNetApply": "ComfyUI-ControlNet-Aux",
    "ControlNetApplyAdvanced": "ComfyUI-ControlNet-Aux",
    "ControlNetLoader": "ComfyUI-ControlNet-Aux",
    "ControlNetLoaderAdvanced": "ComfyUI-ControlNet-Aux",
    "DWPreprocessor": "comfyui_controlnet_aux",
    "IPAdapterApply": "ComfyUI_IPAdapter_plus",
    "IPAdapterModelLoader": "ComfyUI_IPAdapter_plus",
    "IPAdapterUnifiedLoader": "ComfyUI_IPAdapter_plus",
    "IPAdapterAdvanced": "ComfyUI_IPAdapter_plus",
    "IPAdapterTiled": "ComfyUI_IPAdapter_plus",
    "ReActorFaceSwap": "comfyui-reactor_node",
    "ReActorOptions": "comfyui-reactor_node",
    "ReActorLoadFace": "comfyui-reactor_node",
    "InsightFaceLoader": "comfyui-reactor_node",
    "CodeFormerLoader": "comfyui-reactor_node",
    "UltimateSDUpscale": "UltimateSDUpscale",
    "UltimateSDUpscaleNoUpscale": "UltimateSDUpscale",
    "ImpactPack": "ComfyUI-Impact-Pack",
    "SAMLoader": "ComfyUI-Impact-Pack",
    "SAMDetectorCombined": "ComfyUI-Impact-Pack",
    "FaceDetailer": "ComfyUI-Impact-Pack",
    "CR Image Output": "ComfyUI_Comfyroll_CustomNodes",
    "CR Text Output": "ComfyUI_Comfyroll_CustomNodes",
    "AnimateDiffLoaderWithContext": "ComfyUI-AnimateDiff-Evolved",
    "UnetLoaderGGUF": "ComfyUI-GGUF",
    "DualCLIPLoaderGGUF": "ComfyUI-GGUF",
}


def detect_workflow_node_types(
    workflows_dir: Path | str = MODULE7_WORKFLOW_LIBRARY_DIR,
) -> dict[str, set[str]]:
    """
    Detect all node types required by every workflow JSON file in workflows_dir.

    Returns:
        Dictionary mapping relative workflow path string (e.g. 'workflows/general.json')
        to set of required node class_types.
    """
    root = Path(workflows_dir).resolve()
    if not root.is_dir():
        return {}

    workflow_nodes: dict[str, set[str]] = {}
    json_files = sorted(root.rglob("*.json"))

    for json_file in json_files:
        try:
            rel_path = json_file.relative_to(root).as_posix()
        except ValueError:
            rel_path = json_file.name

        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            graph = data.get("graph", {})
            meta = data.get("_meta", {})
            required_meta = meta.get("required_node_types", [])

            graph_types = set()
            if isinstance(graph, dict):
                for node in graph.values():
                    if isinstance(node, dict) and isinstance(node.get("class_type"), str):
                        graph_types.add(node["class_type"])

            all_types = graph_types.union(required_meta)
            if all_types:
                workflow_nodes[rel_path] = all_types
        except (OSError, json.JSONDecodeError):
            continue

    return workflow_nodes


def format_missing_nodes_report(missing_by_workflow: dict[str, list[dict[str, str]]]) -> str:
    """Format a clear human-readable report listing missing custom node types."""
    lines = [
        "==================================================================",
        "           MISSING COMFYUI CUSTOM NODES REPORT                    ",
        "==================================================================",
        "The running ComfyUI instance is missing custom node(s) required by",
        "workflow(s):",
        "",
    ]
    for wf_name, items in missing_by_workflow.items():
        lines.append(f"Workflow: {wf_name}")
        for item in items:
            lines.append(f"  - Missing Node Type:     {item['missing_node_type']}")
            lines.append(f"    Workflow:              {item['workflow']}")
            lines.append(f"    Recommended Package:   {item['recommended_package']}")
        lines.append("")

    lines.append("==================================================================")
    lines.append("Submission of workflow(s) prevented due to missing custom nodes.")
    lines.append("==================================================================")
    return "\n".join(lines)


def _configure_logger() -> None:
    """Ensure Loguru sink is configured for Module 7."""
    try:
        logger.add(
            MODULE7_LOG_PATH,
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            enqueue=True,
        )
    except ValueError:
        pass


_configure_logger()


class CapabilityProbe(ICapabilityProbe):
    """Probes installed ComfyUI node types and caches results per pipeline run."""

    RECOMMENDED_PACKAGES: ClassVar[dict[str, str]] = RECOMMENDED_CUSTOM_NODE_PACKAGES

    def __init__(
        self,
        client: Any | None = None,
        enabled: bool = MODULE7_CAPABILITY_PROBE_ENABLED,
        cache_ttl_seconds: float = MODULE7_CAPABILITY_PROBE_CACHE_SECONDS,
    ) -> None:
        self.client = client
        self.enabled = enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cached_types: frozenset[str] | None = None
        self._last_probe_time: float = 0.0

    @classmethod
    def get_recommended_package(cls, node_type: str) -> str:
        return cls.RECOMMENDED_PACKAGES.get(node_type, f"ComfyUI-Manager search: '{node_type}'")

    def installed_node_types(self) -> frozenset[str]:
        """
        Return set of installed node class_types.

        Returns:
            frozenset of node class_type strings, or empty set if disabled/unavailable.
        """
        if not self.enabled or self.client is None:
            return frozenset()

        now = time.monotonic()
        if self._cached_types is not None and (now - self._last_probe_time) < self.cache_ttl_seconds:
            return self._cached_types

        try:
            info = self.client.object_info()
            if isinstance(info, dict):
                self._cached_types = frozenset(info.keys())
                self._last_probe_time = now
                logger.info(
                    "CapabilityProbe cached {count} installed ComfyUI node types",
                    count=len(self._cached_types),
                )
                return self._cached_types
        except Exception as exc:
            logger.warning("CapabilityProbe failed to fetch /object_info: {exc}", exc=exc)

        self._cached_types = frozenset()
        self._last_probe_time = now
        return self._cached_types

    def is_fragment_supported(self, fragment: dict[str, Any]) -> bool:
        """
        Check if fragment's required node types are installed on the server.

        Args:
            fragment: Fragment definition dictionary.

        Returns:
            True if all required nodes are available or probing disabled, False otherwise.
        """
        if not self.enabled or self.client is None:
            return True

        installed = self.installed_node_types()
        if not installed:
            # If probe failed or returned empty, fail soft and allow fragment
            return True

        meta = fragment.get("_meta", {})
        required = meta.get("required_node_types", [])

        # Also extract class_types directly from the fragment graph
        graph_nodes = fragment.get("graph", {})
        graph_class_types = [
            node.get("class_type")
            for node in graph_nodes.values()
            if isinstance(node, dict) and node.get("class_type")
        ]

        all_required = set(required).union(graph_class_types)
        missing = [nt for nt in all_required if nt not in installed]

        if missing:
            for node_type in missing:
                msg = f"Fragment dropped: required node type {node_type} not available in ComfyUI /object_info"
                logger.warning(msg)
                warnings.warn(msg, UnsupportedNodeTypeWarning)
            return False

        return True

    def validate_all_workflows(
        self, workflows_dir: Path | str = MODULE7_WORKFLOW_LIBRARY_DIR
    ) -> dict[str, list[dict[str, str]]]:
        """
        Validate all workflows in workflows_dir against installed node types on ComfyUI instance.

        Returns:
            Dictionary mapping workflow path -> list of missing node dicts:
            [{ 'missing_node_type': ..., 'workflow': ..., 'recommended_package': ... }]
        """
        installed = self.installed_node_types()
        if not installed:
            return {}

        workflow_requirements = detect_workflow_node_types(workflows_dir)
        missing_by_workflow: dict[str, list[dict[str, str]]] = {}

        for wf_name, req_nodes in workflow_requirements.items():
            missing = [nt for nt in req_nodes if nt not in installed]
            if missing:
                missing_by_workflow[wf_name] = [
                    {
                        "missing_node_type": nt,
                        "workflow": wf_name,
                        "recommended_package": self.get_recommended_package(nt),
                    }
                    for nt in missing
                ]

        return missing_by_workflow

    def validate_workflow_graph(
        self,
        graph: dict[str, Any],
        workflow_name: str = "workflow",
        raise_on_missing: bool = True,
    ) -> list[dict[str, str]]:
        """
        Validate a built workflow graph against installed node types.

        If missing nodes are found and raise_on_missing is True, raises MissingCustomNodeError
        to prevent submission of that workflow.
        """
        if not self.enabled or self.client is None:
            return []

        installed = self.installed_node_types()
        if not installed:
            return []

        required_types = set()
        if isinstance(graph, dict):
            for node in graph.values():
                if isinstance(node, dict) and isinstance(node.get("class_type"), str):
                    required_types.add(node["class_type"])

        missing = [nt for nt in required_types if nt not in installed]
        if not missing:
            return []

        missing_items = [
            {
                "missing_node_type": nt,
                "workflow": workflow_name,
                "recommended_package": self.get_recommended_package(nt),
            }
            for nt in missing
        ]

        report = format_missing_nodes_report({workflow_name: missing_items})
        logger.error("Preventing workflow submission due to missing custom nodes:\n{report}", report=report)

        if raise_on_missing:
            raise MissingCustomNodeError(
                f"Workflow '{workflow_name}' cannot be submitted because required custom nodes are missing from ComfyUI.",
                missing_nodes_report=report,
            )

        return missing_items

