"""
workflow_graph_assembler.py
============================

Materializes and merges declarative workflow fragments into a base workflow graph.
"""

from __future__ import annotations

import copy
from typing import Any, TYPE_CHECKING

from config import MODULE7_LOG_PATH
from generation_components.interfaces import IWorkflowGraphAssembler
from module7_exceptions import FragmentAttachmentError
from loguru import logger

if TYPE_CHECKING:
    from generation_components.conditioning_asset_resolver import GenerationConditioningContext
    from models import GenerationProfile


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


class WorkflowGraphAssembler(IWorkflowGraphAssembler):
    """Merges workflow graph fragments onto a host base graph in a deterministic order."""

    def assemble(
        self,
        base_graph: dict[str, Any],
        fragments: list[dict[str, Any]],
        conditioning: GenerationConditioningContext,
        profile: GenerationProfile,
    ) -> dict[str, Any]:
        """
        Merge base graph and fragments into a final ComfyUI node graph.

        Args:
            base_graph: Base workflow template graph dictionary (contains '_meta' and 'graph').
            fragments: List of loaded fragment definitions (dicts containing '_attach' and 'graph').
            conditioning: Resolved conditioning context.
            profile: Selected generation profile.

        Returns:
            Merged ComfyUI workflow graph dictionary.
        """
        if not fragments:
            return base_graph

        assembled = copy.deepcopy(base_graph)
        graph_nodes: dict[str, Any] = assembled.get("graph", {})
        meta: dict[str, Any] = assembled.get("_meta", {})
        attachment_points: dict[str, Any] = meta.get("attachment_points", {})

        for idx, fragment in enumerate(fragments):
            attach_info = fragment.get("_attach", {})
            point_name = attach_info.get("point")

            if not point_name or point_name not in attachment_points:
                raise FragmentAttachmentError(
                    f"Fragment attachment point '{point_name}' not defined in base template '_meta.attachment_points'."
                )

            target_node_id, input_key = attachment_points[point_name]

            if target_node_id not in graph_nodes:
                raise FragmentAttachmentError(
                    f"Target node ID '{target_node_id}' specified for attachment point '{point_name}' not in base graph."
                )

            target_inputs = graph_nodes[target_node_id].get("inputs", {})
            if input_key not in target_inputs:
                raise FragmentAttachmentError(
                    f"Target input key '{input_key}' not present on node '{target_node_id}' for point '{point_name}'."
                )

            current_input_ref = target_inputs[input_key]
            prefix = f"frag_{idx}_{point_name}_"
            frag_graph: dict[str, Any] = fragment.get("graph", {})

            for node_id, node_def in frag_graph.items():
                namespaced_id = f"{prefix}{node_id}"
                node_copy = copy.deepcopy(node_def)
                inputs_copy = node_copy.get("inputs", {})

                for k, v in inputs_copy.items():
                    if isinstance(v, list) and len(v) == 2:
                        ref_target, slot_idx = v[0], v[1]
                        if ref_target in ("ATTACHMENT_PREVIOUS", "BASE_ATTACHMENT"):
                            inputs_copy[k] = current_input_ref
                        elif str(ref_target) in frag_graph:
                            inputs_copy[k] = [f"{prefix}{ref_target}", slot_idx]

                node_copy["inputs"] = inputs_copy
                graph_nodes[namespaced_id] = node_copy

            out_node = attach_info.get("output_node")
            out_slot = attach_info.get("output_slot", 0)
            new_output_ref = [f"{prefix}{out_node}", out_slot]

            # Update the base graph node input to consume the fragment output
            graph_nodes[target_node_id]["inputs"][input_key] = new_output_ref

        assembled["graph"] = graph_nodes
        logger.info(
            "Assembled workflow graph with {n_fragments} fragment(s) attached; final node count={count}",
            n_fragments=len(fragments),
            count=len(graph_nodes),
        )
        return assembled
