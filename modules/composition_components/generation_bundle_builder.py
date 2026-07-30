"""
generation_bundle_builder.py
=============================

Flattens a CompositionWorkspace into a ComfyUI-ready GenerationBundle.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Optional

from composition_components.interfaces import IGenerationBundleBuilder
from composition_exceptions import GenerationBundleError
from models import CompositionWorkspace, GenerationBundle, LayerDecision


class GenerationBundleBuilder(IGenerationBundleBuilder):
    """Flattens a validated CompositionWorkspace into a GenerationBundle artifact."""

    def build_generation_bundle(
        self, workspace: CompositionWorkspace, prompt_package_hash: str = ""
    ) -> GenerationBundle:
        """
        Flatten CompositionWorkspace into GenerationBundle.

        Args:
            workspace: Validated CompositionWorkspace.
            prompt_package_hash: Optional explicit prompt_package_hash override.

        Returns:
            GenerationBundle artifact.
        """
        if workspace.status == "error":
            raise GenerationBundleError(
                f"Cannot build GenerationBundle from failed workspace '{workspace.video_id}': {workspace.error_message}"
            )

        ref_paths: dict[str, str] = {}
        mask_paths: dict[str, str] = {}
        depth_path: Optional[str] = None
        canny_path: Optional[str] = None
        layer_order: list[str] = []

        for layer in workspace.layers:
            layer_order.append(layer.layer_id)

            # Exclude REMOVE layers from reference_image_paths per §11
            if layer.placement.decision != LayerDecision.REMOVE:
                role_key = layer.placement.role.value
                if layer.placement.source_path:
                    ref_paths[role_key] = layer.placement.source_path

                if layer.placement.mask:
                    mask_paths[role_key] = layer.placement.mask.mask_path

            if layer.depth_hint_path and depth_path is None:
                depth_path = layer.depth_hint_path

        # Compute workspace hash
        ws_json = workspace.model_dump_json(exclude_none=True).encode("utf-8")
        ws_hash = hashlib.sha256(ws_json).hexdigest()

        pkg_hash = (
            prompt_package_hash
            if prompt_package_hash
            else workspace.metadata.prompt_package_hash
        )

        return GenerationBundle(
            video_id=workspace.video_id,
            canvas=workspace.canvas,
            reference_image_paths=ref_paths,
            mask_paths=mask_paths,
            depth_path=depth_path,
            canny_path=canny_path,
            layer_order=layer_order,
            workspace_hash=ws_hash,
            prompt_package_hash=pkg_hash,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
