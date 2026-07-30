"""
metadata_builder.py
===================

Assembles WorkspaceMetadata and WorkspaceStatistics for Module 10 Asset Composer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any

from composition_components.interfaces import IMetadataBuilder
from config import COMPOSITION_ENGINE_VERSION
from models import (
    CompositionLayer,
    LayerDecision,
    PromptPackage,
    RedesignSpecification,
    VisualReferenceManifest,
    WorkspaceMetadata,
    WorkspaceStatistics,
)


def compute_model_hash(model: Any) -> str:
    """Compute deterministic SHA-256 hash of a Pydantic model's JSON representation."""
    json_bytes = model.model_dump_json(exclude_none=True).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()


class MetadataBuilder(IMetadataBuilder):
    """Builder for assembling provenance metadata and statistical metrics."""

    def build(
        self,
        video_id: str,
        manifest: VisualReferenceManifest,
        spec: RedesignSpecification,
        package: PromptPackage,
    ) -> WorkspaceMetadata:
        """Build WorkspaceMetadata from upstream artifact states."""
        created_at = datetime.now(timezone.utc).isoformat()
        spec_hash = compute_model_hash(spec)
        package_hash = compute_model_hash(package)

        return WorkspaceMetadata(
            video_id=video_id,
            created_at=created_at,
            vre_source_hash=manifest.source_hash,
            redesign_spec_hash=spec_hash,
            prompt_package_hash=package_hash,
            engine_version=COMPOSITION_ENGINE_VERSION,
        )

    def statistics(self, layers: list[CompositionLayer]) -> WorkspaceStatistics:
        """Compute statistics counts across a list of CompositionLayers."""
        counts = {
            LayerDecision.KEEP: 0,
            LayerDecision.REMOVE: 0,
            LayerDecision.REPLACE: 0,
            LayerDecision.ENHANCE: 0,
            LayerDecision.ADD: 0,
        }

        for layer in layers:
            counts[layer.placement.decision] += 1

        return WorkspaceStatistics(
            total_layers=len(layers),
            kept=counts[LayerDecision.KEEP],
            removed=counts[LayerDecision.REMOVE],
            replaced=counts[LayerDecision.REPLACE],
            enhanced=counts[LayerDecision.ENHANCE],
            added=counts[LayerDecision.ADD],
        )
