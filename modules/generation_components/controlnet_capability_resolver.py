"""
controlnet_capability_resolver.py
===================================

Resolves semantic ControlNet capabilities to installed model filenames using a
deterministic, priority-ordered candidate pattern table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from config import CONTROLNET_CAPABILITY_TABLE, CapabilityCandidate, MODULE7_LOG_PATH
from generation_components.model_discovery_service import ModelDiscoveryService
from loguru import logger


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


@dataclass(frozen=True)
class ResolvedCapability:
    """Immutable result of resolving a semantic capability against installed models."""

    capability: str
    node_class: str
    filename_field: str
    resolved_filename: str | None
    resolution_source: Literal["legacy_exact_match", "pattern_match", "unresolved"]
    matched_pattern: str | None
    fragment_variant: str
    compatibility_decision: str | None = None


class ControlNetCapabilityResolver:
    """
    Resolves logical ControlNet capability requirements (depth, canny, segmentation)
    to concrete installed model filenames and fragment variants.
    """

    def __init__(
        self,
        discovery_service: ModelDiscoveryService | None = None,
        capability_table: dict[str, tuple[CapabilityCandidate, ...]] | None = None,
    ) -> None:
        self.discovery_service = discovery_service or ModelDiscoveryService()
        self.capability_table = capability_table if capability_table is not None else CONTROLNET_CAPABILITY_TABLE

    def resolve(self, capability: str) -> ResolvedCapability:
        """
        Resolve a logical capability name to a ResolvedCapability object.

        Args:
            capability: Capability name string (e.g. 'depth', 'canny', 'segmentation').

        Returns:
            ResolvedCapability containing resolved filename, node class, resolution source,
            and fragment variant.
        """
        norm_cap = capability.strip().lower()
        candidates = self.capability_table.get(norm_cap)

        if not candidates:
            logger.warning("Unknown ControlNet capability requested: {cap}", cap=capability)
            return ResolvedCapability(
                capability=norm_cap,
                node_class="ControlNetLoader",
                filename_field="control_net_name",
                resolved_filename=None,
                resolution_source="unresolved",
                matched_pattern=None,
                fragment_variant=f"controlnet_{norm_cap}",
                compatibility_decision=f"Capability '{capability}' is not configured in CONTROLNET_CAPABILITY_TABLE.",
            )

        # Priority-ordered search
        for candidate in candidates:
            installed = self.discovery_service.installed_models_for(
                candidate.node_class, candidate.filename_field
            )
            for fn in installed:
                if candidate.filename_regex.search(fn):
                    source: Literal["legacy_exact_match", "pattern_match"] = (
                        "legacy_exact_match" if candidate.pattern_name == "legacy_sdxl_official" else "pattern_match"
                    )
                    decision = (
                        f"Matched legacy exact match pattern '{candidate.pattern_name}' for capability '{norm_cap}': using model '{fn}'"
                        if source == "legacy_exact_match"
                        else f"Matched pattern '{candidate.pattern_name}' for capability '{norm_cap}': using model '{fn}'"
                    )
                    logger.info(
                        "Resolved ControlNet capability '{cap}' -> model '{fn}' via pattern '{pat}' ({source})",
                        cap=norm_cap,
                        fn=fn,
                        pat=candidate.pattern_name,
                        source=source,
                    )
                    return ResolvedCapability(
                        capability=norm_cap,
                        node_class=candidate.node_class,
                        filename_field=candidate.filename_field,
                        resolved_filename=fn,
                        resolution_source=source,
                        matched_pattern=candidate.pattern_name,
                        fragment_variant=candidate.fragment_variant,
                        compatibility_decision=decision,
                    )

        # Unresolved
        first = candidates[0]
        decision = f"No installed model matched any candidate pattern for capability '{norm_cap}'."
        logger.warning("ControlNet capability '{cap}' unresolved; no candidate matched.", cap=norm_cap)
        return ResolvedCapability(
            capability=norm_cap,
            node_class=first.node_class,
            filename_field=first.filename_field,
            resolved_filename=None,
            resolution_source="unresolved",
            matched_pattern=None,
            fragment_variant=first.fragment_variant,
            compatibility_decision=decision,
        )

    def describe_patterns(self, capabilities: Sequence[str]) -> str:
        """Return a formatted string describing all checked patterns for the given capabilities."""
        descriptions: list[str] = []
        for cap in capabilities:
            norm_cap = cap.strip().lower()
            candidates = self.capability_table.get(norm_cap, ())
            pats = [f"'{c.pattern_name}' ({c.filename_regex.pattern})" for c in candidates]
            descriptions.append(f"{norm_cap}: [{', '.join(pats)}]")
        return "; ".join(descriptions)
