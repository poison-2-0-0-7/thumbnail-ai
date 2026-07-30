"""
layer_manager.py
================

Manages layer z-ordering and role-based grouping.
"""

from __future__ import annotations

from composition_components.interfaces import ILayerManager
from models import CompositionLayer, LayerGroup, LayerRole

ROLE_Z_INDEX_MAP: dict[LayerRole, int] = {
    LayerRole.BACKGROUND: 0,
    LayerRole.FOREGROUND: 10,
    LayerRole.PERSON: 10,
    LayerRole.OBJECT: 20,
    LayerRole.TEXT: 30,
    LayerRole.EFFECT: 40,
}


class LayerManager(ILayerManager):
    """Manager for deterministic layer ordering and grouping."""

    @staticmethod
    def get_role_z_index(role: LayerRole) -> int:
        """Return base z-index for a LayerRole."""
        return ROLE_Z_INDEX_MAP.get(role, 0)

    def order(self, layers: list[CompositionLayer]) -> list[CompositionLayer]:
        """
        Sort layers deterministically: primarily by placement.z_index,
        secondarily by layer_id.
        """
        return sorted(
            layers, key=lambda layer: (layer.placement.z_index, layer.layer_id)
        )

    def group(self, layers: list[CompositionLayer]) -> list[LayerGroup]:
        """Group layer_ids by role."""
        grouped: dict[LayerRole, list[str]] = {}
        for layer in layers:
            role = layer.placement.role
            grouped.setdefault(role, []).append(layer.layer_id)

        groups: list[LayerGroup] = []
        for role, layer_ids in grouped.items():
            groups.append(
                LayerGroup(
                    group_id=f"group_{role.value}",
                    role=role,
                    layer_ids=sorted(layer_ids),
                )
            )

        # Order groups deterministically by role z-index
        return sorted(
            groups, key=lambda grp: (ROLE_Z_INDEX_MAP.get(grp.role, 0), grp.group_id)
        )
