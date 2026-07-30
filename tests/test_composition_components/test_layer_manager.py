"""
test_layer_manager.py
=====================

Unit tests for LayerManager component in Module 10 Asset Composer.
"""

from __future__ import annotations

from composition_components.layer_manager import LayerManager
from models import (
    AssetPlacement,
    CompositionLayer,
    LayerDecision,
    LayerRole,
    LayerTransform,
)


def test_layer_manager_ordering_and_grouping():
    manager = LayerManager()

    l_bg = CompositionLayer(
        layer_id="layer_bg",
        placement=AssetPlacement(
            asset_id="bg",
            role=LayerRole.BACKGROUND,
            decision=LayerDecision.REPLACE,
            transform=LayerTransform(),
            z_index=0,
        ),
    )
    l_person = CompositionLayer(
        layer_id="layer_person",
        placement=AssetPlacement(
            asset_id="face",
            role=LayerRole.PERSON,
            decision=LayerDecision.KEEP,
            transform=LayerTransform(),
            z_index=10,
        ),
    )
    l_obj = CompositionLayer(
        layer_id="layer_car",
        placement=AssetPlacement(
            asset_id="car",
            role=LayerRole.OBJECT,
            decision=LayerDecision.KEEP,
            transform=LayerTransform(),
            z_index=20,
        ),
    )
    l_text = CompositionLayer(
        layer_id="layer_text",
        placement=AssetPlacement(
            asset_id="text",
            role=LayerRole.TEXT,
            decision=LayerDecision.ADD,
            transform=LayerTransform(),
            z_index=30,
        ),
    )

    # Shuffled input
    layers = [l_text, l_obj, l_bg, l_person]
    ordered = manager.order(layers)

    assert [l.layer_id for l in ordered] == [
        "layer_bg",
        "layer_person",
        "layer_car",
        "layer_text",
    ]

    # Grouping
    groups = manager.group(ordered)
    assert len(groups) == 4
    bg_group = next(g for g in groups if g.role == LayerRole.BACKGROUND)
    assert bg_group.layer_ids == ["layer_bg"]

    person_group = next(g for g in groups if g.role == LayerRole.PERSON)
    assert person_group.layer_ids == ["layer_person"]
