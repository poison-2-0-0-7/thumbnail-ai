"""
test_precedence_resolver.py
============================

Unit tests for PrecedenceResolver in planner_components.
"""

from pathlib import Path
import pytest

from models import (
    AssetPlacement,
    CanvasTransform,
    CompositionLayer,
    CompositionWorkspace,
    DecisionAction,
    DecisionManifest,
    DecisionSource,
    LayerDecision,
    LayerRole,
    LayerTransform,
    LightingAdjustment,
    PlacementConstraints,
    ResolvedDecision,
    TargetElement,
    TextPlacement,
    WorkspaceMetadata,
    WorkspaceStatistics,
)
from planner_components.precedence_resolver import PrecedenceResolver


@pytest.fixture
def minimal_workspace():
    valid_hash = "a" * 64
    l_bg = CompositionLayer(
        layer_id="l_bg",
        placement=AssetPlacement(
            asset_id="bg",
            role=LayerRole.BACKGROUND,
            decision=LayerDecision.REPLACE,
            source_path="bg.png",
            transform=LayerTransform(),
            z_index=0,
        ),
    )
    l_person = CompositionLayer(
        layer_id="l_person",
        placement=AssetPlacement(
            asset_id="person",
            role=LayerRole.PERSON,
            decision=LayerDecision.KEEP,
            source_path="person.png",
            transform=LayerTransform(),
            z_index=10,
        ),
    )
    return CompositionWorkspace(
        video_id="vid1",
        canvas=CanvasTransform(width=1280, height=720, aspect_ratio="16:9"),
        layers=[l_bg, l_person],
        groups=[],
        text_placement=TextPlacement(),
        lighting=LightingAdjustment(
            target_brightness=0.5, target_contrast=0.5, target_saturation=0.5, warm_or_cool="neutral"
        ),
        constraints=PlacementConstraints(),
        statistics=WorkspaceStatistics(total_layers=2, replaced=1, kept=1),
        metadata=WorkspaceMetadata(
            video_id="vid1",
            created_at="2026-08-01T00:00:00Z",
            vre_source_hash=valid_hash,
            redesign_spec_hash=valid_hash,
            prompt_package_hash=valid_hash,
            engine_version="1.0.0",
        ),
    )


def test_precedence_resolver_fallback_to_workspace(minimal_workspace):
    resolver = PrecedenceResolver()
    decisions = resolver.resolve_layer_decisions(minimal_workspace, decision_manifest=None)
    assert len(decisions) == 2
    assert decisions[0][0] == "bg"
    assert decisions[0][2] == "replace"
    assert decisions[1][0] == "person"
    assert decisions[1][2] == "keep"


def test_precedence_resolver_uses_decision_manifest(minimal_workspace):
    valid_hash = "b" * 64
    manifest = DecisionManifest(
        video_id="vid1",
        source_generated_image_path="thumb.jpg",
        source_generated_image_hash=valid_hash,
        decisions=[
            ResolvedDecision(
                decision_id="d1",
                target=TargetElement(
                    element_id="person",
                    element_type="person",
                    label="creator_face",
                ),
                action=DecisionAction.ENHANCE,
                confidence=0.95,
                source=DecisionSource.RULE_LLM_AGREEMENT,
                rationale="Enhance person appearance",
                priority_rank=1,
            )
        ],
        reasoning_trace=[],
        conflicts_resolved=0,
        status="success",
        engine_version="1.0.0",
        created_at="2026-08-01T00:00:00Z",
        decided_at="2026-08-01T00:00:00Z",
        redesign_spec_hash=valid_hash,
        prompt_package_hash=valid_hash,
        intelligence_hash=valid_hash,
    )
    resolver = PrecedenceResolver()
    decisions = resolver.resolve_layer_decisions(minimal_workspace, decision_manifest=manifest)
    assert len(decisions) == 1
    assert decisions[0][0] == "person"
    assert decisions[0][2] == "enhance"
    assert decisions[0][3] == "Enhance person appearance"
