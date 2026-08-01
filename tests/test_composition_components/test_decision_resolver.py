"""
test_decision_resolver.py
=========================

Unit tests for DecisionResolver component in Module 10 Asset Composer.
"""

from __future__ import annotations

import pytest

from composition_components.decision_resolver import DecisionResolver
from models import (
    BoundingBox,
    ColorDirection,
    LayoutDirection,
    LayerDecision,
    LayerRole,
    ObjectDirective,
    RedesignSpecification,
    SubjectTreatment,
    TextOverlaySpec,
)


@pytest.fixture
def base_redesign_spec():
    return RedesignSpecification(
        video_id="test_vid",
        source_thumbnail_path="data/thumbnails/test_vid.jpg",
        color_direction=ColorDirection(),
        subject_treatment=SubjectTreatment(
            has_subject=True,
            target_position_label="center",
            crop_tighter=False,
        ),
        text_overlay=TextOverlaySpec(include_text=True),
        layout_direction=LayoutDirection(),
        object_directives=[
            ObjectDirective(label="car", action="preserve"),
            ObjectDirective(label="clutter_logo", action="remove"),
            ObjectDirective(label="trophy", action="include"),
        ],
        source_ctr_potential_score=0.7,
        source_curiosity_gap_score=0.6,
        source_content_mismatch_detected=False,
        generated_at="2026-07-30T00:00:00Z",
    )


def test_decision_resolver_all_decisions(base_redesign_spec):
    resolver = DecisionResolver()
    decisions = resolver.resolve(base_redesign_spec)

    # 1. Background -> REPLACE
    bg_entry = next(d for d in decisions if d[0] == "background")
    assert bg_entry[1] == LayerRole.BACKGROUND
    assert bg_entry[2] == LayerDecision.REPLACE

    # 2. Person -> KEEP (since crop_tighter=False)
    person_entry = next(d for d in decisions if d[0] == "person")
    assert person_entry[1] == LayerRole.PERSON
    assert person_entry[2] == LayerDecision.KEEP

    # 3. Objects -> car (KEEP), clutter_logo (REMOVE), trophy (KEEP)
    car_entry = next(d for d in decisions if d[0] == "object_0_car")
    assert car_entry[2] == LayerDecision.KEEP

    logo_entry = next(d for d in decisions if d[0] == "object_1_clutter_logo")
    assert logo_entry[2] == LayerDecision.REMOVE

    trophy_entry = next(d for d in decisions if d[0] == "object_2_trophy")
    assert trophy_entry[2] == LayerDecision.KEEP

    # 4. Text -> ADD (since include_text=True)
    text_entry = next(d for d in decisions if d[0] == "text")
    assert text_entry[1] == LayerRole.TEXT
    assert text_entry[2] == LayerDecision.ADD


def test_decision_resolver_enhance_subject(base_redesign_spec):
    # Test crop_tighter=True -> ENHANCE
    spec = base_redesign_spec.model_copy(
        update={
            "subject_treatment": SubjectTreatment(
                has_subject=True,
                crop_tighter=True,
            )
        }
    )
    resolver = DecisionResolver()
    decisions = resolver.resolve(spec)
    person_entry = next(d for d in decisions if d[0] == "person")
    assert person_entry[2] == LayerDecision.ENHANCE


def test_decision_resolver_no_subject_no_text(base_redesign_spec):
    # Test has_subject=False and include_text=False
    spec = base_redesign_spec.model_copy(
        update={
            "subject_treatment": SubjectTreatment(has_subject=False),
            "text_overlay": TextOverlaySpec(include_text=False),
            "object_directives": [],
        }
    )
    resolver = DecisionResolver()
    decisions = resolver.resolve(spec)

    keys = [d[0] for d in decisions]
    assert "background" in keys
    assert "person" not in keys
    assert "text" not in keys
    assert len(decisions) == 1


def test_decision_resolver_uses_decision_manifest_precedence(base_redesign_spec):
    from models import (
        DecisionAction,
        DecisionManifest,
        DecisionManifestStatus,
        DecisionSource,
        ResolvedDecision,
        TargetElement,
    )

    manifest = DecisionManifest(
        video_id="test_vid",
        source_generated_image_path="test.jpg",
        source_generated_image_hash="a" * 64,
        status=DecisionManifestStatus.SUCCESS,
        decided_at="2026-08-01T00:00:00Z",
        decisions=[
            ResolvedDecision(
                decision_id="d1",
                target=TargetElement(element_id="object_0_car", element_type="object", label="car"),
                action=DecisionAction.REPLACE,
                confidence=0.9,
                source=DecisionSource.RULE,
                rationale="Replace car with futuristic bike",
                priority_rank=1,
            )
        ],
    )

    resolver = DecisionResolver()
    decisions = resolver.resolve(base_redesign_spec, decision_manifest=manifest)

    assert len(decisions) == 1
    assert decisions[0][0] == "object_0_car"
    assert decisions[0][1] == LayerRole.OBJECT
    assert decisions[0][2] == LayerDecision.REPLACE
    assert decisions[0][3] == "Replace car with futuristic bike"
