"""
Unit tests for Knowledge Base validation engine.
Tests ModelValidator, EvidenceValidator (the grounding gate), ConstraintValidator,
and SchemaIntegrityValidator.
"""

from __future__ import annotations

import pytest

from thumbnail_intelligence.knowledge_base.exceptions import (
    ConstraintValidationError,
    EvidenceValidationError,
    IntegrityValidationError,
    SchemaValidationError,
)
from thumbnail_intelligence.knowledge_base.models import (
    BrandConstraint,
    DesignReason,
    DesignReasonType,
    EvidenceReference,
    EvidenceSourceType,
    IdentityConstraint,
)
from thumbnail_intelligence.knowledge_base.validation import (
    ConstraintValidator,
    EvidenceValidator,
    ModelValidator,
    SchemaIntegrityValidator,
)


def test_model_validator_embedding_checks() -> None:
    # Valid 512-dim embedding
    valid_emb = [0.1] * 512
    ModelValidator.validate_embedding(valid_emb, expected_dim=512)

    # Empty embedding allowed
    ModelValidator.validate_embedding([], expected_dim=512)

    # Wrong dimension
    with pytest.raises(SchemaValidationError):
        ModelValidator.validate_embedding([0.1] * 256, expected_dim=512)

    # NaN in embedding
    with pytest.raises(SchemaValidationError):
        ModelValidator.validate_embedding([float("nan")] * 512, expected_dim=512)

    # Non-numeric item
    with pytest.raises(SchemaValidationError):
        ModelValidator.validate_embedding(["str"] * 512, expected_dim=512)  # type: ignore


def test_evidence_validator_grounding_gate() -> None:
    ref = EvidenceReference(
        source_type=EvidenceSourceType.SCENE_GRAPH_ELEMENT,
        source_id="creator_0",
        excerpt_or_value="Face prominent in center",
    )
    EvidenceValidator.validate_evidence_reference(ref)

    reason = DesignReason(
        reason_id="reason_01",
        claim="Preserve creator face prominence",
        reason_type=DesignReasonType.BRAND_CONSISTENCY,
        evidence=[ref],
    )
    EvidenceValidator.validate_design_reason(reason)
    EvidenceValidator.validate_design_reasons([reason])


def test_constraint_validator_detects_contradictions() -> None:
    # Contradictory brand constraint: same element in both mandatory and prohibited
    ref = EvidenceReference(
        source_type=EvidenceSourceType.BRAND_RULE,
        source_id="rule_01",
    )
    contradictory_brand = BrandConstraint(
        constraint_id="bc_bad",
        channel_id="channel_123",
        prohibited_elements=["channel_logo", "red_arrow"],
        mandatory_elements=["channel_logo"],
        evidence_refs=[ref],
    )
    with pytest.raises(ConstraintValidationError):
        ConstraintValidator.validate_brand_constraint(contradictory_brand)

    # Valid brand constraint
    valid_brand = BrandConstraint(
        constraint_id="bc_good",
        channel_id="channel_123",
        prohibited_elements=["corporate_stock_icons"],
        mandatory_elements=["channel_logo"],
        evidence_refs=[ref],
    )
    ConstraintValidator.validate_brand_constraint(valid_brand)


def test_identity_constraint_validation() -> None:
    ref = EvidenceReference(
        source_type=EvidenceSourceType.BRAND_RULE,
        source_id="rule_01",
    )
    valid_id = IdentityConstraint(
        constraint_id="ic_good",
        creator_id="creator_001",
        face_similarity_threshold=0.92,
        evidence_refs=[ref],
    )
    ConstraintValidator.validate_identity_constraint(valid_id)


def test_schema_integrity_validator_missing_fields() -> None:
    raw_data = {"key1": "val1", "key2": "val2"}
    SchemaIntegrityValidator.validate_raw_dict(raw_data, ["key1", "key2"])

    with pytest.raises(IntegrityValidationError):
        SchemaIntegrityValidator.validate_raw_dict(raw_data, ["key1", "missing_key"])
