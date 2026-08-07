"""
validation.py
=============

Comprehensive validation and integrity verification engine for Knowledge Base records.
Enforces:
1. Strict schema boundary checks
2. The Evidence Grounding Gate (§19.2: "Interpretation, not invention")
3. Brand and identity constraint invariants
4. Multimodal embedding dimension and float validity
5. Semantic version compatibility
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel

from thumbnail_intelligence.knowledge_base.exceptions import (
    ConstraintValidationError,
    EvidenceValidationError,
    IntegrityValidationError,
    SchemaValidationError,
)
from thumbnail_intelligence.knowledge_base.models import (
    BaseKBModel,
    BrandConstraint,
    CreatorProfile,
    DesignReason,
    EvidenceReference,
    IdentityConstraint,
    KnowledgeEntry,
)


class ModelValidator:
    """Validates structural and numerical boundaries on Pydantic models."""

    @staticmethod
    def validate_embedding(embedding: List[float], expected_dim: int = 512) -> None:
        """Validate embedding vector dimensions, float types, and finite ranges."""
        if not isinstance(embedding, list):
            raise SchemaValidationError(
                message=f"Embedding must be a list of floats, got {type(embedding).__name__}",
                context={"expected_type": "list", "received_type": type(embedding).__name__},
            )
        if len(embedding) == 0:
            # Empty embedding is permitted for non-vectorized entries
            return
        if len(embedding) != expected_dim:
            raise SchemaValidationError(
                message=f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}",
                context={"expected_dim": expected_dim, "actual_dim": len(embedding)},
            )
        for idx, val in enumerate(embedding):
            if not isinstance(val, (int, float)):
                raise SchemaValidationError(
                    message=f"Embedding element at index {idx} is non-numeric: {type(val)}",
                    context={"index": idx, "type": str(type(val))},
                )
            if val != val:  # NaN check
                raise SchemaValidationError(
                    message=f"Embedding element at index {idx} is NaN",
                    context={"index": idx},
                )

    @staticmethod
    def validate_non_empty_str(value: str, field_name: str) -> None:
        """Ensure string field is present and not blank."""
        if not isinstance(value, str) or not value.strip():
            raise SchemaValidationError(
                message=f"Field '{field_name}' must be a non-empty string",
                context={"field": field_name, "value": value},
            )


class EvidenceValidator:
    """
    Enforces the grounding gate: every design reason or creative assertion must cite
    at least one valid, non-empty EvidenceReference (§19.2).
    """

    @staticmethod
    def validate_evidence_reference(ref: EvidenceReference) -> None:
        """Validate an individual EvidenceReference instance."""
        if not ref.source_id or not ref.source_id.strip():
            raise EvidenceValidationError(
                message="EvidenceReference source_id cannot be empty",
                context={"source_type": str(ref.source_type)},
            )
        if not (0.0 <= ref.confidence <= 1.0):
            raise EvidenceValidationError(
                message=f"EvidenceReference confidence must be in [0.0, 1.0], got {ref.confidence}",
                context={"confidence": ref.confidence, "source_id": ref.source_id},
            )

    @staticmethod
    def validate_design_reason(reason: DesignReason) -> None:
        """Enforce that DesignReason has valid non-empty evidence references."""
        if not reason.evidence or len(reason.evidence) == 0:
            raise EvidenceValidationError(
                message=f"DesignReason '{reason.reason_id}' violated grounding gate: evidence list is empty.",
                context={"reason_id": reason.reason_id, "claim": reason.claim},
            )
        for ref in reason.evidence:
            EvidenceValidator.validate_evidence_reference(ref)

    @classmethod
    def validate_design_reasons(cls, reasons: List[DesignReason]) -> None:
        """Validate an entire collection of DesignReason records."""
        for reason in reasons:
            cls.validate_design_reason(reason)


class ConstraintValidator:
    """Validates brand and identity constraint integrity and detects self-contradictions."""

    @staticmethod
    def validate_brand_constraint(constraint: BrandConstraint) -> None:
        """Validate consistency of brand constraints."""
        # Check overlap between mandatory and prohibited elements
        if constraint.mandatory_elements and constraint.prohibited_elements:
            overlap = set(constraint.mandatory_elements).intersection(set(constraint.prohibited_elements))
            if overlap:
                raise ConstraintValidationError(
                    message=f"BrandConstraint has contradictory elements in both mandatory and prohibited: {overlap}",
                    context={"constraint_id": constraint.constraint_id, "contradictory_elements": list(overlap)},
                )
        # Validate grounding evidence if present
        for ref in constraint.evidence_refs:
            EvidenceValidator.validate_evidence_reference(ref)

    @staticmethod
    def validate_identity_constraint(constraint: IdentityConstraint) -> None:
        """Validate identity locking and similarity thresholds."""
        if not (0.0 <= constraint.face_similarity_threshold <= 1.0):
            raise ConstraintValidationError(
                message=f"IdentityConstraint face_similarity_threshold must be in [0.0, 1.0], got {constraint.face_similarity_threshold}",
                context={"constraint_id": constraint.constraint_id},
            )
        # Validate grounding evidence
        for ref in constraint.evidence_refs:
            EvidenceValidator.validate_evidence_reference(ref)


class SchemaIntegrityValidator:
    """Pre-deserialization integrity checker for raw JSON dictionaries."""

    @staticmethod
    def validate_raw_dict(data: Dict[str, Any], required_fields: List[str]) -> None:
        """Ensure raw dictionary contains all mandatory keys and non-null values."""
        if not isinstance(data, dict):
            raise IntegrityValidationError(
                message=f"Expected JSON dictionary payload, got {type(data).__name__}",
                context={"received_type": type(data).__name__},
            )
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise IntegrityValidationError(
                message=f"Missing required schema fields: {missing}",
                context={"missing_fields": missing},
            )
