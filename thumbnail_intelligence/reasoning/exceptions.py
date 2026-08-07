"""
exceptions.py
=============

Structured exception hierarchy for the Strategic Reasoning Coordinator Foundation.
Provides domain-specific exceptions for registry lifecycle, reasoner validation,
dependency resolution, coordinator execution, and pipeline orchestration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from thumbnail_intelligence.knowledge_base.exceptions import KnowledgeBaseError


class ReasoningError(KnowledgeBaseError):
    """
    Base exception for all errors occurring within the Strategic Reasoning Subsystem.
    All reasoning exceptions inherit from this class.
    """

    default_error_code: str = "REASONING_ERROR"


# ---------------------------------------------------------------------------
# Registry Exceptions
# ---------------------------------------------------------------------------


class RegistryError(ReasoningError):
    """Base exception for reasoner registry operations (registration, lookup, ordering)."""

    default_error_code: str = "REASONING_REGISTRY_ERROR"


class ReasonerNotFoundError(RegistryError):
    """Raised when a requested reasoner is not found in the registry."""

    default_error_code: str = "REASONER_NOT_FOUND"

    def __init__(
        self,
        reasoner_name: str,
        available_reasoners: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        ctx["reasoner_name"] = reasoner_name
        ctx["available_reasoners"] = available_reasoners or []
        msg = (
            f"Reasoner '{reasoner_name}' was not found in registry. "
            f"Available reasoners: {ctx['available_reasoners']}"
        )
        super().__init__(msg, error_code=self.default_error_code, context=ctx)


class DuplicateReasonerError(RegistryError):
    """Raised when attempting to register a reasoner whose name is already registered."""

    default_error_code: str = "DUPLICATE_REASONER_ERROR"

    def __init__(
        self,
        reasoner_name: str,
        existing_version: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        ctx["reasoner_name"] = reasoner_name
        ctx["existing_version"] = existing_version
        msg = f"Reasoner '{reasoner_name}' is already registered in the registry."
        super().__init__(msg, error_code=self.default_error_code, context=ctx)


class MissingDependencyError(RegistryError):
    """Raised when a reasoner declares a dependency on another reasoner that is not registered."""

    default_error_code: str = "MISSING_REASONER_DEPENDENCY"

    def __init__(
        self,
        reasoner_name: str,
        missing_dependency: str,
        available_reasoners: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        ctx["reasoner_name"] = reasoner_name
        ctx["missing_dependency"] = missing_dependency
        ctx["available_reasoners"] = available_reasoners or []
        msg = (
            f"Reasoner '{reasoner_name}' depends on '{missing_dependency}', "
            f"which is not registered in the registry."
        )
        super().__init__(msg, error_code=self.default_error_code, context=ctx)


class CircularDependencyError(RegistryError):
    """Raised when circular dependencies are detected between registered reasoners."""

    default_error_code: str = "CIRCULAR_REASONER_DEPENDENCY"

    def __init__(
        self,
        cycle_path: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        ctx["cycle_path"] = cycle_path
        path_str = " -> ".join(cycle_path)
        msg = f"Circular dependency detected in reasoning graph: {path_str}"
        super().__init__(msg, error_code=self.default_error_code, context=ctx)


class InvalidReasonerError(RegistryError):
    """Raised when a reasoner fails contract validation or does not conform to the base interface."""

    default_error_code: str = "INVALID_REASONER_ERROR"

    def __init__(
        self,
        reasoner_name: str,
        validation_errors: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        ctx["reasoner_name"] = reasoner_name
        ctx["validation_errors"] = validation_errors or []
        msg = f"Reasoner '{reasoner_name}' is invalid: {'; '.join(ctx['validation_errors'])}"
        super().__init__(msg, error_code=self.default_error_code, context=ctx)


# ---------------------------------------------------------------------------
# Coordinator Exceptions
# ---------------------------------------------------------------------------


class CoordinatorError(ReasoningError):
    """Base exception for reasoning coordinator orchestration failures."""

    default_error_code: str = "COORDINATOR_ERROR"


class ReasonerExecutionError(CoordinatorError):
    """Raised when an exception occurs while executing a reasoner."""

    default_error_code: str = "REASONER_EXECUTION_ERROR"

    def __init__(
        self,
        reasoner_name: str,
        underlying_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        ctx["reasoner_name"] = reasoner_name
        ctx["underlying_error_type"] = type(underlying_error).__name__ if underlying_error else None
        ctx["underlying_error_msg"] = str(underlying_error) if underlying_error else None
        msg = f"Execution of reasoner '{reasoner_name}' failed: {underlying_error}"
        super().__init__(msg, error_code=self.default_error_code, context=ctx)


class ReasonerValidationError(CoordinatorError):
    """Raised when a reasoner's output fails validation checks."""

    default_error_code: str = "REASONER_VALIDATION_ERROR"

    def __init__(
        self,
        reasoner_name: str,
        validation_errors: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        ctx["reasoner_name"] = reasoner_name
        ctx["validation_errors"] = validation_errors or []
        msg = f"Output of reasoner '{reasoner_name}' failed validation: {'; '.join(ctx['validation_errors'])}"
        super().__init__(msg, error_code=self.default_error_code, context=ctx)


class ContextConstructionError(CoordinatorError):
    """Raised when synthesizing or merging the final ReasoningContext fails."""

    default_error_code: str = "CONTEXT_CONSTRUCTION_ERROR"


class EmptyEvidenceGraphError(CoordinatorError):
    """Raised when an empty or None NormalizedEvidenceGraph is provided to the coordinator."""

    default_error_code: str = "EMPTY_EVIDENCE_GRAPH_ERROR"


class CoordinatorTimeoutError(CoordinatorError):
    """Raised when a reasoner exceeds its allocated execution timeout."""

    default_error_code: str = "COORDINATOR_TIMEOUT_ERROR"

    def __init__(
        self,
        reasoner_name: str,
        timeout_ms: float,
        elapsed_ms: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        ctx["reasoner_name"] = reasoner_name
        ctx["timeout_ms"] = timeout_ms
        ctx["elapsed_ms"] = elapsed_ms
        msg = f"Reasoner '{reasoner_name}' timed out after {elapsed_ms:.2f}ms (limit: {timeout_ms:.2f}ms)"
        super().__init__(msg, error_code=self.default_error_code, context=ctx)


# ---------------------------------------------------------------------------
# Pipeline Exceptions
# ---------------------------------------------------------------------------


class PipelineError(ReasoningError):
    """Base exception for reasoning pipeline operations."""

    default_error_code: str = "PIPELINE_ERROR"


class PipelineExecutionError(PipelineError):
    """Raised when the end-to-end reasoning pipeline fails."""

    default_error_code: str = "PIPELINE_EXECUTION_ERROR"


class GroundingEnforcementError(PipelineError):
    """Raised when ungrounded claims or missing evidence references are detected."""

    default_error_code: str = "GROUNDING_ENFORCEMENT_ERROR"

    def __init__(
        self,
        reasoner_name: str,
        details: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        ctx["reasoner_name"] = reasoner_name
        ctx["details"] = details
        msg = f"Grounding enforcement failed for reasoner '{reasoner_name}': {details}"
        super().__init__(msg, error_code=self.default_error_code, context=ctx)
