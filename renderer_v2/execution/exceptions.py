"""
exceptions.py
=============

Custom exception hierarchy for the Phase 4.1 Execution Engine Foundation.
"""

from __future__ import annotations


class ExecutionEngineError(Exception):
    """Base exception for all Execution Engine errors."""

    pass


class PackageValidationError(ExecutionEngineError):
    """Raised when a RenderExecutionPackage fails validation checks."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors or []


class GraphValidationError(ExecutionEngineError):
    """Raised when an ExecutionGraph fails dependency or cycle validation."""

    def __init__(self, message: str, cycles: list[list[str]] | None = None) -> None:
        super().__init__(message)
        self.cycles: list[list[str]] = cycles or []


class WorkspaceValidationError(ExecutionEngineError):
    """Raised when a RenderWorkspace fails integrity or state validation."""

    pass


class OperationExecutionError(ExecutionEngineError):
    """Raised when an individual RenderOperation fails execution."""

    def __init__(self, message: str, op_id: str = "", stage: str = "") -> None:
        super().__init__(message)
        self.op_id = op_id
        self.stage = stage


class StageExecutionError(ExecutionEngineError):
    """Raised when a placeholder stage fails validation or execution."""

    pass


class JobCancellationError(ExecutionEngineError):
    """Raised when a RenderJob is cancelled during execution."""

    def __init__(self, message: str = "Job execution was cancelled", job_id: str = "") -> None:
        super().__init__(message)
        self.job_id = job_id


class InvalidStateTransitionError(ExecutionEngineError):
    """Raised when an illegal FSM transition is attempted."""

    def __init__(self, from_state: Any = None, to_state: Any = None, reason: str = "") -> None:
        message = f"Illegal FSM state transition from {from_state} to {to_state}. {reason}".strip()
        super().__init__(message)
        self.from_state = from_state
        self.to_state = to_state

