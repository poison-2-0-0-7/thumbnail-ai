"""
observability_exceptions.py
===========================

Typed exception hierarchy for the Pipeline Observability & Root Cause Engine (PORCE).
"""


class PORCEError(Exception):
    """Base exception for all PORCE errors."""


class ArtifactIndexError(PORCEError):
    """Raised when artifact discovery or indexing fails critically."""


class LogCorrelationError(PORCEError):
    """Raised when log correlation or log file parsing fails critically."""


class TraceAssemblyError(PORCEError):
    """Raised when assembling a PipelineTrace fails."""


class FactExtractionError(PORCEError):
    """Raised when extracting facts from a trace fails."""


class FactPersistenceError(PORCEError):
    """Raised when saving or loading facts fails."""


class FactValidationError(PORCEError):
    """Raised when fact schema validation fails."""


class RuleEngineError(PORCEError):
    """Base exception for rule engine evaluation failures."""


class ReportRenderingError(PORCEError):
    """Raised when generating human or machine reports fails."""


class RootCauseAssemblyError(PORCEError):
    """Raised when assembling a RootCauseReport fails."""


class RootCausePersistenceError(PORCEError):
    """Raised when saving or loading a RootCauseReport fails."""


class RootCauseValidationError(PORCEError):
    """Raised when RootCauseReport schema validation fails."""


