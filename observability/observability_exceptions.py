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


class RuleEngineError(PORCEError):
    """Base exception for rule engine evaluation failures."""


class ReportRenderingError(PORCEError):
    """Raised when generating human or machine reports fails."""
