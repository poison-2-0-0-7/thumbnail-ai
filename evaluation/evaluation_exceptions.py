"""
evaluation_exceptions.py
========================

Typed exception hierarchy for the Pipeline Validation & Quality Evaluation Framework (PVQEF).
"""


class PVQEFError(Exception):
    """Base exception for all PVQEF errors."""


class PipelineStageInvocationError(PVQEFError):
    """Raised when a pipeline stage cannot be invoked (signature mismatch or module failure)."""


class ModuleArtifactMissingError(PVQEFError):
    """Raised when a module's expected persisted JSON artifact is missing from disk."""


class ModuleArtifactSchemaError(PVQEFError):
    """Raised when a persisted module artifact fails Pydantic schema validation."""


class QualityScorerError(PVQEFError):
    """Base class for quality scorer failures."""


class QualityScorerModelUnavailableError(QualityScorerError):
    """Raised when a required vision stack wrapper model cannot be loaded."""


class DeterminismCheckError(QualityScorerError):
    """Raised when a repeated-generation determinism check fails to execute."""


class ReportPersistError(PVQEFError):
    """Raised when a PipelineRunReport cannot be written to disk."""


class GoldenSampleInvalidError(PVQEFError):
    """Raised when the golden sample manifest or baseline is invalid."""


class RegressionRuleError(PVQEFError):
    """Raised when a regression rule cannot be evaluated."""
