"""
decision_exceptions.py
======================

Typed exception hierarchy for Module 9 (AI Decision Engine).
Leaf module with zero project-internal dependencies.
"""


class DecisionEngineError(Exception):
    """Base exception for every Module 9 failure."""


class InputBundleError(DecisionEngineError):
    """Base for failures loading/validating M4/M5/M6/M8 inputs."""


class MissingArtifactError(InputBundleError):
    """Raised when a required upstream artifact file does not exist."""


class ArtifactValidationError(InputBundleError):
    """Raised when an upstream artifact fails Pydantic validation."""


class AssetExtractionManifestError(InputBundleError):
    """Raised when Module 8's manifest is malformed."""


class RuleEvaluationError(DecisionEngineError):
    """Raised when a rule function raises unexpectedly."""


class LLMReasoningError(DecisionEngineError):
    """Base for local Ollama adjudication failures."""


class OllamaConnectionError(LLMReasoningError):
    """Could not reach the local Ollama server."""


class OllamaTimeoutError(LLMReasoningError):
    """Ollama request exceeded its configured deadline."""


class OllamaResponseParseError(LLMReasoningError):
    """Ollama's JSON response could not be parsed or failed schema validation."""


class ConflictResolutionError(DecisionEngineError):
    """Raised when conflict resolution cannot converge."""


class DecisionValidationError(DecisionEngineError):
    """Raised for a hard validation failure that blocks persistence."""


class ManifestPersistError(DecisionEngineError):
    """Raised when the decision manifest or per-action files cannot be atomically written."""


class DecisionCacheError(DecisionEngineError):
    """Raised when the decision cache cannot be read or written."""
