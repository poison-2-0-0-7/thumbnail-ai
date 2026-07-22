"""Typed exception hierarchy shared by Module 7 foundation components."""


class Module7Error(Exception):
    """Base exception for every Module 7 failure."""


class ProfileDowngradedWarning(Warning):
    """Signals that an explicit profile request could not fit measured VRAM."""


class ComfyUIConnectionError(Module7Error):
    """Reserved for Phase 2 local ComfyUI connectivity failures."""


class ComfyUIQueueError(Module7Error):
    """Reserved for Phase 2 ComfyUI execution failures."""


class OutputRetrievalError(Module7Error):
    """Base class for completed ComfyUI output retrieval failures."""

    def __init__(
        self,
        message: str,
        *,
        prompt_id: str | None = None,
        output_node_id: str | None = None,
        filename: str | None = None,
    ) -> None:
        super().__init__(message)
        self.prompt_id = prompt_id
        self.output_node_id = output_node_id
        self.filename = filename


class OutputHistoryError(OutputRetrievalError):
    """Raised when a completed ComfyUI history payload is missing or malformed."""


class NoOutputImageError(OutputRetrievalError):
    """Raised when a completed ComfyUI prompt produced no usable image outputs."""


class OutputDownloadError(OutputRetrievalError):
    """Raised when a selected ComfyUI output image cannot be downloaded."""


class MissingOutputFileError(OutputDownloadError):
    """Raised when ComfyUI reports an output file that is absent from /view."""


class CorruptImageError(OutputRetrievalError):
    """Raised when downloaded output bytes are empty or undecodable as an image."""


class UnsupportedImageFormatError(OutputRetrievalError):
    """Raised when an output image format is outside Module 7's supported formats."""


class VRAMExhaustedError(Module7Error):
    """Reserved for Phase 2 GPU out-of-memory failures."""


class IdentityPreservationError(Module7Error):
    """Reserved for Phase 2 identity-gate failures."""


class QualityAssuranceError(Module7Error):
    """Reserved for Phase 2 quality-assurance failures."""


class PromptPackageInvalidError(Module7Error):
    """Raised when a persisted Module 6 package is missing or unusable."""


class ReferenceAssetError(Module7Error):
    """Raised when a required local reference asset cannot be resolved."""


class WorkflowTemplateError(Module7Error):
    """Raised when a workflow template is invalid or escapes its library."""


class WorkflowBuildError(Module7Error):
    """Raised when a template cannot be deterministically materialized."""


class ArtifactWriteError(Module7Error):
    """Raised when a Module 7 manifest cannot be atomically persisted."""


class MetricsWriteError(Module7Error):
    """Raised when Module 7 monitoring data cannot be appended."""


class NoEligibleCandidateError(Module7Error):
    """Reserved for Phase 2 candidate-ranking failures."""
