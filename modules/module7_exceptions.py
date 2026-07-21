"""Typed exception hierarchy shared by Module 7 foundation components."""


class Module7Error(Exception):
    """Base exception for every Module 7 failure."""


class ProfileDowngradedWarning(Warning):
    """Signals that an explicit profile request could not fit measured VRAM."""


class ComfyUIConnectionError(Module7Error):
    """Reserved for Phase 2 local ComfyUI connectivity failures."""


class ComfyUIQueueError(Module7Error):
    """Reserved for Phase 2 ComfyUI execution failures."""


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
