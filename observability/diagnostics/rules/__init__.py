"""
observability/diagnostics/rules package
========================================

Diagnostic rule families for PORCE.
"""

from observability.diagnostics.rules.asset_provenance_rules import (
    AssetExtractionMissingRule,
    ObjectMappingIncorrectRule,
)
from observability.diagnostics.rules.composition_rules import (
    CompositionMismatchRule,
    IdentityDriftRule,
    MaskOverlapProblemRule,
)
from observability.diagnostics.rules.conditioning_rules import (
    ConditioningFailureRule,
    ControlNetMissingButExpectedRule,
    IPAdapterDisabledButReferenceExistsRule,
)
from observability.diagnostics.rules.controlnet_capability_rules import (
    ControlNetCapabilityResolutionRule,
)
from observability.diagnostics.rules.decision_honoring_rules import (
    BackgroundRegeneratedUnnecessarilyRule,
    EditMaskIgnoredRule,
    RendererIgnoredEditPlanRule,
)
from observability.diagnostics.rules.edit_mode_resolution_rules import (
    EditCapabilityReachabilityRule,
    StagedEditDenoiseStrengthRule,
)
from observability.diagnostics.rules.latent_initialization_rules import (
    SourceNeverEncodedRule,
)
from observability.diagnostics.rules.optimization_diagnostic_rules import (
    GeneratedThumbnailDidNotBeatOriginalRule,
    OptimizationSelectionDisagreementRule,
    OverEditedAcceptedRule,
)
from observability.diagnostics.rules.prompt_consistency_rules import (
    PromptContradictionRule,
)
from observability.diagnostics.rules.candidate_selection_rules import (
    DuplicateCandidateDetectionRule,
    WeakDiversityRule,
    InconsistentRankingRule,
    PoorWinnerSelectionRule,
    CandidateDiversityRule,
    CandidateHardGateRateRule,
    CandidateRankingDominanceRule,
    StrategyPackMismatchRule,
)

from observability.diagnostics.rules.creator_style_rules import (
    BrandingInconsistencyRule,
    IdentityLossWithoutDriftRule,
    StyleViolationRule,
)

DEFAULT_RULE_CLASSES = [
    SourceNeverEncodedRule,
    ControlNetMissingButExpectedRule,
    ControlNetCapabilityResolutionRule,
    IPAdapterDisabledButReferenceExistsRule,
    ConditioningFailureRule,
    EditMaskIgnoredRule,
    RendererIgnoredEditPlanRule,
    BackgroundRegeneratedUnnecessarilyRule,
    AssetExtractionMissingRule,
    ObjectMappingIncorrectRule,
    PromptContradictionRule,
    CompositionMismatchRule,
    MaskOverlapProblemRule,
    IdentityDriftRule,
    EditCapabilityReachabilityRule,
    StagedEditDenoiseStrengthRule,
    GeneratedThumbnailDidNotBeatOriginalRule,
    OverEditedAcceptedRule,
    OptimizationSelectionDisagreementRule,
    DuplicateCandidateDetectionRule,
    WeakDiversityRule,
    InconsistentRankingRule,
    PoorWinnerSelectionRule,
    CandidateDiversityRule,
    CandidateHardGateRateRule,
    CandidateRankingDominanceRule,
    StrategyPackMismatchRule,
    StyleViolationRule,
    BrandingInconsistencyRule,
    IdentityLossWithoutDriftRule,
]

__all__ = [
    "SourceNeverEncodedRule",
    "ControlNetMissingButExpectedRule",
    "ControlNetCapabilityResolutionRule",
    "IPAdapterDisabledButReferenceExistsRule",
    "ConditioningFailureRule",
    "EditMaskIgnoredRule",
    "RendererIgnoredEditPlanRule",
    "BackgroundRegeneratedUnnecessarilyRule",
    "AssetExtractionMissingRule",
    "ObjectMappingIncorrectRule",
    "PromptContradictionRule",
    "CompositionMismatchRule",
    "MaskOverlapProblemRule",
    "IdentityDriftRule",
    "EditCapabilityReachabilityRule",
    "StagedEditDenoiseStrengthRule",
    "GeneratedThumbnailDidNotBeatOriginalRule",
    "OverEditedAcceptedRule",
    "OptimizationSelectionDisagreementRule",
    "DuplicateCandidateDetectionRule",
    "WeakDiversityRule",
    "InconsistentRankingRule",
    "PoorWinnerSelectionRule",
    "CandidateDiversityRule",
    "CandidateHardGateRateRule",
    "CandidateRankingDominanceRule",
    "StrategyPackMismatchRule",
    "StyleViolationRule",
    "BrandingInconsistencyRule",
    "IdentityLossWithoutDriftRule",
    "DEFAULT_RULE_CLASSES",
]


