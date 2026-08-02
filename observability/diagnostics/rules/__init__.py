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
from observability.diagnostics.rules.decision_honoring_rules import (
    BackgroundRegeneratedUnnecessarilyRule,
    EditMaskIgnoredRule,
    RendererIgnoredEditPlanRule,
)
from observability.diagnostics.rules.latent_initialization_rules import (
    SourceNeverEncodedRule,
)
from observability.diagnostics.rules.prompt_consistency_rules import (
    PromptContradictionRule,
)

DEFAULT_RULE_CLASSES = [
    SourceNeverEncodedRule,
    ControlNetMissingButExpectedRule,
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
]

__all__ = [
    "SourceNeverEncodedRule",
    "ControlNetMissingButExpectedRule",
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
    "DEFAULT_RULE_CLASSES",
]
