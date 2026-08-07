"""
interfaces.py
=============

Abstract interfaces and contracts for all strategic reasoning modules.
Establishes the input, output, contract, and validation interfaces for:
- BaseReasoner
- NarrativeReasoner
- AudienceReasoner
- CreatorReasoner
- BrandReasoner
- PriorityReasoner
- RiskReasoner
- StrategyRanker
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from thumbnail_intelligence.evidence.models import NormalizedEvidenceGraph
from thumbnail_intelligence.reasoning.context import ReasoningContext
from thumbnail_intelligence.reasoning.models import (
    AudienceReasoningOutput,
    BrandReasoningOutput,
    CreatorReasoningOutput,
    NarrativeReasoningOutput,
    PriorityReasoningOutput,
    ReasonerContract,
    ReasonerType,
    RiskReasoningOutput,
    StrategyRankingOutput,
)


class BaseReasoner(ABC):
    """
    Abstract base class for all strategic reasoning engines.
    Defines the contract, lifecycle methods, and validation hooks required by the ReasoningCoordinator.
    """

    @property
    @abstractmethod
    def contract(self) -> ReasonerContract:
        """Return the metadata contract defining identity, type, dependencies, and execution bounds."""
        ...

    @property
    def name(self) -> str:
        """Convenience property returning the unique reasoner name."""
        return self.contract.name

    @property
    def reasoner_type(self) -> ReasonerType:
        """Convenience property returning the classified reasoner type."""
        return self.contract.reasoner_type

    @property
    def dependencies(self) -> List[str]:
        """Convenience property returning upstream reasoner dependencies."""
        return self.contract.dependencies

    @property
    def version(self) -> str:
        """Convenience property returning the semantic version string."""
        return self.contract.version

    @property
    def is_mandatory(self) -> bool:
        """Convenience property returning whether this reasoner is mandatory."""
        return self.contract.is_mandatory

    @property
    def timeout_ms(self) -> float:
        """Convenience property returning the execution timeout limit in milliseconds."""
        return self.contract.timeout_ms

    def validate_input(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> bool:
        """
        Validate incoming evidence graph and current reasoning context prior to execution.
        Default implementation checks that graph is a valid NormalizedEvidenceGraph.
        """
        if graph is None or not isinstance(graph, NormalizedEvidenceGraph):
            return False
        if context is None or not isinstance(context, ReasoningContext):
            return False
        return True

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> Any:
        """
        Execute strategic reasoning over the NormalizedEvidenceGraph and current ReasoningContext.
        Must return a validated, strongly typed output model.
        """
        ...

    @abstractmethod
    def validate_output(self, output: Any) -> bool:
        """
        Validate that the produced output satisfies the reasoner's contract and invariants.
        Returns True if valid, or raises an exception / returns False.
        """
        ...


# ---------------------------------------------------------------------------
# Specialized Abstract Reasoner Interfaces
# ---------------------------------------------------------------------------


class NarrativeReasoner(BaseReasoner, ABC):
    """
    Abstract interface for narrative and visual storytelling reasoning.
    Synthesizes story hooks, narrative framing, and visual metaphors from normalized evidence.
    """

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> NarrativeReasoningOutput:
        """Execute narrative storytelling reasoning."""
        ...

    def validate_output(self, output: Any) -> bool:
        """Validate that output is an instance of NarrativeReasoningOutput with valid confidence."""
        if not isinstance(output, NarrativeReasoningOutput):
            return False
        if not (0.0 <= output.confidence <= 1.0):
            return False
        return True


class AudienceReasoner(BaseReasoner, ABC):
    """
    Abstract interface for audience psychology reasoning.
    Synthesizes target audience segments, curiosity triggers, and cognitive expectations.
    """

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> AudienceReasoningOutput:
        """Execute audience psychology reasoning."""
        ...

    def validate_output(self, output: Any) -> bool:
        """Validate that output is an instance of AudienceReasoningOutput with valid confidence."""
        if not isinstance(output, AudienceReasoningOutput):
            return False
        if not (0.0 <= output.confidence <= 1.0):
            return False
        return True


class CreatorReasoner(BaseReasoner, ABC):
    """
    Abstract interface for creator persona and style consistency reasoning.
    Synthesizes creator signature tropes, historical visual voice, and channel equity anchors.
    """

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> CreatorReasoningOutput:
        """Execute creator style and persona reasoning."""
        ...

    def validate_output(self, output: Any) -> bool:
        """Validate that output is an instance of CreatorReasoningOutput with valid confidence."""
        if not isinstance(output, CreatorReasoningOutput):
            return False
        if not (0.0 <= output.confidence <= 1.0):
            return False
        return True


class BrandReasoner(BaseReasoner, ABC):
    """
    Abstract interface for brand constraints and identity protection reasoning.
    Synthesizes palette rules, typography guidelines, logo constraints, and prohibited tropes.
    """

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> BrandReasoningOutput:
        """Execute brand constraint and identity reasoning."""
        ...

    def validate_output(self, output: Any) -> bool:
        """Validate that output is an instance of BrandReasoningOutput with valid confidence."""
        if not isinstance(output, BrandReasoningOutput):
            return False
        if not (0.0 <= output.confidence <= 1.0):
            return False
        if not (0.0 <= output.compliance_score <= 1.0):
            return False
        return True


class PriorityReasoner(BaseReasoner, ABC):
    """
    Abstract interface for visual hierarchy and priority reasoning.
    Synthesizes focal element ordering, visual weight percentages, and contrast priorities.
    """

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> PriorityReasoningOutput:
        """Execute visual hierarchy and priority reasoning."""
        ...

    def validate_output(self, output: Any) -> bool:
        """Validate that output is an instance of PriorityReasoningOutput with valid confidence."""
        if not isinstance(output, PriorityReasoningOutput):
            return False
        if not (0.0 <= output.confidence <= 1.0):
            return False
        return True


class RiskReasoner(BaseReasoner, ABC):
    """
    Abstract interface for risk and fatigue reasoning.
    Evaluates audience fatigue, competitor convergence risk, policy flags, and mitigations.
    """

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> RiskReasoningOutput:
        """Execute risk assessment and mitigation reasoning."""
        ...

    def validate_output(self, output: Any) -> bool:
        """Validate that output is an instance of RiskReasoningOutput with valid confidence."""
        if not isinstance(output, RiskReasoningOutput):
            return False
        if not (0.0 <= output.confidence <= 1.0):
            return False
        if not (0.0 <= output.fatigue_risk_score <= 1.0):
            return False
        if not (0.0 <= output.competitor_convergence_risk <= 1.0):
            return False
        return True


class StrategyRanker(BaseReasoner, ABC):
    """
    Abstract interface for candidate strategy ranking and tradeoff analysis.
    Evaluates candidate thumbnail designs and ranks them based on evidence, expected uplift, and risk.
    """

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> StrategyRankingOutput:
        """Execute strategy ranking and tradeoff analysis."""
        ...

    def validate_output(self, output: Any) -> bool:
        """Validate that output is an instance of StrategyRankingOutput with valid confidence."""
        if not isinstance(output, StrategyRankingOutput):
            return False
        if not (0.0 <= output.confidence <= 1.0):
            return False
        return True


class StrategicReasoningValidator(BaseReasoner, ABC):
    """
    Abstract interface for strategic reasoning validation.
    Verifies internal logical consistency, grounding, and readiness before DesignBrief generation.
    """

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> Any:
        """Execute strategic reasoning validation."""
        ...

    def validate_output(self, output: Any) -> bool:
        """Validate that output is a valid validation package or report."""
        if output is None:
            return False
        return True


class DesignBriefGeneratorInterface(BaseReasoner, ABC):
    """
    Abstract interface for deterministic DesignBrief generation.
    Translates a fully validated reasoning package into a strongly typed, renderer-independent DesignBrief.
    """

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> Any:
        """Execute DesignBrief translation from context/package."""
        ...

    def validate_output(self, output: Any) -> bool:
        """Validate that output is a valid DesignBrief object."""
        if output is None:
            return False
        return True


class ExecutionPlannerInterface(BaseReasoner, ABC):
    """
    Abstract interface for execution planning.
    Translates a DesignBrief into a deterministic, renderer-agnostic ExecutionPlan DAG.
    """

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> Any:
        """Execute ExecutionPlan generation from context/brief."""
        ...

    def validate_output(self, output: Any) -> bool:
        """Validate that output is a valid ExecutionPlan object."""
        if output is None:
            return False
        return True


class SpatialCompositionPlannerInterface(BaseReasoner, ABC):
    """
    Abstract interface for spatial composition planning.
    Translates an ExecutionPlan + DesignBrief into a renderer-independent SpatialComposition.
    """

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> Any:
        """Execute SpatialComposition generation from context/plan."""
        ...

    def validate_output(self, output: Any) -> bool:
        """Validate that output is a valid SpatialComposition object."""
        if output is None:
            return False
        return True


class RendererAdapterInterface(BaseReasoner, ABC):
    """
    Abstract interface for renderer translation adapters.
    Translates a renderer-independent SpatialComposition + ExecutionPlan into a renderer-specific RenderExecutionPackage.
    """

    @abstractmethod
    def reason(self, graph: NormalizedEvidenceGraph, context: ReasoningContext) -> Any:
        """Execute RenderExecutionPackage translation from context/spatial composition."""
        ...

    def validate_output(self, output: Any) -> bool:
        """Validate that output is a valid RenderExecutionPackage object."""
        if output is None:
            return False
        return True





