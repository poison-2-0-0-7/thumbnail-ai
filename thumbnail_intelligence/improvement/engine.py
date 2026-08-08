"""
engine.py
=========

AutomaticImprovementEngine Implementation for Phase 5.5.
Converts an ImprovementPlan / CritiqueReport into a revised RenderExecutionPackage using targeted modifications.

Preserves as much previous work as possible without blindly regenerating the thumbnail.
Does NOT execute rendering directly. Emits UpdatedRenderExecutionPackage containing the updated RenderExecutionPackage,
ImprovementExecutionPlan, and ModificationReport.
"""

from __future__ import annotations

import logging
from typing import Optional

from thumbnail_intelligence.critique.models import CritiqueReport, ImprovementPlan
from thumbnail_intelligence.improvement.models import (
    ImprovementExecutionPlan,
    ImprovementStrategyType,
    ModificationReport,
    UpdatedRenderExecutionPackage,
)
from thumbnail_intelligence.improvement.planner import ModificationPlanner, PlannerValidationError
from thumbnail_intelligence.ranking.models import RankingResult
from thumbnail_intelligence.reasoning.renderer_adapter_models import RenderExecutionPackage

logger = logging.getLogger(__name__)


class ImprovementEngineError(RuntimeError):
    """Exception raised for automatic improvement engine errors or invalid inputs."""
    pass


class AutomaticImprovementEngine:
    """Targeted automatic improvement engine producing revised RenderExecutionPackages."""

    def __init__(self, planner: Optional[ModificationPlanner] = None) -> None:
        self.planner = planner or ModificationPlanner()

    def improve_package(
        self,
        base_package: RenderExecutionPackage,
        improvement_plan: ImprovementPlan,
        strategy: Optional[ImprovementStrategyType] = None,
    ) -> UpdatedRenderExecutionPackage:
        """Convert an ImprovementPlan into an UpdatedRenderExecutionPackage via targeted modifications.

        Args:
            base_package: Base RenderExecutionPackage.
            improvement_plan: ImprovementPlan produced by IntelligentCritiqueEngine.
            strategy: Optional ImprovementStrategyType (CONSERVATIVE, BALANCED, AGGRESSIVE).

        Returns:
            UpdatedRenderExecutionPackage wrapping updated package, execution plan, and report.
        """
        if not base_package:
            raise ImprovementEngineError("Input base_package cannot be None.")

        if not improvement_plan or not improvement_plan.prioritized_suggestions:
            raise ImprovementEngineError("Input improvement_plan cannot be None or empty.")

        strat = strategy or ImprovementStrategyType.BALANCED
        logger.info(f"=== Starting AutomaticImprovementEngine for package '{base_package.metadata.package_id}' (strategy='{strat.value}') ===")

        # 1. Plan and apply modifications
        try:
            updated_pkg, exec_plan, mod_report = self.planner.plan_modifications(
                base_package=base_package,
                suggestions=improvement_plan.prioritized_suggestions,
                strategy=strat,
            )
        except PlannerValidationError as e:
            raise ImprovementEngineError(f"Modification planning failed: {str(e)}") from e

        # 2. Validate updated package bounds & schema
        errors = updated_pkg.validate_package()
        if errors:
            raise ImprovementEngineError(f"Updated RenderExecutionPackage validation failed: {errors}")

        res = UpdatedRenderExecutionPackage(
            package=updated_pkg,
            execution_plan=exec_plan,
            report=mod_report,
        )

        logger.info(
            f"=== Completed AutomaticImprovementEngine for package '{updated_pkg.metadata.package_id}' "
            f"({mod_report.modified_layers_count} modified, {mod_report.preserved_layers_count} preserved, "
            f"preservation_ratio={mod_report.preservation_ratio:.0%}) ==="
        )
        return res

    def improve_critique_report(
        self,
        critique_report: CritiqueReport,
        base_package: RenderExecutionPackage,
        strategy: Optional[ImprovementStrategyType] = None,
    ) -> UpdatedRenderExecutionPackage:
        """Convenience method: Improve a base RenderExecutionPackage directly from a CritiqueReport."""
        if not critique_report or not critique_report.improvement_plan:
            raise ImprovementEngineError("CritiqueReport must contain a valid ImprovementPlan.")

        return self.improve_package(
            base_package=base_package,
            improvement_plan=critique_report.improvement_plan,
            strategy=strategy,
        )

    def improve_ranking_result(
        self,
        ranking_result: RankingResult,
        base_package: RenderExecutionPackage,
        strategy: Optional[ImprovementStrategyType] = None,
    ) -> UpdatedRenderExecutionPackage:
        """Convenience method: Improve a base RenderExecutionPackage from a RankingResult via IntelligentCritiqueEngine."""
        from thumbnail_intelligence.critique import IntelligentCritiqueEngine

        critique_report = IntelligentCritiqueEngine().critique_ranking_result(ranking_result)
        return self.improve_critique_report(
            critique_report=critique_report,
            base_package=base_package,
            strategy=strategy,
        )
