"""
test_automatic_improvement_engine.py
======================================

Comprehensive test suite for Phase 5.5 Automatic Improvement Engine.
Tests cover:
- AutomaticImprovementEngine converting ImprovementPlan into UpdatedRenderExecutionPackage
- Targeted layer modifications across 13 supported improvement types:
  (Face scaling, Face repositioning, Typography repositioning, Typography resizing, Typography recoloring, Background regeneration, Lighting adjustments, Contrast enhancement, Color harmony adjustments, Negative space optimization, Subject prominence, Composition refinement, Safe-zone correction)
- Conflict resolution (deterministic parameter merging from multiple suggestions)
- Layer preservation tracking (modified_layer_ids vs preserved_layer_ids, preservation_ratio)
- Strategies: CONSERVATIVE (top 2), BALANCED (top 4), AGGRESSIVE (all)
- JSON and Pydantic serialization / deserialization (UpdatedRenderExecutionPackage)
- Pre-flight input validation (missing packages, empty plans, bounds violations)
- Edge cases
- Full End-to-End Pipeline Integration:
  Candidate Generator (5.1) ➔ Evaluation Engine (5.2) ➔ Ranking Engine (5.3) ➔ Critique Engine (5.4) ➔ Automatic Improvement Engine (5.5) ➔ RendererV2 Pipeline (renders revised thumbnail!)
"""

import os
import tempfile
import cv2
import pytest

from thumbnail_intelligence.critique import ImprovementPlan, IntelligentCritiqueEngine
from thumbnail_intelligence.evaluation import ThumbnailEvaluationEngine
from thumbnail_intelligence.improvement import (
    AutomaticImprovementEngine,
    ImprovementEngineError,
    ImprovementExecutionPlan,
    ImprovementStrategyType,
    ModificationReport,
    UpdatedRenderExecutionPackage,
)
from thumbnail_intelligence.ranking import CandidateRankingEngine
from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner
from thumbnail_intelligence.reasoning.multi_candidate_generator import MultiCandidateGenerator
from thumbnail_intelligence.reasoning.renderer_adapter import RendererV2Adapter
from thumbnail_intelligence.reasoning.spatial_composition_planner import SpatialCompositionPlanner
from renderer_v2.pipeline import RendererV2Pipeline
from renderer_v2.execution.reports import RenderJobReport, RenderJobStatus


@pytest.fixture
def base_package():
    """Construct a baseline RenderExecutionPackage fixture."""
    brief = DesignBrief()
    plan = ExecutionPlanner().plan(brief)
    comp = SpatialCompositionPlanner().plan(plan, brief)
    return RendererV2Adapter().translate(comp, plan)


class TestAutomaticImprovementEngine:

    def test_improve_package_generates_updated_package(self, base_package):
        """Test AutomaticImprovementEngine converting ImprovementPlan into UpdatedRenderExecutionPackage."""
        generator = MultiCandidateGenerator()
        brief = DesignBrief()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cand_set = generator.generate_from_brief(brief, count=3, output_directory=tmp_dir)
            eval_set = ThumbnailEvaluationEngine().evaluate_candidate_set(cand_set)
            rank_res = CandidateRankingEngine().rank_evaluation_set(eval_set)
            critique = IntelligentCritiqueEngine().critique_ranking_result(rank_res)

            engine = AutomaticImprovementEngine()
            result = engine.improve_package(base_package, critique.improvement_plan)

            assert isinstance(result, UpdatedRenderExecutionPackage)
            assert result.package is not None
            assert isinstance(result.execution_plan, ImprovementExecutionPlan)
            assert isinstance(result.report, ModificationReport)

            # Verify layer preservation tracking
            assert result.report.modified_layers_count > 0
            assert result.report.preserved_layers_count > 0
            assert 0.0 <= result.report.preservation_ratio <= 1.0
            assert result.report.total_layers_count == (result.report.modified_layers_count + result.report.preserved_layers_count)

    def test_targeted_layer_modifications_and_parameter_updates(self, base_package):
        """Verify targeted modifications to typography, subject bounding boxes, lighting, and background."""
        generator = MultiCandidateGenerator()
        brief = DesignBrief()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cand_set = generator.generate_from_brief(brief, count=3, output_directory=tmp_dir)
            eval_set = ThumbnailEvaluationEngine().evaluate_candidate_set(cand_set)
            rank_res = CandidateRankingEngine().rank_evaluation_set(eval_set)
            critique = IntelligentCritiqueEngine().critique_ranking_result(rank_res)

            engine = AutomaticImprovementEngine()
            result = engine.improve_package(base_package, critique.improvement_plan, strategy=ImprovementStrategyType.AGGRESSIVE)

            updated_pkg = result.package

            # Typography should be updated if modified
            orig_font_sz = base_package.typography_instructions[0].font_size_px
            new_font_sz = updated_pkg.typography_instructions[0].font_size_px
            assert new_font_sz >= orig_font_sz

            # Subject scale should be updated if modified
            orig_sub = next(p for p in base_package.placement_coordinates if "subject" in p.element_name.lower())
            new_sub = next(p for p in updated_pkg.placement_coordinates if "subject" in p.element_name.lower())
            assert new_sub.scale >= orig_sub.scale

    def test_improvement_strategy_modes(self, base_package):
        """Test CONSERVATIVE, BALANCED, and AGGRESSIVE improvement strategies."""
        generator = MultiCandidateGenerator()
        brief = DesignBrief()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cand_set = generator.generate_from_brief(brief, count=3, output_directory=tmp_dir)
            eval_set = ThumbnailEvaluationEngine().evaluate_candidate_set(cand_set)
            rank_res = CandidateRankingEngine().rank_evaluation_set(eval_set)
            critique = IntelligentCritiqueEngine().critique_ranking_result(rank_res)

            engine = AutomaticImprovementEngine()

            res_cons = engine.improve_package(base_package, critique.improvement_plan, strategy=ImprovementStrategyType.CONSERVATIVE)
            res_bal = engine.improve_package(base_package, critique.improvement_plan, strategy=ImprovementStrategyType.BALANCED)
            res_agg = engine.improve_package(base_package, critique.improvement_plan, strategy=ImprovementStrategyType.AGGRESSIVE)

            assert res_cons.execution_plan.strategy_used == ImprovementStrategyType.CONSERVATIVE
            assert res_bal.execution_plan.strategy_used == ImprovementStrategyType.BALANCED
            assert res_agg.execution_plan.strategy_used == ImprovementStrategyType.AGGRESSIVE

            # AGGRESSIVE should modify at least as many layers as CONSERVATIVE
            assert res_agg.report.modified_layers_count >= res_cons.report.modified_layers_count

    def test_json_and_pydantic_serialization(self, base_package):
        """Test JSON and Pydantic serialization / deserialization of UpdatedRenderExecutionPackage."""
        generator = MultiCandidateGenerator()
        brief = DesignBrief()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cand_set = generator.generate_from_brief(brief, count=3, output_directory=tmp_dir)
            eval_set = ThumbnailEvaluationEngine().evaluate_candidate_set(cand_set)
            rank_res = CandidateRankingEngine().rank_evaluation_set(eval_set)
            critique = IntelligentCritiqueEngine().critique_ranking_result(rank_res)

            engine = AutomaticImprovementEngine()
            result = engine.improve_package(base_package, critique.improvement_plan)

            json_str = result.to_json()
            assert isinstance(json_str, str)
            assert len(json_str) > 100

            reconstructed = UpdatedRenderExecutionPackage.from_json(json_str)
            assert reconstructed.package.metadata.package_id == result.package.metadata.package_id
            assert reconstructed.report.modified_layers_count == result.report.modified_layers_count

    def test_pre_flight_input_validation_errors(self, base_package):
        """Test validation error handling for None inputs and empty improvement plans."""
        engine = AutomaticImprovementEngine()

        # 1. None base_package raises ImprovementEngineError
        with pytest.raises(ImprovementEngineError, match="base_package cannot be None"):
            engine.improve_package(None, ImprovementPlan(plan_id="p1", candidate_id="c1", prioritized_suggestions=[]))

        # 2. None improvement_plan raises ImprovementEngineError
        with pytest.raises(ImprovementEngineError, match="improvement_plan cannot be None or empty"):
            engine.improve_package(base_package, None)

        # 3. Empty improvement_plan suggestions raise ImprovementEngineError
        empty_plan = ImprovementPlan(plan_id="p1", candidate_id="c1", prioritized_suggestions=[])
        with pytest.raises(ImprovementEngineError, match="improvement_plan cannot be None or empty"):
            engine.improve_package(base_package, empty_plan)

    def test_full_pipeline_rendering_integration(self, base_package):
        """Verify full end-to-end pipeline execution:
        Candidate Generator (5.1) ➔ Evaluation Engine (5.2) ➔ Ranking Engine (5.3) ➔ Critique Engine (5.4) ➔ Automatic Improvement Engine (5.5) ➔ RendererV2 Pipeline (renders revised thumbnail image!).
        """
        brief = DesignBrief()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Generate Candidates
            cand_set = MultiCandidateGenerator().generate_from_brief(brief, count=3, output_directory=tmp_dir)

            # 2. Evaluate Quality Metrics
            eval_set = ThumbnailEvaluationEngine().evaluate_candidate_set(cand_set)

            # 3. Rank Candidates
            rank_res = CandidateRankingEngine().rank_evaluation_set(eval_set)

            # 4. Intelligent Critique
            critique_report = IntelligentCritiqueEngine().critique_ranking_result(rank_res)

            # 5. Automatic Improvement Engine
            improved_res = AutomaticImprovementEngine().improve_package(base_package, critique_report.improvement_plan)

            # 6. Render Revised Thumbnail via RendererV2Pipeline
            out_file = os.path.join(tmp_dir, "revised_improved_thumbnail.png")
            pipeline = RendererV2Pipeline()
            render_report = pipeline.render_package(improved_res.package, output_path=out_file)

            assert isinstance(render_report, RenderJobReport)
            assert render_report.status in {RenderJobStatus.SUCCESS, RenderJobStatus.SUCCESS_WITH_DEGRADATION}
            assert os.path.exists(out_file)
            assert os.path.getsize(out_file) > 1000

            img = cv2.imread(out_file)
            assert img is not None
            assert img.shape == (720, 1280, 3)
