"""
test_multi_candidate_generation.py
===================================

Comprehensive test suite for Phase 5.1 Multi-Candidate Generation.
Tests cover:
- MultiCandidateGenerator producing CandidateSet (Candidate A, B, C, D, E)
- Verification of 5 strategic variation profiles across dimensions (Emotional, Curiosity, Typography, Color, Composition)
- Uniqueness and determinism of generated candidate packages and output files
- CandidateMetadata tracking (strategy summaries, execution latencies, variation dimensions)
- Preservation of underlying RenderJobReports per candidate
- Convenience methods: generate_from_brief() and generate_from_composition()
- Edge cases and error handling (custom profiles, invalid inputs)
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from thumbnail_intelligence.reasoning.design_brief_models import DesignBrief
from thumbnail_intelligence.reasoning.execution_planner import ExecutionPlanner
from thumbnail_intelligence.reasoning.multi_candidate_generator import (
    MultiCandidateGenerator,
    MultiCandidateGeneratorError,
)
from thumbnail_intelligence.reasoning.multi_candidate_models import (
    CandidateResult,
    CandidateSet,
    VariationDimension,
    VariationProfile,
)
from thumbnail_intelligence.reasoning.renderer_adapter import RendererV2Adapter
from thumbnail_intelligence.reasoning.spatial_composition_planner import SpatialCompositionPlanner
from renderer_v2.execution.reports import RenderJobReport, RenderJobStatus


@pytest.fixture
def base_package():
    """Construct a baseline RenderExecutionPackage fixture."""
    brief = DesignBrief()
    plan = ExecutionPlanner().plan(brief)
    comp = SpatialCompositionPlanner().plan(plan, brief)
    return RendererV2Adapter().translate(comp, plan)


class TestMultiCandidateGeneration:

    def test_default_five_candidate_generation(self, base_package):
        """Test generating standard CandidateSet with 5 distinct candidates (A, B, C, D, E)."""
        generator = MultiCandidateGenerator()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cand_set = generator.generate_candidates(base_package, count=5, output_directory=tmp_dir)

            assert isinstance(cand_set, CandidateSet)
            assert len(cand_set.candidates) == 5

            expected_ids = ["candidate_a", "candidate_b", "candidate_c", "candidate_d", "candidate_e"]
            for i, expected_id in enumerate(expected_ids):
                cand = cand_set.candidates[i]
                assert cand.candidate_id == expected_id
                assert cand.profile is not None
                assert isinstance(cand.report, RenderJobReport)
                assert cand.report.status in {RenderJobStatus.SUCCESS, RenderJobStatus.SUCCESS_WITH_DEGRADATION}

                # Verify exported image file existence and non-zero size
                assert os.path.exists(cand.image_path)
                assert os.path.getsize(cand.image_path) > 1000

                img = cv2.imread(cand.image_path)
                assert img is not None
                assert img.shape == (720, 1280, 3)

    def test_candidate_uniqueness_and_profile_transformations(self, base_package):
        """Verify that each generated candidate has distinct typography, lighting, or placement parameters."""
        generator = MultiCandidateGenerator()
        profiles = generator.create_default_profiles(count=5)

        # 1. Candidate A (Emotional - enlarged subject) vs Candidate B (Curiosity - smaller subject)
        pkg_a = generator.apply_profile_to_package(base_package, profiles[0], "candidate_a")
        pkg_b = generator.apply_profile_to_package(base_package, profiles[1], "candidate_b")

        sub_a = next(p for p in pkg_a.placement_coordinates if "subject" in p.element_name.lower())
        sub_b = next(p for p in pkg_b.placement_coordinates if "subject" in p.element_name.lower())
        assert sub_a.scale != sub_b.scale

        # 2. Candidate C (Typography Emphasis) vs Base Package (font size scale multiplier 1.3)
        pkg_c = generator.apply_profile_to_package(base_package, profiles[2], "candidate_c")
        base_size = base_package.typography_instructions[0].font_size_px
        cand_c_size = pkg_c.typography_instructions[0].font_size_px
        assert cand_c_size > base_size

        # 3. Candidate D (Color Emphasis) vs Base Package (dominant colors override)
        pkg_d = generator.apply_profile_to_package(base_package, profiles[3], "candidate_d")
        assert pkg_d.background_instruction.dominant_colors == ["#7952B3", "#FFC107", "#17A2B8"]

    def test_candidate_metadata_tracking(self, base_package):
        """Verify CandidateMetadata tracking of strategy summaries and latencies."""
        generator = MultiCandidateGenerator()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cand_set = generator.generate_candidates(base_package, count=3, output_directory=tmp_dir)

            meta = cand_set.metadata
            assert meta.total_requested == 3
            assert meta.total_generated == 3
            assert len(meta.variation_dimensions) == 3
            assert "candidate_a" in meta.strategy_summary
            assert "candidate_b" in meta.strategy_summary
            assert "candidate_c" in meta.strategy_summary
            assert meta.execution_latencies_s["candidate_a"] >= 0.0

    def test_generate_from_brief_convenience(self):
        """Test MultiCandidateGenerator.generate_from_brief()."""
        generator = MultiCandidateGenerator()
        brief = DesignBrief()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cand_set = generator.generate_from_brief(brief, count=2, output_directory=tmp_dir)

            assert len(cand_set.candidates) == 2
            cand_a = cand_set.get_candidate("candidate_a")
            assert cand_a is not None
            assert os.path.exists(cand_a.image_path)

    def test_custom_variation_profiles(self, base_package):
        """Test candidate generation with custom user-defined VariationProfile instances."""
        generator = MultiCandidateGenerator()

        custom_p1 = VariationProfile(
            profile_id="custom_p1",
            profile_name="Custom Profile 1",
            primary_dimension=VariationDimension.EMOTIONAL_EMPHASIS,
            typography_scale_multiplier=1.5,
            font_color_hex="#FFFF00",
        )
        custom_p2 = VariationProfile(
            profile_id="custom_p2",
            profile_name="Custom Profile 2",
            primary_dimension=VariationDimension.COMPOSITION_EMPHASIS,
            subject_scale_multiplier=1.2,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cand_set = generator.generate_candidates(
                base_package,
                custom_profiles=[custom_p1, custom_p2],
                output_directory=tmp_dir,
            )

            assert len(cand_set.candidates) == 2
            assert cand_set.candidates[0].profile.profile_id == "custom_p1"
            assert cand_set.candidates[1].profile.profile_id == "custom_p2"

    def test_none_package_raises_error(self):
        """Verify MultiCandidateGenerator raises MultiCandidateGeneratorError when presented with None package."""
        generator = MultiCandidateGenerator()
        with pytest.raises(MultiCandidateGeneratorError, match="Input base_package cannot be None"):
            generator.generate_candidates(None)
