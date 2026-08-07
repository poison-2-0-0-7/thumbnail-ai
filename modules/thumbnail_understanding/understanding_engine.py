"""
understanding_engine.py
========================

Thumbnail Understanding Engine V2 Main Orchestrator (Phase 1).

Consumes raw ThumbnailIntelligence (CV detections & Gemini reasoning) and optional
Module 8 AssetExtractionManifest, running grounded element building, subject hierarchy,
relationship graph, psychology assessment, weakness analysis, scene decomposition, and AI Director planning
into a unified, strongly-typed ThumbnailUnderstanding artifact.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger

from config import PROJECT_ROOT
from models import (
    AssetExtractionManifest,
    ThumbnailIntelligence,
    VideoMetadata,
)
from thumbnail_understanding.schemas import CompositionIntelligence, ThumbnailUnderstanding
from thumbnail_understanding.director_engine import AIThumbnailDirector
from thumbnail_understanding.hierarchy_calculator import HierarchyCalculator
from thumbnail_understanding.mask_validator import MaskValidator
from thumbnail_understanding.psychology_assessor import PsychologyAssessor
from thumbnail_understanding.relationship_reasoner import RelationshipReasoner
from thumbnail_understanding.scene_decomposer import SceneDecomposer
from thumbnail_understanding.scene_grounding import SceneGrounder
from thumbnail_understanding.weakness_analyzer import WeaknessAnalyzer

DEFAULT_UNDERSTANDING_DIR: Path = PROJECT_ROOT / "data" / "thumbnail_understanding"


class ThumbnailUnderstandingEngine:
    """Main Orchestrator for Thumbnail Understanding V2."""

    def __init__(self, output_dir: Path = DEFAULT_UNDERSTANDING_DIR) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def understand(
        self,
        intelligence: ThumbnailIntelligence,
        asset_manifest: Optional[AssetExtractionManifest] = None,
        metadata: Optional[VideoMetadata] = None,
    ) -> ThumbnailUnderstanding:
        """
        Execute complete structured Thumbnail Understanding pipeline.
        """
        logger.info("ThumbnailUnderstandingEngine starting for video_id={vid}", vid=intelligence.video_id)

        # 1. Deterministic Scene Grounding
        grounded_elements = SceneGrounder.ground_elements(intelligence)

        # 2. Subject & Visual Hierarchy
        scene_graph, hierarchy = HierarchyCalculator.compute_hierarchy(
            grounded_elements, intelligence.composition
        )

        # 3. Spatial & Relational Reasoner
        relationships = RelationshipReasoner.analyze_relationships(scene_graph.elements)
        scene_graph = scene_graph.model_copy(update={"relationships": relationships})

        # 4. Psychology & Click Driver Assessment
        psychology = PsychologyAssessor.assess_psychology(
            scene_graph=scene_graph,
            metadata=metadata,
            legacy_reasoning=intelligence.reasoning,
        )

        # 5. Weakness & Risk Analysis
        weaknesses = WeaknessAnalyzer.analyze_weaknesses(
            scene_graph=scene_graph,
            hierarchy=hierarchy,
            composition=intelligence.composition,
        )

        # 6. Scene Layer Decomposition
        decomposed_scene = SceneDecomposer.decompose_scene(
            source_thumbnail_path=intelligence.thumbnail_path,
            scene_graph=scene_graph,
            asset_manifest=asset_manifest,
        )

        # 7. AI Director & Professional Improvement Planner
        director_plan, improvement_plan = AIThumbnailDirector.generate_plan(
            scene_graph=scene_graph,
            hierarchy=hierarchy,
            composition=intelligence.composition,
            psychology=psychology,
            weaknesses=weaknesses,
            decomposed_scene=decomposed_scene,
        )

        # Convert CompositionAnalysis to CompositionIntelligence
        comp_intel = CompositionIntelligence(
            rule_of_thirds_score=intelligence.composition.rule_of_thirds_score,
            subject_placement=intelligence.composition.subject_placement,
            cropping_quality=0.8,
            balance=intelligence.composition.balance_score,
            symmetry_asymmetry="asymmetric" if intelligence.composition.symmetry_score < 0.7 else "symmetric",
            negative_space=intelligence.composition.negative_space_ratio,
            depth_layers_count=len(scene_graph.elements),
            camera_perspective="medium_shot",
            subject_background_separation=round(1.0 - intelligence.composition.clutter_score, 2),
            contrast_distribution=0.7,
            lighting_distribution=0.7,
            visual_density=intelligence.composition.clutter_score,
            actionable_findings=[],
        )

        now_iso = datetime.now(timezone.utc).isoformat()

        understanding = ThumbnailUnderstanding(
            video_id=intelligence.video_id,
            source_thumbnail_path=intelligence.thumbnail_path,
            scene_graph=scene_graph,
            hierarchy=hierarchy,
            composition=comp_intel,
            psychology=psychology,
            weaknesses=weaknesses,
            decomposed_scene=decomposed_scene,
            director_plan=director_plan,
            improvement_plan=improvement_plan,
            legacy_reasoning=intelligence.reasoning,
            status="success",
            analyzed_at=now_iso,
        )

        self.save_understanding(understanding)
        logger.info(
            "ThumbnailUnderstanding complete for video_id={vid}: {elements} elements, {layers} layers, {actions} actions",
            vid=intelligence.video_id,
            elements=len(scene_graph.elements),
            layers=decomposed_scene.layer_count,
            actions=len(improvement_plan.actions),
        )
        return understanding

    def save_understanding(self, understanding: ThumbnailUnderstanding) -> Path:
        """Persist ThumbnailUnderstanding to disk as JSON."""
        target_path = self.output_dir / f"{understanding.video_id}.json"
        content = understanding.model_dump_json(indent=2)
        target_path.write_text(content, encoding="utf-8")
        logger.debug("Saved ThumbnailUnderstanding to {path}", path=target_path)
        return target_path

    def load_understanding(self, video_id: str) -> Optional[ThumbnailUnderstanding]:
        """Load ThumbnailUnderstanding from disk if present."""
        target_path = self.output_dir / f"{video_id}.json"
        if not target_path.is_file():
            return None
        try:
            content = target_path.read_text(encoding="utf-8")
            return ThumbnailUnderstanding.model_validate_json(content)
        except Exception as exc:
            logger.warning("Failed to load ThumbnailUnderstanding for video_id={vid}: {exc}", vid=video_id, exc=exc)
            return None
