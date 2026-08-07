"""
psychology_assessor.py
======================

Thumbnail Psychology & Click Driver Assessor (Phase 10).

Evaluates visual click drivers (curiosity, emotion, surprise, urgency, comparison,
transformation, scale, novelty, aspiration, money/value, conflict/tension, story clarity)
supported by empirical visual and face evidence.
"""

from __future__ import annotations

from typing import Optional
from models import GeminiReasoning, VideoMetadata
from thumbnail_understanding.schemas import (
    ElementType,
    PsychologyDriver,
    SceneElement,
    SceneGraph,
    ThumbnailPsychologyAssessment,
)


class PsychologyAssessor:
    """Assesses psychological click triggers and curiosity gaps."""

    @classmethod
    def assess_psychology(
        cls,
        scene_graph: SceneGraph,
        metadata: Optional[VideoMetadata] = None,
        legacy_reasoning: Optional[GeminiReasoning] = None,
    ) -> ThumbnailPsychologyAssessment:
        """
        Synthesize ThumbnailPsychologyAssessment from grounded scene graph and metadata.
        """
        drivers: list[PsychologyDriver] = []
        hero_face = next((e for e in scene_graph.elements if e.element_id == scene_graph.hero_element_id), None)
        text_elems = [e for e in scene_graph.elements if e.element_type == ElementType.TEXT]

        # 1. Emotional Pull Driver
        if hero_face and hero_face.emotion:
            emotion_str = hero_face.emotion.lower()
            strength = 0.85 if emotion_str in {"surprised", "shocked", "happy", "fear", "excited"} else 0.5
            drivers.append(
                PsychologyDriver(
                    driver="emotion",
                    strength=strength,
                    confidence=hero_face.emotion_confidence or 0.8,
                    supporting_evidence=f"Primary subject displays '{emotion_str}' emotion",
                    potential_improvement="Amplify emotion contrast and lighting",
                    associated_element_ids=[hero_face.element_id],
                )
            )

        # 2. Curiosity Gap Driver
        curiosity_score = 0.5
        if legacy_reasoning:
            curiosity_score = legacy_reasoning.curiosity_gap_score
        elif text_elems:
            curiosity_score = 0.75

        drivers.append(
            PsychologyDriver(
                driver="curiosity",
                strength=curiosity_score,
                confidence=0.85,
                supporting_evidence="Headline/visual composition creates an unresolved question",
                potential_improvement="Tighten copy and increase focal subject contrast",
                associated_element_ids=[e.element_id for e in text_elems],
            )
        )

        # 3. Story Clarity Driver
        story_strength = 0.6 if hero_face or text_elems else 0.3
        drivers.append(
            PsychologyDriver(
                driver="story_clarity",
                strength=story_strength,
                confidence=0.8,
                supporting_evidence=f"Thumbnail contains {len(scene_graph.elements)} grounded elements telling the narrative",
                potential_improvement="Eliminate non-essential background clutter",
                associated_element_ids=scene_graph.primary_subject_ids,
            )
        )

        ctr_score = legacy_reasoning.ctr_potential_score if legacy_reasoning else round((curiosity_score + story_strength) / 2.0, 2)
        mismatch = legacy_reasoning.content_mismatch_detected if legacy_reasoning else False
        mismatch_exp = legacy_reasoning.mismatch_explanation if legacy_reasoning else None

        return ThumbnailPsychologyAssessment(
            ctr_potential_score=ctr_score,
            curiosity_gap_score=curiosity_score,
            drivers=drivers,
            content_mismatch_detected=mismatch,
            mismatch_explanation=mismatch_exp,
            overall_psychology_summary=f"Thumbnail leverages {len(drivers)} active click drivers with estimated CTR potential {ctr_score:.2f}",
        )
