"""
weakness_analyzer.py
====================

Actionable Thumbnail Weakness & Risk Detector (Phase 11).

Scans composition, hierarchy, scene graph, and contrast metrics to pinpoint
concrete design flaws (clutter, weak focal point, text overlap, poor separation,
small subjects, background distraction, identity risk).
"""

from __future__ import annotations

from models import CompositionAnalysis
from thumbnail_understanding.schemas import (
    ElementType,
    SceneGraph,
    VisualHierarchy,
    WeaknessAnalysis,
    WeaknessFinding,
)


class WeaknessAnalyzer:
    """Detects visual weaknesses and risks with actionable recommendations."""

    @classmethod
    def analyze_weaknesses(
        cls,
        scene_graph: SceneGraph,
        hierarchy: VisualHierarchy,
        composition: CompositionAnalysis,
    ) -> WeaknessAnalysis:
        """Scan thumbnail parameters and return structured weakness findings."""
        findings: list[WeaknessFinding] = []

        # 1. Visual Clutter Check
        if composition.clutter_score > 0.4:
            findings.append(
                WeaknessFinding(
                    weakness_type="clutter",
                    severity="high" if composition.clutter_score > 0.7 else "medium",
                    confidence=0.9,
                    evidence=f"High visual clutter score of {composition.clutter_score:.2f} across frame",
                    affected_element_ids=[e.element_id for e in scene_graph.elements if e.role.value == "supporting"],
                    recommended_correction="Remove low-value background object clutter and simplify backdrop",
                )
            )

        # 2. Weak Focal Point / Hierarchy
        if hierarchy.focal_strength_score < 0.3:
            findings.append(
                WeaknessFinding(
                    weakness_type="weak_focal_point",
                    severity="high",
                    confidence=0.85,
                    evidence=f"Low focal strength score ({hierarchy.focal_strength_score:.2f}); visual attention is split",
                    affected_element_ids=scene_graph.primary_subject_ids,
                    recommended_correction="Enlarge primary hero subject and increase local contrast against background",
                )
            )

        # 3. Subject / Background Separation
        if hierarchy.subject_separation_score < 0.4:
            findings.append(
                WeaknessFinding(
                    weakness_type="poor_separation",
                    severity="medium",
                    confidence=0.8,
                    evidence="Primary subject blends into background without clear contrast boundary",
                    affected_element_ids=[scene_graph.hero_element_id] if scene_graph.hero_element_id else [],
                    recommended_correction="Apply rim lighting, background desaturation, or outline enhancement to isolate hero",
                )
            )

        # 4. Text Overlap Subject
        if composition.text_overlaps_subject:
            text_elems = [e.element_id for e in scene_graph.elements if e.element_type == ElementType.TEXT]
            findings.append(
                WeaknessFinding(
                    weakness_type="text_overlap",
                    severity="high",
                    confidence=0.95,
                    evidence="OCR text region directly overlaps primary subject bounding box",
                    affected_element_ids=text_elems,
                    recommended_correction="Relocate typography into reserved negative space away from creator face",
                )
            )

        # 5. Small Subjects Check
        if scene_graph.hero_element_id:
            hero = next((e for e in scene_graph.elements if e.element_id == scene_graph.hero_element_id), None)
            if hero:
                hero_area = (hero.bbox.x_max - hero.bbox.x_min) * (hero.bbox.y_max - hero.bbox.y_min)
                if hero_area < 0.08:
                    findings.append(
                        WeaknessFinding(
                            weakness_type="small_subjects",
                            severity="medium",
                            confidence=0.9,
                            evidence=f"Primary hero subject occupies only {hero_area * 100:.1f}% of frame area",
                            affected_element_ids=[hero.element_id],
                            recommended_correction="Crop tighter or scale primary subject up for mobile thumbnail readability",
                        )
                    )

        overall_risk = "low"
        if any(f.severity in ("high", "critical") for f in findings):
            overall_risk = "high"
        elif len(findings) >= 2:
            overall_risk = "medium"

        return WeaknessAnalysis(
            findings=findings,
            overall_risk_level=overall_risk,
        )
