"""
rules.py
========

Deterministic, Rule-Based Issue Detection & Suggestion Generation for Phase 5.4.
No LLMs. Evaluates 22 quality metrics against thresholds and emits structured Issues & ImprovementSuggestions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from thumbnail_intelligence.evaluation.models import EvaluationMetric, EvaluationResult
from thumbnail_intelligence.critique.models import (
    CritiqueProfile,
    ImpactLevel,
    ImplementationCost,
    ImprovementSuggestion,
    Issue,
    IssueSeverity,
)


class CritiqueRule:
    """Evaluates a specific evaluation metric and returns detected Issue and ImprovementSuggestion if applicable."""

    def __init__(
        self,
        metric_name: str,
        critical_threshold: float,
        major_threshold: float,
        action_type: str,
        target_element: str,
        suggested_fix: str,
        parameter_changes: Dict,
        expected_ctr_gain: float,
        visual_impact: ImpactLevel,
        implementation_cost: ImplementationCost,
    ) -> None:
        self.metric_name = metric_name
        self.critical_threshold = critical_threshold
        self.major_threshold = major_threshold
        self.action_type = action_type
        self.target_element = target_element
        self.suggested_fix = suggested_fix
        self.parameter_changes = parameter_changes
        self.expected_ctr_gain = expected_ctr_gain
        self.visual_impact = visual_impact
        self.implementation_cost = implementation_cost

    def evaluate(self, metric: EvaluationMetric, profile: CritiqueProfile) -> Tuple[Issue, ImprovementSuggestion]:
        """Evaluate metric against thresholds and return (Issue, ImprovementSuggestion)."""
        score = metric.score
        crit_thresh = profile.critical_threshold_score
        maj_thresh = profile.major_threshold_score

        if score < crit_thresh:
            severity = IssueSeverity.CRITICAL
        elif score < maj_thresh:
            severity = IssueSeverity.MAJOR
        else:
            severity = IssueSeverity.MINOR

        issue = Issue(
            issue_id=f"issue_{self.metric_name}",
            metric_name=self.metric_name,
            severity=severity,
            confidence=metric.confidence,
            affected_region=self.target_element,
            reason=metric.reason,
            evidence=metric.evidence,
            suggested_fix=self.suggested_fix,
            estimated_impact=self.visual_impact,
        )

        suggestion = ImprovementSuggestion(
            suggestion_id=f"sug_{self.metric_name}",
            action_type=self.action_type,
            description=self.suggested_fix,
            target_element=self.target_element,
            parameter_changes=self.parameter_changes,
            expected_ctr_gain=self.expected_ctr_gain,
            visual_impact=self.visual_impact,
            implementation_cost=self.implementation_cost,
            confidence=metric.confidence,
            priority_score=0.0,
        )

        return issue, suggestion


class DefaultCritiqueRuleSet:
    """Collection of 22 deterministic critique rules covering all evaluation metrics."""

    @staticmethod
    def get_rules() -> Dict[str, CritiqueRule]:
        """Return dictionary mapping metric_name to CritiqueRule."""
        return {
            "face_visibility": CritiqueRule(
                metric_name="face_visibility",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="unobscure_face",
                target_element="hero_subject",
                suggested_fix="Ensure subject face is un-obscured by overlays",
                parameter_changes={"z_index_override": 15},
                expected_ctr_gain=6.5,
                visual_impact=ImpactLevel.HIGH,
                implementation_cost=ImplementationCost.LOW,
            ),
            "face_size": CritiqueRule(
                metric_name="face_size",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="scale_subject",
                target_element="hero_subject",
                suggested_fix="Increase face scale by 15%",
                parameter_changes={"subject_scale_multiplier": 1.15},
                expected_ctr_gain=5.0,
                visual_impact=ImpactLevel.HIGH,
                implementation_cost=ImplementationCost.LOW,
            ),
            "face_position": CritiqueRule(
                metric_name="face_position",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="position_subject",
                target_element="hero_subject",
                suggested_fix="Re-center face along top-third focal intersection",
                parameter_changes={"subject_y_offset_pct": -0.05},
                expected_ctr_gain=3.5,
                visual_impact=ImpactLevel.MEDIUM,
                implementation_cost=ImplementationCost.LOW,
            ),
            "eye_contact": CritiqueRule(
                metric_name="eye_contact",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="adjust_pose",
                target_element="hero_subject",
                suggested_fix="Adjust subject pose toward direct viewer gaze",
                parameter_changes={},
                expected_ctr_gain=2.5,
                visual_impact=ImpactLevel.MEDIUM,
                implementation_cost=ImplementationCost.MEDIUM,
            ),
            "emotion_strength": CritiqueRule(
                metric_name="emotion_strength",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="boost_key_lighting",
                target_element="lighting",
                suggested_fix="Strengthen emotional key lighting intensity",
                parameter_changes={"key_light_intensity_multiplier": 1.25},
                expected_ctr_gain=4.5,
                visual_impact=ImpactLevel.HIGH,
                implementation_cost=ImplementationCost.LOW,
            ),
            "text_readability": CritiqueRule(
                metric_name="text_readability",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="increase_font_size",
                target_element="headline_text",
                suggested_fix="Increase headline font size to at least 48px",
                parameter_changes={"typography_scale_multiplier": 1.25},
                expected_ctr_gain=7.0,
                visual_impact=ImpactLevel.HIGH,
                implementation_cost=ImplementationCost.LOW,
            ),
            "font_contrast": CritiqueRule(
                metric_name="font_contrast",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="boost_font_contrast",
                target_element="headline_text",
                suggested_fix="Increase text stroke width and switch fill/stroke to high-contrast colors",
                parameter_changes={"stroke_width_multiplier": 1.5, "font_color_hex": "#FFFFFF", "stroke_color_hex": "#000000"},
                expected_ctr_gain=6.0,
                visual_impact=ImpactLevel.HIGH,
                implementation_cost=ImplementationCost.LOW,
            ),
            "subject_saliency": CritiqueRule(
                metric_name="subject_saliency",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="boost_saliency",
                target_element="hero_subject",
                suggested_fix="Improve subject saliency with rim lighting",
                parameter_changes={"rim_light_enabled_override": True},
                expected_ctr_gain=5.5,
                visual_impact=ImpactLevel.HIGH,
                implementation_cost=ImplementationCost.LOW,
            ),
            "visual_hierarchy": CritiqueRule(
                metric_name="visual_hierarchy",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="adjust_hierarchy",
                target_element="canvas",
                suggested_fix="Enlarge hero subject and adjust z-index layering",
                parameter_changes={"subject_scale_multiplier": 1.1},
                expected_ctr_gain=4.0,
                visual_impact=ImpactLevel.MEDIUM,
                implementation_cost=ImplementationCost.LOW,
            ),
            "rule_of_thirds": CritiqueRule(
                metric_name="rule_of_thirds",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="align_grid",
                target_element="hero_subject",
                suggested_fix="Shift hero subject anchor to rule-of-thirds grid line",
                parameter_changes={"subject_x_offset_pct": 0.05},
                expected_ctr_gain=3.0,
                visual_impact=ImpactLevel.MEDIUM,
                implementation_cost=ImplementationCost.LOW,
            ),
            "negative_space": CritiqueRule(
                metric_name="negative_space",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="increase_negative_space",
                target_element="canvas",
                suggested_fix="Increase unoccupied negative space",
                parameter_changes={"typography_scale_multiplier": 0.95},
                expected_ctr_gain=3.5,
                visual_impact=ImpactLevel.MEDIUM,
                implementation_cost=ImplementationCost.LOW,
            ),
            "composition_balance": CritiqueRule(
                metric_name="composition_balance",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="rebalance_composition",
                target_element="canvas",
                suggested_fix="Rebalance left-right luminance moments",
                parameter_changes={},
                expected_ctr_gain=2.5,
                visual_impact=ImpactLevel.MEDIUM,
                implementation_cost=ImplementationCost.LOW,
            ),
            "background_clutter": CritiqueRule(
                metric_name="background_clutter",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="reduce_clutter",
                target_element="background",
                suggested_fix="Reduce background clutter",
                parameter_changes={"background_style_direction": "clean minimal studio contrast background"},
                expected_ctr_gain=4.5,
                visual_impact=ImpactLevel.HIGH,
                implementation_cost=ImplementationCost.LOW,
            ),
            "color_harmony": CritiqueRule(
                metric_name="color_harmony",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="simplify_colors",
                target_element="background",
                suggested_fix="Simplify color palette to 3–4 dominant hues",
                parameter_changes={"dominant_colors_override": ["#1A1A2E", "#E94560", "#FFFFFF"]},
                expected_ctr_gain=3.0,
                visual_impact=ImpactLevel.MEDIUM,
                implementation_cost=ImplementationCost.LOW,
            ),
            "color_contrast": CritiqueRule(
                metric_name="color_contrast",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="boost_color_contrast",
                target_element="canvas",
                suggested_fix="Increase dynamic range and color saturation",
                parameter_changes={},
                expected_ctr_gain=5.0,
                visual_impact=ImpactLevel.HIGH,
                implementation_cost=ImplementationCost.LOW,
            ),
            "brand_preservation": CritiqueRule(
                metric_name="brand_preservation",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="enforce_safe_zones",
                target_element="headline_text",
                suggested_fix="Move elements away from margin safe zones",
                parameter_changes={},
                expected_ctr_gain=2.0,
                visual_impact=ImpactLevel.LOW,
                implementation_cost=ImplementationCost.LOW,
            ),
            "object_separation": CritiqueRule(
                metric_name="object_separation",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="add_rim_lighting",
                target_element="hero_subject",
                suggested_fix="Add rim lighting to subject boundary",
                parameter_changes={"rim_light_enabled_override": True},
                expected_ctr_gain=4.0,
                visual_impact=ImpactLevel.MEDIUM,
                implementation_cost=ImplementationCost.LOW,
            ),
            "typography_quality": CritiqueRule(
                metric_name="typography_quality",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="reduce_text_length",
                target_element="headline_text",
                suggested_fix="Reduce text to 4 words or fewer",
                parameter_changes={},
                expected_ctr_gain=4.5,
                visual_impact=ImpactLevel.HIGH,
                implementation_cost=ImplementationCost.LOW,
            ),
            "thumbnail_clarity": CritiqueRule(
                metric_name="thumbnail_clarity",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="sharpen_image",
                target_element="canvas",
                suggested_fix="Increase image sharpening filter",
                parameter_changes={},
                expected_ctr_gain=4.0,
                visual_impact=ImpactLevel.MEDIUM,
                implementation_cost=ImplementationCost.LOW,
            ),
            "visual_simplicity": CritiqueRule(
                metric_name="visual_simplicity",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="simplify_elements",
                target_element="canvas",
                suggested_fix="Reduce total canvas element count",
                parameter_changes={},
                expected_ctr_gain=3.0,
                visual_impact=ImpactLevel.MEDIUM,
                implementation_cost=ImplementationCost.LOW,
            ),
            "mobile_readability": CritiqueRule(
                metric_name="mobile_readability",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="optimize_mobile_readability",
                target_element="headline_text",
                suggested_fix="Increase text font size for 120x68 px mobile preview",
                parameter_changes={"typography_scale_multiplier": 1.2},
                expected_ctr_gain=5.5,
                visual_impact=ImpactLevel.HIGH,
                implementation_cost=ImplementationCost.LOW,
            ),
            "estimated_ctr_score": CritiqueRule(
                metric_name="estimated_ctr_score",
                critical_threshold=50.0,
                major_threshold=70.0,
                action_type="boost_ctr",
                target_element="canvas",
                suggested_fix="Boost subject saliency and headline contrast",
                parameter_changes={"subject_scale_multiplier": 1.1, "stroke_width_multiplier": 1.4},
                expected_ctr_gain=6.5,
                visual_impact=ImpactLevel.HIGH,
                implementation_cost=ImplementationCost.LOW,
            ),
        }
