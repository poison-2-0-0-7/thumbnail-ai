"""
decision_resolver.py
====================

Implements the deterministic KEEP/REMOVE/REPLACE/ENHANCE/ADD decision mapping
from Section 11 of Module 10 Architecture Document.
"""

from __future__ import annotations

from typing import Optional

from composition_components.interfaces import IDecisionResolver
from models import DecisionManifest, LayerDecision, LayerRole, RedesignSpecification


class DecisionResolver(IDecisionResolver):
    """
    Deterministic decision resolver mapping RedesignSpecification or DecisionManifest to LayerDecisions.
    """

    def resolve(
        self,
        spec: RedesignSpecification,
        decision_manifest: Optional[DecisionManifest] = None,
    ) -> list[tuple[str, LayerRole, LayerDecision, str]]:
        """
        Map RedesignSpecification or DecisionManifest to list of (element_key, role, decision, rationale) tuples.

        Rules (§11 & Integration §5):
        - When DecisionManifest is present and status != 'error', uses decisions from DecisionManifest.
        - Otherwise, falls back to RedesignSpecification-only mapping:
          1. Background is always REPLACE.
          2. Subject/person is ENHANCE if crop_tighter else KEEP (when has_subject=True).
          3. Object directives map action "remove" -> REMOVE, "preserve"/"include" -> KEEP.
          4. Text overlay maps include_text=True -> ADD.
        """
        if (
            decision_manifest is not None
            and getattr(decision_manifest, "status", None) != "error"
            and str(getattr(decision_manifest, "status", "")).lower() != "error"
            and getattr(decision_manifest, "decisions", None)
        ):
            manifest_decisions: list[tuple[str, LayerRole, LayerDecision, str]] = []
            for d in decision_manifest.decisions:
                if hasattr(d, "target") and d.target:
                    element_key = d.target.element_id
                    role_str = str(d.target.element_type).lower()
                else:
                    element_key = getattr(d, "target_element_key", "element")
                    role_str = str(getattr(d, "role", "object")).lower()

                action_str = (
                    d.action.value if hasattr(d.action, "value") else str(d.action)
                ).lower()

                try:
                    role = LayerRole(role_str)
                except ValueError:
                    if role_str in ("subject", "person"):
                        role = LayerRole.PERSON
                    elif role_str == "background":
                        role = LayerRole.BACKGROUND
                    elif role_str == "text":
                        role = LayerRole.TEXT
                    else:
                        role = LayerRole.OBJECT

                try:
                    decision = LayerDecision(action_str)
                except ValueError:
                    decision = LayerDecision.KEEP

                manifest_decisions.append((element_key, role, decision, d.rationale))
            return manifest_decisions

        decisions: list[tuple[str, LayerRole, LayerDecision, str]] = []

        # 1. Background
        decisions.append(
            (
                "background",
                LayerRole.BACKGROUND,
                LayerDecision.REPLACE,
                "Real background pixels are replaced by AI-generated background while structure is preserved via conditioning maps.",
            )
        )

        # 2. Person / Subject
        if spec.subject_treatment.has_subject:
            subject_decision = (
                LayerDecision.ENHANCE
                if spec.subject_treatment.crop_tighter
                else LayerDecision.KEEP
            )
            rationale = (
                spec.subject_treatment.rationale
                or f"Subject person layer treatment (crop_tighter={spec.subject_treatment.crop_tighter})."
            )
            decisions.append(("person", LayerRole.PERSON, subject_decision, rationale))

        # 3. Object directives
        for i, obj_dir in enumerate(spec.object_directives):
            key = f"object_{i}_{obj_dir.label}"
            if obj_dir.action == "remove":
                obj_decision = LayerDecision.REMOVE
            elif obj_dir.action in ("preserve", "include"):
                obj_decision = LayerDecision.KEEP
            else:
                obj_decision = LayerDecision.KEEP

            rationale = obj_dir.rationale or f"Object directive for '{obj_dir.label}': {obj_dir.action}."
            decisions.append((key, LayerRole.OBJECT, obj_decision, rationale))

        # 4. Text overlay
        if spec.text_overlay.include_text:
            rationale = (
                spec.text_overlay.rationale
                or "Text overlay geometry added for composition layout."
            )
            decisions.append(("text", LayerRole.TEXT, LayerDecision.ADD, rationale))

        return decisions
