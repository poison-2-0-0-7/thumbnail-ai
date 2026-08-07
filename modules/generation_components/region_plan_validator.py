"""Module 7 V2 Generation Component — RegionPlanValidator.

Converts DecisionManifest decisions and CompositionWorkspace masks into a concrete, frozen EditPlan.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import (
    DECISION_ENGINE_ENABLED,
    MODULE7_V2_DENOISE_BY_DECISION,
    MODULE7_V2_STEPS_BY_DECISION,
)
from models import (
    CompositionWorkspace,
    DecisionManifest,
    EditPlan,
    EditRegion,
    GenerationPlan,
)


def utc_now() -> str:
    """Return ISO-formatted UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


class RegionPlanValidator:
    """Validator and deterministic mapper from element decisions to concrete EditPlans."""

    def classify(
        self,
        video_id: str,
        decision_manifest: DecisionManifest | None = None,
        workspace: CompositionWorkspace | None = None,
        generation_plan: GenerationPlan | None = None,
    ) -> EditPlan:
        """Deterministically map decisions and workspace masks into a concrete EditPlan."""
        if not DECISION_ENGINE_ENABLED or decision_manifest is None or not decision_manifest.decisions:
            logger.info(
                "RegionPlanValidator: No decision manifest or engine disabled for video_id={vid}",
                vid=video_id,
            )
            return EditPlan(
                video_id=video_id,
                edit_scope="none",
                regions=[],
                fallback_elements=[],
                created_at=utc_now(),
            )

        regions: list[EditRegion] = []
        fallback_elements: list[dict[str, str]] = []

        for dec in decision_manifest.decisions:
            elem_id = dec.target.element_id
            elem_type = dec.target.element_type.lower()
            action_str = dec.action.value.lower() if hasattr(dec.action, "value") else str(dec.action).lower()

            if action_str == "keep":
                continue

            is_bg = elem_type in ("background", "scene_background", "background_layer") or "background" in elem_id.lower()
            stage: Literal["background", "object"] = "background" if is_bg else "object"

            mask_path: Path | None = None
            if workspace is not None:
                if hasattr(workspace, "role_mask_paths") and workspace.role_mask_paths:
                    for k, v in workspace.role_mask_paths.items():
                        if k in elem_id or elem_id in k or (is_bg and "background" in k):
                            mask_path = Path(v)
                            break
                if mask_path is None and hasattr(workspace, "layers") and workspace.layers:
                    for layer in workspace.layers:
                        layer_id = getattr(layer, "layer_id", "")
                        placement = getattr(layer, "placement", None)
                        asset_id = getattr(placement, "asset_id", "") if placement else ""
                        if layer_id == elem_id or asset_id == elem_id or (is_bg and "background" in layer_id):
                            if getattr(layer, "mask_path", None):
                                mask_path = Path(layer.mask_path)
                            elif placement and getattr(placement, "mask", None) and getattr(placement.mask, "mask_path", None):
                                mask_path = Path(placement.mask.mask_path)
                            elif is_bg and placement and getattr(placement, "source_path", None):
                                mask_path = Path(placement.source_path)
                            break

            if mask_path is None or not mask_path.is_file():
                logger.warning(
                    "RegionPlanValidator: Unresolvable mask for decision={action} on element={elem_id}; falling back to KEEP",
                    action=action_str,
                    elem_id=elem_id,
                )
                fallback_elements.append({
                    "element_id": elem_id,
                    "reason": "unresolvable_mask",
                    "action": action_str,
                })
                continue

            denoise = MODULE7_V2_DENOISE_BY_DECISION.get(action_str, 0.85)
            steps = MODULE7_V2_STEPS_BY_DECISION.get(action_str, 25)

            region = EditRegion(
                element_id=elem_id,
                decision_type=action_str,  # type: ignore[arg-type]
                mask_path=mask_path,
                denoise_strength=denoise,
                steps=steps,
                stage=stage,
            )
            regions.append(region)

        has_bg = any(r.stage == "background" for r in regions)
        has_obj = any(r.stage == "object" for r in regions)

        if has_bg and has_obj:
            scope: Literal["none", "background_only", "object_only", "heavy_redesign"] = "heavy_redesign"
        elif has_bg:
            scope = "background_only"
        elif has_obj:
            scope = "object_only"
        else:
            scope = "none"

        logger.info(
            "RegionPlanValidator: Classified video_id={vid} as edit_scope={scope} with {n_reg} edit regions",
            vid=video_id,
            scope=scope,
            n_reg=len(regions),
        )

        return EditPlan(
            video_id=video_id,
            edit_scope=scope,
            regions=regions,
            fallback_elements=fallback_elements,
            created_at=utc_now(),
        )
