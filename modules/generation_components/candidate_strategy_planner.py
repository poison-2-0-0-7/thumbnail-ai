"""CandidateStrategyPlanner: derives bounded per-candidate PromptPackage transformations."""

from __future__ import annotations

import sys
from pathlib import Path

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from loguru import logger

from models import CandidateStrategy, DesignBlueprint, PromptPackage


class CandidateStrategyPlanner:
    """Derive a per-candidate PromptPackage deterministically from a base package and strategy."""

    def derive_package(
        self,
        base_package: PromptPackage,
        blueprint: DesignBlueprint | None,
        strategy: CandidateStrategy,
        candidate_index: int,
    ) -> PromptPackage:
        """Derive candidate PromptPackage applying strategy perturbations bounded by DesignBlueprint."""
        cand_seed = base_package.generation_parameters.seed + candidate_index
        cand_params = base_package.generation_parameters.model_copy(update={"seed": cand_seed})

        if strategy.name == "faithful" or blueprint is None:
            return base_package.model_copy(update={"generation_parameters": cand_params})

        subject_inst = base_package.subject_instructions
        bg_inst = base_package.background_instructions
        comp_inst = base_package.composition_instructions
        lighting_inst = base_package.lighting_instructions
        color_inst = base_package.color_instructions
        typo_inst = base_package.typography_instructions
        obj_placement = list(base_package.object_placement)

        # 1. Camera distance / framing perturbation
        if strategy.camera_distance_shift > 0:
            comp_inst += " Framing is stepped slightly wider with additional negative space around the subject."
        elif strategy.camera_distance_shift < 0:
            comp_inst += " Framing is stepped slightly closer around the primary subject."

        if strategy.framing_bias > 0:
            comp_inst += " Tighten framing to emphasize key visual focal points."
        elif strategy.framing_bias < 0:
            comp_inst += " Expand framing to incorporate wider contextual background."

        # 2. Object emphasis bias
        if strategy.object_emphasis_bias != 0.0 and obj_placement:
            new_placement = []
            for item in obj_placement:
                if strategy.object_emphasis_bias > 0 and ("preserve" in item or "include" in item):
                    new_placement.append(f"{item} (emphasized scale +{int(strategy.object_emphasis_bias * 100)}%)")
                elif strategy.object_emphasis_bias < 0 and ("remove" not in item):
                    new_placement.append(f"{item} (reduced scale -{int(abs(strategy.object_emphasis_bias) * 100)}%)")
                else:
                    new_placement.append(item)
            obj_placement = new_placement

        # 3. Background intensity bias
        if strategy.background_intensity_bias < 0:
            bg_inst += " Simplify background details and soften visual distractions."
        elif strategy.background_intensity_bias > 0:
            bg_inst += " Enhance background contrast and texture intensity."

        # 4. Color grade bias
        if strategy.color_grade_bias > 0:
            color_inst += ", with moderately increased contrast, saturation, and color punch."
        elif strategy.color_grade_bias < 0:
            color_inst += ", with softer contrast and natural color saturation."

        # 5. Typography weight bias
        if strategy.typography_weight_bias > 0:
            typo_inst += " Use bolder stroke emphasis and heavy weight for text overlay."
        elif strategy.typography_weight_bias < 0:
            typo_inst += " Use clean, balanced weight for text overlay."

        # 6. Emotion bias
        if strategy.emotion_bias > 0:
            subject_inst += " Amplify subject facial expression with high emotion and strong reaction."

        # 7. Lighting bias
        if strategy.lighting_bias > 0:
            lighting_inst += " Use dramatic, high-contrast directional key and rim lighting."
        elif strategy.lighting_bias < 0:
            lighting_inst += " Use soft, diffused, balanced ambient lighting."

        # Re-assemble positive prompt while respecting invariant section order
        compiled_parts = [
            subject_inst,
            bg_inst,
            comp_inst,
            lighting_inst,
            color_inst,
            typo_inst,
        ]
        if "Preserve: " in base_package.positive_prompt:
            preserve_part = base_package.positive_prompt.split("Preserve: ")[-1]
            compiled_parts.append("Preserve: " + preserve_part)

        positive_prompt = " ".join(compiled_parts)

        derived = base_package.model_copy(
            update={
                "positive_prompt": positive_prompt,
                "subject_instructions": subject_inst,
                "background_instructions": bg_inst,
                "composition_instructions": comp_inst,
                "lighting_instructions": lighting_inst,
                "color_instructions": color_inst,
                "typography_instructions": typo_inst,
                "object_placement": obj_placement,
                "generation_parameters": cand_params,
            }
        )

        logger.debug(
            "Derived PromptPackage candidate idx={idx} strategy={strategy} seed={seed}",
            idx=candidate_index,
            strategy=strategy.name,
            seed=cand_seed,
        )
        return derived
