"""
StylePromptGuidance component for Phase 4 of Module 10 Creator Style Learning.

Generates deterministic style prompt guidance strings based on creator's ThumbnailStyleSignature.
Integrates into Strategy Engine / Prompt Compiler without altering prompt compilation logic.
"""

from __future__ import annotations

from typing import Optional

from modules.config import MODULE10_STYLE_PROMPT_ENABLED
from modules.models import StylePromptGuidance, ThumbnailStyleSignature


class StylePromptGuidanceGenerator:
    """
    Generates deterministic style guidance blocks from creator ThumbnailStyleSignatures.
    """

    @staticmethod
    def generate_guidance(
        channel_id: str,
        signature: Optional[ThumbnailStyleSignature] = None,
        enabled: bool = MODULE10_STYLE_PROMPT_ENABLED,
    ) -> StylePromptGuidance:
        """
        Generate StylePromptGuidance record for a creator.
        If disabled or signature is None, returns empty/unapplied guidance block.
        """
        if not enabled or signature is None:
            return StylePromptGuidance(
                channel_id=channel_id,
                color_guidance="",
                composition_guidance="",
                face_scale_guidance=None,
                applied=False,
            )

        # 1. Color Guidance
        color_parts = []
        if signature.dominant_colors:
            colors_str = ", ".join(signature.dominant_colors[:3])
            color_parts.append(f"Maintain creator signature color palette ({colors_str})")
        if signature.warm_or_cool and signature.warm_or_cool != "neutral":
            color_parts.append(f"use overall {signature.warm_or_cool} lighting tone")
        color_guidance = "; ".join(color_parts) if color_parts else "Preserve creator color harmony"

        # 2. Composition Guidance
        comp_parts = []
        if signature.subject_placement:
            comp_parts.append(f"Position key subject towards {signature.subject_placement}")
        if signature.negative_space_ratio > 0.35:
            comp_parts.append("reserve ample clean negative space")
        elif signature.negative_space_ratio < 0.20:
            comp_parts.append("use high-density compositional framing")
        composition_guidance = "; ".join(comp_parts) if comp_parts else "Follow creator composition framing"

        # 3. Face Scale Guidance
        face_scale_guidance = None
        if signature.face_scale_ratio is not None:
            if signature.face_scale_ratio > 0.35:
                face_scale_guidance = "Maintain prominent close-up face framing ratio"
            elif signature.face_scale_ratio < 0.15:
                face_scale_guidance = "Use medium or wide environmental face framing ratio"
            else:
                face_scale_guidance = "Use balanced medium face framing ratio"

        return StylePromptGuidance(
            channel_id=channel_id,
            color_guidance=color_guidance,
            composition_guidance=composition_guidance,
            face_scale_guidance=face_scale_guidance,
            applied=True,
        )
