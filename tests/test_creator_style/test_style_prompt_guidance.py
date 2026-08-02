"""Tests for Module 10 StylePromptGuidanceGenerator component."""

from __future__ import annotations

import pytest
from modules.creator_style.style_prompt_guidance import StylePromptGuidanceGenerator
from modules.models import ThumbnailStyleSignature


def test_style_prompt_guidance_disabled():
    sig = ThumbnailStyleSignature(
        video_id="v1",
        channel_id="UC123",
        dominant_colors=["#FF0000"],
    )
    guidance = StylePromptGuidanceGenerator.generate_guidance(
        channel_id="UC123",
        signature=sig,
        enabled=False,
    )
    assert guidance.applied is False
    assert guidance.color_guidance == ""


def test_style_prompt_guidance_enabled():
    sig = ThumbnailStyleSignature(
        video_id="v1",
        channel_id="UC123",
        dominant_colors=["#FF0000", "#00FF00"],
        warm_or_cool="warm",
        subject_placement="left",
        negative_space_ratio=0.40,
        face_scale_ratio=0.40,
    )
    guidance = StylePromptGuidanceGenerator.generate_guidance(
        channel_id="UC123",
        signature=sig,
        enabled=True,
    )
    assert guidance.applied is True
    assert "#FF0000" in guidance.color_guidance
    assert "warm" in guidance.color_guidance
    assert "left" in guidance.composition_guidance
    assert guidance.face_scale_guidance is not None
    assert "prominent close-up" in guidance.face_scale_guidance
