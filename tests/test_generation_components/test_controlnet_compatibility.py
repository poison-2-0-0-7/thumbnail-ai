"""
test_controlnet_compatibility.py
==================================

Unit tests for Module 7 ControlNet Capability Resolution Architecture.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from modules.config import (
    CONTROLNET_CAPABILITY_TABLE,
    CapabilityCandidate,
    validate_controlnet_capability_availability,
)
from modules.generation_components.capability_probe import CapabilityProbe
from modules.generation_components.controlnet_capability_resolver import (
    ControlNetCapabilityResolver,
    ResolvedCapability,
)
from modules.generation_components.model_discovery_service import ModelDiscoveryService
from modules.image_generator import WorkflowBuilder
from modules.models import GenerationProfile, PromptPackage, WorkflowTemplateRef
from module7_exceptions import Module7Error
from observability.diagnostics.models import RuleContext
from observability.diagnostics.rules.controlnet_capability_rules import (
    ControlNetCapabilityResolutionRule,
)
from observability.facts.models import TraceFacts
from observability.models import FragmentAttachmentRecord, GenerationTraceRecord


@pytest.fixture
def mock_object_info_full() -> dict[str, Any]:
    return {
        "ControlNetLoader": {
            "input": {
                "required": {
                    "control_net_name": [
                        [
                            "controlnet_depth_sdxl.safetensors",
                            "controlnet-sd-xl-1.0-canny.safetensors",
                            "control_lora_depth_rank128.safetensors",
                        ],
                        {},
                    ]
                }
            }
        },
        "T2IAdapterLoader": {
            "input": {
                "required": {
                    "t2i_adapter_name": [
                        [
                            "t2i_adapter_sdxl_segmentation.safetensors",
                        ],
                        {},
                    ]
                }
            }
        },
    }


def test_model_discovery_service_extraction(mock_object_info_full: dict[str, Any]) -> None:
    client = MagicMock()
    client.object_info.return_value = mock_object_info_full

    probe = CapabilityProbe(client=client, enabled=True)
    discovery = ModelDiscoveryService(client=client, probe=probe, enabled=True)

    models_cn = discovery.installed_models_for("ControlNetLoader", "control_net_name")
    assert "controlnet_depth_sdxl.safetensors" in models_cn
    assert "control_lora_depth_rank128.safetensors" in models_cn

    models_t2i = discovery.installed_models_for("T2IAdapterLoader", "t2i_adapter_name")
    assert models_t2i == ("t2i_adapter_sdxl_segmentation.safetensors",)

    # Missing class or field returns empty tuple
    assert discovery.installed_models_for("NonExistentNode", "control_net_name") == ()
    assert discovery.installed_models_for("ControlNetLoader", "invalid_field") == ()


def test_model_discovery_service_disabled_or_failed() -> None:
    discovery = ModelDiscoveryService(enabled=False)
    assert discovery.installed_models_for("ControlNetLoader", "control_net_name") == ()

    client = MagicMock()
    client.object_info.side_effect = RuntimeError("ComfyUI offline")
    discovery_error = ModelDiscoveryService(client=client, enabled=True)
    assert discovery_error.installed_models_for("ControlNetLoader", "control_net_name") == ()


def test_controlnet_capability_resolver_priority_matching(mock_object_info_full: dict[str, Any]) -> None:
    client = MagicMock()
    client.object_info.return_value = mock_object_info_full
    probe = CapabilityProbe(client=client)
    discovery = ModelDiscoveryService(client=client, probe=probe)
    resolver = ControlNetCapabilityResolver(discovery_service=discovery)

    # Depth: both legacy exact match and control_lora exist -> legacy exact match wins first
    res_depth = resolver.resolve("depth")
    assert res_depth.resolution_source == "legacy_exact_match"
    assert res_depth.resolved_filename == "controlnet_depth_sdxl.safetensors"
    assert res_depth.fragment_variant == "controlnet_depth"

    # Canny: legacy exact match is absent, but sdxl_1_0_official_alt_naming matches 'controlnet-sd-xl-1.0-canny.safetensors'
    res_canny = resolver.resolve("canny")
    assert res_canny.resolution_source == "pattern_match"
    assert res_canny.matched_pattern == "sdxl_1_0_official_alt_naming"
    assert res_canny.resolved_filename == "controlnet-sd-xl-1.0-canny.safetensors"
    assert res_canny.fragment_variant == "controlnet_canny"

    # Segmentation: T2IAdapter matches 't2i_adapter_sdxl_segmentation.safetensors'
    res_seg = resolver.resolve("segmentation")
    assert res_seg.resolution_source == "pattern_match"
    assert res_seg.matched_pattern == "t2i_adapter_segmentation"
    assert res_seg.resolved_filename == "t2i_adapter_sdxl_segmentation.safetensors"
    assert res_seg.fragment_variant == "controlnet_segmentation_t2iadapter"


def test_controlnet_capability_resolver_unresolved() -> None:
    client = MagicMock()
    client.object_info.return_value = {}
    probe = CapabilityProbe(client=client)
    discovery = ModelDiscoveryService(client=client, probe=probe)
    resolver = ControlNetCapabilityResolver(discovery_service=discovery)

    res = resolver.resolve("depth")
    assert res.resolution_source == "unresolved"
    assert res.resolved_filename is None


def test_validate_controlnet_capability_availability(mock_object_info_full: dict[str, Any]) -> None:
    client = MagicMock()
    client.object_info.return_value = mock_object_info_full
    probe = CapabilityProbe(client=client)
    discovery = ModelDiscoveryService(client=client, probe=probe)
    resolver = ControlNetCapabilityResolver(discovery_service=discovery)

    # Should pass when capabilities resolve
    validate_controlnet_capability_availability(resolver, required_capabilities=frozenset({"depth", "canny", "segmentation"}))

    # Should raise Module7Error when a required capability cannot be resolved
    mock_object_info_incomplete = {
        "ControlNetLoader": {
            "input": {
                "required": {
                    "control_net_name": [["controlnet_depth_sdxl.safetensors"], {}]
                }
            }
        }
    }
    client.object_info.return_value = mock_object_info_incomplete
    probe._raw_object_info = None
    probe._cached_types = None
    probe._last_probe_time = 0.0
    discovery._cached_info = None
    discovery._last_probe_time = 0.0

    with pytest.raises(Module7Error, match="could not be resolved"):
        validate_controlnet_capability_availability(resolver, required_capabilities=frozenset({"depth", "canny"}))


def test_workflow_builder_slot_resolution(mock_object_info_full: dict[str, Any], tmp_path: Any) -> None:
    from modules.config import MODULE7_GENERATION_PROFILES

    client = MagicMock()
    client.object_info.return_value = mock_object_info_full
    probe = CapabilityProbe(client=client)
    discovery = ModelDiscoveryService(client=client, probe=probe)
    resolver = ControlNetCapabilityResolver(discovery_service=discovery)

    builder = WorkflowBuilder(capability_probe=probe, controlnet_resolver=resolver)

    package = PromptPackage.model_validate({
        "video_id": "test1234567",
        "title": "Test Title",
        "niche": "general",
        "positive_prompt": "a beautiful scenery",
        "negative_prompt": "ugly",
        "subject_instructions": "a cat",
        "background_instructions": "a garden",
        "typography_instructions": "big title",
        "composition_instructions": "rule of thirds",
        "lighting_instructions": "cinematic",
        "color_instructions": "vibrant",
        "generation_parameters": {
            "seed": 42,
            "width": 1280,
            "height": 720,
            "num_candidates": 1,
        },
        "quality_parameters": {},
        "model_settings": {},
        "generated_at": "2026-08-03T00:00:00Z",
    })
    profile = MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"]

    slots = builder._slots(package, profile, None, controlnet_resolver=resolver)
    assert slots["resolved_depth_controlnet"] == "controlnet_depth_sdxl.safetensors"
    assert slots["resolved_canny_controlnet"] == "controlnet_sd-xl-1.0-canny.safetensors" or slots["resolved_canny_controlnet"] == "controlnet-sd-xl-1.0-canny.safetensors"


def test_rule_edit_04_diagnostics() -> None:
    rule = ControlNetCapabilityResolutionRule()

    # Case 1: FAIL when unresolved
    frag_unresolved = FragmentAttachmentRecord(
        fragment_name="controlnet_depth",
        attach_point="positive_conditioning",
        requested_capability="depth",
        resolution_source="unresolved",
    )
    trace_unresolved = GenerationTraceRecord(
        video_id="test_vid",
        fragments_attached=[frag_unresolved],
    )
    ctx_unresolved = RuleContext(facts=TraceFacts(video_id="test_vid", extracted_at=""), generation_trace=trace_unresolved)
    finding_fail = rule.check(ctx_unresolved.facts, ctx_unresolved)
    assert finding_fail is not None
    assert finding_fail.severity == "FAIL"

    # Case 2: WARNING when pattern match fallback
    frag_pattern = FragmentAttachmentRecord(
        fragment_name="controlnet_canny",
        attach_point="positive_conditioning",
        requested_capability="canny",
        resolved_model="controlnet-sd-xl-1.0-canny.safetensors",
        resolution_source="pattern_match",
    )
    trace_pattern = GenerationTraceRecord(
        video_id="test_vid",
        fragments_attached=[frag_pattern],
    )
    ctx_pattern = RuleContext(facts=TraceFacts(video_id="test_vid", extracted_at=""), generation_trace=trace_pattern)
    finding_warn = rule.check(ctx_pattern.facts, ctx_pattern)
    assert finding_warn is not None
    assert finding_warn.severity == "WARNING"

    # Case 3: INFO when legacy exact match
    frag_exact = FragmentAttachmentRecord(
        fragment_name="controlnet_depth",
        attach_point="positive_conditioning",
        requested_capability="depth",
        resolved_model="controlnet_depth_sdxl.safetensors",
        resolution_source="legacy_exact_match",
    )
    trace_exact = GenerationTraceRecord(
        video_id="test_vid",
        fragments_attached=[frag_exact],
    )
    ctx_exact = RuleContext(facts=TraceFacts(video_id="test_vid", extracted_at=""), generation_trace=trace_exact)
    finding_info = rule.check(ctx_exact.facts, ctx_exact)
    assert finding_info is not None
    assert finding_info.severity == "INFO"
