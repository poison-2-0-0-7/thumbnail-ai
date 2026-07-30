"""
Tests for CapabilityProbe and ControlNet fragment assembly.
"""

from pathlib import Path

from generation_components.capability_probe import CapabilityProbe
from generation_components.conditioning_asset_resolver import GenerationConditioningContext
from generation_components.workflow_graph_assembler import WorkflowGraphAssembler
from image_generator import WorkflowBuilder
from models import GenerationProfile


class MockComfyUIClient:
    def __init__(self, installed_nodes: list[str]):
        self._installed_nodes = installed_nodes

    def object_info(self) -> dict[str, dict]:
        return {node: {} for node in self._installed_nodes}


def test_capability_probe_returns_installed_nodes():
    client = MockComfyUIClient(["LoadImage", "KSampler", "ControlNetApply"])
    probe = CapabilityProbe(client=client, enabled=True)

    installed = probe.installed_node_types()
    assert "LoadImage" in installed
    assert "ControlNetApply" in installed
    assert "UnknownNode" not in installed


def test_capability_probe_drops_unsupported_fragment():
    client = MockComfyUIClient(["LoadImage"])  # Missing ControlNetApply and ControlNetLoader
    probe = CapabilityProbe(client=client, enabled=True)

    fragment = {
        "_meta": {"name": "test_cn", "required_node_types": ["ControlNetApply"]},
        "graph": {"10": {"class_type": "ControlNetApply", "inputs": {}}},
    }

    assert probe.is_fragment_supported(fragment) is False


def test_capability_probe_allows_supported_fragment():
    client = MockComfyUIClient(["LoadImage", "ControlNetLoader", "ControlNetApply"])
    probe = CapabilityProbe(client=client, enabled=True)

    fragment = {
        "_meta": {"name": "test_cn", "required_node_types": ["ControlNetApply"]},
        "graph": {"10": {"class_type": "ControlNetApply", "inputs": {}}},
    }

    assert probe.is_fragment_supported(fragment) is True


def test_controlnet_fragment_assembly_integration(tmp_path: Path):
    depth_file = tmp_path / "depth.png"
    depth_file.write_bytes(b"depth")

    ctx = GenerationConditioningContext(depth_path=depth_file)
    builder = WorkflowBuilder()
    profile = GenerationProfile(
        name="TEST",
        checkpoint="ckpt.safetensors",
        checkpoint_family="sdxl",
        sampler="euler",
        scheduler="normal",
        steps=20,
        cfg=7.0,
        controlnet_enabled=True,
        ipadapter_enabled=False,
        restoration="none",
        restoration_fidelity=0.35,
        upscaler="lanczos_only",
        expected_vram_gb=8.0,
        expected_generation_seconds=10.0,
    )

    selected = builder._select_fragments(profile, ctx)
    assert "controlnet_depth" in selected
