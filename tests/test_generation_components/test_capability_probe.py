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


def test_controlnet_fragment_assembly_integration(tmp_path: Path, monkeypatch):
    from generation_components.controlnet_capability_resolver import ControlNetCapabilityResolver, ResolvedCapability
    dummy_res = ResolvedCapability(
        capability="depth",
        node_class="ControlNetApplyAdvanced",
        filename_field="control_net_name",
        resolved_filename="controlnet_depth_sdxl.safetensors",
        resolution_source="pattern_match",
        matched_pattern="depth",
        fragment_variant="controlnet_depth",
    )
    monkeypatch.setattr(ControlNetCapabilityResolver, "resolve", lambda self, cap: dummy_res)

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



def test_detect_workflow_node_types_scans_workflows_dir(tmp_path: Path):
    from generation_components.capability_probe import detect_workflow_node_types

    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    wf_file = wf_dir / "custom_test.json"
    wf_file.write_text(
        '{"_meta": {"name": "test"}, "graph": {"1": {"class_type": "IPAdapterApply"}, "2": {"class_type": "KSampler"}}}',
        encoding="utf-8",
    )

    detected = detect_workflow_node_types(wf_dir)
    assert "custom_test.json" in detected
    assert "IPAdapterApply" in detected["custom_test.json"]
    assert "KSampler" in detected["custom_test.json"]


def test_validate_all_workflows_reports_missing_nodes(tmp_path: Path):
    from generation_components.capability_probe import CapabilityProbe

    client = MockComfyUIClient(["KSampler", "CheckpointLoaderSimple"])
    probe = CapabilityProbe(client=client, enabled=True)

    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    (wf_dir / "ipadapter_wf.json").write_text(
        '{"_meta": {"name": "ipadapter_wf"}, "graph": {"1": {"class_type": "IPAdapterApply"}}}',
        encoding="utf-8",
    )

    missing_map = probe.validate_all_workflows(wf_dir)
    assert "ipadapter_wf.json" in missing_map
    missing_item = missing_map["ipadapter_wf.json"][0]
    assert missing_item["missing_node_type"] == "IPAdapterApply"
    assert missing_item["recommended_package"] == "ComfyUI_IPAdapter_plus"
    assert missing_item["workflow"] == "ipadapter_wf.json"


def test_validate_workflow_graph_prevents_submission(tmp_path: Path):
    import pytest
    from generation_components.capability_probe import CapabilityProbe
    from module7_exceptions import MissingCustomNodeError

    client = MockComfyUIClient(["KSampler", "CheckpointLoaderSimple"])
    probe = CapabilityProbe(client=client, enabled=True)

    graph = {
        "1": {"class_type": "CheckpointLoaderSimple"},
        "2": {"class_type": "ReActorFaceSwap"},
    }

    with pytest.raises(MissingCustomNodeError) as exc_info:
        probe.validate_workflow_graph(graph, workflow_name="face_swap_wf", raise_on_missing=True)

    err = exc_info.value
    assert "face_swap_wf" in str(err)
    assert "ReActorFaceSwap" in err.missing_nodes_report
    assert "comfyui-reactor_node" in err.missing_nodes_report


def test_validate_workflow_graph_allows_valid_submission():
    from generation_components.capability_probe import CapabilityProbe

    client = MockComfyUIClient(["KSampler", "CheckpointLoaderSimple", "ReActorFaceSwap"])
    probe = CapabilityProbe(client=client, enabled=True)

    graph = {
        "1": {"class_type": "CheckpointLoaderSimple"},
        "2": {"class_type": "ReActorFaceSwap"},
    }

    missing = probe.validate_workflow_graph(graph, workflow_name="face_swap_wf", raise_on_missing=True)
    assert missing == []

