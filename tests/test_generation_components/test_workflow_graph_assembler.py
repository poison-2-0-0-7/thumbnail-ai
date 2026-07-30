"""
Tests for WorkflowGraphAssembler.
"""

from pathlib import Path
import pytest
from generation_components.conditioning_asset_resolver import GenerationConditioningContext
from generation_components.workflow_graph_assembler import WorkflowGraphAssembler
from module7_exceptions import FragmentAttachmentError
from models import GenerationProfile


def _dummy_profile() -> GenerationProfile:
    return GenerationProfile(
        name="TEST_PROFILE",
        checkpoint="ckpt.safetensors",
        checkpoint_family="sdxl",
        sampler="euler",
        scheduler="normal",
        steps=20,
        cfg=7.0,
        controlnet_enabled=True,
        ipadapter_enabled=True,
        restoration="none",
        restoration_fidelity=0.35,
        upscaler="lanczos_only",
        expected_vram_gb=8.0,
        expected_generation_seconds=10.0,
    )


def test_assemble_zero_fragments_returns_unmodified():
    base_graph = {
        "_meta": {"attachment_points": {"positive_conditioning": ["5", "positive"]}},
        "graph": {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos"}},
            "5": {"class_type": "KSampler", "inputs": {"positive": ["2", 0]}},
        },
    }

    assembler = WorkflowGraphAssembler()
    result = assembler.assemble(base_graph, [], GenerationConditioningContext(), _dummy_profile())

    assert result == base_graph


test_frag_1 = {
    "_attach": {"point": "positive_conditioning", "output_node": "30", "output_slot": 0},
    "graph": {
        "10": {"class_type": "LoadImage", "inputs": {"image": "/path/depth.png"}},
        "20": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "depth.safetensors"}},
        "30": {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": ["ATTACHMENT_PREVIOUS", 0],
                "control_net": ["20", 0],
                "image": ["10", 0],
                "strength": 0.55,
            },
        },
    },
}

test_frag_2 = {
    "_attach": {"point": "positive_conditioning", "output_node": "30", "output_slot": 0},
    "graph": {
        "10": {"class_type": "LoadImage", "inputs": {"image": "/path/canny.png"}},
        "20": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "canny.safetensors"}},
        "30": {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": ["ATTACHMENT_PREVIOUS", 0],
                "control_net": ["20", 0],
                "image": ["10", 0],
                "strength": 0.45,
            },
        },
    },
}


def test_assemble_chained_fragments():
    base_graph = {
        "_meta": {"attachment_points": {"positive_conditioning": ["5", "positive"]}},
        "graph": {
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos"}},
            "5": {"class_type": "KSampler", "inputs": {"positive": ["2", 0]}},
        },
    }

    assembler = WorkflowGraphAssembler()
    res = assembler.assemble(base_graph, [test_frag_1, test_frag_2], GenerationConditioningContext(), _dummy_profile())

    graph = res["graph"]
    # Check that KSampler positive points to frag 2 output
    assert graph["5"]["inputs"]["positive"] == ["frag_1_positive_conditioning_30", 0]

    # Check frag 2 input points to frag 1 output
    frag2_node30 = graph["frag_1_positive_conditioning_30"]
    assert frag2_node30["inputs"]["conditioning"] == ["frag_0_positive_conditioning_30", 0]

    # Check frag 1 input points to original base input ["2", 0]
    frag1_node30 = graph["frag_0_positive_conditioning_30"]
    assert frag1_node30["inputs"]["conditioning"] == ["2", 0]


def test_assemble_unknown_attachment_point_raises():
    base_graph = {"_meta": {"attachment_points": {}}, "graph": {}}
    assembler = WorkflowGraphAssembler()
    with pytest.raises(FragmentAttachmentError, match="not defined"):
        assembler.assemble(base_graph, [test_frag_1], GenerationConditioningContext(), _dummy_profile())
