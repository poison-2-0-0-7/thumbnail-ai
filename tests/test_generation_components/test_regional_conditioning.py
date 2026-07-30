"""
Tests for Phase 3.4 regional conditioning & IP-Adapter fragment selection & assembly.
"""

from pathlib import Path
from generation_components.conditioning_asset_resolver import GenerationConditioningContext, LayerConditioning
from generation_components.node_fragment_library import NodeFragmentLibrary
from image_generator import WorkflowBuilder
from models import GenerationProfile


def _dummy_profile(ipadapter_enabled: bool = True) -> GenerationProfile:
    return GenerationProfile(
        name="TEST",
        checkpoint="ckpt.safetensors",
        checkpoint_family="sdxl",
        sampler="euler",
        scheduler="normal",
        steps=20,
        cfg=7.0,
        controlnet_enabled=True,
        ipadapter_enabled=ipadapter_enabled,
        restoration="none",
        restoration_fidelity=0.35,
        upscaler="lanczos_only",
        expected_vram_gb=8.0,
        expected_generation_seconds=10.0,
    )


def test_ipadapter_and_text_exclusion_fragment_selection(tmp_path: Path):
    ip_img = tmp_path / "ip.png"
    text_mask = tmp_path / "text_mask.png"
    person_mask = tmp_path / "person_mask.png"
    ip_img.write_bytes(b"ip")
    text_mask.write_bytes(b"mask")
    person_mask.write_bytes(b"pmask")

    ctx = GenerationConditioningContext(
        ip_adapter_reference_paths={"fg": ip_img},
        text_exclusion_mask_path=text_mask,
        per_layer={"foreground": LayerConditioning(role="foreground", decision="composite", mask_path=person_mask)},
    )


    builder = WorkflowBuilder()
    profile = _dummy_profile(ipadapter_enabled=True)

    selected = builder._select_fragments(profile, ctx)

    assert "ipadapter_reference" in selected
    assert "text_exclusion_mask" in selected
    assert "regional_mask_conditioning" in selected


def test_fragment_library_loads_all_six_fragments():
    lib = NodeFragmentLibrary()
    fragments = lib.discover()
    names = [f.name for f in fragments]

    assert "controlnet_depth.json" in names
    assert "controlnet_canny.json" in names
    assert "controlnet_segmentation.json" in names
    assert "ipadapter_reference.json" in names
    assert "text_exclusion_mask.json" in names
    assert "regional_mask_conditioning.json" in names


def test_controlnet_segmentation_fragment_selection(tmp_path: Path):
    seg_img = tmp_path / "seg.png"
    seg_img.write_bytes(b"seg")

    ctx = GenerationConditioningContext(segmentation_path=seg_img)
    builder = WorkflowBuilder()
    profile = _dummy_profile()

    selected = builder._select_fragments(profile, ctx)
    assert "controlnet_segmentation" in selected

