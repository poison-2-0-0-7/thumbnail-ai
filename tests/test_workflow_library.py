from __future__ import annotations

from pathlib import Path
import sys

import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import MODULE7_GENERATION_PROFILES, MODULE7_NICHE_WORKFLOW_MAP  # noqa: E402
from module7_exceptions import WorkflowTemplateError  # noqa: E402
from workflow_library import WorkflowLibrary  # noqa: E402


def test_discovery_and_configured_niches_are_valid() -> None:
    library = WorkflowLibrary()
    names = {path.name for path in library.discover()}
    assert "general.json" in names
    assert set(MODULE7_NICHE_WORKFLOW_MAP.values()) <= names
    for path in library.discover():
        library.load(path)


def test_unknown_niche_deterministically_uses_general_template() -> None:
    ref = WorkflowLibrary().resolve("future-niche", MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"])
    assert Path(ref.template_path).name == "general.json"
    assert ref.profile_name == "PROFILE_STANDARD"


def test_validation_rejects_missing_graph_contract(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"_meta": {"name": "bad"}, "graph": {}}', encoding="utf-8")
    with pytest.raises(WorkflowTemplateError):
        WorkflowLibrary(tmp_path).load(path)


def test_template_path_cannot_escape_library(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(WorkflowTemplateError):
        WorkflowLibrary(tmp_path).load(outside)


def test_all_templates_contain_save_image_node() -> None:
    library = WorkflowLibrary()
    for path in library.discover():
        template = library.load(path)
        graph = template["graph"]
        has_save_node = any(
            isinstance(node, dict) and node.get("class_type") == "SaveImage"
            for node in graph.values()
        )
        assert has_save_node, f"Template {path.name} must contain a SaveImage node"


def test_resolve_legacy_mode_returns_legacy_template() -> None:
    library = WorkflowLibrary()
    profile = MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"]
    ref = library.resolve("gaming", profile, edit_mode="legacy_txt2img")

    assert Path(ref.template_path).name == "gaming.json"


def test_resolve_staged_edit_mode_returns_edit_template() -> None:
    library = WorkflowLibrary()
    profile = MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"]
    ref_gaming = library.resolve("gaming", profile, edit_mode="staged_edit")
    ref_general = library.resolve("general", profile, edit_mode="staged_edit")

    assert Path(ref_gaming.template_path).name == "gaming_edit.json"
    assert Path(ref_general.template_path).name == "general_edit.json"


def test_builder_attaches_inpaint_fragments_for_edit_workflow() -> None:
    from image_generator import WorkflowBuilder
    from generation_components import GenerationConditioningContext

    library = WorkflowLibrary()
    profile = MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"]
    ref_edit = library.resolve("gaming", profile, edit_mode="staged_edit")

    builder = WorkflowBuilder()
    dummy_ctx = GenerationConditioningContext(
        source_thumbnail_path=Path("source.png"),
        role_mask_paths={"edit_mask": Path("mask.png")},
    )

    base_graph = builder.build_base(profile, ref_edit, library=library, conditioning=dummy_ctx)

    # Must contain nodes from inpaint_base (VAEEncodeForInpaint) and edit_region_mask (SetLatentNoiseMask)
    node_types = [node.get("class_type") for node in base_graph.values() if isinstance(node, dict)]
    assert "VAEEncodeForInpaint" in node_types
    assert "SetLatentNoiseMask" in node_types


