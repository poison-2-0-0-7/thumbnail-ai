"""
Tests for NodeFragmentLibrary.
"""

import json
from pathlib import Path
import pytest
from generation_components.node_fragment_library import NodeFragmentLibrary
from module7_exceptions import WorkflowBuildError


def test_discover_fragments_sorted(tmp_path: Path):
    (tmp_path / "b_fragment.json").write_text("{}", encoding="utf-8")
    (tmp_path / "a_fragment.json").write_text("{}", encoding="utf-8")

    lib = NodeFragmentLibrary(fragment_dir=tmp_path)
    discovered = lib.discover()

    assert len(discovered) == 2
    assert discovered[0].name == "a_fragment.json"
    assert discovered[1].name == "b_fragment.json"


def test_load_valid_fragment(tmp_path: Path):
    fragment_data = {
        "_attach": {"point": "positive_conditioning", "output_node": "30", "output_slot": 0},
        "graph": {
            "30": {"class_type": "ControlNetApply", "inputs": {}}
        }
    }
    (tmp_path / "test_frag.json").write_text(json.dumps(fragment_data), encoding="utf-8")

    lib = NodeFragmentLibrary(fragment_dir=tmp_path)
    loaded = lib.load("test_frag")

    assert loaded["_attach"]["point"] == "positive_conditioning"
    assert "30" in loaded["graph"]


def test_load_missing_fragment_raises(tmp_path: Path):
    lib = NodeFragmentLibrary(fragment_dir=tmp_path)
    with pytest.raises(WorkflowBuildError, match="not found"):
        lib.load("nonexistent_frag")


def test_load_invalid_schema_raises(tmp_path: Path):
    (tmp_path / "invalid_frag.json").write_text(json.dumps({"_attach": {}}), encoding="utf-8")

    lib = NodeFragmentLibrary(fragment_dir=tmp_path)
    with pytest.raises(WorkflowBuildError, match="missing valid '_attach.point'"):
        lib.load("invalid_frag")
