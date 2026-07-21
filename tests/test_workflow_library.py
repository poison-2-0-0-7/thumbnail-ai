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
