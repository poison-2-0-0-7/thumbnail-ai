from __future__ import annotations

import sys
from pathlib import Path
import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))


@pytest.fixture(autouse=True)
def _patch_profile_preference_for_legacy_tests(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Ensure tests run against the activated Phase 3 profile preference tuple unless explicitly overridden."""
    if "unreachable" in request.node.name:
        return
    valid_pref = ("PROFILE_PREMIUM", "PROFILE_STANDARD_EDIT", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM")
    monkeypatch.setattr("config.MODULE7_PROFILE_PREFERENCE", valid_pref)
    monkeypatch.setattr("image_generator.MODULE7_PROFILE_PREFERENCE", valid_pref)
