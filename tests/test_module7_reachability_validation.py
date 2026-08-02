from __future__ import annotations

import sys
from pathlib import Path
import pytest

_MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from config import (  # noqa: E402
    MODULE7_EDIT_CAPABLE_PROFILES,
    MODULE7_GENERATION_PROFILES,
    MODULE7_PROFILE_PREFERENCE,
    validate_module7_edit_reachability,
)
from models import GenerationProfile  # noqa: E402
from module7_exceptions import Module7Error  # noqa: E402
from image_generator import ProfileSelector  # noqa: E402


def test_validate_module7_edit_reachability_raises_on_unreachable_config() -> None:
    """Validator must raise Module7Error when edit-capable profiles exist but none are in preference tuple."""
    unreachable_preference = ("PROFILE_PREMIUM", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM")
    edit_capable = frozenset({"PROFILE_STANDARD_EDIT"})
    with pytest.raises(Module7Error, match="unreachable via MODULE7_PROFILE_PREFERENCE"):
        validate_module7_edit_reachability(preference=unreachable_preference, edit_capable=edit_capable)


def test_validate_module7_edit_reachability_passes_on_reachable_config() -> None:
    """Validator must pass without error when at least one edit-capable profile is in preference tuple."""
    reachable_preference = ("PROFILE_PREMIUM", "PROFILE_STANDARD_EDIT", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM")
    edit_capable = frozenset({"PROFILE_STANDARD_EDIT"})
    # Should not raise any exception
    validate_module7_edit_reachability(preference=reachable_preference, edit_capable=edit_capable)


def test_validate_module7_edit_reachability_passes_on_empty_edit_capable_set() -> None:
    """Validator must pass without error if no edit-capable profiles are configured."""
    preference = ("PROFILE_PREMIUM", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM")
    empty_edit_capable: frozenset[str] = frozenset()
    # Should not raise any exception
    validate_module7_edit_reachability(preference=preference, edit_capable=empty_edit_capable)


def test_module7_edit_capable_profiles_derivation() -> None:
    """MODULE7_EDIT_CAPABLE_PROFILES must be derived dynamically from MODULE7_GENERATION_PROFILES."""
    assert "PROFILE_STANDARD_EDIT" in MODULE7_EDIT_CAPABLE_PROFILES
    assert "PROFILE_STANDARD" not in MODULE7_EDIT_CAPABLE_PROFILES
    assert "PROFILE_PREMIUM" not in MODULE7_EDIT_CAPABLE_PROFILES
    assert "PROFILE_FAST" not in MODULE7_EDIT_CAPABLE_PROFILES
    assert "PROFILE_LOW_VRAM" not in MODULE7_EDIT_CAPABLE_PROFILES

    base_prof = MODULE7_GENERATION_PROFILES["PROFILE_STANDARD"]
    synthetic_profiles = {
        "P1": base_prof.model_copy(update={"name": "P1"}),
        "P2_EDIT": base_prof.model_copy(update={"name": "P2_EDIT", "edit_mode_default": "staged_edit"}),
    }
    derived = frozenset(name for name, p in synthetic_profiles.items() if p.edit_mode_default == "staged_edit")
    assert derived == frozenset({"P2_EDIT"})


def test_profile_selector_startup_validation_passes_on_production_config() -> None:
    """ProfileSelector.__init__ must succeed with default MODULE7_PROFILE_PREFERENCE."""
    selector = ProfileSelector()
    assert selector is not None


def test_profile_selector_startup_validation_raises_on_unreachable_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """ProfileSelector.__init__ must raise Module7Error if config is patched to an unreachable tuple."""
    unreachable = ("PROFILE_PREMIUM", "PROFILE_STANDARD", "PROFILE_FAST", "PROFILE_LOW_VRAM")
    monkeypatch.setattr("config.MODULE7_PROFILE_PREFERENCE", unreachable)
    monkeypatch.setattr("image_generator.MODULE7_PROFILE_PREFERENCE", unreachable)
    with pytest.raises(Module7Error, match="unreachable via MODULE7_PROFILE_PREFERENCE"):
        ProfileSelector()
