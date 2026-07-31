"""Tests for StrategyPackLibrary and StrategyPackResolver."""

from __future__ import annotations

import pytest
from pathlib import Path

from models import CandidateStrategy
from generation_components import StrategyPackLibrary, StrategyPackResolver
from module7_exceptions import StrategyPackError


def test_resolver_default_fallback():
    resolver = StrategyPackResolver()
    strategies = resolver.resolve(requested_pack=None, max_candidates=5)
    assert len(strategies) == 1
    assert strategies[0].name == "faithful"
    assert strategies[0].camera_distance_shift == 0


def test_resolver_loads_default_five():
    resolver = StrategyPackResolver()
    strategies = resolver.resolve(requested_pack="default_five", max_candidates=5)
    assert len(strategies) == 5
    names = [s.name for s in strategies]
    assert names == [
        "faithful",
        "higher_emotion",
        "cleaner_composition",
        "higher_contrast",
        "aggressive_ctr",
    ]


def test_resolver_truncation():
    resolver = StrategyPackResolver()
    strategies = resolver.resolve(requested_pack="default_five", max_candidates=2)
    assert len(strategies) == 2
    assert [s.name for s in strategies] == ["faithful", "higher_emotion"]


def test_resolver_missing_pack():
    resolver = StrategyPackResolver()
    with pytest.raises(StrategyPackError):
        resolver.resolve(requested_pack="non_existent_pack_xyz")
