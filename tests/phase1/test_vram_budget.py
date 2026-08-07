"""VRAM budget tests to ensure peak GPU memory remains under 8.0 GB limit."""

from __future__ import annotations

import torch
from renderer_v2.phase1.config import Phase1Config
from renderer_v2.phase1.model_registry import ModelRegistry


def test_vram_budget_guard(test_config: Phase1Config):
    registry = ModelRegistry(test_config)
    registry.reset_vram_stats()

    # Simulate loading and unloading sequential stages
    def mock_stage_1():
        return "stage1_model"

    def mock_stage_2():
        return "stage2_model"

    model1 = registry.load_model("stage1", mock_stage_1)
    assert registry.active_model_name == "stage1"
    
    model2 = registry.load_model("stage2", mock_stage_2)
    assert registry.active_model_name == "stage2"

    registry.unload_all()
    assert registry.active_model_name is None

    peak_gb = registry.get_peak_vram_gb()
    assert peak_gb <= test_config.max_vram_gb
