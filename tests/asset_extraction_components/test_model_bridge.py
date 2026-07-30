"""
test_model_bridge.py
====================

Unit tests for ModelBridge component.
"""

from unittest.mock import MagicMock
import pytest

from modules.asset_extraction_components.model_bridge import ModelBridge
from modules.asset_extraction_exceptions import AssetFamilyModelError
from vision_stack.exceptions import VisionStackError


def test_model_bridge_run_success():
    mock_runtime = MagicMock()
    mock_gpu_resources = MagicMock()
    mock_runtime.is_bootstrapped.return_value = True
    mock_runtime.registry.get.return_value = MagicMock()

    bridge = ModelBridge(runtime_manager=mock_runtime, gpu_resource_manager=mock_gpu_resources)

    operation = MagicMock(return_value="success_result")
    result = bridge.run("sam2", operation)

    assert result == "success_result"
    mock_gpu_resources.reserve.assert_called_once_with("sam2")


def test_model_bridge_exception_translation():
    mock_runtime = MagicMock()
    mock_gpu_resources = MagicMock()
    mock_runtime.is_bootstrapped.return_value = True
    mock_runtime.registry.get.return_value = MagicMock()

    # Make gpu_resources.reserve raise VisionStackError
    mock_gpu_resources.reserve.side_effect = VisionStackError("GPU lock error")

    bridge = ModelBridge(runtime_manager=mock_runtime, gpu_resource_manager=mock_gpu_resources)

    with pytest.raises(AssetFamilyModelError) as exc_info:
        bridge.run("birefnet", lambda m: None)

    assert exc_info.value.model_name == "birefnet"
    assert "GPU lock error" in str(exc_info.value)
