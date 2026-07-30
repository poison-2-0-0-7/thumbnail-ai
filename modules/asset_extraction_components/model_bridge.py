"""
model_bridge.py
===============

Adapter bridging Module 8 asset processors to vision_stack RuntimeManager
and GPUResourceManager. Translates vision_stack lifecycle exceptions into
Module 8's AssetFamilyModelError.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from loguru import logger

from modules.asset_extraction_components.interfaces import IModelBridge
from modules.asset_extraction_exceptions import AssetFamilyModelError
from vision_stack.exceptions import VisionStackError
from vision_stack.resources import GPUResourceManager
from vision_stack.runtime import RuntimeManager


class ModelBridge(IModelBridge):
    """Adapter bridging Module 8 processors to vision_stack single-active GPU reservations."""

    def __init__(
        self,
        runtime_manager: Optional[RuntimeManager] = None,
        gpu_resource_manager: Optional[GPUResourceManager] = None,
    ) -> None:
        self._runtime = runtime_manager or RuntimeManager()
        self._gpu_resources_provided = gpu_resource_manager

    def _get_gpu_resources(self) -> GPUResourceManager:
        if self._gpu_resources_provided is not None:
            return self._gpu_resources_provided
        if not self._runtime.is_bootstrapped():
            self._runtime.bootstrap()
        return GPUResourceManager(registry=self._runtime.registry)

    def run(self, model_name: str, operation: Callable[[Any], Any]) -> Any:
        """Reserve GPU slot for model_name, run operation, release reservation."""
        try:
            if not self._runtime.is_bootstrapped():
                self._runtime.bootstrap()

            registered_model = self._runtime.registry.get(model_name)
            if registered_model is None:
                raise AssetFamilyModelError(
                    f"Vision model {model_name} is not registered in vision_stack",
                    family_name="unknown",
                    model_name=model_name,
                )

            gpu_res = self._get_gpu_resources()
            with gpu_res.reserve(model_name):
                logger.debug("GPU slot reserved for model={model_name}", model_name=model_name)
                wrapper = self._resolve_model_wrapper(model_name, registered_model)
                result = operation(wrapper or registered_model)
                return result

        except AssetFamilyModelError:
            raise
        except VisionStackError as exc:
            logger.error("ModelBridge caught VisionStackError for {model}: {err}", model=model_name, err=str(exc))
            raise AssetFamilyModelError(
                f"Model execution failed for {model_name}: {exc}",
                family_name="unknown",
                model_name=model_name,
            ) from exc
        except Exception as exc:
            logger.error("ModelBridge caught unexpected error for {model}: {err}", model=model_name, err=str(exc))
            raise AssetFamilyModelError(
                f"Unexpected error executing {model_name}: {exc}",
                family_name="unknown",
                model_name=model_name,
            ) from exc

    def _resolve_model_wrapper(self, model_name: str, registered_model: Any) -> Any:
        """Instantiate or retrieve the vision_stack wrapper instance for model_name."""
        try:
            if model_name == "sam2":
                from vision_stack.sam2 import SAM2Wrapper
                wrapper = SAM2Wrapper()
                wrapper.ensure_loaded(registered_model)
                return wrapper
            elif model_name == "bisenet":
                from vision_stack.bisenet import BiSeNetWrapper
                wrapper = BiSeNetWrapper()
                wrapper.ensure_loaded(registered_model)
                return wrapper
            elif model_name == "insightface":
                from vision_stack.insightface_multi import InsightFaceMultiWrapper
                wrapper = InsightFaceMultiWrapper()
                wrapper.ensure_loaded(registered_model)
                return wrapper
            elif model_name == "birefnet":
                from vision_stack.birefnet import BiRefNetWrapper
                wrapper = BiRefNetWrapper()
                wrapper.ensure_loaded(registered_model)
                return wrapper
            elif model_name == "depth_anything":
                from vision_stack.depth_anything import DepthAnythingWrapper
                wrapper = DepthAnythingWrapper()
                wrapper.ensure_loaded(registered_model)
                return wrapper
            elif model_name == "teed":
                from vision_stack.teed import TEEDWrapper
                wrapper = TEEDWrapper()
                wrapper.ensure_loaded(registered_model)
                return wrapper
            elif model_name == "grounding_dino":
                from vision_stack.grounding_dino import GroundingDINOWrapper
                wrapper = GroundingDINOWrapper()
                wrapper.ensure_loaded(registered_model)
                return wrapper
        except Exception as exc:
            logger.debug("Wrapper resolution fallback for {model}: {err}", model=model_name, err=str(exc))
        return registered_model
