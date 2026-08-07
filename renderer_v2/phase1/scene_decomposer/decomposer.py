"""SceneDecomposer orchestrator for sequential decomposition of an image into a SceneGraph."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from loguru import logger

from .base import Detector, Matter, DepthEstimator
from ..schemas import Instance, SceneGraph
from ..config import Phase1Config, default_config
from ..model_registry import ModelRegistry


class SceneDecomposer:
    """Orchestrates detection, matting refinement, and depth estimation sequentially."""

    def __init__(
        self,
        detector: Detector,
        matter: Matter,
        depth_estimator: DepthEstimator,
        registry: Optional[ModelRegistry] = None,
        config: Phase1Config = default_config,
    ) -> None:
        self.detector = detector
        self.matter = matter
        self.depth_estimator = depth_estimator
        self.registry = registry
        self.config = config

    def decompose(self, image: np.ndarray, class_prompts: List[str]) -> SceneGraph:
        """Decompose flat thumbnail image into a structured SceneGraph.

        Args:
            image: HxWx3 uint8 RGB image array.
            class_prompts: Text prompts for object detection.

        Returns:
            SceneGraph containing instances, alpha mattes, and depth map.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Input image must be HxWx3 RGB array, got shape {image.shape}")

        h, w, _ = image.shape
        logger.info("Starting Scene Decomposition for image size {w}x{h}...", w=w, h=h)

        # Stage 1: Instance Detection & Segmentation
        logger.info("Decomposer Stage 1: Detection & Segmentation...")
        instances = self.detector.detect(image, class_prompts)
        logger.info("Detected {count} initial instances", count=len(instances))

        if self.registry:
            self.registry.unload_all()

        # Stage 2: Matting Refinement per Instance
        logger.info("Decomposer Stage 2: Matting Refinement...")
        refined_instances: List[Instance] = []
        for inst in instances:
            alpha_matte = self.matter.refine(image, inst)
            inst.alpha_matte = alpha_matte
            refined_instances.append(inst)

        if self.registry:
            self.registry.unload_all()

        # Stage 3: Monocular Depth Estimation
        logger.info("Decomposer Stage 3: Monocular Depth Estimation...")
        depth_map = self.depth_estimator.estimate(image)

        if self.registry:
            self.registry.unload_all()

        # Compute per-instance mean depth layer value
        for inst in refined_instances:
            if inst.mask.any():
                inst.depth_layer = float(np.mean(depth_map[inst.mask]))
            else:
                inst.depth_layer = 0.5

        scene_graph = SceneGraph(
            source_image=image,
            instances=refined_instances,
            depth_map=depth_map,
            width=w,
            height=h,
        )

        logger.info("Scene Decomposition complete. Created SceneGraph with {c} instances", c=len(refined_instances))
        return scene_graph
