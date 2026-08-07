"""Unit tests for SceneDecomposer components."""

from __future__ import annotations

import numpy as np
import pytest

from renderer_v2.phase1.config import Phase1Config
from renderer_v2.phase1.model_registry import ModelRegistry
from renderer_v2.phase1.scene_decomposer.groundingdino_sam2_detector import GroundingDINOSAM2Detector
from renderer_v2.phase1.scene_decomposer.sam3_detector import SAM3Detector
from renderer_v2.phase1.scene_decomposer.birefnet_matter import BiRefNetMatter
from renderer_v2.phase1.scene_decomposer.depth_anything import DepthAnythingEstimator
from renderer_v2.phase1.scene_decomposer.decomposer import SceneDecomposer


@pytest.mark.requires_models
def test_groundingdino_sam2_detector(sample_rgb_image: np.ndarray, test_config: Phase1Config, model_registry: ModelRegistry):
    detector = GroundingDINOSAM2Detector(config=test_config, registry=model_registry)
    instances = detector.detect(sample_rgb_image, ["person", "logo"])
    
    assert len(instances) > 0
    for inst in instances:
        assert inst.mask.shape == sample_rgb_image.shape[:2]
        assert inst.alpha_matte.shape == sample_rgb_image.shape[:2]
        assert isinstance(inst.locked, bool)


@pytest.mark.requires_models
def test_sam3_detector(sample_rgb_image: np.ndarray, test_config: Phase1Config, model_registry: ModelRegistry):
    detector = SAM3Detector(config=test_config, registry=model_registry)
    instances = detector.detect(sample_rgb_image, ["person"])
    
    assert len(instances) > 0
    assert instances[0].cls in ["creator", "other"]


@pytest.mark.requires_models
def test_birefnet_matter(sample_rgb_image: np.ndarray, sample_scene_graph, test_config: Phase1Config, model_registry: ModelRegistry):
    matter = BiRefNetMatter(config=test_config, registry=model_registry)
    inst = sample_scene_graph.instances[0]
    
    alpha = matter.refine(sample_rgb_image, inst)
    assert alpha.shape == sample_rgb_image.shape[:2]
    assert alpha.dtype == np.float32
    assert 0.0 <= alpha.min() <= alpha.max() <= 1.0


@pytest.mark.requires_models
def test_depth_anything_estimator(sample_rgb_image: np.ndarray, test_config: Phase1Config, model_registry: ModelRegistry):
    depth_estimator = DepthAnythingEstimator(config=test_config, registry=model_registry)
    depth_map = depth_estimator.estimate(sample_rgb_image)
    
    assert depth_map.shape == sample_rgb_image.shape[:2]
    assert depth_map.dtype == np.float32
    assert 0.0 <= depth_map.min() <= depth_map.max() <= 1.0


@pytest.mark.requires_models
def test_scene_decomposer_end_to_end(sample_rgb_image: np.ndarray, test_config: Phase1Config, model_registry: ModelRegistry):
    detector = GroundingDINOSAM2Detector(config=test_config, registry=model_registry)
    matter = BiRefNetMatter(config=test_config, registry=model_registry)
    depth = DepthAnythingEstimator(config=test_config, registry=model_registry)

    decomposer = SceneDecomposer(
        detector=detector,
        matter=matter,
        depth_estimator=depth,
        registry=model_registry,
        config=test_config,
    )

    scene_graph = decomposer.decompose(sample_rgb_image, ["person", "logo"])
    assert scene_graph.width == sample_rgb_image.shape[1]
    assert scene_graph.height == sample_rgb_image.shape[0]
    assert len(scene_graph.instances) > 0
    assert scene_graph.depth_map.shape == sample_rgb_image.shape[:2]
