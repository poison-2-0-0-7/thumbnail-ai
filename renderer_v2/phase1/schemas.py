"""Data schemas for Phase 1 Scene Decomposer and Inpaint pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Dict
import numpy as np

InstanceClass = Literal["creator", "logo", "product", "other"]


@dataclass
class Instance:
    """Represents a single segmented instance within a decomposed scene graph."""

    instance_id: str
    cls: InstanceClass
    mask: np.ndarray  # hard binary mask (HxW uint8/bool)
    alpha_matte: np.ndarray  # soft alpha matte (HxW float32 [0.0, 1.0])
    bbox: tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    depth_layer: float  # mean depth value within mask region
    locked: bool  # True for creator/logo/product instances to prevent diffusion edit

    def __post_init__(self) -> None:
        """Validate array dimensions and bounds."""
        if self.mask.ndim != 2:
            raise ValueError(f"Instance mask must be 2D HxW array, got shape {self.mask.shape}")
        if self.alpha_matte.ndim != 2:
            raise ValueError(f"Instance alpha_matte must be 2D HxW array, got shape {self.alpha_matte.shape}")
        if self.mask.shape != self.alpha_matte.shape:
            raise ValueError(
                f"Mask shape {self.mask.shape} does not match alpha_matte shape {self.alpha_matte.shape}"
            )


@dataclass
class SceneGraph:
    """Complete decomposed representation of an input image."""

    source_image: np.ndarray  # HxWx3 RGB uint8 image
    instances: list[Instance]  # list of segmented instances
    depth_map: np.ndarray  # HxW float32 depth map normalized [0.0, 1.0]
    width: int  # image width in pixels
    height: int  # image height in pixels

    def get_locked_instances(self) -> list[Instance]:
        """Return all instances flagged as locked."""
        return [inst for inst in self.instances if inst.locked]


@dataclass
class PipelineResult:
    """Final result of the Phase 1 processing pipeline, including intermediate debug artifacts."""

    output_image: np.ndarray  # Final recomposited image (HxWx3 RGB uint8)
    scene_graph: SceneGraph  # Scene graph extracted during decomposition
    inpainted_background: np.ndarray  # Background synthesized by inpainter (HxWx3 RGB uint8)
    locked_region_mask: np.ndarray  # Union of locked region mattes (HxW float32)
    debug_artifacts: Dict[str, Any] = field(default_factory=dict)  # Maps artifact name to image array or data
