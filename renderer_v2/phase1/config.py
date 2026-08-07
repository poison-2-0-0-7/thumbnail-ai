"""Configuration settings for Phase 1 Scene Decomposer and Inpainting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Phase1Config:
    """Central configuration for Phase 1 pipeline components and model settings."""

    # Target hardware constraints
    max_vram_gb: float = 8.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    models_cache_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "models_cache"
    )
    debug_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "debug"
    )

    # Class Prompts for Scene Decomposer
    default_class_prompts: list[str] = field(
        default_factory=lambda: ["person", "logo", "product"]
    )
    locked_classes: list[str] = field(
        default_factory=lambda: ["creator", "person", "logo", "product"]
    )

    # Inpainting Defaults (Phase 1 has no EditPlanner)
    default_inpaint_prompt: str = (
        "modern vibrant YouTube thumbnail background, clean studio lighting, high quality, 8k resolution"
    )
    default_negative_prompt: str = (
        "person, face, creator, logo, product, text, watermark, ugly, blurry, noise, distortion"
    )

    # Mask Processing Parameters
    mask_dilation_px: int = 12
    mask_feather_px: int = 6

    # Real Production Model Identifiers
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam2_model_id: str = "sam2.1_b.pt"
    birefnet_model_id: str = "ZhengPeng7/BiRefNet_lite"
    depth_anything_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    sdxl_inpaint_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

    def __post_init__(self) -> None:
        """Ensure paths exist."""
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)


default_config = Phase1Config()
