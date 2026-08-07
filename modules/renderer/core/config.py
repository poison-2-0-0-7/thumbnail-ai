"""
Rendering Engine Configuration Schema

Defines GPU hardware routing, device paths, model weights, latency mode, and quality thresholds.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class LatencyTier(str, Enum):
    FAST_PATH = "fast_path"  # Lightweight templates & fast diffusion (~1.8s)
    DEEP_PATH = "deep_path"  # Full SAM2 + ViTMatting + Flux Fill + NDAER (~8.5s)


class RendererConfig(BaseModel):
    device: str = "cuda"
    fp16_enabled: bool = True
    tensorrt_enabled: bool = True
    latency_tier: LatencyTier = LatencyTier.DEEP_PATH
    
    # Model checkpoints & paths
    sam2_checkpoint_path: str = "checkpoints/sam2_hiera_large.pt"
    vit_matting_model_path: str = "checkpoints/vit_matting_base.pth"
    depth_anything_model_path: str = "checkpoints/depth_anything_v2_vitl.pth"
    flux_fill_model_id: str = "black-forest-labs/FLUX.1-Fill-dev"
    arcface_model_path: str = "checkpoints/arcface_w600k_r50.onnx"
    
    # Quality Gating Thresholds
    max_identity_cosine_drift: float = Field(default=0.15, description="Reject if face distance > 0.15")
    min_predicted_ctr_lift_pct: float = Field(default=15.0, description="Reject if estimated CTR lift < 15%")
    min_wcag_contrast_ratio: float = Field(default=4.5, description="Reject if text contrast < 4.5:1")
    
    # Work directory & caching
    cache_dir: str = ".cache/renderer"
    output_dir: str = "output/redesigns"
