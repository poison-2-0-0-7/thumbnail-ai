"""
Rendering Engine Vision & Matting Subsystem

Provides zero-shot GroundingDINO instance detection, SAM 2 coarse segmentation,
and ViTMatting 8-bit continuous alpha matte refinement.
"""

from .segmentor import CoarseSegmentor
from .matting import AlphaMattingEngine

__all__ = ["CoarseSegmentor", "AlphaMattingEngine"]
