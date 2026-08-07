"""
Rendering Engine Typography Subsystem

Vector typography rendering and DeepGaze negative saliency anti-collision solver.
"""

from .saliency_solver import SaliencySolver
from .vector_engine import VectorTypographyEngine

__all__ = ["SaliencySolver", "VectorTypographyEngine"]
