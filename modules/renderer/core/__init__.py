"""
Rendering Engine Core Package

Contains immutable data contracts, Canvas layer models, and engine configuration schemas.
"""

from .schema import (
    LayerType,
    LayerAction,
    Archetype,
    EditPlan,
    LayerSpec,
    GenerativeParams,
    RelightingSpec,
    DropShadowSpec,
    TypographySpec,
    QualityReport,
)
from .canvas import Canvas, Layer
from .config import RendererConfig

__all__ = [
    "LayerType",
    "LayerAction",
    "Archetype",
    "EditPlan",
    "LayerSpec",
    "GenerativeParams",
    "RelightingSpec",
    "DropShadowSpec",
    "TypographySpec",
    "QualityReport",
    "Canvas",
    "Layer",
    "RendererConfig",
]
