"""
Rendering Engine V2.1 Data Contracts & Pydantic Schemas

Defines type-safe execution plans, layer directives, and quality report standards.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class LayerType(str, Enum):
    BACKGROUND = "background"
    FOREGROUND_SUBJECT = "foreground_subject"
    PRODUCT_OBJECT = "product_object"
    LOGO_BRAND = "logo_brand"
    TYPOGRAPHY = "typography"
    GRAPHIC_OVERLAY = "graphic_overlay"


class LayerAction(str, Enum):
    PRESERVE = "preserve"
    PRESERVE_AND_RELIGHT = "preserve_and_relight"
    GENERATIVE_REPLACE = "generative_replace"
    REPOSITION_AND_SCALE = "reposition_and_scale"
    RENDER_VECTOR_TEXT = "render_vector_text"
    REMOVE = "remove"


class Archetype(str, Enum):
    SINGLE_CREATOR_FACE = "single_creator_face"
    MULTI_PERSON_PODCAST = "multi_person_podcast"
    PRODUCT_TECH = "product_tech"
    GAMING_UI = "gaming_ui"
    INFO_LISTICLE = "info_listicle"


class GenerativeParams(BaseModel):
    model: str = "flux_fill_v1"
    prompt: str
    negative_prompt: Optional[str] = "clutter, noise, oversaturated, text, watermark, people"
    guidance_scale: float = Field(default=30.0, ge=1.0, le=50.0)
    denoise_strength: float = Field(default=0.88, ge=0.0, le=1.0)
    depth_guidance_weight: float = Field(default=0.45, ge=0.0, le=1.0)


class RelightingSpec(BaseModel):
    enabled: bool = True
    direction_angle_deg: int = Field(default=135, ge=0, le=360)
    color_hex: str = "#08D9D6"
    intensity: float = Field(default=0.65, ge=0.0, le=1.0)
    blend_mode: str = "screen"
    skin_freeze_margin_px: int = Field(default=15, ge=0)


class DropShadowSpec(BaseModel):
    enabled: bool = True
    blur_radius: int = Field(default=24, ge=0)
    offset_x: int = 12
    offset_y: int = 18
    opacity: float = Field(default=0.45, ge=0.0, le=1.0)
    color_hex: str = "#000000"


class TypographySpec(BaseModel):
    text_content: str
    font_family: str = "Outfit-ExtraBold"
    font_size: int = Field(default=94, gt=0)
    letter_spacing: float = 0.0
    bounding_box_target: Optional[Tuple[int, int, int, int]] = None
    alignment: str = "center"
    fill_colors: List[str] = Field(default_factory=lambda: ["#FFFFFF", "#F3F4F6"])
    stroke_color: str = "#000000"
    stroke_width: int = 12
    pill_container_enabled: bool = True
    pill_fill_color: str = "#FF2E63"
    pill_corner_radius: int = 16
    drop_shadow: DropShadowSpec = Field(default_factory=DropShadowSpec)


class LayerSpec(BaseModel):
    layer_id: str
    layer_type: LayerType
    z_index: int
    action: LayerAction
    preserve_pixels: bool = False
    mask_source: Optional[str] = None
    generative_params: Optional[GenerativeParams] = None
    relighting: Optional[RelightingSpec] = None
    drop_shadow: Optional[DropShadowSpec] = None
    typography_spec: Optional[TypographySpec] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GlobalColorPalette(BaseModel):
    primary_accent: str = "#FF2E63"
    secondary_accent: str = "#08D9D6"
    background_tone: str = "#0F172A"
    text_color: str = "#FFFFFF"
    contrast_ratio_achieved: float = Field(default=7.0, ge=1.0)


class CompositingDirectives(BaseModel):
    lut_preset: str = "teal_and_orange_subtle"
    saturation_scale: float = 1.15
    contrast_scale: float = 1.08
    sharpen_strength: float = 0.25
    vignette_intensity: float = 0.35


class EditPlan(BaseModel):
    plan_id: str
    timestamp: str
    target_dimensions: Tuple[int, int] = (1280, 720)
    archetype: Archetype = Archetype.SINGLE_CREATOR_FACE
    global_color_palette: GlobalColorPalette = Field(default_factory=GlobalColorPalette)
    layers: List[LayerSpec]
    compositing_directives: CompositingDirectives = Field(default_factory=CompositingDirectives)


class QualityReport(BaseModel):
    passed: bool
    identity_cosine_drift: float = Field(description="Cosine distance drift for creator face. Must be < 0.15")
    predicted_ctr_lift: float = Field(description="Predicted percentage lift in CTR")
    visual_contrast_score: float = Field(description="WCAG & visual hierarchy score (0-10)")
    saliency_balance_score: float = Field(description="Distribution score of visual attention")
    rejection_reasons: List[str] = Field(default_factory=list)
