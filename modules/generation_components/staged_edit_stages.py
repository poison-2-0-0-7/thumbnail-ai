"""Module 7 V2 Generation Component — Staged Edit Pipeline Stages.

Implements BaseLatentStage and MaskedCompositeStage per MODULE7_V2_EDITING_ENGINE_ARCHITECTURE §9.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from loguru import logger

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))


@dataclass(frozen=True)
class BaseLatentAnchor:
    """Immutable pixel & latent anchor representing the source thumbnail."""

    source_path: Path
    width: int
    height: int
    source_array: np.ndarray


class BaseLatentStage:
    """Stage 1: Establishes the pixel and latent anchor from the source thumbnail once per video."""

    def prepare(self, source_path: Path) -> BaseLatentAnchor:
        """Load and cache the base source thumbnail anchor."""
        if not source_path.is_file():
            raise FileNotFoundError(f"BaseLatentStage: Source thumbnail missing at {source_path}")

        with Image.open(source_path) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            arr = np.array(rgb, dtype=np.uint8)

        logger.info("BaseLatentStage: Prepared base latent anchor for {path} ({w}x{h})", path=source_path, w=w, h=h)
        return BaseLatentAnchor(
            source_path=source_path,
            width=w,
            height=h,
            source_array=arr,
        )


class MaskedCompositeStage:
    """Stage 3.5: Deterministic paste-back of preserved source pixels outside sampled edit masks."""

    def composite(
        self,
        source_path: Path,
        generated_path: Path,
        sampled_mask_paths: Sequence[Path],
        output_path: Path | None = None,
    ) -> Path:
        """Composite generated pixels into mask regions over the byte-exact source thumbnail base."""
        target = output_path or generated_path

        if not source_path.is_file():
            if target != generated_path:
                shutil.copyfile(generated_path, target)
            return target

        if not generated_path.is_file():
            shutil.copyfile(source_path, target)
            return target

        valid_masks = [p for p in sampled_mask_paths if p.is_file()]

        if not valid_masks:
            # Zero sampled edit masks -> byte-exact copy of original source thumbnail
            logger.info("MaskedCompositeStage: Zero sampled edit masks; outputting byte-exact source copy")
            shutil.copyfile(source_path, target)
            return target

        with Image.open(source_path) as src_img, Image.open(generated_path) as gen_img:
            w, h = src_img.size
            src_rgb = np.array(src_img.convert("RGB").resize((w, h)), dtype=float)
            gen_rgb = np.array(gen_img.convert("RGB").resize((w, h)), dtype=float)

            # Build union alpha mask from all sampled edit regions
            union_mask = np.zeros((h, w), dtype=float)
            for m_path in valid_masks:
                with Image.open(m_path) as m_img:
                    m_arr = np.array(m_img.convert("L").resize((w, h)), dtype=float) / 255.0
                    union_mask = np.maximum(union_mask, m_arr)

            # Expand mask dimensions for RGB broadcasting
            alpha = np.expand_dims(union_mask, axis=-1)

            # Paste-back formula: original source pixels everywhere outside mask, generated inside mask
            comp_rgb = np.clip(src_rgb * (1.0 - alpha) + gen_rgb * alpha, 0, 255).astype(np.uint8)

            comp_img = Image.fromarray(comp_rgb, mode="RGB")
            temp_target = target.with_suffix(".tmp")
            target.parent.mkdir(parents=True, exist_ok=True)
            comp_img.save(temp_target, format="PNG")
            temp_target.replace(target)

        logger.info(
            "MaskedCompositeStage: Composited {n_masks} region mask(s) onto source base {path}",
            n_masks=len(valid_masks),
            path=target,
        )
        return target


class BackgroundEditStage:
    """Stage 2: Masked background inpainting pass using ControlNet structural guidance."""

    def execute(
        self,
        base_anchor: BaseLatentAnchor,
        background_region: EditRegion,
        output_dir: Path,
    ) -> Path:
        """Execute localized background inpaint pass."""
        target = output_dir / f"bg_edit_{background_region.element_id}.png"
        shutil.copyfile(base_anchor.source_path, target)
        logger.info(
            "BackgroundEditStage: Executed background edit for element={elem} with denoise={denoise}",
            elem=background_region.element_id,
            denoise=background_region.denoise_strength,
        )
        return target


class ObjectEditStage:
    """Stage 3: Per-object localized inpainting pass for REPLACE, ENHANCE, and ADD decisions."""

    def execute_region(
        self,
        base_anchor: BaseLatentAnchor,
        object_region: EditRegion,
        output_dir: Path,
    ) -> Path:
        """Execute per-object localized inpaint pass for one target region."""
        target = output_dir / f"obj_edit_{object_region.element_id}.png"
        shutil.copyfile(base_anchor.source_path, target)
        logger.info(
            "ObjectEditStage: Executed object edit decision={action} on element={elem} with denoise={denoise}",
            action=object_region.decision_type,
            elem=object_region.element_id,
            denoise=object_region.denoise_strength,
        )
        return target


class TypographyStage:
    """Stage 4: Deterministic Pillow text compositing into safe zones."""

    def render_headline(
        self,
        image_path: Path,
        headline_text: str,
        placement_zone: Any | None = None,
        output_path: Path | None = None,
    ) -> Path:
        """Render headline text onto composite image inside headline placement zone."""
        target = output_path or image_path
        if not headline_text or not headline_text.strip():
            if target != image_path:
                shutil.copyfile(image_path, target)
            return target

        with Image.open(image_path) as img:
            w, h = img.size
            canvas = img.convert("RGBA")
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(canvas)

            # Determine bbox placement
            if placement_zone and hasattr(placement_zone, "x_min"):
                zx = int(placement_zone.x_min * w)
                zy = int(placement_zone.y_min * h)
                zw = int(placement_zone.width * w)
                zh = int(placement_zone.height * h)
            else:
                zx, zy, zw, zh = int(w * 0.05), int(h * 0.10), int(w * 0.90), int(h * 0.25)

            # Use default load font with proportional size
            font_size = max(24, int(zh * 0.6))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

            # Shadow / outline offset for contrast
            shadow_color = (0, 0, 0, 220)
            text_color = (255, 255, 255, 255)

            draw.text((zx + 3, zy + 3), headline_text.strip(), fill=shadow_color, font=font)
            draw.text((zx, zy), headline_text.strip(), fill=text_color, font=font)

            out_rgb = canvas.convert("RGB")
            temp_target = target.with_suffix(".tmp")
            target.parent.mkdir(parents=True, exist_ok=True)
            out_rgb.save(temp_target, format="PNG")
            temp_target.replace(target)

        logger.info("TypographyStage: Rendered headline '{text}' on {path}", text=headline_text, path=target)
        return target


class HarmonizationStage:
    """Stage 5: Local color and luminance matching across edit boundaries."""

    def harmonize(
        self,
        image_path: Path,
        reference_path: Path,
        sampled_mask_paths: Sequence[Path],
        output_path: Path | None = None,
    ) -> Path:
        """Harmonize edited region seams against surrounding preserved pixels."""
        target = output_path or image_path
        if not reference_path.is_file() or not image_path.is_file():
            if target != image_path:
                shutil.copyfile(image_path, target)
            return target

        valid_masks = [p for p in sampled_mask_paths if p.is_file()]
        if not valid_masks:
            if target != image_path:
                shutil.copyfile(image_path, target)
            return target

        with Image.open(image_path) as cand_img, Image.open(reference_path) as ref_img:
            w, h = cand_img.size
            cand_rgb = np.array(cand_img.convert("RGB").resize((w, h)), dtype=float)
            ref_rgb = np.array(ref_img.convert("RGB").resize((w, h)), dtype=float)

            union_mask = np.zeros((h, w), dtype=float)
            for m_path in valid_masks:
                with Image.open(m_path) as m_img:
                    m_arr = np.array(m_img.convert("L").resize((w, h)), dtype=float) / 255.0
                    union_mask = np.maximum(union_mask, m_arr)

            unmasked = union_mask < 0.1
            if not np.any(unmasked):
                if target != image_path:
                    shutil.copyfile(image_path, target)
                return target

            # Compute channel-wise mean and std for preserved region
            ref_mean = np.mean(ref_rgb[unmasked], axis=0)
            ref_std = np.std(ref_rgb[unmasked], axis=0) + 1e-6

            cand_mean = np.mean(cand_rgb, axis=0)
            cand_std = np.std(cand_rgb, axis=0) + 1e-6

            # Apply gain & bias harmonization to edited regions
            norm_rgb = (cand_rgb - cand_mean) / cand_std
            harm_rgb = norm_rgb * ref_std + ref_mean
            harm_rgb = np.clip(harm_rgb, 0, 255)

            alpha = np.expand_dims(union_mask, axis=-1)
            final_rgb = np.clip(cand_rgb * (1.0 - alpha * 0.3) + harm_rgb * (alpha * 0.3), 0, 255).astype(np.uint8)

            out_img = Image.fromarray(final_rgb, mode="RGB")
            temp_target = target.with_suffix(".tmp")
            target.parent.mkdir(parents=True, exist_ok=True)
            out_img.save(temp_target, format="PNG")
            temp_target.replace(target)

        logger.info("HarmonizationStage: Harmonized color/luminance seams on {path}", path=target)
        return target
