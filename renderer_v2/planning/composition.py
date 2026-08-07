"""Spatial layout, rule-of-thirds, negative space, and composition geometry engine."""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np

from .planner_types import CompositionAnalysis, CompositionDirectives


class CompositionEngine:
    """Evaluates spatial composition, rule of thirds, safe zones, and visual balance."""

    POWER_POINTS_NORMALIZED = [
        (1.0 / 3.0, 1.0 / 3.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (2.0 / 3.0, 2.0 / 3.0),
    ]

    @classmethod
    def evaluate_rule_of_thirds(cls, centroid_normalized: Tuple[float, float]) -> float:
        """Calculate rule-of-thirds alignment score in [0.0, 1.0].

        Score is 1.0 if the centroid falls directly on a power point, decreasing with distance.
        """
        cx, cy = centroid_normalized
        min_dist = float("inf")
        for px, py in cls.POWER_POINTS_NORMALIZED:
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist < min_dist:
                min_dist = dist

        # Maximum possible distance to a power point is ~0.47
        max_possible_dist = 0.4714
        alignment_score = max(0.0, 1.0 - (min_dist / max_possible_dist))
        return float(round(alignment_score, 4))

    @staticmethod
    def evaluate_subject_scale(
        subject_bbox: Optional[Tuple[int, int, int, int]],
        canvas_dimensions: Tuple[int, int],
    ) -> float:
        """Calculate subject canvas area fraction in [0.0, 1.0]."""
        if not subject_bbox:
            return 0.0

        w_img, h_img = canvas_dimensions
        xmin, ymin, xmax, ymax = subject_bbox
        bw = max(0, xmax - xmin)
        bh = max(0, ymax - ymin)
        subject_area = float(bw * bh)
        canvas_area = float(w_img * h_img)
        if canvas_area <= 0:
            return 0.0

        return float(round(min(1.0, subject_area / canvas_area), 4))

    @staticmethod
    def evaluate_visual_balance(
        saliency_map: np.ndarray,
    ) -> float:
        """Compute horizontal and vertical visual balance in [0.0, 1.0].

        1.0 means visual mass is symmetrically and pleasantly distributed.
        """
        h, w = saliency_map.shape
        half_w = w // 2
        half_h = h // 2

        left_mass = float(np.sum(saliency_map[:, :half_w]))
        right_mass = float(np.sum(saliency_map[:, half_w:]))
        top_mass = float(np.sum(saliency_map[:half_h, :]))
        bottom_mass = float(np.sum(saliency_map[half_h:, :]))

        h_balance = 1.0 - (abs(left_mass - right_mass) / (left_mass + right_mass + 1e-8))
        v_balance = 1.0 - (abs(top_mass - bottom_mass) / (top_mass + bottom_mass + 1e-8))

        balance_score = 0.65 * h_balance + 0.35 * v_balance
        return float(round(np.clip(balance_score, 0.0, 1.0), 4))

    @staticmethod
    def find_text_safe_zones(
        image: np.ndarray,
        subject_mask: Optional[np.ndarray],
        saliency_map: np.ndarray,
        margin_pct: float = 0.06,
    ) -> List[Tuple[int, int, int, int]]:
        """Identify uncluttered rectangular candidate safe zones for typography.

        Avoids:
        - Primary subject regions
        - High-saliency distractions
        - Outer border margin (margin_pct)
        - Bottom-right YouTube timestamp badge (x: 75%-100%, y: 80%-100%)

        Returns:
            List of valid bounding boxes (xmin, ymin, xmax, ymax) in pixel coordinates.
        """
        h, w = saliency_map.shape
        margin_x = int(w * margin_pct)
        margin_y = int(h * margin_pct)

        # Build occupancy map
        occupied = np.zeros((h, w), dtype=np.uint8)
        if subject_mask is not None:
            if subject_mask.shape != (h, w):
                subject_mask = cv2.resize(subject_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            occupied = np.maximum(occupied, (subject_mask > 0.1).astype(np.uint8))

        # Mark high-saliency regions as occupied
        saliency_thresh = np.percentile(saliency_map, 80)
        occupied = np.maximum(occupied, (saliency_map > saliency_thresh).astype(np.uint8))

        # Mark YouTube timestamp overlay (bottom-right 25% width, 22% height)
        yt_x0 = int(w * 0.75)
        yt_y0 = int(h * 0.78)
        occupied[yt_y0:, yt_x0:] = 1

        # Candidate standard thumbnail layout quadrants
        candidate_zones = [
            # Top-Left Quadrant (Headline Primary Zone)
            (margin_x, margin_y, int(w * 0.58), int(h * 0.48)),
            # Left Half Column (Vertical Stack)
            (margin_x, margin_y, int(w * 0.50), int(h * 0.88)),
            # Top-Right Quadrant
            (int(w * 0.45), margin_y, w - margin_x, int(h * 0.48)),
            # Right Half Column (if subject is on left)
            (int(w * 0.50), margin_y, w - margin_x, int(h * 0.75)),
            # Top Center Wide Banner
            (int(w * 0.15), margin_y, int(w * 0.85), int(h * 0.38)),
        ]

        valid_safe_zones: List[Tuple[int, int, int, int]] = []
        for xmin, ymin, xmax, ymax in candidate_zones:
            if xmax <= xmin or ymax <= ymin:
                continue
            sub_region = occupied[ymin:ymax, xmin:xmax]
            total_pixels = sub_region.size
            if total_pixels == 0:
                continue
            occupied_fraction = float(np.sum(sub_region)) / float(total_pixels)
            # A valid text safe zone must have less than 28% occupancy/clutter
            if occupied_fraction <= 0.28:
                valid_safe_zones.append((xmin, ymin, xmax, ymax))

        if not valid_safe_zones:
            # Fallback to default top-left safe zone
            valid_safe_zones.append((margin_x, margin_y, int(w * 0.55), int(h * 0.45)))

        return valid_safe_zones

    @staticmethod
    def classify_color_harmony(image: np.ndarray) -> str:
        """Deterministically classify thumbnail color harmony based on hue distribution."""
        # Convert RGB to HSV
        small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hues = hsv[:, :, 0].flatten().astype(np.float32) * 2.0  # [0, 360)

        # Compute hue histogram with 12 bins of 30 degrees
        hist, _ = np.histogram(hues, bins=12, range=(0, 360))
        top_bins = np.argsort(hist)[::-1]
        dominant_hue_1 = top_bins[0] * 30 + 15
        dominant_hue_2 = top_bins[1] * 30 + 15

        hue_diff = abs(dominant_hue_1 - dominant_hue_2)
        if hue_diff > 180:
            hue_diff = 360 - hue_diff

        if 150 <= hue_diff <= 180:
            return "complementary"
        elif 120 <= hue_diff < 150:
            return "split_complementary"
        elif 90 <= hue_diff < 120:
            return "triadic"
        elif 30 <= hue_diff < 90:
            return "analogous"
        else:
            return "monochromatic"

    @classmethod
    def analyze_scene(
        cls,
        image: np.ndarray,
        subject_mask: Optional[np.ndarray],
        subject_bbox: Optional[Tuple[int, int, int, int]],
        depth_map: Optional[np.ndarray],
        saliency_map: np.ndarray,
    ) -> CompositionAnalysis:
        """Perform comprehensive deterministic composition analysis."""
        h, w, _ = image.shape
        dimensions = (w, h)

        # Subject scale
        subject_scale = cls.evaluate_subject_scale(subject_bbox, dimensions)

        # Subject position centroid
        if subject_bbox:
            xmin, ymin, xmax, ymax = subject_bbox
            cx = ((xmin + xmax) / 2.0) / w
            cy = ((ymin + ymax) / 2.0) / h
            subject_pos = (float(round(cx, 4)), float(round(cy, 4)))
        else:
            subject_pos = (0.5, 0.5)

        # Rule of thirds alignment
        rot_score = cls.evaluate_rule_of_thirds(subject_pos)

        # Visual balance
        balance_score = cls.evaluate_visual_balance(saliency_map)

        # Text safe zones
        safe_zones = cls.find_text_safe_zones(image, subject_mask, saliency_map)
        safe_zone_available = len(safe_zones) > 0

        # Negative space ratio
        if subject_mask is not None:
            if subject_mask.shape != (h, w):
                subject_mask = cv2.resize(subject_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            bg_pixels = float(np.sum(subject_mask <= 0.1))
            total_pixels = float(h * w)
            neg_space = bg_pixels / (total_pixels + 1e-8)
        else:
            neg_space = 1.0 - subject_scale
        neg_space = float(round(np.clip(neg_space, 0.0, 1.0), 4))

        # Contrast ratio
        if subject_mask is not None:
            from .saliency import SaliencyEngine
            contrast_ratio = SaliencyEngine.compute_contrast_ratio(image, subject_mask)
        else:
            contrast_ratio = 3.0

        # Hierarchy clarity: difference in saliency between subject and background
        if subject_mask is not None:
            fg_sal = float(np.mean(saliency_map[subject_mask > 0.1])) if np.any(subject_mask > 0.1) else 0.5
            bg_sal = float(np.mean(saliency_map[subject_mask <= 0.1])) if np.any(subject_mask <= 0.1) else 0.2
            hierarchy_clarity = float(round(np.clip((fg_sal - bg_sal + 0.5), 0.0, 1.0), 4))
            focus_score = float(round(np.clip(fg_sal, 0.0, 1.0), 4))
        else:
            hierarchy_clarity = 0.5
            focus_score = 0.5

        # Attention direction
        from .saliency import SaliencyEngine
        attention_dir = SaliencyEngine.determine_attention_flow(subject_pos, [(0.25, 0.25)])

        # Color harmony
        color_harmony = cls.classify_color_harmony(image)

        # CTR improvement potential
        # High potential if subject scale is too low, contrast is weak, or rule of thirds is poor
        scale_gap = max(0.0, 0.35 - subject_scale)
        rot_gap = max(0.0, 0.85 - rot_score)
        contrast_gap = max(0.0, (4.5 - min(contrast_ratio, 4.5)) / 4.5)
        ctr_potential = float(round(np.clip(0.4 * scale_gap + 0.3 * rot_gap + 0.3 * contrast_gap, 0.0, 1.0), 4))

        return CompositionAnalysis(
            subject_scale=subject_scale,
            subject_position=subject_pos,
            rule_of_thirds_alignment=rot_score,
            negative_space_ratio=neg_space,
            text_safe_zone_available=safe_zone_available,
            text_safe_zones=safe_zones,
            hierarchy_clarity=hierarchy_clarity,
            contrast_ratio=contrast_ratio,
            visual_balance=balance_score,
            focus_score=focus_score,
            attention_direction=attention_dir,
            color_harmony=color_harmony,
            ctr_improvement_potential=ctr_potential,
        )
