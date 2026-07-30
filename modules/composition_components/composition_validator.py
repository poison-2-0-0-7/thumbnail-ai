"""
composition_validator.py
=========================

Performs structural and referential integrity validation on CompositionWorkspace
artifacts prior to persistence or downstream consumption.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from composition_components.interfaces import ICompositionValidator
from models import CompositionWorkspace, LayerDecision


class CompositionValidator(ICompositionValidator):
    """Validator ensuring structural, referential, and statistical correctness."""

    def validate(self, workspace: CompositionWorkspace) -> list[str]:
        """
        Validate workspace integrity.

        Returns:
            List of validation failure reason strings (empty if valid).
        """
        errors: list[str] = []

        # 1. Total layers check
        if not workspace.layers:
            errors.append("Workspace must contain at least one layer.")

        # 2. At least one kept/replaced layer check
        active_layers = [
            l for l in workspace.layers
            if l.placement.decision in (LayerDecision.KEEP, LayerDecision.ENHANCE, LayerDecision.REPLACE)
        ]
        if not active_layers:
            errors.append("Workspace must contain at least one active (keep, enhance, or replace) layer.")

        # 3. Referenced file paths & mask checksums
        for layer in workspace.layers:
            if layer.placement.source_path:
                path = Path(layer.placement.source_path)
                if not path.is_file():
                    errors.append(
                        f"Layer '{layer.layer_id}' references non-existent source_path: '{layer.placement.source_path}'"
                    )

            if layer.placement.mask:
                mask = layer.placement.mask
                mask_path = Path(mask.mask_path)
                if not mask_path.is_file():
                    errors.append(
                        f"Layer '{layer.layer_id}' references non-existent mask_path: '{mask.mask_path}'"
                    )
                else:
                    try:
                        digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
                        if digest.lower() != mask.mask_checksum.lower():
                            errors.append(
                                f"Layer '{layer.layer_id}' mask checksum mismatch: expected {mask.mask_checksum}, got {digest}"
                            )
                    except Exception as exc:
                        errors.append(f"Failed to read mask file for layer '{layer.layer_id}': {exc}")

            if layer.depth_hint_path:
                depth_path = Path(layer.depth_hint_path)
                if not depth_path.is_file():
                    errors.append(
                        f"Layer '{layer.layer_id}' references non-existent depth_hint_path: '{layer.depth_hint_path}'"
                    )

        # 4. Canvas dimensions check
        if workspace.canvas.width <= 0 or workspace.canvas.height <= 0:
            errors.append("Canvas dimensions must be positive.")

        # 5. Statistics verification
        stats = workspace.statistics
        if stats.total_layers != len(workspace.layers):
            errors.append(
                f"Statistics total_layers ({stats.total_layers}) does not match layer count ({len(workspace.layers)})."
            )

        return errors
