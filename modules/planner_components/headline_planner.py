"""
headline_planner.py
===================

Implements Section 6.4 conservative headline planning logic.
Pure logic, zero I/O.
"""

from __future__ import annotations

from typing import Optional

from models import (
    AssetExtractionManifest,
    HeadlineSource,
    RedesignSpecification,
    ThumbnailIntelligence,
)
from planner_components.interfaces import IHeadlinePlanner


class HeadlinePlanner(IHeadlinePlanner):
    """Plans headline text for the generation plan strictly per §6.4 rules."""

    def plan_headline(
        self,
        spec: RedesignSpecification,
        intelligence: Optional[ThumbnailIntelligence] = None,
        extraction_manifest: Optional[AssetExtractionManifest] = None,
    ) -> tuple[str, HeadlineSource]:
        """
        Derive (headline_text, headline_source).

        Rule (§6.4):
        - If spec.text_overlay.include_text is True:
          - If Module 8 typography assets present, use first non-empty text verbatim -> PRESERVED_OCR.
          - Else if Module 4 ocr text_regions present, use joined text verbatim -> PRESERVED_OCR.
          - Else -> ("", NONE).
        - If spec.text_overlay.include_text is False -> ("", NONE).
        """
        if not spec.text_overlay.include_text:
            return "", HeadlineSource.NONE

        # Try Module 8 typography assets first
        if extraction_manifest is not None and extraction_manifest.typography:
            for typo in extraction_manifest.typography:
                if typo.text and typo.text.strip():
                    return typo.text.strip(), HeadlineSource.PRESERVED_OCR

        # Try Module 4 OCR text regions next
        if intelligence is not None and intelligence.ocr and intelligence.ocr.text_regions:
            texts = [r.text.strip() for r in intelligence.ocr.text_regions if r.text and r.text.strip()]
            if texts:
                return " ".join(texts), HeadlineSource.PRESERVED_OCR

        return "", HeadlineSource.NONE
