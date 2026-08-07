"""
thumbnail_intelligence
======================

Thumbnail Intelligence Engine package for Thumbnail AI.
Provides the knowledge base, strategic reasoning, retrieval, evidence normalization,
and design brief generation subsystems for intelligent thumbnail optimization.

Phase 3.1: Knowledge Base Foundation.
Phase 3.2: Hybrid Retrieval Engine.
Phase 3.3: Evidence Normalization Engine.
"""

from __future__ import annotations

import thumbnail_intelligence.evidence as evidence
import thumbnail_intelligence.knowledge_base as knowledge_base
import thumbnail_intelligence.retrieval as retrieval

__version__ = "0.3.0"

__all__ = [
    "knowledge_base",
    "retrieval",
    "evidence",
]
