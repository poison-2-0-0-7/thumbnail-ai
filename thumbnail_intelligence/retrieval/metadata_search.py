"""
metadata_search.py
==================

Metadata and lexical keyword search engine for the Hybrid Retrieval Engine.
Scans knowledge entries, applies hard stage-1 filters, and computes token-overlap
keyword match scores against titles, tags, facets, and descriptions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from thumbnail_intelligence.knowledge_base.models import BaseKBModel
from thumbnail_intelligence.retrieval.filters import MetadataFilterEngine
from thumbnail_intelligence.retrieval.query import RetrievalQuery


class MetadataSearchEngine:
    """
    Executes fast metadata and lexical keyword filtering across candidate knowledge entries.
    """

    def __init__(self, filter_engine: Optional[MetadataFilterEngine] = None) -> None:
        self.filter_engine = filter_engine or MetadataFilterEngine()

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Tokenize text into lowercase alphanumeric words."""
        if not text:
            return set()
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        return set(words)

    @classmethod
    def compute_keyword_score(cls, entry: Any, query_text: Optional[str]) -> Tuple[float, List[str]]:
        """
        Compute lexical token overlap score in [0.0, 1.0] and return matched keyword terms.
        """
        if not query_text or not query_text.strip():
            return 0.0, []

        q_tokens = cls._tokenize(query_text)
        if not q_tokens:
            return 0.0, []

        # Extract text attributes from candidate entry
        raw_parts = [
            getattr(entry, "name", None),
            getattr(entry, "description", None),
            getattr(entry, "claim", None),
            getattr(entry, "display_name", None),
            getattr(entry, "channel_name", None),
            getattr(entry, "typical_emotion", None),
        ]
        entry_text_parts = [str(p) for p in raw_parts if p is not None]

        # Include list of hook types or tags
        hooks = getattr(entry, "typical_hook_types", [])
        if isinstance(hooks, list):
            entry_text_parts.extend([str(h) for h in hooks if h is not None])

        facets = getattr(entry, "facets", {})
        if isinstance(facets, dict):
            entry_text_parts.extend([str(k) + " " + str(v) for k, v in facets.items() if v is not None])

        full_entry_text = " ".join(entry_text_parts)
        entry_tokens = cls._tokenize(full_entry_text)

        matched = q_tokens.intersection(entry_tokens)
        if not matched:
            return 0.0, []

        # Jaccard / Overlap score
        overlap_score = len(matched) / len(q_tokens)
        return min(1.0, overlap_score), sorted(list(matched))

    def search(
        self,
        candidates: List[Any],
        query: RetrievalQuery,
    ) -> List[Tuple[Any, float, List[str]]]:
        """
        Filter candidates using SearchFilters and compute keyword match score.
        Returns list of (entry, keyword_score, matched_terms).
        """
        results: List[Tuple[Any, float, List[str]]] = []

        # Filter candidates using stage 1 hard predicates
        filtered = self.filter_engine.filter_entries(candidates, query.filters)

        query_text = query.text_query or (query.context.title if query.context else None)

        for entry in filtered:
            kw_score, matched = self.compute_keyword_score(entry, query_text)
            results.append((entry, kw_score, matched))

        return results
