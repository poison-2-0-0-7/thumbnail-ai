"""
query.py
========

Query specifications, contextual search descriptors, and structured filter definitions
for the Hybrid Retrieval Engine.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field, field_validator

from thumbnail_intelligence.knowledge_base.models import (
    BaseKBModel,
    EvidenceGrade,
    KnowledgeEntryType,
    _utc_now_iso,
)
from thumbnail_intelligence.retrieval.exceptions import InvalidQueryError


class QueryContext(BaseKBModel):
    """
    Contextual metadata about the video, creator, channel, and query state
    informing retrieval relevance and personalization.
    """

    video_id: Optional[str] = Field(default=None, description="Target video ID being analyzed")
    channel_id: Optional[str] = Field(default=None, description="Current creator channel ID")
    creator_id: Optional[str] = Field(default=None, description="Parent creator identity")
    niche: str = Field(default="general", description="Content niche or topic domain")
    title: Optional[str] = Field(default=None, description="Video title string")
    headline: Optional[str] = Field(default=None, description="Proposed headline or copy hook")
    tags: List[str] = Field(default_factory=list, description="Content tags or descriptive keywords")
    archetype_id: Optional[str] = Field(default=None, description="Matched archetype if known")
    query_time: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp of query")


class SearchFilters(BaseKBModel):
    """
    Deterministic hard constraints evaluated during Stage 1 filter execution.
    Only entries satisfying all specified filters enter similarity scoring.
    """

    entry_types: List[KnowledgeEntryType] = Field(
        default_factory=list, description="Target knowledge entry types to retrieve"
    )
    niche: Optional[str] = Field(default=None, description="Required niche (or 'general' fallback)")
    channel_id: Optional[str] = Field(default=None, description="Channel filter for own-history queries")
    creator_id: Optional[str] = Field(default=None, description="Creator ID filter")
    archetype_id: Optional[str] = Field(default=None, description="Archetype filter")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum model confidence floor")
    min_evidence_grade: Optional[EvidenceGrade] = Field(
        default=None, description="Minimum required empirical evidence grade"
    )
    exclude_ids: List[str] = Field(default_factory=list, description="Entry IDs to exclude from retrieval")
    date_from: Optional[str] = Field(default=None, description="ISO-8601 lower timestamp boundary")
    date_to: Optional[str] = Field(default=None, description="ISO-8601 upper timestamp boundary")
    custom_facets: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary facet key-value constraints")


class RetrievalQuery(BaseKBModel):
    """
    Complete structured query payload for hybrid multi-modal retrieval.
    Encapsulates visual vector embedding, keyword text, hard filters, and top-k bounding.
    """

    query_id: str = Field(description="Unique deterministic or generated query identifier")
    query_embedding: List[float] = Field(default_factory=list, description="512-dim visual or multimodal query vector")
    text_query: Optional[str] = Field(default=None, description="Optional textual keyword search query")
    context: QueryContext = Field(default_factory=QueryContext, description="Contextual search metadata")
    filters: SearchFilters = Field(default_factory=SearchFilters, description="Deterministic stage 1 filters")
    top_k: int = Field(default=8, ge=1, le=100, description="Maximum number of retrieved results to return")
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity floor")
    weights_override: Optional[Dict[str, float]] = Field(
        default=None, description="Optional ranking weights override"
    )
    deduplicate: bool = Field(default=True, description="Whether to apply semantic and ID deduplication")

    @field_validator("query_id")
    @classmethod
    def validate_query_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query_id must not be empty")
        return v.strip()

    @field_validator("query_embedding")
    @classmethod
    def validate_query_embedding(cls, v: List[float]) -> List[float]:
        for idx, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"query_embedding elements must be floats. Got {type(val)} at index {idx}")
            if val != val:  # NaN check
                raise ValueError(f"query_embedding element at index {idx} is NaN")
        return v

    def compute_cache_key(self) -> str:
        """
        Compute deterministic SHA-256 fingerprint for retrieval caching.
        """
        hasher = hashlib.sha256()
        # Hash text query & context
        text = f"{self.text_query}|{self.context.niche}|{self.context.channel_id}|{self.context.archetype_id}|{self.top_k}|{self.min_similarity}"
        hasher.update(text.encode("utf-8"))

        # Hash filters
        filter_str = json.dumps(self.filters.to_dict(), sort_keys=True)
        hasher.update(filter_str.encode("utf-8"))

        # Hash embedding vector if present
        if self.query_embedding:
            # Hash downsampled/rounded floats to maintain stability
            emb_str = ",".join(f"{round(x, 4):.4f}" for x in self.query_embedding)
            hasher.update(emb_str.encode("utf-8"))

        return hasher.hexdigest()
