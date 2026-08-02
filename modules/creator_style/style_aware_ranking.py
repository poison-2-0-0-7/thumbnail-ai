"""
StyleAwareRanking component for Phase 5 of Module 10 Creator Style Learning.

Extends multi-dimensional candidate ranking with bounded style similarity bonus.
Preserves existing hard-gate correctness validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from modules.config import MODULE10_STYLE_RANKING_WEIGHT, MODULE10_STYLE_SIMILARITY_THRESHOLD
from modules.creator_style.style_similarity import StyleSimilarityEngine
from modules.models import CreatorStyleEmbedding, StyleAwareScore


class StyleAwareRankingEngine:
    """
    Computes candidate style similarity scores and bounded bonus terms for ranking integration.
    """

    def __init__(self, similarity_engine: Optional[StyleSimilarityEngine] = None):
        self.similarity_engine = similarity_engine or StyleSimilarityEngine()

    def evaluate_candidates(
        self,
        video_id: str,
        channel_id: str,
        candidate_images: Sequence[tuple[int, str | Path]],
        profile_embedding: Optional[CreatorStyleEmbedding] = None,
        similarity_threshold: float = MODULE10_STYLE_SIMILARITY_THRESHOLD,
        style_weight: float = MODULE10_STYLE_RANKING_WEIGHT,
    ) -> list[StyleAwareScore]:
        """
        Evaluate candidate images and produce StyleAwareScore records.
        """
        scores: list[StyleAwareScore] = []

        for cand_idx, img_path in candidate_images:
            sim_res = self.similarity_engine.evaluate_similarity(
                video_id=video_id,
                channel_id=channel_id,
                candidate_image=img_path,
                profile_embedding=profile_embedding,
                similarity_threshold=similarity_threshold,
            )

            # Compute bounded style bonus: max(0, similarity - threshold) * weight
            if sim_res.profile_established and sim_res.similarity_score > similarity_threshold:
                bonus = (sim_res.similarity_score - similarity_threshold) * style_weight
            else:
                bonus = 0.0

            scores.append(
                StyleAwareScore(
                    candidate_index=cand_idx,
                    style_similarity=sim_res.similarity_score,
                    style_bonus=round(bonus, 4),
                )
            )

        return scores
