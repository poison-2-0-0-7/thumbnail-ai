"""
StyleDriftDetector component for Phase 6 of Module 10 Creator Style Learning.

Detects intentional creator visual style shifts over a sliding window of recent video thumbnails
reusing OpenCLIP vector similarity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from modules.config import MODULE10_STYLE_DRIFT_WINDOW, MODULE10_STYLE_SIMILARITY_THRESHOLD
from modules.creator_style.style_similarity import StyleSimilarityEngine
from modules.models import CreatorStyleEmbedding, StyleDriftAssessment


class StyleDriftDetector:
    """
    Evaluates running similarity history to detect intentional creator style drift.
    """

    def __init__(self, similarity_engine: Optional[StyleSimilarityEngine] = None):
        self.similarity_engine = similarity_engine or StyleSimilarityEngine()

    def assess_drift(
        self,
        channel_id: str,
        recent_image_inputs: Sequence[str | Path | list[float] | np.ndarray],
        profile_embedding: Optional[CreatorStyleEmbedding] = None,
        drift_window: int = MODULE10_STYLE_DRIFT_WINDOW,
        similarity_threshold: float = MODULE10_STYLE_SIMILARITY_THRESHOLD,
    ) -> StyleDriftAssessment:
        """
        Assess whether recent thumbnails indicate an intentional style shift.
        """
        if profile_embedding is None or not profile_embedding.embedding or profile_embedding.sample_count < drift_window:
            return StyleDriftAssessment(
                channel_id=channel_id,
                recent_similarity_scores=[],
                drift_detected=False,
                drift_confidence=0.0,
                recommended_action="none",
            )

        if len(recent_image_inputs) < drift_window:
            return StyleDriftAssessment(
                channel_id=channel_id,
                recent_similarity_scores=[],
                drift_detected=False,
                drift_confidence=0.0,
                recommended_action="none",
            )

        # Consider the last `drift_window` inputs
        window_inputs = recent_image_inputs[-drift_window:]
        embeddings: list[list[float]] = []
        similarity_scores: list[float] = []

        for item in window_inputs:
            if isinstance(item, list):
                emb = item
            elif isinstance(item, np.ndarray) and item.ndim == 1:
                emb = item.tolist()
            else:
                emb = self.similarity_engine.extract_image_embedding(item)
            embeddings.append(emb)

            sim = self.similarity_engine.compute_vector_similarity(emb, profile_embedding.embedding)
            similarity_scores.append(round(sim, 4))

        # Check if ALL scores in window are below similarity_threshold
        all_below_threshold = all(s < similarity_threshold for s in similarity_scores)

        if not all_below_threshold:
            has_some_below = any(s < similarity_threshold for s in similarity_scores)
            return StyleDriftAssessment(
                channel_id=channel_id,
                recent_similarity_scores=similarity_scores,
                drift_detected=False,
                drift_confidence=0.0,
                recommended_action="monitor" if has_some_below else "none",
            )

        # Check mutual similarity between recent window embeddings to guard against single outliers
        pairwise_sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                p_sim = self.similarity_engine.compute_vector_similarity(embeddings[i], embeddings[j])
                pairwise_sims.append(p_sim)

        avg_pairwise_sim = float(np.mean(pairwise_sims)) if pairwise_sims else 1.0

        # If recent thumbnails are mutually consistent (high pairwise similarity) but diverged from old centroid
        if avg_pairwise_sim >= (similarity_threshold - 0.10):
            return StyleDriftAssessment(
                channel_id=channel_id,
                recent_similarity_scores=similarity_scores,
                drift_detected=True,
                drift_confidence=0.85,
                recommended_action="update_centroid",
            )

        return StyleDriftAssessment(
            channel_id=channel_id,
            recent_similarity_scores=similarity_scores,
            drift_detected=False,
            drift_confidence=0.40,
            recommended_action="monitor",
        )
