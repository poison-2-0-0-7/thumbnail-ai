"""
repository.py
=============

Central typed repository for the Thumbnail Intelligence Knowledge Base.
Composes individual typed registries and storage managers for:
- KnowledgeEntry records (unified multimodal corpus)
- CreatorProfile & ChannelProfile records
- CompetitorProfile records
- Archetype library
- BrandConstraint & IdentityConstraint records
- VisualPattern, DesignPattern, and ThumbnailPattern libraries

Provides dependency injection, seed data bootstrap, and high-level retrieval interfaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from thumbnail_intelligence.knowledge_base.config import KnowledgeBaseConfig
from thumbnail_intelligence.knowledge_base.exceptions import EntryNotFoundError
from thumbnail_intelligence.knowledge_base.models import (
    Archetype,
    BrandConstraint,
    ChannelProfile,
    CompetitorProfile,
    CreatorProfile,
    DesignPattern,
    EvidenceGrade,
    IdentityConstraint,
    KnowledgeEntry,
    KnowledgeEntryType,
    PatternScope,
    ThumbnailPattern,
    VisualPattern,
)
from thumbnail_intelligence.knowledge_base.registry import IndexHook, KnowledgeRegistry
from thumbnail_intelligence.knowledge_base.storage import VersionedFileStorage


class KnowledgeBaseRepository:
    """
    Central repository orchestrating all Knowledge Base entities and storage namespaces.
    Exposes unified lifecycle management, validation, and domain-specific query methods.
    """

    def __init__(self, config: Optional[KnowledgeBaseConfig] = None) -> None:
        self.config = config or KnowledgeBaseConfig()
        self.config.ensure_directories()

        # Initialize storage namespaces
        self._entry_storage = VersionedFileStorage(
            model_cls=KnowledgeEntry,
            namespace="entries",
            base_dir=self.config.base_dir,
            config=self.config,
        )
        self._creator_storage = VersionedFileStorage(
            model_cls=CreatorProfile,
            namespace="creator_profiles",
            base_dir=self.config.base_dir,
            config=self.config,
        )
        self._channel_storage = VersionedFileStorage(
            model_cls=ChannelProfile,
            namespace="channel_profiles",
            base_dir=self.config.base_dir,
            config=self.config,
        )
        self._competitor_storage = VersionedFileStorage(
            model_cls=CompetitorProfile,
            namespace="competitors",
            base_dir=self.config.base_dir,
            config=self.config,
        )
        self._archetype_storage = VersionedFileStorage(
            model_cls=Archetype,
            namespace="archetypes",
            base_dir=self.config.base_dir,
            config=self.config,
        )
        self._brand_storage = VersionedFileStorage(
            model_cls=BrandConstraint,
            namespace="brand_constraints",
            base_dir=self.config.base_dir,
            config=self.config,
        )
        self._identity_storage = VersionedFileStorage(
            model_cls=IdentityConstraint,
            namespace="identity_constraints",
            base_dir=self.config.base_dir,
            config=self.config,
        )
        self._visual_pattern_storage = VersionedFileStorage(
            model_cls=VisualPattern,
            namespace="visual_patterns",
            base_dir=self.config.base_dir,
            config=self.config,
        )
        self._design_pattern_storage = VersionedFileStorage(
            model_cls=DesignPattern,
            namespace="design_patterns",
            base_dir=self.config.base_dir,
            config=self.config,
        )
        self._thumbnail_pattern_storage = VersionedFileStorage(
            model_cls=ThumbnailPattern,
            namespace="thumbnail_patterns",
            base_dir=self.config.base_dir,
            config=self.config,
        )

        # Initialize typed registries
        self.entries: KnowledgeRegistry[KnowledgeEntry] = KnowledgeRegistry(
            model_cls=KnowledgeEntry,
            id_field="entry_id",
            storage=self._entry_storage,
        )
        self.creator_profiles: KnowledgeRegistry[CreatorProfile] = KnowledgeRegistry(
            model_cls=CreatorProfile,
            id_field="creator_id",
            storage=self._creator_storage,
        )
        self.channel_profiles: KnowledgeRegistry[ChannelProfile] = KnowledgeRegistry(
            model_cls=ChannelProfile,
            id_field="channel_id",
            storage=self._channel_storage,
        )
        self.competitor_profiles: KnowledgeRegistry[CompetitorProfile] = KnowledgeRegistry(
            model_cls=CompetitorProfile,
            id_field="competitor_id",
            storage=self._competitor_storage,
        )
        self.archetypes: KnowledgeRegistry[Archetype] = KnowledgeRegistry(
            model_cls=Archetype,
            id_field="archetype_id",
            storage=self._archetype_storage,
        )
        self.brand_constraints: KnowledgeRegistry[BrandConstraint] = KnowledgeRegistry(
            model_cls=BrandConstraint,
            id_field="constraint_id",
            storage=self._brand_storage,
        )
        self.identity_constraints: KnowledgeRegistry[IdentityConstraint] = KnowledgeRegistry(
            model_cls=IdentityConstraint,
            id_field="constraint_id",
            storage=self._identity_storage,
        )
        self.visual_patterns: KnowledgeRegistry[VisualPattern] = KnowledgeRegistry(
            model_cls=VisualPattern,
            id_field="pattern_id",
            storage=self._visual_pattern_storage,
        )
        self.design_patterns: KnowledgeRegistry[DesignPattern] = KnowledgeRegistry(
            model_cls=DesignPattern,
            id_field="pattern_id",
            storage=self._design_pattern_storage,
        )
        self.thumbnail_patterns: KnowledgeRegistry[ThumbnailPattern] = KnowledgeRegistry(
            model_cls=ThumbnailPattern,
            id_field="pattern_id",
            storage=self._thumbnail_pattern_storage,
        )

    # -----------------------------------------------------------------------
    # High-level domain queries
    # -----------------------------------------------------------------------

    def get_creator_profile(self, creator_id: str) -> Optional[CreatorProfile]:
        """Lookup CreatorProfile by creator identifier."""
        return self.creator_profiles.lookup(creator_id)

    def get_channel_profile(self, channel_id: str) -> Optional[ChannelProfile]:
        """Lookup ChannelProfile by channel identifier."""
        return self.channel_profiles.lookup(channel_id)

    def get_competitor_profile(self, competitor_id: str) -> Optional[CompetitorProfile]:
        """Lookup CompetitorProfile by competitor identifier."""
        return self.competitor_profiles.lookup(competitor_id)

    def get_archetype(self, archetype_id: str) -> Optional[Archetype]:
        """Lookup Archetype definition by archetype identifier."""
        return self.archetypes.lookup(archetype_id)

    def find_entries_by_type(
        self,
        entry_type: KnowledgeEntryType,
        niche: Optional[str] = None,
    ) -> List[KnowledgeEntry]:
        """Retrieve KnowledgeEntry records matching entry type and optional niche filter."""
        return self.entries.list(
            filter_fn=lambda e: e.entry_type == entry_type and (niche is None or e.niche == niche or e.niche == "general")
        )

    def find_patterns_by_niche(self, niche: str) -> List[DesignPattern]:
        """Retrieve DesignPatterns observed or applicable to a specific niche."""
        return self.design_patterns.list(
            filter_fn=lambda p: niche in p.frequency_in_niche or "general" in p.frequency_in_niche or p.curated
        )

    def get_brand_constraint_for_channel(self, channel_id: str) -> Optional[BrandConstraint]:
        """Retrieve BrandConstraint associated with a specific channel."""
        matches = self.brand_constraints.list(filter_fn=lambda b: b.channel_id == channel_id)
        return matches[0] if matches else None

    def get_identity_constraint_for_creator(self, creator_id: str) -> Optional[IdentityConstraint]:
        """Retrieve IdentityConstraint for a creator."""
        matches = self.identity_constraints.list(filter_fn=lambda i: i.creator_id == creator_id)
        return matches[0] if matches else None

    # -----------------------------------------------------------------------
    # Production Seed Data Bootstrap
    # -----------------------------------------------------------------------

    def seed_default_archetypes(self) -> List[Archetype]:
        """
        Bootstrap the curated standard Archetype library (§7.1, renderer v2 §3.3).
        Seeds high-signal, industry-standard thumbnail archetypes with structural predicates.
        """
        seeds = [
            Archetype(
                archetype_id="big_face_reaction",
                name="Big Face Reaction",
                description="High-emotion facial reaction occupying primary focal area with uncluttered negative space.",
                defining_scene_graph_pattern={
                    "hero_role": "hero",
                    "hero_bbox_area_min": 0.30,
                    "text_element_count_max": 2,
                    "face_emotion_required": True,
                },
                typical_hook_types=["reaction", "shock", "curiosity"],
                typical_emotion="surprise",
                niches_observed_in=["entertainment", "gaming", "vlog", "tech", "general"],
                centroid_embedding=[0.0] * 512,
                example_count=50,
            ),
            Archetype(
                archetype_id="before_after_split",
                name="Before & After Split Screen",
                description="Side-by-side or split contrast depicting transformation or result progression.",
                defining_scene_graph_pattern={
                    "split_layout": True,
                    "primary_element_count_min": 2,
                    "contrast_separation_min": 0.40,
                },
                typical_hook_types=["transformation", "comparison", "result"],
                typical_emotion="curiosity",
                niches_observed_in=["fitness", "diy", "tech_review", "education", "finance"],
                centroid_embedding=[0.0] * 512,
                example_count=35,
            ),
            Archetype(
                archetype_id="curiosity_gap",
                name="Curiosity Gap",
                description="Partial reveal or teasing visual element paired with concise question hook.",
                defining_scene_graph_pattern={
                    "hero_role": "primary",
                    "text_word_count_max": 4,
                    "mystery_element_present": True,
                },
                typical_hook_types=["curiosity", "question", "open_loop"],
                typical_emotion="curiosity",
                niches_observed_in=["documentary", "science", "true_crime", "business", "general"],
                centroid_embedding=[0.0] * 512,
                example_count=42,
            ),
            Archetype(
                archetype_id="expert_authority",
                name="Expert Authority",
                description="Clean, authoritative subject presentation with credential graphic and high-contrast typography.",
                defining_scene_graph_pattern={
                    "hero_placement": "left_third",
                    "lighting_mood": "studio_clean",
                    "badge_or_credential_present": True,
                },
                typical_hook_types=["authority", "warning", "strategy"],
                typical_emotion="focus",
                niches_observed_in=["business", "finance", "programming", "medical", "education"],
                centroid_embedding=[0.0] * 512,
                example_count=28,
            ),
            Archetype(
                archetype_id="tutorial_result",
                name="Tutorial & Showcase Result",
                description="Hero showcase of the finished product, project, or outcome with hero subject demonstrating.",
                defining_scene_graph_pattern={
                    "product_bbox_area_min": 0.25,
                    "subject_interacting_with_product": True,
                },
                typical_hook_types=["how_to", "demonstration", "proof"],
                typical_emotion="excitement",
                niches_observed_in=["diy", "cooking", "gaming", "coding", "art"],
                centroid_embedding=[0.0] * 512,
                example_count=30,
            ),
        ]

        registered: List[Archetype] = []
        for arch in seeds:
            if not self.archetypes.exists(arch.archetype_id):
                self.archetypes.register(arch, allow_overwrite=True)
                registered.append(arch)
            else:
                registered.append(self.archetypes.get(arch.archetype_id))
        return registered

    def seed_default_patterns(self) -> Tuple[List[VisualPattern], List[DesignPattern]]:
        """
        Bootstrap the curated standard VisualPattern and DesignPattern libraries (§14, §15).
        """
        v_seeds = [
            VisualPattern(
                pattern_id="rim_light_subject_edge",
                name="Rim Light Subject Edge",
                description="High-intensity color-matched edge illumination separating subject from dark backdrop.",
                category="lighting",
                visual_techniques=["edge_lighting", "chromatic_rim", "contrast_boost"],
                applicable_roles=["hero", "primary"],
                frequency_by_niche={"gaming": 0.85, "tech": 0.70, "entertainment": 0.60},
                centroid_embedding=[0.0] * 512,
                curated=True,
                evidence_grade=EvidenceGrade.STRONG,
            ),
            VisualPattern(
                pattern_id="high_contrast_vignette",
                name="High Contrast Radial Vignette",
                description="Subtle dark radial gradient around perimeter directing focus toward center focal anchor.",
                category="framing",
                visual_techniques=["radial_vignette", "focus_shaping", "luminance_falloff"],
                applicable_roles=["background"],
                frequency_by_niche={"documentary": 0.75, "true_crime": 0.90, "business": 0.50},
                centroid_embedding=[0.0] * 512,
                curated=True,
                evidence_grade=EvidenceGrade.STRONG,
            ),
        ]

        d_seeds = [
            DesignPattern(
                pattern_id="curiosity_gap_partial_reveal",
                pattern_scope=PatternScope.AUDIENCE_PSYCHOLOGY,
                name="Curiosity Gap Partial Reveal",
                description="Visually teases the outcome without revealing key detail, stimulating click intent.",
                applicable_element_types=["object", "text", "prop"],
                frequency_in_niche={"entertainment": 0.65, "tech": 0.55, "documentary": 0.80},
                curated=True,
            ),
            DesignPattern(
                pattern_id="arrow_circle_callout",
                pattern_scope=PatternScope.VISUAL_DESIGN,
                name="Arrow and Circle Focal Callout",
                description="High-visibility vector highlight pointing directly to a specific detail or anomalous object.",
                applicable_element_types=["prop", "object", "background"],
                frequency_in_niche={"gaming": 0.70, "tutorial": 0.60, "reaction": 0.50},
                curated=True,
            ),
        ]

        v_reg: List[VisualPattern] = []
        for vp in v_seeds:
            if not self.visual_patterns.exists(vp.pattern_id):
                self.visual_patterns.register(vp, allow_overwrite=True)
                v_reg.append(vp)
            else:
                v_reg.append(self.visual_patterns.get(vp.pattern_id))

        d_reg: List[DesignPattern] = []
        for dp in d_seeds:
            if not self.design_patterns.exists(dp.pattern_id):
                self.design_patterns.register(dp, allow_overwrite=True)
                d_reg.append(dp)
            else:
                d_reg.append(self.design_patterns.get(dp.pattern_id))

        return v_reg, d_reg
