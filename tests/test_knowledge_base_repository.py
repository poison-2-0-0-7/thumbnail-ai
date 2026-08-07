"""
Unit tests for KnowledgeBaseRepository.
Tests multi-namespace repository initialization, typed registries access,
high-level domain queries, and curated seed data bootstrapping.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from thumbnail_intelligence.knowledge_base.config import KnowledgeBaseConfig
from thumbnail_intelligence.knowledge_base.models import (
    Archetype,
    CreatorProfile,
    KnowledgeEntry,
    KnowledgeEntryType,
)
from thumbnail_intelligence.knowledge_base.repository import KnowledgeBaseRepository


def test_repository_initialization_and_namespaces(tmp_path: Path) -> None:
    config = KnowledgeBaseConfig(base_dir=tmp_path)
    repo = KnowledgeBaseRepository(config=config)

    assert config.creator_profiles_dir.exists()
    assert config.archetypes_dir.exists()
    assert config.competitor_profiles_dir.exists()
    assert config.design_patterns_dir.exists()

    # Verify typed registries are accessible
    assert repo.entries is not None
    assert repo.creator_profiles is not None
    assert repo.channel_profiles is not None
    assert repo.competitor_profiles is not None
    assert repo.archetypes is not None
    assert repo.brand_constraints is not None
    assert repo.identity_constraints is not None
    assert repo.visual_patterns is not None
    assert repo.design_patterns is not None
    assert repo.thumbnail_patterns is not None


def test_repository_seed_archetypes_and_patterns(tmp_path: Path) -> None:
    config = KnowledgeBaseConfig(base_dir=tmp_path)
    repo = KnowledgeBaseRepository(config=config)

    # 1. Seed archetypes
    seeded_archetypes = repo.seed_default_archetypes()
    assert len(seeded_archetypes) >= 5
    assert any(a.archetype_id == "big_face_reaction" for a in seeded_archetypes)
    assert any(a.archetype_id == "curiosity_gap" for a in seeded_archetypes)
    assert any(a.archetype_id == "before_after_split" for a in seeded_archetypes)

    # Lookup through repository
    arch = repo.get_archetype("big_face_reaction")
    assert arch is not None
    assert arch.name == "Big Face Reaction"

    # 2. Seed visual and design patterns
    v_patterns, d_patterns = repo.seed_default_patterns()
    assert len(v_patterns) >= 2
    assert len(d_patterns) >= 2

    # Query patterns by niche
    niche_patterns = repo.find_patterns_by_niche("entertainment")
    assert len(niche_patterns) >= 1


def test_repository_domain_queries(tmp_path: Path) -> None:
    config = KnowledgeBaseConfig(base_dir=tmp_path)
    repo = KnowledgeBaseRepository(config=config)

    # Register creator profile
    creator = CreatorProfile(
        creator_id="creator_test_01",
        display_name="Test Creator",
        channel_ids=["channel_01", "channel_02"],
        primary_niche="tech",
    )
    repo.creator_profiles.register(creator)

    # Test get_creator_profile
    retrieved = repo.get_creator_profile("creator_test_01")
    assert retrieved is not None
    assert retrieved.display_name == "Test Creator"

    # Register knowledge entry
    entry = KnowledgeEntry(
        entry_id="entry_hist_01",
        entry_type=KnowledgeEntryType.HISTORICAL_THUMBNAIL,
        embedding=[0.0] * 512,
        niche="tech",
        source_video_id="video_999",
    )
    repo.entries.register(entry)

    # Test find_entries_by_type
    entries = repo.find_entries_by_type(KnowledgeEntryType.HISTORICAL_THUMBNAIL, niche="tech")
    assert len(entries) == 1
    assert entries[0].source_video_id == "video_999"
