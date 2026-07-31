"""
test_validators.py
===================

Unit tests for PVQEF module validators.
"""

import json
from pathlib import Path

import pytest

from evaluation.module_validators import (
    AssetComposerValidator,
    CSVReaderValidator,
    DesignBlueprintValidator,
    Module7Validator,
    PromptCompilerValidator,
    RedesignSpecValidator,
    ThumbnailDownloaderValidator,
    ThumbnailIntelligenceValidator,
    YouTubeMetadataValidator,
)


def test_design_blueprint_validator(tmp_path):
    validator = DesignBlueprintValidator()
    art_path = tmp_path / "blueprint.json"
    art_path.write_text(json.dumps({
        "video_id": "vid123",
        "headline": "Secret Coding Hacks",
        "headline_variants": [],
        "headline_score": 0.85,
        "hook_type": "curiosity",
        "emotion": "high",
        "face_strategy": "smile",
        "object_strategy": [],
        "background_strategy": "keep",
        "text_position": {"include_text": True},
        "camera_distance": "medium",
        "lighting": "balanced",
        "color_palette": [],
        "visual_priority": ["headline"],
        "branding_constraints": [],
        "conflicts_resolved": 0,
        "status": "success",
        "generated_at": "2026-08-01T00:00:00Z"
    }))

    res = validator.validate("vid123", art_path)
    assert res.schema_valid is True
    assert res.status == "success"


def test_csv_reader_validator_success(tmp_path):
    validator = CSVReaderValidator()
    art_path = tmp_path / "creators.csv"
    art_path.write_text("email,video_url\ntest@example.com,https://www.youtube.com/watch?v=dQw4w9WgXcQ\n")

    res = validator.validate("dQw4w9WgXcQ", art_path)
    assert res.schema_valid is True
    assert res.status == "success"
    assert res.invariants_failed == []


def test_csv_reader_validator_missing(tmp_path):
    validator = CSVReaderValidator()
    res = validator.validate("vid123", tmp_path / "missing.csv")
    assert res.schema_valid is False
    assert res.status == "error"


def test_youtube_metadata_validator_success(tmp_path):
    validator = YouTubeMetadataValidator()
    art_path = tmp_path / "vid123.json"
    art_path.write_text(json.dumps({
        "video_id": "vid123",
        "title": "Test Title",
        "uploader": "Test Uploader",
        "uploader_id": "@testuploader",
        "channel_id": "UC12345",
        "categories": ["Tech"],
        "view_count": 1000,
        "status": "success"
    }))

    res = validator.validate("vid123", art_path)
    assert res.schema_valid is True
    assert res.status == "success"


def test_thumbnail_downloader_validator(tmp_path):
    validator = ThumbnailDownloaderValidator()
    img_file = tmp_path / "vid123.jpg"
    img_file.write_bytes(b"fake_image_bytes")

    art_path = tmp_path / "thumb.json"
    art_path.write_text(json.dumps({
        "metadata": {
            "video_id": "vid123",
            "title": "Test Title",
            "uploader": "Test Uploader",
            "uploader_id": "@testuploader",
            "channel_id": "UC12345",
            "status": "success"
        },
        "thumbnail_path": str(img_file)
    }))

    res = validator.validate("vid123", art_path)
    assert res.schema_valid is True
    assert res.status == "success"


def test_thumbnail_intelligence_validator(tmp_path):
    validator = ThumbnailIntelligenceValidator()
    art_path = tmp_path / "intel.json"
    art_path.write_text(json.dumps({
        "video_id": "vid123",
        "thumbnail_path": "data/thumbnails/vid123.jpg",
        "ocr": {
            "visible_text": "Tech",
            "text_regions": [],
            "word_count": 1,
            "text_coverage_ratio": 0.1,
            "average_confidence": 0.9,
            "engine_available": True,
            "duration_seconds": 0.1
        },
        "faces": {
            "face_count": 1,
            "faces": [],
            "has_face": True,
            "engine_available": True,
            "duration_seconds": 0.1
        },
        "objects": [],
        "colors": {
            "dominant_colors": ["#ffffff"],
            "brightness": 0.5,
            "contrast": 0.5,
            "saturation": 0.5,
            "warm_or_cool": "neutral",
            "harmony_score": 0.8,
            "duration_seconds": 0.1
        },
        "composition": {
            "rule_of_thirds_score": 0.8,
            "subject_placement": "center",
            "negative_space_ratio": 0.4,
            "clutter_score": 0.2,
            "visual_hierarchy_score": 0.9,
            "text_overlaps_subject": False,
            "balance_score": 0.85,
            "symmetry_score": 0.8,
            "duration_seconds": 0.1
        },
        "status": "success",
        "analyzed_at": "2026-07-30T00:00:00Z"
    }))

    res = validator.validate("vid123", art_path)
    assert res.schema_valid is True
    assert res.status == "success"


def test_redesign_spec_validator(tmp_path):
    validator = RedesignSpecValidator()
    art_path = tmp_path / "spec.json"
    art_path.write_text(json.dumps({
        "video_id": "vid123",
        "source_thumbnail_path": "data/thumbnails/vid123.jpg",
        "color_direction": {
            "target_brightness": 0.6,
            "target_contrast": 0.7,
            "target_saturation": 0.5,
            "warm_or_cool": "cool",
            "rationale": "High contrast"
        },
        "subject_treatment": {
            "has_subject": True,
            "target_position_label": "center",
            "crop_tighter": False,
            "rationale": "Center face"
        },
        "text_overlay": {
            "include_text": False,
            "rationale": "No text"
        },
        "layout_direction": {
            "target_negative_space_ratio": 0.3,
            "target_clutter_score": 0.2,
            "rationale": "Clean"
        },
        "source_ctr_potential_score": 0.7,
        "source_curiosity_gap_score": 0.6,
        "source_content_mismatch_detected": False,
        "generated_at": "2026-07-30T00:00:00Z"
    }))

    res = validator.validate("vid123", art_path)
    assert res.schema_valid is True
    assert res.status == "success"


def test_prompt_compiler_validator(tmp_path):
    validator = PromptCompilerValidator()
    art_path = tmp_path / "prompt.json"
    art_path.write_text(json.dumps({
        "video_id": "vid123",
        "positive_prompt": "A vibrant tech video thumbnail",
        "negative_prompt": "blurry",
        "subject_instructions": "Center focal person",
        "background_instructions": "Clean dark background",
        "typography_instructions": "Bold header",
        "composition_instructions": "Rule of thirds",
        "lighting_instructions": "Studio lighting",
        "color_instructions": "Vibrant blue and orange",
        "generation_parameters": {
            "width": 1280,
            "height": 720,
            "aspect_ratio": "16:9",
            "guidance_scale": 7.5,
            "inference_steps": 30,
            "sampler": "deterministic",
            "seed": 123456
        },
        "quality_parameters": {
            "quality_tags": ["sharp"],
            "min_resolution_px": 1280,
            "upscale_requested": False
        },
        "model_settings": {
            "model_name": "sdxl",
            "style_preset": "photographic",
            "negative_prompt_weight": 1.0
        },
        "generated_at": "2026-07-30T00:00:00Z"
    }))

    res = validator.validate("vid123", art_path)
    assert res.schema_valid is True
    assert res.status == "success"


def test_asset_composer_validator(tmp_path):
    validator = AssetComposerValidator()
    art_path = tmp_path / "workspace.json"
    art_path.write_text(json.dumps({
        "video_id": "vid123",
        "canvas": {"width": 1280, "height": 720, "aspect_ratio": "16:9"},
        "layers": [],
        "groups": [],
        "text_placement": {"include_text": False},
        "lighting": {"target_brightness": 0.5, "target_contrast": 0.5, "target_saturation": 0.5, "warm_or_cool": "neutral"},
        "constraints": {"safe_margin_px": 24},
        "statistics": {"total_layers": 0},
        "metadata": {
            "video_id": "vid123",
            "created_at": "2026-07-30T00:00:00Z",
            "vre_source_hash": "a" * 64,
            "redesign_spec_hash": "b" * 64,
            "prompt_package_hash": "c" * 64,
            "engine_version": "1.0.0"
        }
    }))

    res = validator.validate("vid123", art_path)
    assert res.schema_valid is True
    assert res.status == "success"


def test_module7_validator(tmp_path):
    validator = Module7Validator()
    img_file = tmp_path / "gen.png"
    img_file.write_bytes(b"fake_generated_image_bytes")

    art_path = tmp_path / "result.json"
    art_path.write_text(json.dumps({
        "video_id": "vid123",
        "status": "success",
        "generated_asset": {
            "path": str(img_file),
            "width": 1280,
            "height": 720,
            "sha256": "d" * 64
        },
        "workflow_version": "v1",
        "prompt_package_hash": "e" * 64,
        "generated_at": "2026-07-30T00:00:00Z"
    }))

    res = validator.validate("vid123", art_path)
    assert res.schema_valid is True
    assert res.status == "success"
