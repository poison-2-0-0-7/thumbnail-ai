"""Copywriter subsystem for Module 5.5.

Authors, fills, and scores headline candidates deterministically using template
libraries, keyword extraction, and lexicon-based scoring rules. Operates with
zero network calls or LLM invocations.
"""

from __future__ import annotations

import re
from typing import Literal

from config import (
    MODULE55_CURIOSITY_LEXICON,
    MODULE55_HEADLINE_SCORE_WEIGHTS,
    MODULE55_HOOK_TYPE_THRESHOLDS,
    MODULE55_MOBILE_CHAR_HARD_LIMIT,
    MODULE55_MOBILE_CHAR_SOFT_LIMIT,
)
from models import (
    HeadlineCandidate,
    RedesignSpecification,
    ThumbnailIntelligence,
    VideoMetadata,
)

HookType = Literal[
    "curiosity", "shock", "controversy", "benefit", "authority", "fomo", "question", "how_to"
]

TEMPLATES: dict[HookType, list[tuple[str, str]]] = {
    "curiosity": [
        ("The Secret Behind {subject}", "curiosity_01"),
        ("Why {subject} Changes Everything", "curiosity_02"),
        ("The {subject} Nobody Talks About", "curiosity_03"),
    ],
    "shock": [
        ("I Tried {subject} For {duration}", "shock_01"),
        ("Don't Do {subject} Until You Watch This", "shock_02"),
        ("{subject} Is Broken", "shock_03"),
    ],
    "controversy": [
        ("The Truth About {subject}", "controversy_01"),
        ("Why Everyone Is Wrong About {subject}", "controversy_02"),
        ("Lies You Were Told About {subject}", "controversy_03"),
    ],
    "benefit": [
        ("How to Master {subject} Fast", "benefit_01"),
        ("The Ultimate {subject} Blueprint", "benefit_02"),
        ("Double Your Results With {subject}", "benefit_03"),
    ],
    "authority": [
        ("The Only {subject} Guide You Need", "authority_01"),
        ("Expert Explains {subject}", "authority_02"),
        ("Official {subject} Breakdown", "authority_03"),
    ],
    "fomo": [
        ("Before You {subject}, Watch This", "fomo_01"),
        ("Stop Doing {subject} Wrong", "fomo_02"),
        ("The {subject} Shortcut", "fomo_03"),
    ],
    "question": [
        ("Is {subject} Still Worth It?", "question_01"),
        ("What Happens When You {subject}?", "question_02"),
        ("Can You Really {subject}?", "question_03"),
    ],
    "how_to": [
        ("How to {subject} Step by Step", "how_to_01"),
        ("How I Mastered {subject}", "how_to_02"),
        ("How to Fix {subject} Today", "how_to_03"),
    ],
}

HOOK_ORDER: list[HookType] = [
    "curiosity",
    "shock",
    "controversy",
    "benefit",
    "authority",
    "fomo",
    "question",
    "how_to",
]


def extract_keywords(metadata: VideoMetadata, intelligence: ThumbnailIntelligence) -> dict[str, str]:
    """Extract primary subject and duration keywords from video metadata and intelligence."""
    title = (metadata.title or "").strip()
    clean_title = re.sub(r"[^\w\s]", "", title)
    words = clean_title.split()

    stop_words = {
        "the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "or", "is",
        "are", "was", "were", "my", "your", "this", "that", "how", "why", "what",
        "i", "you", "we", "they", "it", "with", "from", "by", "about"
    }
    keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]

    if intelligence.reasoning and intelligence.reasoning.elements_to_preserve:
        subject = intelligence.reasoning.elements_to_preserve[0]
    elif len(keywords) >= 2:
        subject = " ".join(keywords[:2]).title()
    elif keywords:
        subject = keywords[0].title()
    else:
        subject = "This Strategy"

    duration_match = re.search(r"(\d+\s*(?:days|hours|minutes|weeks|years|day|hour))", title, re.IGNORECASE)
    duration = duration_match.group(1) if duration_match else "30 Days"

    return {"subject": subject, "duration": duration}


def select_hook_types(
    intelligence: ThumbnailIntelligence,
    metadata: VideoMetadata,
) -> tuple[HookType, list[HookType]]:
    """Determine primary hook type and secondary hook types based on rules."""
    title = (metadata.title or "").lower()
    high_curiosity_thresh = MODULE55_HOOK_TYPE_THRESHOLDS.get("high_curiosity", 0.7)

    reasoning = intelligence.reasoning
    mismatch = reasoning.content_mismatch_detected if reasoning else False
    curiosity_gap = reasoning.curiosity_gap_score if reasoning else 0.5

    if mismatch:
        primary: HookType = "shock"
    elif curiosity_gap >= high_curiosity_thresh:
        primary = "curiosity"
    elif title.startswith("how to") or title.startswith("how "):
        primary = "how_to"
    elif title.endswith("?") or title.startswith("why ") or title.startswith("what "):
        primary = "question"
    else:
        primary = "benefit"

    secondaries = [h for h in HOOK_ORDER if h != primary][:2]
    return primary, secondaries


def score_curiosity(text: str) -> float:
    """Calculate curiosity score based on lexicon matches and patterns."""
    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))
    lexicon_matches = len(words.intersection(MODULE55_CURIOSITY_LEXICON))

    score = min(1.0, lexicon_matches * 0.35)
    if "?" in text or "!" in text:
        score = min(1.0, score + 0.15)
    if re.search(r"\b\d+\b", text):
        score = min(1.0, score + 0.15)
    return round(score, 4)


def score_emotional_impact(text: str, intelligence: ThumbnailIntelligence) -> float:
    """Calculate emotional impact score aligned with facial emotion and reasoning."""
    base_score = 0.5
    if intelligence.reasoning and intelligence.reasoning.emotional_impact:
        impact = intelligence.reasoning.emotional_impact.lower()
        if impact in {"high", "strong", "intense"}:
            base_score = 0.8
        elif impact in {"medium", "moderate"}:
            base_score = 0.6
        elif impact in {"low", "mild"}:
            base_score = 0.4

    face_emotions = [f.emotion.lower() for f in intelligence.faces.faces if f.emotion]
    text_lower = text.lower()

    if "smile" in face_emotions or "happy" in face_emotions:
        if any(w in text_lower for w in ["broken", "wrong", "lie", "bad", "worst"]):
            base_score -= 0.2  # contradiction penalty
        elif any(w in text_lower for w in ["best", "master", "ultimate", "shortcut"]):
            base_score += 0.1
    elif "shock" in face_emotions or "surprised" in face_emotions:
        if any(w in text_lower for w in ["truth", "secret", "tried", "broken", "watch"]):
            base_score += 0.15

    return round(max(0.0, min(1.0, base_score)), 4)


def score_readability(text: str) -> float:
    """Calculate readability score based on word length and word count heuristics."""
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return 0.0

    avg_word_length = sum(len(w) for w in words) / word_count

    if 3 <= word_count <= 8:
        count_score = 1.0
    elif word_count < 3:
        count_score = 0.7
    else:
        count_score = max(0.2, 1.0 - (word_count - 8) * 0.1)

    if avg_word_length <= 6.0:
        length_score = 1.0
    else:
        length_score = max(0.2, 1.0 - (avg_word_length - 6.0) * 0.15)

    return round((count_score * 0.6 + length_score * 0.4), 4)


def score_mobile_readability(char_count: int) -> float:
    """Calculate mobile readability score penalizing text exceeding soft/hard limits."""
    soft = MODULE55_MOBILE_CHAR_SOFT_LIMIT
    hard = MODULE55_MOBILE_CHAR_HARD_LIMIT

    if char_count <= soft:
        return 1.0
    if char_count >= hard:
        return 0.0
    return round(1.0 - (char_count - soft) / (hard - soft), 4)


def score_brand_consistency(text: str, spec: RedesignSpecification) -> float:
    """Check candidate against elements to preserve and weaknesses."""
    score = 1.0
    text_lower = text.lower()

    for item in spec.elements_to_preserve:
        if item.startswith("no_") and item[3:].lower() in text_lower:
            score -= 0.3

    return round(max(0.0, score), 4)


def score_ctr_potential(
    source_ctr: float,
    curiosity: float,
    emotional: float,
    mobile: float,
) -> float:
    """Calculate candidate CTR potential relative to source baseline."""
    delta = (curiosity * 0.12) + (emotional * 0.08) + ((mobile - 0.5) * 0.1)
    ctr = source_ctr + delta
    return round(max(0.0, min(1.0, ctr)), 4)


def author_headline_candidates(
    intelligence: ThumbnailIntelligence,
    spec: RedesignSpecification,
    metadata: VideoMetadata,
) -> tuple[str, float, HookType, str, list[HeadlineCandidate]]:
    """Author and score headline candidates, returning selected headline and variants."""
    keywords = extract_keywords(metadata, intelligence)
    primary_hook, secondary_hooks = select_hook_types(intelligence, metadata)

    hooks_to_generate = [primary_hook] + secondary_hooks
    raw_candidates: list[tuple[str, str]] = []

    for hook in hooks_to_generate:
        for tmpl, tmpl_id in TEMPLATES.get(hook, []):
            try:
                filled = tmpl.format(**keywords)
                raw_candidates.append((filled, tmpl_id))
            except KeyError:
                continue

    source_ctr = spec.source_ctr_potential_score
    candidates: list[HeadlineCandidate] = []

    for text, tmpl_id in raw_candidates:
        c_len = len(text)
        curiosity = score_curiosity(text)
        emotional = score_emotional_impact(text, intelligence)
        readability = score_readability(text)
        mobile = score_mobile_readability(c_len)
        brand = score_brand_consistency(text, spec)
        ctr = score_ctr_potential(source_ctr, curiosity, emotional, mobile)

        weights = MODULE55_HEADLINE_SCORE_WEIGHTS
        composite = (
            curiosity * weights.get("curiosity", 0.25)
            + emotional * weights.get("emotional_impact", 0.20)
            + readability * weights.get("readability", 0.15)
            + ctr * weights.get("ctr_potential", 0.20)
            + mobile * weights.get("mobile_readability", 0.10)
            + brand * weights.get("brand_consistency", 0.10)
        )
        composite = round(max(0.0, min(1.0, composite)), 4)

        candidate = HeadlineCandidate(
            text=text,
            template_id=tmpl_id,
            curiosity_score=curiosity,
            emotional_impact_score=emotional,
            readability_score=readability,
            ctr_potential_score=ctr,
            character_count=c_len,
            mobile_readability_score=mobile,
            brand_consistency_score=brand,
            composite_score=composite,
        )
        candidates.append(candidate)

    if not candidates:
        fallback_text = keywords["subject"]
        c_len = len(fallback_text)
        candidate = HeadlineCandidate(
            text=fallback_text,
            template_id="fallback_01",
            curiosity_score=0.5,
            emotional_impact_score=0.5,
            readability_score=1.0,
            ctr_potential_score=source_ctr,
            character_count=c_len,
            mobile_readability_score=score_mobile_readability(c_len),
            brand_consistency_score=1.0,
            composite_score=0.5,
        )
        candidates.append(candidate)

    candidates.sort(key=lambda c: (-c.composite_score, c.character_count, c.text))

    selected = candidates[0]
    emotion_label = (
        intelligence.reasoning.emotional_impact
        if intelligence.reasoning and intelligence.reasoning.emotional_impact
        else "neutral"
    )

    return (
        selected.text,
        selected.composite_score,
        primary_hook,
        emotion_label,
        candidates,
    )
