"""Executive advisor charter in system prompt."""

from __future__ import annotations

from rag.executive_advisor_charter import executive_advisor_charter_block
from rag.query_service import CONSULTANT_SYSTEM_PROMPT
from rag.response_safety import sanitize_user_facing_answer


def test_charter_in_consultant_system_prompt():
    assert "RELEVANCE-FIRST" in CONSULTANT_SYSTEM_PROMPT
    assert "EXECUTIVE ADVISOR IDENTITY" in CONSULTANT_SYSTEM_PROMPT
    assert "aviation database assistant" in CONSULTANT_SYSTEM_PROMPT.lower()
    assert "strategic framing" in CONSULTANT_SYSTEM_PROMPT.lower()
    assert "founders, executives" in CONSULTANT_SYSTEM_PROMPT.lower()
    assert "How many seats" in CONSULTANT_SYSTEM_PROMPT
    assert "markdown asterisks" in CONSULTANT_SYSTEM_PROMPT.lower()


def test_charter_block_standalone():
    block = executive_advisor_charter_block()
    assert "RELEVANCE-FIRST" in block
    assert "most relevant" in block.lower()
    assert "spec wall" in block.lower() or "spec dumps" in block.lower()
    assert "cannot show images" in block.lower()


def test_safety_strips_visual_refusal():
    raw = "I can't show images for that aircraft, but the G650 is nice."
    out = sanitize_user_facing_answer(raw, strong_aircraft_gallery=True)
    assert "can't show images" not in out.lower()


def test_safety_strips_markdown_bold():
    raw = "The **Challenger 350** is a **strong fit** for that mission."
    out = sanitize_user_facing_answer(raw)
    assert "**" not in out
    assert "Challenger 350" in out


def test_safety_strips_markdown_hash_headers():
    raw = "## Gulfstream G650\n\n# Seats\nTypically **14–19** seats (cite snippet #2)."
    out = sanitize_user_facing_answer(raw)
    assert "#" not in out
    assert "Gulfstream G650" in out
    assert "Seats" in out
    assert "14" in out
