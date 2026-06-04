"""Deterministic consultant charter tests."""

from __future__ import annotations

from rag.deterministic_consultant_charter import (
    adaptive_presentation_suffix,
    deterministic_consultant_charter_block,
    deterministic_execution_charter_block,
)
from rag.query_service import CONSULTANT_SYSTEM_PROMPT


def test_charter_in_system_prompt():
    assert "DETERMINISTIC EXECUTION CHARTER" in CONSULTANT_SYSTEM_PROMPT
    assert "ADAPTIVE PRESENTATION CHARTER" in CONSULTANT_SYSTEM_PROMPT
    assert "fail-closed" in CONSULTANT_SYSTEM_PROMPT.lower() or "safety fallback" in CONSULTANT_SYSTEM_PROMPT.lower()


def test_execution_charter_hard_intents():
    block = deterministic_execution_charter_block()
    assert "comparison" in block
    assert "AKAL" in block
    assert "ICRL" in block
    assert "NEVER use LLM fallback" in block


def test_adaptive_presentation_by_intent():
    assert "matrix" in adaptive_presentation_suffix("comparison").lower()
    assert "verdict" in adaptive_presentation_suffix("buy_decision").lower()
    assert "ICRL" in adaptive_presentation_suffix(None, icrl_handled=True)


def test_forbidden_phrases_in_charter():
    block = deterministic_consultant_charter_block()
    assert "great aircraft" in block
    assert "excellent choice" in block
