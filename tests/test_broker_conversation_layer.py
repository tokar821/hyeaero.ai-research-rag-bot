"""Phase 39 — broker conversation layer tests."""

from __future__ import annotations

import re

import pytest

from services.conversation.broker_conversation_layer import apply_broker_conversation_layer
from services.conversation.broker_fallbacks import (
    INTERNAL_MARKERS,
    apply_broker_fallbacks,
    broker_fallback_for_query,
    contains_internal_language,
    translate_internal_messages,
)
from services.conversation.output_cleaner import clean_broker_output
from services.conversation.broker_style import apply_broker_style, render_conversational_sections
from services.response.response_normalizer import apply_consultant_response_normalization

_BANNED_USER_TERMS = re.compile(
    r"(?i)\b(?:"
    r"INSUFFICIENT_DATA|deterministic execution|verified catalog|verified aircraft|"
    r"mission kernel|catalog authority|catalog contrast|authority dispatch|"
    r"confidence threshold|comparison safety|temporal confidence low|"
    r"CLARIFICATION_REQUIRED|INFEASIBLE_BUDGET_CONSTRAINT|MARKET_CONTEXT_AVAILABLE"
    r")\b"
)


def _assert_no_internal_terms(text: str) -> None:
    assert not _BANNED_USER_TERMS.search(text or ""), f"internal term leaked: {text!r}"


def test_translate_insufficient_data_fallback():
    raw = "INSUFFICIENT_DATA: No verified aircraft available."
    out = translate_internal_messages(raw)
    _assert_no_internal_terms(out)
    assert "reliable recommendation" in out.lower()


def test_translate_comparison_safety_fallback():
    raw = (
        "Insufficient verified data for deterministic execution.\n\n"
        "Verified catalog comparison requires two recognized aircraft models."
    )
    out = apply_broker_fallbacks(raw)
    _assert_no_internal_terms(out)
    assert "comparing" in out.lower()


def test_translate_deterministic_execution_unavailable():
    raw = "Deterministic execution unavailable."
    out = translate_internal_messages(raw)
    _assert_no_internal_terms(out)
    assert "confidently identify" in out.lower()


def test_clarification_cheap_gulfstream():
    q = "cheap gulfstream"
    out = apply_broker_conversation_layer(
        "Insufficient verified data for deterministic execution.",
        query=q,
    )
    _assert_no_internal_terms(out)
    assert "cheap Gulfstream" in out or "G450" in out or "G650" in out


def test_clarification_g700_vs_cheaper():
    q = "G700 vs something cheaper"
    out = apply_broker_conversation_layer(
        "CLARIFICATION_REQUIRED: ambiguous query.",
        query=q,
    )
    _assert_no_internal_terms(out)
    assert "cheaper" in out.lower()
    assert "range" in out.lower() or "acquisition" in out.lower()


def test_clarification_forgot_second_model():
    q = "compare jets but I forgot the second model"
    out = apply_broker_conversation_layer("", query=q)
    _assert_no_internal_terms(out)
    assert "comparing" in out.lower()


def test_buy_decision_internal_stripped():
    raw = (
        "Aircraft: Gulfstream G650\n\n"
        "Market Reality:\n"
        "- Ask is well below typical market band.\n\n"
        "Deal Assessment:\n"
        "- GOOD DEAL\n\n"
        "Verdict:\nGOOD DEAL"
    )
    out = apply_broker_conversation_layer(raw, query="G650 for $5M good deal?", intent_type="buy_decision")
    _assert_no_internal_terms(out)
    assert "G650" in out
    assert "GOOD DEAL" in out


def test_comparison_prose_no_catalog_banner():
    raw = (
        "Verified catalog comparison:\n"
        "- Gulfstream G700: ultra-long class; practical range 7700 nm; seats 19; operating cost band high.\n"
        "- Gulfstream G650: large class; practical range 7000 nm; seats 16; operating cost band high.\n"
        "VERDICT:\nChoose Gulfstream G700 if range is the deciding factor."
    )
    out = apply_broker_conversation_layer(raw, query="compare g700 vs g650", intent_type="comparison")
    _assert_no_internal_terms(out)
    assert "verified catalog" not in out.lower()
    assert "G700" in out and "G650" in out


def test_output_cleaner_removes_markdown_artifacts():
    raw = "Line one\n*\n## **\n* bullet item\n\n\nEmpty section\n"
    out = clean_broker_output(raw)
    assert "## **" not in out
    assert re.search(r"^\*\s*$", out, re.M) is None
    assert "• bullet item" in out or "- bullet item" in out


def test_output_cleaner_strips_best_match_scaffolding():
    raw = (
        "Club seating and a divan.\n\n"
        "Best Match: Cessna Citation Excel Reason: The images depict a cabin layout.\n"
        "This aircraft is for sale."
    )
    out = clean_broker_output(raw)
    assert "best match" not in out.lower()
    assert "club seating" in out.lower()


def test_output_cleaner_strips_retrieval_provenance():
    raw = "Separately, per Hye Aero listing records the ask is high.\nPer aircraft registry the tail is active."
    out = clean_broker_output(raw)
    assert "hye aero listing records" not in out.lower()
    assert "per aircraft registry" not in out.lower()


def test_output_cleaner_strips_bold_markdown():
    raw = "1. **Gulfstream G280**: Known for performance.\n2. **Praetor 600**: Efficient."
    out = clean_broker_output(raw)
    assert "**" not in out
    assert "Gulfstream G280" in out


def test_remove_empty_sections():
    raw = "Overview\n\nAnalysis\n\nRecommendation\nSome text here."
    out = clean_broker_output(raw)
    assert "Overview" not in out
    assert "Analysis" not in out
    assert "Some text here" in out


def test_conversational_sections_not_template():
    text = render_conversational_sections(
        overview="G650 vs Falcon 8X on range and cabin.",
        analysis="- G650 leads on range.",
        recommendation="Lean G650 if range is priority.",
        risks=["Confirm maintenance on specific tails."],
        verdict="CONDITIONAL FIT",
        intent_type="comparison",
    )
    assert "Overview" not in text
    assert "Analysis" not in text
    assert "G650" in text


def test_apply_normalization_strips_internal_terms():
    response = {
        "answer": (
            "Insufficient verified data for deterministic execution.\n\n"
            "Verified catalog comparison requires two recognized aircraft models."
        ),
        "query": "compare jets but I forgot the second model",
        "data_used": {"authority_dispatch_kind": "comparison"},
    }
    out = apply_consultant_response_normalization(
        response,
        context={"query": response["query"]},
    )
    _assert_no_internal_terms(out["answer"])
    assert out["data_used"].get("broker_conversation_layer_applied") == 1


def test_broker_style_removes_mission_template_headers():
    raw = (
        "Mission Fit:\nRoute: TEB-LAX\n\n"
        "Aircraft Options:\nCitation Latitude — strong range for the leg."
    )
    out = apply_broker_style(raw, intent_type="mission")
    assert "Mission Fit:" not in out
    assert "Aircraft Options:" not in out
    assert "Latitude" in out


@pytest.mark.parametrize("marker", INTERNAL_MARKERS[:8])
def test_internal_markers_classified(marker):
    assert contains_internal_language(f"prefix {marker} suffix")


def test_broker_fallback_for_budget_mission():
    q = "what can I buy for $20M?"
    msg = broker_fallback_for_query(q, "")
    assert msg is not None
    _assert_no_internal_terms(msg)
    assert "20" in msg or "budget" in msg.lower()


_ACCEPTANCE_QUERIES = [
    "cheap gulfstream",
    "G700 vs something cheaper",
    "compare jets but I forgot the second model",
    "G650 for $5M good deal?",
    "longitude jet",
    "what can I buy for $20M?",
]

_INTERNAL_ANSWERS = [
    "Insufficient verified data for deterministic execution.",
    "CLARIFICATION_REQUIRED: ambiguous query.",
    "Insufficient verified data for deterministic execution.\n\nVerified catalog comparison requires two recognized aircraft models.",
    "Aircraft: Gulfstream G650\n\nMarket Reality:\n- Unusually low ask.\n\nVerdict:\nGOOD DEAL",
    "Insufficient verified aircraft data to produce a comparison.",
    "",
]


@pytest.mark.parametrize("query,raw", zip(_ACCEPTANCE_QUERIES, _INTERNAL_ANSWERS))
def test_acceptance_criteria_queries(query, raw):
    out = apply_broker_conversation_layer(raw, query=query)
    _assert_no_internal_terms(out)
    assert out.strip()
    assert "## **" not in out
    assert re.search(r"^\*\s*$", out, re.M) is None
