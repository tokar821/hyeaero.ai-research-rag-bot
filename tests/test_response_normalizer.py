"""Phase 12 — response normalization layer tests."""

from __future__ import annotations

import re

from services.response.response_normalizer import (
    apply_consultant_response_normalization,
    normalize_consultant_response,
)

_KERNEL = re.compile(r"\boperational\s+synthesis\b", re.I)


def test_comparison_normalization_schema():
    raw_answer = (
        "Verified catalog comparison:\n"
        "- Gulfstream G650: large class; practical range 7000 nm; seats 16; operating cost band high.\n"
        "- Falcon 8X: large class; practical range 6450 nm; seats 14; operating cost band high.\n"
        "On verified range, Gulfstream G650 leads Falcon 8X (7000 nm vs 6450 nm catalog practical)."
    )
    response = {
        "answer": raw_answer,
        "data_used": {
            "authority_dispatch_kind": "comparison",
            "comparison_v2": {"status": "OK", "models": ["Gulfstream G650", "Falcon 8X"]},
        },
    }
    normalized = normalize_consultant_response(response, context={"query": "G650 vs Falcon 8X"})
    assert normalized.intent_type == "comparison"
    assert "G650" in " ".join(normalized.aircraft)
    assert "Falcon 8X" in " ".join(normalized.aircraft)
    assert normalized.structured_sections["overview"]
    assert normalized.structured_sections["analysis"]
    assert not _KERNEL.search(normalized.answer_text)
    assert normalized.data_sources["market_used"] is False


def test_alternative_normalization_tier_peers():
    response = {
        "answer": (
            "Credible tier-peer alternatives to the Citation Longitude include "
            "Praetor 600, Challenger 350, and Gulfstream G280. "
            "These are verified catalog tier peers only."
        ),
        "data_used": {
            "authority_dispatch_kind": "alternative",
            "alternative_execution": {
                "target": "Citation Longitude",
                "candidates": ["Praetor 600", "Challenger 350", "Gulfstream G280"],
            },
        },
    }
    normalized = normalize_consultant_response(
        response, context={"query": "Alternatives to Longitude"}
    )
    assert normalized.intent_type == "alternative"
    assert normalized.primary_recommendation.get("model") == "Citation Longitude"
    assert len(normalized.alternatives) >= 2
    assert normalized.structured_sections["overview"]
    assert normalized.verdict == "CONDITIONAL FIT"


def test_buy_decision_normalization_verdict():
    response = {
        "answer": (
            "Aircraft: Citation Latitude\n"
            "Year: 2016\n"
            "Ask: $10.0M\n\n"
            "Market Reality:\n"
            "- Limited synced comp data for this model slice.\n\n"
            "Red Flags:\n"
            "- No clear engine hourly program on file.\n\n"
            "Verdict:\n"
            "HIGH RISK"
        ),
        "data_used": {
            "authority_dispatch_kind": "buy_decision",
            "buy_decision_dispatch": {
                "model": "Citation Latitude",
                "year": 2016,
                "ask_usd": 10_000_000.0,
            },
            "deal_killer": {
                "verdict": "HIGH RISK",
                "confidence": 0.72,
                "red_flags": ["No clear engine hourly program on file."],
                "scores": {"price_score": 0.55, "mission_fit_score": 0.8},
            },
        },
    }
    normalized = normalize_consultant_response(
        response, context={"query": "2016 Latitude $10M good deal?"}
    )
    assert normalized.intent_type == "buy_decision"
    assert normalized.verdict == "RISKY"
    assert normalized.confidence == 0.72
    assert normalized.financial_summary.get("ask_usd") == 10_000_000.0
    assert normalized.structured_sections["risks"]
    assert "Verdict: RISKY" in normalized.answer_text


def test_mission_normalization_maps_compromises():
    response = {
        "answer": (
            "Mission Interpretation\n"
            "- Primary stage: TEB-LAX with 6 passengers.\n\n"
            "Final Verdict\n"
            "- VIABLE WITH COMPROMISES: winter westbound reserves required.\n"
        ),
        "data_used": {
            "query_recommendation_requires_pipeline": True,
            "recommendation_pipeline": {"ranked_models": ["Challenger 350", "Praetor 600"]},
        },
    }
    normalized = normalize_consultant_response(
        response,
        context={"query": "Need a jet TEB to LAX, 6 pax, what should I buy?"},
    )
    assert normalized.intent_type == "mission"
    assert normalized.verdict in ("VIABLE WITH COMPROMISES", "CONDITIONAL FIT")
    assert normalized.primary_recommendation.get("model") == "Challenger 350"


def test_kernel_leak_stripped_from_comparison():
    response = {
        "answer": (
            "OPERATIONAL SYNTHESIS (AUTHORITATIVE)\n"
            "Verified catalog comparison:\n"
            "- Gulfstream G650: large class.\n"
            "- Falcon 8X: large class.\n"
        ),
        "data_used": {"authority_dispatch_kind": "comparison"},
    }
    normalized = normalize_consultant_response(response, context={"query": "G650 vs Falcon 8X"})
    assert not _KERNEL.search(normalized.answer_text)


def test_apply_normalization_patches_payload():
    payload = {
        "answer": "Credible tier-peer alternatives to the Citation Longitude include Praetor 600.",
        "data_used": {
            "authority_dispatch_kind": "alternative",
            "alternative_execution": {
                "target": "Citation Longitude",
                "candidates": ["Praetor 600"],
            },
        },
    }
    out = apply_consultant_response_normalization(
        payload, context={"query": "Alternatives to Longitude"}
    )
    assert out["data_used"].get("response_normalization_applied") == 1
    assert isinstance(out["data_used"].get("normalized_response"), dict)
    assert out["data_used"]["normalized_response"]["intent_type"] == "alternative"
    assert out["data_used"].get("broker_conversation_layer_applied") == 1
    assert "Overview" not in out["answer"]
    assert "Longitude" in out["answer"]
