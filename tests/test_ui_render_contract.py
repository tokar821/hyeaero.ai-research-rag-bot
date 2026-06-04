"""Phase 13 — UI render contract layer tests."""

from __future__ import annotations

from services.response.response_normalizer import (
    apply_consultant_response_normalization,
    normalize_consultant_response,
)
from services.response.ui_render_contract import build_ui_render_contract


def _normalized(query: str, response: dict):
    return normalize_consultant_response(response, context={"query": query})


def test_comparison_ui_contract_side_by_side():
    response = {
        "answer": (
            "Verified catalog comparison:\n"
            "- Gulfstream G650: large class; practical range 7000 nm.\n"
            "- Falcon 8X: large class; practical range 6450 nm.\n"
            "On verified range, Gulfstream G650 leads Falcon 8X."
        ),
        "data_used": {
            "authority_dispatch_kind": "comparison",
            "comparison_v2": {"status": "OK", "models": ["Gulfstream G650", "Falcon 8X"]},
        },
    }
    normalized = _normalized("G650 vs Falcon 8X", response)
    contract = build_ui_render_contract(normalized, context={"query": "G650 vs Falcon 8X"})

    assert contract.ui_intent == "comparison"
    assert contract.layout_type == "side_by_side"
    assert len(contract.primary_cards) == 2
    assert contract.primary_cards[0]["role"] == "compare_column"
    assert contract.render_hints["comparison_mode"] == "strict_side_by_side"
    assert contract.ui_flags["show_price_comparison"] is False
    assert any(s["type"] == "analysis" for s in contract.sections)


def test_alternative_ui_contract_tier_clustered():
    response = {
        "answer": (
            "Credible tier-peer alternatives to the Citation Longitude include "
            "Praetor 600, Challenger 350, and Gulfstream G280."
        ),
        "data_used": {
            "authority_dispatch_kind": "alternative",
            "alternative_execution": {
                "target": "Citation Longitude",
                "candidates": ["Praetor 600", "Challenger 350", "Gulfstream G280"],
            },
        },
    }
    normalized = _normalized("Alternatives to Longitude", response)
    contract = build_ui_render_contract(normalized, context={"query": "Alternatives to Longitude"})

    assert contract.ui_intent == "alternative"
    assert contract.layout_type == "ranked_list"
    assert contract.primary_cards[0]["title"] == "Citation Longitude"
    assert len(contract.secondary_cards) >= 2
    assert contract.render_hints["alternative_mode"] == "tier_clustered"
    assert contract.ui_flags["show_verdict_badge"] is True


def test_buy_decision_ui_contract_deal_card():
    response = {
        "answer": (
            "Aircraft: Citation Latitude\nYear: 2016\nAsk: $10.0M\n\n"
            "Market Reality:\n- Limited synced comp data.\n\n"
            "Red Flags:\n- No engine program on file.\n\nVerdict:\nHIGH RISK"
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
                "confidence": 0.7,
                "red_flags": ["No engine program on file."],
                "scores": {"price_score": 0.4, "mission_fit_score": 0.8},
                "price_position": "above_market",
                "inputs_echo": {"market_avg_usd": 8_500_000.0},
            },
        },
    }
    normalized = _normalized("2016 Latitude $10M good deal?", response)
    contract = build_ui_render_contract(normalized, context={"query": "2016 Latitude $10M good deal?"})

    assert contract.ui_intent == "buy_decision"
    assert contract.layout_type == "deal_card"
    assert contract.primary_cards[0]["card_type"] == "aircraft"
    assert contract.financial_cards
    assert contract.risk_cards
    assert contract.ui_flags["show_price_comparison"] is True
    assert contract.ui_flags["show_risk_panel"] is True
    assert contract.render_hints["buy_mode"] == "market_delta_emphasis"


def test_mission_ui_contract_mission_brief():
    response = {
        "answer": (
            "Mission Interpretation\n- Primary stage: TEB-LAX with 6 passengers.\n\n"
            "Final Verdict\n- CONDITIONAL FIT: Challenger 350 leads."
        ),
        "data_used": {
            "query_recommendation_requires_pipeline": True,
            "recommendation_pipeline": {"ranked_models": ["Challenger 350", "Praetor 600"]},
        },
    }
    normalized = _normalized("Need jet TEB to LAX 6 pax", response)
    contract = build_ui_render_contract(normalized, context={"query": "Need jet TEB to LAX 6 pax"})

    assert contract.ui_intent == "mission"
    assert contract.layout_type == "mission_brief"
    assert contract.primary_cards[0]["title"] == "Challenger 350"
    assert contract.render_hints["mission_mode"] == "constraint_first"


def test_apply_ui_contract_attached_to_payload():
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
    out = apply_consultant_response_normalization(payload, context={"query": "Alternatives to Longitude"})
    assert out["data_used"].get("ui_render_contract_applied") == 1
    contract = out["data_used"]["ui_render_contract"]
    assert contract["ui_intent"] == "alternative"
    assert contract["layout_type"] == "ranked_list"
    assert contract["headline"].startswith("Alternatives to")


def test_sections_use_structured_fields():
    response = {
        "answer": "Some unrelated narrative that UI must not depend on.",
        "data_used": {
            "authority_dispatch_kind": "comparison",
            "comparison_v2": {"status": "OK", "models": ["Gulfstream G650", "Falcon 8X"]},
        },
    }
    normalized = _normalized("G650 vs Falcon 8X", response)
    contract = build_ui_render_contract(normalized, context={"query": "G650 vs Falcon 8X"})
    section_types = {s["type"] for s in contract.sections}
    assert "overview" in section_types or "analysis" in section_types
    assert contract.headline == "Gulfstream G650 vs Falcon 8X"
