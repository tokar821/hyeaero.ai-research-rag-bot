"""Phase 34.5 — buy-decision budget audit and market-band alignment."""

from __future__ import annotations

import pytest

from services.deal_killer_engine import VERDICT_GOOD_DEAL, VERDICT_OVERPRICED, run_deal_killer_engine
from services.routing.authority_dispatch import respond_buy_decision
from tests.response_quality._text_extract import (
    extract_acquisition_budget_musd,
    is_buy_price_query,
)
from tests.response_quality.broker_recommendation_audit import audit_broker_recommendation

pytestmark = pytest.mark.deterministic


@pytest.mark.parametrize(
    "query",
    [
        "Is a 2015 Citation Latitude for $5M a good deal?",
        "2016 Citation Latitude at $6M — fair price?",
        "Is this 2015 Citation Latitude for $8M overpriced?",
    ],
)
def test_buy_price_query_not_treated_as_acquisition_budget(query: str) -> None:
    assert is_buy_price_query(query)
    assert extract_acquisition_budget_musd(query) is None


def test_mission_budget_still_detected() -> None:
    q = "Need 8 passengers TEB to LAX nonstop under $10M"
    assert not is_buy_price_query(q)
    assert extract_acquisition_budget_musd(q) == 10.0


def test_buy_decision_audit_uses_market_median_not_catalog() -> None:
    q = "Is a 2015 Citation Latitude for $15M a good deal?"
    ans = (
        "Aircraft: Citation Latitude\n"
        "Ask: $15.0M\n\n"
        "Market Reality:\n"
        "- Median: $18.0M\n\n"
        "Verdict:\n"
        "GOOD DEAL"
    )
    audit = audit_broker_recommendation(query=q, answer=ans)
    assert "BROKER_BUDGET_MISMATCH" not in audit.failures


def test_buy_decision_audit_no_false_budget_mismatch() -> None:
    q = "Is a 2015 Citation Latitude for $5M a good deal?"
    ans = (
        "Aircraft: Citation Latitude\n"
        "Year: 2015\n"
        "Ask: $5.0M\n\n"
        "Market Reality:\n"
        "- Verified catalog band (authority): roughly $13.5M–$24.3M (mid ~$18.0M).\n\n"
        "Verdict:\n"
        "GOOD DEAL"
    )
    audit = audit_broker_recommendation(query=q, answer=ans)
    assert "BROKER_BUDGET_MISMATCH" not in audit.failures


def test_deal_killer_low_ask_maps_to_good_deal() -> None:
    payload = run_deal_killer_engine(
        aircraft={"model": "Citation Latitude", "year": 2015, "ask_price": 5_000_000.0},
        market_data={
            "price_range_low": 13_500_000.0,
            "price_range_high": 24_300_000.0,
            "avg_price": 18_000_000.0,
            "comp_row_count": 0,
        },
        buyer_context={"mission_profile": {}},
    )
    assert payload.get("verdict") == VERDICT_GOOD_DEAL


def test_deal_killer_high_ask_maps_to_overpriced() -> None:
    payload = run_deal_killer_engine(
        aircraft={"model": "Citation Latitude", "year": 2015, "ask_price": 30_000_000.0},
        market_data={
            "price_range_low": 13_500_000.0,
            "price_range_high": 24_300_000.0,
            "avg_price": 18_000_000.0,
            "comp_row_count": 0,
        },
        buyer_context={"mission_profile": {}},
    )
    assert payload.get("verdict") == VERDICT_OVERPRICED


def test_respond_buy_decision_includes_authority_band_and_verdict() -> None:
    body = respond_buy_decision(
        "Is a 2015 Citation Latitude for $5M a good deal?",
        db=None,
        data_used={},
    )
    assert "Ask: $5.0M" in body
    assert "Verdict:" in body
    assert "Ask: $5.0M" in body
    assert "GOOD DEAL" in body.upper() or "FAIR DEAL" in body.upper()
