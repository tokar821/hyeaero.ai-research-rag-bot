"""Phase 35.7 — market intelligence E2E-style response checks."""

from __future__ import annotations

import re

import pytest

from services.market_intelligence.deal_quality_engine import DealQualityVerdict, evaluate_deal_quality
from services.market_intelligence.market_band_builder import BandConfidence, MarketBand
from services.routing.authority_dispatch import respond_buy_decision
from services.consultant.answer_recovery import recover_valuation_answer
from tests.response_quality._text_extract import extract_ask_musd

pytestmark = pytest.mark.deterministic

_HAS_BAND_RE = re.compile(
    r"market\s+band|catalog\s+band|\$\d+(?:\.\d+)?M\s*[–\-]\s*\$\d+(?:\.\d+)?M",
    re.I,
)
_HAS_LIQUIDITY_RE = re.compile(r"liquidity:\s*(HIGH|GOOD|MODERATE|THIN)", re.I)
_VERDICT_RE = re.compile(
    r"\b(GOOD DEAL|FAIR DEAL|OVERPRICED|INSUFFICIENT_DATA|HIGH RISK|DO NOT BUY)\b",
    re.I,
)


def _category_valuation_realism(answer: str) -> list[str]:
    failures: list[str] = []
    if re.search(r"\$\d{1,3}(?:\.\d+)?M", answer) and "INSUFFICIENT_DATA" in answer.upper():
        if "catalog band" not in answer.lower() and "market band" not in answer.lower():
            failures.append("VALUATION_REALISM:invented_price_with_insufficient")
    return failures


def _category_deal_quality_alignment(query: str, answer: str) -> list[str]:
    failures: list[str] = []
    ask = extract_ask_musd(query)
    if ask is None:
        return failures
    m = re.search(r"(\d+(?:\.\d+)?)%\s+below\s+market\s+median", answer, re.I)
    if m and "GOOD DEAL" not in answer.upper():
        failures.append("DEAL_QUALITY_ALIGNMENT:below_median_not_good_deal")
    m2 = re.search(r"(\d+(?:\.\d+)?)%\s+above\s+market\s+median", answer, re.I)
    if m2 and "OVERPRICED" not in answer.upper():
        failures.append("DEAL_QUALITY_ALIGNMENT:above_median_not_overpriced")
    return failures


def _category_liquidity_present(answer: str) -> list[str]:
    if _HAS_LIQUIDITY_RE.search(answer):
        return []
    return ["LIQUIDITY_PRESENT:missing"]


def _category_market_band_present(answer: str) -> list[str]:
    if _HAS_BAND_RE.search(answer):
        return []
    return ["MARKET_BAND_PRESENT:missing"]


def audit_market_intelligence_answer(*, query: str, answer: str) -> list[str]:
    failures: list[str] = []
    failures.extend(_category_valuation_realism(answer))
    failures.extend(_category_deal_quality_alignment(query, answer))
    q_lower = (query or "").lower()
    if "good deal" in q_lower or "fair price" in q_lower or "overpriced" in q_lower:
        failures.extend(_category_liquidity_present(answer))
        failures.extend(_category_market_band_present(answer))
    return failures


@pytest.mark.parametrize(
    "query",
    [
        "Is a 2015 Citation Latitude for $5M a good deal?",
        "2016 Citation Latitude at $6M — fair price?",
    ],
)
def test_buy_decision_market_sections(query: str) -> None:
    body = respond_buy_decision(query, db=None, data_used={})
    assert "Market Reality:" in body
    assert "Verdict:" in body
    assert _VERDICT_RE.search(body)
    failures = audit_market_intelligence_answer(query=query, answer=body)
    assert not failures, failures


def test_valuation_recovery_structure() -> None:
    body = recover_valuation_answer("What is a 2019 Citation Latitude worth?", data_used={})
    assert "Aircraft:" in body
    assert "Verdict:" in body
    failures = audit_market_intelligence_answer(
        query="What is a 2019 Citation Latitude worth?",
        answer=body,
    )
    assert not failures, failures


def test_deal_quality_deterministic_alignment() -> None:
    band = MarketBand(
        low=10.2e6,
        mid=11.8e6,
        high=13.1e6,
        confidence=BandConfidence.HIGH,
        listing_count=12,
    )
    dq = evaluate_deal_quality(
        model="Citation Latitude",
        year=2018,
        ask_usd=9.5e6,
        band=band,
    )
    assert dq.verdict == DealQualityVerdict.GOOD_DEAL
    body = respond_buy_decision(
        "Is a 2018 Citation Latitude for $9.5M a good deal?",
        db=None,
        data_used={},
    )
    assert "GOOD DEAL" in body.upper()
