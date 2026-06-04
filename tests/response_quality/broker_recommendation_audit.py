"""Phase 33 — Broker recommendation audit (final answer only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from tests.response_quality._text_extract import (
    extract_acquisition_budget_musd,
    extract_aircraft_like_tokens,
    extract_ask_musd,
    extract_market_median_musd,
    extract_pax,
    is_buy_price_query,
    mentions_nonstop,
    normalize,
)


@dataclass
class BrokerRecommendationAudit:
    score: float
    failures: List[str]
    recommended_models: List[str]


def _resolve_model(token: str) -> Optional[str]:
    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias

    canonical = resolve_aircraft_alias(token)
    return canonical or token


def _is_catalog_model(model: str) -> bool:
    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    rec = get_aircraft_authority_record(aircraft_model=model)
    return bool(rec)


def _estimate_route_nm(query: str) -> float:
    q = normalize(query)
    # intentionally coarse; we only use it for infeasibility detection
    if any(k in q for k in ("london", "singapore", "tokyo", "dubai", "honolulu")):
        return 7000.0
    if any(k in q for k in ("teb", "teterboro", "lax", "los angeles", "miami", "paris")):
        return 2500.0
    return 1200.0


def _range_ok(model: str, route_nm: float) -> bool:
    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    rec = get_aircraft_authority_record(aircraft_model=model)
    if not rec or not rec.nbaa_range_nm:
        return False
    return rec.nbaa_range_nm >= route_nm * 0.85


def audit_broker_recommendation(*, query: str, answer: str) -> BrokerRecommendationAudit:
    failures: List[str] = []
    tokens = extract_aircraft_like_tokens(answer)
    canonical: List[str] = []
    for t in tokens:
        c = _resolve_model(t)
        if c and c not in canonical:
            canonical.append(c)

    # Choose up to 3 "recommended" models as first mentioned catalog models.
    recommended: List[str] = [m for m in canonical if _is_catalog_model(m)][:3]
    if not recommended:
        # Not necessarily wrong, but broker recommendation audit requires an aircraft recommendation.
        failures.append("BROKER_BAD_AIRCRAFT")
        return BrokerRecommendationAudit(score=0.0, failures=failures, recommended_models=[])

    pax = extract_pax(query) or extract_pax(answer)
    budget_m = extract_acquisition_budget_musd(query) or extract_acquisition_budget_musd(answer)
    nonstop = mentions_nonstop(query) or mentions_nonstop(answer)
    route_nm = _estimate_route_nm(query)

    # Pax check is heuristic: if pax>=14 and the model is typically not ultra-long-range/high-capacity,
    # flag as mismatch. We avoid class assumptions and only enforce hard constraints when extracted.
    if pax and pax >= 16:
        failures.append("BROKER_PAX_MISMATCH")

    # Route feasibility check: if nonstop long-haul and model range short, flag.
    if nonstop and route_nm >= 4000:
        if not any(_range_ok(m, route_nm) for m in recommended):
            failures.append("BROKER_ROUTE_MISMATCH")

    # Budget check: mission/acquisition budget only — buy-decision queries embed ask price, not budget.
    if budget_m and not is_buy_price_query(query):
        from rag.aviation_engines.capabilities import find_catalog_matches, typical_market_price_usd

        cap = budget_m * 1_000_000 * 0.85
        for m in recommended:
            rows = find_catalog_matches([m])
            price = typical_market_price_usd(rows[0]) if rows else 0.0
            if price and price > cap:
                failures.append("BROKER_BUDGET_MISMATCH")
                break

    # Buy-path: optional consistency when ask is stated and answer includes a price verdict.
    if is_buy_price_query(query):
        ask_m = extract_ask_musd(answer) or extract_ask_musd(query)
        if ask_m and recommended:
            from rag.aviation_engines.capabilities import find_catalog_matches, typical_market_price_usd

            rows = find_catalog_matches([recommended[0]])
            catalog = typical_market_price_usd(rows[0]) if rows else 0.0
            median_m = extract_market_median_musd(answer)
            reference_usd = (
                median_m * 1_000_000.0 if median_m and median_m > 0 else catalog
            )
            ask_usd = ask_m * 1_000_000.0
            verdict_block = (answer or "").lower()
            if reference_usd > 0 and ask_usd > reference_usd * 1.15 and "good deal" in verdict_block:
                failures.append("BROKER_BUDGET_MISMATCH")
            elif reference_usd > 0 and ask_usd < reference_usd * 0.70 and "overpriced" in verdict_block:
                failures.append("BROKER_BUDGET_MISMATCH")

    # Score: start from 100 and deduct.
    score = 100.0
    if "BROKER_BAD_AIRCRAFT" in failures:
        score -= 60
    if "BROKER_ROUTE_MISMATCH" in failures:
        score -= 25
    if "BROKER_BUDGET_MISMATCH" in failures:
        score -= 10
    if "BROKER_PAX_MISMATCH" in failures:
        score -= 10
    score = max(0.0, round(score, 2))

    return BrokerRecommendationAudit(score=score, failures=sorted(set(failures)), recommended_models=recommended)

