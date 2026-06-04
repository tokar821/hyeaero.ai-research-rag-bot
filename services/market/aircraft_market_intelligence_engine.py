"""
Aircraft Market Intelligence Engine (AMIE) — Phase 24.

Deterministic market intelligence only. Does not alter routing or execution flow.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_INTEL_ENV = "ENABLE_MARKET_INTELLIGENCE"

_INVENTORY_RISING_THRESHOLD = 0.08
_INVENTORY_DECLINING_THRESHOLD = -0.08
_PRICE_APPRECIATING_THRESHOLD = 0.03
_PRICE_DEPRECIATING_THRESHOLD = -0.03


@dataclass
class MarketIntelligenceReport:
    aircraft: str
    market_state: str
    inventory_trend: str
    price_trend: str
    liquidity_trend: str
    replacement_risk: str
    market_strength_score: float
    confidence: float
    evidence: List[str] = field(default_factory=list)
    buy_timing: str = "neutral"
    sell_timing: str = "neutral"
    buy_timing_evidence: List[str] = field(default_factory=list)
    sell_timing_evidence: List[str] = field(default_factory=list)
    liquidity_score: float = 0.0
    report_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aircraft": self.aircraft,
            "market_state": self.market_state,
            "inventory_trend": self.inventory_trend,
            "price_trend": self.price_trend,
            "liquidity_trend": self.liquidity_trend,
            "replacement_risk": self.replacement_risk,
            "market_strength_score": round(float(self.market_strength_score), 2),
            "confidence": round(float(self.confidence), 3),
            "evidence": list(self.evidence),
            "buy_timing": self.buy_timing,
            "sell_timing": self.sell_timing,
            "buy_timing_evidence": list(self.buy_timing_evidence),
            "sell_timing_evidence": list(self.sell_timing_evidence),
            "liquidity_score": round(float(self.liquidity_score), 2),
            "report_id": self.report_id,
        }


def market_intelligence_enabled() -> bool:
    return (os.getenv(_INTEL_ENV) or "").strip().lower() in ("1", "true", "yes")


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def _listing_counts(source: Any) -> Tuple[int, int]:
    if isinstance(source, dict):
        try:
            current = int(source.get("current") or source.get("count") or 0)
        except (TypeError, ValueError):
            current = 0
        try:
            prior = int(source.get("prior") or source.get("prior_count") or 0)
        except (TypeError, ValueError):
            prior = 0
        return current, prior
    if isinstance(source, int):
        return source, 0
    return 0, 0


def analyze_inventory_trend(
    *,
    controller_listings: Any = None,
    aircraft_exchange_listings: Any = None,
    phly_listings: Any = None,
) -> str:
    """
    Analyze inventory direction from controller, aircraft exchange, and phly listing counts.

    Returns ``rising``, ``stable``, or ``declining``.
    """
    current_total = 0
    prior_total = 0
    for src in (controller_listings, aircraft_exchange_listings, phly_listings):
        cur, pri = _listing_counts(src)
        current_total += cur
        prior_total += pri

    if prior_total <= 0:
        if current_total >= 20:
            return "rising"
        if current_total <= 5:
            return "declining"
        return "stable"

    change = (current_total - prior_total) / prior_total
    if change >= _INVENTORY_RISING_THRESHOLD:
        return "rising"
    if change <= _INVENTORY_DECLINING_THRESHOLD:
        return "declining"
    return "stable"


def analyze_price_trend(historical_listing_data: Sequence[Dict[str, Any]]) -> str:
    """
    Analyze price direction from historical listing snapshots.

    Returns ``appreciating``, ``stable``, or ``depreciating``.
    """
    rows = [r for r in historical_listing_data if isinstance(r, dict)]
    prices: List[float] = []
    for row in rows:
        price = row.get("price_usd") or row.get("ask_usd") or row.get("ask_price")
        if price is None:
            continue
        try:
            val = float(price)
        except (TypeError, ValueError):
            continue
        if val > 0:
            prices.append(val)

    if len(prices) < 2:
        return "stable"

    mid = len(prices) // 2
    early = prices[:mid] or prices[:1]
    late = prices[mid:] or prices[-1:]
    early_avg = sum(early) / len(early)
    late_avg = sum(late) / len(late)
    if early_avg <= 0:
        return "stable"

    change = (late_avg - early_avg) / early_avg
    if change >= _PRICE_APPRECIATING_THRESHOLD:
        return "appreciating"
    if change <= _PRICE_DEPRECIATING_THRESHOLD:
        return "depreciating"
    return "stable"


def analyze_market_liquidity(
    *,
    listing_velocity: float = 0.0,
    inventory_levels: int = 0,
) -> float:
    """Score market liquidity 0–100 from listing velocity and inventory levels."""
    velocity = max(0.0, float(listing_velocity or 0))
    inventory = max(0, int(inventory_levels or 0))
    velocity_score = min(60.0, velocity * 6.0)
    inventory_score = max(0.0, 40.0 - abs(inventory - 12) * 2.5)
    return _clamp(velocity_score + inventory_score)


def _liquidity_trend_label(
    inventory_trend: str,
    liquidity_score: float,
) -> str:
    if inventory_trend == "declining" and liquidity_score >= 55:
        return "tightening"
    if inventory_trend == "rising" and liquidity_score <= 45:
        return "expanding"
    if inventory_trend == "rising":
        return "expanding"
    if inventory_trend == "declining":
        return "tightening"
    return "stable"


def evaluate_replacement_risk(aircraft: str) -> str:
    """
    Evaluate replacement risk from OEM roadmap proxies via AKAL.

    Returns ``LOW``, ``MODERATE``, or ``HIGH``.
    """
    from services.aircraft.aircraft_authority_service import get_aircraft_authority_record

    rec = get_aircraft_authority_record(aircraft_model=aircraft)
    if rec is None:
        return "MODERATE"

    risk = 0
    if not rec.current_in_production:
        risk += 2
    if rec.production_end_year:
        from datetime import datetime

        if datetime.now().year - int(rec.production_end_year) >= 3:
            risk += 1
    if rec.replacement_models:
        risk += 1
    if len(rec.direct_competitors) >= 3:
        risk += 1

    if risk >= 3:
        return "HIGH"
    if risk >= 1:
        return "MODERATE"
    return "LOW"


def _derive_market_state(
    inventory_trend: str,
    price_trend: str,
    liquidity_score: float,
) -> str:
    if inventory_trend == "rising" and price_trend == "depreciating":
        return "BUYER_MARKET"
    if inventory_trend == "declining" and price_trend == "appreciating":
        return "SELLER_MARKET"
    if liquidity_score >= 65 and price_trend == "depreciating":
        return "BUYER_MARKET"
    if liquidity_score <= 35 and price_trend == "appreciating":
        return "SELLER_MARKET"
    if inventory_trend == "rising" and liquidity_score >= 60:
        return "BUYER_MARKET"
    if inventory_trend == "declining" and liquidity_score <= 40:
        return "SELLER_MARKET"
    return "BALANCED_MARKET"


def _market_strength_score(
    *,
    liquidity_score: float,
    inventory_trend: str,
    price_trend: str,
    age_position: str = "unknown",
) -> float:
    inv_points = {"rising": 25.0, "stable": 50.0, "declining": 75.0}.get(inventory_trend, 50.0)
    price_points = {
        "appreciating": 80.0,
        "stable": 50.0,
        "depreciating": 25.0,
    }.get(price_trend, 50.0)
    age_points = {
        "young": 75.0,
        "mid_life": 55.0,
        "mature": 35.0,
        "unknown": 50.0,
    }.get(age_position, 50.0)
    return _clamp(
        liquidity_score * 0.35 + inv_points * 0.25 + price_points * 0.25 + age_points * 0.15
    )


def evaluate_purchase_timing(
    *,
    market_state: str,
    inventory_trend: str,
    price_trend: str,
    liquidity_score: float,
) -> Tuple[str, List[str]]:
    """Return buy timing verdict and evidence."""
    evidence: List[str] = []
    if market_state == "BUYER_MARKET":
        evidence.append("buyer-favorable market state with elevated supply or softening prices")
        if inventory_trend == "rising":
            evidence.append("inventory expanding increases buyer leverage")
        if price_trend == "depreciating":
            evidence.append("observed listing prices trending lower")
        return "favorable", evidence

    if market_state == "SELLER_MARKET" and price_trend == "appreciating":
        evidence.append("seller-favorable market with appreciating listing prices")
        evidence.append("buyers face tighter inventory and rising ask levels")
        return "unfavorable", evidence

    if inventory_trend == "declining" and price_trend == "appreciating":
        evidence.append("inventory tightening while prices strengthen")
        return "unfavorable", evidence

    if liquidity_score >= 60 and price_trend == "stable":
        evidence.append("adequate liquidity with stable pricing supports neutral buy timing")
        return "neutral", evidence

    evidence.append("mixed market signals suggest neutral buy timing")
    return "neutral", evidence


def evaluate_sale_timing(
    *,
    market_state: str,
    inventory_trend: str,
    price_trend: str,
    liquidity_score: float,
) -> Tuple[str, List[str]]:
    """Return sell timing verdict and evidence."""
    evidence: List[str] = []
    if market_state == "SELLER_MARKET":
        evidence.append("seller-favorable market state with tightening inventory or rising prices")
        if inventory_trend == "declining":
            evidence.append("declining inventory supports stronger seller positioning")
        if price_trend == "appreciating":
            evidence.append("observed listing prices trending higher")
        return "favorable", evidence

    if market_state == "BUYER_MARKET" and price_trend == "depreciating":
        evidence.append("buyer-favorable market with depreciating listing prices")
        evidence.append("sellers face expanded inventory and softer pricing")
        return "unfavorable", evidence

    if inventory_trend == "rising" and price_trend == "depreciating":
        evidence.append("inventory expanding while prices weaken")
        return "unfavorable", evidence

    if liquidity_score >= 55 and price_trend == "stable":
        evidence.append("stable liquidity with flat pricing supports neutral sell timing")
        return "neutral", evidence

    evidence.append("mixed market signals suggest neutral sell timing")
    return "neutral", evidence


def _inventory_evidence(
    controller_listings: Any,
    aircraft_exchange_listings: Any,
    phly_listings: Any,
    trend: str,
) -> List[str]:
    evidence: List[str] = []
    labels = (
        ("controller", controller_listings),
        ("aircraft exchange", aircraft_exchange_listings),
        ("phly", phly_listings),
    )
    for label, src in labels:
        cur, pri = _listing_counts(src)
        if cur or pri:
            if pri > 0:
                pct = int(((cur - pri) / pri) * 100)
                evidence.append(f"{label} listings {cur} vs prior {pri} ({pct:+d}%)")
            else:
                evidence.append(f"{label} listings at {cur} units")
    if trend == "rising":
        evidence.append("aggregate inventory expanding across listing sources")
    elif trend == "declining":
        evidence.append("aggregate inventory contracting across listing sources")
    else:
        evidence.append("aggregate inventory stable versus prior period")
    return evidence


def build_market_intelligence_report(
    aircraft: str,
    *,
    controller_listings: Any = None,
    aircraft_exchange_listings: Any = None,
    phly_listings: Any = None,
    historical_listing_data: Optional[Sequence[Dict[str, Any]]] = None,
    listing_velocity: float = 0.0,
    age_position: str = "unknown",
) -> MarketIntelligenceReport:
    """Build a full deterministic market intelligence report for one aircraft."""
    hist = list(historical_listing_data or [])
    inventory_trend = analyze_inventory_trend(
        controller_listings=controller_listings,
        aircraft_exchange_listings=aircraft_exchange_listings,
        phly_listings=phly_listings,
    )
    price_trend = analyze_price_trend(hist)

    cur_total = sum(_listing_counts(s)[0] for s in (controller_listings, aircraft_exchange_listings, phly_listings))
    liquidity_score = analyze_market_liquidity(
        listing_velocity=listing_velocity,
        inventory_levels=cur_total,
    )
    liquidity_trend = _liquidity_trend_label(inventory_trend, liquidity_score)
    replacement_risk = evaluate_replacement_risk(aircraft)
    market_state = _derive_market_state(inventory_trend, price_trend, liquidity_score)
    strength = _market_strength_score(
        liquidity_score=liquidity_score,
        inventory_trend=inventory_trend,
        price_trend=price_trend,
        age_position=age_position,
    )

    evidence = _inventory_evidence(
        controller_listings, aircraft_exchange_listings, phly_listings, inventory_trend
    )
    if hist:
        prices = [
            float(r.get("price_usd") or r.get("ask_usd") or r.get("ask_price") or 0)
            for r in hist
            if isinstance(r, dict)
        ]
        prices = [p for p in prices if p > 0]
        if len(prices) >= 2:
            evidence.append(
                f"historical listing prices moved from avg ${prices[0]:,.0f} to ${prices[-1]:,.0f}"
            )
    evidence.append(f"liquidity score {liquidity_score:.0f}/100 at {cur_total} active listings")
    evidence.append(f"replacement risk assessed {replacement_risk} from production and competitor signals")

    buy_timing, buy_ev = evaluate_purchase_timing(
        market_state=market_state,
        inventory_trend=inventory_trend,
        price_trend=price_trend,
        liquidity_score=liquidity_score,
    )
    sell_timing, sell_ev = evaluate_sale_timing(
        market_state=market_state,
        inventory_trend=inventory_trend,
        price_trend=price_trend,
        liquidity_score=liquidity_score,
    )

    confidence = 0.55
    if cur_total > 0:
        confidence += 0.15
    if hist:
        confidence += 0.15
    if replacement_risk != "MODERATE" or aircraft != "unknown":
        confidence += 0.1
    confidence = min(0.95, confidence)

    report_id = hashlib.sha256(
        "|".join(
            [
                aircraft,
                market_state,
                inventory_trend,
                price_trend,
                str(round(strength, 2)),
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]

    return MarketIntelligenceReport(
        aircraft=aircraft,
        market_state=market_state,
        inventory_trend=inventory_trend,
        price_trend=price_trend,
        liquidity_trend=liquidity_trend,
        replacement_risk=replacement_risk,
        market_strength_score=strength,
        confidence=confidence,
        evidence=evidence,
        buy_timing=buy_timing,
        sell_timing=sell_timing,
        buy_timing_evidence=buy_ev,
        sell_timing_evidence=sell_ev,
        liquidity_score=liquidity_score,
        report_id=report_id,
    )


def _extract_aircraft(data_used: Dict[str, Any]) -> str:
    market = data_used.get("aircraft_authority_market")
    if isinstance(market, dict) and market.get("canonical_name"):
        return str(market["canonical_name"])
    rec_rows = data_used.get("consultant_recommendations")
    if isinstance(rec_rows, list) and rec_rows:
        row = rec_rows[0]
        if isinstance(row, dict):
            name = str(row.get("model") or row.get("aircraft") or "").strip()
            if name:
                return name
    alt = data_used.get("alternative_execution")
    if isinstance(alt, dict) and alt.get("target"):
        return str(alt["target"])
    return "unknown"


def _extract_market_inputs(data_used: Dict[str, Any]) -> Dict[str, Any]:
    snap = data_used.get("market_listing_snapshot")
    if not isinstance(snap, dict):
        snap = data_used.get("listing_inventory") or {}
    if not isinstance(snap, dict):
        snap = {}

    return {
        "controller_listings": snap.get("controller") or snap.get("controller_listings"),
        "aircraft_exchange_listings": snap.get("aircraft_exchange") or snap.get("aircraft_exchange_listings"),
        "phly_listings": snap.get("phly") or snap.get("phly_listings"),
        "historical_listing_data": snap.get("historical_listing_data")
        or data_used.get("historical_listing_data")
        or data_used.get("market_price_history")
        or [],
        "listing_velocity": float(snap.get("listing_velocity") or data_used.get("listing_velocity") or 0),
        "age_position": str(
            (data_used.get("aircraft_authority_market") or {}).get("age_position")
            or snap.get("age_position")
            or "unknown"
        ),
    }


def build_market_intelligence(
    query: str,
    response: Any,
) -> Dict[str, Any]:
    """Build market intelligence bundle from a consultant response payload."""
    payload = response if isinstance(response, dict) else {}
    du = payload.get("data_used") if isinstance(payload.get("data_used"), dict) else {}

    aircraft = _extract_aircraft(du)
    inputs = _extract_market_inputs(du)

    if not any(
        [
            inputs["controller_listings"],
            inputs["aircraft_exchange_listings"],
            inputs["phly_listings"],
            inputs["historical_listing_data"],
        ]
    ):
        return {
            "aircraft": aircraft,
            "status": "INSUFFICIENT_DATA",
            "confidence": 0,
            "trends": {},
            "market_state": "UNKNOWN",
            "inventory_trend": "",
            "price_trend": "",
            "liquidity_trend": "",
            "replacement_risk": "",
            "market_strength_score": 0.0,
            "evidence": [],
            "buy_timing": "neutral",
            "sell_timing": "neutral",
            "buy_timing_evidence": [],
            "sell_timing_evidence": [],
            "liquidity_score": 0.0,
            "report_id": "",
            "market_panel": {
                "market_state": "UNKNOWN",
                "inventory_trend": "",
                "liquidity": 0.0,
                "price_trend": "",
                "buy_timing": "neutral",
                "sell_timing": "neutral",
            },
        }

    report = build_market_intelligence_report(aircraft, **inputs)

    panel = {
        "market_state": report.market_state,
        "inventory_trend": report.inventory_trend,
        "liquidity": report.liquidity_score,
        "price_trend": report.price_trend,
        "buy_timing": report.buy_timing,
        "sell_timing": report.sell_timing,
    }

    out = report.to_dict()
    out["market_panel"] = panel
    return out


def attach_market_intelligence_if_enabled(
    query: str,
    response: Any,
) -> Any:
    """Attach ``data_used.market_intelligence`` when env flag enabled."""
    if not market_intelligence_enabled():
        return response
    if not isinstance(response, dict):
        return response
    out = dict(response)
    du = dict(out.get("data_used") or {}) if isinstance(out.get("data_used"), dict) else {}
    du["market_intelligence"] = build_market_intelligence(query, out)
    out["data_used"] = du
    return out


def evaluate_market_intelligence_hooks(response: Any) -> List[str]:
    """
    Optional evaluation hooks — market consistency, trend consistency, unsupported claims.

    Returns failure tokens for consultant_evaluator integration.
    """
    if not isinstance(response, dict):
        return []
    du = response.get("data_used")
    if not isinstance(du, dict):
        return []
    bundle = du.get("market_intelligence")
    if not isinstance(bundle, dict):
        return []

    failures: List[str] = []
    state = str(bundle.get("market_state") or "")
    inv = str(bundle.get("inventory_trend") or "")
    price = str(bundle.get("price_trend") or "")
    evidence = bundle.get("evidence") or []
    confidence = float(bundle.get("confidence") or 0)

    if state == "BUYER_MARKET" and inv == "declining" and price == "appreciating":
        failures.append("market_consistency")
    if state == "SELLER_MARKET" and inv == "rising" and price == "depreciating":
        failures.append("market_consistency")

    liq_trend = str(bundle.get("liquidity_trend") or "")
    if liq_trend == "tightening" and inv == "rising":
        failures.append("trend_consistency")
    if liq_trend == "expanding" and inv == "declining":
        failures.append("trend_consistency")

    if confidence >= 0.85 and not evidence:
        failures.append("unsupported_market_claims")

    answer = str(response.get("answer") or "").lower()
    if answer and not evidence:
        market_claims = (
            "buyer market" in answer
            or "seller market" in answer
            or "appreciating" in answer
            or "depreciating" in answer
        )
        if market_claims and confidence < 0.5:
            failures.append("unsupported_market_claims")

    return list(dict.fromkeys(failures))


__all__ = [
    "MarketIntelligenceReport",
    "analyze_inventory_trend",
    "analyze_market_liquidity",
    "analyze_price_trend",
    "attach_market_intelligence_if_enabled",
    "build_market_intelligence",
    "build_market_intelligence_report",
    "evaluate_market_intelligence_hooks",
    "evaluate_purchase_timing",
    "evaluate_replacement_risk",
    "evaluate_sale_timing",
    "market_intelligence_enabled",
]
