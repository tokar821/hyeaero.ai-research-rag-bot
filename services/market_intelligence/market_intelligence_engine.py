"""Orchestrates listing analytics, liquidity, market bands, and deal quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from services.market_intelligence.deal_quality_engine import (
    DealQualityResult,
    DealQualityVerdict,
    evaluate_deal_quality,
)
from services.market_intelligence.liquidity_scoring import LiquidityScore, compute_liquidity_score
from services.market_intelligence.listing_analytics import (
    MarketSnapshot,
    build_market_snapshot,
    fetch_listing_rows,
)
from services.market_intelligence.market_band_builder import (
    BandConfidence,
    MarketBand,
    build_market_band_from_asks,
)

if TYPE_CHECKING:
    from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketIntelligenceBundle:
    snapshot: MarketSnapshot
    liquidity: LiquidityScore
    band: MarketBand
    deal_quality: Optional[DealQualityResult] = None
    listing_rows: tuple = ()


def fmt_musd(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def analyze_market(
    db: Optional["PostgresClient"],
    model: str,
    *,
    year: Optional[int] = None,
    ask_usd: Optional[float] = None,
    auth_market: Optional[Dict[str, Any]] = None,
) -> MarketIntelligenceBundle:
    """
    Full market-intelligence pass: snapshot, liquidity, band, optional deal quality.

    Falls back to authority catalog band when listing depth is insufficient (no invented asks).
    """
    rows: List[Dict[str, Any]] = []
    snapshot: MarketSnapshot

    if db is not None and model:
        try:
            from services.market_intelligence.listing_analytics import fetch_aircraftpost_for_sale_count

            rows = fetch_listing_rows(db, model)
            ap_n = fetch_aircraftpost_for_sale_count(db, model)
            snapshot = build_market_snapshot(model, rows, aircraftpost_for_sale=ap_n)
        except Exception:
            logger.exception("analyze_market failed for %s", model)
            snapshot = MarketSnapshot(
                model=model,
                active_listing_count=0,
                median_ask_price=None,
                low_ask_price=None,
                high_ask_price=None,
                median_year=None,
                average_days_on_market=None,
                last_refresh=None,
                insufficient_reason="listing_fetch_error",
            )
    else:
        snapshot = MarketSnapshot(
            model=model,
            active_listing_count=0,
            median_ask_price=None,
            low_ask_price=None,
            high_ask_price=None,
            median_year=None,
            average_days_on_market=None,
            last_refresh=None,
            insufficient_reason="no_database",
        )

    liquidity = compute_liquidity_score(snapshot)
    band = build_market_band_from_asks(snapshot, rows)

    if band.confidence == BandConfidence.INSUFFICIENT and auth_market:
        band = _band_from_authority(auth_market, snapshot)
    if band.confidence == BandConfidence.INSUFFICIENT:
        band = _band_from_catalog_tier(model, snapshot)

    deal: Optional[DealQualityResult] = None
    if ask_usd is not None:
        deal = evaluate_deal_quality(model=model, year=year, ask_usd=ask_usd, band=band)

    return MarketIntelligenceBundle(
        snapshot=snapshot,
        liquidity=liquidity,
        band=band,
        deal_quality=deal,
        listing_rows=tuple(rows),
    )


def _band_from_catalog_tier(model: str, snapshot: MarketSnapshot) -> MarketBand:
    """Directional band from acquisition tier when listings/authority are unavailable."""
    from services.broker_reasoning.acquisition_tier_catalog import ACQUISITION_TIER_MUSD

    tier_m = ACQUISITION_TIER_MUSD.get(model)
    if tier_m is None:
        return MarketBand(
            low=None,
            mid=None,
            high=None,
            confidence=BandConfidence.INSUFFICIENT,
            listing_count=0,
            reason="catalog_tier_unknown",
        )
    mid = float(tier_m) * 1_000_000.0
    return MarketBand(
        low=mid * 0.72,
        mid=mid,
        high=mid * 1.28,
        confidence=BandConfidence.MODERATE,
        listing_count=snapshot.active_listing_count,
        reason="catalog_acquisition_tier",
    )


def _band_from_authority(
    auth_market: Dict[str, Any],
    snapshot: MarketSnapshot,
) -> MarketBand:
    if str(auth_market.get("status") or "").upper() != "OK":
        return MarketBand(
            low=None,
            mid=None,
            high=None,
            confidence=BandConfidence.INSUFFICIENT,
            listing_count=0,
            reason="authority_band_unavailable",
        )
    band_usd = auth_market.get("expected_market_band_usd") or {}
    low = band_usd.get("low")
    mid = band_usd.get("mid")
    high = band_usd.get("high")
    if low is None or high is None:
        return MarketBand(
            low=None,
            mid=None,
            high=None,
            confidence=BandConfidence.INSUFFICIENT,
            listing_count=0,
            reason="authority_band_incomplete",
        )
    return MarketBand(
        low=float(low),
        mid=float(mid) if mid is not None else (float(low) + float(high)) / 2.0,
        high=float(high),
        confidence=BandConfidence.MODERATE,
        listing_count=snapshot.active_listing_count,
        reason="authority_catalog_band",
    )


def bundle_to_market_data(bundle: MarketIntelligenceBundle) -> Dict[str, Any]:
    """Map bundle into deal-killer compatible market_data dict."""
    band = bundle.band
    snap = bundle.snapshot
    out: Dict[str, Any] = {
        "comp_row_count": snap.active_listing_count,
        "liquidity": bundle.liquidity.band.value.lower(),
        "liquidity_score": bundle.liquidity.score,
        "market_intelligence": True,
    }
    if band.low is not None and band.high is not None:
        out["price_range_low"] = band.low
        out["price_range_high"] = band.high
        out["avg_price"] = band.mid
        out["market_band_confidence"] = band.confidence.value
    elif snap.median_ask_price is not None:
        out["avg_price"] = snap.median_ask_price
        if snap.low_ask_price is not None:
            out["price_range_low"] = snap.low_ask_price
        if snap.high_ask_price is not None:
            out["price_range_high"] = snap.high_ask_price
    if band.reason == "authority_catalog_band":
        out["authority_band"] = True
    return out


def enrich_buy_decision(
    *,
    db: Any,
    model: str,
    year: Optional[int],
    ask_usd: Optional[float],
    auth_market: Optional[Dict[str, Any]],
    data_used: Optional[Dict[str, Any]],
) -> tuple[MarketIntelligenceBundle, Dict[str, Any]]:
    bundle = analyze_market(db, model, year=year, ask_usd=ask_usd, auth_market=auth_market)
    market_data = bundle_to_market_data(bundle)
    if isinstance(data_used, dict):
        data_used["market_intelligence"] = {
            "snapshot": {
                "model": bundle.snapshot.model,
                "active_listing_count": bundle.snapshot.active_listing_count,
                "median_ask_price": bundle.snapshot.median_ask_price,
                "average_days_on_market": bundle.snapshot.average_days_on_market,
                "stale": bundle.snapshot.stale,
                "insufficient_reason": bundle.snapshot.insufficient_reason,
            },
            "liquidity": {
                "score": bundle.liquidity.score,
                "band": bundle.liquidity.band.value,
            },
            "band": {
                "low": bundle.band.low,
                "mid": bundle.band.mid,
                "high": bundle.band.high,
                "confidence": bundle.band.confidence.value,
            },
        }
        if bundle.deal_quality is not None:
            data_used["deal_quality"] = {
                "verdict": bundle.deal_quality.verdict.value,
                "display_verdict": bundle.deal_quality.display_verdict,
                "reason": bundle.deal_quality.reason,
                "position_pct": bundle.deal_quality.position_pct,
            }
    return bundle, market_data


def apply_deal_quality_to_verdict(
    verdict_payload: Dict[str, Any],
    deal: Optional[DealQualityResult],
) -> Dict[str, Any]:
    """Override price-position verdict when deterministic deal quality is confident."""
    if deal is None or deal.verdict == DealQualityVerdict.INSUFFICIENT_DATA:
        return verdict_payload
    out = dict(verdict_payload)
    out["verdict"] = deal.display_verdict
    reasons = list(out.get("key_reasons") or [])
    reasons.insert(0, deal.reason)
    out["key_reasons"] = reasons[:6]
    out["deal_quality_verdict"] = deal.verdict.value
    return out


def format_buy_decision_market_sections(
    bundle: MarketIntelligenceBundle,
    market_data: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    snap = bundle.snapshot
    band = bundle.band
    liq = bundle.liquidity

    if band.low is not None and band.high is not None:
        mid = band.mid
        lines.append(
            f"- Market Band: {fmt_musd(band.low)}–{fmt_musd(band.high)} "
            f"(confidence: {band.confidence.value})"
        )
        if mid is not None:
            lines.append(f"- Median: {fmt_musd(mid)}")
    elif market_data.get("authority_band"):
        low = market_data.get("price_range_low")
        high = market_data.get("price_range_high")
        mid = market_data.get("avg_price")
        if low is not None and high is not None:
            lines.append(f"- Market Band (authority catalog): {fmt_musd(low)}–{fmt_musd(high)}")
            if mid is not None:
                lines.append(f"- Median: {fmt_musd(mid)}")

    if snap.active_listing_count > 0:
        lines.append(f"- Active Listings: {snap.active_listing_count}")
    if snap.average_days_on_market is not None:
        lines.append(f"- Avg Days on Market: {int(round(snap.average_days_on_market))}")
    lines.append(f"- Liquidity: {liq.band.value} (score {liq.score}/100)")

    if not lines:
        lines.append("- Limited synced listing intelligence — using catalog authority where available.")
    return lines


def format_deal_assessment(
    ask_usd: Optional[float],
    deal: Optional[DealQualityResult],
) -> List[str]:
    lines: List[str] = []
    if ask_usd is not None:
        lines.append(f"- Ask: {fmt_musd(ask_usd)}")
    if deal is None:
        return lines
    if deal.position_pct is not None:
        pct = deal.position_pct * 100.0
        if pct < 0:
            lines.append(f"- Position: {abs(pct):.1f}% below market median")
        elif pct > 0:
            lines.append(f"- Position: {pct:.1f}% above market median")
        else:
            lines.append("- Position: at market median")
    if deal.reason:
        lines.append(f"- Assessment: {deal.reason}")
    return lines


def format_valuation_response(
    *,
    model: str,
    year: str,
    db: Any = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Market-aware valuation block; never fabricates prices."""
    from services.aircraft.aircraft_authority_service import build_authoritative_market_context

    auth = build_authoritative_market_context(year=None, model=model, ask_usd=None, db=db)
    bundle = analyze_market(db, model, auth_market=auth)
    snap = bundle.snapshot
    band = bundle.band
    liq = bundle.liquidity

    if isinstance(data_used, dict):
        data_used["market_intelligence"] = {
            "valuation": True,
            "band_confidence": band.confidence.value,
            "liquidity_score": liq.score,
        }

    lines = [f"Aircraft: {model}", f"Year: {year}", "", "Market Reality:"]

    if band.confidence != BandConfidence.INSUFFICIENT and band.low is not None and band.high is not None:
        lines.append(f"Market Band: {fmt_musd(band.low)}–{fmt_musd(band.high)}")
        if band.mid is not None:
            lines.append(f"Median: {fmt_musd(band.mid)}")
        lines.append(f"Liquidity: {liq.band.value} (score {liq.score}/100)")
        lines.append(f"Confidence: {band.confidence.value}")
        lines.append("")
        lines.append("Verdict:")
        lines.append("MARKET_CONTEXT_AVAILABLE")
        if snap.active_listing_count > 0:
            lines.append(f"(Based on {snap.active_listing_count} active listing signal(s).)")
        return "\n".join(lines)

    reasons: List[str] = []
    if not model:
        reasons.append("unresolved aircraft")
    if snap.insufficient_reason == "too_few_listings" or band.reason in (
        "too_few_listings",
        "too_few_listings_after_outlier_rejection",
    ):
        reasons.append("too few listings")
    if snap.stale or snap.insufficient_reason == "stale_market" or band.reason == "stale_market":
        reasons.append("stale market data")
    if db is None:
        reasons.append("no synced listing database in this execution path")
    if band.reason == "authority_catalog_band" and band.mid is not None:
        lines.append(f"Catalog band (authority): {fmt_musd(band.low)}–{fmt_musd(band.high)}")
        lines.append(f"Median: {fmt_musd(band.mid)}")
        lines.append(f"Liquidity: {liq.band.value} (score {liq.score}/100)")
        lines.append("Confidence: MODERATE (catalog authority — not live listing median)")
        lines.append("")
        lines.append("Verdict:")
        lines.append("INSUFFICIENT_DATA")
        lines.append("(Listing depth below band threshold; catalog band shown for orientation only.)")
        return "\n".join(lines)

    why = ", ".join(reasons) if reasons else "insufficient verified listing depth"
    lines.append(f"Insufficient market band: {why}.")
    lines.append(f"Liquidity: {liq.band.value} (score {liq.score}/100)" if snap.active_listing_count else "Liquidity: unavailable")
    lines.append("")
    lines.append("Verdict:")
    lines.append("INSUFFICIENT_DATA")
    return "\n".join(lines)
