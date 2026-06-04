"""
Phase 43 — market reality orchestrator.

Reads existing market intelligence outputs; does not alter valuation, temporal,
adversarial, or client-context logic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from services.market_reality.buyer_leverage_analyzer import analyze_buyer_leverage
from services.market_reality.inventory_pressure_detector import detect_inventory_pressure
from services.market_reality.listing_confidence_analyzer import analyze_listing_confidence
from services.market_reality.listing_detector import ListingMode, ListingSignal, detect_listing_signal
from services.market_reality.market_reality_writer import write_market_reality
from services.market_reality.tail_investigation_mode import build_tail_investigation_brief
from services.market_reality.tail_broker_rewriter import rewrite_tail_investigation

logger = logging.getLogger(__name__)


def _load_bundle(model: str, ask_usd: Optional[float], data_used: Dict[str, Any]):
    """Use existing ``analyze_market`` — no formula changes."""
    db = data_used.get("db")
    auth = None
    ubs = data_used.get("unified_broker_state") or {}
    if isinstance(ubs, dict):
        md = ubs.get("market_data") or {}
        if isinstance(md, dict):
            auth = md

    from services.market_intelligence.market_intelligence_engine import analyze_market

    return analyze_market(db, model, ask_usd=ask_usd, auth_market=auth)


def _temporal_note(model: str, data_used: Dict[str, Any]) -> Optional[str]:
    temporal = data_used.get("temporal") or {}
    if isinstance(temporal, dict) and temporal.get("deal_timing_signal"):
        return f"Recent price drift signal: {temporal.get('deal_timing_signal')}."
    try:
        from services.temporal_market.temporal_market_intelligence import build_temporal_extension

        db = data_used.get("db")
        bundle = _load_bundle(model, None, data_used)
        ext = build_temporal_extension(model, bundle, db=db)
        if ext and not ext.temporal_confidence_low:
            return f"Price trend: {ext.price_drift_report.direction.value} — timing classified {ext.deal_timing_signal.value}."
    except Exception:
        pass
    return None


def build_market_reality_brief(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    du = data_used if isinstance(data_used, dict) else {}
    signal = detect_listing_signal(query)
    if signal.mode == ListingMode.NONE:
        return None

    if signal.mode == ListingMode.TAIL_INVESTIGATION and signal.registrations:
        reg = signal.registrations[0]
        brief = build_tail_investigation_brief(
            reg,
            model=signal.model,
            data_used=du,
        )
        facts = du.get("tail_selected_facts") or du.get("tail_facts") or []
        has_facts = bool(facts)
        return rewrite_tail_investigation(brief, registration=reg, facts_available=has_facts)

    model = signal.model
    if not model:
        return (
            "To read this as a deal, I need the aircraft model and the asking price "
            "(and ideally the tail number or listing link)."
        )

    ask_usd = signal.ask_musd * 1_000_000.0 if signal.ask_musd is not None else None
    bundle = _load_bundle(model, ask_usd, du)
    if bundle.band.reason == "catalog_acquisition_tier":
        from services.broker_reasoning.acquisition_tier_catalog import record_catalog_band_usage

        record_catalog_band_usage(du, model=model, band_reason=bundle.band.reason)

    price_analysis = analyze_listing_confidence(
        model=model,
        ask_usd=ask_usd,
        band=bundle.band,
    )
    inventory = detect_inventory_pressure(bundle.snapshot)
    temporal_timing = None
    tnote = _temporal_note(model, du)
    try:
        from services.temporal_market.temporal_market_intelligence import build_temporal_extension

        ext = build_temporal_extension(model, bundle, db=du.get("db"))
        if ext:
            temporal_timing = ext.deal_timing_signal.value
    except Exception:
        pass

    leverage = analyze_buyer_leverage(
        liquidity=bundle.liquidity,
        inventory_pressure=inventory["pressure"],
        temporal_timing=temporal_timing,
        price_confidence=price_analysis.get("confidence"),
    )

    du["market_reality"] = {
        "signal": signal.to_dict(),
        "price_analysis": price_analysis,
        "inventory": inventory,
        "leverage": leverage,
        "band_mid_usd": bundle.band.mid,
    }
    if bundle.deal_quality is not None:
        du["deal_quality"] = {
            "verdict": bundle.deal_quality.verdict.value,
            "display_verdict": bundle.deal_quality.display_verdict,
            "reason": bundle.deal_quality.reason,
            "position_pct": bundle.deal_quality.position_pct,
        }

    return write_market_reality(
        signal,
        price_analysis=price_analysis,
        inventory=inventory,
        leverage=leverage,
        band_mid_usd=bundle.band.mid,
        temporal_note=tnote,
    )


def apply_market_reality_layer(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    When the turn is listing- or deal-focused, lead with transaction-advisor prose.
    """
    q = (query or "").strip()
    du = data_used if isinstance(data_used, dict) else {}
    if not q:
        return (answer or "").strip()

    try:
        from services.broker_execution.output_governance import is_llm_primary_output

        if is_llm_primary_output(du):
            du["market_reality_layer_skipped_llm_primary"] = 1
            return (answer or "").strip()
    except Exception:
        pass

    try:
        from services.executive_broker.acquisition_budget_reality import should_block_market_reality

        if should_block_market_reality(du, q):
            return (answer or "").strip()
    except Exception:
        pass

    brief = build_market_reality_brief(q, data_used=du)
    if not brief:
        return (answer or "").strip()

    du["market_reality_layer_applied"] = 1
    logger.debug("market reality layer: mode=%s", (du.get("market_reality") or {}).get("signal", {}).get("mode"))

    raw = (answer or "").strip()
    if not raw or len(raw) < 80:
        return brief

    # Listing turns: lead with deal read, append non-duplicative market facts from pipeline.
    if brief.split("\n\n")[0].lower() not in raw[:200].lower():
        return f"{brief}\n\n{raw}".strip()
    return raw


__all__ = ["apply_market_reality_layer", "build_market_reality_brief"]
