"""Consistency injection before response formatting (normalization only)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from services.consistency.cross_model_identity import (
    CanonicalAircraftIdentity,
    resolve_canonical_identity,
    resolve_comparison_identities,
)
from services.consistency.pipeline_agreement_checker import (
    AgreementFlag,
    PipelineAgreementReport,
    check_pipeline_agreement,
)
from services.consistency.unified_broker_state import UnifiedBrokerState

logger = logging.getLogger(__name__)


def _resolve_adversarial_metadata(du: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    adv = du.get("adversarial")
    if isinstance(adv, dict) and adv.get("normalized_query"):
        return adv
    from services.adversarial.adversarial_preprocessor import (
        _adversarial_from_data_used,
        to_unified_adversarial_metadata,
    )

    clean = _adversarial_from_data_used(du)
    if clean is not None:
        return to_unified_adversarial_metadata(clean)
    return None


def _attach_temporal(state: UnifiedBrokerState, db: Any = None) -> UnifiedBrokerState:
    """Additive temporal overlay — does not alter market intelligence math."""
    from services.temporal_market.temporal_market_intelligence import build_temporal_extension

    if not state.canonical_model:
        return state
    state.temporal = build_temporal_extension(
        model=state.canonical_model,
        db=db,
        market_bundle=state.market_bundle,
    )
    return state


def inject_consistency(
    data_used: Optional[Dict[str, Any]],
    state: UnifiedBrokerState,
    report: PipelineAgreementReport,
) -> UnifiedBrokerState:
    """
    Reconcile layer stamps in ``data_used`` to unified state (no routing changes).
    """
    if not isinstance(data_used, dict):
        return state

    canon = state.canonical_model
    if not canon:
        return state

    if AgreementFlag.MODEL_MISMATCH in report.flags:
        data_used["buy_decision_dispatch"] = {
            **(data_used.get("buy_decision_dispatch") or {}),
            "model": canon,
        }
        verified = data_used.get("verified_recovery_models")
        if isinstance(verified, list) and verified:
            data_used["verified_recovery_models"] = [canon] + [m for m in verified if m != canon][:2]

    if state.market_bundle is not None:
        band = state.market_bundle.band
        liq = state.market_bundle.liquidity
        data_used["market_intelligence"] = {
            **(data_used.get("market_intelligence") or {}),
            "snapshot": {
                "model": canon,
                "active_listing_count": state.market_snapshot.active_listing_count
                if state.market_snapshot
                else 0,
            },
            "band": {
                "low": band.low,
                "mid": band.mid,
                "high": band.high,
                "confidence": band.confidence.value,
            },
            "liquidity": {"score": liq.score, "band": liq.band.value},
        }
    if AgreementFlag.VERDICT_INCONSISTENCY in report.flags and state.deal_quality is not None:
        if isinstance(data_used.get("deal_killer"), dict):
            data_used["deal_killer"]["verdict"] = state.deal_quality.display_verdict

    data_used["unified_broker_state"] = state.to_data_used_dict()
    data_used["pipeline_agreement"] = {
        "aligned": report.aligned,
        "flags": [f.value for f in report.flags],
        "details": list(report.details),
    }
    return state


def prepare_buy_decision_state(
    *,
    query: str,
    parsed: Dict[str, Any],
    db: Any = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> UnifiedBrokerState:
    """Build unified state once — single market intelligence pass."""
    from services.aircraft.aircraft_authority_service import (
        build_authoritative_market_context,
        get_aircraft_authority_record,
    )
    from services.deal_killer_engine import run_deal_killer_engine
    from services.market_intelligence.market_intelligence_engine import (
        apply_deal_quality_to_verdict,
        bundle_to_market_data,
        enrich_buy_decision,
    )

    du = data_used if isinstance(data_used, dict) else {}
    from services.adversarial.adversarial_preprocessor import get_pipeline_query

    query = get_pipeline_query(query, du)
    raw_model = str(parsed.get("model") or "").strip()
    year = parsed.get("year")
    ask_usd = parsed.get("ask_usd")

    identity = resolve_canonical_identity(
        query=query,
        data_used=du,
        explicit_model=raw_model,
        source_layer="dispatch",
    )
    model = identity.canonical_model or raw_model

    authority_rec = get_aircraft_authority_record(aircraft_model=model)
    if authority_rec is not None:
        model = authority_rec.canonical_name
        identity = CanonicalAircraftIdentity(
            canonical_model=model,
            aliases_used=identity.aliases_used,
            source_layers=identity.source_layers + ("akal",),
            confidence_score=min(100, identity.confidence_score + 5),
            resolved_from_query_tokens=identity.resolved_from_query_tokens,
        )

    auth_market = build_authoritative_market_context(
        year=year,
        model=model,
        ask_usd=ask_usd,
        db=db,
    )

    mi_bundle, market_data = enrich_buy_decision(
        db=db,
        model=model,
        year=year,
        ask_usd=ask_usd,
        auth_market=auth_market,
        data_used=du,
    )

    aircraft: Dict[str, Any] = {
        "model": model,
        "manufacturer_year": year,
        "year": year,
        "ask_price": ask_usd,
    }

    peer_hours: Optional[List[float]] = None
    if db is not None and model:
        try:
            from services.market_comparison import run_comparison

            mc = run_comparison(db=db, models=[model], region="Global", limit=40)
            if not mc.get("error"):
                hours: List[float] = []
                for r in mc.get("rows") or []:
                    if isinstance(r, dict):
                        ht = r.get("total_time") or r.get("airframe_total_time")
                        try:
                            if ht is not None:
                                hours.append(float(ht))
                        except (TypeError, ValueError):
                            pass
                if hours:
                    peer_hours = hours
        except Exception:
            pass

    verdict_payload = run_deal_killer_engine(
        aircraft=aircraft,
        market_data=market_data,
        buyer_context={"mission_profile": {}},
        peer_airframe_hours=peer_hours,
    )
    verdict_payload = apply_deal_quality_to_verdict(verdict_payload, mi_bundle.deal_quality)

    adv_meta = _resolve_adversarial_metadata(du)

    state = UnifiedBrokerState(
        identity=identity,
        market_snapshot=mi_bundle.snapshot,
        liquidity=mi_bundle.liquidity,
        market_band=mi_bundle.band,
        deal_quality=mi_bundle.deal_quality,
        market_bundle=mi_bundle,
        authority_market=auth_market,
        authority_record=authority_rec.to_dict() if authority_rec else None,
        dispatch_kind="buy_decision",
        year=year,
        ask_usd=ask_usd,
        verdict_payload=verdict_payload,
        market_data=market_data,
        adversarial=adv_meta,
    )

    report = check_pipeline_agreement(data_used=du, state=state)
    state.agreement_report = report
    state = _attach_temporal(state, db=db)
    state = inject_consistency(du, state, report)

    du["deal_killer"] = verdict_payload
    du["buy_decision_dispatch"] = {"model": model, "year": year, "ask_usd": ask_usd}
    du["aircraft_authority_market"] = auth_market
    if authority_rec is not None:
        du["aircraft_authority_record"] = authority_rec.to_dict()

    return state


def prepare_valuation_state(
    *,
    query: str,
    model: str,
    year: str,
    db: Any = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> UnifiedBrokerState:
    """Build or reuse unified valuation state (single market pass)."""
    from services.aircraft.aircraft_authority_service import build_authoritative_market_context
    from services.market_intelligence.market_intelligence_engine import analyze_market

    du = data_used if isinstance(data_used, dict) else {}
    from services.adversarial.adversarial_preprocessor import get_pipeline_query

    query = get_pipeline_query(query, du)

    identity = resolve_canonical_identity(
        query=query,
        data_used=du,
        explicit_model=model,
        source_layer="recovery",
    )
    canon = identity.canonical_model or model

    auth_market = build_authoritative_market_context(
        year=None,
        model=canon,
        ask_usd=None,
        db=db,
    )
    bundle = analyze_market(db, canon, auth_market=auth_market)

    state = UnifiedBrokerState(
        identity=identity,
        market_snapshot=bundle.snapshot,
        liquidity=bundle.liquidity,
        market_band=bundle.band,
        deal_quality=bundle.deal_quality,
        market_bundle=bundle,
        authority_market=auth_market,
        dispatch_kind="valuation",
        year=int(year) if str(year).isdigit() else None,
        adversarial=_resolve_adversarial_metadata(du),
    )
    report = check_pipeline_agreement(data_used=du, state=state)
    state.agreement_report = report
    state = _attach_temporal(state, db=db)
    return inject_consistency(du, state, report)


def prepare_comparison_consistency(
    *,
    query: str,
    compare_models: List[str],
    data_used: Optional[Dict[str, Any]] = None,
) -> PipelineAgreementReport:
    """Lock comparison models through canonical identity; stamp agreement metadata."""
    du = data_used if isinstance(data_used, dict) else {}
    id_a, id_b = resolve_comparison_identities(query, compare_models, du)

    canonical_models = [m for m in (id_a.canonical_model, id_b.canonical_model) if m]
    if canonical_models:
        du["comparison_v2"] = {
            **(du.get("comparison_v2") or {}),
            "status": (du.get("comparison_v2") or {}).get("status", "OK"),
            "models": canonical_models,
            "identity_locked": True,
        }
        du["comparison_identity"] = {
            "models": canonical_models,
            "confidence": [id_a.confidence_score, id_b.confidence_score],
        }

    state = UnifiedBrokerState(
        identity=id_a,
        dispatch_kind="comparison",
        comparison_identities=(id_a, id_b),
    )
    report = check_pipeline_agreement(data_used=du, state=state)
    if id_a.canonical_model and id_b.canonical_model:
        if id_a.canonical_model == id_b.canonical_model:
            report.add(
                AgreementFlag.MODEL_MISMATCH,
                "comparison pair resolved to identical canonical model",
            )
    du["pipeline_agreement"] = {
        "aligned": report.aligned,
        "flags": [f.value for f in report.flags],
        "details": list(report.details),
    }
    return report


def render_buy_decision_answer(state: UnifiedBrokerState) -> str:
    """Format buy decision from precomputed unified state only."""
    from services.market_intelligence.market_intelligence_engine import (
        format_buy_decision_market_sections,
        format_deal_assessment,
    )

    model = state.canonical_model
    year = state.year
    ask_usd = state.ask_usd
    verdict_payload = state.verdict_payload or {}
    market_data = state.market_data or {}
    bundle = state.market_bundle

    lines: List[str] = [f"Aircraft: {model}"]
    if year:
        lines.append(f"Year: {year}")

    lines.append("")
    lines.append("Market Reality:")
    if bundle is not None:
        lines.extend(format_buy_decision_market_sections(bundle, market_data))
    else:
        lines.append("- Market context unavailable.")

    broker = (verdict_payload.get("broker_comment") or "").strip()
    if broker:
        lines.append(f"- {broker}")

    for r in list(verdict_payload.get("key_reasons") or [])[:4]:
        if r:
            lines.append(f"- {r}")

    red = list(verdict_payload.get("red_flags") or [])[:6]
    if red:
        lines.append("")
        lines.append("Red Flags:")
        for f in red:
            lines.append(f"- {f}")

    if bundle is not None:
        lines.append("")
        lines.append("Deal Assessment:")
        lines.extend(format_deal_assessment(ask_usd, state.deal_quality))

    if state.temporal is not None:
        from services.temporal_market.temporal_market_intelligence import format_temporal_buy_sections

        lines.extend(format_temporal_buy_sections(state.temporal))

    lines.append("")
    lines.append("Verdict:")
    lines.append(str(verdict_payload.get("verdict") or "FAIR DEAL"))
    return "\n".join(lines)


def render_valuation_answer(state: UnifiedBrokerState, *, year_label: str) -> str:
    """Format valuation from unified state only."""
    from services.market_intelligence.market_band_builder import BandConfidence
    from services.market_intelligence.market_intelligence_engine import fmt_musd

    model = state.canonical_model
    bundle = state.market_bundle
    if bundle is None:
        return (
            f"Aircraft: {model}\n"
            f"Year: {year_label}\n\n"
            "Market Reality:\nInsufficient verified market comps.\n\n"
            "Verdict:\nINSUFFICIENT_DATA"
        )

    band = bundle.band
    liq = bundle.liquidity
    snap = bundle.snapshot

    lines = [f"Aircraft: {model}", f"Year: {year_label}", "", "Market Reality:"]

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
        if state.temporal is not None:
            from services.temporal_market.temporal_market_intelligence import (
                format_temporal_valuation_sections,
            )

            lines.extend(format_temporal_valuation_sections(state.temporal))
        return "\n".join(lines)

    if band.reason == "authority_catalog_band" and band.mid is not None:
        lines.append(f"Catalog band (authority): {fmt_musd(band.low)}–{fmt_musd(band.high)}")
        lines.append(f"Median: {fmt_musd(band.mid)}")
        lines.append(f"Liquidity: {liq.band.value} (score {liq.score}/100)")
        lines.append("Confidence: MODERATE (catalog authority — not live listing median)")
        lines.append("")
        lines.append("Verdict:")
        lines.append("INSUFFICIENT_DATA")
        lines.append("(Listing depth below band threshold; catalog band shown for orientation only.)")
        if state.temporal is not None:
            from services.temporal_market.temporal_market_intelligence import (
                format_temporal_valuation_sections,
            )

            lines.extend(format_temporal_valuation_sections(state.temporal))
        return "\n".join(lines)

    lines.append("Insufficient market band: too few listings or stale market data.")
    lines.append(f"Liquidity: {liq.band.value} (score {liq.score}/100)")
    lines.append("")
    lines.append("Verdict:")
    lines.append("INSUFFICIENT_DATA")
    if state.temporal is not None:
        from services.temporal_market.temporal_market_intelligence import format_temporal_valuation_sections

        lines.extend(format_temporal_valuation_sections(state.temporal))
    return "\n".join(lines)
