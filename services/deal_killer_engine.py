"""
HyeAero.AI — **Deal Killer Engine**: blunt brokerage verdict on a specific aircraft + market slice.

Deterministic rules only — no LLM. Output is advisory for the consultant layer and ``data_used``.
"""

from __future__ import annotations

import logging
import os
import re
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VERDICT_GOOD_DEAL = "GOOD DEAL"
VERDICT_FAIR_DEAL = "FAIR DEAL"
VERDICT_OVERPRICED = "OVERPRICED"
VERDICT_HIGH_RISK = "HIGH RISK"
VERDICT_DO_NOT_BUY = "DO NOT BUY"


def consultant_deal_killer_enabled() -> bool:
    return (os.getenv("CONSULTANT_DEAL_KILLER") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _parse_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip().replace(",", "")
    s = re.sub(r"[^\d.\-]", "", s)
    if not s or s in (".", "-", "-."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_year(v: Any) -> Optional[int]:
    f = _parse_float(v)
    if f is None:
        return None
    y = int(f)
    if 1950 <= y <= 2035:
        return y
    return None


def _norm_programs(programs: Any) -> List[str]:
    if programs is None:
        return []
    if isinstance(programs, str) and programs.strip():
        return [programs.strip()]
    if isinstance(programs, list):
        return [str(p).strip() for p in programs if str(p).strip()]
    return []


def _blob_programs(aircraft: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in (
        "engines",
        "engine_program",
        "airframe_program",
        "avionics_program",
        "programs_enrolled",
        "programs",
    ):
        v = aircraft.get(k)
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v:
            parts.append(str(v))
    return " ".join(parts).lower()


def _program_flags(aircraft: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    """engine_program_like, apu_like, tracking_like"""
    blob = _blob_programs(aircraft)
    mt = str(aircraft.get("maintenance_tracking") or "").lower()
    blob_all = f"{blob} {mt}"
    eng = bool(
        re.search(r"\b(msp|jssi|tip\s*toe|tip-toe|power\s*by\s*the\s*hour|pbh|eap|esp)\b", blob_all, re.I)
    )
    apu = bool(re.search(r"\b(apu\s*on\s*program|apu\s*coverage|apu\s*msp|apu\s*jssi)\b", blob_all, re.I))
    trk = bool(re.search(r"\b(camp|cescom|flightdocs|corridor|traxxall|veryon)\b", blob_all, re.I))
    return eng, apu, trk


def _price_position(ask: Optional[float], low: Optional[float], high: Optional[float]) -> Tuple[str, List[str]]:
    """Returns (label, red_flags_for_price)."""
    flags: List[str] = []
    if ask is None or ask <= 0:
        return "unknown", ["No reliable ask price on file — price verdict is blind."]
    if low is not None and high is not None and low > 0 and high >= low:
        if ask > high * 1.15:
            flags.append(f"Asking price is more than 15% above synced comp high (~{high:,.0f} USD).")
            return "over_range_high", flags
        if ask < low * 0.80:
            flags.append(
                f"Asking price is more than 20% below comp low (~{low:,.0f} USD) — verify equipment, pedigree, and log gaps."
            )
            return "suspiciously_low", flags
        return "in_range", flags
    if high is not None and high > 0 and ask > high * 1.15:
        flags.append("Asking price sits materially above the high ask in the current comp slice.")
        return "over_range_high", flags
    return "unknown", flags


def _utilization_flags(
    aircraft: Dict[str, Any],
    peer_hours: Optional[List[float]],
) -> Tuple[List[str], float]:
    """Returns (red_flags, condition_score 0–1)."""
    flags: List[str] = []
    tt = _parse_float(aircraft.get("total_time"))
    if tt is None:
        tt = _parse_float(aircraft.get("airframe_total_time"))
    year = _parse_year(aircraft.get("year"))
    score = 0.72
    if tt is not None and peer_hours and len(peer_hours) >= 3:
        med = float(median(peer_hours))
        if med > 0 and tt > med * 1.45:
            flags.append(f"Airframe time (~{tt:,.0f} hrs) is very high vs this comp set’s typical — check cycles, CPCP, and residual.")
            score -= 0.22
        elif tt < 400 and year is not None and year <= 2005:
            flags.append("Very low time on an older airframe — confirm usage pattern, storage, and calendar-out maintenance.")
            score -= 0.12
    elif tt is not None:
        if tt >= 8500:
            flags.append(f"Total time ~{tt:,.0f} hrs is elevated for most buyers — overhaul proximity matters.")
            score -= 0.18
        elif tt is not None and tt < 250:
            flags.append("Ultra-low total time — validate log continuity and seller motivation.")
            score -= 0.08
    if year is not None and year < 1995:
        blob = _blob_programs(aircraft)
        if not re.search(r"\b(upgrade|winglets|proline|g5000|g6000|vision|phase\s*\d)\b", blob, re.I):
            flags.append("Older airframe with no obvious major-upgrade cues in the data provided — avionics and corrosion plan matter.")
            score -= 0.12
    return flags, max(0.15, min(1.0, score))


def _liquidity_score(market_data: Dict[str, Any], comp_rows_n: int) -> Tuple[float, List[str]]:
    flags: List[str] = []
    liq = str(market_data.get("liquidity") or "").lower()
    dem = str(market_data.get("demand_level") or "").lower()
    s = 0.65
    if comp_rows_n >= 10:
        s += 0.15
    elif comp_rows_n <= 2:
        s -= 0.25
        flags.append("Thin listing comps in this slice — resale pricing is harder to defend.")
    if any(x in liq for x in ("low", "weak", "thin")):
        s -= 0.2
        flags.append("Liquidity read: weak — harder exit if mission or financing shifts.")
    if any(x in dem for x in ("weak", "soft", "buyer's")):
        s -= 0.1
    if any(x in dem for x in ("strong", "firm", "seller's")):
        s += 0.08
    return max(0.1, min(1.0, s)), flags


def _mission_fit_score(aircraft_model: str, buyer_context: Dict[str, Any], mission_dict: Dict[str, Any]) -> Tuple[float, List[str]]:
    try:
        from services.aircraft_decision_engine import _CLASS_BAND_NM, infer_jet_class
    except Exception:
        return 0.7, []

    jet = infer_jet_class(aircraft_model or "")
    band = _CLASS_BAND_NM.get(jet, 3200.0)
    leg = mission_dict.get("longest_leg_nm")
    if leg is None:
        leg = _parse_float(buyer_context.get("longest_leg_nm"))
    flags: List[str] = []
    if leg and band and float(leg) > band * 1.08:
        flags.append(
            f"Mission mismatch: stated ~{float(leg):,.0f} nm leg exceeds typical practical band for this class — do not buy this class for that mission."
        )
        return 0.35, flags
    if leg and band and float(leg) > band * 0.98:
        flags.append(f"Leg ~{float(leg):,.0f} nm is tight for this class — expect operational compromises or fuel stops.")
        return 0.62, flags
    return 0.88, flags


def _price_score_from_position(pos: str, ask: Optional[float], low: Optional[float], high: Optional[float]) -> float:
    if ask is None or ask <= 0:
        return 0.45
    if pos == "over_range_high":
        return 0.22
    if pos == "suspiciously_low":
        return 0.55
    if pos == "in_range" and low and high and high >= low:
        mid = (low + high) / 2.0
        if mid > 0:
            # below mid is better for buyer
            r = ask / mid
            if r <= 0.92:
                return 0.9
            if r <= 1.05:
                return 0.78
            return 0.58
    return 0.62


def run_deal_killer_engine(
    *,
    aircraft: Dict[str, Any],
    market_data: Dict[str, Any],
    buyer_context: Dict[str, Any],
    peer_airframe_hours: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Produce verdict + scores. All monetary values assumed **USD** as stored in Hye Aero listing comps.
    """
    ask = _parse_float(aircraft.get("ask_price"))
    low = _parse_float(market_data.get("price_range_low"))
    high = _parse_float(market_data.get("price_range_high"))
    avg = _parse_float(market_data.get("avg_price"))

    pos, price_flags = _price_position(ask, low, high)
    price_score = _price_score_from_position(pos, ask, low, high)

    util_flags, condition_score = _utilization_flags(aircraft, peer_airframe_hours)

    eng_ok, apu_ok, trk_ok = _program_flags(aircraft)
    prog_flags: List[str] = []
    if not eng_ok:
        prog_flags.append("No clear engine hourly program (MSP/JSSI-class) on file — heavy cash maintenance exposure if true.")
    if not apu_ok and infer_jet_heavy(aircraft):
        prog_flags.append("No APU program cues — budget hot-section/APU reserves explicitly.")
    if not trk_ok:
        prog_flags.append("No recognized digital maintenance tracking (CAMP/CESCOM-class) called out — records discipline is a diligence item.")

    comp_n = int(market_data.get("comp_row_count") or 0)
    liq_score, liq_flags = _liquidity_score(market_data, comp_n)

    mission_dict = buyer_context.get("mission_profile") if isinstance(buyer_context.get("mission_profile"), dict) else {}
    mm = str(aircraft.get("model") or "").strip()
    if not mm:
        mm = " ".join(
            x for x in (str(aircraft.get("manufacturer") or "").strip(), str(aircraft.get("model") or "").strip()) if x
        ).strip()
    mission_score, mission_flags = _mission_fit_score(mm, buyer_context, mission_dict)

    red_flags = list(price_flags) + util_flags + prog_flags + liq_flags + mission_flags

    _blob_u = f"{mm} {str(aircraft.get('manufacturer') or '')}".lower()
    uncommon = bool(
        re.search(r"\b(very\s*light\s*jet|experimental|amateur\s*built|warbird)\b", _blob_u)
    )
    if uncommon:
        red_flags.append("Uncommon category — parts and specialist support risk is real.")

    # Hidden cost / aging engines heuristic
    y = _parse_year(aircraft.get("year"))
    if y and y < 2000 and not eng_ok:
        red_flags.append("Aging airframe without enrolled engine program on file — overhaul reserve risk is elevated.")

    risk_flag_n = len([f for f in red_flags if f])

    # Final verdict (spec order)
    verdict = VERDICT_FAIR_DEAL
    confidence = 0.62
    if mission_score < 0.60:
        verdict = VERDICT_DO_NOT_BUY
        confidence = 0.78
    elif pos == "over_range_high":
        verdict = VERDICT_OVERPRICED
        confidence = 0.74 if ask and high else 0.6
    elif risk_flag_n >= 4 or (price_score < 0.35 and condition_score < 0.45):
        verdict = VERDICT_HIGH_RISK
        confidence = 0.7
    elif price_score >= 0.82 and condition_score >= 0.70 and liq_score >= 0.72 and mission_score >= 0.75 and eng_ok and trk_ok:
        verdict = VERDICT_GOOD_DEAL
        confidence = 0.68 if comp_n >= 5 else 0.55
    elif mission_score < 0.68 and mission_flags:
        verdict = VERDICT_HIGH_RISK
        confidence = 0.62

    key_reasons = _key_reasons(verdict, pos, mission_score, eng_ok, trk_ok, ask, low, high, avg)
    broker_comment = _broker_comment(verdict, pos, mission_score, ask, high)

    return {
        "verdict": verdict,
        "confidence": round(min(0.95, max(0.35, confidence)), 2),
        "key_reasons": key_reasons,
        "red_flags": red_flags[:12],
        "broker_comment": broker_comment,
        "scores": {
            "price_score": round(price_score, 3),
            "condition_score": round(condition_score, 3),
            "liquidity_score": round(liq_score, 3),
            "mission_fit_score": round(mission_score, 3),
        },
        "price_position": pos,
        "inputs_echo": {
            "ask_price_usd": ask,
            "market_low_usd": low,
            "market_high_usd": high,
            "market_avg_usd": avg,
        },
    }


def infer_jet_heavy(aircraft: Dict[str, Any]) -> bool:
    """APU/program warnings matter more on turbine jets."""
    try:
        from services.aircraft_decision_engine import infer_jet_class

        m = str(aircraft.get("model") or aircraft.get("manufacturer") or "")
        return infer_jet_class(m) != "light"
    except Exception:
        return True


def _key_reasons(
    verdict: str,
    pos: str,
    mission_score: float,
    eng_ok: bool,
    trk_ok: bool,
    ask: Optional[float],
    low: Optional[float],
    high: Optional[float],
    avg: Optional[float],
) -> List[str]:
    out: List[str] = []
    if verdict == VERDICT_DO_NOT_BUY:
        out.append("Aircraft class does not credibly match the stated mission — walk away or change class.")
    if pos == "over_range_high":
        out.append("Price sits above the current comp ceiling without justification in this data.")
    if pos == "suspiciously_low":
        out.append("Price is far below the comp floor — assume missing damage, run-out programs, or bad pedigree until proven otherwise.")
    if mission_score >= 0.75:
        out.append("Mission fit against this class is acceptable on paper — still verify wind, reserves, and alternates.")
    if eng_ok and trk_ok:
        out.append("Programs/tracking cues present — lowers some cash-spike risk vs naked enrollment.")
    elif not eng_ok:
        out.append("Engine program not evidenced — reserve cash for hot sections/overhauls in underwriting.")
    if ask and avg:
        out.append(f"Ask {ask:,.0f} USD vs comp average ~{avg:,.0f} USD (rough positioning only).")
    if low and high:
        out.append(f"Comp band in this slice: ~{low:,.0f}–{high:,.0f} USD.")
    return out[:6] if out else ["Limited structured inputs — verdict is directional only."]


def _broker_comment(verdict: str, pos: str, mission_score: float, ask: Optional[float], high: Optional[float]) -> str:
    if verdict == VERDICT_DO_NOT_BUY:
        return "Do not buy this aircraft for the mission you described — you will buy a problem, not range."
    if verdict == VERDICT_OVERPRICED:
        return "This is overpriced for what the current comp slice shows — negotiate hard or pass."
    if verdict == VERDICT_HIGH_RISK:
        return "Too many open risks on price, time, programs, or liquidity — not a clean check to write today."
    if verdict == VERDICT_GOOD_DEAL:
        return "On paper this is a strong buy candidate — still demand pre-buy and log review before you wire anything."
    if mission_score < 0.68:
        return "Fair at best: mission fit is shaky — only proceed if you change how you fly it."
    if pos == "suspiciously_low":
        return "Cheap is not free — assume something expensive is missing from the story until diligence proves otherwise."
    return "Fair deal territory — nothing heroic here; price and risk look ordinary for the evidence provided."


def _merge_aircraft_from_phly_and_listing(
    phly_row: Optional[Dict[str, Any]],
    listing_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    pr = phly_row if isinstance(phly_row, dict) else {}
    lr = listing_row if isinstance(listing_row, dict) else {}

    def _first(*keys: str) -> Any:
        for d in (lr, pr):
            for k in keys:
                if d.get(k) not in (None, ""):
                    return d.get(k)
        return None

    out["manufacturer"] = _first("manufacturer")
    out["model"] = _first("model")
    out["year"] = _first("manufacturer_year", "year")
    out["ask_price"] = _first("ask_price")
    out["total_time"] = _first("airframe_total_time", "total_time")
    out["location"] = _first("location", "based_at")
    out["listing_status"] = _first("listing_status")

    prog_parts = _norm_programs(pr.get("programs")) + _norm_programs(pr.get("airframe_program"))
    out["programs"] = prog_parts or _norm_programs(pr.get("engine_program"))
    out["engines"] = str(pr.get("engines") or pr.get("engine_model") or "").strip() or None
    out["maintenance_tracking"] = str(pr.get("maintenance_tracking") or pr.get("camp") or "").strip() or None
    return out


def _market_stats_from_rows(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[float]]:
    asks: List[float] = []
    hours: List[float] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        a = _parse_float(r.get("ask_price"))
        if a and a > 0:
            asks.append(a)
        h = _parse_float(r.get("airframe_total_time"))
        if h is not None and h >= 0:
            hours.append(h)
    if not asks:
        return (
            {
                "avg_price": None,
                "price_range_low": None,
                "price_range_high": None,
                "liquidity": "unknown",
                "demand_level": "unknown",
                "comp_row_count": len(rows or []),
            },
            hours,
        )
    asks_s = sorted(asks)
    low = asks_s[0]
    high = asks_s[-1]
    avg = sum(asks) / len(asks)
    n = len(asks)
    if n >= 12:
        liq, dem = "strong", "firm"
    elif n >= 5:
        liq, dem = "moderate", "balanced"
    else:
        liq, dem = "thin", "buyer's market"
    return (
        {
            "avg_price": avg,
            "price_range_low": low,
            "price_range_high": high,
            "liquidity": liq,
            "demand_level": dem,
            "comp_row_count": len(rows or []),
        },
        hours,
    )


def run_deal_killer_from_consultant_context(
    *,
    phly_rows: Optional[List[Dict[str, Any]]],
    primary_listing: Optional[Dict[str, Any]],
    query: str,
    buyer_psychology: Optional[Dict[str, Any]],
    db: Any,
) -> Optional[Dict[str, Any]]:
    """
    Build inputs from consultant bundle and run the engine. Returns ``None`` if there is nothing to score.
    """
    ph0 = (phly_rows or [])[0] if phly_rows else None
    if not ph0 and not primary_listing:
        return None
    ac = _merge_aircraft_from_phly_and_listing(ph0, primary_listing)
    if not str(ac.get("model") or "").strip() and not str(ac.get("manufacturer") or "").strip():
        return None

    rows: List[Dict[str, Any]] = []
    mdl_token = str(ac.get("model") or "").strip()
    if not mdl_token and ph0:
        mdl_token = str(ph0.get("model") or "").strip()
    if not mdl_token and isinstance(primary_listing, dict):
        mdl_token = str(primary_listing.get("model") or "").strip()
    if not mdl_token:
        return None
    if db is not None and mdl_token:
        try:
            from services.market_comparison import run_comparison

            mc = run_comparison(db=db, models=[mdl_token], region="Global", limit=40)
            if not mc.get("error"):
                rows = list(mc.get("rows") or [])
        except Exception as e:
            logger.debug("deal_killer market_comparison skipped: %s", e)

    if ac.get("ask_price") is None and ac.get("total_time") is None and not rows:
        return None

    market_data, peer_hours = _market_stats_from_rows(rows)

    try:
        from services.aircraft_decision_engine import extract_mission_profile
    except Exception:

        def extract_mission_profile(q: str) -> Dict[str, Any]:  # type: ignore[misc,redefinition]
            return {}

    mission_profile = extract_mission_profile(query or "")
    budget = None
    if buyer_psychology and isinstance(buyer_psychology, dict):
        budget = buyer_psychology.get("parsed_budget_millions_usd")
    buyer_context: Dict[str, Any] = {
        "mission": str(mission_profile.get("typical_routes_hint") or ""),
        "budget_millions_usd": budget,
        "usage": str(mission_profile.get("usage") or ""),
        "psychology_type": str((buyer_psychology or {}).get("buyer_type") or ""),
        "mission_profile": mission_profile,
        "longest_leg_nm": mission_profile.get("longest_leg_nm"),
    }

    return run_deal_killer_engine(
        aircraft=ac,
        market_data=market_data,
        buyer_context=buyer_context,
        peer_airframe_hours=peer_hours or None,
    )


def format_deal_killer_for_system_prompt(payload: Dict[str, Any], *, max_chars: int = 1200) -> str:
    if not isinstance(payload, dict) or not payload.get("verdict"):
        return ""
    lines = [
        "**[DEAL KILLER — advisory; do not quote this header to the user]**",
        f"- **Verdict:** {payload.get('verdict')} (confidence {payload.get('confidence')})",
        f"- **Broker line:** {payload.get('broker_comment')}",
        f"- **Scores:** {payload.get('scores')}",
        f"- **Key reasons:** " + "; ".join(payload.get("key_reasons") or [])[:500],
        f"- **Red flags:** " + "; ".join((payload.get("red_flags") or [])[:6])[:500],
    ]
    out = "\n".join(lines)
    if len(out) > max_chars:
        return out[: max_chars - 3] + "..."
    return out
