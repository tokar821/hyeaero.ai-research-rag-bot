"""
Assemble **Aviation engines** context for the LLM (mission distance, feasibility, recommendations).

Framed for a professional broker / mission advisor — not system diagnostics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from rag.aviation_engines.capabilities import (
    filter_by_mission_pax_budget,
    find_catalog_matches,
    mission_possible_for_row,
    typical_market_price_usd,
    typical_passengers,
)
from rag.aviation_engines.geo import (
    PLANNING_GREAT_CIRCLE_NY_LA_NM,
    mission_endpoints_from_text,
    required_range_nm,
)
from rag.consultant_fine_intent import ConsultantFineIntent, ConsultantFineIntentResult


def _parse_budget_usd(text: str, entity_budget: Any) -> Optional[float]:
    if entity_budget is not None:
        try:
            v = float(entity_budget)
            if v > 0:
                if v < 500:
                    return v * 1_000_000
                return v
        except (TypeError, ValueError):
            pass
    t = (text or "").lower()
    m = re.search(
        r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(mm|m\b|million|mil)\b",
        t,
    )
    if m:
        try:
            num = float(m.group(1).replace(",", ""))
            return num * 1_000_000
        except ValueError:
            pass
    m2 = re.search(r"budget\s*(?:of|is)?\s*\$?\s*(\d{1,3}(?:,\d{3})+|\d+)", t)
    if m2:
        try:
            return float(m2.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _parse_passengers(text: str, ent_pax: Any) -> Optional[int]:
    if ent_pax is not None:
        try:
            p = int(ent_pax)
            if p > 0:
                return p
        except (TypeError, ValueError):
            pass
    m = re.search(r"\b(\d{1,2})\s*(pax|passengers?|seats?)\b", (text or "").lower())
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _model_hints_from_entities(entities: Dict[str, Any], query: str) -> List[str]:
    hints: List[str] = []
    m = entities.get("models")
    if isinstance(m, list):
        hints.extend(str(x).strip() for x in m if x)
    for rx in (
        r"\b(citation|challenger|falcon|gulfstream|global|phenom|learjet|hawker|praetor|legacy|sovereign)\s+[\w\+\d\-]+",
        r"\b([A-Z][a-z]+\s+[0-9]{3,4}[A-Z\+]*)\b",
    ):
        for mm in re.finditer(rx, query or "", re.I):
            hints.append(mm.group(0).strip())
    return list(dict.fromkeys(h for h in hints if h))


def build_aviation_engines_block(
    fine: ConsultantFineIntentResult,
    query: str,
) -> str:
    """Structured block for context assembly; instructs LLM to cite as typical performance data."""
    fi = fine.intent
    ent = fine.entities if isinstance(fine.entities, dict) else {}
    q = (query or "").strip()
    lines: List[str] = []

    icaos_entity = ent.get("icaos")
    icaos_list: Optional[List[str]] = list(icaos_entity) if isinstance(icaos_entity, list) else None

    endpoint = mission_endpoints_from_text(q, icaos_list)
    mission_nm: Optional[float] = None
    if endpoint:
        c0, c1, mission_nm = endpoint[0], endpoint[1], endpoint[2]
        low_hi_pr = (mission_nm * 1.15, mission_nm * 1.20)
        ny_la_note = ""
        pair = {c0.upper(), c1.upper()}
        if pair == {"KTEB", "KLAX"}:
            ny_la_note = (
                f" For New York (KTEB) to Los Angeles (KLAX), brokers commonly cite about {PLANNING_GREAT_CIRCLE_NY_LA_NM:,} nm "
                f"great-circle still air (spheric math here ≈ {mission_nm:.0f} nm — use the industry round figure in client copy)."
            )
        lines.append(
            "[CONSULTANT OUTPUT STRUCTURE — HyeAero.AI; recommendations & mission-fit buys]\n"
            "1) Mission analysis — what the client is trying to do (city pair, nonstop intent, pax).\n"
            "2) Distance — great-circle nm, still air; be numerically credible."
            f"{ny_la_note}\n"
            "3) Required practical range — use **mission_distance × 1.15 to 1.20** as the planning band for reserves and routing "
            f"(for this segment: roughly {low_hi_pr[0]:.0f}–{low_hi_pr[1]:.0f} nm from the ×1.15–1.20 rule; e.g. New York to Los "
            "Angeles ≈2,145 nm great-circle → operators often speak in terms of ~2,400–2,600 nm of practical capability with 8 pax and reserves).\n"
            "4) Recommended aircraft — 3–5 models; only types that can **nonstop** the mission per catalog filter (range ≥ mission×1.15); "
            "each: Aircraft name, Range, Passengers, Reason it fits.\n"
            "5) Close by offering **further analysis** (alternates, budgeting, or a deeper comparison) as HyeAero.AI.\n"
            "Avoid: internal dataset, our database, records not found. Prefer: "
            "\"I don't currently see a market listing for that aircraft\" or \"that type appears rarely on the secondary market.\"\n"
            f"[AVIATION REASONING — segment {c0}–{c1}: ≈ {mission_nm:.0f} nm GC; practical band ≈ {low_hi_pr[0]:.0f}–{low_hi_pr[1]:.0f} nm]"
        )

    req_nm: Optional[float] = required_range_nm(mission_nm) if mission_nm is not None else None
    synthetic_recommendation_route = False
    if fi == ConsultantFineIntent.AIRCRAFT_RECOMMENDATION and req_nm is None:
        synthetic_recommendation_route = True
        req_nm = required_range_nm(2200.0)
        lines.append(
            "[CONSULTANT OUTPUT ORDER — still open with **Mission** (ask/clarify city pair if needed), then distance/range assumptions, then recommendations.]\n"
            "[AVIATION REASONING — No city pair parsed; using a **transcon-class** reference mission (~2200 nm great-circle) "
            "with ×1.15 margin for catalog filtering—confirm the actual city pair with the client.]"
        )
    if req_nm is not None:
        _lbl = (
            "reference mission × 1.15 (placeholder route)"
            if synthetic_recommendation_route
            else "mission × 1.15"
        )
        lines.append(
            f"- Rule-of-thumb **required usable range** ({_lbl}): ≈ {req_nm:.0f} nm — compare to "
            "**realistic** OEM/book figures with reserves and payload, not brochure max."
        )

    if fi in (ConsultantFineIntent.AVIATION_MISSION, ConsultantFineIntent.AIRCRAFT_SPECS):
        models = _model_hints_from_entities(ent, q)
        rows = find_catalog_matches(models)
        if rows and mission_nm is not None:
            lines.append("[Range feasibility — based on typical aircraft performance data in broker reference table]")
            for r in rows[:4]:
                mp = mission_possible_for_row(mission_nm, r)
                mdl = r.get("aircraft_model")
                mx = r.get("max_range_nm")
                status = "likely feasible on paper" if mp else "likely marginal or requires fuel stop / larger type"
                lines.append(f"  · {mdl}: catalog max-range reference ≈ {mx} nm → {status} vs ≈ {req_nm:.0f} nm required.")
        elif rows and mission_nm is None:
            lines.append(
                "[Aircraft reference — typical performance data]\n"
                + "\n".join(
                    f"  · {r.get('aircraft_model')}: class {r.get('class')}, range ≈ {r.get('max_range_nm')} nm, "
                    f"pax ≤ {typical_passengers(r)}, cruise ≈ {r.get('cruise_speed_ktas')} KTAS."
                    for r in rows[:3]
                )
            )

    if fi == ConsultantFineIntent.AIRCRAFT_RECOMMENDATION and req_nm is not None:
        pax = _parse_passengers(q, ent.get("passengers"))
        bud = _parse_budget_usd(q, ent.get("budget_usd"))
        rec = filter_by_mission_pax_budget(req_nm, pax, bud, limit=5)
        _bud_note = (
            f", typical_market_price ≤ 85% of stated budget (${bud * 0.85:,.0f} cap on ${bud:,.0f} — no aircraft above that band)"
            if bud
            else ""
        )
        lines.append(
            "[Recommendation engine — ranked: closest range match, then closest passenger fit, then lowest acquisition cost]\n"
            f"- Hard filter: max_range_nm ≥ {req_nm:.0f} nm"
            + (f", typical_passengers ≥ {pax}" if pax else "")
            + _bud_note
            + "\n- Present 3–5 types in client answer under **Recommended aircraft:** using the four-line block per type below."
        )
        if rec:
            for i, r in enumerate(rec, 1):
                why = (r.get("typical_mission") or r.get("class") or "").strip()
                price = typical_market_price_usd(r)
                tp = typical_passengers(r)
                lines.append(
                    f"  {i}. {r.get('aircraft_model')}\n"
                    f"     Range: ~{r.get('max_range_nm')} nm\n"
                    f"     Passengers: up to {tp}\n"
                    f"     Typical acquisition band: ≈ ${price:,.0f}\n"
                    f"     Reason it fits: {why or '—'}"
                )
        else:
            lines.append(
                "  **No types in the reference catalog matched all constraints.** Tell the client plainly — in broker "
                "language — that this mission likely needs either a **higher budget**, **fewer passengers**, or a "
                "**fuel / tech stop** (or a larger class of aircraft). Do not blame 'records' or 'availability'; frame "
                "as operating economics and mission physics."
            )

    if fi == ConsultantFineIntent.AIRCRAFT_COMPARISON:
        models = _model_hints_from_entities(ent, q)
        rows = find_catalog_matches(models[:6] or models)
        if len(rows) >= 2:
            lines.append(
                "[Comparison aide — use these section headings for the client: **Range**; **Passengers**; **Cruise speed**; "
                "**Cabin characteristics**; **Mission strengths**]"
            )
            for r in rows[:4]:
                lines.append(
                    f"  · {r.get('aircraft_model')}: range ≈ {r.get('max_range_nm')} nm, pax {typical_passengers(r)}, "
                    f"speed ≈ {r.get('cruise_speed_ktas')} KTAS; cabin: {r.get('cabin') or '—'}; "
                    f"typical mission: {r.get('typical_mission') or '—'}."
                )

    if not lines:
        return ""

    lines.append(
        "Voice: HyeAero.AI — aircraft broker, mission advisor, aviation intelligence analyst; confident industry reasoning. "
        "Avoid weak phrasing: 'market availability', 'records indicate'. "
        "Based on typical operational performance for this aircraft or class when generalizing. "
        "Never imply a type above the user's budget envelope (catalog caps typical ask at 85% of stated budget). "
        "No charter booking or promotional URLs. End with a short offer of further analysis when natural."
    )
    return "\n".join(lines)
