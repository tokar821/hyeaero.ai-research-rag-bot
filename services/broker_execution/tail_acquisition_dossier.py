"""
Tail acquisition due-diligence — broker dossier (not registry card).

Answers engine program, risks, and full-profile questions with Phly + FAA facts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from services.broker_execution.tail_depth_mode import TailDepthMode, classify_tail_depth_mode


def _reg(query: str, data_used: dict) -> str:
    reg = str(data_used.get("tail_registration") or "").strip().upper()
    if reg:
        return reg
    _, r = classify_tail_depth_mode(query)
    return (r or "").strip().upper()


def _phly_row(data_used: dict, reg: str) -> Dict[str, Any]:
    rows = data_used.get("phlydata_rows") or data_used.get("phly_rows") or []
    if not isinstance(rows, list):
        return {}
    for candidate in rows:
        if not isinstance(candidate, dict):
            continue
        if (candidate.get("registration_number") or "").strip().upper() == reg:
            return candidate
    return rows[0] if rows and isinstance(rows[0], dict) else {}


def _fmt_price(raw: Any) -> Optional[str]:
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
        if val >= 1_000_000:
            return f"${val / 1_000_000:.2f}M"
        return f"${val:,.0f}"
    except (TypeError, ValueError):
        s = str(raw).strip()
        return s if s else None


def _hours_context(hours: Any, year: Any) -> str:
    if hours in (None, ""):
        return "Airframe hours not in synced data — verify logbooks."
    try:
        h = float(hours)
    except (TypeError, ValueError):
        return f"Airframe time: {hours} (verify vs fleet and year)."
    y = int(year) if year not in (None, "") else None
    if y and y >= 2000:
        age = max(1, 2026 - y)
        avg = h / age
        if avg > 550:
            return f"{h:,.0f} hours — high utilization for a {y} airframe; scrutinize cycles and upcoming inspections."
        if avg < 250:
            return f"{h:,.0f} hours — relatively low time for a {y} airframe; confirm program continuity."
    return f"{h:,.0f} hours — compare to same-model fleet averages and upcoming calendar inspections."


def build_tail_acquisition_dossier_block(query: str, data_used: Optional[Dict[str, Any]] = None) -> str:
    """Structured due-diligence block for fact pack / LLM (not registry-only)."""
    du = data_used if isinstance(data_used, dict) else {}
    depth, _ = classify_tail_depth_mode(query)
    depth_name = str(du.get("tail_depth_mode") or depth.value or "").strip().lower()
    if depth not in (
        TailDepthMode.ENGINE_PROGRAM,
        TailDepthMode.ACQUISITION,
        TailDepthMode.ACQUISITION_RISKS,
        TailDepthMode.DETAIL,
        TailDepthMode.COMPARISON,
        TailDepthMode.CONTEXT,
    ) and depth_name not in (
        "engine_program",
        "acquisition",
        "acquisition_risks",
        "detail",
        "comparison",
        "context",
    ):
        return ""

    try:
        from services.broker_execution.tail_fact_loader import ensure_tail_facts_for_query

        ensure_tail_facts_for_query(query, du)
    except Exception:
        pass

    reg = _reg(query, du)
    if not reg:
        return ""

    phly = _phly_row(du, reg)
    lines: List[str] = [
        f"[TAIL ACQUISITION DOSSIER — {reg}]",
        "Broker due-diligence facts (narrate as acquisition consultant; do not lead with registry boilerplate).",
        "",
    ]

    mfr = (phly.get("manufacturer") or "").strip()
    mdl = (phly.get("model") or "").strip()
    mm = " ".join(x for x in (mfr, mdl) if x).strip()
    if mm:
        lines.append(f"Aircraft: {mm}")
    year = phly.get("year") or phly.get("year_mfr")
    if year:
        lines.append(f"Year: {year}")
    hours = phly.get("airframe_total_time") or phly.get("total_time")
    lines.append(f"Airframe time: {_hours_context(hours, year)}")

    eng = (phly.get("engine_program") or "").strip()
    apu = (phly.get("apu_program") or "").strip()
    maint = (phly.get("maintenance_tracking_program") or phly.get("maintenance_program") or "").strip()
    defer = (phly.get("engine_program_deferment") or "").strip()
    if eng:
        lines.append(f"Engine program: {eng}" + (" — verify enrollment dates and transferability." if "msp" in eng.lower() else ""))
    else:
        lines.append("Engine program: not listed — major pre-buy risk; confirm enrollment before LOI.")
    if apu:
        lines.append(f"APU program: {apu}")
    if maint:
        lines.append(f"Maintenance tracking: {maint}")
    if defer:
        lines.append(f"Engine deferment note: {defer}")

    damage = (phly.get("damage_history") or phly.get("incident_history") or "").strip()
    if damage:
        lines.append(f"Damage / incident (synced): {damage}")
    else:
        lines.append("Damage history: not flagged in synced data — still pull NTSB, FAA, and logbook entries.")

    owner = (phly.get("owner") or phly.get("registered_owner") or "").strip()
    if owner:
        lines.append(f"Ownership (listing): {owner} — trace chain and recent transfers.")

    status = (phly.get("aircraft_status") or "").strip()
    ask = _fmt_price(phly.get("ask_price"))
    if status:
        lines.append(f"Market status: {status}")
    if ask:
        lines.append(f"Asking price (internal): {ask}")

    loc = (phly.get("location") or phly.get("base_location") or "").strip()
    if loc:
        lines.append(f"Location: {loc}")

    cabin = (phly.get("cabin_configuration") or phly.get("passenger_capacity") or "").strip()
    if cabin:
        lines.append(f"Cabin / capacity: {cabin}")

    lines.extend(
        [
            "",
            "Acquisition risk checklist (prioritize in answer):",
            "• Utilization vs fleet — hours/cycles and upcoming inspections",
            "• Engine/APU program quality, deferments, and transferability at closing",
            "• Maintenance tracking gaps or back-to-back 135/91 churn",
            "• Damage, corrosion, or AD compliance not visible in snapshot data",
            "• Ownership chain, export, and listing pedigree",
            "• Market position — ask vs comps, days on market, motivation",
            "• Resale liquidity for this model/year/program state",
        ]
    )

    mr = du.get("market_reality")
    if isinstance(mr, dict) and mr:
        band = mr.get("band_mid_usd") or mr.get("mid_band_usd")
        if band:
            try:
                lines.append(f"Model market mid-band (indicative): ${float(band) / 1_000_000:.1f}M")
            except (TypeError, ValueError):
                pass

    body = "\n".join(lines).strip()
    return body if len(body) > 120 else ""


def render_engine_program_answer(query: str, data_used: Optional[Dict[str, Any]] = None) -> str:
    """Direct answer: Yes/No + program names (not a registry card)."""
    du = data_used if isinstance(data_used, dict) else {}
    reg = _reg(query, du)
    phly = _phly_row(du, reg)
    eng = (phly.get("engine_program") or "").strip()
    apu = (phly.get("apu_program") or "").strip()
    if eng:
        lead = f"Yes. Engine program: {eng}."
        if apu:
            lead += f" APU: {apu}."
        return lead.strip()
    if re.search(r"(?is)\b(?:enrolled|on\s+(?:an?\s+)?engine)\b", query or ""):
        return (
            f"No engine program is listed in synced data for {reg or 'this tail'}. "
            "Treat that as a pre-buy priority — confirm enrollment and transferability in the logs."
        )
    return ""


def render_acquisition_risks_answer(query: str, data_used: Optional[Dict[str, Any]] = None) -> str:
    """Client-facing acquisition risks (broker voice)."""
    du = data_used if isinstance(data_used, dict) else {}
    reg = _reg(query, du)
    phly = _phly_row(du, reg)
    if not reg and not phly:
        return ""

    mm = " ".join(
        x
        for x in (
            (phly.get("manufacturer") or "").strip(),
            (phly.get("model") or "").strip(),
        )
        if x
    ).strip()
    opener = f"On {reg}" + (f" ({mm})" if mm else "") + ", the biggest acquisition risks I'd pressure-test:"

    bullets: List[str] = []
    hours = phly.get("airframe_total_time") or phly.get("total_time")
    year = phly.get("year") or phly.get("year_mfr")
    bullets.append(f"• Utilization: {_hours_context(hours, year)}")

    eng = (phly.get("engine_program") or "").strip()
    if eng:
        bullets.append(f"• Engine program: {eng} — confirm enrollment dates, exclusions, and transfer at closing.")
    else:
        bullets.append("• Engine program: not listed — high risk; verify MSP/JSSI status before LOI.")

    damage = (phly.get("damage_history") or phly.get("incident_history") or "").strip()
    bullets.append(
        f"• Damage / airworthiness: {damage if damage else 'not flagged in sync — still pull NTSB and logbooks.'}"
    )

    ask = _fmt_price(phly.get("ask_price"))
    status = (phly.get("aircraft_status") or "").strip()
    if ask or status:
        bullets.append(
            f"• Market position: {status or 'status unclear'}{f' at {ask}' if ask else ''} — validate comps, DOM, and seller motivation."
        )
    else:
        bullets.append("• Market position: confirm ask, days on market, and whether the seller is motivated or testing the market.")

    bullets.append("• Resale: model/year/program state vs active inventory — don't overpay for a thin buyer pool.")

    return opener + "\n" + "\n".join(bullets[:6])


def render_tail_detail_answer(query: str, data_used: Optional[Dict[str, Any]] = None) -> str:
    """Full-profile client answer for 'tell me everything' tail queries."""
    du = data_used if isinstance(data_used, dict) else {}
    reg = _reg(query, du)
    phly = _phly_row(du, reg)
    if not reg:
        return ""

    sections: List[str] = []
    mm = " ".join(
        x
        for x in (
            (phly.get("manufacturer") or "").strip(),
            (phly.get("model") or "").strip(),
        )
        if x
    ).strip()
    sections.append(f"{reg}" + (f" — {mm}" if mm else ""))

    for label, key in (
        ("Year", "year"),
        ("Serial", "serial_number"),
        ("Owner", "owner"),
        ("Status", "aircraft_status"),
        ("Ask", "ask_price"),
        ("Location", "location"),
        ("Hours", "airframe_total_time"),
        ("Engine program", "engine_program"),
        ("APU program", "apu_program"),
        ("Maintenance", "maintenance_tracking_program"),
        ("Cabin", "cabin_configuration"),
        ("Damage", "damage_history"),
    ):
        val = phly.get(key) or phly.get("total_time" if key == "airframe_total_time" else "")
        if key == "ask_price":
            val = _fmt_price(val) or val
        if val not in (None, ""):
            sections.append(f"• {label}: {val}")

    if len(sections) <= 2:
        block = build_tail_acquisition_dossier_block(query, du)
        if block:
            for line in block.splitlines():
                if line.startswith("• ") or ":" in line[:40]:
                    sections.append(line if line.startswith("•") else f"• {line}")

    sections.append(
        "• Broker take: verify logbooks, program transferability, and live comps before LOI — snapshot data is not a pre-buy."
    )
    return "\n".join(sections[:18]).strip()


def resolve_query_with_active_tail(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Append locked tail from conversation memory when the user asks a short follow-up."""
    q = (query or "").strip()
    if not q:
        return q
    try:
        from rag.aviation_tail import primary_registration_from_query

        if primary_registration_from_query(q):
            return q
    except Exception:
        if re.search(r"\bN(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}\b", q.upper()):
            return q

    du = data_used if isinstance(data_used, dict) else {}
    tail = str(du.get("active_tail") or du.get("tail_registration") or "").strip().upper()
    if not tail and isinstance(du.get("intent_persistence"), dict):
        tail = str(du["intent_persistence"].get("active_tail") or "").strip().upper()
    if not tail:
        cont = du.get("continuity") or {}
        if isinstance(cont, dict):
            tail = str(cont.get("current_tail") or "").strip().upper()

    if not tail:
        return q

    if not re.search(
        r"(?is)\b(?:engine\s+program|apu|risks?|acquisition|damage|hours|"
        r"compare|versus|vs\.?|everything\s+about|enrolled|maintenance|"
        r"cabin|cockpit|photos?|gallery|for\s+sale|owner)\b",
        q,
    ):
        return q

    if tail.lower() in q.lower():
        return q
    return f"{q} {tail}".strip()


__all__ = [
    "build_tail_acquisition_dossier_block",
    "render_acquisition_risks_answer",
    "render_engine_program_answer",
    "render_tail_detail_answer",
    "resolve_query_with_active_tail",
]
