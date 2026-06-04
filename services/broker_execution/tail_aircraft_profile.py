"""
Merged tail aircraft profile — FAA + Phly listing + market for DETAIL / ACQUISITION turns.

Fact-pack context only; does not change registry SQL or ranking.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.broker_execution.tail_depth_mode import TailDepthMode, classify_tail_depth_mode


def _extract_registration(query: str, data_used: dict) -> Optional[str]:
    reg = str(data_used.get("tail_registration") or "").strip().upper()
    if reg:
        return reg
    try:
        from rag.aviation_tail import primary_registration_from_query

        return primary_registration_from_query(query or "")
    except Exception:
        return None


def _phly_row_for_reg(data_used: dict, reg: str) -> Dict[str, Any]:
    rows = data_used.get("phlydata_rows") or data_used.get("phly_rows") or []
    if not isinstance(rows, list):
        return {}
    for candidate in rows:
        if not isinstance(candidate, dict):
            continue
        rn = (candidate.get("registration_number") or "").strip().upper()
        if rn == reg:
            return candidate
    return rows[0] if rows and isinstance(rows[0], dict) else {}


def _format_price(raw: Any) -> Optional[str]:
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
        if val >= 1_000_000:
            return f"${val / 1_000_000:.2f}M"
        if val >= 1_000:
            return f"${val:,.0f}"
        return f"${val:.2f}"
    except (TypeError, ValueError):
        s = str(raw).strip()
        return s if s else None


def build_tail_aircraft_profile_block(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Merge registry, listing, and market facts for detail-level tail queries."""
    du = data_used if isinstance(data_used, dict) else {}
    depth_mode = str(du.get("tail_depth_mode") or "").strip().lower()
    depth, reg_from_mode = classify_tail_depth_mode(query)
    if depth not in (
        TailDepthMode.DETAIL,
        TailDepthMode.ACQUISITION,
        TailDepthMode.ACQUISITION_RISKS,
        TailDepthMode.COMPARISON,
        TailDepthMode.CONTEXT,
    ) and depth_mode not in (
        "detail",
        "acquisition",
        "acquisition_risks",
        "comparison",
        "context",
    ):
        return ""

    try:
        from services.broker_execution.tail_fact_loader import ensure_tail_facts_for_query

        ensure_tail_facts_for_query(query, du)
    except Exception:
        pass

    reg = _extract_registration(query, du) or reg_from_mode
    if not reg:
        return ""

    facts = du.get("tail_selected_facts") or du.get("tail_facts") or []
    phly = _phly_row_for_reg(du, reg)
    faa = du.get("faa_master_row") if isinstance(du.get("faa_master_row"), dict) else {}

    lines: List[str] = [f"[TAIL AIRCRAFT PROFILE — {reg}]", ""]

    lines.append("Registry / identity:")
    if facts:
        for f in facts:
            if not isinstance(f, dict):
                continue
            label = str(f.get("label") or f.get("kind") or "Fact")
            val = str(f.get("value") or "").strip()
            if val:
                lines.append(f"• {label}: {val}")
    elif faa:
        for key, label in (
            ("faa_reference_model", "Aircraft type"),
            ("registrant_name", "FAA registrant"),
            ("serial_number", "Serial"),
            ("year_mfr", "Year"),
        ):
            v = faa.get(key)
            if v not in (None, ""):
                lines.append(f"• {label}: {v}")

    listing_lines: List[str] = []
    if phly:
        status = (phly.get("aircraft_status") or "").strip()
        ask = _format_price(phly.get("ask_price"))
        owner = (phly.get("owner") or phly.get("registered_owner") or "").strip()
        mfr = (phly.get("manufacturer") or "").strip()
        mdl = (phly.get("model") or "").strip()
        mm = " ".join(x for x in (mfr, mdl) if x).strip()
        year = phly.get("year") or phly.get("year_mfr")
        serial = (phly.get("serial_number") or phly.get("serial") or "").strip()
        location = (phly.get("location") or phly.get("base_location") or "").strip()
        hours = phly.get("airframe_total_time") or phly.get("total_time")
        eng = (phly.get("engine_program") or "").strip()
        apu = (phly.get("apu_program") or "").strip()
        maint = (phly.get("maintenance_tracking_program") or phly.get("maintenance_program") or "").strip()
        damage = (phly.get("damage_history") or phly.get("incident_history") or "").strip()
        cabin = (phly.get("cabin_configuration") or phly.get("passenger_capacity") or "").strip()
        if status:
            listing_lines.append(f"• Listing status: {status}")
        if ask:
            listing_lines.append(f"• Internal ask: {ask}")
        if owner and not any("owner" in ln.lower() for ln in lines):
            listing_lines.append(f"• Owner (Phly): {owner}")
        if mm and not any("aircraft" in ln.lower() for ln in lines):
            listing_lines.append(f"• Marketing type: {mm}")
        if year:
            listing_lines.append(f"• Year: {year}")
        if serial:
            listing_lines.append(f"• Serial: {serial}")
        if location:
            listing_lines.append(f"• Location: {location}")
        if hours not in (None, ""):
            listing_lines.append(f"• Airframe hours: {hours}")
        if eng:
            listing_lines.append(f"• Engine program: {eng}")
        if apu:
            listing_lines.append(f"• APU program: {apu}")
        if maint:
            listing_lines.append(f"• Maintenance tracking: {maint}")
        if damage:
            listing_lines.append(f"• Damage / incident: {damage}")
        if cabin:
            listing_lines.append(f"• Cabin / capacity: {cabin}")

    if listing_lines:
        lines.append("")
        lines.append("Phly listing / inventory:")
        lines.extend(listing_lines)

    mr = du.get("market_reality")
    if isinstance(mr, dict) and mr:
        band = mr.get("band_mid_usd") or mr.get("mid_band_usd")
        model_ctx = mr.get("model") or mr.get("aircraft_model") or ""
        market_lines: List[str] = []
        if band:
            try:
                market_lines.append(f"• Market mid-band: ${float(band) / 1_000_000:.1f}M")
            except (TypeError, ValueError):
                market_lines.append(f"• Market mid-band: {band}")
        note = str(mr.get("brief") or mr.get("summary") or "").strip()
        if note and len(note) < 400:
            market_lines.append(f"• Context: {note}")
        elif model_ctx:
            market_lines.append(f"• Market context model: {model_ctx}")
        if market_lines:
            lines.append("")
            lines.append("Market context:")
            lines.extend(market_lines[:4])

    body = "\n".join(lines).strip()
    return body if len(body) > 100 else ""


__all__ = ["build_tail_aircraft_profile_block"]
