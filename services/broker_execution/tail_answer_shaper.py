"""
Tail answer shaper — client-facing prose for registry turns (non-LLM and post-LLM hygiene).

Only **owner** and **sale_status** use the short registry card. Other tail intents use
specialized shapes or defer to the LLM.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.broker_execution.tail_depth_mode import TailDepthMode, registry_template_depths


def _fact_value(facts: List[Dict[str, str]], kind: str) -> str:
    for f in facts:
        if f.get("kind") == kind and f.get("value"):
            return str(f.get("value")).strip()
    for f in facts:
        if kind in str(f.get("label") or "").lower() and f.get("value"):
            return str(f.get("value")).strip()
    return ""


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


def _is_for_sale(status: str) -> bool:
    s = (status or "").strip().lower()
    if not s:
        return False
    if any(x in s for x in ("sold", "not for sale", "no longer", "withdrawn", "delivered")):
        return False
    return any(x in s for x in ("for sale", "available", "on market", "listed", "active listing"))


def _format_price(raw: Any) -> Optional[str]:
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


def _shape_engine_program(reg: str, data_used: dict) -> str:
    row = _phly_row(data_used, reg)
    lines: List[str] = []
    eng = (row.get("engine_program") or "").strip()
    apu = (row.get("apu_program") or "").strip()
    maint = (row.get("maintenance_tracking_program") or row.get("maintenance_program") or "").strip()
    if eng:
        lines.append(f"Engine program: {eng}")
    else:
        lines.append("Engine program: not listed in synced Phly data for this tail.")
    if apu:
        lines.append(f"APU program: {apu}")
    if maint:
        lines.append(f"Maintenance tracking: {maint}")
    defer = (row.get("engine_program_deferment") or "").strip()
    if defer:
        lines.append(f"Engine program deferment: {defer}")
    return "\n".join(lines).strip()


def shape_tail_client_answer(
    answer: str,
    *,
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Rewrite registry answers only for owner/sale registry-card intents."""
    du = data_used if isinstance(data_used, dict) else {}
    try:
        from services.broker_execution.tail_depth_mode import classify_tail_depth_mode

        depth, reg_from_mode = classify_tail_depth_mode(query)
    except Exception:
        depth, reg_from_mode = TailDepthMode.NONE, None

    depth_name = str(du.get("tail_depth_mode") or depth.value or "").strip().lower()
    reg = str(du.get("tail_registration") or reg_from_mode or "").upper()
    if not reg:
        m = re.search(r"\b(N(?=[A-Z0-9]*\d)[A-Z0-9]{2,6})\b", (query or "").upper())
        reg = m.group(1) if m else ""

    if depth_name == "engine_program" or depth == TailDepthMode.ENGINE_PROGRAM:
        try:
            from services.broker_execution.tail_acquisition_dossier import render_engine_program_answer

            short = render_engine_program_answer(query, du)
            if short:
                return short
        except Exception:
            pass
        shaped = _shape_engine_program(reg, du)
        return shaped or (answer or "").strip()

    # All non-registry-card intents: do not replace LLM/dispatch prose with fact card.
    try:
        depth_enum = TailDepthMode(depth_name) if depth_name else depth
    except ValueError:
        depth_enum = depth
    if depth_enum not in registry_template_depths():
        return (answer or "").strip()

    try:
        from services.broker_execution.tail_fact_loader import ensure_tail_facts_for_query
        from services.broker_execution.tail_fact_renderer import select_tail_facts

        ensure_tail_facts_for_query(query, du)
    except Exception:
        pass

    facts = du.get("tail_selected_facts") or du.get("tail_facts") or []
    if reg and not facts:
        try:
            from services.broker_execution.tail_fact_renderer import select_tail_facts

            facts = select_tail_facts(du, reg)
        except Exception:
            facts = []

    phly = _phly_row(du, reg)
    owner = _fact_value(facts, "ownership") or str(phly.get("owner") or "").strip()
    aircraft = _fact_value(facts, "aircraft_model")
    year = _fact_value(facts, "year")
    status = _fact_value(facts, "registry_status") or str(phly.get("aircraft_status") or "").strip()
    if phly and not aircraft:
        mfr = str(phly.get("manufacturer") or "").strip()
        mdl = str(phly.get("model") or "").strip()
        aircraft = " ".join(x for x in (mfr, mdl) if x).strip()
    ask = _format_price(phly.get("ask_price") if phly else None)

    if depth_name in ("owner", "tail_owner") or depth == TailDepthMode.OWNER:
        if owner:
            tail = reg or "this aircraft"
            type_bit = f", a {aircraft}" if aircraft else ""
            line = f"{owner} is the registered owner of {tail}{type_bit}."
            if year:
                line += f" Year of manufacture: {year}."
            return line.strip()
        return (answer or "").strip()

    if depth_name in ("sale_status", "tail_sale_status") or depth == TailDepthMode.SALE_STATUS:
        for_sale = _is_for_sale(status)
        lead = "Yes" if for_sale else "No"
        if for_sale and ask:
            lead += f" — listed around {ask}"
        elif for_sale and status:
            lead += f" — {status}"
        elif status:
            lead += f" — status: {status}"
        else:
            lead += " — listing status not confirmed in synced inventory data"

        support: List[str] = []
        if aircraft:
            support.append(f"Aircraft: {aircraft}")
        if owner:
            support.append(f"Owner: {owner}")
        if year:
            support.append(f"Year: {year}")
        body = lead + "."
        if support:
            body += "\n" + "\n".join(f"• {s}" for s in support[:3])
        return body.strip()

    return (answer or "").strip()


__all__ = ["shape_tail_client_answer"]
