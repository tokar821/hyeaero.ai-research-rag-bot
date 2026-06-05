"""
Tail answer shaper — client-facing prose for registry turns (non-LLM and post-LLM hygiene).

Ownership / registry lookups: broker narrative lead + structured key details.
Sale status keeps the short yes/no card.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.broker_execution.tail_depth_mode import TailDepthMode, registry_template_depths

_REGISTRY_STYLE_RE = re.compile(
    r"(?is)\b(?:registry\s+lookup|registration\s+lookup|where\s+.{0,40}?\s+registered|"
    r"registered\s+to|registration\s+details|registrant)\b"
)
_TRUST_OWNER_RE = re.compile(r"(?is)\b(?:trust|trustee|bank\s+of\s+utah)\b")
_OPERATE_SIGNAL_RE = re.compile(
    r"(?is)\b(?:operat(?:ed|es|ing)?\s+by|managed\s+by|fleet\s+of|charter(?:ed)?\s+by)\b"
)


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


def _faa_row(data_used: dict) -> Dict[str, Any]:
    faa = data_used.get("faa_master_row")
    return faa if isinstance(faa, dict) else {}


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


def _location_from_faa(faa: Dict[str, Any]) -> str:
    city = str(faa.get("city") or "").strip()
    state = str(faa.get("state") or "").strip()
    if city and state:
        return f"{city}, {state}"
    return city or state or ""


def _is_trust_registrant(owner: str) -> bool:
    return bool(_TRUST_OWNER_RE.search(owner or ""))


def _operator_from_data_used(data_used: dict, owner: str) -> str:
    owner_low = (owner or "").lower()

    syn = data_used.get("tavily_llm_synthesis")
    if isinstance(syn, dict):
        name = str(syn.get("operating_company_name") or "").strip()
        if name and name.lower() not in owner_low:
            return name

    meta = data_used.get("phly_meta")
    if isinstance(meta, dict):
        for key in ("tavily_llm_synthesis", "faa_tavily_llm_hint"):
            nested = meta.get(key)
            if isinstance(nested, dict):
                name = str(nested.get("operating_company_name") or "").strip()
                if name and name.lower() not in owner_low:
                    return name

    for row in data_used.get("phlydata_rows") or data_used.get("phly_rows") or []:
        if not isinstance(row, dict):
            continue
        for field in ("operator", "operating_company", "management_company"):
            name = str(row.get(field) or "").strip()
            if name and name.lower() not in owner_low:
                return name

    for blob_key in ("tavily_block", "tavily_context"):
        blob = data_used.get(blob_key)
        if not isinstance(blob, str) or not blob.strip():
            continue
        for line in blob.splitlines():
            m = _OPERATE_SIGNAL_RE.search(line)
            if not m:
                continue
            tail = line[m.end() :].strip(" :—-")
            tail = re.split(r"[.;]\s", tail, maxsplit=1)[0].strip()
            if 3 < len(tail) < 120 and tail.lower() not in owner_low:
                return tail
    return ""


def _base_airport_label(phly: Dict[str, Any], faa: Dict[str, Any]) -> str:
    base = str(phly.get("base_code") or "").strip().upper()
    if not base:
        return ""
    try:
        from services.airport.airport_database import _ICAO_RAW

        profile = _ICAO_RAW.get(base) or {}
        name = str(profile.get("name") or "").strip()
        if name:
            return f"{name} ({base})"
    except Exception:
        pass
    return base


def _aircraft_descriptor(year: str, aircraft: str) -> str:
    if year and aircraft:
        return f"a {year} {aircraft}"
    if aircraft:
        return f"a {aircraft}"
    if year:
        return f"year {year}"
    return ""


def _registry_lead(
    *,
    reg: str,
    owner: str,
    aircraft: str,
    year: str,
    location: str,
    operator: str,
    query: str,
) -> str:
    registry_style = bool(_REGISTRY_STYLE_RE.search(query or "")) or _is_trust_registrant(owner)
    type_desc = _aircraft_descriptor(year, aircraft)

    if registry_style or operator:
        lead = f"The aircraft is officially registered to **{owner}**"
        if location:
            lead += f" (based in {location})"
        if operator:
            lead += f" and is operated by **{operator}**"
            if _is_trust_registrant(owner):
                lead += " for charter and management flights"
        lead += "."
        if _is_trust_registrant(owner) and not operator:
            lead += (
                " Trust-style registrants mask beneficial owner — "
                "get seller reps and UCC/title before LOI."
            )
        return lead

    lead = f"The aircraft registered as **{reg}**"
    if type_desc:
        lead += f" ({type_desc})"
    lead += f" is owned by **{owner}**"
    if location:
        lead += f" based in {location}"
    lead += "."
    return lead


def render_registry_broker_answer(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Broker narrative lead + key registration details for ownership/registry turns."""
    du = data_used if isinstance(data_used, dict) else {}
    reg = str(du.get("tail_registration") or "").strip().upper()
    if not reg:
        m = re.search(r"\b(N(?=[A-Z0-9]*\d)[A-Z0-9]{2,6})\b", (query or "").upper())
        reg = m.group(1) if m else ""

    facts = du.get("tail_selected_facts") or du.get("tail_facts") or []
    if reg and not facts:
        try:
            from services.broker_execution.tail_fact_loader import ensure_tail_facts_for_query
            from services.broker_execution.tail_fact_renderer import select_tail_facts

            ensure_tail_facts_for_query(query, du)
            facts = du.get("tail_selected_facts") or du.get("tail_facts") or []
        except Exception:
            pass

    if reg and not facts:
        try:
            from services.broker_execution.tail_fact_renderer import select_tail_facts

            facts = select_tail_facts(du, reg)
        except Exception:
            facts = []

    phly = _phly_row(du, reg)
    faa = _faa_row(du)
    owner = _fact_value(facts, "ownership") or str(phly.get("owner") or phly.get("registered_owner") or "").strip()
    if not owner:
        owner = str(faa.get("registrant_name") or "").strip()
    aircraft = _fact_value(facts, "aircraft_model") or str(faa.get("faa_reference_model") or faa.get("model") or "").strip()
    year = _fact_value(facts, "year") or str(faa.get("year_mfr") or phly.get("manufacturer_year") or "").strip()
    serial = _fact_value(facts, "serial_number") or str(faa.get("serial_number") or phly.get("serial_number") or "").strip()
    if phly and not aircraft:
        mfr = str(phly.get("manufacturer") or "").strip()
        mdl = str(phly.get("model") or "").strip()
        aircraft = " ".join(x for x in (mfr, mdl) if x).strip()

    if not owner and not aircraft:
        return ""

    location = _location_from_faa(faa)
    operator = _operator_from_data_used(du, owner)
    base = _base_airport_label(phly, faa)

    lines: List[str] = []
    if owner:
        lines.append(
            _registry_lead(
                reg=reg or "this aircraft",
                owner=owner,
                aircraft=aircraft,
                year=year,
                location=location,
                operator=operator,
                query=query,
            )
        )
    elif aircraft:
        lines.append(
            f"I have verified type data on **{reg}** ({aircraft}) but no confirmed registrant "
            "in synced FAA/Phly feeds — verify registry externally before relying on ownership."
        )
    else:
        return ""

    details: List[tuple[str, str]] = []
    if aircraft:
        details.append(("Aircraft Type", aircraft))
    if owner:
        details.append(("Registrant", owner))
    if operator and operator.lower() not in owner.lower():
        details.append(("Operator", operator))
    if serial:
        details.append(("Serial Number", serial))
    if year:
        details.append(("Year of Manufacture", year))
    if base:
        details.append(("Base Airport", base))
    if reg:
        details.append(("Registration", reg))

    if details:
        lines.append("")
        lines.append("**Key registration details:**")
        for label, val in details:
            bold_val = f"**{val}**" if label in ("Registrant", "Operator") else val
            lines.append(f"- **{label}:** {bold_val}")

    return "\n".join(lines).strip()


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

        ensure_tail_facts_for_query(query, du)
    except Exception:
        pass

    facts = du.get("tail_selected_facts") or du.get("tail_facts") or []
    phly = _phly_row(du, reg)
    owner = _fact_value(facts, "ownership") or str(phly.get("owner") or "").strip()
    aircraft = _fact_value(facts, "aircraft_model")
    year = _fact_value(facts, "year")
    status = _fact_value(facts, "registry_status") or str(phly.get("aircraft_status") or "").strip()
    ask = _format_price(phly.get("ask_price") if phly else None)

    if depth_name in ("owner", "tail_owner") or depth == TailDepthMode.OWNER:
        broker = render_registry_broker_answer(query, du)
        if broker:
            return broker
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


__all__ = ["render_registry_broker_answer", "shape_tail_client_answer"]
