"""
Phase 56 — select and render verified tail facts before broker templates.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def select_tail_facts(data_used: dict, registration: str) -> List[Dict[str, str]]:
    """Build ordered fact records from loaded registry context."""
    reg = (registration or "").strip().upper()
    facts: List[Dict[str, str]] = []
    if not reg:
        return facts

    facts.append({"kind": "registration", "label": "Registration", "value": reg})

    rows = data_used.get("phlydata_rows") or data_used.get("phly_rows") or []
    row: Dict[str, str] = {}
    if isinstance(rows, list) and rows:
        for candidate in rows:
            if not isinstance(candidate, dict):
                continue
            rn = (candidate.get("registration_number") or "").strip().upper()
            if rn == reg:
                row = candidate
                break
        if not row and isinstance(rows[0], dict):
            row = rows[0]

    faa = data_used.get("faa_master_row") or {}
    faa_type = ""
    if isinstance(faa, dict):
        faa_type = (faa.get("faa_reference_model") or faa.get("model") or "").strip()

    if isinstance(row, dict):
        owner = (row.get("owner") or row.get("registered_owner") or "").strip()
        if owner:
            facts.append({"kind": "ownership", "label": "Owner", "value": owner})
        mfr = (row.get("manufacturer") or "").strip()
        mdl = (row.get("model") or "").strip()
        mm = " ".join(x for x in (mfr, mdl) if x).strip()
        if faa_type and (
            not mm
            or re.fullmatch(r"[A-Z0-9\-]{2,10}", mdl.replace(" ", ""))
            or mdl.upper() == (faa.get("mfr_mdl_code") or "").strip().upper()
        ):
            mm = faa_type
        if mm:
            facts.append({"kind": "aircraft_model", "label": "Aircraft", "value": mm})
        serial = (row.get("serial_number") or "").strip()
        if serial:
            facts.append({"kind": "serial_number", "label": "Serial", "value": serial})
        status = (row.get("aircraft_status") or "").strip()
        if status:
            facts.append({"kind": "registry_status", "label": "Status", "value": status})

    if isinstance(faa, dict):
        rn = (faa.get("registrant_name") or "").strip()
        if rn and not any(f.get("kind") == "ownership" for f in facts):
            facts.append({"kind": "ownership", "label": "FAA registrant", "value": rn})
        ref = (faa.get("faa_reference_model") or faa.get("model") or "").strip()
        if ref and not any(f.get("kind") == "aircraft_model" for f in facts):
            facts.append({"kind": "aircraft_model", "label": "FAA aircraft type", "value": ref})
        serial = (faa.get("serial_number") or "").strip()
        if serial and not any(f.get("kind") == "serial_number" for f in facts):
            facts.append({"kind": "serial_number", "label": "FAA serial", "value": serial})
        year = faa.get("year_mfr")
        if year is not None:
            facts.append({"kind": "year", "label": "Year of manufacture", "value": str(year)})

    # De-dupe by kind+value
    seen = set()
    out: List[Dict[str, str]] = []
    for f in facts:
        key = (f.get("kind"), f.get("value"))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def render_tail_facts_block(facts: List[Dict[str, str]], *, registration: str) -> str:
    if not facts:
        return ""
    reg = (registration or "").strip().upper()
    lines = [f"Registry facts for {reg}:"]
    for f in facts:
        label = f.get("label") or f.get("kind") or "Fact"
        value = f.get("value") or ""
        if value:
            lines.append(f"• {label}: {value}")
    return "\n".join(lines).strip()


def count_rendered_tail_facts(answer: str, facts: List[Dict[str, str]]) -> int:
    if not answer or not facts:
        return 0
    low = answer.lower()
    count = 0
    for f in facts:
        val = str(f.get("value") or "").strip()
        if not val:
            continue
        if val.lower() in low or (len(val) > 4 and val[:12].lower() in low):
            count += 1
    return count


def prepend_tail_facts_to_answer(
    answer: str,
    *,
    facts: List[Dict[str, str]],
    registration: str,
    data_used: Optional[dict] = None,
) -> str:
    """Facts first — broker template / asks follow only when facts exist."""
    body = (answer or "").strip()
    block = render_tail_facts_block(facts, registration=registration)
    if not block:
        if isinstance(data_used, dict):
            data_used["tail_fallback_used"] = True
        return body
    if isinstance(data_used, dict):
        data_used["tail_fallback_used"] = False
        data_used["tail_facts_rendered"] = count_rendered_tail_facts(block, facts)
    if block.lower() in body.lower():
        return body
    return f"{block}\n\n{body}".strip() if body else block


def render_tail_facts_for_llm_context(
    facts: List[Dict[str, str]],
    *,
    registration: str,
) -> str:
    """Authoritative registry block for consultant LLM (not a client-facing template)."""
    reg = (registration or "").strip().upper()
    if not facts:
        return ""
    lines = [
        f"[REGISTRY FACTS — {reg} — MANDATORY FOR ANSWER]",
        "Narrate these facts in natural professional prose. Max 5 short bullets or 2 sentences.",
        "Do NOT ask for listing package, engine program, or 'before treating this tail' boilerplate.",
    ]
    for kind, label in _FACT_LABEL_PRIORITY:
        for f in facts:
            if f.get("kind") != kind:
                continue
            val = str(f.get("value") or "").strip()
            if val:
                lines.append(f"- {label}: {val}")
            break
    return "\n".join(lines).strip()


__all__ = [
    "count_rendered_tail_facts",
    "prepend_tail_facts_to_answer",
    "render_tail_facts_block",
    "render_tail_facts_for_llm_context",
    "select_tail_facts",
]
