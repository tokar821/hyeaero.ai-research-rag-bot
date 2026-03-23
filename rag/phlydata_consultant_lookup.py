"""
Direct PostgreSQL lookup for Ask Consultant when the user cites a PhlyData serial / tail.

PhlyData tab uses ``phlydata_aircraft``; RAG/Pinecone often does not index it, so vector search
misses these rows. This module runs before Pinecone when we detect serial- or N-number-like tokens.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


def wants_consultant_owner_operator_context(user_query: str) -> bool:
    """True when the user is asking about registered owner, operator, or who owns the aircraft."""
    q = (user_query or "").strip().lower()
    return any(
        w in q
        for w in (
            "who owns",
            "who own",
            "owner",
            "owned by",
            "registrant",
            "operator",
            "operated by",
            "who operate",
            "certificate holder",
            "airline",
        )
    )


def registry_web_hint_for_tail(registration: str) -> str:
    """
    Short phrase to bias Tavily toward the correct civil registry / region (not a legal claim).
    """
    u = (registration or "").strip().upper().replace(" ", "")
    if u.startswith("OY"):
        return "Denmark Danish civil aircraft register"
    if u.startswith("LN"):
        return "Norway Norwegian CAA"
    if u.startswith("SE"):
        return "Sweden Transportstyrelsen aircraft"
    if u.startswith("G-"):
        return "UK CAA aircraft register"
    if u.startswith("D-"):
        return "Germany Luftfahrt-Bundesamt"
    if u.startswith("F-"):
        return "France civil aviation register"
    if u.startswith("HB-"):
        return "Switzerland FOCA aircraft"
    if u.startswith("I-"):
        return "Italy ENAC aircraft"
    if u.startswith("EC-"):
        return "Spain AESA aircraft"
    if u.startswith("VH-" ):
        return "Australia CASA aircraft"
    if u.startswith("C-G") or u.startswith("CF-"):
        return "Canada Transport Canada aircraft"
    return ""


def build_owner_operator_focus_tavily_query(
    user_query: str,
    phly_rows: List[Dict[str, Any]],
    history_snippet: Optional[str] = None,
) -> Optional[str]:
    """
    Second Tavily query focused on tail + serial + operator/owner keywords.
    Use together with the LLM-expanded query to reduce stale or generic snippets.
    ``history_snippet`` lets short follow-ups (e.g. "thanks") still run owner-focused web search
    when the thread already asked about ownership.
    """
    blob = "\n".join(x for x in ((history_snippet or "").strip(), (user_query or "").strip()) if x)
    if not wants_consultant_owner_operator_context(blob) or not phly_rows:
        return None
    r = phly_rows[0]
    reg = (r.get("registration_number") or "").strip()
    serial = (r.get("serial_number") or "").strip()
    mfr = (r.get("manufacturer") or "").strip()
    mdl = (r.get("model") or "").strip()
    mm = " ".join(x for x in (mfr, mdl) if x).strip()
    hint = registry_web_hint_for_tail(reg)
    parts = [
        reg,
        serial,
        mm,
        "registered owner",
        "operator",
        "AOC",
        "aircraft management",
        "charter",
        "fleet",
        "current",
    ]
    mdl_l = (mdl or "").lower()
    sn_l = (serial or "").lower().replace("-", "")
    if "cj2" in mdl_l or "525a" in sn_l:
        parts.append("Cessna 525A Citation CJ2+")
    if hint:
        parts.append(hint)
    # Bias toward recent fleet/operator pages (Tavily still returns static pages; helps ranking)
    reg_u = (reg or "").strip().upper()
    if reg_u.startswith("OY"):
        parts.append("Denmark Transportstyrelsen")
        parts.append("Danish aircraft register")
        parts.append("AOC charter operator Denmark")
    q = " ".join(x for x in parts if x).strip()
    # Help search engines match the exact tail
    if reg_u and len(reg_u) <= 12 and reg_u.upper() == reg_u:
        quoted = f'"{reg_u}"'
        if quoted not in q:
            q = f"{quoted} {q}".strip()
    if len(q) < 8:
        return None
    return q[:500]


def extract_phlydata_tokens_with_history(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    max_history_messages: int = 14,
) -> List[str]:
    """
    Collect serial/tail tokens from the current message and recent chat turns so follow-ups
    like "thanks" or "U" still resolve the same aircraft.
    """
    seen: set[str] = set()
    out: List[str] = []

    def add_from_text(text: str) -> None:
        for t in extract_phlydata_lookup_tokens(text or ""):
            k = t.strip().upper()
            if k and k not in seen:
                seen.add(k)
                out.append(t)

    add_from_text(query or "")
    if history:
        tail = history[-max_history_messages:]
        for h in tail:
            role = (h.get("role") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            add_from_text(h.get("content") or "")
    return out


def extract_phlydata_lookup_tokens(query: str) -> List[str]:
    """
    Pull candidate serials / tails from natural language (typos like "aicraft" tolerated via
    loose patterns).
    """
    q = query or ""
    seen: set[str] = set()
    out: List[str] = []

    def add(raw: str) -> None:
        t = (raw or "").strip()
        if len(t) < 3 or len(t) > 36:
            return
        k = t.upper()
        if k in seen:
            return
        seen.add(k)
        out.append(t)

    # Citation-style: 510-0010, 525-0444 (common for Cessna / jets)
    for m in re.finditer(r"\b(\d{2,5}-\d{3,6})\b", q):
        add(m.group(1))

    # 525B0044, 172S11842
    for m in re.finditer(r"\b(\d{2,4}[A-Za-z][0-9A-Za-z\-]{2,12})\b", q):
        add(m.group(1))

    # After aircraft / serial / msn / s/n (allows "aicraft" typo: craft\s+)
    for m in re.finditer(
        r"(?:\baircraft\b|\baicraft\b|\bserial\b|\bmsn\b|\bs/n\b)\s*[:\#]?\s*([0-9A-Za-z\-]{3,24})\b",
        q,
        flags=re.IGNORECASE,
    ):
        add(m.group(1))

    # Tail / N-number
    for m in re.finditer(r"\b([Nn][-\s]?[A-Z0-9]{1,6})\b", q):
        add(m.group(1).upper().replace(" ", ""))

    return out


def ilike_patterns_for_token(token: str) -> List[str]:
    """Build a small set of ILIKE patterns for hyphen / spacing variants."""
    u = token.strip()
    if not u:
        return []
    pats: List[str] = []
    for variant in {u, u.upper(), u.lower()}:
        if variant and f"%{variant}%" not in pats:
            pats.append(f"%{variant}%")
    if "-" in u:
        collapsed = u.replace("-", "")
        pats.append(f"%{collapsed}%")
        pats.append(f"%{collapsed.upper()}%")
    # de-dupe preserve order
    seen: set[str] = set()
    uniq: List[str] = []
    for p in pats:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq[:8]


def lookup_phlydata_aircraft_rows(db: PostgresClient, tokens: List[str]) -> List[Dict[str, Any]]:
    """Return matching ``phlydata_aircraft`` rows (deduped by aircraft_id)."""
    if not tokens:
        return []
    all_pats: List[str] = []
    for t in tokens:
        all_pats.extend(ilike_patterns_for_token(t))
    # cap for query size
    all_pats = list(dict.fromkeys(all_pats))[:28]
    if not all_pats:
        return []
    try:
        rows = db.execute_query(
            """
            SELECT aircraft_id, serial_number, registration_number, manufacturer, model,
                   manufacturer_year, delivery_year, category
            FROM public.phlydata_aircraft
            WHERE serial_number ILIKE ANY(%s)
               OR registration_number ILIKE ANY(%s)
            LIMIT 25
            """,
            (all_pats, all_pats),
        )
    except Exception as e:
        logger.warning("phlydata_consultant_lookup: query failed (%s); skipping direct path.", e)
        return []
    if not rows:
        return []
    by_id: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        aid = r.get("aircraft_id")
        if aid is not None and aid not in by_id:
            by_id[aid] = dict(r)
    return list(by_id.values())[:12]


def format_phlydata_consultant_answer(
    db: PostgresClient,
    rows: List[Dict[str, Any]],
    fetch_faa_master,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build plain-text answer and ``data_used`` summary.
    ``fetch_faa_master`` is ``services.faa_master_lookup.fetch_faa_master_owner_rows``.
    """
    lines: List[str] = []
    lines.append("From Hye Aero PhlyData internal aircraft database (phlydata_aircraft):")
    lines.append(
        "**Note:** phlydata_aircraft stores identity only (serial, tail, make/model, year, category). "
        "It does NOT store registered owner or operator names — those appear only in the FAA MASTER line below when present, "
        "otherwise you must take them from Tavily web snippets and vector context."
    )
    faa_hits = 0
    for i, r in enumerate(rows, 1):
        if i > 1:
            lines.append("")
        sn = (r.get("serial_number") or "—").strip() or "—"
        reg = (r.get("registration_number") or "—").strip() or "—"
        mfr = (r.get("manufacturer") or "").strip()
        mdl = (r.get("model") or "").strip()
        mm = " ".join(x for x in (mfr, mdl) if x).strip() or "—"
        y = r.get("manufacturer_year") or r.get("delivery_year")
        cat = (r.get("category") or "").strip() or "—"
        lines.append(f"- Aircraft {i}:")
        lines.append(f"  - Serial: {sn}")
        lines.append(f"  - Registration (tail): {reg}")
        lines.append(f"  - Make / model: {mm}")
        lines.append(f"  - Year: {y if y is not None else '—'}")
        lines.append(f"  - Category: {cat}")

        serial = str(r.get("serial_number") or "").strip()
        reg_s = str(r.get("registration_number") or "").strip() or None
        mdl_q = str(r.get("model") or "").strip() or None
        try:
            faa_rows, _kind = fetch_faa_master(
                db,
                serial=serial or sn,
                model=mdl_q,
                registration=reg_s,
            )
        except Exception as e:
            logger.debug("faa_master enrich for consultant: %s", e)
            faa_rows = []
        if faa_rows:
            faa_hits += 1
            fr = faa_rows[0]
            rn = (fr.get("registrant_name") or "").strip()
            if rn:
                lines.append(f"  - FAA MASTER registrant (faa_master): {rn}")
            st1 = (fr.get("street") or "").strip()
            st2 = (fr.get("street2") or "").strip()
            street_combined = " ".join(x for x in (st1, st2) if x).strip()
            if street_combined:
                lines.append(f"  - FAA mailing street: {street_combined}")
            city = (fr.get("city") or "").strip()
            st = (fr.get("state") or "").strip()
            z = (fr.get("zip_code") or "").strip()
            ctry = (fr.get("country") or "").strip()
            loc = ", ".join(x for x in (city, st, z, ctry) if x)
            if loc:
                lines.append(f"  - FAA mailing location: {loc}")
            # Loud anchor so the answer LLM cannot substitute a different "registrant" from Tavily/RAG.
            if rn:
                lines.append("")
                lines.append(
                    "[FOR USER REPLY — U.S. legal registrant (FAA MASTER) — MANDATORY VERBATIM]"
                )
                lines.append(
                    "  The FAA-recorded legal registrant for this aircraft is exactly the following. "
                    "You MUST state this name and mailing address as written here. "
                    "Do NOT replace it with a company name from web search, vector DB, or a name formed from the tail number (e.g. N123AB LLC) unless that exact phrase appears in this block."
                )
                lines.append(f"  Registrant name: {rn}")
                if street_combined:
                    lines.append(f"  Mailing street: {street_combined}")
                if loc:
                    lines.append(f"  Mailing city/state/ZIP/country: {loc}")
        else:
            reg_u = (reg_s or "").strip().upper()
            non_us_hint = ""
            if reg_u and not reg_u.startswith("N") and len(reg_u) >= 2:
                non_us_hint = (
                    " This tail is not in U.S. N-number format; FAA MASTER often has no legal registrant for the "
                    "state of registry (e.g. European civil registers). "
                )
            lines.append(
                "  - FAA MASTER: no U.S. registrant row in our latest MASTER snapshot for this serial/tail."
                + non_us_hint
                + "For current registered owner or operating company (AOC / charter / management), synthesize Tavily web "
                "snippets and vector DB excerpts below like a professional researcher: lead with the best-supported "
                "company name(s) tied to this tail/serial (fleet, operator, AOC, or registry text). If snippets conflict, "
                "state both and which sources support each. Do not invent database or portal names unless the exact phrase "
                "appears in a snippet. Do not invent company names — only names that appear in context below."
            )

    data_used = {
        "phlydata_aircraft_rows": len(rows),
        "faa_master_owner_rows": faa_hits,
    }
    return "\n".join(lines), data_used


def enrich_tavily_query_for_consultant(
    user_query: str,
    base_tavily_query: str,
    phly_rows: List[Dict[str, Any]],
    history_snippet: Optional[str] = None,
) -> str:
    """
    When the user asks about ownership and we matched PhlyData rows, append tail/serial/model
    plus owner/operator terms so Tavily returns registry and fleet pages (not just the raw user phrase).
    """
    blob = "\n".join(x for x in ((history_snippet or "").strip(), (user_query or "").strip()) if x)
    ql = blob.lower()
    wants_owner = any(
        w in ql
        for w in (
            "who owns",
            "who own",
            "owner",
            "owned by",
            "registrant",
            "operator",
            "operated by",
            "registration",
        )
    )
    if not wants_owner or not phly_rows:
        return (base_tavily_query or user_query)[:500]

    parts: List[str] = [(base_tavily_query or user_query).strip()]
    for r in phly_rows[:2]:
        reg = (r.get("registration_number") or "").strip()
        serial = (r.get("serial_number") or "").strip()
        mfr = (r.get("manufacturer") or "").strip()
        mdl = (r.get("model") or "").strip()
        mm = " ".join(x for x in (mfr, mdl) if x).strip()
        if reg:
            parts.append(reg)
        if serial:
            parts.append(serial)
        if mm:
            parts.append(mm)
        rh = registry_web_hint_for_tail(reg)
        if rh:
            parts.append(rh)
    parts.append("registered owner operator airline charter")
    merged = " ".join(x for x in parts if x)
    return merged[:500]
