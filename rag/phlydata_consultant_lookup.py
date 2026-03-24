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
from rag.phlydata_aircraft_schema import phlydata_aircraft_select_sql

logger = logging.getLogger(__name__)


def _append_phly_internal_snapshot_lines(lines: List[str], r: Dict[str, Any]) -> None:
    """Emit CSV-backed fields from phlydata_aircraft when present (internal export, not aircraft_listings)."""

    def _s(v: Any) -> str:
        if v is None:
            return ""
        if hasattr(v, "isoformat"):
            try:
                return str(v.isoformat())[:10]
            except Exception:
                return str(v).strip()
        return str(v).strip()

    def _ask_placeholder(t: str) -> bool:
        u = (t or "").strip().upper()
        return not u or u in ("M/O", "N/A", "NA", "TBD", "—", "-")

    pairs: List[tuple[str, str]] = [
        ("Aircraft status (internal export)", "aircraft_status"),
        ("Transaction status", "transaction_status"),
        ("Ask price (as in export)", "ask_price"),
        ("Take price", "take_price"),
        ("Sold price", "sold_price"),
        ("Airframe total time (hrs)", "airframe_total_time"),
        ("APU total time (hrs)", "apu_total_time"),
        ("Prop total time (hrs)", "prop_total_time"),
        ("Engine program", "engine_program"),
        ("Engine program deferment", "engine_program_deferment"),
        ("Engine program deferment amount", "engine_program_deferment_amount"),
        ("APU program", "apu_program"),
        ("APU program deferment", "apu_program_deferment"),
        ("APU program deferment amount", "apu_program_deferment_amount"),
        ("Airframe program", "airframe_program"),
        ("Maintenance tracking program", "maintenance_tracking_program"),
        ("Date listed", "date_listed"),
        ("Export / source updated at", "source_updated_at"),
        ("Interior year", "interior_year"),
        ("Exterior year", "exterior_year"),
        ("Seller broker", "seller_broker"),
        ("Seller", "seller"),
        ("Buyer broker", "buyer_broker"),
        ("Buyer", "buyer"),
        ("Registration country", "registration_country"),
        ("Based country", "based_country"),
        ("Number of passengers", "number_of_passengers"),
        ("Updated by", "updated_by"),
        ("Has damage", "has_damage"),
        ("Feature source", "feature_source"),
        ("Features", "features"),
        ("Next inspections", "next_inspections"),
    ]
    pair_db_keys = {dbk for _, dbk in pairs}
    skip_top = {
        "serial_number",
        "registration_number",
        "manufacturer",
        "model",
        "manufacturer_year",
        "delivery_year",
        "category",
    }
    any_line = False

    # Extra export columns first — wide CSVs often put list/for-sale/ask in csv_* headers; show before typed fields.
    skip = skip_top | pair_db_keys | {"aircraft_id", "id", "csv_extra"}
    csv_keys = sorted(k for k in r.keys() if isinstance(k, str) and k.startswith("csv_") and k not in skip)
    if csv_keys:
        lines.append("  - Additional PhlyData export columns (csv_*):")
        for k in csv_keys:
            v = r.get(k)
            if v is None:
                continue
            t = _s(v)
            if t:
                lines.append(f"    - {k} (from spreadsheet header slug): {t}")
                any_line = True

    for label, key in pairs:
        v = r.get(key)
        if v is None:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lines.append(f"  - {label}: {v}")
            any_line = True
            continue
        t = _s(v)
        if t:
            lines.append(f"  - {label}: {t}")
            any_line = True

    # If typed ask is M/O/empty but another PhlyData column holds a dollar figure, call it out for the LLM.
    ap = _s(r.get("ask_price"))
    if _ask_placeholder(ap):
        price_hint_keys = re.compile(r"(ask|price|list|sale|offer|amount)", re.I)
        for k in sorted(r.keys()):
            if not isinstance(k, str) or k in skip | pair_db_keys:
                continue
            if not price_hint_keys.search(k):
                continue
            tv = _s(r.get(k))
            if tv and not _ask_placeholder(tv) and re.search(r"[\d$]", tv):
                lines.append(
                    f"  - Note: canonical ask_price is placeholder ({ap or 'empty'}) but column "
                    f"{k.replace('_', ' ')} contains: {tv} — still PhlyData; report both if relevant."
                )
                any_line = True
                break

    for k in sorted(r.keys()):
        if k in skip or k.startswith("csv_") or k in pair_db_keys:
            continue
        v = r.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lines.append(f"  - {k.replace('_', ' ')}: {v}")
            any_line = True
            continue
        t = _s(v)
        if t:
            lines.append(f"  - {k.replace('_', ' ')}: {t}")
            any_line = True
    if any_line:
        lines.append(
            "  [LLM: Hye Aero policy — every line above is PhlyData. If csv_* columns disagree with aircraft_status "
            "or ask_price, quote ALL of them (export may use multiple columns). Do not invent values; do not replace "
            "these with listing-ingest or web. For live deals, note snapshot timing and verify with broker/platform.]"
        )


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


_N_TAIL_TOKEN = re.compile(r"^N[A-Z0-9]{1,6}$", re.IGNORECASE)


def _normalize_registration_compare(value: Any) -> str:
    """Uppercase tail for equality checks (no spaces/hyphens)."""
    s = (value if value is not None else "") or ""
    return re.sub(r"[\s\-]+", "", str(s).strip().upper())


def _normalize_serial_compare(value: Any) -> str:
    """Alphanumeric only, uppercase — matches hyphenated serials to collapsed tokens."""
    s = (value if value is not None else "") or ""
    return re.sub(r"[^A-Z0-9]", "", str(s).strip().upper())


def _phly_rows_match_tokens_exact_first(rows: List[Dict[str, Any]], tokens: List[str]) -> List[Dict[str, Any]]:
    """
    ILIKE '%N807JS%' can match the wrong aircraft (e.g. registration containing that substring).
    When tokens include a clear U.S. tail or serial, keep only rows that match exactly; if none,
    fall back to the original list (avoid empty PhlyData when data is messy).
    """
    if not rows or not tokens:
        return rows

    tail_norms: set[str] = set()
    serial_norms: set[str] = set()
    for raw in tokens:
        t = (raw or "").strip()
        if not t:
            continue
        tc = _normalize_registration_compare(t.replace("N-", "N"))
        if _N_TAIL_TOKEN.fullmatch(tc):
            tail_norms.add(tc)
        # Citation-style serial 560-5354
        if re.search(r"\d", t) and len(t) >= 4:
            serial_norms.add(_normalize_serial_compare(t))

    narrowed: List[Dict[str, Any]] = []
    if tail_norms:
        for r in rows:
            reg = _normalize_registration_compare(r.get("registration_number"))
            if reg and reg in tail_norms:
                narrowed.append(r)
        if narrowed:
            return narrowed

    if serial_norms:
        for r in rows:
            sn = _normalize_serial_compare(r.get("serial_number"))
            if sn and sn in serial_norms:
                narrowed.append(r)
        if narrowed:
            return narrowed

    return rows


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
            f"""
            SELECT aircraft_id,
            {phlydata_aircraft_select_sql(include_cast_id=False, db=db)}
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
    rows = _phly_rows_match_tokens_exact_first(rows, tokens)
    # Prefer exact tail/serial matches first, then stable order
    def _rank(r: Dict[str, Any]) -> tuple:
        reg = _normalize_registration_compare(r.get("registration_number"))
        sn = _normalize_serial_compare(r.get("serial_number"))
        tail_hit = 0
        serial_hit = 0
        for raw in tokens:
            tc = _normalize_registration_compare((raw or "").strip().replace("N-", "N"))
            if _N_TAIL_TOKEN.fullmatch(tc) and reg == tc:
                tail_hit = 1
            snt = _normalize_serial_compare(raw or "")
            if snt and sn == snt:
                serial_hit = 1
        return (-tail_hit, -serial_hit)

    rows = sorted(rows, key=_rank)
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
    lines.append(
        "PhlyData — Hye Aero's canonical internal aircraft record (table phlydata_aircraft). "
        "This is what Hye Aero treats as source of truth for the product: identity and every field loaded from the internal export — "
        "not the separate aircraft_listings marketplace-ingest table."
    )
    lines.append(
        "Evaluation order: When answering, ground and lead on PhlyData (this block) for identity and all snapshot fields below. "
        "Controller / exchanges / aircraft_listings / Tavily / vector DB are supplemental — never override PhlyData internal fields with them; "
        "if they disagree, state PhlyData first, then Separately, … for the other source. "
        "Snapshots can still be stale; for live availability and contracts, say verify with broker/platform. "
        "Registered owner names are not in PhlyData — they appear only in the FAA MASTER line below when present; "
        "otherwise use Tavily / vector context and label sources clearly."
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
        _append_phly_internal_snapshot_lines(lines, r)

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
                    "  Source: PhlyData (Hye Aero aircraft identity) + FAA MASTER (registrant/address). "
                    "The FAA-recorded legal registrant is exactly the following — state name and mailing verbatim. "
                    "Do NOT replace with web search, vector DB, or a tail-derived LLC unless that exact phrase appears here."
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
