"""
Direct PostgreSQL lookup for Ask Consultant against ``phlydata_aircraft``.

Users cite serials, tails, **model codes** (e.g. HA-420, 525-0682), **make/model** phrases, etc.
We query Postgres on **registration_number**, **serial_number**, **manufacturer**, **model**,
**category**, and **features** (plus ILIKE/patterns), then post-filter so every extracted token
matches **identity or text** on the row.

RAG/Pinecone usually does **not** embed Phly export rows; optional future: a dedicated PhlyData
vector index for fuzzy/long-form questions — this module remains the deterministic primary path.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from database.postgres_client import PostgresClient
from rag.phlydata_aircraft_schema import phlydata_aircraft_select_sql
from services.faa_master_lookup import registration_tail_canonical

logger = logging.getLogger(__name__)


def _phly_verbatim_scalar(value: Any, *, null_label: str = "(null)") -> str:
    """String for LLM copy-paste — no interpretation."""
    if value is None:
        return null_label
    if isinstance(value, Decimal):
        s = format(value, "f").rstrip("0").rstrip(".")
        return s if s else null_label
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    s = str(value).strip()
    return s if s else "(empty)"


def _phly_verbatim_money_field(value: Any, column: str) -> str:
    """Readable $ line when numeric; else exact export string."""
    if value is None:
        return f"(null) — phlydata_aircraft.{column} has no value in DB"
    try:
        x = float(value) if not isinstance(value, Decimal) else float(value)
        if abs(x - round(x)) < 1e-9:
            return f"${int(round(x)):,} (numeric phlydata_aircraft.{column})"
        return f"${x:,.2f} (numeric phlydata_aircraft.{column})".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return f"{_phly_verbatim_scalar(value)} (phlydata_aircraft.{column} as stored)"


def _append_phly_mandatory_verbatim_reply(lines: List[str], r: Dict[str, Any], index: int) -> None:
    """
    Hard anchor for the answer LLM: exact phlydata_aircraft typed fields — not paraphrased,
    not replaced by aircraft_listings or web.
    """
    reg = (r.get("registration_number") or "").strip() or "(null)"
    sn = (r.get("serial_number") or "").strip() or "(null)"
    mfr = (r.get("manufacturer") or "").strip()
    mdl = (r.get("model") or "").strip()
    mm = " ".join(x for x in (mfr, mdl) if x).strip() or "(null)"
    y = r.get("manufacturer_year") if r.get("manufacturer_year") is not None else r.get("delivery_year")
    cat = (r.get("category") or "").strip() or "(null)"
    ast = (r.get("aircraft_status") or "").strip()
    ast_out = ast if ast else "(null or empty in phlydata_aircraft.aircraft_status)"

    lines.append("")
    lines.append(
        f"[FOR USER REPLY — PhlyData (table phlydata_aircraft only) — MANDATORY VERBATIM — Aircraft {index}]"
    )
    lines.append(
        "  These keys come straight from Hye Aero internal Postgres (phlydata_aircraft). "
        "**Not** aircraft_listings, not Tavily, not vector DB. Copy status and prices into your answer exactly; "
        "do not infer, soften, or substitute marketplace listing fields."
    )
    lines.append(f"  registration_number: {reg}")
    lines.append(f"  serial_number: {sn}")
    lines.append(f"  manufacturer / model (combined): {mm}")
    lines.append(f"  manufacturer_year (or delivery if used): {y if y is not None else '(null)'}")
    lines.append(f"  category: {cat}")
    lines.append(f"  aircraft_status: {ast_out}")
    lines.append(f"  ask_price: {_phly_verbatim_money_field(r.get('ask_price'), 'ask_price')}")
    lines.append(f"  take_price: {_phly_verbatim_money_field(r.get('take_price'), 'take_price')}")
    lines.append(f"  sold_price: {_phly_verbatim_money_field(r.get('sold_price'), 'sold_price')}")
    lines.append(
        "  [LLM — non-negotiable] **Forbidden:** fabricating or paraphrasing these values; pulling status/ask from "
        "Hye Aero listing records or web as if they were PhlyData; inventing transaction_status (Phly export uses "
        "**aircraft_status** and **ask_price**). **Required:** When the user asks for internal DB / Phly data / "
        "ask or for-sale disposition, repeat **aircraft_status** and **ask_price** (and identity) consistent with "
        "the lines above; then you may add 'Separately, …' for listing-ingest or web."
    )


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
        return not u or u in ("M/O", "N/A", "NA", "TBD", "—", "-", "NULL", "NONE")

    def _csv_cell_looks_like_list_price(text: str) -> bool:
        """Heuristic: export value is probably a USD list/ask, not a serial/year."""
        x = (text or "").strip()
        if not x or _ask_placeholder(x):
            return False
        if "$" in x and re.search(r"\$[\d,]+(?:\.\d{1,2})?", x):
            return True
        if re.search(r"[\d]{1,3}(?:,[\d]{3})+(?:\.\d{1,2})?", x):
            return True
        if re.search(r"\b\d{1,3}(?:\.\d{1,3})?\s*(?:million|mm|mn)\b", x, re.I):
            return True
        return False

    pairs: List[tuple[str, str]] = [
        ("Aircraft status — for sale / disposition (phlydata_aircraft.aircraft_status)", "aircraft_status"),
        ("Ask price — internal export (phlydata_aircraft.ask_price)", "ask_price"),
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
        # Empty spreadsheet cell in DB → treat like NULL for money fields so placeholder recovery runs.
        if isinstance(v, str) and not v.strip() and key in ("ask_price", "take_price", "sold_price"):
            continue
        if isinstance(v, Decimal):
            lines.append(f"  - {label}: {v}")
            any_line = True
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
    found_alt_ask = False
    if _ask_placeholder(ap):
        price_hint_keys = re.compile(
            r"(ask|price|list|sale|offer|amount|value|usd|quote|figure)", re.I
        )
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
                found_alt_ask = True
                break
        # Some exports put list price only in generically named csv_* cells; name may not match price_hint_keys.
        if not found_alt_ask:
            for k in sorted(r.keys()):
                if not isinstance(k, str) or not k.startswith("csv_"):
                    continue
                tv = _s(r.get(k))
                if not tv or _ask_placeholder(tv):
                    continue
                if not _csv_cell_looks_like_list_price(tv):
                    continue
                lines.append(
                    f"  - Note: typed ask_price is unset or placeholder ({ap or 'empty'}) in PhlyData; "
                    f"export column {k} contains a list/ask-like figure: {tv}. Treat as PhlyData for pricing questions "
                    f"(do not say ask was not specified in PhlyData)."
                )
                any_line = True
                found_alt_ask = True
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
    # U.S. civil registry — critical when PhlyData has no row and faa_master ingest may lag public FAA
    if u.startswith("N") and len(u) > 1 and u[1:].replace("-", "").isalnum():
        return "FAA civil aircraft registry N-number flightaware"
    return ""


def build_owner_operator_focus_tavily_query(
    user_query: str,
    phly_rows: List[Dict[str, Any]],
    history_snippet: Optional[str] = None,
    lookup_tokens: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Second Tavily query focused on tail + serial + operator/owner keywords.
    Use together with the LLM-expanded query to reduce stale or generic snippets.
    ``history_snippet`` lets short follow-ups (e.g. "thanks") still run owner-focused web search
    when the thread already asked about ownership.

    When **PhlyData has no row**, we still build a query from ``lookup_tokens`` (synthetic identity)
    so U.S. tails (e.g. ``N448SJ``) get an FAA/registry-focused second pass instead of skipping entirely.
    """
    blob = "\n".join(x for x in ((history_snippet or "").strip(), (user_query or "").strip()) if x)
    rows_eff: List[Dict[str, Any]] = list(phly_rows or [])
    if not rows_eff and lookup_tokens:
        rows_eff = synthetic_phyl_like_rows_from_tokens(lookup_tokens) or []
    if not rows_eff:
        return None

    reg0 = (rows_eff[0].get("registration_number") or "").strip().upper()
    no_phly = not (phly_rows or [])
    n_tail_us = reg0.startswith("N") and len(reg0) > 1
    owner_q = wants_consultant_owner_operator_context(blob)
    # Secondary pass: explicit owner/operator questions, OR missing Phly + U.S. N-number (FAA / type / owner color)
    if not owner_q and not (no_phly and n_tail_us):
        return None

    r = rows_eff[0]
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
    if no_phly and n_tail_us:
        parts.extend(
            [
                "aircraft type",
                "make model",
                "year",
                "FAA registered owner",
            ]
        )
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

    # Tail / N-number (allow lowercase letters, e.g. n448sj → N448SJ)
    for m in re.finditer(r"\b([Nn][-\s]?[A-Za-z0-9]{1,6})\b", q):
        add(m.group(1).upper().replace(" ", ""))

    # International-style marks: V-682, XA-98723, G-CIVG — not MSN forms like 525-0444 (those start with a digit)
    for m in re.finditer(r"\b([A-Za-z]{1,3}-[A-Za-z0-9]{2,16})\b", q):
        add(m.group(1))

    # Leading-zero MSNs: 0011 (4 chars) through 0000171 (7 chars) — was missing 0\d{4,6} minimum length
    for m in re.finditer(r"\b(0\d{2,6})\b", q):
        add(m.group(1))

    # Standalone numeric tokens length 5–8 (avoid bare 4-digit years)
    for m in re.finditer(r"\b(\d{5,8})\b", q):
        add(m.group(1))

    # 3–4 digit serials without required leading zero (e.g. 0011 already covered; catches 425, 1910)
    for m in re.finditer(r"\b(\d{3,4})\b", q):
        s = m.group(1)
        if len(s) == 4 and s.startswith(("19", "20")):
            continue
        add(s)

    return out


def extract_us_registration_tail_candidates(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[str]:
    """
    Scan the **current message** and (when provided) **recent chat** for U.S. civil registration marks
    (N-numbers), without relying on :func:`consultant_phly_lookup_token_list` alone.

    Used so **faa_master** standalone lookup still runs for tails like ``N448SJ`` when the user cites the
    tail only in an earlier turn (follow-up: "what about ownership?") or when Phly SQL tokens differ.

    History scan fixes gaps where the raw ``query`` string has no tail but the thread does.
    """
    seen: set[str] = set()
    out: List[str] = []

    def add(raw: str) -> None:
        t = (raw or "").strip().upper().replace(" ", "")
        if len(t) < 3 or len(t) > 10:
            return
        if t in seen:
            return
        seen.add(t)
        out.append(t)

    def scan_text(blob: str) -> None:
        if not (blob or "").strip():
            return
        for m in re.finditer(r"\b([Nn][-\s]?[A-Za-z0-9]{1,6})\b", blob):
            add(m.group(1))

    scan_text(query or "")
    if history:
        for h in history[-16:]:
            role = (h.get("role") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            scan_text(h.get("content") or "")

    return out[:24]


def faa_internal_miss_context_block(tokens: List[str]) -> str:
    """
    When PhlyData has no row **and** our ingested ``faa_master`` snapshot has no row for the tail,
    instruct the answer model to lead on Tavily / web (public FAA-equivalent facts), not "unknown."
    """
    toks = ", ".join(str(x) for x in (tokens or [])[:16])
    return (
        "[NO INGESTED FAA MASTER ROW — Hye Aero internal faa_master snapshot]\n"
        f"Identifiers for this turn: {toks}.\n"
        "Our ingested **faa_master** table has **no** row for this registration in this environment "
        "(ingest lag, non-U.S. registry, or tail not yet loaded). **Public FAA / registry data may still exist** "
        "on the internet — you **must** use the **Tavily web results** and vector excerpts in this context when present. "
        "Lead with **substantive facts from Tavily snippets** (aircraft class, manufacturer/model class, year, serial, "
        "registered owner/operator when snippets support them) and cite **snippet #** and domain. "
        "Do **not** say make/model, year, or ownership are \"not available in the data gathered\" if any Tavily or "
        "vector line provides them. If web snippets are empty, say that clearly and still avoid implying the aircraft "
        "does not exist in public registries — suggest verifying on the FAA registry or flight-tracking sites.\n"
    )


def _should_attempt_faa_registration_lookup(raw: str) -> bool:
    """True if ``raw`` might be a U.S. tail worth querying ``faa_master`` (cheap filter)."""
    if registration_tail_canonical(raw):
        return True
    u = (raw or "").strip().upper().replace(" ", "")
    if u.startswith("N") and 4 <= len(u) <= 8:
        return True
    return False


def consultant_phly_lookup_token_list(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[str]:
    """
    Tokens for Phly SQL lookup: **current user message first**. If the user cites a new tail/serial,
    do not merge older thread tokens (prevents repeating N678PS when they ask about 000017).
    If the current message has no aircraft tokens, fall back to recent chat (e.g. "thanks", "and the hours?").
    """
    primary = extract_phlydata_lookup_tokens(query or "")
    if primary:
        return primary
    return extract_phlydata_tokens_with_history(query, history)


def consultant_user_asks_aircraft_master_table(query: str) -> bool:
    """User explicitly referenced the synced Postgres aircraft entity table."""
    q = (query or "").lower()
    needles = (
        "aircraft table",
        "public.aircraft",
        "the aircraft table",
        "in aircraft table",
        "aircraft_master",
        "master aircraft",
        "from aircraft",
        "in the aircraft ",
    )
    return any(n in q for n in needles)


def lookup_aircraft_master_rows(db: PostgresClient, tokens: List[str]) -> List[Dict[str, Any]]:
    """Match ``public.aircraft`` by **strict** TRIM+UPPER equality on registration and/or serial (hyphens literal)."""
    if not tokens:
        return []
    parts: List[str] = []
    params: List[Any] = []
    seen: set[str] = set()
    _trim_up = "TRIM(UPPER(COALESCE({col},'')))"

    def add_cond(kind: str, value: str) -> None:
        key = f"{kind}:{value}"
        if key in seen:
            return
        seen.add(key)
        if kind == "reg":
            parts.append(f"{_trim_up.format(col='registration_number')} = %s")
            params.append(value)
        elif kind == "sn":
            parts.append(f"{_trim_up.format(col='serial_number')} = %s")
            params.append(value)

    for raw in tokens:
        t = (raw or "").strip()
        if not t:
            continue
        tk = _phly_identity_key(t)
        if _token_is_tail_registration(t):
            add_cond("reg", tk)
            add_cond("sn", tk)
            continue
        if re.search(r"\d", t) and len(t) >= 3:
            for vn in _serial_token_match_variants(t):
                if len(vn) >= 2:
                    add_cond("sn", vn)

    if not parts:
        return []

    where_sql = " OR ".join(parts)
    try:
        rows = db.execute_query(
            f"""
            SELECT id,
                   serial_number,
                   registration_number,
                   manufacturer,
                   model,
                   manufacturer_year,
                   delivery_year,
                   category,
                   aircraft_status,
                   condition,
                   registration_country,
                   based_country
            FROM public.aircraft
            WHERE ({where_sql})
            ORDER BY updated_at DESC NULLS LAST
            LIMIT 15
            """,
            tuple(params),
        )
    except Exception as e:
        logger.warning("consultant: public.aircraft lookup failed: %s", e)
        return []

    if not rows:
        return []
    by_id: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        aid = r.get("id")
        if aid is not None and aid not in by_id:
            by_id[aid] = dict(r)
    return list(by_id.values())


def phly_like_row_from_aircraft_master(r: Dict[str, Any]) -> Dict[str, Any]:
    """Shape aircraft.id as aircraft_id for listing / Tavily filters when PhlyData has no row."""
    return {
        "aircraft_id": r.get("id"),
        "serial_number": r.get("serial_number"),
        "registration_number": r.get("registration_number"),
        "manufacturer": r.get("manufacturer"),
        "model": r.get("model"),
    }


def format_aircraft_master_consultant_block(rows: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Readable authority block for ``public.aircraft`` (not Phly export)."""
    lines: List[str] = [
        "",
        "[AUTHORITATIVE — Hye Aero aircraft master (PostgreSQL: public.aircraft)]",
        "This table is the synced **aircraft** entity used with listings/FAA ingest — **not** phlydata_aircraft (Phly export). "
        "When the user asks for status **in the aircraft table**, use **aircraft_status** below verbatim.",
        "",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(f"--- public.aircraft row {i} ---")
        lines.append(
            f"[FOR USER REPLY — public.aircraft — MANDATORY VERBATIM — row {i}]"
        )
        lines.append(
            "  Copy **aircraft_status** (and identity) exactly for questions about the aircraft table; "
            "do not substitute PhlyData or listing rows unless the user asks for those layers."
        )
        lines.append(f"  id: {r.get('id')}")
        lines.append(f"  serial_number: {(r.get('serial_number') or '').strip() or '(null)'}")
        lines.append(f"  registration_number: {(r.get('registration_number') or '').strip() or '(null)'}")
        mm = " ".join(
            x
            for x in (
                (r.get("manufacturer") or "").strip(),
                (r.get("model") or "").strip(),
            )
            if x
        )
        lines.append(f"  manufacturer / model: {mm or '(null)'}")
        y = r.get("manufacturer_year")
        if y is None:
            y = r.get("delivery_year")
        lines.append(f"  manufacturer_year: {y if y is not None else '(null)'}")
        lines.append(f"  category: {(r.get('category') or '').strip() or '(null)'}")
        st = (r.get("aircraft_status") or "").strip()
        lines.append(f"  aircraft_status: {st if st else '(null)'}")
        lines.append(f"  condition: {(r.get('condition') or '').strip() or '(null)'}")
        lines.append("")
    meta = {"aircraft_master_rows": len(rows)}
    return "\n".join(lines).rstrip() + "\n", meta


_N_TAIL_TOKEN = re.compile(r"^N[A-Z0-9]{1,6}$", re.IGNORECASE)


def _phly_identity_key(value: Any) -> str:
    """Phly / aircraft **identity** compare: trim + uppercase only — **hyphens and digits are literal** (LJ-1682 ≠ LJ1682)."""
    return str(value if value is not None else "").strip().upper()


def _token_is_tail_registration(raw: str) -> bool:
    """
    Shape heuristic for hyphenated / N-tail tokens (routing to text fallback when reg/sn do not match).
    **Equality** always uses ``_phly_identity_key`` — never hyphen-stripped forms.
    """
    t = (raw or "").strip()
    if not t:
        return False
    u = re.sub(r"\s+", "", t.upper())
    # Compact U.S. N-number without hyphen (N682TM)
    if _N_TAIL_TOKEN.fullmatch(u):
        return True
    # Explicit hyphen after 1–3 letter mark (G-CIVG, V-682, XA-98723, N-682TM, LJ-1682)
    if re.fullmatch(r"[A-Z]{1,3}-[A-Z0-9]{2,16}", u):
        return True
    # Letters then digit, no hyphen in token (e.g. XA98723) — not pure numeric MSNs like 5250682
    if re.fullmatch(r"[A-Z]{1,3}\d[A-Z0-9]{0,11}", u) and not re.fullmatch(r"\d+", u):
        return True
    return False


def _serial_token_match_variants(raw: str) -> set[str]:
    """Single strict identity key for Phly token (hyphen-preserving)."""
    s = _phly_identity_key(raw or "")
    return {s} if s else set()


def _phly_row_text_contains_token(r: Dict[str, Any], token: str) -> bool:
    """True when manufacturer, model, category, or features contain the **exact** token spelling (trim+upper; hyphens kept)."""
    t = (token or "").strip()
    if len(t) < 3:
        return False
    parts: List[str] = []
    for key in ("manufacturer", "model", "category", "features"):
        v = r.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    blob = " ".join(parts).upper()
    if not blob.strip():
        return False
    needle = t.upper()
    blob_nf = re.sub(r"\s+", " ", blob)
    if len(needle) >= 3 and needle in blob_nf:
        return True
    return False


def _token_matches_phly_row(r: Dict[str, Any], token: str) -> bool:
    """
    One user token matches this row: **strict** ``registration_number`` / ``serial_number`` equality (hyphen-preserving),
    or manufacturer/model/category/features substring match with the same spelling.
    """
    t = (token or "").strip()
    if not t or len(t) < 3:
        return True
    reg_k = _phly_identity_key(r.get("registration_number"))
    sn_k = _phly_identity_key(r.get("serial_number"))
    tok_k = _phly_identity_key(t)
    u = re.sub(r"\s+", "", t.upper())

    if _N_TAIL_TOKEN.fullmatch(u):
        return reg_k == tok_k

    if tok_k and (reg_k == tok_k or sn_k == tok_k):
        return True

    if _token_is_tail_registration(t):
        return _phly_row_text_contains_token(r, t)

    if re.search(r"\d", t) and len(t) >= 3:
        return _phly_row_text_contains_token(r, t)

    return _phly_row_text_contains_token(r, t)


def _phly_rows_match_consultant_tokens(rows: List[Dict[str, Any]], tokens: List[str]) -> List[Dict[str, Any]]:
    """Keep rows where **every** token (len≥3) matches via ``_token_matches_phly_row`` (AND across tokens)."""
    if not rows or not tokens:
        return rows
    meaningful = [x.strip() for x in tokens if (x or "").strip() and len((x or "").strip()) >= 3]
    if not meaningful:
        return rows
    narrowed: List[Dict[str, Any]] = []
    for r in rows:
        if all(_token_matches_phly_row(r, tok) for tok in meaningful):
            narrowed.append(r)
    return narrowed


def ilike_patterns_for_token(token: str) -> List[str]:
    """ILIKE patterns for recall — **hyphens preserved** (no collapsed ``HA420`` variant for ``HA-420``)."""
    u = token.strip()
    if not u:
        return []
    pats: List[str] = []
    for variant in {u, u.upper(), u.lower()}:
        if variant and f"%{variant}%" not in pats:
            pats.append(f"%{variant}%")
    seen: set[str] = set()
    uniq: List[str] = []
    for p in pats:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq[:8]


def lookup_phlydata_aircraft_rows(db: PostgresClient, tokens: List[str]) -> List[Dict[str, Any]]:
    """Return matching ``phlydata_aircraft`` rows: serial, tail, **and** make/model/category/features text."""
    if not tokens:
        return []
    all_pats: List[str] = []
    for t in tokens:
        all_pats.extend(ilike_patterns_for_token(t))
    all_pats = list(dict.fromkeys(all_pats))[:28]

    _tu_sn = "TRIM(UPPER(COALESCE(serial_number,'')))"
    _tu_reg = "TRIM(UPPER(COALESCE(registration_number,'')))"
    norm_parts: List[str] = []
    norm_params: List[Any] = []
    seen_n: set[str] = set()
    for t in tokens[:14]:
        for v in _serial_token_match_variants(t):
            if len(v) < 2 or v in seen_n:
                continue
            seen_n.add(v)
            norm_parts.append(f"({_tu_sn} = %s OR {_tu_reg} = %s)")
            norm_params.extend([v, v])
    norm_parts = norm_parts[:36]

    text_parts: List[str] = []
    text_params: List[Any] = []
    for p in all_pats[:24]:
        text_parts.append(
            "("
            "UPPER(COALESCE(manufacturer,'')) LIKE UPPER(%s) OR "
            "UPPER(COALESCE(model,'')) LIKE UPPER(%s) OR "
            "UPPER(COALESCE(category,'')) LIKE UPPER(%s) OR "
            "UPPER(COALESCE(features,'')) LIKE UPPER(%s)"
            ")"
        )
        text_params.extend([p, p, p, p])

    ilike_clause = (
        "(serial_number ILIKE ANY(%s) OR registration_number ILIKE ANY(%s))"
        if all_pats
        else ""
    )

    chunks: List[str] = []
    params_list: List[Any] = []
    if all_pats:
        chunks.append(ilike_clause)
        params_list.extend([all_pats, all_pats])
    if text_parts:
        chunks.append("(" + " OR ".join(text_parts) + ")")
        params_list.extend(text_params)
    if norm_parts:
        chunks.append("(" + " OR ".join(norm_parts) + ")")
        params_list.extend(norm_params)

    if not chunks:
        return []

    where_body = "(" + " OR ".join(chunks) + ")"
    params = tuple(params_list)

    try:
        rows = db.execute_query(
            f"""
            SELECT aircraft_id,
            {phlydata_aircraft_select_sql(include_cast_id=False, db=db)}
            FROM public.phlydata_aircraft
            WHERE {where_body}
            LIMIT 75
            """,
            params,
        )
    except Exception as e:
        logger.warning("phlydata_consultant_lookup: query failed (%s); skipping direct path.", e)
        return []
    if not rows:
        return []
    rows = _phly_rows_match_consultant_tokens(rows, tokens)
    # Prefer exact tail/serial, then identity hit, then text-only
    def _rank(r: Dict[str, Any]) -> tuple:
        reg = _phly_identity_key(r.get("registration_number"))
        sn = _phly_identity_key(r.get("serial_number"))
        tail_hit = 0
        serial_hit = 0
        text_hit = 0
        for raw in tokens:
            tr = (raw or "").strip()
            trk = _phly_identity_key(tr)
            tu = re.sub(r"\s+", "", (tr or "").upper())
            if _N_TAIL_TOKEN.fullmatch(tu) and reg == trk:
                tail_hit = 1
            elif _token_is_tail_registration(tr) and (reg == trk or sn == trk):
                tail_hit = 1
            variants = _serial_token_match_variants(raw or "")
            if sn and sn in variants:
                serial_hit = 1
            if reg == trk or sn == trk:
                serial_hit = max(serial_hit, 1)
            if _phly_row_text_contains_token(r, tr):
                text_hit = 1
        return (-tail_hit, -serial_hit, -text_hit)

    rows = sorted(rows, key=_rank)
    by_id: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        aid = r.get("aircraft_id")
        if aid is not None and aid not in by_id:
            by_id[aid] = dict(r)
    return list(by_id.values())[:12]


def _faa_master_aircraft_identity_lines(fr: Dict[str, Any]) -> List[str]:
    """FAA MASTER aircraft fields (not Phly) — make/model/year context when PhlyData has no row."""
    out: List[str] = []
    nn = (fr.get("n_number") or "").strip()
    sn = (fr.get("serial_number") or "").strip()
    mcode = (fr.get("mfr_mdl_code") or "").strip()
    ref_model = (fr.get("faa_reference_model") or "").strip()
    yr = fr.get("year_mfr")
    ta = (fr.get("type_aircraft") or "").strip()
    te = (fr.get("type_engine") or "").strip()
    cert = (fr.get("certification") or "").strip()
    stc = (fr.get("status_code") or "").strip()
    if nn:
        out.append(f"- FAA n_number (as in MASTER): {nn}")
    if sn:
        out.append(f"- FAA serial_number (as in MASTER): {sn}")
    if mcode:
        out.append(f"- FAA mfr_mdl_code: {mcode}")
    if ref_model:
        out.append(f"- FAA aircraft reference model (ACFTREF decode): {ref_model}")
    if yr is not None:
        out.append(f"- FAA year_mfr: {yr}")
    if ta:
        out.append(f"- FAA type_aircraft: {ta}")
    if te:
        out.append(f"- FAA type_engine: {te}")
    if cert:
        out.append(f"- FAA certification: {cert}")
    if stc:
        out.append(f"- FAA status_code: {stc}")
    return out


def synthetic_phly_row_from_faa_master(fr: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal ``phlydata_aircraft``-shaped row so downstream consultant paths (Tavily, listing SQL)
    can anchor tail/serial when only **faa_master** matched.
    """
    nn = (fr.get("n_number") or "").strip()
    reg = nn.upper() if nn else ""
    if reg and not reg.startswith("N") and len(reg) <= 6:
        reg = f"N{reg}"
    ref = (fr.get("faa_reference_model") or "").strip()
    sn = (fr.get("serial_number") or "").strip()
    yr = fr.get("year_mfr")
    mdl = ref or (fr.get("mfr_mdl_code") or "").strip()
    return {
        "registration_number": reg,
        "serial_number": sn,
        "manufacturer": "",
        "model": mdl,
        "manufacturer_year": yr,
        "category": "",
    }


def sort_tokens_faa_priority(tokens: List[str]) -> List[str]:
    """
    Order tokens so U.S. N-numbers are tried first in ``faa_master_standalone_authority_for_tokens``.
    Long numeric MSNs or years first in the list can waste attempts and delay tail matches.
    """
    if not tokens:
        return []
    scored: List[tuple[int, int, str]] = []
    for i, t in enumerate(tokens):
        u = re.sub(r"\s+", "", (t or "").strip())
        if _N_TAIL_TOKEN.fullmatch(u):
            pri = 0
        elif _token_is_tail_registration(t):
            pri = 1
        elif u.startswith(("N", "n")) and len(u) >= 4:
            pri = 2
        else:
            pri = 3
        scored.append((pri, i, t))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in scored]


def consultant_merge_lookup_tokens(
    query: str,
    history: Optional[List[Dict[str, str]]],
    faa_lookup_tokens: Optional[List[str]] = None,
) -> List[str]:
    """
    Merge token lists used for Phly SQL, FAA scan, and Tavily enrichment so tails cited only in
    history (or duplicated across paths) stay anchored on one identifier list.
    """
    base = consultant_phly_lookup_token_list(query, history)
    extra = [x for x in (faa_lookup_tokens or []) if (x or "").strip()]
    return list(dict.fromkeys([*base, *extra]))


def _faa_registration_lookup_variants(raw: str) -> List[str]:
    """
    Try multiple string forms for ``fetch_faa_master_owner_rows`` tail matching
    (CSV may store ``N448SJ``, ``448SJ``, or mixed case).
    """
    t = (raw or "").strip()
    if not t:
        return []
    seen: set[str] = set()
    out: List[str] = []

    def add(x: str) -> None:
        s = (x or "").strip()
        if not s or s in seen:
            return
        seen.add(s)
        out.append(s)

    u = t.upper().replace(" ", "")
    add(t)
    add(u)
    add(u.replace("-", ""))
    # US civil: ensure N-prefix form when user typed short form (e.g. 448SJ → N448SJ)
    if re.match(r"^[A-Z0-9]{2,6}$", u) and not u.startswith("N"):
        add(f"N{u}")
    if u.startswith("N") and len(u) > 2:
        add(u[1:])
    return out[:8]


def faa_master_standalone_authority_for_tokens(
    db: PostgresClient,
    tokens: List[str],
    fetch_faa_master,
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    When **phlydata_aircraft** has no matching row, still look up **faa_master** using the user's
    registration / tail tokens (e.g. ``N448SJ``) so the consultant receives verbatim FAA registrant
    lines — same source as when Phly rows exist.

    Also emits **faa_aircraft_reference** / MASTER identity fields (make/model class, year, serial)
    so the answer LLM does not claim "unknown" when the registry row exists.

    Returns ``(authority_text, meta, first_faa_row_or_none)`` for downstream synthetic Phly-shaped rows.
    """
    if not tokens:
        return "", {}, None
    tried: set[str] = set()
    for raw in sort_tokens_faa_priority(tokens)[:24]:
        t = (raw or "").strip()
        if not t:
            continue
        # Prefer tokens that can map to a US registry tail; still try variants for robustness.
        variants = _faa_registration_lookup_variants(t)
        for reg_try in variants:
            if reg_try in tried:
                continue
            if not _should_attempt_faa_registration_lookup(reg_try):
                continue
            tried.add(reg_try)
            try:
                faa_rows, kind = fetch_faa_master(
                    db,
                    serial="",
                    model=None,
                    registration=reg_try,
                )
            except Exception as e:
                logger.debug("faa_master standalone lookup: %s", e)
                continue
            if not faa_rows:
                continue
            fr = faa_rows[0]
            rn = (fr.get("registrant_name") or "").strip()
            st1 = (fr.get("street") or "").strip()
            st2 = (fr.get("street2") or "").strip()
            street_combined = " ".join(x for x in (st1, st2) if x).strip()
            city = (fr.get("city") or "").strip()
            st = (fr.get("state") or "").strip()
            z = (fr.get("zip_code") or "").strip()
            ctry = (fr.get("country") or "").strip()
            loc = ", ".join(x for x in (city, st, z, ctry) if x)

            lines: List[str] = [
                "[AUTHORITATIVE — FAA MASTER (faa_master) — no internal aircraft record; FAA snapshot only]",
                "Hye Aero's **internal aircraft record** has **no** row for this identifier; the following lines are from our "
                "ingested **FAA MASTER** snapshot only (U.S. civil registry). "
                "Use **every** non-empty FAA line below for your answer: aircraft identity (reference model, year, type) "
                "and U.S. legal registrant — do **not** say make/model or ownership is unknown or not in the data when "
                "those lines are present. Supplement with Tavily/vector only for operator, fleet, or market color not in MASTER. "
                "In the **client-facing** answer, do **not** mention \"internal export row\" or similar engineering phrasing.",
                "",
                f"- Registration token matched: {reg_try} (from query token {raw!r})",
                f"- FAA match kind: {kind or 'unknown'}",
                "",
                "[FAA aircraft identity from MASTER — use when PhlyData absent]",
            ]
            for id_line in _faa_master_aircraft_identity_lines(fr):
                lines.append(id_line)
            lines.append("")
            if rn:
                lines.append(f"- FAA MASTER registrant (faa_master): {rn}")
            if street_combined:
                lines.append(f"- FAA mailing street: {street_combined}")
            if loc:
                lines.append(f"- FAA mailing location: {loc}")
            ref_model = (fr.get("faa_reference_model") or "").strip()
            if ref_model or fr.get("year_mfr") is not None:
                lines.extend(
                    [
                        "",
                        "[FOR USER REPLY — FAA aircraft type / year (faa_master) — when PhlyData absent]",
                        "  If the FAA identity lines above include **faa_reference_model** or **year_mfr**, state aircraft class / year from those lines.",
                        "  Do **not** claim make/model or year are unavailable when those fields appear above.",
                    ]
                )
            if rn:
                lines.extend(
                    [
                        "",
                        "[FOR USER REPLY — U.S. legal registrant (FAA MASTER) — MANDATORY VERBATIM]",
                        "  Source: FAA MASTER (registrant/address). State name and mailing verbatim — do NOT replace with web guesses.",
                        f"  Registrant name: {rn}",
                    ]
                )
                if street_combined:
                    lines.append(f"  Mailing street: {street_combined}")
                if loc:
                    lines.append(f"  Mailing city/state/ZIP/country: {loc}")
            meta = {
                "faa_master_owner_rows": 1,
                "faa_master_match_kind": kind,
                "faa_standalone_from_tokens": 1,
            }
            return "\n".join(lines), meta, fr

    logger.info(
        "faa_master_standalone: no FAA row matched for tokens (sample): %s",
        [str(x) for x in tokens[:12]],
    )
    return "", {}, None


def format_phlydata_consultant_answer(
    db: PostgresClient,
    rows: List[Dict[str, Any]],
    fetch_faa_master,
    *,
    registry_sql_enabled: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build plain-text answer and ``data_used`` summary.
    ``fetch_faa_master`` is ``services.faa_master_lookup.fetch_faa_master_owner_rows``.

    When ``registry_sql_enabled`` is False, Phly identity/snapshot lines are still emitted but
    **faa_master** is not queried (intent-gated registry path).
    """
    lines: List[str] = []
    lines.append(
        "PhlyData — Hye Aero's internal aircraft record (PostgreSQL table **phlydata_aircraft** only). "
        "Identity, **aircraft_status** (for-sale / disposition from the internal export), **ask_price** / take / sold, "
        "and other export columns below are authoritative for this layer — not **aircraft_listings** (marketplace ingest)."
    )
    lines.append(
        "Evaluation order: When answering, ground and lead on PhlyData (this block) for identity and all snapshot fields below. "
        "Controller / exchanges / aircraft_listings / Tavily / vector DB are supplemental — never override PhlyData internal fields with them; "
        "if they disagree, state PhlyData first, then Separately, … for the other source. "
        "Snapshots can still be stale; for live availability and contracts, say verify with broker/platform. "
        + (
            "Registered owner names are not in PhlyData — they appear only in the FAA MASTER line below when present; "
            "otherwise use Tavily / vector context and label sources clearly."
            if registry_sql_enabled
            else "U.S. legal registrant lines from **faa_master** were not loaded for this query type — use Tavily/vector for ownership when needed and label sources."
        )
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
        _append_phly_mandatory_verbatim_reply(lines, r, i)
        _append_phly_internal_snapshot_lines(lines, r)
        if (r.get("aircraft_status") or "").strip() or (r.get("ask_price") is not None):
            lines.append(
                "  [LLM: **phlydata_aircraft** has no separate transaction_status in the standard internal export — "
                "use **aircraft_status** for whether the export shows for sale / disposition, and **ask_price** for the "
                "internal asking price. Quote those fields exactly as printed; do not substitute listing-ingest status. "
                "If **ask_price** (or take/sold) appears above, do not tell the user there is no internal asking price.]"
            )

        serial = str(r.get("serial_number") or "").strip()
        reg_s = str(r.get("registration_number") or "").strip() or None
        mdl_q = str(r.get("model") or "").strip() or None
        faa_rows: List[Dict[str, Any]] = []
        if registry_sql_enabled:
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
        elif registry_sql_enabled:
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
        else:
            lines.append(
                "  - FAA MASTER: not queried for this turn (registry SQL disabled for this intent). "
                "Use Tavily/vector for U.S. legal registrant or operator context when relevant."
            )

    data_used = {
        "phlydata_aircraft_rows": len(rows),
        "faa_master_owner_rows": faa_hits,
    }
    return "\n".join(lines), data_used


def synthetic_phyl_like_rows_from_tokens(lookup_tokens: Optional[List[str]]) -> List[Dict[str, Any]]:
    """
    When PhlyData matched nothing, still give Tavily enrichment a minimal ``registration_number``
    (and optional identity) so searches are anchored on the user's tail/serial tokens.
    """
    if not lookup_tokens:
        return []
    for t in lookup_tokens[:8]:
        u = re.sub(r"\s+", "", (t or "").strip())
        if u and _N_TAIL_TOKEN.fullmatch(u):
            return [{"registration_number": u.upper(), "serial_number": "", "manufacturer": "", "model": ""}]
    for t in lookup_tokens[:8]:
        if _token_is_tail_registration(t):
            return [{"registration_number": _phly_identity_key(t), "serial_number": "", "manufacturer": "", "model": ""}]
    return []


def enrich_tavily_query_for_consultant(
    user_query: str,
    base_tavily_query: str,
    phly_rows: List[Dict[str, Any]],
    history_snippet: Optional[str] = None,
    lookup_tokens: Optional[List[str]] = None,
) -> str:
    """
    When the user asks about ownership and we matched PhlyData rows, append tail/serial/model
    plus owner/operator terms so Tavily returns registry and fleet pages (not just the raw user phrase).

    When PhlyData matched **no** rows but ``lookup_tokens`` contains a tail (e.g. ``N448SJ``), we still
    append those identifiers so Tavily is not left with a generic phrase and empty context.
    """
    rows_use: List[Dict[str, Any]] = list(phly_rows or [])
    if not rows_use and lookup_tokens:
        rows_use = synthetic_phyl_like_rows_from_tokens(lookup_tokens)

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
    should_enrich = wants_owner or (not phly_rows and bool(rows_use))
    if not should_enrich or not rows_use:
        return (base_tavily_query or user_query)[:500]

    parts: List[str] = [(base_tavily_query or user_query).strip()]
    for r in rows_use[:2]:
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
