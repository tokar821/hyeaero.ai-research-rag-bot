"""
Deterministic listings + sales context for Ask Consultant (purchase / price / availability).

Pinecone may miss the latest ask price; this queries ``aircraft_listings`` and ``aircraft_sales``
directly when the user asks about buying, price, or availability for a known serial/tail.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from database.postgres_client import PostgresClient
from rag.phlydata_consultant_lookup import (
    extract_phlydata_tokens_with_history,
    ilike_patterns_for_token,
)

logger = logging.getLogger(__name__)


def _user_only_history_blob(
    history: Optional[List[Dict[str, str]]],
    *,
    max_messages: int = 10,
) -> str:
    """Concatenate recent **user** turns only — never assistant text (avoids false purchase intent)."""
    if not history:
        return ""
    parts: List[str] = []
    for h in history[-max_messages:]:
        if (h.get("role") or "").strip().lower() != "user":
            continue
        c = (h.get("content") or "").strip()
        if c:
            parts.append(c)
    return " ".join(parts)


def wants_consultant_purchase_market_context(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """True when the user likely cares about price, listing, or ability to purchase.

    Uses **user messages only** (plus the current query). Assistant replies often say "listing",
    "available", or "asking price" and would wrongly flip follow-ups like "U" into purchase mode.
    """
    blob = f"{_user_only_history_blob(history)} {user_query or ''}".strip().lower()
    keywords = (
        "buy",
        "purchase",
        "acquire",
        "for sale",
        "on the market",
        "listing",
        "listed",
        "available",
        "on sale",
        "price",
        "pricing",
        "how much",
        "cost",
        "asking",
        "ask price",
        "offer",
        "worth",
        "market value",
        "valuation",
        "pay",
        "budget",
        "seller",
        "broker",
        "deal",
        "negotiate",
        "can i buy",
        "could i buy",
        "i can buy",
        "buy it",
        "buy now",
        "is it available",
        "still available",
        "still for sale",
        "get a deal",
        "make an offer",
    )
    return any(k in blob for k in keywords)


def wants_consultant_strict_internal_market_sql(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """Stricter than :func:`wants_consultant_purchase_market_context` for Postgres market SQL.

    Avoids firing on vague substrings like bare ``available`` or ``listed`` (e.g. unrelated wording).
    Use with ``CONSULTANT_MARKET_SQL_STRICT=1``.
    """
    blob = f"{_user_only_history_blob(history)} {user_query or ''}".strip().lower()
    if not blob.strip():
        return False
    phrases = (
        "for sale",
        "on sale",
        "on the market",
        "listing",
        "listings",
        "publicly listed",
        "public listing",
        "still for sale",
        "is it available",
        "still available",
        "can i buy",
        "could i buy",
        "i can buy",
        "buy it",
        "buy now",
        "purchase",
        "acquire",
        "how much",
        "asking price",
        "ask price",
        "market value",
        "valuation",
        "appraisal",
        "comparable sale",
        "comparable sales",
        "recent sale",
        "recent sales",
        "make an offer",
        "negotiate",
        "get a deal",
        "seller",
        "broker",
        "budget",
        "afford",
    )
    if any(p in blob for p in phrases):
        return True
    if re.search(
        r"\b(buy|purchase|pricing|sold|asking|listed|listing|listings|offer|cost|pay|worth|ask)\b",
        blob,
    ):
        return True
    if re.search(r"\bprice\b", blob):
        return True
    if re.search(r"\bcomps?\b", blob):
        return True
    return False


def consultant_wants_internal_market_sql(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    strict: bool = False,
) -> bool:
    """Whether to run internal listings/comps SQL and related purchase-biased retrieval."""
    if strict:
        return wants_consultant_strict_internal_market_sql(user_query, history)
    return wants_consultant_purchase_market_context(user_query, history)


def _consultant_detail_view_phrases(blob_lc: str) -> bool:
    """Detail/show-me/overview/photo wording (excludes tail-only token detection)."""
    if not blob_lc.strip():
        return False
    phrases = (
        "detail",
        "details",
        "information",
        "tell me",
        "show me",
        "overview",
        "summary",
        "who owns",
        "ownership",
        "operator",
        "specs",
        "specifications",
        "describe",
        "background",
        "profile",
        "picture",
        "pictures",
        "photos",
        "images",
        "photo",
    )
    if any(k in blob_lc for k in phrases):
        return True
    if re.search(r"\b(info|about)\b", blob_lc):
        return True
    return False


def wants_consultant_explicit_photo_web(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """User asked for photos / images / gallery (web), not only narrative detail."""
    blob = f"{_user_only_history_blob(history)} {user_query or ''}".strip().lower()
    if not blob.strip():
        return False
    keys = (
        "image",
        "images",
        "photo",
        "photos",
        "photograph",
        "picture",
        "pictures",
        "gallery",
    )
    return any(k in blob for k in keys)


def wants_consultant_aircraft_images_in_answer(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """True when the UI should show the aircraft image gallery — tall/research-only turns stay text-only.

    Triggers on explicit photo/image requests or detail/deep-dive phrasing (not bare tail, not price-only).
    """
    if wants_consultant_explicit_photo_web(user_query, history):
        return True
    blob = f"{_user_only_history_blob(history)} {user_query or ''}".strip().lower()
    if not blob.strip():
        return False
    if re.search(
        r"\b(details?|in[-\s]depth|overview|describe|description|specifications?|specs|"
        r"full briefing|background on|profile of|walk-?through)\b",
        blob,
    ):
        return True
    if re.search(r"\b(tell me (more )?about|more about|information about)\b", blob):
        return True
    if re.search(r"\b(please show|show me (some )?details?|show me detail)\b", blob):
        return True
    if re.search(
        r"\b(any|some|do you have|have you|got)\s+(photos?|images?|pictures?)\b",
        blob,
    ):
        return True
    if re.search(r"\b(see (the )?(photos?|images?)|look at (the )?(photos?|images?))\b", blob):
        return True
    return False


def wants_consultant_aircraft_detail_phrases(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """True when the user explicitly asked for narrative/detail/visual wording (not tail-only)."""
    blob = f"{_user_only_history_blob(history)} {user_query or ''}".strip().lower()
    if _consultant_detail_view_phrases(blob):
        return True
    return wants_consultant_explicit_photo_web(user_query, history)


def wants_consultant_aircraft_detail_context(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """True for general aircraft questions (detail, overview, photos) or tail/serial in query/history."""
    blob = f"{_user_only_history_blob(history)} {user_query or ''}".strip().lower()
    if not blob.strip():
        return False
    if _consultant_detail_view_phrases(blob):
        return True
    return bool(extract_phlydata_tokens_with_history(user_query, history))


def consultant_wants_internal_listings_sql(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None,
    phly_rows: Optional[List[Dict[str, Any]]] = None,
    *,
    strict: bool = False,
) -> bool:
    """Run Postgres listing/comps SQL for purchase-style questions, resolved Phly identity, or detail/tail queries."""
    if consultant_wants_internal_market_sql(user_query, history, strict=strict):
        return True
    if phly_rows:
        return True
    return wants_consultant_aircraft_detail_context(user_query, history)


def build_aircraft_photo_focus_tavily_query(
    query: str,
    phly_rows: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """
    Second-pass Tavily string biased toward **image** search hits (use with ``include_images``).

    CDN image URLs rarely contain the tail; pairing this query with a relaxed identity filter
    improves recall vs. generic web search.
    """
    reg = ""
    serial = ""
    for r in phly_rows[:4]:
        rv = (r.get("registration_number") or "").strip()
        if rv:
            reg = re.sub(r"[\s\-]+", "", rv.upper())
            break
    for r in phly_rows[:4]:
        sv = (r.get("serial_number") or "").strip()
        if sv:
            serial = sv
            break
    if not reg:
        for t in extract_phlydata_tokens_with_history(query, history):
            u = (t or "").strip().upper().replace(" ", "").replace("-", "")
            if re.match(r"^N[A-Z0-9]{1,6}$", u):
                reg = u
                break
    if not serial:
        for t in extract_phlydata_tokens_with_history(query, history):
            tv = (t or "").strip()
            if tv and re.search(r"\d", tv) and len(tv) >= 4 and not re.match(
                r"^N[A-Z0-9]{1,6}$", tv.upper().replace(" ", "").replace("-", "")
            ):
                serial = tv
                break
    parts: List[str] = []
    if reg:
        parts.append(f'"{reg}"')
    elif serial:
        parts.append(f'"{serial}"')
    else:
        return None
    # Phrasing close to what users type in Google (e.g. "n807js images") — helps Tavily return image assets.
    parts.extend(
        [
            "aircraft",
            "images",
            "photos",
            "JetPhotos",
            "exterior",
            "cabin",
            "interior",
        ]
    )
    q = " ".join(parts)
    return q[:420] if q else None


def _listing_where_from_phly_rows(
    phly_rows: List[Dict[str, Any]],
) -> Tuple[List[str], List[Any]]:
    """Extra OR-clauses so listings match hyphenated serials and aligned aircraft UUIDs.

    ILIKE ``%5605354%`` does not match ``aircraft.serial_number`` = ``560-5354``; normalize in SQL.
    When ``phlydata_aircraft.aircraft_id`` equals ``aircraft.id``, match listings directly.
    """
    cond_parts: List[str] = []
    params_l: List[Any] = []
    seen_norm: set[str] = set()
    for r in phly_rows[:10]:
        sn = (r.get("serial_number") or "").strip()
        if sn:
            collapsed = re.sub(r"[^0-9A-Za-z]", "", sn).upper()
            if len(collapsed) >= 4 and collapsed not in seen_norm:
                seen_norm.add(collapsed)
                cond_parts.append(
                    "REGEXP_REPLACE(UPPER(COALESCE(a.serial_number,'')), '[^A-Z0-9]', '', 'g') LIKE %s"
                )
                params_l.append(f"%{collapsed}%")
    uuids: List[str] = []
    for r in phly_rows[:10]:
        aid = r.get("aircraft_id")
        if aid is None:
            continue
        s = str(aid).strip()
        if s and s not in uuids:
            uuids.append(s)
    if uuids:
        cond_parts.append("a.id = ANY(%s::uuid[])")
        params_l.append(uuids)
    return cond_parts, params_l


def _collect_ilike_patterns(
    query: str,
    history: Optional[List[Dict[str, str]]],
    phly_rows: List[Dict[str, Any]],
) -> List[str]:
    seen: List[str] = []
    uq: set[str] = set()

    def add_pat(p: str) -> None:
        p = (p or "").strip()
        if p and p not in uq and len(seen) < 32:
            uq.add(p)
            seen.append(p)

    for t in extract_phlydata_tokens_with_history(query, history):
        for p in ilike_patterns_for_token(t)[:6]:
            add_pat(p)
    for r in phly_rows[:4]:
        for key in ("serial_number", "registration_number"):
            v = (r.get(key) or "").strip()
            if v:
                for p in ilike_patterns_for_token(v)[:5]:
                    add_pat(p)
    return seen


def _availability_guidance_for_listing_status(status: Optional[str]) -> str:
    """
    Short instruction for the LLM so it does not over-claim "available" or "for sale"
    from internal Postgres rows alone.
    """
    s = (status or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not s or s in ("—", "unknown", "null"):
        return (
            "LLM: treat as Hye Aero listing snapshot only (not PhlyData) — do not say the aircraft is currently for sale; "
            "say no clear active status in listing data or confirm with a broker."
        )
    if s in ("for_sale", "forsale", "active", "listed", "available", "on_market", "onmarket"):
        return (
            "LLM: Hye Aero listing record shows for-sale-style status — synced snapshot, not live proof (not PhlyData). "
            "Say 'Per Hye Aero listing records…' and require confirmation on the platform or with the seller; never guarantee availability."
        )
    if s in (
        "sold",
        "closed",
        "withdrawn",
        "off_market",
        "inactive",
        "expired",
        "removed",
        "cancelled",
        "canceled",
        "completed",
    ):
        return (
            "LLM: NOT an active public listing — historical or off-market per this Hye Aero listing row (not PhlyData). "
            "Say clearly it is not currently offered for sale on this listing record."
        )
    return (
        f"LLM: status is '{(status or '').strip()}' — listing snapshot only (not PhlyData); "
        "do not present as confirmed active availability; use 'Per Hye Aero listing records…' and suggest verification."
    )


def _fmt_money(v: Any) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
        if x >= 1_000_000:
            return f"${x:,.0f}"
        return f"${x:,.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(v)


def _phly_snapshot_includes_ask(phly_rows: List[Dict[str, Any]]) -> bool:
    """True when PhlyData row(s) carry a positive numeric ask/take usable as internal snapshot."""
    for r in phly_rows or []:
        for key in ("ask_price", "take_price"):
            v = r.get(key)
            if v is None:
                continue
            try:
                if float(v) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def build_consultant_market_authority_block(
    db: PostgresClient,
    query: str,
    history: Optional[List[Dict[str, str]]],
    phly_rows: List[Dict[str, Any]],
    *,
    strict_market_sql: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns ``(text_block, meta)`` for prepending to consultant context.
    ``meta`` counts rows used (for ``data_used``).

    Skips listing SQL unless the user message(s) look market-related **or** PhlyData resolved an
    aircraft **or** the query looks like a detail/overview/tail question; set
    ``strict_market_sql=True`` (``CONSULTANT_MARKET_SQL_STRICT=1``) for a narrower purchase-only trigger.
    """
    meta: Dict[str, Any] = {"consultant_internal_listings": 0, "consultant_internal_sales_comps": 0}
    if not consultant_wants_internal_listings_sql(
        query, history, phly_rows, strict=strict_market_sql
    ):
        return "", meta

    pats = _collect_ilike_patterns(query, history, phly_rows)
    # Collapsed serial variants (525-0444 → 5250444) help match listing URLs that omit hyphens.
    extra: List[str] = []
    seen_inner: set[str] = set()
    for pat in list(pats):
        inner = pat.strip("%").strip()
        if inner.lower() in seen_inner:
            continue
        seen_inner.add(inner.lower())
        collapsed = re.sub(r"[^0-9A-Za-z]", "", inner)
        if len(collapsed) >= 5 and collapsed.lower() != inner.lower():
            ep = f"%{collapsed}%"
            if ep not in pats and ep not in extra:
                extra.append(ep)
    pats = list(dict.fromkeys(pats + extra))[:36]

    cond_parts: List[str] = []
    params_l: List[Any] = []
    for p in pats:
        cond_parts.append(
            "(a.serial_number ILIKE %s OR a.registration_number ILIKE %s OR l.listing_url ILIKE %s)"
        )
        params_l.extend([p, p, p])
    extra_parts, extra_params = _listing_where_from_phly_rows(phly_rows)
    cond_parts.extend(extra_parts)
    params_l.extend(extra_params)
    if not cond_parts:
        return "", meta

    lines: List[str] = [
        "[Hye Aero listing & sales data — supplemental marketplace ingest (Postgres: aircraft_listings + aircraft_sales) — NOT PhlyData]",
        "**Policy:** The **PhlyData** authority block (if present) is Hye Aero's **canonical internal record** — lead evaluation there for identity and internal snapshot fields. **This block is supplemental:** synced marketplace/sales ingests (Controller, exchanges, etc.) — not live, may be stale/sold/withdrawn, and **must not override** PhlyData internal ask/status/sold lines when both exist.",
        "CRITICAL — never imply the aircraft is 'available', 'on the market', or 'you can buy it now' unless:",
        "  (1) listing status + tail/serial alignment support it, AND",
        "  (2) you frame it as **'Separately, per Hye Aero listing records (marketplace ingest; not PhlyData)…'** and tell the user to verify with the platform/broker.",
        "Distinguish in your answer:",
        "  - **PhlyData internal snapshot** (other block) vs **listing-ingest row** (this block) — never call listing rows PhlyData.",
        "  - **Active public listing (unverified):** only when listing row supports it — still say confirm before acting.",
        "  - **Off-market / historical listing record:** sold, withdrawn, or unclear — say not actively listed **in this listing-ingest data**.",
        "  - **No matching listing row:** say no row for this tail/serial in **Hye Aero listing records** (and web if relevant).",
        "Use this block for **asking price**, **sold price**, platform, seller, URLs, **listing_status** from ingest. Amounts as stored (typically USD).",
        "If the user asks whether they can buy now: after PhlyData internal lines (if any), use listing-row interpretation (per-row LLM notes), then web; never overstate certainty.",
    ]

    listings_out: List[Dict[str, Any]] = []
    try:
        where_sql = " OR ".join(cond_parts)
        listings_out = db.execute_query(
            f"""
            SELECT
                l.id,
                l.source_platform,
                l.source_listing_id,
                l.listing_status,
                l.ask_price,
                l.sold_price,
                l.listing_url,
                l.seller,
                l.seller_broker,
                l.location,
                l.date_listed,
                l.date_sold,
                a.serial_number,
                a.registration_number,
                a.manufacturer,
                a.model,
                a.manufacturer_year
            FROM aircraft_listings l
            INNER JOIN aircraft a ON l.aircraft_id = a.id
            WHERE ({where_sql})
            ORDER BY l.updated_at DESC NULLS LAST, l.created_at DESC NULLS LAST
            LIMIT 15
            """,
            tuple(params_l),
        )
    except Exception as e:
        logger.warning("consultant_market_lookup: listings query failed: %s", e)

    if listings_out:
        meta["consultant_internal_listings"] = len(listings_out)
        meta["consultant_listing_rows_for_images"] = [
            {
                "source_platform": r.get("source_platform"),
                "source_listing_id": r.get("source_listing_id"),
                "listing_url": r.get("listing_url"),
            }
            for r in listings_out
        ]
        lines.append("")
        lines.append(
            "[FOR USER REPLY — Market / pricing (place near the top of your answer; keep exact dollar amounts and URLs):]"
        )
        phly_ask = _phly_snapshot_includes_ask(phly_rows)
        if phly_ask:
            lines.append(
                "[CRITICAL — If the PhlyData block already shows Ask Price / take for this tail, that is Hye Aero's "
                "internal snapshot. A listing row below may say marketplace-ingest ask_price is missing on the row "
                "only — that does NOT mean the user has no answering price: do not imply the ask is unknown, and do "
                "not use a 'no confirmed live listing' closing on a simple how-much/asking-price question unless the "
                "user asked about live purchase availability.]"
            )
        for i, row in enumerate(listings_out, 1):
            ask_raw = row.get("ask_price")
            ask_s = _fmt_money(ask_raw)
            plat = (row.get("source_platform") or "").strip() or "—"
            st = (row.get("listing_status") or "").strip() or "—"
            url = (row.get("listing_url") or "").strip()
            seller = (row.get("seller") or "").strip() or "—"
            brk = (row.get("seller_broker") or "").strip()
            sold_s = _fmt_money(row.get("sold_price"))
            parts = [
                f"Listing {i}",
                f"platform {plat}",
                f"status {st}",
            ]
            if ask_s and ask_s != "—":
                parts.append(f"asking price {ask_s} USD (from our database)")
            elif phly_ask:
                parts.append(
                    "marketplace-ingest ask_price NULL on this row — use PhlyData block above for Hye Aero internal ask; "
                    "listing URL confirms/timestamps on platform"
                )
            else:
                parts.append(
                    "asking price not stored in our database — tell the user to open the listing URL for current ask"
                )
            parts.append(f"seller {seller}")
            if brk:
                parts.append(f"broker {brk}")
            if url:
                parts.append(f"listing URL {url}")
            if sold_s and sold_s != "—":
                parts.append(f"recorded sold field {sold_s} (may be historical)")
            parts.append(_availability_guidance_for_listing_status(st))
            lines.append("  - " + " | ".join(parts))
        lines.append("")
        lines.append(
            "Internal listing rows tied to this serial or tail (read Status + notes — not all are active for sale):"
        )
        for i, row in enumerate(listings_out, 1):
            sn = (row.get("serial_number") or "—").strip()
            reg = (row.get("registration_number") or "—").strip()
            mm = " ".join(
                x
                for x in (
                    (row.get("manufacturer") or "").strip(),
                    (row.get("model") or "").strip(),
                )
                if x
            ) or "—"
            yr = row.get("manufacturer_year") if row.get("manufacturer_year") is not None else "—"
            plat = (row.get("source_platform") or "—").strip()
            st = (row.get("listing_status") or "—").strip()
            ask = _fmt_money(row.get("ask_price"))
            sold = _fmt_money(row.get("sold_price"))
            seller = (row.get("seller") or "").strip() or "—"
            brk = (row.get("seller_broker") or "").strip()
            loc = (row.get("location") or "").strip() or "—"
            url = (row.get("listing_url") or "").strip()
            lines.append(f"  {i}. Serial {sn} / {reg} | {mm} ({yr})")
            lines.append(f"      Platform: {plat} | Status: {st}")
            lines.append(f"      Ask: {ask} | Sold (if applicable): {sold}")
            lines.append(f"      Seller: {seller}" + (f" | Broker: {brk}" if brk else ""))
            lines.append(f"      Location: {loc}")
            if url:
                lines.append(f"      Listing URL: {url}")
            lines.append(f"      {_availability_guidance_for_listing_status(st)}")
    else:
        lines.append("")
        lines.append("(No matching rows in aircraft_listings for this serial/tail in our database.)")

    # Comparable sales: same make/model from PhlyData when available
    mfr = ""
    mdl = ""
    for r in phly_rows[:1]:
        mfr = (r.get("manufacturer") or "").strip()
        mdl = (r.get("model") or "").strip()
    sales_out: List[Dict[str, Any]] = []
    if mfr and mdl:
        try:
            sales_out = db.execute_query(
                """
                SELECT
                    sold_price,
                    ask_price,
                    manufacturer,
                    model,
                    manufacturer_year,
                    based_country,
                    registration_country,
                    date_sold,
                    airframe_total_time
                FROM aircraft_sales
                WHERE manufacturer ILIKE %s
                  AND model ILIKE %s
                  AND sold_price IS NOT NULL
                  AND sold_price > 0
                ORDER BY date_sold DESC NULLS LAST
                LIMIT 12
                """,
                (f"%{mfr}%", f"%{mdl}%"),
            )
        except Exception as e:
            logger.warning("consultant_market_lookup: sales comps query failed: %s", e)

    if sales_out:
        meta["consultant_internal_sales_comps"] = len(sales_out)
        lines.append("")
        lines.append(
            f"Recent comparable sales from Hye Aero sales data (not PhlyData; same make/model family: {mfr} {mdl}):"
        )
        prices = []
        for i, row in enumerate(sales_out[:10], 1):
            sp = row.get("sold_price")
            prices.append(float(sp))
            yr = row.get("manufacturer_year") if row.get("manufacturer_year") is not None else "—"
            bc = (row.get("based_country") or "").strip()
            rc = (row.get("registration_country") or "").strip()
            reg = ", ".join(x for x in (bc, rc) if x) or "—"
            sd = row.get("date_sold")
            sd_s = str(sd)[:10] if sd is not None else "—"
            aft = row.get("airframe_total_time")
            aft_s = f"{float(aft):,.0f} hrs" if aft is not None else "—"
            lines.append(
                f"  {i}. Sold {_fmt_money(sp)} | Year {yr} | Region {reg} | Date {sd_s} | AFT {aft_s}"
            )
        if prices:
            lo, hi = min(prices), max(prices)
            avg = sum(prices) / len(prices)
            sum_line = (
                f"  Summary (this sample): low {_fmt_money(lo)}, high {_fmt_money(hi)}, "
                f"approx avg {_fmt_money(avg)} — use as market context, not a formal appraisal."
            )
            lines.append(sum_line)
            if not listings_out:
                lines.append("")
                lines.append(
                    "[FOR USER REPLY — No live listing in our DB; include the comp range above as "
                    "recent sale comps (not a current asking price) plus low/high/avg in USD.]"
                )

    block = "\n".join(lines)
    return block, meta


def strip_market_meta_zeros(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Drop zero counts so data_used stays readable."""
    return {k: v for k, v in meta.items() if v not in (0, None, "", [])}


# Known listing marketplaces: URLs here often appear in purchase-mode Tavily merges; if the page
# does not mention the same serial/tail as PhlyData, drop (avoids wrong-jet links, e.g. XLS vs CJ1).
_TAVILY_MARKETPLACE_HOST_MARKERS = (
    "aircraftexchange.com",
    "aircraftexchange",
    "controller.com",
    "trade-a-plane",
    "avbuyer.com",
    "avbuyer",
    "jetnet.com",
    "globalair.com",
    "aso.com",
    "avpay",
    "planesales",
    "planequest",
    "aircraft24.com",
    "aircraft24",
    "hangar67",
    "apn.com",
)


def _alnum_compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _blob_matches_phly_aircraft_identity(blob: str, phly_rows: List[Dict[str, Any]]) -> bool:
    """True if blob (URL + title + body) clearly references PhlyData serial or registration."""
    if not blob or not phly_rows:
        return False
    b_lower = blob.lower()
    b_alnum = _alnum_compact(blob)
    b_nospace = re.sub(r"\s+", "", b_lower)
    for r in phly_rows:
        reg = (r.get("registration_number") or "").strip()
        if reg:
            rc = _alnum_compact(reg)
            if len(rc) >= 3 and rc in b_alnum:
                return True
            rns = reg.lower().replace(" ", "").replace("-", "")
            if len(rns) >= 3 and rns in b_nospace.replace("-", ""):
                return True
        sn = (r.get("serial_number") or "").strip()
        if sn:
            if sn.lower() in b_lower:
                return True
            sc = _alnum_compact(sn)
            if len(sc) >= 5 and sc in b_alnum:
                return True
    return False


def _tavily_url_is_marketplace_host(url: str) -> bool:
    u = (url or "").lower()
    return any(m in u for m in _TAVILY_MARKETPLACE_HOST_MARKERS)


def filter_tavily_results_for_phly_identity(
    payload: Dict[str, Any],
    phly_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    When PhlyData has a resolved aircraft, remove Tavily hits whose URL is a known marketplace
    but whose snippet/URL does not reference the same serial or tail — reduces hallucinated
    wrong listing URLs in consultant answers.
    """
    if not isinstance(payload, dict) or not phly_rows:
        return payload
    has_id = False
    for r in phly_rows:
        if (r.get("serial_number") or "").strip() or (r.get("registration_number") or "").strip():
            has_id = True
            break
    if not has_id:
        return payload
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return payload
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for res in results:
        if not isinstance(res, dict):
            kept.append(res)
            continue
        url = res.get("url") or ""
        if _tavily_url_is_marketplace_host(url):
            blob = f"{url} {res.get('title') or ''} {res.get('content') or ''}"
            if not _blob_matches_phly_aircraft_identity(blob, phly_rows):
                dropped += 1
                continue
        kept.append(res)
    if dropped:
        logger.info(
            "Tavily: dropped %s marketplace results not matching Phly serial/tail (kept %s)",
            dropped,
            len(kept),
        )
    out = dict(payload)
    out["results"] = kept
    return out


# Dollar / USD / common listing phrasing (helps LLM not miss prices in long snippets).
_TAVILY_PRICE_PATTERNS = re.compile(
    r"\$\s?[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|mm|mn|m\b))?|\$\s?[\d.]+\s*(?:million|mm|m\b)"
    r"|\bUSD\s*[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|mm|m\b))?"
    r"|\bUS\s*\$\s*[\d,]+(?:\.\d{1,2})?"
    r"|\b[\d,]+(?:\.\d{1,2})?\s*(?:million|mm|m\b)\s+USD\b"
    r"|\b(?:asking|ask|list|listed|price|priced|sale|sold)\s*(?:price)?\s*[:\-]?\s*\$?\s*[\d,]+(?:\.\d{1,2})?\b"
    r"|\b(?:asking|ask)\s*[:\-]\s*\$[\d,]+(?:\.\d{2})?\b"
    r"|\b€\s*[\d,]+(?:\.\d{1,2})?"
    r"|\bGBP\s*[\d,]+(?:\.\d{1,2})?"
    r"|\b(?:EUR|euros?)\s*[\d,]+(?:\.\d{1,2})?"
    # e.g. "1.895 million" or "2.1mm" without $
    r"|\b\d+(?:\.\d+)?\s*(?:mm|mn|million)\b",
    re.IGNORECASE,
)


def _dollar_amounts_loose(text: str, *, max_n: int = 8) -> List[str]:
    """Catch plain $1,234,567 in listing copy when structured phrases miss."""
    if not text:
        return []
    out: List[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\$\s*[\d]{1,3}(?:,\d{3})+(?:\.\d{2})?\b|\$\s*[\d]{4,}(?:\.\d{2})?\b", text):
        s = m.group(0).strip()
        k = s.replace(" ", "")
        if k not in seen and len(s) >= 4:
            seen.add(k)
            out.append(s)
        if len(out) >= max_n:
            break
    return out


def tavily_price_highlights_block(payload: Dict[str, Any], *, max_snippets: int = 16, max_highlights: int = 12) -> str:
    """
    Build a short appendix listing $ / USD strings found per Tavily result index so the model
    must surface asking prices when they appear only inside snippet bodies.
    """
    results = payload.get("results") or []
    if not isinstance(results, list) or not results:
        return ""
    lines: List[str] = []
    seen_amt: set[str] = set()
    for i, r in enumerate(results[: max(1, max_snippets)], 1):
        if not isinstance(r, dict):
            continue
        blob = f"{r.get('title') or ''} {r.get('content') or ''}"
        hits: List[str] = []
        for m in _TAVILY_PRICE_PATTERNS.finditer(blob):
            amt = m.group(0).strip()
            if len(amt) < 2:
                continue
            key = amt.lower().replace(" ", "")
            if key in seen_amt:
                continue
            seen_amt.add(key)
            hits.append(amt)
            if len(seen_amt) >= max_highlights:
                break
        for loose in _dollar_amounts_loose(blob):
            k = loose.lower().replace(" ", "")
            if k in seen_amt:
                continue
            seen_amt.add(k)
            hits.append(loose)
            if len(seen_amt) >= max_highlights:
                break
        if hits:
            lines.append(f"  - Snippet #{i}: " + "; ".join(hits[:6]))
        if len(seen_amt) >= max_highlights:
            break
    if not lines:
        return ""
    return (
        "[WEB — Dollar amounts spotted in Tavily snippet text (use these in your answer when relevant; "
        "verify wording in the numbered snippets above)]\n" + "\n".join(lines)
    )


def enrich_rag_queries_for_purchase(
    base_queries: List[str],
    query: str,
    history: Optional[List[Dict[str, str]]],
    phly_rows: List[Dict[str, Any]],
    *,
    max_total: int = 8,
    strict_market_sql: bool = False,
) -> List[str]:
    """Add embedding-search phrases biased toward listings and prices for this aircraft."""
    if not consultant_wants_internal_listings_sql(query, history, phly_rows, strict=strict_market_sql):
        return base_queries
    out: List[str] = []
    for q in base_queries or []:
        s = (q or "").strip()
        if s and s not in out:
            out.append(s)
    extras: List[str] = []
    for r in phly_rows[:2]:
        sn = (r.get("serial_number") or "").strip()
        reg = (r.get("registration_number") or "").strip()
        mfr = (r.get("manufacturer") or "").strip()
        mdl = (r.get("model") or "").strip()
        mm = " ".join(x for x in (mfr, mdl) if x).strip()
        if sn:
            extras.append(f"{sn} aircraft listing for sale asking price")
            extras.append(f"{sn} sold price comparable sale")
        if reg:
            extras.append(f"{reg} for sale listing price")
        if mm and (sn or reg):
            extras.append(f"{mm} {sn or reg} listing Controller AvPay")
    for e in extras:
        if e and e.lower() not in {x.lower() for x in out}:
            out.append(e)
        if len(out) >= max_total:
            break
    return out[:max_total]


def build_purchase_listing_tavily_query(
    query: str,
    history: Optional[List[Dict[str, str]]],
    phly_rows: List[Dict[str, Any]],
    *,
    strict_market_sql: bool = False,
) -> Optional[str]:
    """
    Extra web search string biased toward **asking price** and **for-sale listings** for this tail/serial.
    Merged with primary/owner Tavily so AvPay / Controller / etc. snippets surface when Postgres has no row.
    """
    if not consultant_wants_internal_market_sql(query, history, strict=strict_market_sql):
        return None
    reg = ""
    serial = ""
    mm = ""
    if phly_rows:
        r = phly_rows[0]
        reg = (r.get("registration_number") or "").strip().upper().replace(" ", "")
        serial = (r.get("serial_number") or "").strip()
        mfr = (r.get("manufacturer") or "").strip()
        mdl = (r.get("model") or "").strip()
        mm = " ".join(x for x in (mfr, mdl) if x).strip()
    else:
        for t in extract_phlydata_tokens_with_history(query, history):
            u = (t or "").strip().upper().replace(" ", "")
            if re.match(r"^N[-]?[A-Z0-9]{1,6}$", u):
                reg = u.replace("-", "")
            if re.search(r"\b\d{2,5}-\d{3,6}\b", t or ""):
                serial = (t or "").strip()
        if not reg and not serial:
            return None
    parts: List[str] = [
        "asking price",
        "for sale",
        "aircraft listing",
        "USD",
    ]
    if reg:
        parts.insert(0, f'"{reg}"')
    if serial:
        parts.append(serial)
    if mm:
        parts.append(mm)
    parts.extend(["broker", "JetNet", "Controller", "AvPay", "AircraftExchange"])
    q = " ".join(x for x in parts if x).strip()
    return q[:500] if len(q) >= 12 else None
