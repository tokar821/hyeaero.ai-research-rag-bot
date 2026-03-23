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
from rag.phlydata_consultant_lookup import extract_phlydata_tokens_with_history, ilike_patterns_for_token

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
        r"\b(buy|purchase|pricing|sold|asking|listing|listings|offer|cost|pay|worth)\b",
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

    Skips all market SQL unless the user message(s) look market-related; set
    ``strict_market_sql=True`` (``CONSULTANT_MARKET_SQL_STRICT=1``) for a narrower trigger list.
    """
    meta: Dict[str, Any] = {"consultant_internal_listings": 0, "consultant_internal_sales_comps": 0}
    if not consultant_wants_internal_market_sql(query, history, strict=strict_market_sql):
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
    if not pats:
        return "", meta

    lines: List[str] = [
        "[INTERNAL — Hye Aero market data (Postgres: aircraft_listings + aircraft_sales)]",
        "Use this for **asking price**, **sold price**, platform/seller, and listing URLs when answering "
        "purchase, availability, or pricing questions. Amounts are as stored (typically USD). "
        "Reconcile with Tavily/web snippets when both exist; cite this block as Internal listings/sales.",
        "If the user asks whether they can buy now or what it costs: quote the **Ask:** line (and Listing URL) "
        "before paraphrasing web-only sources.",
    ]

    listings_out: List[Dict[str, Any]] = []
    try:
        cond_parts: List[str] = []
        params_l: List[Any] = []
        for p in pats:
            cond_parts.append(
                "(a.serial_number ILIKE %s OR a.registration_number ILIKE %s OR l.listing_url ILIKE %s)"
            )
            params_l.extend([p, p, p])
        where_sql = " OR ".join(cond_parts) if cond_parts else "FALSE"
        listings_out = db.execute_query(
            f"""
            SELECT
                l.id,
                l.source_platform,
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
        lines.append("")
        lines.append(
            "[FOR USER REPLY — Market / pricing (place near the top of your answer; keep exact dollar amounts and URLs):]"
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
            lines.append("  - " + " | ".join(parts))
        lines.append("")
        lines.append("Active / recent internal listings tied to this serial or tail:")
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
        lines.append(f"Recent internal comparable sales (same make/model family: {mfr} {mdl}):")
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
    if not consultant_wants_internal_market_sql(query, history, strict=strict_market_sql):
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
