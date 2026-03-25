"""
Optional Tavily gating for Ask Consultant.

When ``CONSULTANT_TAVILY_WHEN_NEEDED=1``, skip Tavily if PhlyData + FAA MASTER already
answer the question class (identity / U.S. registrant) and the user is not asking for
purchase/listing web, operator/fleet web, or other explicit live-web topics.

Default (env unset / 0): always run Tavily — preserves existing behavior.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from rag.consultant_market_lookup import (
    consultant_wants_internal_market_sql,
    wants_consultant_aircraft_detail_context,
)
from rag.phlydata_consultant_lookup import wants_consultant_owner_operator_context

logger = logging.getLogger(__name__)


def _gate_context_blob(
    query: str,
    history: Optional[List[Dict[str, str]]],
    *,
    max_chars: int = 4500,
) -> str:
    """User + assistant lines + current query (for follow-ups like \"what about the operator\")."""
    parts: List[str] = []
    if history:
        for h in history[-12:]:
            role = (h.get("role") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            c = (h.get("content") or "").strip()
            if c:
                parts.append(f"{role}: {c}")
    q = (query or "").strip()
    if q:
        parts.append(q)
    return "\n".join(parts)[:max_chars]


def _needs_operator_or_fleet_web(blob_lc: str) -> bool:
    """Questions where FAA registrant alone is often insufficient."""
    needles = (
        "operator",
        "operate",
        "operated",
        "operating",
        "charter",
        "fleet",
        "aoc",
        "air operator",
        "management",
        "aircraft management",
        "certificate holder",
        "who flies",
        "airline",
        "flying under",
        "dry lease",
        "wet lease",
        "leased to",
        "beneficial owner",
        "ultimate owner",
        "dba",
        "doing business as",
    )
    return any(n in blob_lc for n in needles)


def _explicit_web_or_news_triggers(blob_lc: str) -> bool:
    """User likely wants fresh or external sources beyond internal DB."""
    triggers = (
        "news",
        "latest news",
        "press",
        "accident",
        "incident",
        "crash",
        "website",
        " official website",
        "url ",
        "link to",
        "where can i find",
        "google ",
        "search the web",
        "on the internet",
        "market trend",
        "industry outlook",
    )
    return any(t in blob_lc for t in triggers)


def should_run_consultant_tavily(
    *,
    when_needed_enabled: bool,
    query: str,
    history: Optional[List[Dict[str, str]]],
    phly_authority: str,
    phly_rows: List[Dict[str, Any]],
    phly_meta: Dict[str, Any],
    strict_market_sql: bool = False,
) -> Tuple[bool, str]:
    """
    Returns (run_tavily, reason_code for logging / data_used).

    If ``when_needed_enabled`` is False, always returns (True, \"always_on\").
    """
    if not when_needed_enabled:
        return True, "always_on"

    blob = _gate_context_blob(query, history)
    blob_lc = blob.lower()

    if consultant_wants_internal_market_sql(query, history, strict=strict_market_sql):
        return True, "purchase_price_listing"

    if wants_consultant_aircraft_detail_context(query, history):
        return True, "aircraft_detail_context"

    if not (phly_authority or "").strip():
        return True, "no_phly_authority"

    if int((phly_meta or {}).get("phlydata_no_row_for_tokens") or 0):
        return True, "phly_no_row_use_web_and_rag"

    n_phly = len(phly_rows or [])
    faa_hits = int((phly_meta or {}).get("faa_master_owner_rows") or 0)

    if n_phly <= 0:
        return True, "no_phly_rows"

    if faa_hits < n_phly:
        return True, "incomplete_faa_master_coverage"

    if _explicit_web_or_news_triggers(blob_lc):
        return True, "web_news_trigger"

    if wants_consultant_owner_operator_context(blob):
        if _needs_operator_or_fleet_web(blob_lc):
            return True, "operator_fleet_context"
        return False, "faa_registrant_identity_sufficient"

    return False, "internal_context_sufficient"


def empty_consultant_tavily_payload() -> Dict[str, Any]:
    """Same shape as :func:`services.tavily_owner_hint.fetch_tavily_hints_for_query` on empty/error."""
    from services.tavily_owner_hint import DISCLAIMER

    return {
        "query": None,
        "disclaimer": DISCLAIMER,
        "results": [],
        "error": None,
    }
