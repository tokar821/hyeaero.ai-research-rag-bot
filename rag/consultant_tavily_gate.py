"""
Tavily gating for Ask Consultant.

**Default:** web search runs only as a **fallback** after internal retrieval — see
:func:`should_run_consultant_tavily_after_internal`.

Set ``CONSULTANT_TAVILY_ALWAYS=1`` to always run Tavily (previous default-style behavior).

Legacy heuristic: :func:`should_run_consultant_tavily` (still available; no longer used by
``RAGQueryService`` by default).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from rag.consultant_market_lookup import (
    consultant_wants_internal_market_sql,
    wants_consultant_aircraft_detail_context,
)
from rag.phlydata_consultant_lookup import wants_consultant_owner_operator_context

logger = logging.getLogger(__name__)


def consultant_tavily_min_vector_threshold() -> int:
    """Minimum Pinecone hits before Tavily may be skipped (default ``3``)."""
    try:
        v = int((os.getenv("CONSULTANT_TAVILY_MIN_VECTOR_RESULTS") or "3").strip())
    except ValueError:
        return 3
    return max(0, min(50, v))


def should_run_consultant_tavily_after_internal(
    *,
    vector_result_count: int,
    sql_context_nonempty: bool,
    force_always: bool = False,
    min_vector_results: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Run Tavily when internal context is thin: too few vector hits **or** no SQL-backed
    authority (PhlyData / FAA / listing block).

    Skip Tavily when ``vector_result_count >= min_vector_results`` **and**
    ``sql_context_nonempty`` (unless ``force_always``).

    ``min_vector_results`` defaults to :func:`consultant_tavily_min_vector_threshold`.
    """
    if force_always:
        return True, "forced_always_on"
    min_v = (
        consultant_tavily_min_vector_threshold()
        if min_vector_results is None
        else max(0, min(50, int(min_vector_results)))
    )
    if not sql_context_nonempty:
        return True, "sql_context_empty"
    if vector_result_count < min_v:
        return True, "vector_below_threshold"
    return False, "internal_sql_and_vector_sufficient"


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
