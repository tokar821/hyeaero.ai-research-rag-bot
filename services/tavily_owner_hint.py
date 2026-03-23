"""
Tavily web search hints for FAA **trustee / shell-style** registrants + mailing address.

Snippets are **unverified**; they feed ``tavily_llm_synthesis`` (OpenAI) and may drive ZoomInfo
``faa_tavily_llm_hint``. No curated JSON knowledge base.

Requires ``TAVILY_API_KEY``. Uses Tavily via ``requests`` or ``tavily-python`` SDK.

Disable with ``TAVILY_DISABLED=1``. Optional ``TAVILY_WHEN_CORP_AND_ADDRESS=1`` expands to corporate
registrants (INC/LLC/CORP/…) with a street or city+state.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DISCLAIMER = (
    "Web search suggestions only — not verified. FAA registrant remains the legal record. "
    "Review sources before relying on names or URLs."
)


def _strip(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip()


def is_trustee_like_registrant(registrant_name: str) -> bool:
    """
    True when the FAA name looks like a corporate title-holder / trust shell (not a person).

    Conservative: avoids Tavily spend on obvious individual names.
    """
    raw = _strip(registrant_name)
    if len(raw) < 10:
        return False
    norm = re.sub(r"[^A-Za-z0-9]+", " ", raw.upper()).strip()
    if not norm:
        return False
    tokens = set(norm.split())

    if "TRUSTEE" in tokens:
        return True
    if "NOMINEE" in tokens or "CUSTODIAN" in tokens:
        return True
    if "TRUST" in tokens and tokens & {"INC", "LLC", "CORP", "CORPORATION", "LP", "LTD", "CO"}:
        return True
    # "OWNER TRUST" style (two tokens)
    if "OWNER" in tokens and "TRUST" in tokens:
        return True
    if "TITLE" in tokens and "TRUST" in tokens:
        return True
    return False


# Corporate suffix tokens common on FAA registrants (not persons). Used only when
# TAVILY_WHEN_CORP_AND_ADDRESS is enabled.
_CORP_HINT_TOKENS = frozenset(
    {"INC", "LLC", "CORP", "CORPORATION", "LP", "LTD", "LLP", "CO", "TRUSTEE", "NOMINEE", "CUSTODIAN"}
)


def is_corporate_entity_registrant(registrant_name: str) -> bool:
    """
    True when the name clearly looks like a registered organization (INC/LLC/…),
    so Tavily + street may still help find a DBA / website when trustee keywords are absent.

    Still skips very short strings. Does not fire on typical ``LAST, FIRST`` person lines.
    """
    raw = _strip(registrant_name)
    if len(raw) < 12:
        return False
    norm = re.sub(r"[^A-Za-z0-9]+", " ", raw.upper()).strip()
    if not norm:
        return False
    tokens = set(norm.split())
    return bool(tokens & _CORP_HINT_TOKENS)


def should_run_tavily_for_registrant(registrant_name: str) -> bool:
    if is_trustee_like_registrant(registrant_name):
        return True
    flag = (os.getenv("TAVILY_WHEN_CORP_AND_ADDRESS") or "").strip().lower()
    if flag in ("1", "true", "yes") and is_corporate_entity_registrant(registrant_name):
        return True
    return False


def has_address_context_for_search(row: Dict[str, Any]) -> bool:
    """Need enough location to disambiguate web search."""
    street = _strip(row.get("street"))
    street2 = _strip(row.get("street2"))
    city = _strip(row.get("city"))
    state = _strip(row.get("state"))
    z = _strip(row.get("zip_code"))
    if street or street2:
        return True
    if city and (state or z):
        return True
    return False


def build_owner_search_query(row: Dict[str, Any]) -> str:
    """Single-line query: registrant + mailing address + disambiguation terms."""
    name = _strip(row.get("registrant_name"))
    parts = [
        f'"{name}"' if name else "",
        _strip(row.get("street")),
        _strip(row.get("street2")),
        _strip(row.get("city")),
        _strip(row.get("state")),
        _strip(row.get("zip_code")),
        "FAA aircraft",
        "trustee",
        "dba",
    ]
    q = " ".join(p for p in parts if p)
    if len(q) > 380:
        q = q[:377] + "..."
    return q


def _normalize_search_depth(raw: Optional[str]) -> str:
    d = (raw or "basic").strip().lower()
    return d if d in ("basic", "advanced") else "basic"


def _tavily_search_rest(
    api_key: str, query: str, max_results: int, search_depth: str = "basic"
) -> Dict[str, Any]:
    import requests

    body = {
        "api_key": api_key,
        "query": query,
        "max_results": max(1, min(10, max_results)),
        "search_depth": _normalize_search_depth(search_depth),
    }
    r = requests.post(TAVILY_SEARCH_URL, json=body, timeout=45)
    r.raise_for_status()
    return r.json()


def _tavily_search_sdk(
    api_key: str, query: str, max_results: int, search_depth: str = "basic"
) -> Dict[str, Any]:
    from tavily import TavilyClient  # type: ignore

    client = TavilyClient(api_key=api_key)
    n = max(1, min(10, max_results))
    sd = _normalize_search_depth(search_depth)
    try:
        return client.search(query, max_results=n, search_depth=sd)
    except TypeError:
        # Older tavily-python may not accept search_depth
        return client.search(query, max_results=n)


def fetch_tavily_hints_for_query(
    query: str,
    result_limit: Optional[int] = None,
    search_depth: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call Tavily once. Returns a dict safe to JSON-serialize to the frontend.

    ``result_limit`` overrides ``TAVILY_MAX_RESULTS`` (capped 1–10) when set (e.g. Ask Consultant uses 8).
    ``search_depth`` may be ``basic`` or ``advanced`` (Tavily); if omitted, uses env ``TAVILY_SEARCH_DEPTH`` or ``basic``.

    On failure, returns ``{"query", "disclaimer", "results": [], "error": "..."}``.
    """
    if not query.strip():
        return {
            "query": None,
            "disclaimer": DISCLAIMER,
            "results": [],
            "error": "empty_query",
        }

    if (os.getenv("TAVILY_DISABLED") or "").strip().lower() in ("1", "true", "yes"):
        return {
            "query": query,
            "disclaimer": DISCLAIMER,
            "results": [],
            "error": "tavily_disabled",
        }

    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return {
            "query": query,
            "disclaimer": DISCLAIMER,
            "results": [],
            "error": "tavily_api_key_missing",
        }

    env_n = 5
    try:
        env_n = max(1, min(10, int((os.getenv("TAVILY_MAX_RESULTS") or "5").strip())))
    except ValueError:
        env_n = 5
    if result_limit is not None:
        try:
            max_results = max(1, min(10, int(result_limit)))
        except (TypeError, ValueError):
            max_results = env_n
    else:
        max_results = env_n

    depth = _normalize_search_depth(
        search_depth if search_depth is not None else (os.getenv("TAVILY_SEARCH_DEPTH") or "basic")
    )

    try:
        try:
            data = _tavily_search_sdk(api_key, query, max_results, depth)
        except ImportError:
            data = _tavily_search_rest(api_key, query, max_results, depth)
    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        return {
            "query": query,
            "disclaimer": DISCLAIMER,
            "results": [],
            "error": str(e)[:500],
        }

    raw_results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(raw_results, list):
        raw_results = []

    slim: List[Dict[str, str]] = []
    for item in raw_results[:max_results]:
        if not isinstance(item, dict):
            continue
        slim.append(
            {
                "title": _strip(item.get("title")) or None,
                "url": _strip(item.get("url")) or None,
                "content": _strip(item.get("content")) or None,
            }
        )

    return {
        "query": query,
        "disclaimer": DISCLAIMER,
        "results": slim,
        "error": None,
    }


def enrich_faa_owners_with_tavily_hints(owners_from_faa: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each FAA row: if registrant passes :func:`should_run_tavily_for_registrant` (trustee-like by
    default; optional corporate + address when ``TAVILY_WHEN_CORP_AND_ADDRESS=1``), and address context
    exists — attach ``tavily_web_hints``. Deduplicates by query string per request.
    """
    cache: Dict[str, Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []

    for row in owners_from_faa:
        r = dict(row)
        r["tavily_web_hints"] = None

        name = _strip(r.get("registrant_name"))
        if not name or not should_run_tavily_for_registrant(name):
            out.append(r)
            continue

        if not has_address_context_for_search(r):
            out.append(r)
            continue

        q = build_owner_search_query(r)
        if not q.strip():
            out.append(r)
            continue

        if q not in cache:
            logger.info("Tavily owner hint: query=%r", q[:200])
            cache[q] = fetch_tavily_hints_for_query(q)

        payload = cache[q]
        if payload.get("results"):
            r["tavily_web_hints"] = payload
        elif payload.get("error") and payload.get("error") not in (
            "tavily_api_key_missing",
            "tavily_disabled",
        ):
            # Surface transport errors to UI for debugging; omit missing-key noise
            r["tavily_web_hints"] = payload
        else:
            r["tavily_web_hints"] = None

        out.append(r)

    return out
