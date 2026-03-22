"""
After Tavily web hints, use OpenAI to infer a likely **operating company** from snippets
(FAA registrant is often a trustee shell). Output is attached as ``tavily_llm_synthesis`` on
each ``owners_from_faa`` row and can drive an extra ZoomInfo query (``faa_tavily_llm_hint``).

Disable with ``TAVILY_LLM_SYNTHESIS_DISABLED=1`` or missing ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_SNIPPET_CHARS = 600
_MAX_RESULTS_IN_PROMPT = 5


def _strip(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _disabled() -> bool:
    v = (os.getenv("TAVILY_LLM_SYNTHESIS_DISABLED") or "").strip().lower()
    return v in ("1", "true", "yes")


def _truncate(s: str, n: int) -> str:
    s = s.replace("\r", " ").replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def synthesize_from_tavily_and_registrant(
    *,
    registrant_name: str,
    street: Optional[str],
    street2: Optional[str],
    city: Optional[str],
    state: Optional[str],
    zip_code: Optional[str],
    tavily_results: List[Dict[str, Any]],
    openai_api_key: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    """
    Call OpenAI once. Returns a dict with keys:
    operating_company_name, website, phone, confidence, summary, suggested_zoominfo_query
    or None on failure / skip.
    """
    if not openai_api_key or not tavily_results:
        return None

    addr_parts = [_strip(street), _strip(street2), _strip(city), _strip(state), _strip(zip_code)]
    address_line = ", ".join(p for p in addr_parts if p)

    lines: List[str] = []
    for i, hit in enumerate(tavily_results[:_MAX_RESULTS_IN_PROMPT], 1):
        title = _truncate(_strip(hit.get("title")), 200)
        url = _strip(hit.get("url"))
        content = _truncate(_strip(hit.get("content")), _MAX_SNIPPET_CHARS)
        lines.append(f"[{i}] title: {title}\n    url: {url}\n    snippet: {content}")

    user_block = (
        f"FAA legal registrant on file: {registrant_name}\n"
        f"FAA mailing address (may help match dba / office): {address_line or '—'}\n\n"
        f"Web search snippets (may include generic industry pages — be skeptical):\n"
        + "\n\n".join(lines)
    )

    system = """You help aviation analysts map FAA **legal registrant** strings (often "… INC TRUSTEE") to a likely **operating business name** (dba / brand) using web snippets only.

Rules:
- The FAA registrant string is authoritative for **legal title**; your job is to infer **who might operate or market** services, only when snippets clearly support it (e.g. same address, explicit "d/b/a", company name on official-looking page).
- If snippets are only generic "aircraft trust services" with no clear link to THIS registrant name or address, return confidence "none" or "low" and mostly null fields.
- Never invent a website or phone not present in snippets unless you are highly confident from a single authoritative URL's content.
- Return **only** valid JSON, no markdown.

JSON schema:
{
  "operating_company_name": string or null,
  "website": string or null,
  "phone": string or null,
  "confidence": "high" | "medium" | "low" | "none",
  "summary": string (one or two sentences),
  "suggested_zoominfo_query": string or null
}

Use suggested_zoominfo_query as the best short company name phrase for a B2B database search (often same as operating_company_name)."""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_block},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip ```json fences if any
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```\s*$", "", raw)
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        out = {
            "operating_company_name": _strip(data.get("operating_company_name")) or None,
            "website": _strip(data.get("website")) or None,
            "phone": _strip(data.get("phone")) or None,
            "confidence": (_strip(data.get("confidence")) or "none").lower(),
            "summary": _strip(data.get("summary")) or None,
            "suggested_zoominfo_query": _strip(data.get("suggested_zoominfo_query")) or None,
        }
        if out["confidence"] not in ("high", "medium", "low", "none"):
            out["confidence"] = "low"
        if not out["suggested_zoominfo_query"] and out["operating_company_name"]:
            out["suggested_zoominfo_query"] = out["operating_company_name"]
        return out
    except Exception as e:
        logger.warning("Tavily LLM synthesis failed: %s", e)
        return {"error": str(e)[:400], "confidence": "none"}


def enrich_faa_owners_with_tavily_llm_synthesis(owners_from_faa: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each row with non-empty ``tavily_web_hints.results``, optionally call OpenAI and set
    ``tavily_llm_synthesis``. Rows without Tavily results get ``tavily_llm_synthesis``: null.
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()

    out: List[Dict[str, Any]] = []
    for row in owners_from_faa:
        r = dict(row)
        r["tavily_llm_synthesis"] = None

        if _disabled() or not api_key:
            out.append(r)
            continue

        tw = r.get("tavily_web_hints")
        if not isinstance(tw, dict):
            out.append(r)
            continue
        results = tw.get("results") or []
        if not isinstance(results, list) or not results:
            out.append(r)
            continue

        name = _strip(r.get("registrant_name"))
        if not name:
            out.append(r)
            continue

        syn = synthesize_from_tavily_and_registrant(
            registrant_name=name,
            street=r.get("street"),
            street2=r.get("street2"),
            city=r.get("city"),
            state=r.get("state"),
            zip_code=r.get("zip_code"),
            tavily_results=results,
            openai_api_key=api_key,
            model=model,
        )
        if syn and not syn.get("error"):
            r["tavily_llm_synthesis"] = syn
        elif syn and syn.get("error"):
            r["tavily_llm_synthesis"] = syn
        out.append(r)

    return out
