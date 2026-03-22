"""
Use OpenAI to infer a plausible **operating company** (name + website) from Tavily search snippets,
given the FAA **legal registrant** name and mailing address.

Output is **unverified**; intended to drive an optional ZoomInfo company search (``tavily_derived``)
and to show a labeled card in the owner panel.

Enable with ``OPENAI_API_KEY`` and ``TAVILY_LLM_DERIVE=1`` (default on). Disable with ``TAVILY_LLM_DERIVE=0``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "AI read Tavily web snippets only — not verified. May be wrong; FAA registrant remains the legal record."
)


def _strip(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _mailing_blob(row: Dict[str, Any]) -> str:
    parts = [
        row.get("street"),
        row.get("street2"),
        row.get("city"),
        row.get("state"),
        row.get("zip_code"),
        row.get("country"),
    ]
    return ", ".join(_strip(p) for p in parts if _strip(p))


def derive_company_from_tavily_llm(
    registrant_name: str,
    faa_row: Dict[str, Any],
    tavily_results: List[Dict[str, str]],
    *,
    openai_api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return a dict suitable for JSON and ``owners_from_faa[].tavily_derived_company``, or None.

    ``tavily_results`` items: ``title``, ``url``, ``content`` (as from :func:`fetch_tavily_hints_for_query`).
    """
    flag = (os.getenv("TAVILY_LLM_DERIVE") or "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return None

    key = (openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        logger.debug("tavily_derive_company: OPENAI_API_KEY missing, skip LLM")
        return None

    name = _strip(registrant_name)
    if not name or not tavily_results:
        return None

    # Cap snippets for cost/latency
    max_snip = 700
    max_items = 6
    lines: List[str] = []
    allowed_urls: List[str] = []
    for i, hit in enumerate(tavily_results[:max_items], 1):
        u = _strip(hit.get("url"))
        t = _strip(hit.get("title"))
        c = _strip(hit.get("content"))[:max_snip]
        if u:
            allowed_urls.append(u)
        lines.append(f"[{i}] title={t!r}\n    url={u}\n    snippet={c!r}")

    blob = _mailing_blob(faa_row)
    user = (
        f"FAA legal registrant name: {name!r}\n"
        f"FAA mailing address on file: {blob!r}\n\n"
        "Web search snippets (may include irrelevant SEO pages):\n"
        + "\n\n".join(lines)
        + "\n\n"
        "Task: If any snippet plausibly identifies an operating company, DBA, or brand **related to "
        "this specific registrant or the same address**, extract it. Generic 'aircraft trust services' "
        "pages that do not tie to this name/address should NOT produce a company.\n"
        'Reply with JSON only: {"suggested_operating_name": string|null, "suggested_website": string|null, '
        '"confidence": "high"|"medium"|"low", "rationale": string, "source_urls_used": string[]}'
    )

    model = (os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()

    try:
        import openai

        client = openai.OpenAI(api_key=key, timeout=35.0)
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured company hints for aviation compliance analysts. "
                        "Be conservative: if evidence is weak or only generic industry content, set "
                        "suggested_operating_name to null and confidence to low. "
                        "Websites must be bare host or https URL if clearly stated in snippets."
                    ),
                },
                {"role": "user", "content": user},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
    except Exception as e:
        logger.warning("tavily_derive_company LLM failed: %s", e)
        return None

    if not isinstance(data, dict):
        return None

    op = _strip(data.get("suggested_operating_name")) or None
    web = _strip(data.get("suggested_website")) or None
    conf = (_strip(data.get("confidence")) or "low").lower()
    rationale = _strip(data.get("rationale")) or None
    urls = data.get("source_urls_used")
    if not isinstance(urls, list):
        urls = []
    urls_clean = [u for u in (_strip(x) for x in urls) if u]

    # Filter URLs to those we actually showed the model (reduce hallucinated citations)
    allowed_set = {u.lower() for u in allowed_urls}
    urls_clean = [u for u in urls_clean if u.lower() in allowed_set]

    if conf not in ("high", "medium", "low"):
        conf = "low"

    if not op and not web:
        return {
            "suggested_operating_name": None,
            "suggested_website": None,
            "confidence": conf,
            "rationale": rationale,
            "source_urls_used": urls_clean,
            "disclaimer": DISCLAIMER,
            "skipped_for_zoominfo": True,
        }

    # Normalize website to https if it looks like a domain
    if web and not web.startswith("http"):
        if re.match(r"^[a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z]{2,}", web):
            web = "https://" + web.lstrip("/")

    skip_zi = conf == "low" or not op
    out = {
        "suggested_operating_name": op,
        "suggested_website": web,
        "confidence": conf,
        "rationale": rationale,
        "source_urls_used": urls_clean,
        "disclaimer": DISCLAIMER,
        "skipped_for_zoominfo": skip_zi,
    }
    return out
