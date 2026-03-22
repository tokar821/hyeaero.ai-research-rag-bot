"""
LLM-only mapping from FAA **legal registrant** (often a trustee shell) to **ZoomInfo-oriented**
company candidates — no web search, no Tavily.

Runs when ``OPENAI_API_KEY`` is set and the row has no curated ``trustee_operating_contact``
(unless ``REGISTRANT_LLM_EVEN_WHEN_KB_MATCH=1``). Skipped for rows that do not look like
corporate shells unless ``REGISTRANT_LLM_FOR_ALL_FAA=1``.

Disable entirely: ``REGISTRANT_LLM_RESOLVER_DISABLED=1``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _strip(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _disabled() -> bool:
    v = (os.getenv("REGISTRANT_LLM_RESOLVER_DISABLED") or "").strip().lower()
    return v in ("1", "true", "yes")


def _env_truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes")


def _looks_corporate_shell(registrant_name: str) -> bool:
    """Rough filter to avoid LLM on obvious personal names (saves cost)."""
    if _env_truthy("REGISTRANT_LLM_FOR_ALL_FAA"):
        return bool(_strip(registrant_name))
    raw = _strip(registrant_name)
    if len(raw) < 8:
        return False
    u = raw.upper()
    if any(x in u for x in ("TRUSTEE", "TRUST ", " NOMINEE", "CUSTODIAN", "OWNER TRUST")):
        return True
    tokens = set(re.sub(r"[^A-Za-z0-9]+", " ", u).split())
    if tokens & {"INC", "LLC", "CORP", "CORPORATION", "LP", "LTD", "LLP", "CO"}:
        return True
    return False


def resolve_registrant_row_llm(
    row: Dict[str, Any],
    *,
    openai_api_key: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    """
    Returns ``{ "candidates": [ { company_name, website, phone, confidence, rationale } ], "model": ... }``
    or ``{ "error": "..." }`` on failure.
    """
    name = _strip(row.get("registrant_name"))
    if not name:
        return None

    street = _strip(row.get("street"))
    if (row.get("street2") or "").strip():
        street = f"{street} {_strip(row.get('street2'))}".strip()
    city = _strip(row.get("city"))
    state = _strip(row.get("state"))
    z = _strip(row.get("zip_code"))
    country = _strip(row.get("country"))
    addr = ", ".join(p for p in (street, city, state, z, country) if p)

    system = """You map FAA aircraft **legal registrant** strings to likely **operating business** names
used in B2B databases (e.g. ZoomInfo). The FAA string is often a title trustee or shell (e.g. "… INC TRUSTEE").

Rules:
- Use **general knowledge** and the **mailing address** to infer a plausible operating brand or d/b/a when widely known (e.g. well-known trustee services and their public brand).
- If you are **not confident** there is a specific operating company distinct from the legal string, return **zero or one** candidate with confidence "none" or "low" and null company_name.
- **Never invent** a website or phone number unless you are highly confident from well-known public facts; prefer null for website/phone when uncertain.
- Return **only** valid JSON, no markdown.

JSON schema:
{
  "candidates": [
    {
      "company_name": string or null,
      "website": string or null,
      "phone": string or null,
      "confidence": "high" | "medium" | "low" | "none",
      "rationale": string (short)
    }
  ]
}

At most **2** candidates. Prefer **one** strong candidate over two weak ones."""

    user = f"FAA registrant (legal): {name}\nFAA mailing address: {addr or '—'}\n"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=600,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```\s*$", "", raw)
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"error": "invalid_json_shape", "candidates": []}
        cands = data.get("candidates")
        if not isinstance(cands, list):
            cands = []
        norm: List[Dict[str, Any]] = []
        for c in cands[:2]:
            if not isinstance(c, dict):
                continue
            cn = _strip(c.get("company_name")) or None
            conf = (_strip(c.get("confidence")) or "none").lower()
            if conf not in ("high", "medium", "low", "none"):
                conf = "low"
            norm.append(
                {
                    "company_name": cn,
                    "website": _strip(c.get("website")) or None,
                    "phone": _strip(c.get("phone")) or None,
                    "confidence": conf,
                    "rationale": _strip(c.get("rationale")) or None,
                }
            )
        return {"candidates": norm, "model": model}
    except Exception as e:
        logger.warning("registrant_llm_resolver failed: %s", e)
        return {"error": str(e)[:400], "candidates": []}


def enrich_faa_owners_with_llm_registrant_resolution(owners_from_faa: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if _disabled():
        for row in owners_from_faa:
            r = dict(row)
            r["llm_registrant_resolution"] = None
            out.append(r)
        return out

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()

    for row in owners_from_faa:
        r = dict(row)
        r["llm_registrant_resolution"] = None

        if not api_key:
            out.append(r)
            continue

        if r.get("trustee_operating_contact") and not _env_truthy("REGISTRANT_LLM_EVEN_WHEN_KB_MATCH"):
            out.append(r)
            continue

        name = _strip(r.get("registrant_name"))
        if not name or not _looks_corporate_shell(name):
            out.append(r)
            continue

        resolved = resolve_registrant_row_llm(r, openai_api_key=api_key, model=model)
        if resolved:
            r["llm_registrant_resolution"] = resolved
        out.append(r)

    return out
