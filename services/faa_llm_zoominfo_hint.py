"""
LLM-only hint: map FAA **legal registrant** (often a trustee shell) + mailing address → a **short
company name** (and optional website) suitable for **ZoomInfo company search**.

No web search (no Tavily). Uses model knowledge conservatively; prefer ``none``/``low`` when unsure.

Disable with ``FAA_LLM_ZOOMINFO_HINT_DISABLED=1`` or missing ``OPENAI_API_KEY``.

Runs only for rows **without** ``trustee_operating_contact`` (after optional KB), and only when
registrant looks trustee-like or corporate-with-address (same gates as Tavily — see
``tavily_owner_hint``).
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
    v = (os.getenv("FAA_LLM_ZOOMINFO_HINT_DISABLED") or "").strip().lower()
    return v in ("1", "true", "yes")


def suggest_zoominfo_company_from_faa_row(
    *,
    registrant_name: str,
    street: Optional[str],
    street2: Optional[str],
    city: Optional[str],
    state: Optional[str],
    zip_code: Optional[str],
    country: Optional[str],
    openai_api_key: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    """
    Returns dict like tavily_llm_synthesis: operating_company_name, website, phone, confidence,
    summary, suggested_zoominfo_query; or None on skip; or {"error": ...} on failure.
    """
    if not openai_api_key or not _strip(registrant_name):
        return None

    addr_parts = [_strip(street), _strip(street2), _strip(city), _strip(state), _strip(zip_code), _strip(country)]
    address_line = ", ".join(p for p in addr_parts if p)

    system = """You help aviation analysts choose a **ZoomInfo company search query** from FAA data only (no live web).

Context:
- FAA **registrant_name** is the **legal** record (often "… INC TRUSTEE", leasing company, or holding entity).
- The **operating brand** (who runs day-to-day marketing, website, services) may differ — e.g. "d/b/a" or a consumer-facing name at the **same mailing address**.
- Use your knowledge **only when** you can tie a specific operating name to this **exact pattern** of legal name + address (e.g. well-known trustee + city/ZIP combinations). If you are not confident, return confidence "none" or "low" and nulls.

Rules:
- **suggested_zoominfo_query**: short phrase for B2B search (usually the operating company name, not the full trustee string).
- Do **not** invent a website or phone unless you are highly confident from known public facts matching this registrant + address.
- Return **only** valid JSON, no markdown.

JSON schema:
{
  "operating_company_name": string or null,
  "website": string or null,
  "phone": string or null,
  "confidence": "high" | "medium" | "low" | "none",
  "summary": string (one or two sentences, may cite "typical structure" without claiming verified current fact if uncertain),
  "suggested_zoominfo_query": string or null
}"""

    user = (
        f"FAA registrant_name (legal):\n{registrant_name}\n\n"
        f"FAA mailing address:\n{address_line or '—'}\n"
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=500,
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
        logger.warning("FAA LLM ZoomInfo hint failed: %s", e)
        return {"error": str(e)[:400], "confidence": "none"}


def enrich_faa_owners_with_faa_llm_zoominfo_hint(owners_from_faa: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Attach ``faa_llm_zoominfo_hint`` when KB did not set ``trustee_operating_contact`` and
    registrant passes Tavily-style gates (trustee-like or TAVILY_WHEN_CORP_AND_ADDRESS).
    """
    if _disabled():
        return [_with_faa_llm_hint(r, None) for r in owners_from_faa]

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()
    if not api_key:
        return [_with_faa_llm_hint(r, None) for r in owners_from_faa]

    from services.tavily_owner_hint import has_address_context_for_search, should_run_tavily_for_registrant

    out: List[Dict[str, Any]] = []
    for row in owners_from_faa:
        r = dict(row)
        r["faa_llm_zoominfo_hint"] = None

        if r.get("trustee_operating_contact"):
            out.append(r)
            continue

        name = _strip(r.get("registrant_name"))
        if not name or not should_run_tavily_for_registrant(name):
            out.append(r)
            continue
        if not has_address_context_for_search(r):
            out.append(r)
            continue

        syn = suggest_zoominfo_company_from_faa_row(
            registrant_name=name,
            street=r.get("street"),
            street2=r.get("street2"),
            city=r.get("city"),
            state=r.get("state"),
            zip_code=r.get("zip_code"),
            country=r.get("country"),
            openai_api_key=api_key,
            model=model,
        )
        if syn and not syn.get("error"):
            r["faa_llm_zoominfo_hint"] = syn
        elif syn and syn.get("error"):
            r["faa_llm_zoominfo_hint"] = syn
        out.append(r)

    return out


def _with_faa_llm_hint(row: Dict[str, Any], hint: Any) -> Dict[str, Any]:
    r = dict(row)
    r["faa_llm_zoominfo_hint"] = hint
    return r
