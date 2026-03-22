"""
LLM pass over Tavily web snippets → suggested operating company for FAA trustee-style rows.

Used after :func:`services.tavily_owner_hint.enrich_faa_owners_with_tavily_hints`. When the model
returns a usable ``company_name`` with sufficient confidence, :mod:`api.main` adds a ZoomInfo lookup
item (``tavily_llm_hint``) so owner details include enriched company data.

Disable with ``TAVILY_LLM_DISABLED=1``. Requires ``OPENAI_API_KEY``.

``TAVILY_LLM_ZOOMINFO_MIN_CONFIDENCE`` — ``high`` | ``medium`` | ``low`` (default ``medium``):
only inject ZoomInfo lookup when LLM confidence is at or above this level.
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


def _faa_address_block(row: Dict[str, Any]) -> str:
    parts = [
        _strip(row.get("street")),
        _strip(row.get("street2")),
        _strip(row.get("city")),
        _strip(row.get("state")),
        _strip(row.get("zip_code")),
        _strip(row.get("country")),
    ]
    return ", ".join(p for p in parts if p)


def _tavily_results_text(results: List[Any], max_chars_per_hit: int = 900) -> str:
    lines: List[str] = []
    for i, hit in enumerate(results[:6], 1):
        if not isinstance(hit, dict):
            continue
        t = _strip(hit.get("title"))
        u = _strip(hit.get("url"))
        c = _strip(hit.get("content"))
        if c and len(c) > max_chars_per_hit:
            c = c[: max_chars_per_hit - 3] + "..."
        lines.append(f"--- Result {i} ---\nTitle: {t}\nURL: {u}\nSnippet: {c}\n")
    return "\n".join(lines)


def _parse_llm_json_object(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    text = raw.strip()
    # Strip ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Fallback: first {...}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _confidence_rank(c: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3}.get((c or "").lower().strip(), 0)


def _min_confidence_for_zoominfo() -> int:
    raw = (os.getenv("TAVILY_LLM_ZOOMINFO_MIN_CONFIDENCE") or "medium").strip().lower()
    return _confidence_rank(raw) if _confidence_rank(raw) > 0 else _confidence_rank("medium")


def infer_company_from_tavily_llm(
    faa_row: Dict[str, Any],
    tavily_payload: Dict[str, Any],
    openai_api_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Return a normalized suggestion dict or None if skipped / failed.

    Keys: company_name, website, confidence, rationale (all optional except confidence).
    """
    if (os.getenv("TAVILY_LLM_DISABLED") or "").strip().lower() in ("1", "true", "yes"):
        return None
    if not openai_api_key:
        return None

    results = tavily_payload.get("results") if isinstance(tavily_payload, dict) else None
    if not isinstance(results, list) or not results:
        return None

    reg = _strip(faa_row.get("registrant_name"))
    if not reg:
        return None

    addr = _faa_address_block(faa_row)
    body = _tavily_results_text(results)

    prompt = (
        "You help analysts link an FAA aircraft registrant (often a title trustee / shell) to an "
        "operating company using web search snippets.\n\n"
        "FAA registrant name:\n"
        f"{reg}\n\n"
        "FAA mailing address on file:\n"
        f"{addr or '(none)'}\n\n"
        "Web search results (may include generic industry pages that do NOT match this specific registrant):\n"
        f"{body}\n\n"
        "Task: If a snippet clearly ties THIS registrant or THIS address to a specific operating brand "
        "(e.g. d/b/a, same street, company blog naming the trustee), output that entity. "
        "If results are only generic 'aircraft trust services' with no link to this name/address, "
        'use confidence \"none\" and null company_name.\n\n'
        "Respond with ONLY a JSON object, no markdown, keys:\n"
        '- "company_name": string or null (best operating / marketing name to search in a B2B database)\n'
        '- "website": string or null (canonical https URL if inferable from snippets)\n'
        '- "confidence": one of high, medium, low, none\n'
        '- "rationale": one short sentence\n'
    )

    try:
        import openai

        client = openai.OpenAI(api_key=openai_api_key, timeout=45.0)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1,
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = _parse_llm_json_object(raw)
        if not obj:
            logger.info("tavily_llm_bridge: no JSON from LLM")
            return None

        company_name = _strip(obj.get("company_name")) or None
        website = _strip(obj.get("website")) or None
        confidence = (_strip(obj.get("confidence")) or "none").lower()
        if confidence not in ("high", "medium", "low", "none"):
            confidence = "none"
        rationale = _strip(obj.get("rationale")) or None

        if confidence == "none" or not company_name:
            return {
                "company_name": None,
                "website": website,
                "confidence": confidence,
                "rationale": rationale,
                "matched_by": "tavily_llm",
            }

        return {
            "company_name": company_name,
            "website": website,
            "confidence": confidence,
            "rationale": rationale,
            "matched_by": "tavily_llm",
        }
    except Exception as e:
        logger.warning("tavily_llm_bridge: LLM call failed: %s", e)
        return None


def suggestion_qualifies_for_zoominfo(suggestion: Optional[Dict[str, Any]]) -> bool:
    if not suggestion or not isinstance(suggestion, dict):
        return False
    cn = _strip(suggestion.get("company_name"))
    if not cn:
        return False
    conf = _strip(suggestion.get("confidence")) or "none"
    return _confidence_rank(conf) >= _min_confidence_for_zoominfo()


def apply_tavily_llm_company_suggestions(
    owners_from_faa: List[Dict[str, Any]],
    openai_api_key: str,
) -> List[Dict[str, Any]]:
    """
    For each row with ``tavily_web_hints`` results, set ``tavily_llm_company_suggestion`` (may be
    partial with confidence none).
    """
    out: List[Dict[str, Any]] = []
    for row in owners_from_faa:
        r = dict(row)
        r.setdefault("tavily_llm_company_suggestion", None)

        tw = r.get("tavily_web_hints")
        if not isinstance(tw, dict):
            out.append(r)
            continue
        if not (tw.get("results") or []):
            out.append(r)
            continue

        sug = infer_company_from_tavily_llm(r, tw, openai_api_key)
        if sug:
            r["tavily_llm_company_suggestion"] = sug
        out.append(r)

    return out
