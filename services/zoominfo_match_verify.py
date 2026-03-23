"""
Tavily + LLM verification for PhlyData ZoomInfo company matches (FAA registrant vs ZoomInfo profile).

Used when matching is uncertain (e.g. vector+LLM fallback) to avoid showing wrong homonyms
("Clydesdale Capital" vs "Clydesdale Asset Management LLC").
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _strip(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _zoominfo_company_summary(record: Dict[str, Any]) -> str:
    attrs = record.get("attributes") or {}
    lines: List[str] = []
    n = _strip(attrs.get("name") or attrs.get("companyName"))
    if n:
        lines.append(f"Company name: {n}")
    for k in ("website", "phone", "mainPhone", "directPhone"):
        v = _strip(attrs.get(k))
        if v:
            lines.append(f"{k}: {v}")
            break
    addr = [
        _strip(attrs.get("addressLine1")),
        _strip(attrs.get("city")),
        _strip(attrs.get("state")),
        _strip(attrs.get("zipCode")),
        _strip(attrs.get("country")),
    ]
    loc = ", ".join(x for x in addr if x)
    if loc:
        lines.append(f"Headquarters / address: {loc}")
    return "\n".join(lines) if lines else str(attrs)[:800]


def verify_faa_zoominfo_company_match(
    item: Dict[str, Any],
    zoominfo_record: Dict[str, Any],
    *,
    openai_api_key: str,
    chat_model: str,
    strict: bool = True,
) -> Tuple[bool, str, int]:
    """
    Run a focused Tavily search + LLM judgment.

    ``strict`` (default True): require web snippets + high-confidence LLM agreement; on Tavily/LLM
    failure return **reject** so we do not show homonyms (e.g. Clydesdale Capital vs Asset Management LLC).

    Returns:
        (accept, short_reason, tavily_hit_count)
    """
    if (os.getenv("ZOOMINFO_TAVILY_VERIFY_DISABLED") or "").strip().lower() in ("1", "true", "yes"):
        return True, "verification_disabled", 0

    reg_name = _strip(item.get("company_name"))
    if not reg_name:
        return True, "no_registrant_name", 0

    city = _strip(item.get("city"))
    state = _strip(item.get("state"))
    street = _strip(item.get("street"))
    z = _strip(item.get("zip_code"))

    # Search geared to legal entity + location, not only the ZoomInfo brand name
    tq_parts = [f'"{reg_name}"', city, state, "company", "aircraft", "owner"]
    if street and len(street) < 80:
        tq_parts.insert(1, street.split(",")[0][:60])
    tavily_query = " ".join(x for x in tq_parts if x)[:480]

    try:
        from services.tavily_owner_hint import fetch_tavily_hints_for_query

        depth = None
        if (os.getenv("ZOOMINFO_VERIFY_TAVILY_ADVANCED") or "").strip().lower() in ("1", "true", "yes"):
            depth = "advanced"
        payload = fetch_tavily_hints_for_query(tavily_query, result_limit=8, search_depth=depth)
    except Exception as e:
        logger.warning("ZoomInfo verify: Tavily failed: %s", e)
        if strict:
            return False, "tavily_error", 0
        return True, "tavily_failed_open", 0

    results = payload.get("results") or []
    if not isinstance(results, list):
        results = []
    nhit = len(results)
    if strict and nhit == 0:
        return False, "no_web_snippets", 0

    tavily_lines: List[str] = []
    for i, r in enumerate(results[:8], 1):
        if not isinstance(r, dict):
            continue
        title = _strip(r.get("title")) or "Result"
        body = _strip(r.get("content")) or ""
        if len(body) > 650:
            body = body[:647] + "..."
        url = _strip(r.get("url")) or ""
        tavily_lines.append(f"[{i}] {title}\n{body}\nURL: {url}")
    tavily_block = "\n\n".join(tavily_lines) if tavily_lines else "(no web results)"

    zi_block = _zoominfo_company_summary(zoominfo_record)

    faa_block = "\n".join(
        [
            f"Legal registrant name: {reg_name}",
            f"Mailing street: {street or '—'}",
            f"Mailing city/state/ZIP: {city or '—'}, {state or '—'}, {z or '—'}",
        ]
    )

    instruction = """You are a senior aviation compliance analyst. Your job is to avoid WRONG company matches.

The FAA block is the legal registrant record. The ZoomInfo block is a B2B database candidate.

Rules:
- "Similar name" is NOT enough. Examples of DIFFERENT entities: "Clydesdale Asset Management LLC" vs "Clydesdale Capital"; "Acme Aviation LLC" vs "Acme Holdings Inc." unless web text clearly says they are the same organization.
- If ZoomInfo HQ state differs from FAA mailing state (both US), that is a red flag unless snippets prove same entity (e.g. parent/subsidiary with explicit link).
- If web snippets describe a different company than the legal name, reject.

Reply with ONLY a JSON object (no markdown):
{"same_entity": true or false, "confidence": "high" or "medium" or "low", "reason": "one sentence"}

Set same_entity to false unless you would stake professional reputation they are the same legal entity.
If confidence would be "low" or "medium", set same_entity to false unless web snippets explicitly equate the two names (same org, DBA, or parent/subsidiary with full legal names)."""

    user = f"""{faa_block}

ZoomInfo candidate:
{zi_block}

Web search snippets:
{tavily_block}

Return only the JSON object."""

    try:
        import openai

        client = openai.OpenAI(api_key=openai_api_key, timeout=45.0)
        resp = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        data = json.loads(text)
    except Exception as e:
        logger.warning("ZoomInfo verify: LLM parse failed: %s", e)
        if strict:
            return False, "llm_verify_failed", nhit
        return True, "llm_verify_failed_open", nhit

    same = bool(data.get("same_entity"))
    conf = str(data.get("confidence") or "").strip().lower()
    reason = _strip(data.get("reason")) or "no reason given"

    if strict:
        # Quality-first: only accept high confidence, or medium with enough corroborating web hits
        accept = same and (conf == "high" or (conf == "medium" and nhit >= 3))
    else:
        accept = same and conf in ("high", "medium")
    if not accept:
        logger.info("ZoomInfo verify: REJECT same_entity=%s conf=%s reason=%s", same, conf, reason[:120])
    else:
        logger.info("ZoomInfo verify: ACCEPT conf=%s", conf)

    return accept, reason, nhit
