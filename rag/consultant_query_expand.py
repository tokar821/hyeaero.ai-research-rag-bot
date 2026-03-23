"""
LLM-assisted search query expansion for Ask Consultant (Tavily + Pinecone RAG).

Produces a web search string and 2–4 paraphrases for vector retrieval so RAG is not tied
to the user's exact wording.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def expand_consultant_research_queries(
    user_query: str,
    openai_api_key: str,
    chat_model: str,
    history_snippet: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns ``{"tavily_query": str, "rag_queries": List[str]}``.

    ``history_snippet``: recent user/assistant text so follow-ups like "tell me more" still include
    tail/serial/model in the web search string.

    On any failure, falls back to using the raw user question for both paths.
    """
    q = (user_query or "").strip()
    base_rag = [q] if q else []
    default: Dict[str, Any] = {
        "tavily_query": q[:400] if q else "",
        "rag_queries": base_rag or [""],
    }
    if not openai_api_key or not q:
        return default

    try:
        import openai

        client = openai.OpenAI(api_key=openai_api_key, timeout=35.0)
        instruction = """You help an aviation research assistant run (1) a public web search (Tavily) and (2) semantic search over a private database of aircraft listings, sales, and registry-related records.

Given the user's question, respond with ONLY a JSON object (no markdown fences) with exactly these keys:
- "tavily_query": one concise English string optimized for web search. Always include any serial, tail/registration, and make/model if inferable (including from recent conversation if provided). For ownership / "who owns" / operator questions, add terms like: owner, operator, registered owner, registrant, AOC, air operator certificate, charter, aircraft management, fleet — so results hit operator fleet pages and registry excerpts, not only Wikipedia. Put the tail/registration in quotes when it is alphanumeric (e.g. "OY-JSW"). For European tails (OY-, SE-, LN-, G-), add the country civil aviation authority / register name when known from context.
- "rag_queries": an array of 2 to 4 short alternative search phrases for embedding search (synonyms, model variants, registration format, "Citation", "Gulfstream", etc. as relevant).

Keep strings under 200 characters each where possible."""

        user_block = q
        hs = (history_snippet or "").strip()
        if hs:
            user_block = (
                "Recent conversation (use to resolve aircraft identity on short follow-ups):\n"
                f"{hs[:3500]}\n\nCurrent message:\n{q}"
            )

        resp = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_block},
            ],
            max_tokens=350,
            temperature=0.25,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        obj = json.loads(text)
        tq = str(obj.get("tavily_query") or "").strip() or q
        rq = obj.get("rag_queries")
        rag_list: List[str] = []
        if isinstance(rq, list):
            for x in rq:
                s = str(x).strip()
                if s and s not in rag_list:
                    rag_list.append(s)
        if q not in rag_list:
            rag_list.insert(0, q)
        if not rag_list:
            rag_list = [q]
        return {"tavily_query": tq[:500], "rag_queries": rag_list[:6]}
    except Exception as e:
        logger.warning("consultant query expand failed, using raw query: %s", e)
        return default


def format_tavily_payload_for_consultant(
    payload: Dict[str, Any],
    *,
    max_items: int = 12,
    max_body_chars: int = 1100,
) -> str:
    """Turn :func:`services.tavily_owner_hint.fetch_tavily_hints_for_query` output into LLM context."""
    if not isinstance(payload, dict):
        return "[WEB — Tavily: no payload]"
    err = payload.get("error")
    results = payload.get("results") or []
    cap = max(4, min(15, int(max_items)))
    body_cap = max(400, min(2000, int(max_body_chars)))
    disclaimer = (payload.get("disclaimer") or "").strip()
    lines = [
        "[WEB SEARCH — Tavily (third-party sources; unverified; may be incomplete or wrong)]",
        "These results may combine two search passes (broad + owner/operator-focused). Read ALL numbered results before concluding.",
        "When the user asks who owns or operates this aircraft: prefer snippets that explicitly tie THIS tail/serial to a company (registry excerpt, AOC holder, fleet page, operator press release). "
        "If a charter/airline/management company is named on a fleet or operator page for this exact registration, treat that as strong evidence for who operates or commercially manages the aircraft (often different wording from a bare legal registrant). "
        "If one company appears on fleet/operator pages and another only on generic registry aggregators, weight the fleet/operator evidence more heavily for 'who operates' questions. Name companies exactly as written in snippets.",
    ]
    if disclaimer:
        lines.append(disclaimer)
    if err and err not in ("tavily_disabled", "tavily_api_key_missing") and not results:
        lines.append(f"(Tavily note: {err})")
    elif err in ("tavily_disabled", "tavily_api_key_missing") and not results:
        lines.append(f"(Tavily skipped: {err})")
    for i, r in enumerate(results[:cap], 1):
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or "").strip() or "Result"
        body = (r.get("content") or "").strip()
        if len(body) > body_cap:
            body = body[: body_cap - 3] + "..."
        url = (r.get("url") or "").strip()
        lines.append(f"{i}. {title}")
        if body:
            lines.append(f"   {body}")
        if url:
            lines.append(f"   URL: {url}")
    return "\n".join(lines) if len(lines) > 1 else "\n".join(lines) + "\n(no web results returned.)"


def merge_tavily_consultant_payloads(
    primary: Dict[str, Any],
    secondary: Dict[str, Any],
    *,
    max_results: int = 12,
) -> Dict[str, Any]:
    """
    Combine two Tavily responses (e.g. LLM-expanded query + registration-focused query).
    Deduplicates by URL; preserves primary order then appends new rows from secondary.
    """
    seen: set[str] = set()
    merged: List[Dict[str, Optional[str]]] = []

    def _add_from(payload: Dict[str, Any]) -> None:
        for src in payload.get("results") or []:
            if not isinstance(src, dict):
                continue
            if len(merged) >= max_results:
                return
            url = (src.get("url") or "").strip()
            key = url if url else str(hash((src.get("title"), (src.get("content") or "")[:100])))
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "title": (src.get("title") or None) and str(src.get("title")).strip() or None,
                    "url": (src.get("url") or None) and str(src.get("url")).strip() or None,
                    "content": (src.get("content") or None) and str(src.get("content")).strip() or None,
                }
            )

    _add_from(primary)
    _add_from(secondary)

    q1 = (primary.get("query") or "").strip()
    q2 = (secondary.get("query") or "").strip()
    qm = f"{q1} || {q2}" if q2 and q2.lower() != q1.lower() else q1

    err_out = None
    if not merged:
        err_out = primary.get("error") or secondary.get("error")

    return {
        "query": qm[:800] if qm else None,
        "disclaimer": (primary.get("disclaimer") or secondary.get("disclaimer") or "").strip() or None,
        "results": merged,
        "error": err_out,
    }
