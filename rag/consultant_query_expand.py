"""
Lightweight rule-based query rewrite for Ask Consultant (Tavily + Pinecone RAG).

Replaces LLM expansion: detects tails/serials, manufacturers, model tokens, and intent
(price / range / operator) to build 1–2 RAG strings and one Tavily-optimized string.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from services.tavily_owner_hint import clamp_tavily_query

logger = logging.getLogger(__name__)

# --- Manufacturers (substring or token match, lowercase) ---
_MANUFACTURER_KEYS: Tuple[Tuple[str, str], ...] = (
    ("gulfstream", "Gulfstream"),
    ("gulf stream", "Gulfstream"),
    ("bombardier", "Bombardier"),
    ("dassault", "Dassault"),
    ("falcon", "Dassault Falcon"),
    ("embraer", "Embraer"),
    ("phenom", "Embraer Phenom"),
    ("praetor", "Embraer Praetor"),
    ("legacy", "Embraer Legacy"),
    ("lineage", "Embraer Lineage"),
    ("cessna", "Cessna"),
    ("citation", "Cessna Citation"),
    ("longitude", "Cessna Longitude"),
    ("latitude", "Cessna Latitude"),
    ("king air", "King Air"),
    ("beechcraft", "Beechcraft"),
    ("hawker", "Hawker"),
    ("pilatus", "Pilatus"),
    ("pc-12", "Pilatus PC-12"),
    ("pc12", "Pilatus PC-12"),
    ("honda", "HondaJet"),
    ("hondajet", "HondaJet"),
    ("cirrus", "Cirrus"),
    ("vision jet", "Cirrus Vision"),
    ("global ", "Bombardier Global"),
    ("global 5000", "Global 5000"),
    ("global 6000", "Global 6000"),
    ("global 7500", "Global 7500"),
    ("challenger", "Challenger"),
    ("learjet", "Learjet"),
    ("atr ", "ATR"),
    ("airbus", "Airbus"),
    ("boeing", "Boeing"),
    ("grob", "Grob"),
    ("diamond", "Diamond"),
)

# Regex (pattern, canonical label) for models / families
_MODEL_REGEX: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(p, re.IGNORECASE), label)
    for p, label in (
        (r"\bg\s*[-.]?\s*650(?:\s*er)?\b", "G650"),
        (r"\bg\s*[-.]?\s*550\b", "G550"),
        (r"\bg\s*[-.]?\s*500\b", "G500"),
        (r"\bg\s*[-.]?\s*280\b", "G280"),
        (r"\bg\s*[-.]?\s*400\b", "G400"),
        (r"\bgvi\b", "G650"),
        (r"\bcitation\s+latitude\b", "Citation Latitude"),
        (r"\bcitation\s+longitude\b", "Citation Longitude"),
        (r"\bcitation\s+x\b", "Citation X"),
        (r"\bcj\s*[1-4]\b", "Citation CJ"),
        (r"\bcitation\s+m2\b", "Citation M2"),
        (r"\bphenom\s*100\b", "Phenom 100"),
        (r"\bphenom\s*300\b", "Phenom 300"),
        (r"\bcl\s*300\b", "Challenger 300"),
        (r"\bcl\s*350\b", "Challenger 350"),
        (r"\bcl\s*600\b", "Challenger 600"),
        (r"\bcl\s*650\b", "Challenger 650"),
        (r"\bfalcon\s*8x\b", "Falcon 8X"),
        (r"\bfalcon\s*7x\b", "Falcon 7X"),
        (r"\bfalcon\s*900\b", "Falcon 900"),
        (r"\bfalcon\s*2000\b", "Falcon 2000"),
        (r"\bprae?tor\s*500\b", "Praetor 500"),
        (r"\bprae?tor\s*600\b", "Praetor 600"),
        (r"\blegacy\s*500\b", "Legacy 500"),
        (r"\blegacy\s*600\b", "Legacy 600"),
        (r"\bvision\s*jet\b", "Vision Jet"),
    )
)

_TAIL_RE = re.compile(r"\b(N)(?:[- ]?)([A-Z0-9]{1,6})\b", re.IGNORECASE)
_EU_TAIL_RE = re.compile(
    r"\b([CGLOVSXB][A-Z]{1,2}|[OY]{2}|SE|LN)-?([A-Z0-9]{2,6})\b", re.IGNORECASE
)

_INTENT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "price": (
        "price",
        "pricing",
        "cost",
        "how much",
        "asking",
        "ask price",
        "usd",
        "dollar",
        "$",
        "for sale",
        "buy",
        "purchase",
        "listing",
        "market value",
        "valuation",
    ),
    "range": (
        "range",
        "nm",
        "nautical mile",
        "endurance",
        "fuel burn",
        "consumption",
        "cruise speed",
        "max cruise",
        "mach ",
        "specifications",
        "specs",
        "performance",
        "payload",
    ),
    "operator": (
        "who owns",
        "who own",
        "owner",
        "operate",
        "operator",
        "operated",
        "operating",
        "charter",
        "fleet",
        "registrant",
        "management",
        "aoc",
        "certificate holder",
    ),
}


def _norm_blob(user_query: str, history_snippet: Optional[str]) -> str:
    parts = []
    if (history_snippet or "").strip():
        parts.append(history_snippet.strip()[:3500])
    parts.append((user_query or "").strip())
    return "\n".join(parts)


def _detect_intents(blob_lc: str) -> Set[str]:
    found: Set[str] = set()
    for intent, needles in _INTENT_KEYWORDS.items():
        if any(n in blob_lc for n in needles):
            found.add(intent)
    return found


def _detect_manufacturers(blob_lc: str) -> List[str]:
    names: List[str] = []
    seen: Set[str] = set()
    for key, display_mfr in _MANUFACTURER_KEYS:
        if key in blob_lc and display_mfr.lower() not in seen:
            seen.add(display_mfr.lower())
            names.append(display_mfr)
    return names[:3]


def _detect_models(blob: str) -> List[str]:
    labels: List[str] = []
    seen: Set[str] = set()
    for pat, label in _MODEL_REGEX:
        if pat.search(blob):
            lk = label.lower()
            if lk not in seen:
                seen.add(lk)
                labels.append(label)
    return labels[:4]


def _extract_tails(blob: str) -> List[str]:
    tails: List[str] = []
    seen: Set[str] = set()
    for m in _TAIL_RE.finditer(blob):
        t = f"{m.group(1).upper()}{m.group(2).upper().replace(' ', '')}"
        if len(t) >= 3 and t not in seen:
            seen.add(t)
            tails.append(t)
    for m in _EU_TAIL_RE.finditer(blob):
        pre, suf = m.group(1).upper(), m.group(2).upper().replace(" ", "")
        t = f"{pre}-{suf}"
        if t not in seen and len(suf) >= 2:
            seen.add(t)
            tails.append(t)
    return tails[:4]


def _build_tavily_query(
    user_query: str,
    blob: str,
    blob_lc: str,
    tails: List[str],
    manufacturers: List[str],
    models: List[str],
    intents: Set[str],
) -> str:
    parts: List[str] = [(user_query or "").strip()]
    if tails:
        parts.append(" ".join(f'"{t}"' for t in tails))
    if manufacturers:
        parts.append(manufacturers[0])
    if models:
        parts.append(" ".join(models))

    suffix: List[str] = []
    if "operator" in intents:
        suffix.extend(
            ["owner", "operator", "registered owner", "charter", "fleet"]
        )
    if "price" in intents:
        suffix.extend(
            ["asking price", "for sale", "aircraft listing", "USD"]
        )
    if "range" in intents:
        suffix.extend(["range", "cruise", "specifications", "nm"])

    if "photo" in blob_lc or "image" in blob_lc or "gallery" in blob_lc or "picture" in blob_lc:
        suffix.extend(["aircraft photos", "exterior", "interior"])

    core = " ".join(p for p in parts if p).strip()
    if suffix:
        core = f"{core} {' '.join(suffix)}".strip()

    return clamp_tavily_query(core) if core else clamp_tavily_query(user_query or "")


def _build_rag_queries(
    user_query: str,
    manufacturers: List[str],
    models: List[str],
    tails: List[str],
    intents: Set[str],
) -> List[str]:
    """Return 1–2 concise strings for embedding search."""
    q = (user_query or "").strip()
    pieces: List[str] = []
    if manufacturers:
        pieces.append(manufacturers[0])
    if models:
        pieces.extend(models[:2])
    if tails:
        pieces.append(tails[0])

    base = " ".join(dict.fromkeys(pieces)) if pieces else ""
    intent_bits: List[str] = []
    if "price" in intents:
        intent_bits.append("asking price for sale")
    if "range" in intents:
        intent_bits.append("range cruise specifications")
    if "operator" in intents:
        intent_bits.append("operator owner registrant")

    primary = " ".join(x for x in [base, q] if x).strip() or q
    head = " ".join(primary.split()[:14])
    if intent_bits:
        secondary = f"{head} {' '.join(intent_bits)}".strip()
    else:
        secondary = ""

    if len(primary) > 220:
        primary = primary[:217] + "..."
    if len(secondary) > 220:
        secondary = secondary[:217] + "..."

    out: List[str] = [primary]
    if secondary and secondary.lower() != primary.lower():
        out.append(secondary)
    dedup: List[str] = []
    seen: Set[str] = set()
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(s)
        if len(dedup) >= 2:
            break
    return dedup or ([q] if q else [""])


def expand_consultant_research_queries(
    user_query: str,
    openai_api_key: str,
    chat_model: str,
    history_snippet: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns ``{"tavily_query": str, "rag_queries": List[str]}`` (1–2 RAG queries).

    ``openai_api_key`` / ``chat_model`` are kept for API compatibility; they are not used.

    Set ``CONSULTANT_QUERY_EXPAND_LLM=1`` to restore the previous OpenAI JSON expansion
    (slower, ~1–2s extra latency).
    """
    q = (user_query or "").strip()
    base_rag = [q] if q else []
    default: Dict[str, Any] = {
        "tavily_query": clamp_tavily_query(q) if q else "",
        "rag_queries": base_rag or [""],
    }
    if not q:
        return default

    if (os.getenv("CONSULTANT_QUERY_EXPAND_LLM") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return _expand_consultant_research_queries_llm(
            user_query, openai_api_key, chat_model, history_snippet
        )

    return _expand_consultant_research_queries_rules(user_query, history_snippet)


def _expand_consultant_research_queries_rules(
    user_query: str,
    history_snippet: Optional[str] = None,
) -> Dict[str, Any]:
    q = (user_query or "").strip()
    blob = _norm_blob(q, history_snippet)
    blob_lc = blob.lower()

    tails = _extract_tails(blob)
    manufacturers = _detect_manufacturers(blob_lc)
    models = _detect_models(blob)
    intents = _detect_intents(blob_lc)

    tq = _build_tavily_query(q, blob, blob_lc, tails, manufacturers, models, intents)
    rag_list = _build_rag_queries(q, manufacturers, models, tails, intents)
    final_rag = (rag_list[:2] if rag_list else [q])[:2]

    return {
        "tavily_query": clamp_tavily_query(tq or q),
        "rag_queries": final_rag,
    }


def _expand_consultant_research_queries_llm(
    user_query: str,
    openai_api_key: str,
    chat_model: str,
    history_snippet: Optional[str] = None,
) -> Dict[str, Any]:
    """Legacy LLM path (opt-in via CONSULTANT_QUERY_EXPAND_LLM=1)."""
    q = (user_query or "").strip()
    base_rag = [q] if q else []
    default: Dict[str, Any] = {
        "tavily_query": clamp_tavily_query(q) if q else "",
        "rag_queries": base_rag or [""],
    }
    if not openai_api_key or not q:
        return default

    try:
        import json

        import openai

        try:
            expand_timeout = float((os.getenv("CONSULTANT_EXPAND_TIMEOUT_SEC") or "18").strip())
            expand_timeout = max(5.0, min(45.0, expand_timeout))
        except ValueError:
            expand_timeout = 18.0
        expand_model = (os.getenv("CONSULTANT_EXPAND_MODEL") or "").strip() or chat_model

        client = openai.OpenAI(api_key=openai_api_key, timeout=expand_timeout)
        instruction = """You help an aviation research assistant run (1) a public web search (Tavily) and (2) semantic search over a private database of aircraft listings, sales, and registry-related records.

Given the user's question, respond with ONLY a JSON object (no markdown fences) with exactly these keys:
- "tavily_query": one concise English string optimized for web search. Always include any serial, tail/registration, and make/model if inferable (including from recent conversation if provided). For ownership / "who owns" / operator questions, add terms like: owner, operator, registered owner, registrant, AOC, air operator certificate, charter, aircraft management, fleet — so results hit operator fleet pages and registry excerpts, not only Wikipedia. Put the tail/registration in quotes when it is alphanumeric (e.g. "OY-JSW"). For European tails (OY-, SE-, LN-, G-), add the country civil aviation authority / register name when known from context. For purchase / "can I buy" / price / for-sale / listing questions, add: asking price, for sale, aircraft listing, USD (and broker or marketplace names if natural) so snippets include **live listing pages with prices** (AvPay, Controller, JetNet, etc.), not only registry data. If the user asks for **photos, images, pictures, or gallery** of the aircraft, add terms like: aircraft photos, exterior, cabin interior, and keep the quoted tail/serial so image-capable search returns relevant hits.
- "rag_queries": an array of 2 to 4 short alternative search phrases for embedding search (synonyms, model variants, registration format, "Citation", "Gulfstream", etc. as relevant). If the user asks about buying, price, or availability, include at least one phrase with **asking price** or **for sale** plus serial or tail when known from the message or conversation.

Keep strings under 200 characters each where possible."""

        user_block = q
        hs = (history_snippet or "").strip()
        if hs:
            user_block = (
                "Recent conversation (use to resolve aircraft identity on short follow-ups):\n"
                f"{hs[:3500]}\n\nCurrent message:\n{q}"
            )

        resp = client.chat.completions.create(
            model=expand_model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_block},
            ],
            max_tokens=300,
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
        return {"tavily_query": clamp_tavily_query(tq), "rag_queries": rag_list[:6]}
    except Exception as e:
        logger.warning("consultant LLM query expand failed, using rules: %s", e)
        return _expand_consultant_research_queries_rules(user_query, history_snippet)


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
    cap = max(4, min(20, int(max_items)))
    body_cap = max(400, min(2600, int(max_body_chars)))
    disclaimer = (payload.get("disclaimer") or "").strip()
    lines = [
        "[WEB SEARCH — Tavily (third-party sources; unverified; may be incomplete or wrong)]",
        "These results may combine two search passes (broad + owner/operator-focused). Read ALL numbered results before concluding.",
        "When the user asks who owns or operates this aircraft: prefer snippets that explicitly tie THIS tail/serial to a company (registry excerpt, AOC holder, fleet page, operator press release). "
        "If a charter/airline/management company is named on a fleet or operator page for this exact registration, treat that as strong evidence for who operates or commercially manages the aircraft (often different wording from a bare legal registrant). "
        "If one company appears on fleet/operator pages and another only on generic registry aggregators, weight the fleet/operator evidence more heavily for 'who operates' questions. Name companies exactly as written in snippets.",
        "For purchase / pricing questions: pull explicit **dollar amounts** and **listing URLs** from snippets when present; the assistant must repeat them in the answer with source (result # / domain).",
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
    imgs = payload.get("images") if isinstance(payload, dict) else None
    if isinstance(imgs, list) and imgs:
        lines.append("")
        lines.append(
            "[WEB — Image URLs from Tavily search (third-party; verify tail/serial matches before showing as this aircraft)]"
        )
        for j, im in enumerate(imgs[:14], 1):
            if not isinstance(im, dict):
                continue
            iu = (im.get("url") or "").strip()
            if not iu:
                continue
            lab = (im.get("description") or "").strip()
            lines.append(f"  Image {j}: {iu}" + (f" — {lab}" if lab else ""))
    return "\n".join(lines) if len(lines) > 1 else "\n".join(lines) + "\n(no web results returned.)"


def merge_tavily_consultant_payloads(
    primary: Dict[str, Any],
    secondary: Dict[str, Any],
    *,
    max_results: int = 12,
) -> Dict[str, Any]:
    """
    Combine two Tavily responses (e.g. expanded query + registration-focused query).
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

    img_merged: List[Dict[str, Any]] = []

    def _add_imgs(src: Dict[str, Any]) -> None:
        raw = src.get("images") if isinstance(src, dict) else None
        if not isinstance(raw, list):
            return
        seen_url: set[str] = {str(x.get("url") or "").strip() for x in img_merged if isinstance(x, dict)}
        for it in raw:
            if isinstance(it, dict):
                u = str(it.get("url") or "").strip()
                if not u.startswith("http") or u in seen_url:
                    continue
                seen_url.add(u)
                img_merged.append(
                    {
                        "url": u,
                        "description": (str(it.get("description")).strip() if it.get("description") else None),
                    }
                )
            elif isinstance(it, str) and it.strip().startswith("http"):
                u = it.strip()
                if u in seen_url:
                    continue
                seen_url.add(u)
                img_merged.append({"url": u, "description": None})
            if len(img_merged) >= 28:
                return

    _add_imgs(primary)
    _add_imgs(secondary)

    return {
        "query": qm[:800] if qm else None,
        "disclaimer": (primary.get("disclaimer") or secondary.get("disclaimer") or "").strip() or None,
        "results": merged,
        "images": img_merged,
        "error": err_out,
    }
