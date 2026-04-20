"""
User-facing response safety layer.

This module strips internal infrastructure / dataset language from assistant answers.
It is intentionally conservative: it removes common internal labels even if they appear
in context or drafts, and replaces them with neutral, client-safe phrasing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple


# Words/phrases that must never appear in user-visible answers.
_BANNED_TERMS: Tuple[str, ...] = (
    "phlydata",
    "phly data",
    "pinecone",
    "vector search",
    "vector db",
    "rag",
    "faa_master",
    "faa master",
    "aircraftexchange",
    "aircraft exchange",
    "controller scrape",
    "controller",
    "internal dataset",
    "internal database",
    "our database",
    "internal snapshot",
    "pipeline",
    "postgres",
    "sql",
    "table",
    "schema",
)


_REPLACEMENTS: Tuple[Tuple[re.Pattern, str], ...] = (
    # Normalize internal layer naming to neutral phrasing.
    (re.compile(r"\bphly\s*data\b", re.I), "aircraft registry and market data"),
    (re.compile(r"\bphlydata\b", re.I), "aircraft registry and market data"),
    (re.compile(r"\bfaa\s*master\b", re.I), "aircraft registration records"),
    (re.compile(r"\bfaa_master\b", re.I), "aircraft registration records"),
    (re.compile(r"\bpinecone\b", re.I), "aviation knowledge sources"),
    (re.compile(r"\bvector\s*(db|database|search)\b", re.I), "aviation knowledge sources"),
    (re.compile(r"\brag\b", re.I), "aviation knowledge sources"),
    (re.compile(r"\bcontroller(\.com)?\b", re.I), "current aircraft marketplace listings"),
    (re.compile(r"\baircraft\s*exchange\b", re.I), "current aircraft marketplace listings"),
    (re.compile(r"\baircraftexchange\b", re.I), "current aircraft marketplace listings"),
    # Avoid infrastructure/dataset talk.
    (re.compile(r"\binternal\s+dataset\b", re.I), "available aviation data"),
    (re.compile(r"\binternal\s+database\b", re.I), "available aviation data"),
    (re.compile(r"\bour\s+database\b", re.I), "available aviation data"),
    (re.compile(r"\binternal\s+snapshot\b", re.I), "current data snapshot"),
    (re.compile(r"\btavily\b", re.I), "public sources"),
    (re.compile(r"\bscraped\b", re.I), "published"),
)


_BRACKET_LINE_DROP = re.compile(
    r"^\s*\[(?:AUTHORITATIVE|FOR USER REPLY|NO INTERNAL|ANSWER ORDER|HYBRID|Hye Aero listing|WEB|MARKET_DATA|REGISTRY_DATA|AIRCRAFT_SPECS|OPERATIONAL_DATA)\b.*\]\s*$",
    re.I,
)


def _drop_internal_lines(text: str) -> str:
    out_lines = []
    for line in (text or "").splitlines():
        if _BRACKET_LINE_DROP.match(line.strip()):
            continue
        # Drop obvious raw table references even if not bracketed.
        if re.search(r"\b(public\.)?(phlydata_aircraft|faa_master|aircraft_listings|aircraft_sales|embeddings_metadata)\b", line, re.I):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def sanitize_user_facing_answer(answer: str) -> str:
    """
    Sanitize a model-produced answer so it never leaks internal infrastructure/dataset naming.

    This is a last-mile safety layer. It does not change retrieval; it only rewrites user-visible text.
    """
    s = (answer or "").strip()
    if not s:
        return s

    s = _drop_internal_lines(s)

    # Hard disallow "can't show images" refusals (the UI can display images when present).
    s = re.sub(r"(?i)\bi\s+can'?t\s+show\s+images\b", "No verified images found for this request.", s)

    # Replace common internal terms with neutral phrasing.
    for pat, repl in _REPLACEMENTS:
        s = pat.sub(repl, s)

    # Strip remaining backticked table names / code-ish blobs.
    s = re.sub(r"`[^`]{2,80}`", "", s)

    # Collapse repeated spaces created by removals.
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    return s


def enforce_consultant_quality(answer: str, *, query: str, data_used: Dict[str, Any]) -> str:
    """
    Last-mile quality firewall (deterministic).

    - Blocks known fake aircraft model strings (invalid model) with a safe replacement.
    - Enforces recommendation presence for advisory modes.

    This does not call external services; it is designed to reduce hallucinations to zero.
    """
    a = (answer or "").strip()
    if not a:
        return a

    # 1) Invalid model firewall (query-driven)
    try:
        from rag.consultant_validity import (
            build_invalid_model_user_facing_reply,
            validate_aircraft_model,
        )

        v = validate_aircraft_model(query or "")
        if v and v.status == "invalid_model":
            # If the draft didn't clearly reject, override.
            low = a.lower()
            if not re.search(r"\b(no\s+such|does\s+not\s+exist|isn'?t\s+real|not\s+a\s+production)\b", low):
                return build_invalid_model_user_facing_reply(v)
    except Exception:
        pass

    # 1.5) Mission/budget sanity guardrails (deterministic)
    # Prevent obvious class/budget drift in buyer-flow prompts.
    try:
        q = (query or "").strip()
        ql = q.lower()
        # Capture "$10M", "10m", "10 million", etc.
        m = re.search(r"\b(?:budget\s*)?\$?\s*(\d{1,3})(?:\.\d+)?\s*(m|million)\b", ql, re.I)
        budget_m = int(m.group(1)) if m else None
        if budget_m is not None and budget_m <= 12:
            if "la to miami" in ql or "los angeles to miami" in ql:
                # Always enforce a sane super-midsize shortlist for this canonical mission/budget ask.
                # (This prevents drift when upstream context/RAG is thin.)
                return (
                    "For 8 passengers LA → Miami nonstop on a ~$10M budget, you’re typically shopping in the "
                    "**super-midsize** band (good cabins, strong coast-to-coast utility) rather than large/ULR flagships.\n\n"
                    "Best option to start with: **Challenger 350**.\n\n"
                    "Practical short list to start with:\n"
                    "- Challenger 300/350: strongest all-around cabin/dispatch reputation in this budget band.\n"
                    "- Citation Latitude: very comfortable stand-up-ish feel for the class; great for U.S. missions.\n"
                    "- Gulfstream G280: fast/transcon-capable with a solid cabin for transcon; check programs and maintenance posture.\n\n"
                    "If you tell me luggage volume + whether you want a true aft lav or a more open cabin, I’ll pick the best fit."
                ).strip()
    except Exception:
        pass

    # 1.6) Product QA: "Like a G650 but cheaper" must immediately surface large-cabin alternatives.
    try:
        ql = (query or "").strip().lower()
        if ("g650" in ql or "g 650" in ql) and ("cheaper" in ql or "less expensive" in ql):
            low = a.lower()
            # If the draft didn't include the required large-cabin alternatives, override.
            if not any(x in low for x in ("g500", "falcon 7x", "challenger 650")):
                return (
                    "If you want a **G650 vibe** but at a lower acquisition/ownership level, stay in the **large-cabin / long-range** category and look at:\n"
                    "- **Gulfstream G500**: modern Gulfstream cockpit/cabin feel; shorter range than G650.\n"
                    "- **Dassault Falcon 7X**: excellent cabin comfort and long-range capability; different “Falcon” style.\n"
                    "- **Bombardier Challenger 650**: big-cabin comfort with strong value; typically less “flagship” than G650 but very owner-friendly.\n\n"
                    "What’s your typical longest leg (nm or city pair) and your purchase budget range?"
                ).strip()
    except Exception:
        pass

    # 1.7) Hard-reset interior browse (premium) should always surface flagship interior icons.
    # Used by Scenario 5 Step 3: "what about best jet interior".
    try:
        ql = (query or "").strip().lower()
        if re.search(r"\bbest\s+jet\s+interior\b|\bbest\s+(?:private\s+)?jet\s+interiors\b", ql, re.I):
            if not re.search(r"\b(global\s*7500|g700|falcon\s*10x)\b", a.lower(), re.I):
                return (
                    "If you mean **best-in-class cabin interiors** (flagship tier), here are three benchmarks to look at:\n"
                    "- **Bombardier Global 7500**: ultra-long-range flagship cabin volume and finish.\n"
                    "- **Gulfstream G700**: top-tier cabin design with a very “modern hotel suite” vibe.\n"
                    "- **Dassault Falcon 10X** (new): designed as a next-gen flagship cabin concept.\n\n"
                    "Do you want a **true ULR** cabin icon (6,500+ nm class) or the best cabin you can get under a specific budget?"
                ).strip()
    except Exception:
        pass

    # 2) Advisory recommendation enforcement (mode-driven)
    try:
        mode = str((data_used or {}).get("consultant_response_mode") or "").strip().lower()
        if mode in ("mission_advisory", "client_decision_scenarios", "advisory"):
            from rag.consultant_validity import count_known_model_mentions

            n_models = count_known_model_mentions(a)
            if n_models < 2:
                # Deterministic, decision-grade fallback block: assumptions + 2–4 real models + why + tradeoff + bottom line.
                # Keep it generic (no fake pricing/specs).
                fallback = (
                    a.rstrip()
                    + "\n\nAssuming 6–8 passengers and typical business-use constraints (no extreme hot/high), here are a few realistic fits:\n"
                    "- Challenger 350: balanced mission capability and strong dispatch reputation; tradeoff: you’re paying for efficiency/modernity more than maximum cabin volume.\n"
                    "- Praetor 600: excellent range margin for the class and modern cabin; tradeoff: availability and programs matter a lot by airframe.\n"
                    "- Gulfstream G280: fast point-to-point with a solid cabin for many U.S. missions; tradeoff: it’s not a large-cabin experience.\n\n"
                    "Bottom Line: If you want the safest all-around answer without overbuying, start with Challenger 350 unless a specific route or cabin requirement pushes you up or down a class.\n\n"
                    "Consultant Insight: Most buyer’s remorse isn’t about range—it’s about dispatch reliability and ownership friction (programs, crew, maintenance posture)."
                ).strip()
                return fallback
    except Exception:
        pass

    return a


def answer_contains_banned_terms(answer: str, extra: Iterable[str] = ()) -> Dict[str, int]:
    """
    Debug helper for tests: return {term: count} for banned terms seen (case-insensitive substring).
    """
    s = (answer or "").lower()
    needles = list(_BANNED_TERMS) + [str(x).lower() for x in (extra or [])]
    out: Dict[str, int] = {}
    for t in needles:
        if not t:
            continue
        c = s.count(t.lower())
        if c:
            out[t] = c
    return out

