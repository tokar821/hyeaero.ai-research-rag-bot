"""
User-facing response safety layer.

This module strips internal infrastructure / dataset language from assistant answers.
It is intentionally conservative: it removes common internal labels even if they appear
in context or drafts, and replaces them with neutral, client-safe phrasing.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple


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

    # Replace common internal terms with neutral phrasing.
    for pat, repl in _REPLACEMENTS:
        s = pat.sub(repl, s)

    # Strip remaining backticked table names / code-ish blobs.
    s = re.sub(r"`[^`]{2,80}`", "", s)

    # Collapse repeated spaces created by removals.
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    return s


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

