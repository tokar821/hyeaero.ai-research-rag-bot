"""
Aircraft Query Builder (deterministic).

CRITICAL PRODUCT RULE:
This builder must behave like:

    image_search(query) -> results

NOT:

    image_search(full_conversation_context)

Therefore it only accepts the *isolated* current user query plus an optional resolved entity
(tail or model), and rebuilds a clean image-search seed from scratch.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_FACET = re.compile(r"(?i)\b(cockpit|flight\s*deck|cabin|interior|exterior)\b")
_DEICTIC_ONLY = re.compile(
    r"(?is)^\s*(?:show\s+me|let\s+me\s+see|can\s+i\s+see\s+(?:it|that|this)|compare\s+it|that\s+one)\s*[\?!\.]*\s*$"
)

# Phrases that frequently leak old context into the query and/or are not useful for Google Images.
_STRIP_NOISE = re.compile(
    r"(?is)\b(?:"
    r"show\s+me|let\s+me\s+see|can\s+i\s+see|could\s+i\s+see|"
    r"that\s+one|that|this|it|them|same|again|"
    r"previous|earlier|above|as\s+i\s+said|as\s+i\s+mentioned|you\s+said|we\s+discussed|"
    r"thread|context|conversation|history|"
    r"best|top|nicest|ultimate|vibe|feel|"
    r"under|below|budget|cheaper|not\s+that\s+expensive|affordable|"
    r"\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:m|mm|million)?"
    r")\b"
)

_TAIL_TOKEN = re.compile(r"(?i)\bN(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}\b")


def _word_cap(s: str, max_words: int = 6) -> str:
    parts = [p for p in (s or "").strip().split() if p]
    return " ".join(parts[:max_words]).strip()


def _normalize_facet_token(raw: str) -> Optional[str]:
    low = (raw or "").lower()
    if "cockpit" in low or "flight deck" in low or "flightdeck" in low:
        return "cockpit"
    if "exterior" in low:
        return "exterior"
    if "cabin" in low:
        return "cabin"
    if "interior" in low:
        return "interior"
    return None


def _scrub_unrelated_aircraft_mentions(*, text: str, resolved_entity: str) -> str:
    """
    Remove unrelated aircraft/tail/model mentions from the *current* isolated query.

    This is a strict preflight cleanup to prevent accidental "full conversation" leakage when the
    user (or upstream layer) includes multiple aircraft names in a single input.
    """
    q = (text or "").strip()
    ent = (resolved_entity or "").strip()
    if not q or not ent:
        return q

    # Tail: keep only the resolved tail if the entity is a tail.
    ent_tail = None
    if _TAIL_TOKEN.search(ent):
        ent_tail = _TAIL_TOKEN.search(ent).group(0).upper()

    def _strip_other_tails(s: str) -> str:
        def _repl(m: re.Match[str]) -> str:
            t = m.group(0).upper()
            if ent_tail and t == ent_tail:
                return m.group(0)
            # Otherwise remove tails (likely from pasted context).
            return " "

        return _TAIL_TOKEN.sub(_repl, s)

    q = _strip_other_tails(q)

    # Models/manufacturers: if multiple models are present, drop those that do not match the resolved entity.
    try:
        from rag.consultant_query_expand import _detect_manufacturers, _detect_models

        mans = _detect_manufacturers(q.lower())
        mdls = _detect_models(q)

        ent_low = ent.lower()
        keep_mans = [m for m in mans if m.lower() in ent_low]
        keep_mdls = [m for m in mdls if m.lower() in ent_low]

        drop_terms: List[str] = []
        for m in mans:
            if m not in keep_mans:
                drop_terms.append(m)
        for m in mdls:
            if m not in keep_mdls:
                drop_terms.append(m)

        for term in sorted({t.strip() for t in drop_terms if t and t.strip()}, key=len, reverse=True):
            # Whole-ish word removal; allow hyphenated tokens.
            q = re.sub(rf"(?i)(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", " ", q)
    except Exception:
        pass

    q = re.sub(r"\s+", " ", q).strip()
    return q


def build_aircraft_image_search_seed(
    *,
    isolated_query: str,
    resolved_entity: Optional[str] = None,
) -> str:
    """
    Build a single clean seed string for downstream image query engines (SearchAPI/Tavily).

    - Uses only `isolated_query` and `resolved_entity`
    - Never injects prior-thread keywords
    - Always anchors to `resolved_entity` when present (tail or model)
    """
    q = (isolated_query or "").strip()
    ent = (resolved_entity or "").strip()
    if not q and not ent:
        return ""

    # Preflight scrub: remove unrelated past aircraft/topic tokens when an entity is known.
    if ent:
        q = _scrub_unrelated_aircraft_mentions(text=q, resolved_entity=ent)

    # If user wrote only a deictic phrase but we have an entity, treat it as a pure visual request.
    facet = None
    m = _FACET.search(q)
    if m:
        facet = _normalize_facet_token(m.group(0))

    # Rebuild from scratch: strip common noise, keep facet cues.
    q2 = _STRIP_NOISE.sub(" ", q)
    q2 = re.sub(r"[^A-Za-z0-9\s\-]", " ", q2)
    q2 = re.sub(r"\s+", " ", q2).strip()

    # If the remaining query is empty or deictic-only, fall back to entity + facet.
    if not q2 or _DEICTIC_ONLY.match(q2):
        if not ent:
            return ""
        if not facet:
            facet = "cabin"
        # Quality cue for interior/cockpit.
        if facet in ("cabin", "interior", "cockpit"):
            return _word_cap(f"{ent} {facet} high resolution", 6)
        return _word_cap(f"{ent} {facet}", 6)

    # If entity exists, force it to lead.
    if ent:
        base = f"{ent} {q2}"
    else:
        base = q2

    # Ensure a facet exists when an entity exists (prevents generic queries like "G650 nice").
    if ent and not facet:
        facet = "cabin"
        base = f"{ent} {facet} {q2}"

    # Add a quality cue when this is clearly a visual interior/cockpit query.
    if facet in ("cabin", "interior", "cockpit"):
        if not re.search(r"(?i)\b(high\s+res(?:olution)?|hd)\b", base):
            base = f"{base} high resolution"

    return _word_cap(base, 6)

