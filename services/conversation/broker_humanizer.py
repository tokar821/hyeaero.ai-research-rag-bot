"""Humanize residual software-y labels into broker speech (late presentation pass)."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

_LINE_REWRITES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?im)^\s*Key risk:\s*"), "The biggest risk here is "),
    (re.compile(r"(?im)^\s*What I would do:\s*"), "If I were spending my own money, I'd "),
    (re.compile(r"(?im)^\s*Inventory:\s*"), "Right now, "),
    (re.compile(r"(?im)^\s*Leverage:\s*"), "In negotiation, "),
    (re.compile(r"(?im)^\s*Supporting market context:\s*$"), "Here’s the market context I’m using:"),
)

_PHRASE_REWRITES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?is)\bIf the first tail doesn't check out, my backup would be:\b"),
        "If that first airplane doesn’t pencil out, my backup would be:",
    ),
    (
        re.compile(r"(?is)\bI would not lead with:\b"),
        "I wouldn’t lead with:",
    ),
)

# Certification-forbidden software phrases — removed or rewritten at the final pass.
_FORBIDDEN_PHRASE_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\bbuyer leverage\b"), "room to negotiate"),
    (re.compile(r"(?i)\bseller leverage\b"), "seller strength"),
    (re.compile(r"(?i)\bthin inventory\b"), "limited inventory"),
    (re.compile(r"(?i)\binventory pressure\b"), "tight supply"),
    (re.compile(r"(?i)\bverified catalog\b"), "verified aircraft data"),
    (re.compile(r"(?i)\bdeterministic execution\b"), "structured analysis"),
    (re.compile(r"(?i)\binsufficient verified data\b"), "not enough verified data"),
    (re.compile(r"(?i)\bmission kernel\b"), "mission profile"),
    (re.compile(r"(?i)\bcatalog authority\b"), "aircraft records"),
    (re.compile(r"(?i)\bstretch case\b"), "tight budget fit"),
    (re.compile(r"(?i)\bmotivated seller\b"), "a motivated seller"),
    (re.compile(r"(?i)\bcan be plausible\b"), "can trade in-band"),
    (re.compile(r"(?i)\bprimary recommendation would be\b"), "I'd focus on"),
)

_REMOVE_HEADERS = re.compile(
    r"(?im)^\s*(?:Overview|Analysis|Recommendation|Risks|Mission Fit|Aircraft Options)\s*:?\s*$"
)


def _strip_forbidden_sentences(text: str) -> str:
    """Drop lines that still contain forbidden tokens after substitution."""
    forbidden_tokens = (
        "insufficient_data",
        "deterministic execution",
        "verified catalog",
        "mission kernel",
        "catalog authority",
        "typical acquisition tier",
        "acceptance criteria",
        "supporting context",
    )
    kept: list[str] = []
    for line in text.splitlines():
        low = line.lower()
        if any(tok in low for tok in forbidden_tokens):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def humanize_broker_language(text: str, *, data_used: Optional[Dict] = None) -> str:
    del data_used
    out = (text or "").strip()
    if not out:
        return out

    lines = [ln for ln in out.splitlines() if not _REMOVE_HEADERS.match(ln.strip())]
    out = "\n".join(lines).strip()

    for pat, repl in _LINE_REWRITES:
        out = pat.sub(repl, out)

    for pat, repl in _PHRASE_REWRITES:
        out = pat.sub(repl, out)

    for pat, repl in _FORBIDDEN_PHRASE_PATTERNS:
        out = pat.sub(repl, out)

    out = _strip_forbidden_sentences(out)

    out = re.sub(
        r"(?is)(Here’s the market context I’m using:\s*)(?:\n\s*\n)?Here’s the market context I’m using:\s*",
        r"\1",
        out,
    )

    return out.strip()


__all__ = ["humanize_broker_language"]
