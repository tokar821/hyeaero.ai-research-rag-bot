"""Broker-mode tail investigation rewrite — no speculation, acquisition-relevant asks."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


_TAIL_HEADER_RE = re.compile(r"(?is)^on\s+N[A-Z0-9]{3,6},")
_LISTING_PKG_RE = re.compile(r"(?is)send\s+me\s+(?:the\s+)?listing\s+package[^.\n]*\.?")
_ACQUISITION_ASK_RE = re.compile(
    r"(?is)\b(?:engine\s+program|maintenance\s+records|listing\s+package|acquisition\s+merit|"
    r"logbooks?|spec\s+sheet)\b"
)


def rewrite_tail_investigation(
    text: str,
    *,
    registration: str,
    facts_available: bool = False,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Legacy rewrite hook — skipped on LLM-primary path.

    When still invoked, do not append acquisition diligence on ownership/sale turns.
    """
    body = (text or "").strip()
    reg = (registration or "").strip().upper()
    if not body or not reg:
        return body

    du = data_used if isinstance(data_used, dict) else {}
    suppress_acq = bool(du.get("suppress_acquisition_tail_rewrite"))

    body = _LISTING_PKG_RE.sub("", body).strip() if facts_available else body
    body = re.sub(r"(?is)\bworth\s+looking\s+at\b", "worth investigating", body)

    if suppress_acq and facts_available:
        for line in body.splitlines():
            if _ACQUISITION_ASK_RE.search(line):
                body = body.replace(line, "").strip()
        return body.strip()

    if facts_available and re.search(r"(?is)registry\s+facts\s+for", body):
        if suppress_acq:
            return body
        asks = (
            "To assess acquisition merit on this tail, share total time, engine program status, "
            "and recent maintenance when you have them."
        ).strip()
        if asks.lower() not in body.lower():
            return f"{body}\n\n{asks}".strip()
        return body

    if suppress_acq:
        return body

    lead = (
        f"I can verify ownership and basic registry facts on {reg}. "
        "I cannot tell you whether it is worth buying from registry data alone."
    ).strip()

    asks = (
        "Send me the listing package (or spec sheet), asking price, model year, total time, "
        "engine program status, and recent maintenance records — "
        "and I'll give you an acquisition view on that specific tail."
    ).strip()

    if _TAIL_HEADER_RE.search(body):
        body = re.sub(r"(?is)^on\s+N[A-Z0-9]{3,6},\s*", "", body).strip()

    factual = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("•") or s.startswith("-") or s.startswith("*"):
            factual.append(line)

    tail = "\n".join(factual).strip()
    if tail:
        return f"{lead}\n\n{asks}\n\n{tail}".strip()
    return f"{lead}\n\n{asks}".strip()


__all__ = ["rewrite_tail_investigation"]
