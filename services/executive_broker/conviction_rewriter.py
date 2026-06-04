"""Rewrite executive output to sound like a broker with conviction (not a menu)."""

from __future__ import annotations

import re
from typing import Optional


_PRIMARY_RE = re.compile(r"(?im)^my primary recommendation would be the (?P<model>.+?)\s*-\s*(?P<why>.+?)\.\s*$")


def rewrite_with_conviction(
    answer: str,
    *,
    primary_model: Optional[str] = None,
) -> str:
    """
    Convert the executive lead into broker-like conviction phrasing.

    Does not change the underlying selection — only wording.
    """
    text = (answer or "").strip()
    if not text:
        return text

    m = _PRIMARY_RE.search(text)
    if not m:
        return text

    model = primary_model or m.group("model").strip()
    why = m.group("why").strip()

    lead = f"If I were buying today, I'd focus on the {model}. {why}."
    text = _PRIMARY_RE.sub(lead, text, count=1)
    return text.strip()


__all__ = ["rewrite_with_conviction"]

