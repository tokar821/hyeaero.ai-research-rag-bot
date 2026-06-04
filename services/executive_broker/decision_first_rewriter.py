"""Enforce decision → reasoning → process ordering in executive answers."""

from __future__ import annotations

import re
from typing import List


_PROCESS_MARKERS = (
    r"(?is)\bbefore treating\b",
    r"(?is)\bi would verify\b",
    r"(?is)\bverify:\b",
    r"(?is)\bsend the listing\b",
)


def rewrite_decision_first(answer: str) -> str:
    """
    If a process/checklist block appears before the executive decision, move it after.
    """
    text = (answer or "").strip()
    if not text:
        return text

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) < 3:
        return text

    decision_i = None
    for i, p in enumerate(paras):
        if re.search(r"(?is)\b(primary recommendation|i'd focus on|i would buy|no\.)\b", p):
            decision_i = i
            break
    if decision_i is None:
        return text

    process_idxs: List[int] = []
    for i, p in enumerate(paras):
        if i <= decision_i:
            for pat in _PROCESS_MARKERS:
                if re.search(pat, p):
                    process_idxs.append(i)
                    break

    if not process_idxs:
        return text

    moved = [paras[i] for i in process_idxs]
    kept = [p for i, p in enumerate(paras) if i not in process_idxs]
    out = kept + moved
    return "\n\n".join(out).strip()


__all__ = ["rewrite_decision_first"]

