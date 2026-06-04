"""
Phase 56.5 — remove duplicate facts and repeated aircraft mentions (formatting only).
"""

from __future__ import annotations

import re
from typing import List, Set


def _normalize_line(line: str) -> str:
    s = re.sub(r"\s+", " ", (line or "").strip().lower())
    s = re.sub(r"[^\w\s$./-]", "", s)
    return s


def deduplicate_lines(text: str) -> str:
    """Drop duplicate bullet/paragraph lines."""
    if not text:
        return ""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    out_paras: List[str] = []
    seen_para: Set[str] = set()

    for para in paragraphs:
        lines = para.splitlines()
        kept: List[str] = []
        seen_line: Set[str] = set()
        for line in lines:
            norm = _normalize_line(line)
            if not norm:
                kept.append(line)
                continue
            if norm in seen_line:
                continue
            seen_line.add(norm)
            kept.append(line)
        block = "\n".join(kept).strip()
        if not block:
            continue
        pnorm = _normalize_line(block)
        if pnorm in seen_para:
            continue
        seen_para.add(pnorm)
        out_paras.append(block)

    return "\n\n".join(out_paras).strip()


def collapse_repeated_aircraft_mentions(text: str, aircraft_names: List[str]) -> str:
    """If the same model appears in multiple narrative blocks, keep first substantive mention."""
    if not text or not aircraft_names:
        return text
    lines = text.splitlines()
    seen: Set[str] = set()
    out: List[str] = []
    for line in lines:
        low = line.lower()
        matched = None
        for name in aircraft_names:
            if name.lower() in low:
                matched = name.lower()
                break
        if matched:
            if matched in seen and len(line.strip()) > 80:
                continue
            seen.add(matched)
        out.append(line)
    return "\n".join(out).strip()


__all__ = ["collapse_repeated_aircraft_mentions", "deduplicate_lines"]
