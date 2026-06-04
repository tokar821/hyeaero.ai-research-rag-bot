"""
Final render stripper — remove report scaffolds, broker injections, and duplicate blocks.

Presentation only; does not change routing or ranking.
"""

from __future__ import annotations

import re
from typing import List, Set

# Paragraphs led by internal report section headers (not user-facing prose).
_SECTION_LEAD_RE = re.compile(
    r"(?im)^\s*(?:Overview|Analysis|Recommendation|Risks|Mission Fit|Aircraft Options|Verdict|"
    r"Comparison|Operational segments?|Operational synthesis|Authority|Routes|"
    r"Operational constraint|Implication|Ranked options below|Supporting market context|"
    r"Recommended Aircraft|Where I would look)\s*:?\s*"
)

# Lines that indicate broker/template injection (drop whole paragraph if present).
_BROKER_INJECT_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:Key risk|What I would do|Inventory|Note:|PRIMARY RECOMMENDATION|"
    r"VIABLE WITH COMPROMISES|Reason it fits|Operational read|Planning assumption|"
    r"Executive travel profile|Validate live market|Before treating it as a bargain|"
    r"Before treating this tail|Get a spec sheet|logbooks, and a broker|"
    r"If I were spending my own money|latent priorities|continuation_ulr|"
    r"multi-layer aircraft reasoning)\s*:"
)

_FORBIDDEN_PHRASE_RE = re.compile(
    r"(?is)\b(?:if\s+i\s+were\s+buying\s+today|before\s+treating\s+(?:it\s+as\s+a\s+)?bargain|"
    r"before\s+treating\s+this\s+tail|operational\s+synthesis|send\s+me\s+(?:the\s+)?listing\s+package|"
    r"supporting\s+market\s+context|ranked\s+options\s+below\s+follow|where\s+i\s+would\s+look)\b"
)
_INSUFFICIENT_LINE_RE = re.compile(
    r"(?im)^\s*(?:INSUFFICIENT_DATA|insufficient\s+verified\s+aircraft\s+data)[^\n]*$"
)

_ADDRESS_DUMP_RE = re.compile(
    r"(?is)\b(?:FAA-registered address|registered to:\s*\n|ST\s+\d+.*(?:CHICAGO|IL,)\b)"
)


def _normalize_para(p: str) -> str:
    return re.sub(r"\s+", " ", (p or "").strip().lower())


def _jaccard(a: str, b: str) -> float:
    ta = set(_normalize_para(a).split())
    tb = set(_normalize_para(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def dedupe_paragraphs(text: str, *, threshold: float = 0.82) -> str:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if len(paras) < 2:
        return (text or "").strip()
    kept: List[str] = []
    for p in paras:
        if any(_jaccard(p, k) >= threshold for k in kept):
            continue
        kept.append(p)
    return "\n\n".join(kept).strip()


def _paragraph_is_scaffold(para: str) -> bool:
    p = (para or "").strip()
    if not p:
        return True
    if _SECTION_LEAD_RE.match(p):
        return True
    if _BROKER_INJECT_RE.search(p):
        return True
    if _FORBIDDEN_PHRASE_RE.search(p):
        return True
    # Kernel-style bullet dumps
    if p.count("* Route:") + p.count("* Pax:") + p.count("Aircraft Options:") >= 2:
        return True
    if re.search(r"(?im)^\s*[\*\-]\s*(?:Global|Gulfstream|Citation|Challenger|Boeing)\s", p) and (
        "Why it fits:" in p or "Key compromise:" in p
    ):
        return True
    return False


def strip_report_scaffolds(text: str, *, fact_only: bool = False) -> str:
    """Remove template sections and broker-injection paragraphs."""
    raw = (text or "").strip()
    if not raw:
        return raw

    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    kept: List[str] = []
    for p in paras:
        if _paragraph_is_scaffold(p):
            continue
        if fact_only and _ADDRESS_DUMP_RE.search(p):
            continue
        kept.append(p)

    body = dedupe_paragraphs("\n\n".join(kept))
    # Drop trailing INSUFFICIENT_DATA lines when substantive content remains.
    if kept:
        lines = body.splitlines()
        pruned = [ln for ln in lines if not _INSUFFICIENT_LINE_RE.match(ln.strip())]
        if pruned:
            body = "\n".join(pruned).strip()
    # Line-level dedupe for repeated sentences (comparison bug)
    lines = body.splitlines()
    seen_line: Set[str] = set()
    out_lines: List[str] = []
    for line in lines:
        norm = _normalize_para(line)
        if len(norm) > 40 and norm in seen_line:
            continue
        if norm:
            seen_line.add(norm)
        out_lines.append(line)
    return "\n".join(out_lines).strip()


__all__ = ["dedupe_paragraphs", "strip_report_scaffolds"]
