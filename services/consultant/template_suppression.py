"""
Duplicate and fallback template suppression for consultant answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SuppressionResult:
    text: str
    removed_blocks: int
    duplicate_paragraphs_removed: int
    fallback_contamination_score: float


_FALLBACK_MARKERS: Tuple[re.Pattern, ...] = (
    re.compile(r"Assuming\s+6[–-]8\s+passengers", re.I),
    re.compile(r"✅\s*GOOD\s+FIT", re.I),
    re.compile(r"Consultant\s+Insight:", re.I),
    re.compile(r"here are a few realistic fits", re.I),
    re.compile(r"typical business-use constraints \(no extreme hot/high\)", re.I),
    re.compile(r"Most buyer'?s remorse isn'?t about range", re.I),
)

_ASSUMING_BLOCK_RE = re.compile(
    r"(?:\n\s*)?Assuming\s+6[–-]8\s+passengers[\s\S]*?(?=\n\s*[A-Z][a-z]+ Summary|\n\s*Bottom|\Z)",
    re.I,
)
_REALISTIC_FITS_RE = re.compile(
    r"(?:\n\s*)?"
    r"(?:Assuming\s+6[–-]8\s+passengers\s+and\s+)?"
    r"typical\s+business-use\s+constraints\s*\(no\s+extreme\s+hot/high\)\s*,?\s*"
    r"here\s+are\s+a\s+few\s+realistic\s+fits:\s*"
    r"[\s\S]*?"
    r"(?=\n\s*(?:For |On |With |I['']d |My |The |\Z))",
    re.I,
)
_GOOD_FIT_RE = re.compile(r"\n\s*✅\s*GOOD\s+FIT[\s\S]*$", re.I)
_INSIGHT_RE = re.compile(r"\n\s*Consultant\s+Insight:[\s\S]*$", re.I)


def _normalize_para(p: str) -> str:
    return re.sub(r"\s+", " ", (p or "").strip().lower())


def _jaccard_similarity(a: str, b: str) -> float:
    ta = set(_normalize_para(a).split())
    tb = set(_normalize_para(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def fallback_contamination_score(text: str) -> float:
    """0 = clean, 1 = heavily contaminated with stock fallback templates."""
    s = text or ""
    if not s.strip():
        return 0.0
    hits = sum(1 for pat in _FALLBACK_MARKERS if pat.search(s))
    return min(1.0, hits / max(len(_FALLBACK_MARKERS), 1) * 1.4)


def _dedupe_paragraphs(text: str, *, similarity_threshold: float = 0.82) -> Tuple[str, int]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if len(paras) < 2:
        return text or "", 0
    kept: List[str] = []
    removed = 0
    for p in paras:
        if any(_jaccard_similarity(p, k) >= similarity_threshold for k in kept):
            removed += 1
            continue
        kept.append(p)
    return "\n\n".join(kept).strip(), removed


def suppress_templates(
    text: str,
    *,
    similarity_threshold: float = 0.82,
    strip_stock_fallback: bool = True,
) -> SuppressionResult:
    """
    Remove repeated fallback blocks, duplicate paragraphs, and stale advisory fragments.
    """
    s = (text or "").strip()
    removed_blocks = 0
    if not s:
        return SuppressionResult(
            text=s,
            removed_blocks=0,
            duplicate_paragraphs_removed=0,
            fallback_contamination_score=0.0,
        )

    score_before = fallback_contamination_score(s)

    if strip_stock_fallback:
        for pat in (_GOOD_FIT_RE, _ASSUMING_BLOCK_RE, _REALISTIC_FITS_RE, _INSIGHT_RE):
            new_s, n = pat.subn("", s)
            if n:
                removed_blocks += n
                s = new_s.strip()

    s, dup_removed = _dedupe_paragraphs(s, similarity_threshold=similarity_threshold)

    # Collapse repeated aircraft bullet sequences (same model line twice)
    lines = s.splitlines()
    seen_bullets: set[str] = set()
    out_lines: List[str] = []
    seq_removed = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-"):
            # Dedupe by model token prefix (e.g. "- Challenger 350: ...")
            model_key = re.sub(r"^-\s*", "", stripped).split(":")[0].strip().lower()
            norm = model_key or _normalize_para(line)
            if norm in seen_bullets:
                seq_removed += 1
                continue
            seen_bullets.add(norm)
        out_lines.append(line)
    if seq_removed:
        removed_blocks += 1
    s = "\n".join(out_lines).strip()

    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return SuppressionResult(
        text=s,
        removed_blocks=removed_blocks,
        duplicate_paragraphs_removed=dup_removed,
        fallback_contamination_score=max(score_before, fallback_contamination_score(s)),
    )
