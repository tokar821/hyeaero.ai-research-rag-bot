"""
Output cleaner — remove markdown artifacts, empty sections, and internal headers.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

# Template section headers to collapse or remove when empty.
_TEMPLATE_HEADER_RE = re.compile(
    r"(?im)^\s*(?:Overview|Analysis|Recommendation|Risks|Mission Fit|Aircraft Options|"
    r"Mission Summary|Constraint Summary|Operational Synthesis|Ranked Aircraft Shortlist|"
    r"Mission Interpretation|Bottom-Line Recommendation|Side-by-Side Comparison|"
    r"Top Aircraft Options|Conditional options|Red Flags|Deal Assessment|Market Reality|"
    r"Final Verdict|Comparison:)\s*:?\s*$"
)

# Standalone markdown noise.
_RAW_MARKDOWN_ARTIFACT_RE = re.compile(
    r"(?m)^\s*\*\s*$|^\s*##\s*\*\*\s*$|^\s*\*\*\s*$"
)

# Internal marker lines.
_INTERNAL_LINE_RE = re.compile(
    r"(?im)^\s*(?:"
    r"INSUFFICIENT_DATA|CLARIFICATION_REQUIRED|INFEASIBLE_BUDGET_CONSTRAINT|"
    r"MARKET_CONTEXT_AVAILABLE|VERDICT:\s*$|Verdict:\s*$"
    r")\s*$"
)

# Duplicate header on consecutive lines.
_DUPLICATE_HEADER_RE = re.compile(
    r"(?im)^(\s*(?:Overview|Analysis|Recommendation|Risks)\s*:?\s*\n)\1+"
)


def remove_empty_sections(text: str) -> str:
    """Drop section headers with no body content."""
    lines = (text or "").splitlines()
    if not lines:
        return ""

    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _TEMPLATE_HEADER_RE.match(line.strip()):
            header = line.strip()
            j = i + 1
            body: List[str] = []
            while j < len(lines):
                nxt = lines[j].strip()
                if _TEMPLATE_HEADER_RE.match(nxt):
                    break
                if nxt:
                    body.append(lines[j])
                j += 1
            if body:
                out.append(header)
                out.extend(body)
            i = j
            continue
        out.append(line)
        i += 1

    return "\n".join(out).strip()


def collapse_redundant_headers(text: str) -> str:
    """Remove duplicate template headers and collapse repeated blank blocks."""
    out = _DUPLICATE_HEADER_RE.sub(r"\1", text or "")
    out = re.sub(r"(?im)^(Overview|Analysis|Recommendation|Risks)\s*:?\s*\n(?=\1\s*:?\s*\n)", "", out)
    return out.strip()


def _normalize_markdown_emphasis(text: str) -> str:
    """Convert leaked ``**bold**`` markdown to plain emphasis-friendly text."""
    out = text or ""

    def _repl_heading(m: re.Match) -> str:
        inner = m.group(2).strip()
        return inner if inner else ""

    out = re.sub(r"(?m)^\s*#{1,6}\s+\*\*(.+?)\*\*\s*$", _repl_heading, out)
    out = re.sub(r"\*\*([^*\n][^*\n]+?)\*\*", r"\1", out)
    out = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)", r"\1", out)
    out = re.sub(r"(?m)^\s*\d+\.\s+\*\*([^*]+)\*\*\s*:", r"\1:", out)
    return out


def normalize_bullets(text: str, *, use_unicode: bool = True) -> str:
    """
    Normalize list markers to consistent bullets.

    Converts lone ``*`` lines and broken markdown to ``•`` or plain sentences.
    """
    bullet = "•" if use_unicode else "-"
    lines: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if _RAW_MARKDOWN_ARTIFACT_RE.match(stripped):
            continue
        if re.match(r"^\*\s+\*\s*$", stripped):
            continue
        # Broken heading like "## **"
        if re.match(r"^#+\s*\*+\s*$", stripped):
            continue
        m = re.match(r"^(\*|\-|•)\s+(.*)$", stripped)
        if m:
            content = m.group(2).strip()
            if content:
                lines.append(f"{bullet} {content}")
            continue
        # Lone asterisk at end of line (markdown leak)
        if stripped.endswith("**") and stripped.count("*") == 2 and len(stripped) <= 4:
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


_RETRIEVAL_PROVENANCE_RE = re.compile(
    r"(?is)\b(?:"
    r"separately,?\s+per\s+hye\s+aero(?:\s+listing)?\s+records?|"
    r"per\s+hye\s+aero(?:\s+listing)?\s+records?|"
    r"per\s+phlydata|per\s+aircraft\s+registry|"
    r"current\s+marketplace\s+listing|"
    r"hye\s+aero(?:'s)?\s+internal\s+aircraft\s+record|"
    r"canonical\s+internal\s+record|"
    r"marketplace\s+ingest|not\s+phlydata"
    r")\b"
)


def strip_retrieval_provenance(text: str) -> str:
    """Remove internal retrieval / ingest phrasing from client-visible answers."""
    lines = []
    for line in (text or "").splitlines():
        if _RETRIEVAL_PROVENANCE_RE.search(line):
            continue
        cleaned = _RETRIEVAL_PROVENANCE_RE.sub("", line)
        if cleaned.strip():
            lines.append(cleaned.rstrip())
    return "\n".join(lines).strip()


def strip_internal_markers(text: str) -> str:
    """Remove lines that are purely internal status codes."""
    lines = []
    for line in (text or "").splitlines():
        if _INTERNAL_LINE_RE.match(line.strip()):
            continue
        if re.match(r"(?i)^verdict:\s*(insufficient_data|clarification_required|infeasible_budget_constraint|market_context_available)\s*$", line.strip()):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def clean_broker_output(
    text: str,
    *,
    use_unicode_bullets: bool = True,
    preserve_structured_headers: Optional[Set[str]] = None,
) -> str:
    """Run full output cleanup pipeline."""
    del preserve_structured_headers  # reserved for intent-specific callers
    out = _normalize_markdown_emphasis(text)
    out = strip_retrieval_provenance(out)
    out = strip_internal_markers(out)
    out = remove_empty_sections(out)
    out = collapse_redundant_headers(out)
    out = normalize_bullets(out, use_unicode=use_unicode_bullets)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    return out.strip()


__all__ = [
    "clean_broker_output",
    "collapse_redundant_headers",
    "normalize_bullets",
    "remove_empty_sections",
    "strip_internal_markers",
]
