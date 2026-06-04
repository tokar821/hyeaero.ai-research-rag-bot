"""Presentation-only compression — one voice, one recommendation block."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from services.conversation.output_cleaner import collapse_redundant_headers, remove_empty_sections
from services.truth_compression.truth_synthesizer import BrokerTruthState

_PRIMARY_BLOCK_RE = re.compile(
    r"(?is)(my primary recommendation would be.+?)"
    r"(?=\n\n(?:my primary recommendation|supporting market context)|\Z)",
)
_DUPLICATE_PRIMARY_RE = re.compile(r"(?is)my primary recommendation would be[^.]+\.")
_SUPPORTING_HEADER_RE = re.compile(r"(?im)^\s*supporting market context:\s*$")
_TEMPLATE_HEADER_LINE = re.compile(
    r"(?im)^\s*(?:overview|analysis|recommendation|risks)\s*:?\s*$",
)
_IF_NOT_DILIGENCE_RE = re.compile(
    r"(?is)(\n\s*if that does not clear diligence[^\n]*:?\s*\n(?:\s*[•\-].+\n)+)",
)


def _extract_primary_block(text: str) -> Tuple[str, str, str]:
    """Return (before, primary_block, after)."""
    m = _PRIMARY_BLOCK_RE.search(text)
    if not m:
        return text, "", ""
    return text[: m.start()].strip(), m.group(1).strip(), text[m.end() :].strip()


def _dedupe_primary_blocks(text: str) -> str:
    matches = list(_DUPLICATE_PRIMARY_RE.finditer(text))
    if len(matches) <= 1:
        return text
    for m in reversed(matches[1:]):
        text = text[: m.start()] + text[m.end() :]
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _dedupe_if_not_diligence_simple(text: str) -> str:
    matches = list(_IF_NOT_DILIGENCE_RE.finditer(text))
    if len(matches) <= 1:
        return text
    for m in reversed(matches[1:]):
        text = text[: m.start()] + text[m.end() :]
    return text


def _dedupe_supporting(text: str) -> str:
    headers = list(_SUPPORTING_HEADER_RE.finditer(text))
    if len(headers) <= 1:
        return text
    for h in reversed(headers[1:]):
        start = h.start()
        end = len(text)
        rest = text[h.end() :]
        end_match = re.search(r"\n\s*\n", rest)
        if end_match:
            end = h.end() + end_match.start()
        text = text[:start].rstrip() + text[end:].lstrip("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _remove_template_headers(text: str) -> str:
    lines = (text or "").splitlines()
    return "\n".join(ln for ln in lines if not _TEMPLATE_HEADER_LINE.match(ln.strip())).strip()


def _reorder_executive_first(text: str) -> str:
    before, primary, after = _extract_primary_block(text)
    if not primary:
        return text
    parts = [primary]
    if after:
        parts.append(after)
    if before:
        parts.append(before)
    return "\n\n".join(p for p in parts if p).strip()


def simplify_response(
    answer: str,
    truth: BrokerTruthState,
    *,
    pathways: Optional[List[str]] = None,
) -> str:
    """
    Enforce single-voice output: one recommendation speaker, supporting context once.
    """
    text = (answer or "").strip()
    if not text:
        return text

    text = collapse_redundant_headers(text)
    text = _remove_template_headers(text)
    text = remove_empty_sections(text)

    if truth.has_executive_recommendation:
        text = _dedupe_primary_blocks(text)
        text = _dedupe_if_not_diligence_simple(text)
        text = _reorder_executive_first(text)

    text = _dedupe_supporting(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if pathways and "REDUNDANT_TEMPLATE_HEADERS" in pathways:
        text = _remove_template_headers(text)

    return text


__all__ = ["simplify_response"]
