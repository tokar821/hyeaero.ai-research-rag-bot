"""
Final response cleanup — dedupe prose, bullets, aircraft mentions, and operational warnings.

Runs after LLM generation (and formatter synthesis) but before user delivery.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

_BULLET_RE = re.compile(r"^(\s*)([-•*]|\d+\.)\s+(.*)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[^\w\s]+$")

_OPERATIONAL_WARNING_RE = re.compile(
    r"\b(?:"
    r"fuel\s+stop(?:s)?|"
    r"westbound(?:\s+headwind)?|"
    r"headwind(?:s)?|"
    r"short\s+runway|"
    r"runway\s+length|"
    r"hot(?:/|\s+and\s+)high|"
    r"baggage(?:\s+may|\s+limit)?|"
    r"operating\s+cost(?:s)?|"
    r"direct\s+operating|"
    r"nonstop(?:\s+may|\s+not)?|"
    r"nbaa(?:[- ]style)?|"
    r"reserve(?:s)?|"
    r"brochure\s+range|"
    r"practical\s+range|"
    r"tech[- ]stop|"
    r"payload(?:\s+trade)|"
    r"runway\s+flex"
    r")\b",
    re.I,
)

_MALFORMED_BULLET_RE = re.compile(r"^(\s*)([-•*])\s*([-•*])\s+")
_SECTION_HDR_RE = re.compile(
    r"^([A-Z][A-Za-z0-9\s/&\-]+):\s*$"
    r"|^Your typical routes involve:\s*$",
    re.I,
)
_BROKER_VERDICT_RE = re.compile(
    r"^(?:PRIMARY RECOMMENDATION|VIABLE WITH COMPROMISES|MISSION-RISKY|"
    r"NOT OPERATIONALLY CREDIBLE|GOOD FIT|CONDITIONAL FIT|NOT A FIT)\s*:",
    re.I,
)


def _is_broker_verdict_line(line: str) -> bool:
    s = (line or "").strip()
    # Cleanup normalizes bullet markers to "-" later in the pipeline.
    # Verdict matching must tolerate optional leading bullet tokens.
    if s.startswith(("-", "*", "•")):
        s = s[1:].strip()
    if re.match(r"^\d+\.\s+", s):
        s = re.sub(r"^\d+\.\s+", "", s, count=1).strip()
    return bool(_BROKER_VERDICT_RE.match(s))


def _collapse_ws(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", (text or "").strip())


def _sentence_key(sentence: str) -> str:
    s = _collapse_ws(sentence).lower()
    s = _TRAILING_PUNCT_RE.sub("", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def _clause_keys(sentence: str, *, min_words: int = 6) -> List[str]:
    words = re.findall(r"\b[\w']+\b", sentence.lower())
    if len(words) < min_words:
        return []
    keys: List[str] = []
    for n in range(min_words, min(len(words), 14) + 1):
        for i in range(0, len(words) - n + 1):
            keys.append(" ".join(words[i : i + n]))
    return keys


def _warning_signature(sentence: str) -> Optional[str]:
    hits = sorted({m.group(0).lower() for m in _OPERATIONAL_WARNING_RE.finditer(sentence)})
    if not hits:
        return None
    return "|".join(hits)


def _detect_aircraft_models(text: str) -> List[str]:
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        return detect_models_from_text(text or "")
    except Exception:
        return []


def _dedupe_sentences_in_paragraph(paragraph: str) -> str:
    if not (paragraph or "").strip():
        return ""
    parts = _SENTENCE_SPLIT_RE.split(paragraph.strip())
    if len(parts) <= 1:
        return paragraph.strip()

    kept: List[str] = []
    seen_sentences: Set[str] = set()
    seen_clauses: Set[str] = set()
    seen_warnings: Set[str] = set()

    for raw in parts:
        sent = raw.strip()
        if not sent:
            continue
        sk = _sentence_key(sent)
        if not sk or len(sk) < 8:
            kept.append(sent)
            continue

        if sk in seen_sentences:
            continue

        if any(
            len(sk) >= 20 and len(other) >= 20 and (sk in other or other in sk)
            for other in seen_sentences
        ):
            continue

        clause_dup = False
        for ck in _clause_keys(sent):
            if ck in seen_clauses:
                clause_dup = True
                break
        if clause_dup:
            continue

        warn_sig = _warning_signature(sent)
        if warn_sig and warn_sig in seen_warnings:
            continue

        kept.append(sent)
        seen_sentences.add(sk)
        for ck in _clause_keys(sent):
            seen_clauses.add(ck)
        if warn_sig:
            seen_warnings.add(warn_sig)

    if not kept:
        return paragraph.strip()
    out = " ".join(kept)
    if paragraph.rstrip()[-1:] in ".!?" and out[-1:] not in ".!?":
        out += "."
    return out


def _normalize_bullet_line(line: str) -> Tuple[str, str]:
    m = _BULLET_RE.match(line)
    if not m:
        return line, _sentence_key(line)
    indent, _marker, body = m.group(1), m.group(2), m.group(3)
    body = _collapse_ws(body)
    body = re.sub(r"^\s*[-•*]\s+", "", body)
    normalized = f"{indent}- {body}".rstrip()
    return normalized, _sentence_key(body)


def _fix_malformed_bullets(lines: List[str]) -> List[str]:
    fixed: List[str] = []
    for line in lines:
        if not line.strip():
            fixed.append(line)
            continue
        m = _MALFORMED_BULLET_RE.match(line)
        if m:
            line = f"{m.group(1)}- {line[m.end():].lstrip()}"
        bm = _BULLET_RE.match(line)
        if bm and not bm.group(3).strip():
            continue
        fixed.append(line)
    return fixed


def _dedupe_bullet_lines(lines: List[str]) -> List[str]:
    lines = _fix_malformed_bullets(lines)
    out: List[str] = []
    seen_bullets: Set[str] = set()
    seen_model_lead: Set[str] = set()

    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        if not _BULLET_RE.match(line):
            out.append(line)
            continue

        normalized, key = _normalize_bullet_line(line)
        if not key:
            out.append(normalized)
            continue
        if key in seen_bullets:
            continue

        models = _detect_aircraft_models(normalized)
        if models:
            lead = models[0].lower()
            lead_key = f"{lead}|{_sentence_key(normalized)[:80]}"
            if lead_key in seen_model_lead:
                continue
            seen_model_lead.add(lead_key)

        seen_bullets.add(key)
        out.append(normalized)
    return out


def _dedupe_repeated_aircraft_mentions(lines: List[str]) -> List[str]:
    models = _detect_aircraft_models("\n".join(lines))
    if not models:
        return lines

    mentioned: Set[str] = set()
    out: List[str] = []
    for line in lines:
        if _is_broker_verdict_line(line):
            out.append(line)
            continue
        line_models = [m.lower() for m in _detect_aircraft_models(line)]
        if not line_models:
            out.append(line)
            continue
        primary = line_models[0]
        body_key = _sentence_key(
            re.sub(r"\b" + re.escape(primary) + r"\b", "", line, flags=re.I)
        )
        if len(body_key) < 18 and primary in mentioned and _BULLET_RE.match(line):
            continue
        if primary in mentioned and len(line_models) == 1 and len(body_key) < 30:
            continue
        for m in line_models:
            mentioned.add(m)
        out.append(line)
    return out


def _process_line_groups(lines: List[str]) -> List[str]:
    """Dedupe prose by sentence; bullets by line."""
    out: List[str] = []
    prose_buf: List[str] = []

    def flush_prose() -> None:
        if not prose_buf:
            return
        joined = " ".join(l.strip() for l in prose_buf if l.strip())
        if joined:
            out.append(_dedupe_sentences_in_paragraph(joined))
        prose_buf.clear()

    for line in lines:
        if _is_broker_verdict_line(line):
            flush_prose()
            out.append(line)
        elif _BULLET_RE.match(line):
            flush_prose()
            out.append(line)
        elif not line.strip():
            flush_prose()
            out.append(line)
        else:
            prose_buf.append(line)

    flush_prose()
    return out


def _strip_empty_section_headers(lines: List[str]) -> List[str]:
    """Remove section titles with no body (e.g. 'Distance Considerations:' with nothing below)."""
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if _SECTION_HDR_RE.match(stripped):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                i += 1
                continue
            nxt = lines[j].strip()
            if _SECTION_HDR_RE.match(nxt):
                i += 1
                continue
        out.append(line)
        i += 1
    return out


def clean_response_text(text: str) -> str:
    """
    Final cleanup pass: dedupe sentences/clauses, bullets, aircraft names, and warnings.
    """
    s = (text or "").strip()
    if not s:
        return ""

    lines = s.splitlines()
    lines = _strip_empty_section_headers(lines)
    lines = _process_line_groups(lines)
    lines = _dedupe_bullet_lines(lines)
    lines = _dedupe_repeated_aircraft_mentions(lines)
    lines = _dedupe_bullet_lines(lines)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Paragraph-level duplicate blocks
    blocks = re.split(r"\n\s*\n", text)
    seen_para: Set[str] = set()
    kept_blocks: List[str] = []
    for block in blocks:
        b = block.strip()
        if not b:
            continue
        if any(_is_broker_verdict_line(ln) for ln in b.splitlines()):
            kept_blocks.append(b)
            continue
        if any(_BULLET_RE.match(ln) for ln in b.splitlines()):
            kept_blocks.append(b)
            continue
        pk = _sentence_key(b)
        if pk and len(pk) >= 24 and pk in seen_para:
            continue
        if pk:
            seen_para.add(pk)
        kept_blocks.append(_dedupe_sentences_in_paragraph(b))

    return "\n\n".join(kept_blocks).strip()


def cleanResponseText(text: str) -> str:
    """Public camelCase alias for ``clean_response_text``."""
    return clean_response_text(text)
