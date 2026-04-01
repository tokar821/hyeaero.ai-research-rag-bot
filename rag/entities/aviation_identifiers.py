"""
Aviation identifier detection: **models**, **registrations (tail)**, **serial numbers (MSN)**.

Resolution is **span-based**: model patterns are matched first; overlapping text is never
reclassified as serial. Protects ``737-800``, ``A320-200``, ``Challenger 601-3ER``, etc.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# --- Aircraft model patterns (order: longer / more specific first) ---
_AIRCRAFT_MODEL_RES: List[re.Pattern] = [
    re.compile(
        r"\b(?:Bombardier\s+)?Challenger\s+(?:300|3500?|350|600|601|604|605|650|850)"
        r"(?:\s+)?(?:-[0-9A-Z]+)*\b",
        re.I,
    ),
    re.compile(
        r"\b(?:Bombardier\s+)?Global\s+(?:Express|5000|5500|6000|6500|7500|8000)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:Cessna\s+)?Citation\s+"
        r"(?:Latitude|Longitude|Mustang|Sovereign|XLS\+?|Excel|Ultra|Encore|"
        r"CJ\d\+?|CJ\d|M2|Hemisphere|Ascend|Bravo|Ultra|"
        r"II|III|ISP|CJ[12]|SII)\w*\b",
        re.I,
    ),
    re.compile(r"\b(?:Gulfstream\s+)?G\d{3,4}(?:ER|MS)?\b", re.I),
    re.compile(r"\b(?:Dassault\s+)?Falcon\s+(?:50|900|2000|6X|7X|8X|10X|1000|2000LXS)[\w\-]*\b", re.I),
    re.compile(r"\bHondaJet\b", re.I),
    re.compile(r"\b(?:Embraer\s+)?(?:Phenom|Praetor|Legacy)\s*[\w\-]+\b", re.I),
    re.compile(r"\bKing\s*Air\s+[\w\-]+\b", re.I),
    re.compile(r"\bLearjet\s+\d+[\w\-]*\b", re.I),
    re.compile(r"\bPilatus\s+PC[- ]?\d+[\w\-]*\b", re.I),
    re.compile(r"\bPC-1[24]\w*\b", re.I),
    # Boeing / Airbus airliner variants (incl. 737-800, 777-300ER, A320-200)
    re.compile(r"\b(?:Boeing\s*)?(?:B-?)?(?:717|737|747|757|767|777|787)-\d{2,3}\w*\b", re.I),
    re.compile(r"\bAirbus\s+A\d{2,3}(?:-\d{3})?\w*\b", re.I),
    re.compile(r"\bA\d{2,3}-\d{3}\w*\b", re.I),
    re.compile(r"\bA\d{3}(?:-\d{3})?\s*(?:neo|XLR|ceo|leap|pw)?\b", re.I),
    # Bombardier 601-3ER-style **without** the word Challenger (numeric + hyphen variant)
    re.compile(r"\b6(?:01|04|05)(?:-[0-9A-Z]+)+\b", re.I),
    re.compile(r"\bChallenger\s+650\b", re.I),
    # Regional jets
    re.compile(r"\b(?:CRJ|Dash\s*8|Q\d{3}|ATR|MELJET|E175|ERJ|E-?170|E-?175|E-?190|E-?195)[-\s]?\w*\b", re.I),
]

# US + international civil marks (same discipline as intent classifier)
_US_N = re.compile(r"\b[Nn]\d[A-Z0-9]{0,4}\b")
_ICAO = re.compile(r"\b[A-Z]{1,2}-[A-Z0-9]{2,5}\b", re.I)

# Citation / OEM serial-style (first segment → often type series) — **not** airliner prefixes
_OEM_SERIAL_LEFT = frozenset(
    {
        "50",
        "51",
        "52",
        "55",
        "56",
        "175",
        "190",
        "195",
        "510",
        "525",
        "525A",
        "525B",
        "525C",
        "530",
        "550",
        "560",
        "650",
        "680",
        "680A",
        "750",
        "50EX",
    }
)

# Hyphen tokens where left side is always an **airliner / Airbus** model prefix, never MSN
_AIRLINER_LEFT_PREFIX: Set[int] = set()
for n in (
    717,
    727,
    737,
    747,
    757,
    767,
    777,
    787,
    318,
    319,
    320,
    321,
    322,
    330,
    332,
    333,
    338,
    339,
    340,
    342,
    350,
    359,
    380,
):
    _AIRLINER_LEFT_PREFIX.add(n)


def _merge_text(query: str, history: Optional[List[Dict[str, str]]]) -> str:
    parts: List[str] = []
    if history:
        for h in history[-12:]:
            if (h.get("role") or "").strip().lower() not in ("user", "assistant"):
                continue
            c = (h.get("content") or "").strip()
            if c:
                parts.append(c)
    parts.append(query or "")
    return "\n".join(parts)


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    srt = sorted(spans, key=lambda x: (x[0], x[1]))
    out = [srt[0]]
    for a, b in srt[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def _span_hits(pos: int, end: int, blocked: List[Tuple[int, int]]) -> bool:
    for a, b in blocked:
        if pos < b and end > a:
            return True
    return False


def _hyphen_token_is_airliner_model(left: str, right: str) -> bool:
    """737-800, A320-200 style (right is mostly digits + optional ER/LR)."""
    try:
        n = int(re.sub(r"^0+", "", left) or "0")
    except ValueError:
        return False
    if n in _AIRLINER_LEFT_PREFIX:
        return True
    if re.match(r"^\d{2,3}[A-Z]{0,3}$", right, re.I) and 300 <= n <= 399 and len(left) == 3:
        # A320-200 style already caught; 350-??? could be Airbus A350
        if n in (319, 320, 321, 318, 330, 332, 333, 338, 339, 350, 359, 380):
            return True
    return False


def _hyphen_token_is_challenger_style(left: str, right: str) -> bool:
    """601-3ER, 604-2R — numeric CL family + variant suffix."""
    if left in ("601", "604", "605") and re.match(r"^[0-9A-Z]+$", right, re.I):
        return True
    return False


def _collect_models(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    spans: List[Tuple[int, int]] = []
    seen_norm: Set[str] = set()
    out: List[str] = []
    for rx in _AIRCRAFT_MODEL_RES:
        for m in rx.finditer(text):
            raw = m.group(0).strip()
            if len(raw) < 3:
                continue
            norm = raw.upper()
            if norm in seen_norm:
                spans.append((m.start(), m.end()))
                continue
            seen_norm.add(norm)
            out.append(raw)
            spans.append((m.start(), m.end()))
    merged = _merge_spans(spans)
    return out, merged


def _collect_registrations(text: str, model_spans: List[Tuple[int, int]]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []

    def add(t: str) -> None:
        u = t.strip().upper().replace(" ", "")
        if 3 <= len(u) <= 10 and u not in seen:
            seen.add(u)
            out.append(u)

    for m in _US_N.finditer(text):
        if _span_hits(m.start(), m.end(), model_spans):
            continue
        add(m.group(0))

    for m in _ICAO.finditer(text):
        if _span_hits(m.start(), m.end(), model_spans):
            continue
        whole = m.group(0)
        suf = whole.split("-", 1)[1]
        if suf.isdigit() and len(suf) <= 3:
            continue  # PC-12 / CL-604 series noise (model handled elsewhere)
        add(whole)
    return out


def _serial_context_phrase(text: str) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    rx = re.compile(
        r"(?:\bmsn\b|\bserial(?:\s+number)?\b|\bs/?\s*n\b|\baircraft\b|\baicraft\b)\s*[:\#]?\s*"
        r"([A-Z0-9\-]{3,24})\b",
        re.I,
    )
    for m in rx.finditer(text):
        out.append((m.start(1), m.end(1), m.group(1).strip()))
    return out


def _hyphen_serial_candidates(text: str) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    for m in re.finditer(r"\b(\d{2,5}[A-Za-z]?-\d{3,7})\b", text):
        out.append((m.start(1), m.end(1), m.group(1)))
    return out


def _numeric_serial_candidates(text: str) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    for m in re.finditer(r"\b(0\d{3,6})\b", text):
        out.append((m.start(1), m.end(1), m.group(1)))
    for m in re.finditer(r"\b(\d{5,8})\b", text):
        s = m.group(1)
        if len(s) == 4 and s.startswith(("19", "20")):
            continue
        out.append((m.start(1), m.end(1), s))
    for m in re.finditer(r"\b(\d{3,4})\b", text):
        s = m.group(1)
        if len(s) == 4 and s.startswith(("19", "20")):
            continue
        out.append((m.start(1), m.end(1), s))
    return out


def _protect_hyphen_from_serial(token: str) -> bool:
    """True → must **not** go to serial_numbers (model or airliner shape)."""
    if "-" not in token:
        return False
    left, right = token.split("-", 1)
    if not left or not right:
        return False
    if right.isdigit() and _hyphen_token_is_airliner_model(left, right):
        return True
    if _hyphen_token_is_challenger_style(left, right):
        return True
    # A320neo style without hyphen sometimes caught as A320-200 via model regex only
    try:
        li = int(left)
    except ValueError:
        li = -1
    if li in _AIRLINER_LEFT_PREFIX:
        return True
    return False


def _oem_serial_hyphen_allowed(token: str) -> bool:
    if "-" not in token:
        return False
    left, right = token.split("-", 1)
    rl = left.upper()
    if rl in _OEM_SERIAL_LEFT and right.isdigit() and len(right) >= 3:
        return True
    if re.match(r"^\d{3}$", left) and re.match(r"^\d{4,6}$", right):
        if int(left) not in _AIRLINER_LEFT_PREFIX and not _hyphen_token_is_challenger_style(left, right):
            if left not in ("601", "604", "605"):
                return True
    return False


def _collect_serials(
    text: str,
    model_spans: List[Tuple[int, int]],
    model_strings: Set[str],
) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []

    def add_sn(raw: str, start: int, end: int) -> None:
        if _span_hits(start, end, model_spans):
            return
        t = raw.strip()
        if len(t) < 3 or len(t) > 28:
            return
        if t in model_strings:
            return
        if _protect_hyphen_from_serial(t):
            return
        k = t.upper()
        if k in seen:
            return
        seen.add(k)
        out.append(t)

    model_upper = {m.strip().upper() for m in model_strings}

    for start, end, tok in _serial_context_phrase(text):
        add_sn(tok, start, end)

    for start, end, tok in _hyphen_serial_candidates(text):
        if _protect_hyphen_from_serial(tok):
            continue
        if _oem_serial_hyphen_allowed(tok):
            add_sn(tok, start, end)
        elif re.search(
            r"(?:\bmsn\b|\bserial(?:\s+number)?\b|\bs/?\s*n\b)\s*[:\#]?\s*$",
            text[max(0, start - 48) : start],
            re.I,
        ):
            add_sn(tok, start, end)

    for start, end, tok in _numeric_serial_candidates(text):
        add_sn(tok, start, end)

    # 2–4 alnum fused (525B0044) seen in phly lookups
    for m in re.finditer(r"\b(\d{2,4}[A-Z][0-9A-Z]{3,12})\b", text, re.I):
        add_sn(m.group(1), m.start(1), m.end(1))

    return out


def detect_aviation_entities(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, List[str]]:
    """
    Detect models, registrations, and serials with **model-first** priority.

    Returns ordered unique lists (discovery order preserved per category).
    """
    text = _merge_text(query, history)
    aircraft_models, model_spans = _collect_models(text)
    model_set = set(aircraft_models)
    registrations = _collect_registrations(text, model_spans)
    serial_numbers = _collect_serials(text, model_spans, model_set)
    return {
        "aircraft_models": aircraft_models,
        "registrations": registrations,
        "serial_numbers": serial_numbers,
    }


def detect_aviation_entities_json(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Strict JSON-friendly output (lists of strings)."""
    return detect_aviation_entities(query, history)

