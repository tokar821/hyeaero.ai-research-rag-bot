"""Shared lightweight text extraction for Phase 33 audits."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MONEY_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(?:m|mm|million)?\b", re.I)
_PAX_RE = re.compile(r"\b(\d+)\s*(?:pax|passengers?|people)\b", re.I)
_NONSTOP_RE = re.compile(r"\bnonstop\b", re.I)

# Very conservative aircraft-like tokens; we still validate via AKAL before flagging.
_AIRCRAFT_TOKEN_RE = re.compile(
    r"\b(?:G\d{3}\b|Gulfstream\s+G\d{3}\b|Global\s+\d{4}\b|Falcon\s+\dX\b|Falcon\s+\d{4}\b|"
    r"Challenger\s+\d{3,4}\b|Citation\s+(?:CJ\d\+?|Latitude|Longitude)\b|Praetor\s+\d{3}\b|PC-24\b)\b",
    re.I,
)


def extract_year(text: str) -> Optional[int]:
    m = _YEAR_RE.search(text or "")
    return int(m.group(0)) if m else None


_BUY_PRICE_QUERY_RE = re.compile(
    r"(?is)\b(?:"
    r"good\s+deal|fair\s+price|overpriced|good\s+buy|"
    r"listed\s+at|worth\s+it|should\s+i\s+buy|fair\s+deal"
    r")\b",
)


def is_buy_price_query(text: str) -> bool:
    """True when query states an ask/list price for acquisition judgment (not a mission budget)."""
    t = text or ""
    if not _BUY_PRICE_QUERY_RE.search(t):
        return False
    return bool(_YEAR_RE.search(t) and _AIRCRAFT_TOKEN_RE.search(t))


def extract_ask_musd(text: str) -> Optional[float]:
    """Listing ask in millions from structured buy-decision answer or query."""
    m = re.search(r"(?is)\bask:\s*\$\s*(\d+(?:\.\d+)?)\s*m\b", text or "")
    if m:
        return float(m.group(1))
    if is_buy_price_query(text):
        return extract_money_musd(text)
    return None


def extract_market_median_musd(text: str) -> Optional[float]:
    """Market median from Phase 35 buy-decision / valuation blocks."""
    m = re.search(r"(?is)\bmedian:\s*\$\s*(\d+(?:\.\d+)?)\s*m\b", text or "")
    if m:
        return float(m.group(1))
    return None


def extract_acquisition_budget_musd(text: str) -> Optional[float]:
    """Mission/acquisition budget cap (e.g. under $10M), not listing ask."""
    if is_buy_price_query(text):
        return None
    t = text or ""
    if not re.search(r"(?is)\b(?:under|below|within|budget|max(?:imum)?)\b", t):
        return None
    return extract_money_musd(t)


def extract_money_musd(text: str) -> Optional[float]:
    m = _MONEY_RE.search(text or "")
    if not m:
        return None
    val = float(m.group(1))
    # If the text explicitly says million/MM/M, treat as millions.
    tail = (text or "")[m.end() : m.end() + 12].lower()
    if "m" in tail or "million" in tail:
        return val
    # Heuristic: large numbers without "M" are probably USD, not millions.
    if val > 1000:
        return val / 1_000_000.0
    return val


def extract_pax(text: str) -> Optional[int]:
    m = _PAX_RE.search(text or "")
    return int(m.group(1)) if m else None


def mentions_nonstop(text: str) -> bool:
    return bool(_NONSTOP_RE.search(text or ""))


def extract_aircraft_like_tokens(text: str) -> List[str]:
    return list(dict.fromkeys(m.group(0).strip() for m in _AIRCRAFT_TOKEN_RE.finditer(text or "")))


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def find_section(text: str, header: str) -> str:
    """Return section body after a header like 'Verdict:' if present."""
    t = text or ""
    idx = t.lower().find(header.lower())
    if idx < 0:
        return ""
    sub = t[idx + len(header) :]
    # stop at next markdown header
    m = re.search(r"\n#{1,6}\s+", sub)
    return sub[: m.start()] if m else sub


def first_nonempty(*parts: str) -> str:
    for p in parts:
        if (p or "").strip():
            return p
    return ""


def within_any(text: str, needles: Iterable[str]) -> bool:
    nt = normalize(text)
    return any(normalize(n) in nt for n in needles if n)

