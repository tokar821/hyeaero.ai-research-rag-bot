"""
Deterministic normalization for aircraft manufacturer / model / engine strings before embedding
or cross-table joins. Does not mutate database rows; use when building RAG text or analytics.

Canonical example: Gulfstream + ("Gulfstream G650" | "G650" | "G-650" | "G 650") -> ("Gulfstream", "G650").
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# Lowercased key -> canonical manufacturer display token
_MANUFACTURER_ALIASES: dict[str, str] = {
    "gulfstream": "Gulfstream",
    "gulfstream aerospace": "Gulfstream",
    "gulfstream aerospace corporation": "Gulfstream",
    "gulfstream aerospace corp": "Gulfstream",
    "gulfstream aerospace corp.": "Gulfstream",
    "emb": "Embraer",
    "embraer s.a.": "Embraer",
    "bombardier": "Bombardier",
    "bombardier aerospace": "Bombardier",
    "textron": "Textron Aviation",
    "textron aviation": "Textron Aviation",
    "cessna": "Cessna",
    "beechcraft": "Beechcraft",
    "dassault": "Dassault",
    "dassault aviation": "Dassault",
    "boeing": "Boeing",
    "airbus": "Airbus",
    "pilatus": "Pilatus",
    "honda": "Honda Aircraft",
    "honda aircraft": "Honda Aircraft",
    "cirrus": "Cirrus",
    "daher": "Daher",
    "pilatus aircraft": "Pilatus",
}

# (lowercase manufacturer token or None for any) -> list of (regex, canonical_model).
# Order matters: longer/more specific patterns first (e.g. G650ER before G650).
_MODEL_RULES: list[tuple[Optional[str], re.Pattern, str]] = [
    (
        "gulfstream",
        re.compile(
            r"^gvi\s*\(\s*g\s*650\s*er\s*\)$",
            re.IGNORECASE,
        ),
        "G650ER",
    ),
    (
        None,
        re.compile(
            r"gvi\s*\(\s*g\s*650\s*er\s*\)",
            re.IGNORECASE,
        ),
        "G650ER",
    ),
    (
        "gulfstream",
        re.compile(
            r"^(?:gulfstream\s+)?g[\s\-–—]*650er(?:\b|[\s\-–—]|$)",
            re.IGNORECASE,
        ),
        "G650ER",
    ),
    (
        None,
        re.compile(r"^g[\s\-–—]*650er(?:\b|[\s\-–—]|$)", re.IGNORECASE),
        "G650ER",
    ),
    (
        "gulfstream",
        re.compile(
            r"^(?:gulfstream\s+)?g[\s\-–—]*650\s*(?:ms)(?:\b|[\s\-–—]|$)",
            re.IGNORECASE,
        ),
        "G650",
    ),
    (
        None,
        re.compile(r"^g[\s\-–—]*650\s*(?:ms)(?:\b|[\s\-–—]|$)", re.IGNORECASE),
        "G650",
    ),
    (
        "gulfstream",
        re.compile(
            r"^(?:gulfstream\s+)?g[\s\-–—]*650(?:\b|[\s\-–—]|$)",
            re.IGNORECASE,
        ),
        "G650",
    ),
    (
        "gulfstream",
        re.compile(r"^gulfstream\s+g[\s\-–—]*650(?:\b|[\s\-–—]|$)", re.IGNORECASE),
        "G650",
    ),
    (
        None,
        re.compile(r"^g[\s\-–—]*650(?:\b|[\s\-–—]|$)", re.IGNORECASE),
        "G650",
    ),
]


def _strip_noise(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_manufacturer(value: Optional[str]) -> Optional[str]:
    """Return canonical manufacturer label or None if empty."""
    if value is None or not str(value).strip():
        return None
    raw = _strip_noise(str(value))
    key = raw.lower()
    if key in _MANUFACTURER_ALIASES:
        return _MANUFACTURER_ALIASES[key]
    # Title-case light: keep known acronyms
    if raw.isupper() and len(raw) <= 5:
        return raw
    return raw


def normalize_model(
    manufacturer: Optional[str],
    model: Optional[str],
) -> Optional[str]:
    """
    Normalize model string using manufacturer context.
    Strips duplicate manufacturer prefix from model (e.g. 'Gulfstream G650' -> 'G650').
    """
    if model is None or not str(model).strip():
        return None
    mfr_norm = normalize_manufacturer(manufacturer)
    mod = _strip_noise(str(model))
    mfr_key = (mfr_norm or "").lower()

    # Remove leading manufacturer echo from model column
    if mfr_norm and mod.lower().startswith(mfr_key):
        rest = mod[len(mfr_norm) :].lstrip(" -_/")
        if rest:
            mod = rest

    for rule_mfr, pat, canonical in _MODEL_RULES:
        if rule_mfr is None or rule_mfr == mfr_key:
            if pat.match(mod):
                return canonical

    return mod


def normalize_aircraft_identity(
    manufacturer: Optional[str],
    model: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Return (canonical_manufacturer, canonical_model)."""
    m = normalize_manufacturer(manufacturer)
    mo = normalize_model(manufacturer, model)
    return m, mo


def normalize_engine_display(value: Optional[str]) -> Optional[str]:
    """Light cleanup for type_engine / powerplant / engine_model text fields."""
    if value is None or not str(value).strip():
        return None
    s = _strip_noise(str(value))
    s = re.sub(r"\s*/\s*", " / ", s)
    return s


def normalized_type_key(
    manufacturer: Optional[str],
    model: Optional[str],
) -> str:
    """
    Stable key for dedupe / reporting: lowercase mfr + '|' + normalized model alphanumerics.
    """
    m, mo = normalize_aircraft_identity(manufacturer, model)
    mk = re.sub(r"[^a-z0-9]", "", (m or "").lower())
    ok = re.sub(r"[^a-z0-9]", "", (mo or "").lower())
    return f"{mk}|{ok}"


def enrich_record_for_embedding(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add normalization hints to a copy of a DB row (manufacturer_canonical, model_canonical, type_key).
    Safe to merge into chunk metadata if desired.
    """
    out = dict(record)
    m = record.get("manufacturer")
    mo = record.get("model")
    mc, moc = normalize_aircraft_identity(
        str(m) if m is not None else None,
        str(mo) if mo is not None else None,
    )
    if mc:
        out["manufacturer_canonical"] = mc
    if moc:
        out["model_canonical"] = moc
    tk = normalized_type_key(
        str(m) if m is not None else None,
        str(mo) if mo is not None else None,
    )
    if tk != "|":
        out["aircraft_type_key"] = tk
    return out
