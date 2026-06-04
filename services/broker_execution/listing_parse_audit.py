"""
Phase 56 — listing parse audit (observability + guard against re-asking parsed fields).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from services.market_reality.listing_detector import _extract_ask_musd, _resolve_model


_YEAR_RE = re.compile(r"\b(19\d{2}|20[0-3]\d)\b")
_ASK_MODEL_RE = re.compile(
    r"(?is)\b(?:what\s+(?:aircraft\s+)?model|which\s+model|tell\s+me\s+the\s+model)\b"
)
_MODEL_HINT_RE = re.compile(
    r"(?is)\b(?:citation\s+)?latitude\b|\bchallenger\s+\d+\b|\bg\d{3,4}\b|\bgulfstream\s+g\d{3}\b|"
    r"\blongitude\b|\bg280\b|\bpraetor\s+\d+\b|\bfalcon\s+\w+\b"
)


def _detect_model(query: str) -> Optional[str]:
    model = _resolve_model(query)
    if model:
        return model
    m = _MODEL_HINT_RE.search(query or "")
    if not m:
        return None
    token = m.group(0).strip()
    try:
        from services.broker_reasoning.broker_reasoning_layer import _resolve_model_name

        return _resolve_model_name(token.title())
    except Exception:
        return token.title()


def build_listing_parse_audit(query: str) -> Dict[str, Any]:
    q = (query or "").strip()
    model = _detect_model(q)
    year_m = _YEAR_RE.search(q)
    price = _extract_ask_musd(q)
    audit = {
        "detected_model": model,
        "detected_year": int(year_m.group(1)) if year_m else None,
        "detected_price": price,
        "parse_success": bool(model and (year_m or price is not None)),
    }
    return audit


def attach_listing_parse_audit(query: str, data_used: dict) -> Dict[str, Any]:
    if not isinstance(data_used, dict):
        return {}
    audit = build_listing_parse_audit(query)
    data_used["listing_parse_audit"] = audit
    return audit


def strip_redundant_listing_questions(answer: str, *, data_used: dict) -> str:
    """When parse succeeded, remove lines that re-ask for model/year/price."""
    audit = data_used.get("listing_parse_audit") or {}
    if not isinstance(audit, dict) or not audit.get("parse_success"):
        return (answer or "").strip()
    lines = []
    for line in (answer or "").splitlines():
        if _ASK_MODEL_RE.search(line):
            continue
        if audit.get("detected_model") and re.search(
            r"(?is)\bwhat\s+is\s+the\s+asking\s+price\b", line
        ) and audit.get("detected_price") is not None:
            continue
        lines.append(line)
    return "\n".join(lines).strip() or (answer or "").strip()


__all__ = [
    "attach_listing_parse_audit",
    "build_listing_parse_audit",
    "strip_redundant_listing_questions",
]
