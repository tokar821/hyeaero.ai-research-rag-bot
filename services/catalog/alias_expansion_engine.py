"""
Phase 53 — shorthand alias expansion before comparison dispatch.

Resolves broker tokens (CJ4, Phenom, Falcon, Longitude, etc.) to registry-verified
canonical names. Measurement/dispatch aid only — does not alter IntentLock or routing order.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

# Longest-match shorthand → (canonical display, confidence 0–100)
_SHORTHAND_MAP: Dict[str, Tuple[str, int]] = {
    "falcon 2000": ("Falcon 2000", 97),
    "falcon 2000lx": ("Falcon 2000", 95),
    "falcon 8x": ("Falcon 8X", 98),
    "falcon 7x": ("Falcon 7X", 97),
    "falcon": ("Falcon 2000", 85),
    "longitude": ("Citation Longitude", 95),
    "citation longitude": ("Citation Longitude", 98),
    "latitude": ("Citation Latitude", 93),
    "citation latitude": ("Citation Latitude", 97),
    "challenger 350": ("Challenger 350", 98),
    "challenger 650": ("Challenger 650", 98),
    "challenger": ("Challenger 350", 88),
    "cj4": ("Citation CJ4", 96),
    "cj3+": ("Citation CJ3+", 95),
    "cj3": ("Citation CJ3+", 90),
    "cj2": ("Citation CJ2", 95),
    "phenom 300e": ("Phenom 300E", 92),
    "phenom 300": ("Phenom 300", 94),
    "phenom": ("Phenom 300", 88),
    "praetor 600": ("Praetor 600", 97),
    "praetor": ("Praetor 600", 95),
    "g650er": ("Gulfstream G650ER", 98),
    "g650": ("Gulfstream G650", 98),
    "g700": ("Gulfstream G700", 98),
    "g550": ("Gulfstream G550", 97),
    "g280": ("Gulfstream G280", 97),
    "g500": ("Gulfstream G500", 96),
    "global 7500": ("Global 7500", 98),
    "global 6500": ("Global 6500", 97),
    "global": ("Global 6500", 85),
    "pc-24": ("Pilatus PC-24", 95),
    "pc24": ("Pilatus PC-24", 93),
    "learjet 75": ("Learjet 75", 94),
}


def _normalize_key(raw: str) -> str:
    return re.sub(r"\s+", " ", (raw or "").lower().strip())


def expand_shorthand_token(raw: str) -> Tuple[Optional[str], int]:
    """Expand a single token/phrase to canonical model name."""
    key = _normalize_key(raw)
    if not key:
        return None, 0

    from services.catalog.catalog_alias_resolver import resolve_canonical_display_name

    display = resolve_canonical_display_name(raw)
    if display and display != raw.strip():
        return display, 96

    best: Optional[Tuple[str, int]] = None
    for shorthand, (model, conf) in sorted(_SHORTHAND_MAP.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(shorthand)}\b", key):
            if best is None or conf > best[1]:
                best = (model, conf)
    if best:
        return best

    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias
    from services.comparison.aircraft_registry_lock import lock_comparison_aircraft

    alias = resolve_aircraft_alias(raw)
    if alias:
        lock = lock_comparison_aircraft([alias])
        if lock.canonical:
            return lock.canonical[0], 94

    lock = lock_comparison_aircraft([display or raw])
    if lock.canonical:
        return lock.canonical[0], 90

    return None, 0


def expand_models_in_text(models: Sequence[str]) -> List[str]:
    """Expand a sequence of detected model strings to canonical registry names."""
    out: List[str] = []
    seen: set[str] = set()
    for raw in models:
        canon, _ = expand_shorthand_token(raw)
        if canon:
            k = canon.lower()
            if k not in seen:
                seen.add(k)
                out.append(canon)
    return out


def resolve_comparison_models_from_query(query: str) -> List[str]:
    """
    Resolve two comparison aircraft from a vs/versus query using alias expansion.
    Falls back to detect_models_from_text when soft resolution is insufficient.
    """
    q = (query or "").strip()
    if not q:
        return []

    from services.consultant.recommendation_engine import detect_models_from_text

    _VS_RE = re.compile(r"\b(?:vs\.?|versus)\b", re.I)
    if _VS_RE.search(q):
        parts = _VS_RE.split(q, maxsplit=1)
        if len(parts) == 2:
            expanded: List[str] = []
            for side in parts:
                side = side.strip()
                canon, conf = expand_shorthand_token(side)
                if canon and conf >= 80:
                    expanded.append(canon)
            if len(expanded) >= 2:
                from services.comparison.aircraft_registry_lock import lock_comparison_aircraft

                lock = lock_comparison_aircraft(expanded)
                canonical = [m for m in lock.canonical if m]
                if len(canonical) >= 2:
                    return canonical[:2]
                return expanded[:2]

    from services.broker_reasoning.comparison_soft_resolution import (
        comparison_models_for_dispatch,
        soft_resolve_comparison,
    )

    soft = soft_resolve_comparison(q)
    dispatch_models = comparison_models_for_dispatch(soft)
    if dispatch_models and len(dispatch_models) >= 2:
        return dispatch_models[:2]

    detected = detect_models_from_text(q)
    expanded = expand_models_in_text(detected)
    if len(expanded) >= 2:
        return expanded[:2]

    from services.comparison.aircraft_registry_lock import lock_comparison_aircraft

    lock = lock_comparison_aircraft(expanded or detected)
    return [m for m in lock.canonical if m][:2]


__all__ = [
    "expand_models_in_text",
    "expand_shorthand_token",
    "resolve_comparison_models_from_query",
]
