"""
Soft comparison resolution — auto-resolve shorthand model pairs when confidence is sufficient.

Thresholds (Phase 40):
  >= 95  auto-resolve
  80–94  auto-resolve with note
  < 80   clarify (fail closed)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from services.comparison.aircraft_registry_lock import lock_comparison_aircraft, resolve_to_registry_name

AUTO_RESOLVE_HIGH = 95
AUTO_RESOLVE_SOFT = 80

# Shorthand token → (registry model, confidence)
_COMPARISON_TOKEN_MAP: Dict[str, Tuple[str, int]] = {
    "longitude": ("Citation Longitude", 92),
    "latitude": ("Citation Latitude", 90),
    "phenom": ("Praetor 600", 82),
    "phenom 300": ("Praetor 600", 84),
    "phenom300": ("Praetor 600", 84),
    "challenger 300": ("Challenger 300", 97),
    "challenger 350": ("Challenger 350", 97),
    "challenger": ("Challenger 350", 88),
    "challenger 650": ("Challenger 650", 97),
    "citation jet": ("Citation Latitude", 80),
    "citation": ("Citation Latitude", 78),
    "g650": ("Gulfstream G650", 98),
    "g700": ("Gulfstream G700", 98),
    "g280": ("Gulfstream G280", 97),
    "falcon": ("Falcon 2000", 85),
    "falcon 8x": ("Falcon 8X", 97),
    "global": ("Global 6500", 85),
    "praetor": ("Praetor 600", 95),
    "cj4": ("Citation CJ4", 95),
    "cj2": ("Citation CJ2", 95),
}

_REGISTRY_FALLBACK_NOTES: Dict[str, str] = {
    "Praetor 600": (
        "Shorthand 'Phenom' mapped to Praetor 600 — the verified Embraer super-mid in catalog. "
        "Say Phenom 300 explicitly if you meant the light jet class."
    ),
    "Citation Latitude": (
        "Shorthand 'citation jet' mapped to Citation Latitude — specify Longitude or CJ4 if you meant a different Citation."
    ),
}


@dataclass
class SoftComparisonResolution:
    models: Tuple[str, ...] = ()
    confidences: Tuple[int, ...] = ()
    action: str = "clarify"  # auto | auto_with_note | clarify
    notes: Tuple[str, ...] = field(default_factory=tuple)
    raw_tokens: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "models": list(self.models),
            "confidences": list(self.confidences),
            "action": self.action,
            "notes": list(self.notes),
            "raw_tokens": list(self.raw_tokens),
        }


_VS_SPLIT_RE = re.compile(r"\b(?:vs\.?|versus)\b", re.I)


def _extract_comparison_tokens(query: str) -> Tuple[str, str]:
    parts = _VS_SPLIT_RE.split(query or "", maxsplit=1)
    if len(parts) != 2:
        return "", ""
    return parts[0].strip(), parts[1].strip()


def _normalize_token(token: str) -> str:
    t = re.sub(r"[^\w\s+\-]", " ", (token or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _resolve_side(token: str) -> Tuple[Optional[str], int, str]:
    raw = _normalize_token(token)
    if not raw:
        return None, 0, raw

    # Direct registry / AKAL resolution first (highest confidence).
    reg = resolve_to_registry_name(raw)
    if reg:
        return reg, 98, raw

    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias

    alias = resolve_aircraft_alias(raw)
    if alias:
        reg2 = resolve_to_registry_name(alias)
        if reg2:
            return reg2, 96, raw

    from services.consultant.recommendation_engine import detect_models_from_text

    detected = detect_models_from_text(token)
    if detected:
        lock = lock_comparison_aircraft(detected)
        if lock.canonical:
            return lock.canonical[0], 94, raw
        if len(detected) == 1:
            explicit = str(detected[0]).strip()
            if explicit and re.search(r"\d", explicit):
                return explicit, 94, raw

    # Longest shorthand match.
    best: Optional[Tuple[str, int]] = None
    best_key = ""
    for key, (model, conf) in sorted(_COMPARISON_TOKEN_MAP.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(key)}\b", raw):
            lock = lock_comparison_aircraft([model])
            if lock.canonical:
                if best is None or conf > best[1]:
                    best = (lock.canonical[0], conf)
                    best_key = key

    if best:
        return best[0], best[1], raw

    return None, 0, raw


def soft_resolve_comparison(query: str) -> Optional[SoftComparisonResolution]:
    """Resolve a comparison query to registry-verified model pair when confidence allows."""
    if not _VS_SPLIT_RE.search(query or ""):
        return None

    left_tok, right_tok = _extract_comparison_tokens(query)
    if not left_tok or not right_tok:
        return None

    left, lconf, lraw = _resolve_side(left_tok)
    right, rconf, rraw = _resolve_side(right_tok)

    notes: List[str] = []
    if left in _REGISTRY_FALLBACK_NOTES:
        notes.append(_REGISTRY_FALLBACK_NOTES[left])
    if right in _REGISTRY_FALLBACK_NOTES:
        notes.append(_REGISTRY_FALLBACK_NOTES[right])

    min_conf = min(lconf, rconf) if left and right else 0
    models: Tuple[str, ...] = tuple(m for m in (left, right) if m)

    if len(models) < 2:
        return SoftComparisonResolution(
            models=models,
            confidences=tuple(c for c in (lconf, rconf) if c),
            action="clarify",
            notes=tuple(notes),
            raw_tokens=(lraw, rraw),
        )

    if min_conf >= AUTO_RESOLVE_HIGH:
        action = "auto"
    elif min_conf >= AUTO_RESOLVE_SOFT:
        action = "auto_with_note"
        notes = notes or ("Resolved from shorthand model names.",)
    else:
        action = "clarify"

    return SoftComparisonResolution(
        models=models,
        confidences=(lconf, rconf),
        action=action,
        notes=tuple(notes),
        raw_tokens=(lraw, rraw),
    )


def comparison_models_for_dispatch(resolution: Optional[SoftComparisonResolution]) -> Optional[List[str]]:
    """Return compare_models list when soft resolution clears threshold."""
    if resolution is None:
        return None
    if resolution.action not in ("auto", "auto_with_note"):
        return None
    if len(resolution.models) >= 2:
        return list(resolution.models[:2])
    return None


__all__ = [
    "AUTO_RESOLVE_HIGH",
    "AUTO_RESOLVE_SOFT",
    "SoftComparisonResolution",
    "comparison_models_for_dispatch",
    "soft_resolve_comparison",
]
