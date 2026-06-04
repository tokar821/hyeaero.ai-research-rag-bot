"""Comparison presentation guard — broker conclusion when both models are known.

Does not modify comparison engine logic. Rewrites client-facing prose when upstream
responders produced insufficient-data strings or deferral without a pick.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.client_context.recommendation_consistency import _tier_musd
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

_INSUFFICIENT_RE = re.compile(
    r"(?is)\binsufficient verified\b|\bverified catalog comparison requires\b|\bneed the second aircraft\b"
)
_DEFERRAL_RE = re.compile(
    r"(?is)\btell me what you care about most\b|\bi['']ll give you a clear pick\b"
)
_CONCLUSION_RE = re.compile(r"(?is)\b(?:choose .+ if|i would lean toward|i'd lean toward)\b")


def _known_compare_models(data_used: Optional[Dict[str, Any]], query: str = "") -> List[str]:
    du = data_used if isinstance(data_used, dict) else {}
    br = du.get("broker_reasoning") or {}
    if isinstance(br, dict):
        models = br.get("compare_models")
        if isinstance(models, list) and len(models) >= 2:
            return [str(models[0]), str(models[1])]
    frame = du.get("canonical_intent_frame") or {}
    if isinstance(frame, dict):
        scope = frame.get("aircraft_scope") or {}
        if isinstance(scope, dict):
            models = scope.get("models") or []
            if isinstance(models, list) and len(models) >= 2:
                return [str(models[0]), str(models[1])]
    if re.search(r"(?is)\bvs\.?\b|\bversus\b", query or ""):
        try:
            from services.consultant.recommendation_engine import detect_models_from_text
            from services.broker_reasoning.broker_reasoning_layer import _resolve_model_name

            raw = detect_models_from_text(query or "")
            resolved = [_resolve_model_name(m) for m in raw]
            resolved = [m for m in resolved if m]
            if len(resolved) >= 2:
                return [resolved[0], resolved[1]]
        except Exception:
            pass
    return []


def _profile_bits(model: str) -> str:
    p = AIRCRAFT_PROFILES.get(model) or {}
    bits: List[str] = []
    cat = p.get("category")
    if cat:
        bits.append(str(cat))
    rn = p.get("practical_nm")
    if rn:
        bits.append(f"~{int(rn)} nm practical")
    seats = p.get("seats")
    if seats:
        bits.append(f"{int(seats)} seats")
    return ", ".join(bits)


def _short_name(model: str) -> str:
    return model.split()[-1] if model else model


def _broker_comparison_conclusion(models: List[str], *, query: str = "") -> str:
    a, b = models[0], models[1]
    a_bits = _profile_bits(a)
    b_bits = _profile_bits(b)
    tier_a = _tier_musd(a)
    tier_b = _tier_musd(b)
    pa = AIRCRAFT_PROFILES.get(a) or {}
    pb = AIRCRAFT_PROFILES.get(b) or {}
    range_a = float(pa.get("practical_nm") or 0)
    range_b = float(pb.get("practical_nm") or 0)

    lines = [
        f"{a} vs {b}: here’s the broker read.",
        "",
        f"• {a}" + (f" — {a_bits}" if a_bits else ""),
        f"• {b}" + (f" — {b_bits}" if b_bits else ""),
    ]

    q = (query or "").lower()
    cost_sensitive = any(w in q for w in ("cost", "budget", "cheaper", "value", "buy"))

    if tier_a < tier_b * 0.9 or (cost_sensitive and tier_a <= tier_b):
        winner, other = a, b
        why = "the lower capital entry and stronger cost-to-capability balance"
        alt_why = "maximum cabin scale and range headroom"
    elif range_b > range_a + 400:
        winner, other = b, a
        why = "the extra range and cabin scale for long legs"
        alt_why = "capital efficiency and lower operating cost"
    elif range_a > range_b + 400:
        winner, other = a, b
        why = "range and cabin comfort on longer stages"
        alt_why = "a lighter cost footprint"
    elif tier_b < tier_a * 0.9:
        winner, other = b, a
        why = "value inside the same mission band"
        alt_why = "a larger cabin and higher speed"
    else:
        winner, other = a, b
        why = "the more balanced ownership economics in this pair"
        alt_why = "specifically needing the larger-cabin variant"

    lines.extend(
        [
            "",
            f"I would lean toward the {_short_name(winner)} for {why}.",
            f"Choose the {_short_name(other)} if you prioritize {alt_why}.",
        ]
    )
    return "\n".join(lines).strip()


def guard_comparison_presentation(
    answer: str,
    *,
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    text = (answer or "").strip()
    models = _known_compare_models(data_used, query=query)
    if len(models) < 2:
        return text

    needs_rewrite = bool(
        _INSUFFICIENT_RE.search(text)
        or _DEFERRAL_RE.search(text)
        or not _CONCLUSION_RE.search(text)
    )
    if not needs_rewrite:
        return text

    return _broker_comparison_conclusion(models, query=query)


__all__ = ["guard_comparison_presentation"]
