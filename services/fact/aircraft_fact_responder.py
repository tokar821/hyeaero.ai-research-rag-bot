"""
Deterministic aircraft fact responder — verified catalog data only.

Returns 1–3 broker-tone sentences. No mission language, recommendations, or verdict blocks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from services.aircraft_truth.constants import (
    UNIFIED_CATALOG_MISS_MESSAGE,
    UNIFIED_FACT_UNVERIFIED_MESSAGE,
)
from services.aircraft_truth.validator import extract_verified_facts

_RUNWAY_CLASS_LABELS = {
    "short_field": "short-field capable",
    "regional": "regional-jet runway",
    "super_mid": "super-midsize runway",
    "large_cabin": "large-cabin runway",
    "ultra_long": "ultra-long-range runway",
}

_FORBIDDEN_PHRASES = re.compile(
    r"\b(?:good\s+fit|recommend|mission|nonstop|shortlist|best\s+jet|compare|versus)\b",
    re.I,
)


def _format_usd_band(value: float) -> str:
    if value >= 1_000_000:
        low = value * 0.88
        high = value * 1.12

        def _fmt(n: float) -> str:
            m = n / 1_000_000
            if m >= 10:
                return f"${m:.0f}M"
            return f"${m:.1f}M".replace(".0M", "M")

        return f"{_fmt(low)}–{_fmt(high)}"
    if value >= 1_000:
        return f"${value:,.0f}"
    return f"${value:.0f}"


def _lookup_facts(model: str) -> tuple[Optional[Any], Optional[Any]]:
    facts, _missing = extract_verified_facts(model)
    spec = None
    try:
        from services.data_authority.aircraft_spec_repository import get_verified_spec

        spec = get_verified_spec(model)
    except Exception:
        pass
    return facts, spec


def _guard_answer(text: str) -> str:
    if _FORBIDDEN_PHRASES.search(text or ""):
        return UNIFIED_FACT_UNVERIFIED_MESSAGE
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    return " ".join(sentences[:2])


def _respond_seats(model: str, facts: Any, spec: Any) -> str:
    pax = None
    if facts is not None:
        pax = facts.max_passengers
    elif spec is not None:
        pax = spec.pax_max_long_range or spec.pax_typical
    if not pax:
        return UNIFIED_FACT_UNVERIFIED_MESSAGE
    return _guard_answer(
        f"The {model} seats up to {int(pax)} passengers per verified catalog data."
    )


def _respond_baggage(model: str, facts: Any, spec: Any) -> str:
    cu_ft = None
    if facts is not None:
        cu_ft = facts.baggage_volume_cu_ft
    if not cu_ft and spec is not None:
        cu_ft = getattr(spec, "baggage_volume_cu_ft", None)
    if not cu_ft:
        return UNIFIED_FACT_UNVERIFIED_MESSAGE
    return _guard_answer(
        f"The {model} offers approximately {int(round(cu_ft))} cu ft of baggage volume "
        f"in verified catalog configuration."
    )


def _respond_range(model: str, facts: Any, spec: Any) -> str:
    nm = None
    if facts is not None:
        nm = facts.practical_range_nm
    elif spec is not None:
        nm = spec.practical_nm
    if not nm:
        return UNIFIED_FACT_UNVERIFIED_MESSAGE
    return _guard_answer(
        f"The {model} has a verified practical range of approximately {int(round(nm))} nm."
    )


def _respond_speed(model: str, facts: Any, spec: Any) -> str:
    speed = None
    if spec is not None:
        speed = getattr(spec, "cruise_speed_knots", None) or getattr(spec, "max_speed_knots", None)
    if not speed:
        return UNIFIED_FACT_UNVERIFIED_MESSAGE
    return _guard_answer(
        f"The {model} has a verified cruise speed of approximately {int(round(speed))} knots."
    )


def _respond_runway(model: str, facts: Any, spec: Any) -> str:
    if facts is not None and facts.runway_class:
        label = _RUNWAY_CLASS_LABELS.get(facts.runway_class, facts.runway_class.replace("_", " "))
        return _guard_answer(
            f"The {model} is classified as {label} in verified catalog runway data."
        )
    if spec is not None and spec.runway_ft:
        return _guard_answer(
            f"The {model} requires approximately {int(spec.runway_ft):,} feet of runway "
            f"per verified catalog data."
        )
    return UNIFIED_FACT_UNVERIFIED_MESSAGE


def _respond_worth(model: str, facts: Any, spec: Any) -> str:
    price = None
    if spec is not None:
        price = spec.average_pre_owned_price
    if not price or price <= 0:
        return UNIFIED_FACT_UNVERIFIED_MESSAGE
    band = _format_usd_band(float(price))
    return _guard_answer(
        f"Pre-owned {model} aircraft typically trade in the {band} range based on verified "
        f"market reference data; configuration, age, and program status move the number."
    )


_FIELD_HANDLERS = {
    "seats": _respond_seats,
    "baggage": _respond_baggage,
    "range": _respond_range,
    "speed": _respond_speed,
    "runway": _respond_runway,
    "worth": _respond_worth,
    "value": _respond_worth,
    "price": _respond_worth,
}


def respond_aircraft_fact(model: str, field: str) -> str:
    """
    Return a deterministic 1–3 sentence broker answer for (model, field).

    Lookup order: extract_verified_facts, then get_verified_spec.
    """
    name = (model or "").strip()
    fld = (field or "").strip().lower()
    if not name or not fld:
        return UNIFIED_CATALOG_MISS_MESSAGE if name else UNIFIED_FACT_UNVERIFIED_MESSAGE

    handler = _FIELD_HANDLERS.get(fld)
    if handler is None:
        return UNIFIED_FACT_UNVERIFIED_MESSAGE

    facts, spec = _lookup_facts(name)
    if facts is None and spec is None:
        return UNIFIED_CATALOG_MISS_MESSAGE

    return handler(name, facts, spec)


__all__ = ["respond_aircraft_fact"]
