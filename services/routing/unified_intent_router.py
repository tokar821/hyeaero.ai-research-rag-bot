"""
Unified intent router — lightweight heuristic classification (no LLM).

Phase 5 Step 1: AIRCRAFT_FACT, AIRCRAFT_MARKET_FACT, or OTHER (existing pipeline).
Shadow-only secondary labels (capability / comparison / mission likely) for observability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from services.catalog.catalog_alias_resolver import (
    resolve_canonical_display_name,
    resolve_catalog_profile_key,
)

_MODEL_CONFIDENCE_THRESHOLD = 0.7
_CAPABILITY_BORDERLINE_THRESHOLD = 0.55
PROMOTION_CONTRACT_VERSION = "v1"


class UnifiedIntent(str, Enum):
    AIRCRAFT_FACT = "aircraft_fact"
    AIRCRAFT_MARKET_FACT = "aircraft_market_fact"
    OTHER = "other"


class UnifiedSecondaryIntent(str, Enum):
    """Shadow-only labels — not used for routing enforcement."""

    AIRCRAFT_CAPABILITY_LIKELY = "aircraft_capability_likely"
    AIRCRAFT_COMPARISON_LIKELY = "aircraft_comparison_likely"
    AIRCRAFT_MISSION_LIKELY = "aircraft_mission_likely"


class UnifiedExecutionPath(str, Enum):
    """Authoritative execution path — resolved once at route finalization."""

    NONE = "none"
    AIRCRAFT_FACT = "aircraft_fact"
    AIRCRAFT_MARKET_FACT = "aircraft_market_fact"
    CAPABILITY = "capability"
    COMPARISON = "comparison"
    ALTERNATIVE = "alternative"


# Declarative Step 2+ execution targets — not invoked by current routing.
_SECONDARY_INTENT_EXECUTION_TARGETS: Dict[UnifiedSecondaryIntent, str] = {
    UnifiedSecondaryIntent.AIRCRAFT_CAPABILITY_LIKELY: "capability_responder",
    UnifiedSecondaryIntent.AIRCRAFT_COMPARISON_LIKELY: "comparison_responder",
    UnifiedSecondaryIntent.AIRCRAFT_MISSION_LIKELY: "mission_responder",
}


_SEATS_RE = re.compile(
    r"\b(?:how\s+many\s+)?(?:seats?|passengers?|pax|seating\s+capacity)\b",
    re.I,
)
_BAGGAGE_RE = re.compile(r"\b(?:baggage|luggage|cargo\s+volume)\b", re.I)
_RANGE_RE = re.compile(r"\b(?:range|how\s+far)\b", re.I)
_SPEED_RE = re.compile(
    r"\b(?:max(?:imum)?\s+)?(?:cruise\s+)?speed|how\s+(?:fast|quick)\b",
    re.I,
)
_RUNWAY_RE = re.compile(
    r"\b(?:runway|takeoff\s+distance|landing\s+distance|field\s+length)\b",
    re.I,
)
_MARKET_RE = re.compile(
    r"\b(?:worth|value|valued\s+at|market\s+value|resale\s+value|pre[-\s]?owned\s+price|asking\s+price|"
    r"what\s+(?:does|do)\s+.+\s+(?:cost|go\s+for)|how\s+much\s+(?:is|does|are)|"
    r"sell(?:s)?\s+for)\b",
    re.I,
)
_PRICE_RE = re.compile(r"\b(?:price|cost|asking)\b", re.I)

_CAPABILITY_TRIGGER_RE = re.compile(
    r"\b(?:"
    r"can(?:\s+\w+){0,6}\s+fly|"
    r"capable\s+of|"
    r"make\s+it\s+(?:from|to)|"
    r"fly\s+(?:from|nonstop|to|between)"
    r")\b",
    re.I,
)

_CAPABILITY_BLOCKERS = re.compile(
    r"\b(?:"
    r"can(?:\s+\w+){0,4}\s+fly|"
    r"fly\s+(?:nonstop|from|to|between)|"
    r"nonstop|"
    r"make\s+it\s+to|"
    r"feasib(?:le|ility)|"
    r"city\s+pair|"
    r"\bfrom\s+[a-z]{3,}\s+to\s+[a-z]{3,}|"
    r"\b(?:nyc|new\s+york|paris|london|tokyo|sfo|lax|mia|dubai|singapore)\b.*\b(?:to|from)\b|"
    r"\b(?:to|from)\b.*\b(?:nyc|new\s+york|paris|london|tokyo|sfo|lax|mia|dubai|singapore)\b|"
    r"route|mission|trip|leg\b"
    r")\b",
    re.I,
)

_RECOMMENDATION_BLOCKERS = re.compile(
    r"\b(?:compare|versus|vs\.?|recommend|best\s+jet|shortlist|which\s+(?:jet|aircraft|plane)|"
    r"better\s+(?:value|option|jet)|alternatives?\s+to)\b",
    re.I,
)

_COMPARISON_SECONDARY_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|better\s+than|which\s+is\s+better)\b",
    re.I,
)

_MISSION_SECONDARY_RE = re.compile(
    r"\b(?:"
    r"recommend|best\s+jet|shortlist|which\s+(?:jet|aircraft|plane)|"
    r"options?\s+for|what\s+(?:jet|aircraft)|good\s+fit|"
    r"fits?\s+my|acquisition|buy\s+a\s+jet"
    r")\b",
    re.I,
)


@dataclass(frozen=True)
class UnifiedIntentRoute:
    intent: UnifiedIntent
    model: Optional[str] = None
    field: Optional[str] = None
    confidence: float = 0.0
    model_confidence: float = 0.0
    secondary_intent: Optional[UnifiedSecondaryIntent] = None
    execution_path: UnifiedExecutionPath = UnifiedExecutionPath.NONE
    signals: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "model": self.model,
            "field": self.field,
            "confidence": round(float(self.confidence), 4),
            "model_confidence": round(float(self.model_confidence), 4),
            "secondary_intent": (
                self.secondary_intent.value if self.secondary_intent is not None else None
            ),
            "execution_path": self.execution_path.value,
            "signals": list(self.signals),
        }


def _detect_fact_field(ql: str) -> Optional[str]:
    fields: List[str] = []
    if _SEATS_RE.search(ql):
        fields.append("seats")
    if _BAGGAGE_RE.search(ql):
        fields.append("baggage")
    if _RANGE_RE.search(ql):
        fields.append("range")
    if _SPEED_RE.search(ql):
        fields.append("speed")
    if _RUNWAY_RE.search(ql):
        fields.append("runway")
    if len(fields) != 1:
        return None
    return fields[0]


def _detect_market_field(ql: str) -> Optional[str]:
    if re.search(r"\bsell(?:s)?\s+for\b", ql):
        return "price"
    if re.search(r"\bresale\s+value\b", ql):
        return "value"
    if re.search(r"\bgo\s+for\b", ql):
        return "price"
    if _MARKET_RE.search(ql):
        if re.search(r"\bworth\b", ql):
            return "worth"
        if re.search(r"\bvalue\b", ql):
            return "value"
        if _PRICE_RE.search(ql):
            return "price"
        return "worth"
    if _PRICE_RE.search(ql) and not _SEATS_RE.search(ql) and not _RANGE_RE.search(ql):
        return "price"
    return None


def _mentioned_models(query: str) -> List[str]:
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        return list(detect_models_from_text(query or ""))
    except Exception:
        return []


def _dedupe_canonical_models(models: List[str]) -> List[str]:
    """Collapse alias duplicates (e.g. G650 + Gulfstream G650) to one canonical name."""
    seen: set[str] = set()
    out: List[str] = []
    for raw in models:
        display = resolve_canonical_display_name(raw)
        key = resolve_catalog_profile_key(raw) or resolve_catalog_profile_key(display) or display
        norm = (key or display or raw).strip()
        if not norm:
            continue
        low = norm.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(norm)
    return out


def _compound_alias_in_query(ql: str) -> Optional[Tuple[str, float]]:
    """Match longest compound alias phrase present in query text."""
    try:
        from services.catalog.catalog_alias_resolver import _DISPLAY_ALIASES
    except Exception:
        return None
    for alias_key in sorted(_DISPLAY_ALIASES.keys(), key=len, reverse=True):
        if alias_key in ql:
            display = _DISPLAY_ALIASES[alias_key]
            key = resolve_catalog_profile_key(display) or display
            return key, 0.95
    return None


def _model_explicit_in_query(model: str, ql: str) -> bool:
    """True when the resolved model (or a known alias) appears explicitly in the query."""
    if not model:
        return False
    if model.lower() in ql:
        return True
    try:
        from services.catalog.catalog_alias_resolver import _DISPLAY_ALIASES

        for alias_key, display in _DISPLAY_ALIASES.items():
            if display.lower() == model.lower() and alias_key in ql:
                return True
    except Exception:
        pass
    return False


def _extract_capability_aircraft_name(query: str) -> Optional[str]:
    """Extract named aircraft from capability phrasing when catalog detection misses."""
    q = (query or "").strip()
    patterns = (
        r"\b(?:can|is)\s+(?:a|an|the)\s+(.+?)\s+(?:fly|capable|make\s+it)\b",
        r"\b(.+?)\s+capable\s+of\b",
    )
    for pat in patterns:
        m = re.search(pat, q, re.I)
        if not m:
            continue
        name = re.sub(r"\s+(?:fly|from|to|nonstop).*$", "", m.group(1).strip(), flags=re.I).strip()
        if name and len(name) >= 3:
            return resolve_canonical_display_name(name) or name
    return None


def _resolve_model_from_query(query: str) -> Tuple[Optional[str], float]:
    """
    Resolve catalog model with confidence score.

    Returns (model, confidence). Confidence below ``_MODEL_CONFIDENCE_THRESHOLD``
    must not be used for FACT/MARKET_FACT routing unless capability borderline rules apply.
    """
    q = (query or "").strip()
    ql = q.lower()
    if not q:
        return None, 0.0

    compound = _compound_alias_in_query(ql)
    if compound:
        return compound

    candidates: List[Tuple[str, float, str]] = []
    seen_keys: set[str] = set()

    for raw in _dedupe_canonical_models(_mentioned_models(q)):
        key = resolve_catalog_profile_key(raw) or raw
        if not key:
            continue
        key_low = key.lower()
        if key_low in seen_keys:
            continue
        seen_keys.add(key_low)
        conf = 0.95 if key.lower() in ql or raw.lower() in ql else 0.88
        candidates.append((key, conf, "detect_models"))

    try:
        from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

        for name in sorted(AIRCRAFT_PROFILES.keys(), key=len, reverse=True):
            if name.lower() in ql:
                key = resolve_catalog_profile_key(name)
                if not key:
                    continue
                key_low = key.lower()
                if key_low in seen_keys:
                    continue
                seen_keys.add(key_low)
                candidates.append((key, 0.92, "profile_substring"))
    except Exception:
        pass

    has_detect_models_match = any(c[2] == "detect_models" and c[1] >= 0.9 for c in candidates)

    if not has_detect_models_match:
        try:
            from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

            tokens = [t for t in re.findall(r"[\w.-]+", ql) if len(t) >= 4]
            for token in tokens:
                partial_matches = [
                    name
                    for name in AIRCRAFT_PROFILES
                    if token in name.lower().split() or name.lower().endswith(f" {token}")
                ]
                if len(partial_matches) == 1:
                    key = resolve_catalog_profile_key(partial_matches[0])
                    if key and key.lower() not in seen_keys:
                        seen_keys.add(key.lower())
                        candidates.append((key, 0.62, "token_partial"))
                elif len(partial_matches) > 1:
                    candidates.append((partial_matches[0], 0.48, "token_ambiguous"))
        except Exception:
            pass

    if not candidates:
        extracted = _extract_capability_aircraft_name(q)
        if extracted:
            key = resolve_catalog_profile_key(extracted)
            if key:
                return key, 0.88
            return extracted, 0.65
        return None, 0.0

    candidates.sort(key=lambda c: c[1], reverse=True)
    best_model, best_conf, _source = candidates[0]

    distinct_models = {c[0] for c in candidates if c[1] >= 0.6}
    if len(distinct_models) > 1:
        best_conf = min(best_conf, 0.55)

    top_two = [c for c in candidates if c[1] >= best_conf - 0.05][:2]
    if len(top_two) == 2 and top_two[0][0] != top_two[1][0]:
        best_conf = min(best_conf, 0.58)

    return best_model, best_conf


def _apply_model_confidence_gate(
    model: Optional[str],
    model_confidence: float,
    signals: List[str],
) -> Tuple[Optional[str], float]:
    if model and model_confidence < _MODEL_CONFIDENCE_THRESHOLD:
        signals.append("model_confidence_below_threshold")
        return None, model_confidence
    return model, model_confidence


def _has_capability_or_route_signals(ql: str) -> bool:
    return bool(_CAPABILITY_BLOCKERS.search(ql) or _CAPABILITY_TRIGGER_RE.search(ql))


def _has_recommendation_signals(ql: str) -> bool:
    return bool(_RECOMMENDATION_BLOCKERS.search(ql))


def _distinct_field_types(ql: str) -> set[str]:
    types: set[str] = set()
    if _SEATS_RE.search(ql):
        types.add("seats")
    if _BAGGAGE_RE.search(ql):
        types.add("baggage")
    if _RANGE_RE.search(ql):
        types.add("range")
    if _SPEED_RE.search(ql):
        types.add("speed")
    if _RUNWAY_RE.search(ql):
        types.add("runway")
    if _detect_market_field(ql):
        types.add("market")
    return types


def _detect_secondary_intent(
    ql: str,
    *,
    models: List[str],
    query: str = "",
) -> Optional[UnifiedSecondaryIntent]:
    """Shadow-only secondary label — capability prioritized over false comparison duplicates."""
    from services.comparison.alternative_pipeline_responder import (
        is_alternative_execution_query,
        is_explicit_comparison_query,
    )

    q = query or ql
    if is_alternative_execution_query(q) and not is_explicit_comparison_query(q):
        return None

    canonical = _dedupe_canonical_models(models)

    if _has_capability_or_route_signals(ql):
        return UnifiedSecondaryIntent.AIRCRAFT_CAPABILITY_LIKELY
    if _COMPARISON_SECONDARY_RE.search(ql) or len(canonical) >= 2:
        return UnifiedSecondaryIntent.AIRCRAFT_COMPARISON_LIKELY
    if _has_recommendation_signals(ql) and len(canonical) >= 2:
        return UnifiedSecondaryIntent.AIRCRAFT_COMPARISON_LIKELY
    if _MISSION_SECONDARY_RE.search(ql):
        return UnifiedSecondaryIntent.AIRCRAFT_MISSION_LIKELY
    if not _has_capability_or_route_signals(ql) and re.search(
        r"\b(?:nonstop|transatlantic|route)\b", ql, re.I
    ):
        return UnifiedSecondaryIntent.AIRCRAFT_MISSION_LIKELY
    return None


def _canonical_compare_models(query: str) -> List[str]:
    from services.comparison.aircraft_registry_lock import lock_comparison_aircraft

    lock = lock_comparison_aircraft(_mentioned_models(query or ""))
    return list(lock.canonical)


def _resolve_execution_path(
    query: str,
    *,
    intent: UnifiedIntent,
    model: Optional[str],
    field: Optional[str],
    secondary_intent: Optional[UnifiedSecondaryIntent],
) -> UnifiedExecutionPath:
    """Authoritative execution path — resolved once; never re-evaluated downstream."""
    if intent == UnifiedIntent.AIRCRAFT_FACT and model and field:
        return UnifiedExecutionPath.AIRCRAFT_FACT
    if intent == UnifiedIntent.AIRCRAFT_MARKET_FACT and model and field:
        return UnifiedExecutionPath.AIRCRAFT_MARKET_FACT
    if intent != UnifiedIntent.OTHER:
        return UnifiedExecutionPath.NONE

    from services.comparison.alternative_pipeline_responder import (
        _resolve_alternative_target,
        is_alternative_execution_query,
        is_explicit_comparison_query,
    )

    q = query or ""
    if is_alternative_execution_query(q) and not is_explicit_comparison_query(q):
        if _resolve_alternative_target(q):
            return UnifiedExecutionPath.ALTERNATIVE

    if secondary_intent == UnifiedSecondaryIntent.AIRCRAFT_CAPABILITY_LIKELY and model:
        return UnifiedExecutionPath.CAPABILITY

    if is_explicit_comparison_query(q):
        return UnifiedExecutionPath.COMPARISON

    if secondary_intent == UnifiedSecondaryIntent.AIRCRAFT_COMPARISON_LIKELY:
        return UnifiedExecutionPath.COMPARISON

    return UnifiedExecutionPath.NONE


def _finalize_route(
    *,
    intent: UnifiedIntent,
    model: Optional[str],
    field: Optional[str],
    confidence: float,
    model_confidence: float,
    signals: List[str],
    query: str,
) -> UnifiedIntentRoute:
    models = _mentioned_models(query)
    secondary = _detect_secondary_intent((query or "").lower(), models=models, query=query)
    execution_path = _resolve_execution_path(
        query,
        intent=intent,
        model=model,
        field=field,
        secondary_intent=secondary,
    )
    return UnifiedIntentRoute(
        intent=intent,
        model=model,
        field=field,
        confidence=confidence,
        model_confidence=model_confidence,
        secondary_intent=secondary,
        execution_path=execution_path,
        signals=tuple(signals),
    )


def classify_unified_intent(query: str) -> UnifiedIntentRoute:
    """
    Classify a user query into AIRCRAFT_FACT, AIRCRAFT_MARKET_FACT, or OTHER.

    Uses only lightweight heuristics — no LLM calls.
    """
    q = (query or "").strip()
    ql = q.lower()
    signals: List[str] = []

    if not q:
        return _finalize_route(
            intent=UnifiedIntent.OTHER,
            model=None,
            field=None,
            confidence=0.0,
            model_confidence=0.0,
            signals=signals,
            query=q,
        )

    raw_model, raw_model_conf = _resolve_model_from_query(q)
    model, model_confidence = _apply_model_confidence_gate(raw_model, raw_model_conf, signals)

    if _has_recommendation_signals(ql):
        signals.append("recommendation_or_comparison")
        return _finalize_route(
            intent=UnifiedIntent.OTHER,
            model=model,
            field=None,
            confidence=0.92,
            model_confidence=model_confidence,
            signals=signals,
            query=q,
        )

    fact_field = _detect_fact_field(ql)
    market_field = _detect_market_field(ql)
    has_capability = _has_capability_or_route_signals(ql)

    if fact_field and market_field:
        signals.extend(["mixed_fact_and_market", fact_field, market_field])
        return _finalize_route(
            intent=UnifiedIntent.OTHER,
            model=model,
            field=fact_field,
            confidence=0.88,
            model_confidence=model_confidence,
            signals=signals,
            query=q,
        )

    if has_capability and (fact_field or market_field):
        signals.extend(["mixed_fact_and_capability", *([fact_field] if fact_field else [])])
        return _finalize_route(
            intent=UnifiedIntent.OTHER,
            model=model,
            field=fact_field or market_field,
            confidence=0.9,
            model_confidence=model_confidence,
            signals=signals,
            query=q,
        )

    if has_capability:
        cap_model = model
        cap_conf = model_confidence
        if (
            not cap_model
            and raw_model
            and raw_model_conf >= _CAPABILITY_BORDERLINE_THRESHOLD
            and _model_explicit_in_query(raw_model, ql)
        ):
            cap_model = raw_model
            cap_conf = raw_model_conf
            signals.append("capability_borderline_model")
        signals.append("capability_or_route")
        return _finalize_route(
            intent=UnifiedIntent.OTHER,
            model=cap_model,
            field=None,
            confidence=0.85,
            model_confidence=cap_conf,
            signals=signals,
            query=q,
        )

    if len(_distinct_field_types(ql)) > 1:
        signals.append("multi_fact_fields")
        return _finalize_route(
            intent=UnifiedIntent.OTHER,
            model=model,
            field=fact_field,
            confidence=0.8,
            model_confidence=model_confidence,
            signals=signals,
            query=q,
        )

    if market_field and not fact_field:
        if not model:
            signals.extend(["market_field", market_field, "no_model"])
            return _finalize_route(
                intent=UnifiedIntent.OTHER,
                model=None,
                field=market_field,
                confidence=0.45,
                model_confidence=model_confidence,
                signals=signals,
                query=q,
            )
        signals.extend(["market_field", market_field, "model_resolved"])
        return _finalize_route(
            intent=UnifiedIntent.AIRCRAFT_MARKET_FACT,
            model=model,
            field=market_field,
            confidence=0.88,
            model_confidence=model_confidence,
            signals=signals,
            query=q,
        )

    if fact_field:
        if not model:
            signals.extend(["fact_field", fact_field, "no_model"])
            return _finalize_route(
                intent=UnifiedIntent.OTHER,
                model=None,
                field=fact_field,
                confidence=0.45,
                model_confidence=model_confidence,
                signals=signals,
                query=q,
            )
        signals.extend(["fact_field", fact_field, "model_resolved"])
        return _finalize_route(
            intent=UnifiedIntent.AIRCRAFT_FACT,
            model=model,
            field=fact_field,
            confidence=0.9,
            model_confidence=model_confidence,
            signals=signals,
            query=q,
        )

    return _finalize_route(
        intent=UnifiedIntent.OTHER,
        model=model,
        field=None,
        confidence=0.35,
        model_confidence=model_confidence,
        signals=signals,
        query=q,
    )


def _intent_agreement(unified: UnifiedIntent, qri_intent: str) -> str:
    qri = (qri_intent or "").strip().lower()
    if unified == UnifiedIntent.AIRCRAFT_FACT:
        if qri in ("payload_range_analysis",):
            return "aligned"
        if qri in ("mission_feasibility", "shortlist_ranking", "acquisition_recommendation"):
            return "misroute_qri_mission"
        return "partial"
    if unified == UnifiedIntent.AIRCRAFT_MARKET_FACT:
        if qri in ("ownership_economics", "payload_range_analysis"):
            return "aligned"
        if qri in ("mission_feasibility", "shortlist_ranking", "acquisition_recommendation"):
            return "misroute_qri_mission"
        return "partial"
    if unified == UnifiedIntent.OTHER and qri in (
        "mission_feasibility",
        "aircraft_comparison",
        "shortlist_ranking",
    ):
        return "aligned"
    return "neutral"


def get_secondary_intent_promotion_contract(route: UnifiedIntentRoute) -> Dict[str, Any]:
    """
    Pure contract layer — defines what secondary intents WOULD map to
    in future execution layers (Step 2+).

    This MUST NOT influence current routing.
    """
    secondary = route.secondary_intent
    return {
        "contract_version": PROMOTION_CONTRACT_VERSION,
        "capability_promotable": secondary == UnifiedSecondaryIntent.AIRCRAFT_CAPABILITY_LIKELY,
        "comparison_promotable": secondary == UnifiedSecondaryIntent.AIRCRAFT_COMPARISON_LIKELY,
        "mission_promotable": secondary == UnifiedSecondaryIntent.AIRCRAFT_MISSION_LIKELY,
        "execution_target": (
            _SECONDARY_INTENT_EXECUTION_TARGETS.get(secondary) if secondary is not None else None
        ),
        "primary_intent_frozen": route.intent.value,
    }


def validate_unified_intent_route_invariants(route: UnifiedIntentRoute) -> Dict[str, Any]:
    """
    Structural invariant checks for UnifiedIntentRoute.

    Read-only validation — never mutates routing state or triggers execution.
    """
    violations: List[str] = []

    if route.intent in (UnifiedIntent.AIRCRAFT_FACT, UnifiedIntent.AIRCRAFT_MARKET_FACT):
        if not route.model:
            violations.append("fact_or_market_requires_resolved_model")
        if not route.field:
            violations.append("fact_or_market_requires_field")
        if route.model_confidence < _MODEL_CONFIDENCE_THRESHOLD:
            violations.append("fact_or_market_requires_model_confidence_above_threshold")

    if route.model and route.model_confidence <= 0.0:
        violations.append("resolved_model_requires_positive_confidence")

    return {
        "contract_version": PROMOTION_CONTRACT_VERSION,
        "valid": not violations,
        "violations": violations,
    }


def build_unified_intent_shadow(
    route: UnifiedIntentRoute,
    qri_intent: str,
    *,
    enforce_fact: bool = False,
    hardening_flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Build shadow-mode comparison payload for data_used logging."""
    would_enforce = (
        enforce_fact
        and route.intent in (UnifiedIntent.AIRCRAFT_FACT, UnifiedIntent.AIRCRAFT_MARKET_FACT)
        and bool(route.model)
        and bool(route.field)
    )
    secondary = route.secondary_intent.value if route.secondary_intent is not None else None
    flags = hardening_flags or {
        "routing_failure": False,
        "ambiguity_detected": False,
        "fallback_triggered": False,
    }
    return {
        "qri_intent": qri_intent,
        "unified_intent_primary": route.intent.value,
        "unified_intent_secondary": secondary,
        "model_resolved": route.model,
        "model_confidence": round(float(route.model_confidence), 4),
        "field_detected": route.field,
        "intent_confidence": round(float(route.confidence), 4),
        # Legacy keys retained for downstream readers
        "unified_intent": route.intent.value,
        "unified_model": route.model,
        "unified_field": route.field,
        "unified_confidence": round(float(route.confidence), 4),
        "unified_signals": list(route.signals),
        "intent_agreement": _intent_agreement(route.intent, qri_intent),
        "would_enforce_fact_path": would_enforce,
        "hardening_flags": flags,
    }


__all__ = [
    "PROMOTION_CONTRACT_VERSION",
    "UnifiedExecutionPath",
    "UnifiedIntent",
    "UnifiedIntentRoute",
    "UnifiedSecondaryIntent",
    "build_unified_intent_shadow",
    "classify_unified_intent",
    "get_secondary_intent_promotion_contract",
    "validate_unified_intent_route_invariants",
]
