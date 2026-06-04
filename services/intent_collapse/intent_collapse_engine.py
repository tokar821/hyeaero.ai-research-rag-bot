"""
Phase 45 — unified intent collapse (pre-reasoning canonicalization).

Single interpretation pass before broker reasoning, decision, market, or executive layers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.broker_reasoning.comparison_soft_resolution import soft_resolve_comparison
from services.broker_reasoning.intent_expander import _detect_manufacturer, _detect_reference_model
from services.broker_reasoning.mission_interpreter import interpret_mission
from services.intent_collapse.ambiguity_resolver import resolve_ambiguity
from services.intent_collapse.canonical_intent_frame import (
    AircraftScopeFrame,
    AircraftScopeType,
    BudgetFrame,
    CanonicalIntentFrame,
    PrimaryIntent,
)
from services.intent_collapse.mission_frame_builder import build_mission_frame, mission_budget_musd


_COMPARE_RE = re.compile(r"\b(?:vs\.?|versus)\b", re.I)
_BUY_RE = re.compile(
    r"(?is)\b(?:buy|purchase|acquire|get|find|looking\s+for|what\s+should\s+i\s+buy|"
    r"what\s+can\s+i\s+buy|best\s+jet|cheap|cheapest)\b",
)
_VALUATION_RE = re.compile(
    r"(?is)\b(?:worth|valuation|good\s+deal|fair\s+price|overpay|realistic|apprais)\b",
)
_LISTING_RE = re.compile(r"(?is)\b(?:saw|found|listing\s+says?|i\s+found|i\s+saw)\b")


def _classify_primary_intent(query: str, adversarial: Dict[str, Any]) -> str:
    q = (query or "").strip()
    resolved_intent = str(adversarial.get("resolved_intent") or "").upper()
    if resolved_intent == "COMPARE" or _COMPARE_RE.search(q):
        return PrimaryIntent.COMPARE.value
    if _VALUATION_RE.search(q) or resolved_intent in ("VALUATION", "DEAL_QUALITY"):
        return PrimaryIntent.VALUATION.value
    if _BUY_RE.search(q) or _LISTING_RE.search(q) or resolved_intent in ("BUY", "ACQUISITION"):
        return PrimaryIntent.BUY.value
    if re.search(r"(?is)\b(?:discover|explore|options|what\s+is\s+a)\b", q):
        return PrimaryIntent.DISCOVERY.value
    return PrimaryIntent.DISCOVERY.value


def _budget_frame(
    query: str,
    adversarial: Dict[str, Any],
) -> BudgetFrame:
    """Budget from query + adversarial only — client memory does not override."""
    interp = interpret_mission(query or "")
    cap_m = interp.acquisition_budget_musd

    adv_budget = adversarial.get("budget_state") if isinstance(adversarial.get("budget_state"), dict) else {}
    if cap_m is None and adv_budget:
        cap_m = adv_budget.get("primary_cap_musd") or adv_budget.get("acquisition_cap_musd")
        if cap_m is not None:
            try:
                cap_m = float(cap_m)
            except (TypeError, ValueError):
                cap_m = None

    if cap_m is None:
        caps = adv_budget.get("budget_caps_musd") or []
        if caps:
            try:
                cap_m = float(caps[0])
            except (TypeError, ValueError):
                pass

    tier = "UNKNOWN"
    unknown = cap_m is None
    if re.search(r"(?is)\b(?:cheap|cheapest|affordable|lowest)\b", query or ""):
        tier = "LOW"
    elif cap_m is not None:
        if cap_m <= 8:
            tier = "LOW"
        elif cap_m <= 25:
            tier = "NORMAL"
        else:
            tier = "HIGH"
        unknown = False

    return BudgetFrame(
        cap_musd=cap_m,
        max_musd=cap_m,
        tier_hint=tier,
        unknown=unknown,
    )


def _aircraft_scope(
    query: str,
    *,
    primary_intent: str,
    budget: BudgetFrame,
    resolved_models: List[str],
) -> AircraftScopeFrame:
    q = (query or "").strip()
    mfr = _detect_manufacturer(q)
    ref = _detect_reference_model(q)
    if not ref:
        # Plural shorthand (e.g. "G650s") — canonicalize without changing engines.
        if re.search(r"(?is)\bg650s?\b", q):
            ref = "Gulfstream G650"
        elif re.search(r"(?is)\bg700s?\b", q):
            ref = "Gulfstream G700"
        elif re.search(r"(?is)\bg280s?\b", q):
            ref = "Gulfstream G280"
    cheap_gulf = bool(
        re.search(r"(?is)\b(?:cheap|cheapest|affordable)\b", q)
        and mfr == "Gulfstream"
    )

    if cheap_gulf or (
        mfr == "Gulfstream"
        and budget.tier_hint == "LOW"
        and re.search(r"(?is)\b(?:cheap|budget|affordable)\b", q)
    ):
        return AircraftScopeFrame(
            scope_type=AircraftScopeType.ENTRY_LEVEL_GULFSTREAM_SCOPE.value,
            manufacturer="Gulfstream",
            models=["Gulfstream G280"],
            price_sensitive=True,
            entry_level_only=True,
        )

    if primary_intent == PrimaryIntent.COMPARE.value and len(resolved_models) >= 2:
        return AircraftScopeFrame(
            scope_type=AircraftScopeType.COMPARISON_PAIR.value,
            manufacturer=mfr,
            models=resolved_models[:2],
            price_sensitive=budget.tier_hint == "LOW",
        )

    if resolved_models:
        return AircraftScopeFrame(
            scope_type=AircraftScopeType.EXPLICIT_MODELS.value,
            manufacturer=mfr,
            models=list(resolved_models),
            price_sensitive=budget.tier_hint == "LOW",
        )

    if ref:
        return AircraftScopeFrame(
            scope_type=AircraftScopeType.EXPLICIT_MODELS.value,
            manufacturer=mfr,
            models=[ref],
            price_sensitive=budget.tier_hint == "LOW",
        )

    if mfr:
        return AircraftScopeFrame(
            scope_type=AircraftScopeType.MANUFACTURER_FAMILY.value,
            manufacturer=mfr,
            models=[],
            price_sensitive=budget.tier_hint == "LOW" or bool(re.search(r"(?is)\bcheap\b", q)),
        )

    return AircraftScopeFrame(scope_type=AircraftScopeType.OPEN.value)


def collapse_intent(
    raw_query: str,
    *,
    normalized_query: Optional[str] = None,
    client_context: Optional[Dict[str, Any]] = None,
    adversarial: Optional[Dict[str, Any]] = None,
) -> CanonicalIntentFrame:
    """Build canonical intent frame from query and preprocess metadata."""
    raw = (raw_query or "").strip()
    norm = (normalized_query or raw).strip()
    adv = adversarial if isinstance(adversarial, dict) else {}
    ctx = client_context if isinstance(client_context, dict) else {}

    primary = _classify_primary_intent(norm, adv)
    mission = build_mission_frame(norm, client_context=ctx)
    budget = _budget_frame(norm, adv)

    mb = mission_budget_musd(mission, norm)
    if mb is not None:
        budget.cap_musd = mb
        budget.max_musd = mb
        budget.unknown = False
        if budget.tier_hint == "UNKNOWN":
            budget.tier_hint = "LOW" if mb <= 8 else ("NORMAL" if mb <= 25 else "HIGH")

    amb = resolve_ambiguity(
        norm,
        primary_intent=primary,
        adversarial=adv,
        budget_cap_musd=budget.cap_musd,
    )

    scope = _aircraft_scope(
        norm,
        primary_intent=primary,
        budget=budget,
        resolved_models=amb.resolved_models,
    )

    confidence = 0.92 - amb.confidence_penalty
    if amb.flags:
        confidence -= min(0.35, 0.04 * len(amb.flags))
    confidence = max(0.25, min(0.99, confidence))

    if amb.clarification_request and "COMPARISON_AMBIGUOUS" in amb.flags:
        confidence = min(confidence, 0.45)

    return CanonicalIntentFrame(
        primary_intent=primary,
        mission=mission,
        budget=budget,
        aircraft_scope=scope,
        confidence=round(confidence, 3),
        ambiguity_flags=list(amb.flags),
        clarification_request=amb.clarification_request,
        normalized_query=norm,
        raw_query=raw,
    )


def apply_intent_collapse(
    raw_query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    normalized_query: Optional[str] = None,
) -> CanonicalIntentFrame:
    """
    Collapse raw query into canonical frame and stamp ``data_used``.

    Client context is read for mission stage hints only — it cannot override
    budget or primary intent classification.
    """
    du = data_used if isinstance(data_used, dict) else {}
    adv = du.get("adversarial") if isinstance(du.get("adversarial"), dict) else {}
    ctx = (
        du.get("client_context")
        or du.get("broker_conversation_context")
        or {}
    )

    norm = normalized_query or adv.get("normalized_query") or raw_query
    frame = collapse_intent(
        raw_query,
        normalized_query=norm,
        client_context=ctx if isinstance(ctx, dict) else None,
        adversarial=adv,
    )

    du["canonical_intent_frame"] = frame.to_dict()
    du["intent_collapse_applied"] = 1

    if frame.clarification_request:
        du["intent_clarification_required"] = frame.clarification_request

    return frame


__all__ = ["apply_intent_collapse", "collapse_intent"]
