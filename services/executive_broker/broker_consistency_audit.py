"""Audit recommendation drift vs conversation memory and stated constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from services.client_context.broker_context_builder import BrokerConversationContext
from services.client_context.recommendation_consistency import (
    _manufacturer_of,
    _tier_musd,
    filter_models_for_consistency,
)


@dataclass
class BrokerConsistencyScore:
    overall: float
    budget_drift: bool = False
    mission_drift: bool = False
    aircraft_drift: bool = False
    conversation_drift: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "budget_drift": self.budget_drift,
            "mission_drift": self.mission_drift,
            "aircraft_drift": self.aircraft_drift,
            "conversation_drift": self.conversation_drift,
            "notes": list(self.notes),
        }


def _ctx_from_data_used(data_used: Dict[str, Any]) -> BrokerConversationContext:
    raw = (
        data_used.get("broker_conversation_context")
        or data_used.get("client_context")
        or {}
    )
    return BrokerConversationContext.from_dict(raw if isinstance(raw, dict) else {})


def audit_broker_consistency(
    *,
    primary: str,
    alternatives: Sequence[str],
    data_used: Dict[str, Any],
    query: str = "",
) -> BrokerConsistencyScore:
    """
    Score how well the executive pick aligns with remembered budget, mission, and thread.
    """
    ctx = _ctx_from_data_used(data_used)
    notes: List[str] = []
    penalties = 0

    budget = ctx.remembered_budget_musd
    tier = _tier_musd(primary)

    budget_drift = False
    if budget is not None and tier > budget * 1.2:
        budget_drift = True
        penalties += 2
        notes.append(
            f"{primary} sits above the ${budget:.0f}M budget discussed in this thread."
        )

    mission_drift = False
    br = data_used.get("broker_reasoning") or {}
    mission = br.get("mission") if isinstance(br, dict) else {}
    if isinstance(mission, dict):
        mission_budget = mission.get("acquisition_budget_musd")
        if mission_budget is not None and budget is not None:
            if abs(float(mission_budget) - float(budget)) > 2.0:
                mission_drift = True
                penalties += 1
                notes.append("Mission budget inference diverges from remembered client budget.")

    aircraft_drift = False
    if ctx.remembered_targets and primary not in ctx.remembered_targets:
        if not any(primary.split()[-1] in t for t in ctx.remembered_targets):
            # New model not in thread — only drift if user was focused elsewhere
            if ctx.stage in ("ACTIVE_SHOPPING", "NEGOTIATING", "DUE_DILIGENCE"):
                aircraft_drift = True
                penalties += 1
                notes.append(
                    f"Thread has been centered on {ctx.remembered_targets[0]}; "
                    f"{primary} is a shift."
                )

    conversation_drift = False
    prior = data_used.get("executive_recommendation_prior")
    if isinstance(prior, dict) and prior.get("primary_recommendation"):
        prev = str(prior["primary_recommendation"])
        if prev != primary and prev not in alternatives:
            conversation_drift = True
            penalties += 1
            notes.append(f"Prior turn primary was {prev}; new pick is {primary}.")

    mfrs = {m.lower() for m in ctx.preferred_manufacturers}
    if mfrs:
        mfr = (_manufacturer_of(primary) or "").lower()
        if mfr and mfr not in mfrs:
            alt_mfrs = {_manufacturer_of(a) or "" for a in alternatives}
            if not any((x or "").lower() in mfrs for x in alt_mfrs):
                notes.append(f"Primary {primary} is outside stated manufacturer preference.")

    # Re-check alternatives against consistency filter
    combined = [primary, *alternatives]
    filtered = filter_models_for_consistency(combined, ctx)
    if primary not in filtered:
        budget_drift = True
        penalties += 2
        notes.append(f"{primary} fails client-context consistency filter.")

    overall = max(0.0, min(1.0, 1.0 - penalties * 0.2))

    return BrokerConsistencyScore(
        overall=overall,
        budget_drift=budget_drift,
        mission_drift=mission_drift,
        aircraft_drift=aircraft_drift,
        conversation_drift=conversation_drift,
        notes=notes,
    )


__all__ = ["BrokerConsistencyScore", "audit_broker_consistency"]
