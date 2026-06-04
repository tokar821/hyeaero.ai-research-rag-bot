"""Prevent recommendation drift across turns — budget and preference aware."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from services.client_context.broker_context_builder import BrokerConversationContext


def _tier_musd(model: str) -> float:
    from services.broker_reasoning.category_resolver import _ACQUISITION_TIER_MUSD

    return float(_ACQUISITION_TIER_MUSD.get(model, 30.0))


def _manufacturer_of(model: str) -> Optional[str]:
    low = model.lower()
    if "gulfstream" in low:
        return "Gulfstream"
    if "falcon" in low or "dassault" in low:
        return "Dassault"
    if "citation" in low or "latitude" in low or "longitude" in low:
        return "Cessna"
    if "challenger" in low or "global" in low or "learjet" in low:
        return "Bombardier"
    if "phenom" in low or "praetor" in low or "legacy" in low:
        return "Embraer"
    return None


def filter_models_for_consistency(
    models: Sequence[str],
    ctx: BrokerConversationContext,
    *,
    allow_stretch: bool = False,
    pinned_models: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Drop models that violate remembered budget or manufacturer focus.

    Ultra-long jets are removed when budget is set unless user explicitly mentioned them this turn.
    """
    budget = ctx.remembered_budget_musd
    mfrs = {m.lower() for m in ctx.preferred_manufacturers}
    pinned = {str(m).strip() for m in (pinned_models or ()) if str(m).strip()}
    out: List[str] = []

    for model in models:
        m = str(model).strip()
        if not m:
            continue
        tier = _tier_musd(m)

        if budget is not None and not allow_stretch:
            if tier > budget * 1.2:
                # G700 at 12M budget — block unless pinned this turn or remembered target
                if m not in pinned and m not in ctx.remembered_targets[:2]:
                    continue

        if mfrs:
            mfr = _manufacturer_of(m)
            if mfr and mfr.lower() not in mfrs:
                # Allow if model is in explicit remembered targets
                if m not in ctx.remembered_targets:
                    continue

        out.append(m)

    return out


def apply_consistency_to_broker_reasoning(
    data_used: Dict[str, Any],
    ctx: BrokerConversationContext,
) -> None:
    """Filter broker_reasoning category/alternatives in-place (metadata only)."""
    if data_used.get("intent_collapse_applied"):
        return
    br = data_used.get("broker_reasoning")
    if not isinstance(br, dict):
        return

    cat = br.get("category")
    if isinstance(cat, dict) and cat.get("candidates"):
        filtered = filter_models_for_consistency(cat["candidates"], ctx)
        if filtered:
            cat["candidates"] = filtered
            cat["consistency_filtered"] = True

    alts = br.get("alternatives")
    if isinstance(alts, dict) and alts.get("candidates"):
        filtered = filter_models_for_consistency(alts["candidates"], ctx)
        if filtered:
            alts["candidates"] = filtered

    if ctx.remembered_budget_musd is not None:
        mission = br.get("mission")
        if isinstance(mission, dict) and mission.get("acquisition_budget_musd") is None:
            mission["acquisition_budget_musd"] = ctx.remembered_budget_musd
            mission["from_client_context"] = True


def apply_consistency_to_broker_decision(
    decision_dict: Dict[str, Any],
    ctx: BrokerConversationContext,
) -> Dict[str, Any]:
    """Filter decision alternatives for consistency."""
    alts = decision_dict.get("alternatives")
    if not isinstance(alts, list):
        return decision_dict

    models = [a.get("model") for a in alts if isinstance(a, dict) and a.get("model")]
    filtered_names = filter_models_for_consistency(models, ctx)
    if filtered_names:
        decision_dict["alternatives"] = [
            a for a in alts if isinstance(a, dict) and a.get("model") in filtered_names
        ]
    return decision_dict


__all__ = [
    "apply_consistency_to_broker_decision",
    "apply_consistency_to_broker_reasoning",
    "filter_models_for_consistency",
]
