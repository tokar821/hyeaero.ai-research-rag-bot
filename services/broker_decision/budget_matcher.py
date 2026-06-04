"""Rank acquisition opportunities for a stated budget — not a raw aircraft list."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from services.broker_reasoning.category_resolver import resolve_category
from services.comparison.aircraft_registry_lock import lock_comparison_aircraft
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES


@dataclass(frozen=True)
class BudgetOpportunity:
    rank: int
    model: str
    value_reason: str


def _tier_musd(model: str) -> float:
    from services.broker_reasoning.category_resolver import _ACQUISITION_TIER_MUSD

    if model in _ACQUISITION_TIER_MUSD:
        return _ACQUISITION_TIER_MUSD[model]
    profile = AIRCRAFT_PROFILES.get(model) or {}
    return float(profile.get("operating_index") or 0.5) * 25.0


def _value_reason(model: str, budget_musd: float) -> str:
    tier = _tier_musd(model)
    profile = AIRCRAFT_PROFILES.get(model) or {}
    category = profile.get("category", "jet")
    practical = profile.get("practical_nm")
    range_bit = f" with roughly {int(practical)} nm practical range" if practical else ""

    if tier <= budget_musd * 0.65:
        return (
            f"Leaves headroom below your ${budget_musd:.0f}M cap — strong {category}-class option{range_bit}; "
            "room for refurbishment or avionics upgrades."
        )
    if tier <= budget_musd * 0.95:
        return (
            f"Sits near the top of your ${budget_musd:.0f}M budget — credible {category}-class{range_bit}; "
            "expect mid-life examples, not entry hours."
        )
    return (
        f"At the top of a ${budget_musd:.0f}M budget — expect older hours and a disciplined "
        f"pre-buy; verify total cost of ownership before committing."
    )


def match_budget_opportunities(
    budget_musd: float,
    *,
    manufacturer: Optional[str] = None,
    query: str = "",
    max_items: int = 4,
) -> List[BudgetOpportunity]:
    """Return ranked opportunities that fit the buyer's budget envelope."""
    q = query or f"budget {budget_musd}M"
    cat = resolve_category(
        q,
        manufacturer=manufacturer,
        budget_musd=budget_musd,
        price_sensitive=False,
    )
    candidates = list(cat.candidates)
    stretch = 1.5 if re.search(r"(?is)\bsuper-?\s*midsize\b", q) else 1.15
    if not candidates:
        # Broad catalog fit by tier.
        from services.comparison.aircraft_registry_lock import CANONICAL_COMPARISON_REGISTRY

        scored = sorted(
            (abs(_tier_musd(m) - budget_musd * 0.75), m)
            for m in CANONICAL_COMPARISON_REGISTRY
        )
        candidates = [m for _, m in scored[:8]]

    from services.broker_decision.mission_fit_scorer import score_model_fit

    ranked: List[tuple[float, str]] = []
    du_stub: dict = {"broker_reasoning": {"mission": {"acquisition_budget_musd": budget_musd}}}
    for model in candidates:
        lock = lock_comparison_aircraft([model])
        if not lock.canonical:
            continue
        canon = lock.canonical[0]
        tier = _tier_musd(canon)
        if tier > budget_musd * stretch:
            continue
        fit = score_model_fit(canon, query=query, data_used=du_stub)
        ranked.append((fit, canon))

    ranked.sort(key=lambda x: (-x[0], _tier_musd(x[1])))
    seen: set[str] = set()
    opportunities: List[BudgetOpportunity] = []
    for _, model in ranked:
        key = model.lower()
        if key in seen:
            continue
        seen.add(key)
        opportunities.append(
            BudgetOpportunity(
                rank=len(opportunities) + 1,
                model=model,
                value_reason=_value_reason(model, budget_musd),
            )
        )
        if len(opportunities) >= max_items:
            break

    return opportunities


__all__ = ["BudgetOpportunity", "match_budget_opportunities"]
