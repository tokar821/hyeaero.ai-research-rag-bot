"""Normalize and classify budget signals — acquisition cap vs listing ask vs vague mention."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

_ULTRA_PREMIUM_RE = __import__("re").compile(
    r"\b(?:g\s*700|g700|g\s*650|g650|global\s*7500|falcon\s*8x|global\s*6500)\b",
    __import__("re").I,
)

_BUY_PRICE_SHAPE_RE = re.compile(
    r"(?is)\b(?:"
    r"good\s+deal|fair\s+price|overpriced|good\s+buy|listed\s+at|"
    r"should\s+i\s+buy|fair\s+deal|for\s+\$"
    r")\b",
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_ACQUISITION_BUDGET_RE = re.compile(
    r"(?is)\b(?:under|below|within|budget|max(?:imum)?|less\s+than)\s*\$?\s*"
    r"(\d+(?:\.\d+)?)\s*(m|mm|million|mil|k)?\b",
)
_ASK_FOR_RE = re.compile(
    r"(?is)\b(?:for|at)\s+\$?\s*(\d+(?:\.\d+)?)\s*(m|mm|million|mil|k)?\b",
)
_MONEY_ALL_RE = re.compile(
    r"(?is)\$?\s*(\d+(?:\.\d+)?)\s*(m|mm|million|mil|k)?\b",
)


class BudgetFeasibility(str, Enum):
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    SEMANTICALLY_CONFLICTED = "SEMANTICALLY_CONFLICTED"


class PriceSignalKind(str, Enum):
    ACQUISITION_BUDGET = "ACQUISITION_BUDGET"
    LISTING_ASK = "LISTING_ASK"
    VAGUE_MENTION = "VAGUE_MENTION"


@dataclass(frozen=True)
class PriceSignal:
    kind: PriceSignalKind
    amount_musd: float
    source_span: str = ""


@dataclass(frozen=True)
class BudgetConflictState:
    feasibility: BudgetFeasibility
    budget_caps_musd: Tuple[float, ...]
    primary_cap_musd: Optional[float]
    acquisition_cap_musd: Optional[float] = None
    listing_ask_musd: Optional[float] = None
    price_signals: Tuple[PriceSignal, ...] = field(default_factory=tuple)
    conflicting_models: Tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""


def _to_musd(amount: str, unit: str) -> Optional[float]:
    try:
        val = float(amount)
    except ValueError:
        return None
    u = (unit or "").lower()
    if u == "k":
        return val / 1000.0
    if u in ("m", "mm", "million", "mil", "") and val < 1000:
        return val
    if val >= 1000:
        return val / 1_000_000.0
    return val


def classify_price_signals(query: str) -> List[PriceSignal]:
    """
    Separate acquisition budget caps from listing asks and vague price mentions.

    Listing ask: year + aircraft + ``for/at $XM`` + buy-decision shape.
    Acquisition: ``under/below/budget $XM`` without buy-price-only shape.
    Vague: bare ``$XM`` without budget or ask anchors.
    """
    q = query or ""
    signals: List[PriceSignal] = []
    buy_shape = bool(_BUY_PRICE_SHAPE_RE.search(q) and _YEAR_RE.search(q))

    for m in _ACQUISITION_BUDGET_RE.finditer(q):
        amt = _to_musd(m.group(1), m.group(2) or "")
        if amt is None:
            continue
        if buy_shape and _ASK_FOR_RE.search(q):
            continue
        signals.append(
            PriceSignal(
                kind=PriceSignalKind.ACQUISITION_BUDGET,
                amount_musd=amt,
                source_span=m.group(0)[:40],
            )
        )

    for m in _ASK_FOR_RE.finditer(q):
        amt = _to_musd(m.group(1), m.group(2) or "")
        if amt is None:
            continue
        if buy_shape:
            signals.append(
                PriceSignal(
                    kind=PriceSignalKind.LISTING_ASK,
                    amount_musd=amt,
                    source_span=m.group(0)[:40],
                )
            )

    if not signals:
        for m in _MONEY_ALL_RE.finditer(q):
            amt = _to_musd(m.group(1), m.group(2) or "")
            if amt is not None:
                signals.append(
                    PriceSignal(
                        kind=PriceSignalKind.VAGUE_MENTION,
                        amount_musd=amt,
                        source_span=m.group(0)[:40],
                    )
                )
                break

    return signals


def normalize_budget_conflicts(
    query: str,
    *,
    resolved_models: Optional[List[str]] = None,
) -> BudgetConflictState:
    """Classify budget feasibility using acquisition caps only for infeasibility rules."""
    q = query or ""
    signals = classify_price_signals(q)
    acquisition = [s.amount_musd for s in signals if s.kind == PriceSignalKind.ACQUISITION_BUDGET]
    listing_ask = [s.amount_musd for s in signals if s.kind == PriceSignalKind.LISTING_ASK]
    vague = [s.amount_musd for s in signals if s.kind == PriceSignalKind.VAGUE_MENTION]

    all_caps = [s.amount_musd for s in signals]
    primary_acq = acquisition[0] if acquisition else None
    primary_ask = listing_ask[0] if listing_ask else None

    models = tuple(resolved_models or [])

    if primary_acq is not None and primary_acq <= 8.0 and _ULTRA_PREMIUM_RE.search(q):
        return BudgetConflictState(
            feasibility=BudgetFeasibility.INFEASIBLE,
            budget_caps_musd=tuple(all_caps),
            primary_cap_musd=primary_acq,
            acquisition_cap_musd=primary_acq,
            listing_ask_musd=primary_ask,
            price_signals=tuple(signals),
            conflicting_models=models if models else ("ultra_premium_mention",),
            reason=f"acquisition budget ${primary_acq}M incompatible with ultra-premium model class",
        )

    if (
        primary_acq is not None
        and primary_ask is not None
        and abs(primary_acq - primary_ask) / max(primary_acq, 0.1) > 0.35
        and re.search(r"(?is)\b(?:worth|valuation)\b", q)
    ):
        return BudgetConflictState(
            feasibility=BudgetFeasibility.SEMANTICALLY_CONFLICTED,
            budget_caps_musd=tuple(all_caps),
            primary_cap_musd=primary_acq,
            acquisition_cap_musd=primary_acq,
            listing_ask_musd=primary_ask,
            price_signals=tuple(signals),
            conflicting_models=models,
            reason="valuation ask differs materially from stated acquisition budget",
        )

    if re.search(r"(?is)\b(?:like|similar)\b.*\b(?:under|below)\b", q) and len(acquisition) >= 1 and len(vague) >= 1:
        return BudgetConflictState(
            feasibility=BudgetFeasibility.SEMANTICALLY_CONFLICTED,
            budget_caps_musd=tuple(all_caps),
            primary_cap_musd=primary_acq,
            acquisition_cap_musd=primary_acq,
            listing_ask_musd=primary_ask,
            price_signals=tuple(signals),
            conflicting_models=models,
            reason="cross-class budget comparison language with mixed price signals",
        )

    if len(acquisition) >= 2 and max(acquisition) / max(min(acquisition), 0.1) > 2.5:
        return BudgetConflictState(
            feasibility=BudgetFeasibility.SEMANTICALLY_CONFLICTED,
            budget_caps_musd=tuple(all_caps),
            primary_cap_musd=primary_acq,
            acquisition_cap_musd=primary_acq,
            listing_ask_musd=primary_ask,
            price_signals=tuple(signals),
            conflicting_models=models,
            reason="conflicting acquisition budget constraints in same query",
        )

    return BudgetConflictState(
        feasibility=BudgetFeasibility.FEASIBLE,
        budget_caps_musd=tuple(all_caps),
        primary_cap_musd=primary_acq or primary_ask or (vague[0] if vague else None),
        acquisition_cap_musd=primary_acq,
        listing_ask_musd=primary_ask,
        price_signals=tuple(signals),
        conflicting_models=(),
        reason="budget signals consistent",
    )
