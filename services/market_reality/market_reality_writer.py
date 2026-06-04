"""Transform listing intelligence into transaction-advisor prose."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.market_intelligence.market_intelligence_engine import fmt_musd
from services.market_reality.buyer_leverage_analyzer import BuyerLeverage
from services.market_reality.listing_confidence_analyzer import ListingPriceConfidence
from services.market_reality.listing_detector import ListingMode, ListingSignal


def _diligence_block() -> str:
    return (
        "Before treating it as a bargain, I would verify:\n"
        "• year and total time\n"
        "• engine program status\n"
        "• damage history\n"
        "• the listing link or broker package"
    )


def write_listing_discussion(
    signal: ListingSignal,
    *,
    price_analysis: Dict[str, Any],
    inventory: Dict[str, Any],
    leverage: Dict[str, Any],
    band_mid_usd: Optional[float] = None,
) -> str:
    model = signal.model or "this aircraft"
    ask = signal.ask_musd
    conf = price_analysis.get("confidence")
    lines: List[str] = []

    if ask is not None:
        ask_line = f"${ask:.1f}M"
    else:
        ask_line = "that ask"

    if conf == ListingPriceConfidence.UNUSUALLY_CHEAP.value:
        lines.append(
            f"That is materially below where most {model} transactions occur"
            + (f" (your {ask_line} vs market near {fmt_musd(band_mid_usd)})." if band_mid_usd else f" ({ask_line}).")
        )
        lines.append("")
        lines.append(_diligence_block())
        lines.append("")
        lines.append(
            "A pricing gap that large is unusual — it can be real, but it is more often "
            "hours, maintenance, or a mis-listed model year."
        )
    elif conf == ListingPriceConfidence.UNUSUALLY_EXPENSIVE.value:
        lines.append(
            f"At {ask_line}, that {model} ask looks high versus current listing-derived medians"
            + (f" near {fmt_musd(band_mid_usd)}." if band_mid_usd else ".")
        )
        lines.append("I would negotiate against recent comps or walk.")
    elif conf == ListingPriceConfidence.POTENTIAL_DATA_ERROR.value:
        lines.append(
            f"The {ask_line} figure does not line up with the listing band I have for {model}."
        )
        lines.append("Confirm the model year and that the ask includes the right equipment before diligence.")
    else:
        lines.append(
            f"At {ask_line}, a {model} can be plausible depending on year and program status"
            + (f" — market center is near {fmt_musd(band_mid_usd)}." if band_mid_usd else ".")
        )
        lines.append(_diligence_block())

    lev = leverage.get("leverage")
    if lev == BuyerLeverage.BUYER_FRIENDLY.value:
        lines.append(f"\nLeverage: {leverage.get('summary', '')}")
    elif lev == BuyerLeverage.SELLER_FRIENDLY.value:
        lines.append(f"\nLeverage: {leverage.get('summary', '')}")

    inv_note = inventory.get("note")
    if inv_note:
        lines.append(f"\nInventory: {inv_note}")

    return "\n".join(lines).strip()


def write_market_timing(
    signal: ListingSignal,
    *,
    leverage: Dict[str, Any],
    inventory: Dict[str, Any],
    temporal_note: Optional[str] = None,
) -> str:
    model = signal.model or "this model"
    lines = [
        f"On timing for a {model}, I would not call the cycle from one listing.",
    ]
    if temporal_note:
        lines.append(temporal_note)
    lines.append(leverage.get("summary", ""))
    lines.append(inventory.get("note", ""))
    lines.append(
        "\nBuy when you have a vetted tail — not when a headline says prices are moving."
    )
    return "\n".join(lines).strip()


def write_buyer_seller_market(
    *,
    leverage: Dict[str, Any],
    inventory: Dict[str, Any],
    model: Optional[str] = None,
) -> str:
    model_bit = model or "this segment"
    lev = leverage.get("leverage")
    if lev == BuyerLeverage.BUYER_FRIENDLY.value:
        headline = f"For {model_bit}, this looks like a buyer-friendlier window."
    elif lev == BuyerLeverage.SELLER_FRIENDLY.value:
        headline = f"For {model_bit}, sellers still have leverage in the current inventory picture."
    else:
        headline = f"For {model_bit}, the market looks balanced — tail condition will drive the deal."

    return f"{headline}\n\n{inventory.get('note', '')}\n\n{leverage.get('summary', '')}".strip()


def write_why_so_cheap(
    signal: ListingSignal,
    *,
    price_analysis: Dict[str, Any],
) -> str:
    model = signal.model or "this aircraft"
    reasons = [
        f"When a {model} looks 'too cheap,' it is usually one of:",
        "• mis-stated year, hours, or engine status",
        "• deferred maintenance coming due",
        "• damage or incident history",
        "• a motivated seller — less often a free lunch",
    ]
    conf = price_analysis.get("confidence")
    if conf == ListingPriceConfidence.UNUSUALLY_CHEAP.value:
        reasons.insert(0, f"Your price point is genuinely low vs market ({price_analysis.get('reason', '')}).")
    return "\n".join(reasons).strip()


def write_market_reality(
    signal: ListingSignal,
    *,
    price_analysis: Dict[str, Any],
    inventory: Dict[str, Any],
    leverage: Dict[str, Any],
    band_mid_usd: Optional[float] = None,
    temporal_note: Optional[str] = None,
    tail_brief: Optional[str] = None,
) -> str:
    if signal.mode == ListingMode.TAIL_INVESTIGATION and tail_brief:
        return tail_brief

    if signal.mode in (
        ListingMode.LISTING_DISCUSSION,
        ListingMode.LISTING_REALISM,
    ):
        return write_listing_discussion(
            signal,
            price_analysis=price_analysis,
            inventory=inventory,
            leverage=leverage,
            band_mid_usd=band_mid_usd,
        )

    if signal.mode == ListingMode.WHY_SO_CHEAP:
        return write_why_so_cheap(signal, price_analysis=price_analysis)

    if signal.mode == ListingMode.MARKET_TIMING:
        return write_market_timing(
            signal,
            leverage=leverage,
            inventory=inventory,
            temporal_note=temporal_note,
        )

    if signal.mode == ListingMode.BUYER_SELLER_MARKET:
        return write_buyer_seller_market(
            leverage=leverage,
            inventory=inventory,
            model=signal.model,
        )

    return ""


__all__ = [
    "write_buyer_seller_market",
    "write_listing_discussion",
    "write_market_reality",
    "write_market_timing",
    "write_why_so_cheap",
]
