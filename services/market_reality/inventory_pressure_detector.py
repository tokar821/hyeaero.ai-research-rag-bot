"""Inventory pressure from existing listing snapshot counts — no formula changes."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict

from services.market_intelligence.listing_analytics import MarketSnapshot


class InventoryPressure(str, Enum):
    LOW_INVENTORY = "LOW_INVENTORY"
    NORMAL = "NORMAL"
    OVERSUPPLY = "OVERSUPPLY"


def detect_inventory_pressure(snapshot: MarketSnapshot) -> Dict[str, Any]:
    """
    Classify supply pressure from active listing count only.

    Thresholds are interpretive labels on existing snapshot data.
    """
    n = int(snapshot.active_listing_count or 0)
    if n >= 18:
        pressure = InventoryPressure.OVERSUPPLY
        note = "Healthy listing depth — buyers usually have alternatives in-band."
    elif n >= 6:
        pressure = InventoryPressure.NORMAL
        note = "Normal inventory — pricing tends to be negotiable but not distressed."
    else:
        pressure = InventoryPressure.LOW_INVENTORY
        note = "Limited inventory — fewer comps and less room to negotiate on specific tails."

    return {
        "pressure": pressure.value,
        "active_listings": n,
        "note": note,
    }


__all__ = ["InventoryPressure", "detect_inventory_pressure"]
