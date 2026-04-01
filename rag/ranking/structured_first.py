"""Post-retrieval ordering: structured entities before long-form document chunks."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from rag.intent.schemas import ConsultantIntent

logger = logging.getLogger(__name__)


def _tier_for_intent(entity_type: str, intent: ConsultantIntent) -> int:
    et = (entity_type or "").strip() or "other"
    if intent == ConsultantIntent.MARKET_PRICING:
        order = {
            "phlydata_aircraft": 0,
            "aircraft": 1,
            "aircraft_listing": 2,
            "aircraft_sale": 3,
            "faa_registration": 4,
            "aviacost_aircraft_detail": 5,
            "aircraftpost_fleet_aircraft": 5,
            "document": 8,
        }
    elif intent == ConsultantIntent.TECHNICAL_SPEC:
        order = {
            "document": 0,
            "phlydata_aircraft": 1,
            "aircraft": 2,
            "aircraft_listing": 6,
            "aircraft_sale": 6,
            "faa_registration": 5,
            "aviacost_aircraft_detail": 4,
            "aircraftpost_fleet_aircraft": 4,
        }
    elif intent == ConsultantIntent.REGISTRATION_LOOKUP:
        order = {
            "faa_registration": 0,
            "phlydata_aircraft": 1,
            "aircraft": 2,
            "aircraft_listing": 5,
            "aircraft_sale": 5,
            "document": 7,
            "aviacost_aircraft_detail": 6,
            "aircraftpost_fleet_aircraft": 6,
        }
    else:
        order = {
            "phlydata_aircraft": 0,
            "aircraft": 1,
            "faa_registration": 2,
            "aircraft_listing": 3,
            "aircraft_sale": 4,
            "aviacost_aircraft_detail": 5,
            "aircraftpost_fleet_aircraft": 5,
            "document": 9,
        }
    return order.get(et, 15)


def apply_structured_first_rag_order(
    results: List[Dict[str, Any]],
    intent: ConsultantIntent,
) -> List[Dict[str, Any]]:
    """Re-order hydrated RAG rows by intent; structured sources precede generic docs."""
    if not results:
        return results

    def sort_key(r: Dict[str, Any]) -> tuple:
        et = str(r.get("entity_type") or "")
        tier = _tier_for_intent(et, intent)
        rs = r.get("rerank_score")
        if rs is None:
            rs = r.get("score")
        try:
            sc = float(rs) if rs is not None else 0.0
        except (TypeError, ValueError):
            sc = 0.0
        return (tier, -sc)

    out = sorted(results, key=sort_key)
    if out != results:
        logger.debug(
            "consultant retrieval: applied structured-first order for intent=%s",
            intent.value,
        )
    return out
