"""Map consultant fine intent → Pinecone ``entity_type`` / ``doc_type`` filters (intent-shaped retrieval)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_AIRCRAFT_MODEL_TYPES: List[str] = [
    "phlydata_aircraft",
    "aircraft",
    "aviacost_aircraft_detail",
    "aircraftpost_fleet_aircraft",
]
_MARKET_TYPES: List[str] = ["aircraft_listing", "aircraft_sale"]
_REGISTRY_TYPES: List[str] = ["faa_registration", "phlydata_aircraft"]


def _intent_filter(entity_types: List[str], doc_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Match vectors that have either a known ``entity_type`` **or** a logical ``doc_type``.

    Re-ingested chunks get ``doc_type``; legacy chunks may only have ``entity_type``.
    """
    et = list(dict.fromkeys(entity_types))
    if not et and not doc_types:
        return {}
    if doc_types:
        dt = list(dict.fromkeys(doc_types))
        return {"$or": [{"entity_type": {"$in": et}}, {"doc_type": {"$in": dt}}]}
    return {"entity_type": {"$in": et}}


def pinecone_filter_for_fine_intent(intent_value: str) -> Optional[Dict[str, Any]]:
    """Return a Pinecone metadata filter or ``None`` for unrestricted search."""
    iv = (intent_value or "").strip().lower()

    if iv == "ownership_lookup":
        return _intent_filter(list(_REGISTRY_TYPES), ["registry"])
    if iv == "market_question":
        return _intent_filter(
            _MARKET_TYPES + ["phlydata_aircraft"],
            ["aircraft_listing", "market_data", "aircraft_model"],
        )
    if iv in ("aircraft_specs", "aircraft_comparison"):
        return _intent_filter(list(_AIRCRAFT_MODEL_TYPES), ["aircraft_model"])
    if iv in ("aviation_mission", "aircraft_recommendation"):
        return _intent_filter(
            list(_AIRCRAFT_MODEL_TYPES) + ["document"],
            ["aircraft_model", "document"],
        )
    if iv == "general_question":
        return _intent_filter(
            ["document", "aviacost_aircraft_detail", "aircraft"],
            ["document", "aircraft_model"],
        )
    return None
