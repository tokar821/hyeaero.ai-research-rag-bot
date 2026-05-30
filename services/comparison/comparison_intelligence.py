"""
Comparison intelligence — operational tradeoffs beyond range/seats/speed.
"""

from __future__ import annotations

from typing import Any, Dict, List

_MAINTENANCE_ECOSYSTEM: Dict[str, str] = {
    "Gulfstream G650ER": "mature_global_support",
    "Gulfstream G650": "mature_global_support",
    "Global 7500": "strong_oem_program",
    "Global 6500": "strong_oem_program",
    "Falcon 8X": "trijet_specialist_network",
    "Falcon 7X": "trijet_specialist_network",
    "Challenger 650": "large_cabin_mature",
    "Praetor 600": "embraer_efficient_support",
    "Gulfstream G280": "super_mid_efficient",
}

_PILOT_WORKLOAD: Dict[str, str] = {
    "Falcon 8X": "trijet_crew_complexity",
    "Falcon 7X": "trijet_crew_complexity",
    "Global 7500": "ulr_two_pilot_standard",
    "Gulfstream G650ER": "ulr_two_pilot_standard",
}

_AIRPORT_FLEX: Dict[str, str] = {
    "Praetor 600": "strong_field_flex",
    "Gulfstream G280": "good_field_flex",
    "Challenger 650": "moderate_field",
    "Falcon 8X": "moderate_field",
    "Global 7500": "ulr_runway_bias",
}

_DISPATCH_MATURITY: Dict[str, str] = {
    "Global 7500": "flagship_dispatch_maturity",
    "Gulfstream G650ER": "proven_ulr_dispatch",
    "Falcon 8X": "strong_ulr_with_westbound_caveats",
    "Challenger 650": "transcon_mature",
    "Praetor 600": "super_mid_dispatch",
}

_CABIN_USABILITY: Dict[str, str] = {
    "Global 7500": "boardroom_ulr",
    "Gulfstream G650ER": "executive_ulr",
    "Falcon 8X": "trijet_sleeping_berth_bias",
    "Challenger 650": "stand_up_large_cabin",
    "Praetor 600": "super_mid_efficient_cabin",
}


def _lookup(table: Dict[str, str], model: str, default: str = "standard") -> str:
    if model in table:
        return table[model]
    low = model.lower()
    for k, v in table.items():
        if k.lower() == low:
            return v
    return default


def enrich_comparison_row(model: str, base_row: Dict[str, Any]) -> Dict[str, Any]:
    """Add broker intelligence dimensions to a comparison row."""
    row = dict(base_row)
    row["maintenance_ecosystem"] = _lookup(_MAINTENANCE_ECOSYSTEM, model)
    row["pilot_workload"] = _lookup(_PILOT_WORKLOAD, model, "standard_two_crew")
    row["airport_flexibility"] = _lookup(_AIRPORT_FLEX, model)
    row["dispatch_maturity"] = _lookup(_DISPATCH_MATURITY, model)
    row["cabin_usability"] = _lookup(_CABIN_USABILITY, model)
    return row


def enrich_comparison_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich all comparison rows with intelligence dimensions."""
    if payload.get("comparison_type") == "strategy_vs_strategy":
        return payload
    rows = payload.get("comparison_rows") or []
    aircraft = payload.get("aircraft") or []
    enriched_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        name = str(row.get("label") or row.get("aircraft_id") or "")
        if not name and i < len(aircraft):
            name = str((aircraft[i] or {}).get("name") or "")
        enriched_rows.append(enrich_comparison_row(name, row))
    out = dict(payload)
    out["comparison_rows"] = enriched_rows
    out["intelligence_dimensions"] = [
        "maintenance_ecosystem",
        "pilot_workload",
        "airport_flexibility",
        "dispatch_maturity",
        "cabin_usability",
    ]
    return out


__all__ = ["enrich_comparison_row", "enrich_comparison_payload"]
