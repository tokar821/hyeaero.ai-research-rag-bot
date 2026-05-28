"""
Comparison v2 schema contract — strict structured output for explicit_comparison only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

COMPARISON_SCHEMA_V2: Dict[str, Any] = {
    "type": "object",
    "mode": "explicit_comparison",
    "fields": {
        "aircraft": [
            {
                "name": "string (canonical aircraft name only)",
                "category": "string (light / super-midsize / large-cabin / ULR)",
                "range_nm": "number | null",
                "seats": "number | null",
                "mission_fit_score": "number (0-1)",
                "cost_band": "low | medium | high | ultra",
                "winter_westbound_capability": "boolean | conditional | false",
            }
        ],
        "comparison_matrix": {
            "dimensions": [
                "range",
                "cost_efficiency",
                "dispatch_reliability",
                "payload_flexibility",
                "mission_fit",
            ]
        },
        "verdict": {
            "best_overall": "string aircraft name OR null",
            "conditional_winner": "string OR null",
            "no_fit_reason": "string OR null",
        },
        "data_quality": {
            "status": "OK | INSUFFICIENT_DATA",
            "reason": "string",
        },
    },
}

CostBand = Literal["low", "medium", "high", "ultra"]
DataQualityStatus = Literal["OK", "INSUFFICIENT_DATA"]
WinterWestbound = Union[bool, Literal["conditional"], Literal[False]]


class AircraftEntryV2(TypedDict):
    name: str
    category: str
    range_nm: Optional[float]
    seats: Optional[int]
    mission_fit_score: float
    cost_band: CostBand
    winter_westbound_capability: WinterWestbound


class VerdictV2(TypedDict):
    best_overall: Optional[str]
    conditional_winner: Optional[str]
    no_fit_reason: Optional[str]


class DataQualityV2(TypedDict):
    status: DataQualityStatus
    reason: str


class ComparisonPayloadV2(TypedDict):
    mode: Literal["explicit_comparison"]
    aircraft: List[AircraftEntryV2]
    comparison_matrix: Dict[str, List[str]]
    verdict: VerdictV2
    data_quality: DataQualityV2


class InsufficientComparisonV2(TypedDict):
    mode: Literal["explicit_comparison"]
    status: Literal["INSUFFICIENT_DATA"]
    reason: str


MATRIX_DIMENSIONS: List[str] = list(
    COMPARISON_SCHEMA_V2["fields"]["comparison_matrix"]["dimensions"]
)


def insufficient_comparison(reason: str) -> InsufficientComparisonV2:
    return {
        "mode": "explicit_comparison",
        "status": "INSUFFICIENT_DATA",
        "reason": (reason or "missing canonical aircraft set").strip(),
    }


__all__ = [
    "COMPARISON_SCHEMA_V2",
    "AircraftEntryV2",
    "ComparisonPayloadV2",
    "InsufficientComparisonV2",
    "MATRIX_DIMENSIONS",
    "insufficient_comparison",
]
