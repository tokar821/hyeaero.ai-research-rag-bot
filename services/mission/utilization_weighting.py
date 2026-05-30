"""
Utilization weighting — frequency and annual-hours drive mission weight, not all routes equally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_WEIGHT_PCT_RE = re.compile(
    r"(\d{1,3})\s*%\s*(?:of\s+)?(?:annual\s+)?(?:hours?|utilization|flying)\b",
    re.I,
)
_WEEKLY_RE = re.compile(r"\b(?:weekly|every\s+week)\b", re.I)
_MONTHLY_RE = re.compile(r"\b(?:monthly|every\s+month)\b", re.I)
_QUARTERLY_RE = re.compile(r"\b(?:quarterly|few\s+times\s+a\s+year)\b", re.I)


@dataclass
class RouteWeight:
    label: str
    weight: float
    frequency_band: str = "unknown"
    notes: str = ""


@dataclass
class UtilizationWeightingResult:
    weighted_routes: List[RouteWeight] = field(default_factory=list)
    dominant_route: str = ""
    total_weight: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weighted_routes": [
                {
                    "label": r.label,
                    "weight": round(r.weight, 3),
                    "frequency_band": r.frequency_band,
                    "notes": r.notes,
                }
                for r in self.weighted_routes
            ],
            "dominant_route": self.dominant_route,
            "total_weight": round(self.total_weight, 3),
        }


def _default_weight_for_route(route: str, query: str) -> tuple[float, str]:
    ql = (query or "").lower()
    rl = (route or "").lower()
    if _WEEKLY_RE.search(ql):
        return 1.0, "weekly"
    if _MONTHLY_RE.search(ql):
        return 0.65, "monthly"
    if _QUARTERLY_RE.search(ql) or "occasional" in ql:
        if any(h in rl for h in ("dubai", "singapore", "tokyo", "london")):
            return 0.15, "episodic"
        return 0.35, "occasional"
    return 0.55, "regular"


def compute_utilization_weighting(
    mission: Any,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> UtilizationWeightingResult:
    routes = list(getattr(mission, "routes", None) or [])
    result = UtilizationWeightingResult()
    pct_match = _WEIGHT_PCT_RE.search(query or "")

    for route in routes:
        w, band = _default_weight_for_route(route, query)
        if pct_match and routes.index(route) == 0:
            try:
                w = max(w, int(pct_match.group(1)) / 100.0)
            except ValueError:
                pass
        result.weighted_routes.append(
            RouteWeight(label=route, weight=w, frequency_band=band)
        )

    if isinstance(data_used, dict):
        cog = data_used.get("mission_center_of_gravity") or {}
        if isinstance(cog, dict) and cog.get("episodic_distortion_risk"):
            for rw in result.weighted_routes:
                if any(
                    h in rw.label.lower()
                    for h in ("dubai", "singapore", "tokyo", "london", "paris", "riyadh")
                ):
                    rw.weight = min(rw.weight, 0.2)
                    rw.notes = "Episodic continuation — down-weighted for procurement."

    result.total_weight = sum(r.weight for r in result.weighted_routes) or 1.0
    if result.weighted_routes:
        result.dominant_route = max(result.weighted_routes, key=lambda r: r.weight).label

    if isinstance(data_used, dict):
        data_used["utilization_weighting"] = result.to_dict()

    return result


__all__ = ["UtilizationWeightingResult", "compute_utilization_weighting"]
