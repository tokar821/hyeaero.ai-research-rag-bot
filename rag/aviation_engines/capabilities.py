"""Load and query the structured aircraft capability catalog (broker-style reference)."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "aircraft_capability_catalog.json"

# Budget buffer: only recommend aircraft with typical ask at or below this fraction of stated budget.
_BUDGET_BUFFER = 0.85


def typical_passengers(row: Dict[str, Any]) -> int:
    """Normalized seat count from catalog row (supports legacy ``passenger_capacity``)."""
    for k in ("typical_passengers", "passenger_capacity"):
        try:
            v = int(row.get(k) or 0)
            if v > 0:
                return v
        except (TypeError, ValueError):
            continue
    return 0


def typical_market_price_usd(row: Dict[str, Any]) -> float:
    """Normalized acquisition band in USD (supports legacy ``typical_market_price_usd``)."""
    for k in ("typical_market_price", "typical_market_price_usd"):
        try:
            v = float(row.get(k) or 0)
            if v > 0:
                return v
        except (TypeError, ValueError):
            continue
    return 0.0


@lru_cache(maxsize=1)
def load_capability_rows() -> List[Dict[str, Any]]:
    if not _CATALOG_PATH.is_file():
        logger.warning("aircraft capability catalog missing: %s", _CATALOG_PATH)
        return []
    try:
        raw = _CATALOG_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        return list(data) if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("capability catalog load failed: %s", e)
        return []


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def find_catalog_matches(model_substrings: List[str]) -> List[Dict[str, Any]]:
    """Rows whose aircraft_model fuzzy-matches any substring."""
    rows = load_capability_rows()
    if not rows or not model_substrings:
        return []
    keys = [_norm_key(x) for x in model_substrings if x]
    out: List[Dict[str, Any]] = []
    for r in rows:
        mdl = str(r.get("aircraft_model") or "")
        nk = _norm_key(mdl)
        if any(k and (k in nk or nk in k) for k in keys):
            out.append(r)
    return out


def mission_possible_for_row(mission_nm: float, row: Dict[str, Any], margin: float = 1.15) -> bool:
    try:
        max_r = float(row.get("max_range_nm") or 0)
    except (TypeError, ValueError):
        return False
    need = mission_nm * margin
    return max_r >= need


def _rank_recommendations(
    candidates: List[Dict[str, Any]],
    required_range_nm: float,
    requested_pax: Optional[int],
    *,
    min_count: int = 3,
    max_count: int = 5,
) -> List[Dict[str, Any]]:
    """
    After feasibility + budget filters:
    1. Closest range match → minimize surplus (max_range_nm - required_range_nm) among feasible.
    2. Closest passenger match → minimize |typical_passengers - requested|.
    3. Lowest acquisition cost → ascending typical_market_price.
    Return 3–5 rows when available (fewer if catalog is thin).
    """

    def key(r: Dict[str, Any]) -> tuple:
        try:
            mx = float(r.get("max_range_nm") or 0)
        except (TypeError, ValueError):
            mx = 0.0
        surplus = max(0.0, mx - required_range_nm)
        tp = typical_passengers(r)
        if requested_pax is not None and requested_pax > 0:
            pax_dist = abs(tp - requested_pax)
        else:
            pax_dist = 0
        price = typical_market_price_usd(r)
        return (surplus, pax_dist, price)

    ordered = sorted(candidates, key=key)
    cap = max_count
    if len(ordered) >= min_count:
        return ordered[:cap]
    return ordered[:cap]


def filter_by_mission_pax_budget(
    required_range_nm: float,
    passengers: Optional[int],
    budget_usd: Optional[float],
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Mission filter:
    - max_range_nm >= required_range_nm
    - typical_passengers >= requested (when requested > 0)
    - typical_market_price <= budget_usd * 0.85 when budget given (never above buffer; missing price excluded)
    Then rank and return top 3–5.
    """
    rows = load_capability_rows()
    pax_req = int(passengers) if passengers is not None and int(passengers) > 0 else None
    bud = float(budget_usd) if budget_usd is not None and float(budget_usd) > 0 else None
    max_affordable = bud * _BUDGET_BUFFER if bud is not None else None

    cand: List[Dict[str, Any]] = []
    for r in rows:
        try:
            mx = float(r.get("max_range_nm") or 0)
        except (TypeError, ValueError):
            continue
        if mx < required_range_nm:
            continue
        tp = typical_passengers(r)
        if pax_req is not None and tp < pax_req:
            continue
        price = typical_market_price_usd(r)
        if max_affordable is not None:
            if price <= 0:
                continue
            if price > max_affordable:
                continue
        cand.append(r)

    return _rank_recommendations(cand, required_range_nm, pax_req, min_count=3, max_count=min(limit, 5))
