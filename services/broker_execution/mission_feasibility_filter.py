"""
Phase 56 — hard mission feasibility filter on broker_reasoning candidates (not ranking).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from services.broker_reasoning.mission_interpreter import interpret_mission

# Practical max range (nm) — feasibility gate only, not scoring.
_MODEL_RANGE_NM: Dict[str, int] = {
    "Citation CJ4": 2165,
    "CJ4": 2165,
    "Citation Latitude": 2700,
    "Latitude": 2700,
    "Citation Longitude": 3500,
    "Longitude": 3500,
    "Gulfstream G280": 3600,
    "G280": 3600,
    "Praetor 600": 3400,
    "Challenger 350": 3200,
}

# Great-circle style minimums for common city pairs (nm).
_CITY_PAIR_MIN_NM: Dict[Tuple[str, str], int] = {
    ("NEW YORK", "TOKYO"): 5900,
    ("NYC", "TOKYO"): 5900,
    ("NEW YORK", "LONDON"): 3000,
    ("BOSTON", "DENVER"): 1600,
}

_NONSTOP_RE = re.compile(r"(?is)\bnonstop\b")


def _normalize_city(token: str) -> str:
    return re.sub(r"\s+", " ", (token or "").strip().upper())


def _required_range_nm(query: str, route: Optional[str]) -> Optional[int]:
    q = query or ""
    if re.search(r"(?is)\bcoast.?to.?coast\b", q):
        return 2600
    if not route:
        return None
    parts = re.split(r"[-–—]", route.replace(" ", ""))
    if len(parts) >= 2:
        a = _normalize_city(parts[0].replace("-", " "))
        b = _normalize_city(parts[-1].replace("-", " "))
        for key, nm in _CITY_PAIR_MIN_NM.items():
            if (a.startswith(key[0][:3]) or key[0] in a) and (b.startswith(key[1][:3]) or key[1] in b):
                return nm
            if (a.startswith(key[1][:3]) or key[1] in a) and (b.startswith(key[0][:3]) or key[0] in b):
                return nm
        if "TOKYO" in a + b and ("NEW YORK" in a + b or "NYC" in a + b or "YORK" in a + b):
            return 5900
    return None


def _model_range_nm(model: str) -> Optional[int]:
    m = (model or "").strip()
    if m in _MODEL_RANGE_NM:
        return _MODEL_RANGE_NM[m]
    for key, nm in _MODEL_RANGE_NM.items():
        if key.lower() in m.lower():
            return nm
    return None


def _infeasible_models(required_nm: int) -> Set[str]:
    out: Set[str] = set()
    for model, max_nm in _MODEL_RANGE_NM.items():
        if max_nm < required_nm * 0.95:
            out.add(model)
    return out


def apply_mission_feasibility_filter(
    query: str,
    *,
    data_used: Optional[dict] = None,
) -> None:
    """
    Filter broker_reasoning candidate lists when route requires nonstop beyond aircraft range.
    """
    du = data_used if isinstance(data_used, dict) else {}
    interp = interpret_mission(query or "")
    nonstop = bool(_NONSTOP_RE.search(query or "")) or bool(
        re.search(r"(?is)\btokyo\b", query or "") and re.search(r"(?is)\bnew\s+york\b", query or "")
    )
    required = _required_range_nm(query, interp.route)
    if required is None or not nonstop:
        du["mission_feasibility_checked"] = True
        return

    infeasible = _infeasible_models(required)
    du["mission_feasibility_checked"] = True
    du["mission_required_range_nm"] = required
    du["mission_infeasible_models"] = sorted(infeasible)

    br = du.get("broker_reasoning")
    if not isinstance(br, dict):
        return

    def _filter_list(models: List[Any]) -> List[str]:
        out = []
        for m in models or []:
            ms = str(m).strip()
            if not ms:
                continue
            blocked = any(iff.lower() in ms.lower() for iff in infeasible)
            if blocked:
                continue
            rng = _model_range_nm(ms)
            if rng is not None and rng < required * 0.95:
                continue
            out.append(ms)
        return out

    for key in ("category", "alternatives", "mission"):
        block = br.get(key)
        if not isinstance(block, dict):
            continue
        if "candidates" in block:
            block["candidates"] = _filter_list(list(block.get("candidates") or []))
        if "models" in block:
            block["models"] = _filter_list(list(block.get("models") or []))

    comp = br.get("comparison")
    if isinstance(comp, dict) and comp.get("models"):
        comp["models"] = _filter_list(list(comp.get("models") or []))


__all__ = ["apply_mission_feasibility_filter"]
