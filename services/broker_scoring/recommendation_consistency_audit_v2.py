"""
Phase 50 — recommendation consistency audit across conversation turns.

Detects RECOMMENDATION_DRIFT when primary aircraft changes without new budget/mission info.
Measurement only — flags stored on ``data_used``; does not alter routing or answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_AUDIT_KEY = "recommendation_consistency_audit_v2"
_HISTORY_KEY = "recommendation_audit_v2_history"

_MODEL_TOKEN_RE = re.compile(
    r"(?is)\b(?:gulfstream\s+)?(?:g\d{3}(?:er)?|citation\s+(?:latitude|longitude|cj\d+)|"
    r"challenger\s+\d+|falcon\s+\d+|praetor\s+\d+|global\s+\d+|phenom\s+\d+|learjet\s+\d+)\b"
)
_BUDGET_RE = re.compile(
    r"(?is)\$\s*(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)\b"
)
_MISSION_SIG_RE = re.compile(
    r"(?is)\b(?P<pax>\d+)\s+passengers?\b|\b(?:nonstop|coast.?to.?coast)\b"
)


@dataclass
class TurnSnapshot:
    turn: int
    primary: str
    budget_musd: Optional[float]
    mission_signature: str
    query: str
    rejected: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "primary": self.primary,
            "budget_musd": self.budget_musd,
            "mission_signature": self.mission_signature,
            "query": self.query,
            "rejected": list(self.rejected),
        }


def _parse_budget(query: str, data_used: Dict[str, Any]) -> Optional[float]:
    m = _BUDGET_RE.search(query or "")
    if m:
        try:
            val = float(m.group("amt"))
            unit = (m.group("unit") or "m").lower()
            return val / 1000.0 if unit == "k" else val
        except (TypeError, ValueError):
            pass
    ctx = data_used.get("client_context") or data_used.get("broker_conversation_context") or {}
    if isinstance(ctx, dict) and ctx.get("remembered_budget_musd") is not None:
        try:
            return float(ctx["remembered_budget_musd"])
        except (TypeError, ValueError):
            pass
    return None


def _mission_signature(query: str, data_used: Dict[str, Any]) -> str:
    parts: List[str] = []
    m = _MISSION_SIG_RE.search(query or "")
    if m and m.group("pax"):
        parts.append(f"pax:{m.group('pax')}")
    if re.search(r"(?is)\bnonstop\b", query or ""):
        parts.append("nonstop")
    if re.search(r"(?is)\bcoast.?to.?coast\b", query or ""):
        parts.append("coast")
    br = data_used.get("broker_reasoning") or {}
    if isinstance(br, dict):
        mission = br.get("mission") or {}
        if isinstance(mission, dict):
            if mission.get("passengers"):
                parts.append(f"pax:{mission['passengers']}")
            if mission.get("range_nm"):
                parts.append(f"rng:{mission['range_nm']}")
    return "|".join(sorted(set(parts)))


def _normalize_model(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    low = n.lower()
    if low.startswith("gulfstream g"):
        return n
    m = _MODEL_TOKEN_RE.search(n)
    return m.group(0).title() if m else n


def _extract_primary(data_used: Dict[str, Any], answer: str) -> str:
    rec = data_used.get("executive_recommendation") or {}
    if isinstance(rec, dict) and rec.get("primary_recommendation"):
        return _normalize_model(str(rec["primary_recommendation"]))

    for pat in (
        r"(?is)i'd focus on(?:\s+the)?\s+([^.\n]+)",
        r"(?is)if i were buying(?: today)?,?\s*i'd focus on(?:\s+the)?\s+([^.\n]+)",
        r"(?is)i would buy(?:\s+the)?\s+([^.\n]+)",
    ):
        m = re.search(pat, answer or "")
        if m:
            return _normalize_model(m.group(1))

    models = _MODEL_TOKEN_RE.findall(answer or "")
    if models:
        return _normalize_model(models[0])
    return ""


def _extract_rejected(data_used: Dict[str, Any], answer: str) -> List[str]:
    rejected: List[str] = []
    rec = data_used.get("executive_recommendation") or {}
    if isinstance(rec, dict):
        for item in rec.get("rejected_options") or []:
            if isinstance(item, dict) and item.get("model"):
                rejected.append(_normalize_model(str(item["model"])))
    for line in (answer or "").splitlines():
        if re.search(r"(?is)(?:would not|not lead|above the|budget cap)", line):
            for m in _MODEL_TOKEN_RE.findall(line):
                rejected.append(_normalize_model(m))
    return list(dict.fromkeys(rejected))


def _load_history(data_used: Dict[str, Any]) -> List[TurnSnapshot]:
    raw = data_used.get(_HISTORY_KEY) or []
    if not isinstance(raw, list):
        return []
    out: List[TurnSnapshot] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(
                TurnSnapshot(
                    turn=int(item.get("turn") or 0),
                    primary=str(item.get("primary") or ""),
                    budget_musd=item.get("budget_musd"),
                    mission_signature=str(item.get("mission_signature") or ""),
                    query=str(item.get("query") or ""),
                    rejected=list(item.get("rejected") or []),
                )
            )
    return out


def _save_history(data_used: Dict[str, Any], history: List[TurnSnapshot]) -> None:
    data_used[_HISTORY_KEY] = [s.to_dict() for s in history[-20:]]


def _models_equivalent(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a.lower().replace(" ", "") == b.lower().replace(" ", "")


def audit_recommendation_consistency_v2(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Record this turn's recommendation and detect drift vs prior turns.

    Returns audit dict with ``recommendation_drift`` bool and ``drift_events`` list.
    """
    du = data_used if isinstance(data_used, dict) else {}
    history = _load_history(du)
    turn = len(history) + 1

    primary = _extract_primary(du, answer)
    budget = _parse_budget(query, du)
    mission_sig = _mission_signature(query, du)
    rejected = _extract_rejected(du, answer)

    snapshot = TurnSnapshot(
        turn=turn,
        primary=primary,
        budget_musd=budget,
        mission_signature=mission_sig,
        query=query,
        rejected=rejected,
    )
    history.append(snapshot)
    _save_history(du, history)

    drift_events: List[Dict[str, Any]] = []
    budget_changes: List[Dict[str, Any]] = []
    mission_changes: List[Dict[str, Any]] = []
    new_aircraft_introduced = bool(
        re.search(r"(?is)\b(?:what about|what if|also|consider|versus|vs\.?)\b", query or "")
    )

    if len(history) >= 2:
        prev = history[-2]
        if (
            budget is not None
            and prev.budget_musd is not None
            and abs(float(budget) - float(prev.budget_musd)) > 0.5
        ):
            budget_changes.append(
                {
                    "from_musd": prev.budget_musd,
                    "to_musd": budget,
                    "turn": turn,
                }
            )
        if mission_sig and mission_sig != prev.mission_signature:
            mission_changes.append(
                {
                    "from": prev.mission_signature,
                    "to": mission_sig,
                    "turn": turn,
                }
            )

        if primary and prev.primary and not _models_equivalent(prev.primary, primary):
            budget_changed = bool(budget_changes)
            mission_changed = bool(mission_changes)
            query_signals_new_info = bool(
                re.search(
                    r"(?is)\b(?:stretch|increase|raise|new budget|actually|changed|now i have)\b",
                    query or "",
                )
            )
            allowed = budget_changed or mission_changed or query_signals_new_info or new_aircraft_introduced
            event_type = "RECOMMENDATION_DRIFT" if allowed else "UNJUSTIFIED_RECOMMENDATION_DRIFT"
            if not allowed:
                drift_events.append(
                    {
                        "type": event_type,
                        "from": prev.primary,
                        "to": primary,
                        "turn_from": prev.turn,
                        "turn_to": turn,
                        "severity": "HIGH",
                        "budget_changed": budget_changed,
                        "mission_changed": mission_changed,
                        "new_aircraft_introduced": new_aircraft_introduced,
                    }
                )
            elif not (budget_changed or mission_changed):
                drift_events.append(
                    {
                        "type": event_type,
                        "from": prev.primary,
                        "to": primary,
                        "turn_from": prev.turn,
                        "turn_to": turn,
                        "severity": "LOW",
                        "budget_changed": False,
                        "mission_changed": False,
                        "new_aircraft_introduced": new_aircraft_introduced,
                    }
                )

    first_primary = history[0].primary if history else ""
    latest_primary = primary or (history[-1].primary if history else "")

    unjustified = any(e.get("type") == "UNJUSTIFIED_RECOMMENDATION_DRIFT" for e in drift_events)

    result = {
        "turn": turn,
        "primary_recommendation": primary,
        "first_primary": first_primary,
        "latest_primary": latest_primary,
        "rejected_aircraft": rejected,
        "budget_musd": budget,
        "mission_signature": mission_sig,
        "budget_changes": budget_changes,
        "mission_changes": mission_changes,
        "recommendation_drift": bool(drift_events),
        "unjustified_recommendation_drift": unjustified,
        "drift_events": drift_events,
        "drift_severity": drift_events[0]["severity"] if drift_events else None,
        "history_length": len(history),
    }
    du[_AUDIT_KEY] = result
    return result


__all__ = ["TurnSnapshot", "audit_recommendation_consistency_v2"]
