"""
Phase 51 — full answer traceability (diagnostics only).

Captures routing, retrieval, and synthesis metadata already present in ``data_used``.
Does not alter routing, retrieval, or answer generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_TRACE_KEY = "broker_trace"


@dataclass
class BrokerTrace:
    query: str
    canonical_intent: str = ""
    authority_selected: str = ""
    aircraft_detected: List[str] = field(default_factory=list)
    budget_detected: Optional[float] = None
    retrieval_sources: List[str] = field(default_factory=list)
    executive_primary: Optional[str] = None
    final_answer: str = ""
    broker_quality_score: Optional[float] = None
    execution_path: str = ""
    market_reality_mode: str = ""
    acquisition_infeasible: bool = False
    mission_budget_conflict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "canonical_intent": self.canonical_intent,
            "authority_selected": self.authority_selected,
            "aircraft_detected": list(self.aircraft_detected),
            "budget_detected": self.budget_detected,
            "retrieval_sources": list(self.retrieval_sources),
            "executive_primary": self.executive_primary,
            "final_answer": self.final_answer[:500] if self.final_answer else "",
            "broker_quality_score": self.broker_quality_score,
            "execution_path": self.execution_path,
            "market_reality_mode": self.market_reality_mode,
            "acquisition_infeasible": self.acquisition_infeasible,
            "mission_budget_conflict": self.mission_budget_conflict,
        }


def _parse_budget_musd(query: str, data_used: Dict[str, Any]) -> Optional[float]:
    try:
        from services.executive_broker.acquisition_budget_reality import _parse_budget_musd as _p

        return _p(query) or _budget_from_context(data_used)
    except Exception:
        pass
    m = re.search(
        r"(?is)\$\s*(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)?\b",
        query or "",
    )
    if not m:
        return _budget_from_context(data_used)
    try:
        val = float(m.group("amt"))
        unit = (m.group("unit") or "m").lower()
        if unit == "k":
            return val / 1000.0
        return val if val < 1000 else val
    except (TypeError, ValueError):
        return _budget_from_context(data_used)


def _budget_from_context(data_used: Dict[str, Any]) -> Optional[float]:
    ctx = data_used.get("client_context") or data_used.get("broker_conversation_context") or {}
    if isinstance(ctx, dict) and ctx.get("remembered_budget_musd") is not None:
        try:
            return float(ctx["remembered_budget_musd"])
        except (TypeError, ValueError):
            pass
    return None


def _canonical_intent(data_used: Dict[str, Any]) -> str:
    if data_used.get("intent_collapse_applied"):
        frame = data_used.get("canonical_intent_frame") or {}
        if isinstance(frame, dict):
            intent = frame.get("canonical_intent") or frame.get("intent_type")
            if intent:
                return str(intent)
    lock = data_used.get("intent_lock") or {}
    if isinstance(lock, dict):
        for key in ("dispatch_authority_id", "locked_intent", "intent_kind"):
            if lock.get(key):
                return str(lock[key])
    dispatch = data_used.get("authority_dispatch_kind")
    if dispatch:
        return str(dispatch)
    qri = data_used.get("query_recommendation_intent")
    if qri:
        return str(qri)
    return ""


def _authority_selected(data_used: Dict[str, Any]) -> str:
    dispatch = data_used.get("authority_dispatch_kind")
    if dispatch:
        return str(dispatch)
    lock = data_used.get("intent_lock") or {}
    if isinstance(lock, dict) and lock.get("dispatch_authority_id"):
        return str(lock["dispatch_authority_id"])
    trace = data_used.get("intent_execution_trace") or {}
    if isinstance(trace, dict) and trace.get("final_execution_path"):
        return str(trace["final_execution_path"])
    return ""


def _aircraft_detected(query: str, data_used: Dict[str, Any]) -> List[str]:
    found: List[str] = []
    br = data_used.get("broker_reasoning") or {}
    if isinstance(br, dict):
        for m in br.get("compare_models") or []:
            if m and str(m) not in found:
                found.append(str(m))
        cat = br.get("category") or {}
        if isinstance(cat, dict):
            for m in cat.get("candidates") or []:
                if m and str(m) not in found:
                    found.append(str(m))
    frame = data_used.get("canonical_intent_frame") or {}
    if isinstance(frame, dict):
        scope = frame.get("aircraft_scope") or {}
        if isinstance(scope, dict):
            for m in scope.get("models") or []:
                if m and str(m) not in found:
                    found.append(str(m))
    try:
        from services.consultant.recommendation_engine import detect_models_from_text
        from services.broker_reasoning.broker_reasoning_layer import _resolve_model_name

        for token in detect_models_from_text(query or ""):
            resolved = _resolve_model_name(token)
            if resolved and resolved not in found:
                found.append(resolved)
    except Exception:
        pass
    return found[:12]


def _retrieval_sources(data_used: Dict[str, Any]) -> List[str]:
    sources: List[str] = []
    if data_used.get("comparison_v2"):
        sources.append("comparison_v2")
    if data_used.get("comparison_structured_engine"):
        sources.append("comparison_structured_engine")
    if data_used.get("alternative_execution"):
        sources.append("alternative_execution")
    if data_used.get("buy_decision_dispatch"):
        sources.append("buy_decision_dispatch")
    if data_used.get("valuation_dispatch"):
        sources.append("valuation_dispatch")
    if data_used.get("phly_rows") or data_used.get("phly_authority"):
        sources.append("phly")
    if data_used.get("tavily_payload") or data_used.get("tavily_used"):
        sources.append("tavily")
    if data_used.get("market_reality_layer_applied"):
        sources.append("market_reality")
    if data_used.get("executive_broker_layer_applied"):
        sources.append("executive_broker")
    trace = data_used.get("intent_execution_trace") or {}
    if isinstance(trace, dict):
        path = trace.get("final_execution_path")
        if path and str(path) not in sources:
            sources.append(str(path))
    lock = data_used.get("intent_lock") or {}
    if isinstance(lock, dict) and lock.get("dispatch_authority_id"):
        aid = str(lock["dispatch_authority_id"])
        if aid not in sources:
            sources.append(aid)
    return sources


def _executive_primary(data_used: Dict[str, Any], answer: str) -> Optional[str]:
    rec = data_used.get("executive_recommendation") or {}
    if isinstance(rec, dict) and rec.get("primary_recommendation"):
        return str(rec["primary_recommendation"]).strip()
    m = re.search(
        r"(?is)(?:i'd focus on|if i were buying(?: today)?,?\s*i'd focus on)\s+(?:the\s+)?([^.\n]+)",
        answer or "",
    )
    return m.group(1).strip() if m else None


def _quality_score(data_used: Dict[str, Any]) -> Optional[float]:
    blob = data_used.get("broker_quality_score")
    if isinstance(blob, dict) and blob.get("total") is not None:
        try:
            return float(blob["total"])
        except (TypeError, ValueError):
            pass
    return None


def build_broker_trace(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> BrokerTrace:
    """Assemble trace from existing pipeline metadata."""
    du = data_used if isinstance(data_used, dict) else {}
    q = (query or du.get("query") or "").strip()
    text = (answer or "").strip()

    mr = du.get("market_reality") or {}
    mr_mode = ""
    if isinstance(mr, dict):
        sig = mr.get("signal") or {}
        if isinstance(sig, dict):
            mr_mode = str(sig.get("mode") or "")

    trace = data_used.get("intent_execution_trace") or {}
    exec_path = ""
    if isinstance(trace, dict):
        exec_path = str(trace.get("final_execution_path") or "")

    return BrokerTrace(
        query=q,
        canonical_intent=_canonical_intent(du),
        authority_selected=_authority_selected(du),
        aircraft_detected=_aircraft_detected(q, du),
        budget_detected=_parse_budget_musd(q, du),
        retrieval_sources=_retrieval_sources(du),
        executive_primary=_executive_primary(du, text),
        final_answer=text,
        broker_quality_score=_quality_score(du),
        execution_path=exec_path,
        market_reality_mode=mr_mode,
        acquisition_infeasible=bool(du.get("acquisition_budget_infeasible")),
        mission_budget_conflict=bool(du.get("mission_budget_conflict")),
    )


def attach_broker_trace(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> BrokerTrace:
    """Persist trace on ``data_used`` — diagnostics only."""
    du = data_used if isinstance(data_used, dict) else {}
    trace = build_broker_trace(answer, query=query, data_used=du)
    du[_TRACE_KEY] = trace.to_dict()
    return trace


__all__ = ["BrokerTrace", "attach_broker_trace", "build_broker_trace"]
