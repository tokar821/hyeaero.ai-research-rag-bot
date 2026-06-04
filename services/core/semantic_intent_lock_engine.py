"""
Semantic Intent Lock Engine — Phase 28.

Freezes canonical query semantics before Authority Dispatch.
All downstream layers must read IntentLock; no independent intent re-inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_INTENT_LOCK_ENV = "ENABLE_INTENT_LOCK"
_SEMANTIC_VERSION = "v1"
_AKAL_VERSION = "akal-v1"

INTENT_LOCK_TYPES = frozenset(
    {"comparison", "buy", "alternative", "valuation", "fleet", "optimization", "mission"}
)

_DISPATCH_TO_LOCK = {
    "buy_decision": "buy",
    "comparison": "comparison",
    "alternative": "alternative",
    "valuation": "valuation",
    "fleet": "fleet",
    "optimization": "optimization",
    "mission": "mission",
}

_LOCK_TO_DISPATCH = {
    "buy": "buy_decision",
    "comparison": "comparison",
    "alternative": "alternative",
    "valuation": "valuation",
}


def intent_lock_enabled() -> bool:
    return (os.getenv(_INTENT_LOCK_ENV) or "1").strip().lower() in ("1", "true", "yes")


def semantic_enforcement_enabled() -> bool:
    return intent_lock_enabled()


@dataclass(frozen=True)
class IntentLock:
    """Immutable semantic contract for a single consultant turn."""

    intent_type: str
    canonical_models: Tuple[str, ...]
    constraints: Dict[str, Any]
    origin_query_hash: str
    deterministic_flags: Dict[str, Any]
    dispatch_authority_id: str
    timestamp: str
    semantic_version: str = _SEMANTIC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_type": self.intent_type,
            "canonical_models": list(self.canonical_models),
            "constraints": dict(self.constraints),
            "origin_query_hash": self.origin_query_hash,
            "deterministic_flags": dict(self.deterministic_flags),
            "dispatch_authority_id": self.dispatch_authority_id,
            "timestamp": self.timestamp,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["IntentLock"]:
        if not isinstance(raw, dict):
            return None
        intent = str(raw.get("intent_type") or "mission").strip().lower()
        if intent == "buy_decision":
            intent = "buy"
        models = tuple(str(m).strip() for m in (raw.get("canonical_models") or []) if str(m).strip())
        constraints = dict(raw.get("constraints") or {}) if isinstance(raw.get("constraints"), dict) else {}
        flags = (
            dict(raw.get("deterministic_flags") or {})
            if isinstance(raw.get("deterministic_flags"), dict)
            else {}
        )
        return cls(
            intent_type=intent,
            canonical_models=models,
            constraints=constraints,
            origin_query_hash=str(raw.get("origin_query_hash") or ""),
            deterministic_flags=flags,
            dispatch_authority_id=str(raw.get("dispatch_authority_id") or ""),
            timestamp=str(raw.get("timestamp") or ""),
            semantic_version=str(raw.get("semantic_version") or _SEMANTIC_VERSION),
        )


def compute_origin_query_hash(query: str) -> str:
    normalized = re.sub(r"\s+", " ", (query or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _deterministic_timestamp(origin_query_hash: str) -> str:
    return f"lock-{origin_query_hash[:16]}"


def _resolve_akal_models(raw_models: Sequence[str]) -> Tuple[str, ...]:
    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias

    out: List[str] = []
    seen: set[str] = set()
    for raw in raw_models:
        token = str(raw or "").strip()
        if not token:
            continue
        canonical = resolve_aircraft_alias(token) or token
        key = canonical.lower()
        if key not in seen:
            seen.add(key)
            out.append(canonical)
    return tuple(out)


def _resolve_intent_type(
    query: str,
    *,
    qri: Any = None,
    unified_route: Any = None,
) -> str:
    from services.routing.deterministic_execution_guard import infer_extended_hard_routing_intent

    raw = infer_extended_hard_routing_intent(query or "")
    if raw == "buy_decision":
        return "buy"
    if raw in INTENT_LOCK_TYPES:
        return raw

    qri_intent = ""
    if qri is not None and getattr(qri, "intent", None) is not None:
        qri_intent = str(qri.intent.value).strip().lower()

    if qri_intent in ("aircraft_comparison", "comparison"):
        return "comparison"
    if qri_intent in ("listing_valuation",):
        return "valuation"
    if qri_intent in ("aircraft_alternative", "replacement"):
        return "alternative"

    if unified_route is not None and getattr(unified_route, "execution_path", None) is not None:
        path = str(unified_route.execution_path.value).strip().lower()
        if path == "comparison":
            return "comparison"
        if path == "alternative":
            return "alternative"
        if path == "buy_decision":
            return "buy"

    return "mission"


def build_intent_lock(
    query: str,
    *,
    qri: Any = None,
    unified_route: Any = None,
) -> IntentLock:
    """Convert QRI + unified route into a frozen IntentLock (AKAL-resolved models only)."""
    q = (query or "").strip()
    origin_hash = compute_origin_query_hash(q)

    from services.routing.intent_conflict_resolution import build_intent_graph

    graph = build_intent_graph(q, qri=qri, unified_intent=unified_route)
    canonical_models = _resolve_akal_models(graph.entities)

    intent_type = _resolve_intent_type(q, qri=qri, unified_route=unified_route)

    execution_path = ""
    if unified_route is not None and getattr(unified_route, "execution_path", None) is not None:
        execution_path = str(unified_route.execution_path.value)

    qri_intent = ""
    if qri is not None and getattr(qri, "intent", None) is not None:
        qri_intent = str(qri.intent.value)

    flags: Dict[str, Any] = {
        "hard_routing": intent_type in {
            "comparison",
            "buy",
            "alternative",
            "valuation",
            "fleet",
            "optimization",
        },
        "qri_intent": qri_intent,
        "execution_path": execution_path,
        "akal_version": _AKAL_VERSION,
    }

    return IntentLock(
        intent_type=intent_type,
        canonical_models=canonical_models,
        constraints=dict(graph.constraints),
        origin_query_hash=origin_hash,
        deterministic_flags=flags,
        dispatch_authority_id="",
        timestamp=_deterministic_timestamp(origin_hash),
    )


def compute_dispatch_authority_id(
    dispatch_result: Any,
    *,
    intent_lock: Optional[IntentLock] = None,
) -> str:
    if dispatch_result is None:
        return ""
    kind = str(getattr(dispatch_result, "dispatch_kind", "") or "")
    answer_head = str(getattr(dispatch_result, "answer", "") or "")[:64]
    data_used = getattr(dispatch_result, "data_used", {}) or {}
    models = []
    if isinstance(data_used, dict):
        models = list(data_used.get("authority_dispatch_models") or [])
    lock_hash = intent_lock.origin_query_hash if intent_lock else ""
    blob = "|".join([kind, lock_hash, ",".join(models), answer_head])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def bind_dispatch_authority(
    intent_lock: IntentLock,
    dispatch_result: Any,
) -> IntentLock:
    """Return new IntentLock with dispatch_authority_id bound after dispatch."""
    auth_id = compute_dispatch_authority_id(dispatch_result, intent_lock=intent_lock)
    flags = dict(intent_lock.deterministic_flags)
    flags["dispatch_kind"] = str(getattr(dispatch_result, "dispatch_kind", "") or "")
    return IntentLock(
        intent_type=intent_lock.intent_type,
        canonical_models=intent_lock.canonical_models,
        constraints=intent_lock.constraints,
        origin_query_hash=intent_lock.origin_query_hash,
        deterministic_flags=flags,
        dispatch_authority_id=auth_id,
        timestamp=intent_lock.timestamp,
        semantic_version=intent_lock.semantic_version,
    )


def _icrl_primary_mode(icrl_resolution: Any) -> str:
    """Extract ICRL plan primary_mode from object or dict form."""
    if icrl_resolution is None:
        return ""
    if isinstance(icrl_resolution, dict):
        plan = icrl_resolution.get("plan")
        if isinstance(plan, dict):
            return str(plan.get("primary_mode") or "").strip().lower()
        return ""
    plan = getattr(icrl_resolution, "plan", None)
    if plan is not None:
        return str(getattr(plan, "primary_mode", "") or "").strip().lower()
    return ""


def _authority_bound_models(
    lock_models: set[str],
    data_used: Dict[str, Any],
    dispatch_result: Any = None,
) -> set[str]:
    """Models authorized by IntentLock and dispatch — excludes advisory self-output."""
    allowed = set(lock_models)
    for m in data_used.get("authority_dispatch_models") or []:
        name = str(m or "").strip().lower()
        if name:
            allowed.add(name)
    if dispatch_result is not None:
        auth_du = getattr(dispatch_result, "data_used", {}) or {}
        if isinstance(auth_du, dict):
            for m in auth_du.get("authority_dispatch_models") or []:
                name = str(m or "").strip().lower()
                if name:
                    allowed.add(name)
    return allowed


def _models_from_data_used(data_used: Dict[str, Any]) -> List[str]:
    models: List[str] = []
    for key in ("authority_dispatch_models",):
        for m in data_used.get(key) or []:
            name = str(m or "").strip()
            if name and name not in models:
                models.append(name)

    opt = data_used.get("optimization_result")
    if isinstance(opt, dict):
        for row in opt.get("ranked_candidates") or []:
            if isinstance(row, dict):
                name = str(row.get("aircraft") or "").strip()
                if name and name not in models:
                    models.append(name)

    fleet = data_used.get("fleet_portfolio_strategy")
    if isinstance(fleet, dict):
        for m in fleet.get("current_aircraft") or []:
            name = str(m or "").strip()
            if name and name not in models:
                models.append(name)
        fin = fleet.get("fleet_input") or {}
        if isinstance(fin, dict):
            for m in fin.get("aircraft_owned") or []:
                name = str(m or "").strip()
                if name and name not in models:
                    models.append(name)

    return models


def validate_intent_lock_consistency(
    intent_lock: Any,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    dispatch_result: Any = None,
    icrl_resolution: Any = None,
) -> List[str]:
    """
    Cross-layer semantic alignment checks.

    Returns failure tokens (empty when consistent).
    """
    lock = intent_lock if isinstance(intent_lock, IntentLock) else IntentLock.from_dict(intent_lock)
    if lock is None:
        return ["missing_intent_lock"]

    failures: List[str] = []
    du = dict(data_used or {})

    if lock.intent_type not in INTENT_LOCK_TYPES:
        failures.append("invalid_intent_type")

    if not lock.origin_query_hash:
        failures.append("missing_origin_query_hash")

    lock_models = {m.lower() for m in lock.canonical_models}

    if dispatch_result is not None:
        dispatch_kind = str(getattr(dispatch_result, "dispatch_kind", "") or "")
        expected = _LOCK_TO_DISPATCH.get(lock.intent_type)
        mapped = _DISPATCH_TO_LOCK.get(dispatch_kind, dispatch_kind)
        if expected and dispatch_kind and mapped != lock.intent_type:
            if not (lock.intent_type == "valuation" and dispatch_kind == "buy_decision"):
                failures.append("dispatch_intent_mismatch")

        auth_du = getattr(dispatch_result, "data_used", {}) or {}
        if isinstance(auth_du, dict):
            dispatch_models = [str(m) for m in auth_du.get("authority_dispatch_models") or []]
            for dm in dispatch_models:
                if lock_models and dm.lower() not in lock_models:
                    failures.append("dispatch_model_not_in_lock")

    if lock.deterministic_flags.get("hard_routing") and dispatch_result is not None:
        if not lock.dispatch_authority_id:
            failures.append("missing_dispatch_authority_id")

    if icrl_resolution is not None:
        icrl_primary = _icrl_primary_mode(icrl_resolution)
        if icrl_primary == "comparison" and lock.intent_type not in (
            "comparison",
            "buy",
            "fleet",
            "optimization",
        ):
            failures.append("icrl_intent_type_drift")

    opt = du.get("optimization_result")
    if isinstance(opt, dict) and opt.get("status") != "INSUFFICIENT_DATA" and lock_models:
        allowed = _authority_bound_models(lock_models, du, dispatch_result)
        for row in opt.get("ranked_candidates") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("aircraft") or "").strip()
            if name and name.lower() not in allowed:
                failures.append("optimization_model_not_in_lock")

    fleet = du.get("fleet_portfolio_strategy")
    if isinstance(fleet, dict) and fleet.get("status") != "INSUFFICIENT_DATA":
        budget_lock = lock.constraints.get("budget_m")
        fin = fleet.get("fleet_input") or {}
        if isinstance(fin, dict) and budget_lock is not None:
            fleet_budget = (fin.get("budget_constraints") or {}).get("budget_m")
            if fleet_budget is not None and float(fleet_budget) != float(budget_lock):
                failures.append("fleet_constraint_override")

    return list(dict.fromkeys(failures))


def intent_lock_failures_require_safety_fallback(failures: Sequence[str]) -> bool:
    critical = {
        "missing_intent_lock",
        "dispatch_intent_mismatch",
        "dispatch_model_not_in_lock",
        "missing_dispatch_authority_id",
        "akal_model_lock_drift",
        "icrl_intent_type_drift",
        "optimization_model_not_in_lock",
        "fleet_constraint_override",
    }
    return bool(critical & set(failures))


def build_intent_lock_safety_payload(
    intent_lock: Optional[IntentLock],
    *,
    pre_llm_patch: Optional[Dict[str, Any]] = None,
    trigger: str = "intent_lock_semantic_violation",
) -> Tuple[str, Dict[str, Any]]:
    from services.routing.authority_dispatch import _SAFETY_FALLBACK_ANSWERS

    intent = "comparison"
    if intent_lock is not None:
        intent_key = intent_lock.intent_type
        intent = _LOCK_TO_DISPATCH.get(intent_key, intent_key)
        if intent not in _SAFETY_FALLBACK_ANSWERS:
            intent = "comparison"

    du = dict(pre_llm_patch or {})
    du["intent_lock"] = intent_lock.to_dict() if intent_lock else {}
    du["deterministic_execution"] = {
        "bypassed_llm": True,
        "trigger_reason": trigger,
        "final_responder": "deterministic_safety_fallback",
        "deterministic_intent": intent,
    }
    du["authority_dispatch_safety_fallback"] = intent
    du["intent_lock_validation_failed"] = 1
    answer = _SAFETY_FALLBACK_ANSWERS.get(
        intent,
        "Insufficient verified data for deterministic execution.",
    )
    return "professional", {
        "answer": answer,
        "sources": [],
        "data_used": du,
        "aircraft_images": [],
        "error": None,
    }


def enforce_intent_lock_at_guard(context: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Phase 15 extension — fail-closed when IntentLock is missing or inconsistent."""
    if not semantic_enforcement_enabled():
        return None

    ctx = context if isinstance(context, dict) else {}
    du = dict(ctx.get("pre_llm_pipeline_patch") or {})
    lock_raw = du.get("intent_lock")
    lock = IntentLock.from_dict(lock_raw)

    from services.routing.deterministic_execution_guard import requires_hard_deterministic_responder

    hard = requires_hard_deterministic_responder(ctx)
    if hard and lock is None:
        return build_intent_lock_safety_payload(None, pre_llm_patch=du, trigger="missing_intent_lock")

    if lock is None:
        return None

    failures = validate_intent_lock_consistency(
        lock,
        data_used=du,
        dispatch_result=ctx.get("authority_dispatch_result"),
        icrl_resolution=du.get("intent_conflict_resolution"),
    )
    if intent_lock_failures_require_safety_fallback(failures):
        payload = build_intent_lock_safety_payload(
            lock,
            pre_llm_patch=du,
            trigger="intent_lock_consistency_violation",
        )
        kind, body = payload
        body["data_used"]["intent_lock_consistency_failures"] = failures
        return kind, body
    return None


def compute_deterministic_evaluation_id(
    query: str,
    *,
    intent_lock: Any = None,
    dispatch_result: Any = None,
    answer: str = "",
) -> str:
    lock_dict: Dict[str, Any] = {}
    if isinstance(intent_lock, IntentLock):
        lock_dict = intent_lock.to_dict()
    elif isinstance(intent_lock, dict):
        lock_dict = intent_lock

    dispatch_kind = ""
    dispatch_id = ""
    if dispatch_result is not None:
        dispatch_kind = str(getattr(dispatch_result, "dispatch_kind", "") or "")
        dispatch_id = str(getattr(dispatch_result, "dispatch_authority_id", "") or "")
        if not dispatch_id:
            du = getattr(dispatch_result, "data_used", {}) or {}
            if isinstance(du, dict):
                dispatch_id = str(du.get("authority_dispatch_kind") or dispatch_kind)
    elif isinstance(lock_dict.get("deterministic_flags"), dict):
        dispatch_kind = str(lock_dict["deterministic_flags"].get("dispatch_kind") or "")

    blob = json.dumps(
        {
            "query": (query or "").strip(),
            "intent_lock": lock_dict,
            "dispatch_kind": dispatch_kind,
            "dispatch_authority_id": lock_dict.get("dispatch_authority_id") or dispatch_id,
            "answer_head": (answer or "")[:512],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _hash_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    else:
        blob = str(payload or "")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def build_execution_trace_v2(
    *,
    intent_lock: Any = None,
    dispatch_result: Any = None,
    icrl_resolution: Any = None,
    data_used: Optional[Dict[str, Any]] = None,
    final_answer: str = "",
) -> Dict[str, Any]:
    """Deterministic global trace object for replay audit."""
    du = dict(data_used or {})
    lock_dict: Dict[str, Any] = {}
    if isinstance(intent_lock, IntentLock):
        lock_dict = intent_lock.to_dict()
    elif isinstance(intent_lock, dict):
        lock_dict = dict(intent_lock)
    elif du.get("intent_lock"):
        lock_dict = dict(du["intent_lock"]) if isinstance(du["intent_lock"], dict) else {}

    dispatch_snapshot: Dict[str, Any] = {}
    if dispatch_result is not None:
        dispatch_snapshot = {
            "dispatch_kind": getattr(dispatch_result, "dispatch_kind", None),
            "progress_step": getattr(dispatch_result, "progress_step", None),
            "data_used_keys": sorted((getattr(dispatch_result, "data_used", {}) or {}).keys()),
        }
    elif du.get("authority_dispatch_kind"):
        dispatch_snapshot = {"dispatch_kind": du.get("authority_dispatch_kind")}

    icrl_snapshot: Optional[Dict[str, Any]] = None
    if icrl_resolution is not None:
        icrl_snapshot = icrl_resolution.to_dict() if hasattr(icrl_resolution, "to_dict") else icrl_resolution
    elif isinstance(du.get("intent_conflict_resolution"), dict):
        icrl_snapshot = du["intent_conflict_resolution"]

    advisory_layers = [
        key
        for key in (
            "recommendation_justification",
            "recommendation_confidence",
            "optimization_result",
            "market_intelligence",
            "ownership_intelligence",
            "fleet_portfolio_strategy",
            "executive_synthesis",
            "consultant_evaluation",
        )
        if du.get(key) is not None
    ]

    trace_id = compute_deterministic_evaluation_id(
        str(lock_dict.get("origin_query_hash") or ""),
        intent_lock=lock_dict,
        dispatch_result=dispatch_result,
        answer=final_answer,
    )

    return {
        "trace_id": trace_id,
        "semantic_version": _SEMANTIC_VERSION,
        "intent_lock_snapshot": lock_dict,
        "dispatch_result": dispatch_snapshot,
        "icrl_result": icrl_snapshot,
        "akal_version": _AKAL_VERSION,
        "advisory_layers_used": advisory_layers,
        "final_output_hash": _hash_payload({"answer": (final_answer or "")[:2048]}),
    }


__all__ = [
    "INTENT_LOCK_TYPES",
    "IntentLock",
    "bind_dispatch_authority",
    "build_execution_trace_v2",
    "build_intent_lock",
    "build_intent_lock_safety_payload",
    "compute_deterministic_evaluation_id",
    "compute_dispatch_authority_id",
    "compute_origin_query_hash",
    "enforce_intent_lock_at_guard",
    "intent_lock_enabled",
    "intent_lock_failures_require_safety_fallback",
    "semantic_enforcement_enabled",
    "validate_intent_lock_consistency",
]
