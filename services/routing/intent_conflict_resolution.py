"""
Intent Conflict Resolution Layer (ICRL) — multi-intent query planning before deterministic execution.

Resolves competing comparison / buy / constraint / mission signals into a single execution plan.
Does not replace authority dispatch or deterministic execution guard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

_BUDGET_BUFFER = 0.85

_VS_RE = re.compile(r"\b(?:vs\.?|versus|compare|comparison)\b", re.I)
_ALTERNATIVES_RE = re.compile(r"\balternatives?\s+to\b", re.I)
_BUDGET_MODIFIER_RE = re.compile(
    r"\b(?:under|around|about|<=?)\s*\$?\s*\d+(?:\.\d+)?\s*(?:m|mm|million|mil)\b",
    re.I,
)
_BUY_RE = re.compile(
    r"\b(?:"
    r"good\s+deal|fair\s+deal|worth\s+it|overpriced|"
    r"good\s+buy|buy\s+decision|afford(?:able)?"
    r")\b",
    re.I,
)
_PAX_RE = re.compile(
    r"\b(\d+)\s*(?:pax|passengers?|people|seats?)\b|\b(?:for|with)\s+(\d+)\s+(?:pax|passengers?|people)\b",
    re.I,
)
_RANGE_RE = re.compile(r"\b(\d{3,5})\s*(?:nm|nautical)\b", re.I)
_MISSION_ROUTE_RE = re.compile(
    r"\b(?:from|to|between)\b|\b(?:nyc|new\s+york|la|los\s+angeles|miami|lax|mia|jfk)\b",
    re.I,
)
_SHOULD_BUY_RE = re.compile(r"\bwhat\s+should\s+i\s+buy\b", re.I)


class IntentNodeType(str, Enum):
    COMPARISON = "comparison"
    BUY_DECISION = "buy_decision"
    CONSTRAINT = "constraint"
    MISSION = "mission"


class ConflictType(str, Enum):
    SINGLE_INTENT = "SINGLE_INTENT"
    MULTI_COMPARISON = "MULTI_COMPARISON"
    COMPARISON_PLUS_BUY = "COMPARISON_PLUS_BUY"
    COMPARISON_PLUS_CONSTRAINT = "COMPARISON_PLUS_CONSTRAINT"
    TRIPLE_PLUS_COMPARISON = "TRIPLE_PLUS_COMPARISON"
    MISSION_OVERLAY = "MISSION_OVERLAY"


@dataclass(frozen=True)
class IntentNode:
    node_type: str
    confidence: float = 1.0
    source: str = "heuristic"


@dataclass
class IntentGraph:
    intents: List[IntentNode] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    modifiers: List[str] = field(default_factory=list)

    def intent_types(self) -> frozenset[str]:
        return frozenset(n.node_type for n in self.intents)


@dataclass
class ResolvedExecutionPlan:
    ui_intent: str
    layout_type: str
    primary_mode: str
    secondary_modes: List[str] = field(default_factory=list)
    filtered_entities: List[str] = field(default_factory=list)
    constraint_result: Dict[str, bool] = field(default_factory=dict)
    execution_strategy: str = "hybrid_safe"
    comparison_mode: str = "pairwise"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ui_intent": self.ui_intent,
            "layout_type": self.layout_type,
            "primary_mode": self.primary_mode,
            "secondary_modes": list(self.secondary_modes),
            "filtered_entities": list(self.filtered_entities),
            "constraint_result": dict(self.constraint_result),
            "execution_strategy": self.execution_strategy,
            "comparison_mode": self.comparison_mode,
        }


@dataclass
class IntentResolutionResult:
    conflict_type: ConflictType
    graph: IntentGraph
    plan: ResolvedExecutionPlan
    execution_strategy: str
    handled_by_icrl: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_type": self.conflict_type.value,
            "execution_strategy": self.execution_strategy,
            "handled_by_icrl": self.handled_by_icrl,
            "graph": {
                "intents": [
                    {"node_type": n.node_type, "confidence": n.confidence, "source": n.source}
                    for n in self.graph.intents
                ],
                "entities": list(self.graph.entities),
                "constraints": dict(self.graph.constraints),
                "modifiers": list(self.graph.modifiers),
            },
            "plan": self.plan.to_dict(),
        }


def _parse_budget_millions(query: str) -> Optional[float]:
    from services.intent_persistence.pivot import _parse_budget_millions

    return _parse_budget_millions(query or "")


def _parse_pax(query: str) -> Optional[int]:
    m = _PAX_RE.search(query or "")
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    try:
        v = int(raw)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _parse_range_nm(query: str) -> Optional[int]:
    m = _RANGE_RE.search(query or "")
    if not m:
        return None
    try:
        v = int(m.group(1))
        return v if v >= 100 else None
    except (TypeError, ValueError):
        return None


def _resolve_entities(query: str) -> List[str]:
    from services.comparison.aircraft_registry_lock import lock_comparison_aircraft
    from services.consultant.recommendation_engine import detect_models_from_text

    lock = lock_comparison_aircraft(detect_models_from_text(query or ""))
    return [m for m in lock.canonical if m]


def _qri_intent_value(qri: Any) -> str:
    if qri is None or getattr(qri, "intent", None) is None:
        return ""
    return str(qri.intent.value)


def build_intent_graph(
    query: str,
    qri: Any = None,
    unified_intent: Any = None,
) -> IntentGraph:
    """Convert a query into an intent graph with entities, constraints, and modifiers."""
    q = (query or "").strip()
    entities = _resolve_entities(q)
    constraints: Dict[str, Any] = {}
    modifiers: List[str] = []
    intents: List[IntentNode] = []

    budget_m = _parse_budget_millions(q)
    if budget_m is not None:
        constraints["budget_m"] = budget_m
        constraints["budget_usd"] = budget_m * 1_000_000
    pax = _parse_pax(q)
    if pax is not None:
        constraints["pax"] = pax
    range_nm = _parse_range_nm(q)
    if range_nm is not None:
        constraints["range_nm"] = range_nm

    if _VS_RE.search(q):
        modifiers.append("vs")
    if _ALTERNATIVES_RE.search(q):
        modifiers.append("alternatives")
    if _BUDGET_MODIFIER_RE.search(q):
        modifiers.append("budget_cap")

    qri_intent = _qri_intent_value(qri)
    has_comparison_signal = bool(
        _VS_RE.search(q)
        or qri_intent == "aircraft_comparison"
        or len(entities) >= 2
    )
    has_buy_signal = bool(_BUY_RE.search(q)) and not _SHOULD_BUY_RE.search(q)
    has_constraint_signal = bool(constraints.get("budget_m") is not None or constraints.get("pax") or constraints.get("range_nm"))
    has_mission_signal = bool(
        qri_intent in ("acquisition_recommendation", "mission_feasibility", "shortlist_ranking")
        or _SHOULD_BUY_RE.search(q)
        or (_MISSION_ROUTE_RE.search(q) and not has_comparison_signal)
    )

    if has_comparison_signal:
        intents.append(IntentNode(IntentNodeType.COMPARISON.value, confidence=0.95))
    if has_buy_signal:
        intents.append(IntentNode(IntentNodeType.BUY_DECISION.value, confidence=0.85))
    if has_constraint_signal:
        intents.append(IntentNode(IntentNodeType.CONSTRAINT.value, confidence=0.9))
    if has_mission_signal:
        intents.append(IntentNode(IntentNodeType.MISSION.value, confidence=0.88))

    if not intents:
        if qri_intent:
            mapped = {
                "aircraft_comparison": IntentNodeType.COMPARISON.value,
                "acquisition_recommendation": IntentNodeType.MISSION.value,
                "mission_feasibility": IntentNodeType.MISSION.value,
            }.get(qri_intent, IntentNodeType.MISSION.value)
            intents.append(IntentNode(mapped, confidence=0.7))

    return IntentGraph(
        intents=intents,
        entities=entities,
        constraints=constraints,
        modifiers=modifiers,
    )


def classify_conflict_type(intent_graph: IntentGraph) -> ConflictType:
    """Classify multi-intent conflicts from an intent graph."""
    types = intent_graph.intent_types()
    n_entities = len(intent_graph.entities)
    has_comparison = IntentNodeType.COMPARISON.value in types
    has_buy = IntentNodeType.BUY_DECISION.value in types
    has_constraint = IntentNodeType.CONSTRAINT.value in types
    has_mission = IntentNodeType.MISSION.value in types

    if has_mission and not has_comparison and not has_buy:
        return ConflictType.SINGLE_INTENT

    if has_mission and (has_comparison or has_buy):
        return ConflictType.MISSION_OVERLAY

    if n_entities >= 3 and has_comparison:
        if has_constraint:
            return ConflictType.TRIPLE_PLUS_COMPARISON
        return ConflictType.TRIPLE_PLUS_COMPARISON

    if has_comparison and has_constraint:
        return ConflictType.COMPARISON_PLUS_CONSTRAINT

    if has_comparison and has_buy:
        return ConflictType.COMPARISON_PLUS_BUY

    if n_entities >= 2 and has_comparison:
        return ConflictType.SINGLE_INTENT

    if len(types) > 1:
        return ConflictType.MULTI_COMPARISON

    return ConflictType.SINGLE_INTENT


def _apply_budget_filter(
    entities: Sequence[str],
    budget_m: Optional[float],
) -> Tuple[List[str], Dict[str, bool]]:
    if budget_m is None or not entities:
        return list(entities), {e: True for e in entities}

    from rag.aviation_engines.capabilities import find_catalog_matches, typical_market_price_usd

    cap_usd = budget_m * 1_000_000 * _BUDGET_BUFFER
    filtered: List[str] = []
    results: Dict[str, bool] = {}
    for entity in entities:
        rows = find_catalog_matches([entity])
        price = typical_market_price_usd(rows[0]) if rows else 0.0
        if price <= 0:
            results[entity] = False
            continue
        ok = price <= cap_usd
        results[entity] = ok
        if ok:
            filtered.append(entity)
    return filtered, results


def resolve_multi_intent_execution(intent_graph: IntentGraph) -> ResolvedExecutionPlan:
    """Apply strict priority resolution rules to produce an execution plan."""
    conflict = classify_conflict_type(intent_graph)
    entities = list(intent_graph.entities)
    constraints = dict(intent_graph.constraints)
    types = intent_graph.intent_types()

    secondary_modes: List[str] = []
    comparison_mode = "pairwise"
    layout_type = "side_by_side"
    ui_intent = "comparison"
    execution_strategy = "hybrid_safe"
    constraint_result: Dict[str, bool] = {e: True for e in entities}
    filtered_entities = list(entities)

    has_mission_overlay = conflict == ConflictType.MISSION_OVERLAY
    if has_mission_overlay:
        secondary_modes.append(IntentNodeType.MISSION.value)

    if conflict in (ConflictType.TRIPLE_PLUS_COMPARISON, ConflictType.MULTI_COMPARISON):
        comparison_mode = "comparison_matrix"
        layout_type = "comparison_matrix_with_filter"
        ui_intent = "multi_intent_comparison_decision"
        execution_strategy = "deterministic_only"

    if IntentNodeType.CONSTRAINT.value in types or constraints.get("budget_m") is not None:
        filtered_entities, constraint_result = _apply_budget_filter(
            entities,
            constraints.get("budget_m"),
        )
        if conflict in (
            ConflictType.COMPARISON_PLUS_CONSTRAINT,
            ConflictType.TRIPLE_PLUS_COMPARISON,
        ):
            layout_type = "comparison_matrix_with_filter"
            ui_intent = "multi_intent_comparison_decision"
            execution_strategy = "deterministic_only"
            if IntentNodeType.CONSTRAINT.value not in secondary_modes:
                secondary_modes.append("constraint_filter")

    if conflict == ConflictType.COMPARISON_PLUS_BUY:
        ui_intent = "multi_intent_comparison_decision"
        layout_type = "comparison_matrix_with_filter"
        execution_strategy = "deterministic_only"
        if "buy_decision" not in secondary_modes:
            secondary_modes.append("buy_decision")

    if conflict == ConflictType.SINGLE_INTENT:
        if len(entities) >= 2 and IntentNodeType.COMPARISON.value in types:
            execution_strategy = "deterministic_only"
            ui_intent = "comparison"
            layout_type = "side_by_side"
        elif IntentNodeType.MISSION.value in types:
            execution_strategy = "hybrid_safe"
            ui_intent = "mission"
            layout_type = "mission_brief"

    primary_mode = IntentNodeType.COMPARISON.value
    if IntentNodeType.COMPARISON.value not in types and IntentNodeType.MISSION.value in types:
        primary_mode = IntentNodeType.MISSION.value

    return ResolvedExecutionPlan(
        ui_intent=ui_intent,
        layout_type=layout_type,
        primary_mode=primary_mode,
        secondary_modes=secondary_modes,
        filtered_entities=filtered_entities,
        constraint_result=constraint_result,
        execution_strategy=execution_strategy,
        comparison_mode=comparison_mode,
    )


def _icrl_should_handle(conflict: ConflictType, plan: ResolvedExecutionPlan) -> bool:
    if conflict == ConflictType.SINGLE_INTENT:
        return False
    if plan.execution_strategy != "deterministic_only":
        return False
    return conflict in (
        ConflictType.TRIPLE_PLUS_COMPARISON,
        ConflictType.MULTI_COMPARISON,
        ConflictType.COMPARISON_PLUS_CONSTRAINT,
        ConflictType.COMPARISON_PLUS_BUY,
        ConflictType.MISSION_OVERLAY,
    )


def resolve_intent_conflicts(context: Dict[str, Any]) -> IntentResolutionResult:
    """Resolve multi-intent conflicts for a consultant turn."""
    ctx = context if isinstance(context, dict) else {}
    query = str(ctx.get("query") or "")
    qri = ctx.get("qri")
    unified = ctx.get("unified_route") or ctx.get("unified_intent")

    graph = build_intent_graph(query, qri=qri, unified_intent=unified)
    lock_raw = ctx.get("intent_lock")
    if lock_raw is not None:
        from services.core.semantic_intent_lock_engine import IntentLock

        lock = lock_raw if isinstance(lock_raw, IntentLock) else IntentLock.from_dict(lock_raw)
        if lock is not None:
            if lock.canonical_models:
                graph.entities = list(lock.canonical_models)
            if lock.constraints:
                graph.constraints.update(dict(lock.constraints))
    conflict = classify_conflict_type(graph)
    plan = resolve_multi_intent_execution(graph)
    handled = _icrl_should_handle(conflict, plan)

    return IntentResolutionResult(
        conflict_type=conflict,
        graph=graph,
        plan=plan,
        execution_strategy=plan.execution_strategy,
        handled_by_icrl=handled,
    )


def _format_constraint_block(plan: ResolvedExecutionPlan, budget_m: Optional[float]) -> str:
    if not plan.constraint_result:
        return ""
    lines = ["Budget constraint filter:"]
    if budget_m is not None:
        lines.append(f"- Stated budget: ${budget_m:.0f}M (85% acquisition cap applied)")
    for entity, passed in plan.constraint_result.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"- {entity}: {status}")
    passing = [e for e, ok in plan.constraint_result.items() if ok]
    if passing:
        lines.append(f"- Eligible for comparison: {', '.join(passing)}")
    else:
        lines.append("- No aircraft passed budget filter; comparison uses best-effort catalog data.")
    return "\n".join(lines)


def _rank_affordability(entities: Sequence[str]) -> List[str]:
    from rag.aviation_engines.capabilities import find_catalog_matches, typical_market_price_usd

    scored: List[Tuple[float, str]] = []
    for entity in entities:
        rows = find_catalog_matches([entity])
        price = typical_market_price_usd(rows[0]) if rows else float("inf")
        scored.append((price, entity))
    scored.sort(key=lambda x: x[0])
    return [name for _, name in scored]


def execute_icrl_plan(
    context: Dict[str, Any],
    resolution: IntentResolutionResult,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Execute a resolved ICRL plan deterministically.

    Returns (kind, payload) professional response or None when execution cannot proceed.
    """
    if not resolution.handled_by_icrl:
        return None

    ctx = context if isinstance(context, dict) else {}
    if ctx.get("authority_dispatch_result") is not None:
        return None

    query = str(ctx.get("query") or "")
    plan = resolution.plan
    pre_llm_patch = dict(ctx.get("pre_llm_pipeline_patch") or {})
    compare_models = [m for m in plan.filtered_entities if m]

    if plan.primary_mode == IntentNodeType.COMPARISON.value and len(compare_models) >= 2:
        from services.comparison.comparison_pipeline_v2_responder import respond_aircraft_comparison

        handler_du = dict(pre_llm_patch)
        handler_du["intent_conflict_resolution"] = resolution.to_dict()
        handler_du["icrl_execution_plan"] = plan.to_dict()
        handler_du["authority_dispatch_kind"] = "comparison"
        handler_du["deterministic_execution"] = {
            "bypassed_llm": True,
            "trigger_reason": "icrl_deterministic_plan",
            "final_responder": "icrl_comparison_matrix",
            "deterministic_intent": "comparison",
            "comparison_mode": plan.comparison_mode,
        }

        answer = respond_aircraft_comparison(
            query,
            compare_models=compare_models,
            data_used=handler_du,
        )
        blocks: List[str] = [answer]
        budget_m = resolution.graph.constraints.get("budget_m")
        if plan.constraint_result and budget_m is not None:
            blocks.append(_format_constraint_block(plan, budget_m))
        if "buy_decision" in plan.secondary_modes:
            ranked = _rank_affordability(compare_models)
            if ranked:
                blocks.append(
                    "Affordability ranking (catalog typical acquisition, lowest first): "
                    + " → ".join(ranked)
                )
        if IntentNodeType.MISSION.value in plan.secondary_modes:
            blocks.append(
                "Mission overlay deferred: route/pax feasibility runs as secondary analysis only."
            )

        return "professional", {
            "answer": "\n\n".join(b for b in blocks if b.strip()),
            "sources": [],
            "data_used": handler_du,
            "aircraft_images": [],
            "error": None,
        }

    return None


__all__ = [
    "ConflictType",
    "IntentGraph",
    "IntentNode",
    "IntentNodeType",
    "IntentResolutionResult",
    "ResolvedExecutionPlan",
    "build_intent_graph",
    "classify_conflict_type",
    "execute_icrl_plan",
    "resolve_intent_conflicts",
    "resolve_multi_intent_execution",
]
