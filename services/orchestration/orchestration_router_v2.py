"""
Orchestration Router V2 — deterministic query-type routing BEFORE recommendation.

Authoritative for:
- response mode
- renderer selection
- fallback authority
- recommendation eligibility
- comparison candidate preservation
- operational synthesis activation

Downstream layers MUST NOT override router decisions when
``orchestration_v2_authoritative`` is set.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ORCHESTRATION_V2_KEY = "orchestration_v2"
ORCHESTRATION_V2_QUERY_TYPE_KEY = "orchestration_v2_query_type"
ROUTER_V2_FINAL_DECISION_KEY = "router_v2_final_decision"


class OrchestrationQueryTypeV2(str, Enum):
    NAMED_AIRCRAFT_CAPABILITY = "named_aircraft_capability"
    EXPLICIT_COMPARISON = "explicit_comparison"
    STRATEGIC_FLEET_ANALYSIS = "strategic_fleet_analysis"
    NETWORK_STRUCTURE = "network_structure"
    RECOMMENDATION_REQUEST = "recommendation_request"


class OrchestrationRendererV2(str, Enum):
    NAMED_CAPABILITY = "named_aircraft_capability"
    COMPARISON_TABLE = "explicit_comparison_table"
    STRATEGIC_ANALYSIS = "strategic_analysis"
    STRATEGIC_COMPARISON = "strategic_comparison"
    NETWORK_TOPOLOGY = "network_topology"
    RECOMMENDATION_BROKER = "recommendation_broker"
    OWNERSHIP_ECONOMICS = "ownership_economics"


_NAMED_CAPABILITY_RE = re.compile(
    r"\b(?:"
    r"can\s+(?:a\s+)?(?:\w+\s+){0,3}(?:fly|make|reach|do)\b|"
    r"would\s+(?:a\s+)?(?:\w+\s+){0,3}(?:work|make\s+it|reach)\b|"
    r"is\s+(?:the\s+)?(?:\w+\s+){0,3}(?:\w+\s+){0,2}(?:capable|realistic|feasible|viable)\b|"
    r"nonstop\s+(?:to|from|possible|feasible)\b.*\?"
    r")\b",
    re.I,
)

_CAPABILITY_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"can|capable|capability|feasible|viable|realistic|reliabl(?:e|y)|"
    r"nonstop|year[- ]round|westbound|nbaa|ifr\s+reserves?|reserves?\b|"
    r"winter|january|payload|passengers?|pax\b|safely\s+handle"
    r")\b",
    re.I,
)

_COMPARISON_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?\b|head[- ]to[- ]head|which\s+is\s+better)\b",
    re.I,
)

_STRONG_COMPARISON_RE = re.compile(
    r"\b(?:compare|comparison|versus|head[- ]to[- ]head|which\s+(?:is\s+)?better)\b",
    re.I,
)

_STRATEGIC_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:structurally\s+)?breaks\b|"
    r"structural(?:ly)?\s+(?:wrong|incoherent|impossible)\b|"
    r"single[\s-]platform|one\s+aircraft\s+(?:realistically\s+)?cover|"
    r"one\s+aircraft\s+for\s+everything|"
    r"one\s+aircraft\s+realistically|"
    r"mixed\s+fleet|fleet\s+segmentation|"
    r"aircraft\s+strategy|"
    r"fleet\s+planning|"
    r"operational\s+tradeoffs?\b|"
    r"dispatch\s+reliability\b|"
    r"utilization\s+(?:mismatch|conflict)\b|"
    r"maintenance\s+(?:profile\s+)?divergence\b|"
    r"scheduling\s+incoherence\b|"
    r"should\s+we\s+(?:run|operate)\s+(?:one|a\s+single)\b|"
    r"leadership\s+(?:wants|believes)\s+one\b|"
    r"super-midsize\s+fleet\s+strategy|"
    r"ultra-long-range\s+flagship|"
    r"one\s+aircraft\s+to\s+cover|"
    r"structurally\s+feasible"
    r")\b",
    re.I,
)

_NETWORK_RE = re.compile(
    r"\b(?:"
    r"hierarchy\b|"
    r"how\s+should\s+.*\s+be\s+structured\b|"
    r"operational\s+hub\b|"
    r"primary\s+hubs?\b|"
    r"network\s+priority\s+model\b|"
    r"dominant\s+planning\s+axis\b|"
    r"how\s+should\s+(?:this|the)\s+network\s+be\s+represented\b|"
    r"primary\s+utilization\b|"
    r"how\s+should\s+(?:this|the)\s+network\s+be\s+(?:interpreted|understood)\b|"
    r"continuation\s+hub\b|"
    r"continuation\s+(?:through|via)\b|"
    r"topology\b|"
    r"dominant\s+utilization\b|"
    r"origin\s+integrity\b|"
    r"without\s+breaking\s+origin\b|"
    r"network\s+structure\b|"
    r"how\s+should\s+continuation\b|"
    r"interpretation\s+wrong\b"
    r")\b",
    re.I,
)

_RECOMMENDATION_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:aircraft|jet|plane)\s+should\b|"
    r"which\s+(?:aircraft|jet|plane)\s+(?:fits|fit|should)\b|"
    r"recommend\b|"
    r"best\s+(?:aircraft|jet|option)\b|"
    r"what\s+should\s+(?:i\s+)?(?:buy|acquire)\b|"
    r"shortlist\b|"
    r"what\s+(?:jet|aircraft)\s+fits\b|"
    r"options\s+for\s+this\s+mission\b|"
    r"what\s+is\s+the\s+best\s+aircraft\b|"
    r"what\s+aircraft\s+realistically\s+remain\b"
    r")\b",
    re.I,
)

_OWNERSHIP_RE = re.compile(
    r"\b(?:ownership\s+economics|cost\s+of\s+ownership|fractional\s+vs|charter\s+transition)\b",
    re.I,
)

_ACQUISITION_RECOMMENDATION_RE = re.compile(
    r"\b(?:"
    r"should\s+we\s+buy|buy,\s*fractional|fractional,\s*or\s+stay\s+charter|"
    r"stay\s+charter|charter\s+\d+\s+hours|hours/year"
    r")\b",
    re.I,
)

_STRATEGIC_FILTERING_RE = re.compile(
    r"\b(?:"
    r"what\s+options\s+remain|options\s+remain\s+after|excluding\s+marginal|"
    r"survives\s+filtering|what\s+aircraft\s+actually\s+survives"
    r")\b",
    re.I,
)

_ARCHETYPE_COMPARISON_RE = re.compile(
    r"\b(?:"
    r"tradeoffs?\s+between|"
    r"(?:single\s+ultra-long-range|mixed\s+fleet|super-midsize)\b.*\bvs\b|"
    r"\bvs\b.*\b(?:single\s+ultra-long-range|mixed\s+fleet|super-midsize|charter\s+support)"
    r")\b",
    re.I,
)

# Task 1 — hard pre-filter literals (cost + ULR conflict)
_HARD_CONFLICT_LITERAL_RE = [
    re.compile(p, re.I)
    for p in (
        r"below\s+a\s+global\s+7500",
        r"cheaper\s+than\s+global\s+7500",
        r"avoid\s+global\s+7500[- ]level\s+cost",
        r"significantly\s+below\s+(?:a\s+)?global\s+7500",
        r"lower\s+operating\s+cost\s+than\s+(?:g650|g7500|global)",
        r"something\s+cheaper\s+than\s+(?:g650|g7500|global)",
        r"replace\s+.+\s+with\s+something\s+cheaper",
        r"operating\s+costs?\s+significantly\s+below\s+(?:a\s+)?global\s+7500",
        r"avoid\s+global\s+7500[- ]level\s+costs?",
    )
]

_COST_ECONOMIC_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"below\s+(?:a\s+)?global|cheaper|lower\s+operating\s+cost|"
    r"7500[- ]level\s+cost|something\s+cheaper|replace\s+.+\s+cheaper|"
    r"cost\s+close\s+to\s+midsize"
    r")\b",
    re.I,
)

_ULR_MISSION_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"transpacific|transatlantic|nonstop|ultra[- ]long|"
    r"tokyo|hong\s+kong|singapore|dubai|johannesburg|sydney|"
    r"lax\s*[-–>→]\s*tokyo|los\s+angeles\s+to\s+tokyo|sfo\s*[-–>→]\s*tokyo|"
    r"westbound\s+tokyo|year[- ]round"
    r")\b",
    re.I,
)

_COMPARISON_MODEL_TOKEN_RE = re.compile(
    r"\b("
    r"global\s+\d{4,5}|"
    r"gulfstream\s+g\s*\d{3,4}(?:er)?|"
    r"g\s*\d{3,4}(?:er)?|"
    r"falcon\s+\d+x?|"
    r"challenger\s+\d{3,4}|"
    r"praetor\s+\d+|"
    r"citation\s+(?:longitude|latitude|3500)"
    r")\b",
    re.I,
)

_VS_GLOBAL_REF_ONLY_RE = re.compile(
    r"\bvs\.?\s+global\s+\d{4,5}\b",
    re.I,
)


@dataclass(frozen=True)
class OrchestrationRouterV2Result:
    query_type: OrchestrationQueryTypeV2
    renderer: OrchestrationRendererV2
    confidence: float
    signals: List[str] = field(default_factory=list)
    authoritative: bool = True
    allow_recommendation_ranking: bool = False
    allow_tier_fallback: bool = False
    allow_operational_synthesis: bool = False
    preserve_comparison_models: tuple[str, ...] = ()
    named_aircraft_models: tuple[str, ...] = ()
    suppress_generic_shortlist: bool = True
    requires_deterministic_pipeline: bool = True
    physics_first_priority: bool = True
    routing_debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "query_type": self.query_type.value,
            "renderer": self.renderer.value,
            "confidence": round(float(self.confidence), 4),
            "signals": list(self.signals),
            "authoritative": self.authoritative,
            "allow_recommendation_ranking": self.allow_recommendation_ranking,
            "allow_tier_fallback": self.allow_tier_fallback,
            "allow_operational_synthesis": self.allow_operational_synthesis,
            "preserve_comparison_models": list(self.preserve_comparison_models),
            "named_aircraft_models": list(self.named_aircraft_models),
            "suppress_generic_shortlist": self.suppress_generic_shortlist,
            "requires_deterministic_pipeline": self.requires_deterministic_pipeline,
            "physics_first_priority": self.physics_first_priority,
        }
        if self.routing_debug:
            out["routing_debug"] = dict(self.routing_debug)
        return out


@dataclass
class _RouteDebugCtx:
    detected_intent: str = ""
    pre_filter_triggered: bool = False
    comparison_override_triggered: bool = False
    network_override_triggered: bool = False
    cost_constraint_detected: bool = False
    long_range_detected: bool = False


def detect_hard_conflict_query(text: str) -> bool:
    """
  Hard pre-filter: cost + ultra-long-range / transpacific conflict.
  When True, router must force strategic_fleet_analysis with NO model extraction.
    """
    ql = (text or "").strip().lower()
    if not ql:
        return False
    if _ACQUISITION_RECOMMENDATION_RE.search(ql):
        return False
    if _STRATEGIC_FILTERING_RE.search(ql):
        return True
    if re.search(r"want\s+to\s+replace\s+a\s+global|replace\s+a\s+global\s+\d{4}", ql) and re.search(
        r"options\s+remain|excluding\s+marginal|westbound|transatlantic", ql
    ):
        return True
    if any(p.search(ql) for p in _HARD_CONFLICT_LITERAL_RE):
        return True
    if _COST_ECONOMIC_SIGNAL_RE.search(ql) and _ULR_MISSION_SIGNAL_RE.search(ql):
        return True
  # Midsize-cost framing + transpacific / LAX-Tokyo corridor
    if re.search(r"cost\s+close\s+to\s+midsize", ql) and _ULR_MISSION_SIGNAL_RE.search(ql):
        return True
    if re.search(r"something\s+cheaper", ql) and re.search(
        r"\b(?:winter|westbound|nbaa|ifr\s+reserves?|london|chicago)\b", ql
    ):
        return True
    return False


def _cost_constraint_detected(ql: str) -> bool:
    return bool(_COST_ECONOMIC_SIGNAL_RE.search(ql))


def _long_range_or_winter_detected(ql: str) -> bool:
    return bool(_ULR_MISSION_SIGNAL_RE.search(ql))


def _normalize_model_token(raw: str) -> str:
    s = re.sub(r"\s+", " ", (raw or "").strip())
    s = re.sub(r"\bg\s+(\d)", r"G\1", s, flags=re.I)
    parts = s.split()
    out: List[str] = []
    for w in parts:
        wl = w.lower()
        if wl in ("g", "global", "gulfstream", "falcon", "challenger", "praetor", "citation"):
            out.append(w.capitalize() if wl != "g" else "G")
        elif re.match(r"g\d", wl):
            out.append("G" + w[1:])
        else:
            out.append(w.upper() if wl in ("er",) else w.capitalize())
    return " ".join(out)


def _detect_models(query: str) -> List[str]:
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        found = list(detect_models_from_text(query or ""))
    except Exception:
        found = []
    ql = (query or "").lower()
    for m in _COMPARISON_MODEL_TOKEN_RE.finditer(ql):
        token = _normalize_model_token(m.group(1))
        if token and token not in found:
            found.append(token)
    return list(dict.fromkeys(found))


def _is_archetype_explicit_comparison(ql: str) -> bool:
    """Fleet archetype vs archetype (tradeoffs / ULR vs mixed fleet) → explicit_comparison."""
    return bool(_ARCHETYPE_COMPARISON_RE.search(ql))


def _blocks_named_capability_routing(ql: str) -> bool:
    return bool(
        detect_hard_conflict_query(ql)
        or _STRATEGIC_FILTERING_RE.search(ql)
        or re.search(r"what\s+options\s+remain|excluding\s+marginal-range", ql)
    )


def _should_route_explicit_comparison(ql: str, models: List[str]) -> bool:
    if _is_archetype_explicit_comparison(ql):
        return True
    unique = list(dict.fromkeys(models))
    has_compare_signal = bool(_COMPARISON_RE.search(ql))
    has_strong_compare = bool(_STRONG_COMPARISON_RE.search(ql))
    if len(unique) < 2:
        return False
    # "vs Global 7500" alone is NOT comparison unless another model exists
    if (
        len(unique) == 1
        and _VS_GLOBAL_REF_ONLY_RE.search(ql)
        and not has_strong_compare
    ):
        return False
    if len(unique) >= 2 and (has_compare_signal or has_strong_compare):
        return True
    if len(unique) >= 2 and has_compare_signal:
        return True
    return len(unique) >= 2 and has_strong_compare


def _build_routing_debug(
    result: OrchestrationRouterV2Result,
    ctx: _RouteDebugCtx,
) -> Dict[str, Any]:
    return {
        "detected_intent": ctx.detected_intent or result.query_type.value,
        "intent_type": ctx.detected_intent or result.query_type.value,
        "pre_filter_triggered": ctx.pre_filter_triggered,
        "comparison_override_triggered": ctx.comparison_override_triggered,
        "network_override_triggered": ctx.network_override_triggered,
        "stabilizer_modified_route": False,
        "cost_constraint_detected": ctx.cost_constraint_detected,
        "long_range_detected": ctx.long_range_detected,
        "final_route": result.query_type.value,
        "signals": list(result.signals),
    }


def _finalize(
    result: OrchestrationRouterV2Result,
    ctx: _RouteDebugCtx,
) -> OrchestrationRouterV2Result:
    debug = _build_routing_debug(result, ctx)
    logger.info(
        "orchestration_v2_route detected_intent=%s pre_filter=%s comparison_override=%s "
        "network_override=%s final_route=%s",
        debug["detected_intent"],
        debug["pre_filter_triggered"],
        debug["comparison_override_triggered"],
        debug["network_override_triggered"],
        debug["final_route"],
    )
    return OrchestrationRouterV2Result(
        query_type=result.query_type,
        renderer=result.renderer,
        confidence=result.confidence,
        signals=result.signals,
        authoritative=result.authoritative,
        allow_recommendation_ranking=result.allow_recommendation_ranking,
        allow_tier_fallback=result.allow_tier_fallback,
        allow_operational_synthesis=result.allow_operational_synthesis,
        preserve_comparison_models=result.preserve_comparison_models,
        named_aircraft_models=result.named_aircraft_models,
        suppress_generic_shortlist=result.suppress_generic_shortlist,
        requires_deterministic_pipeline=result.requires_deterministic_pipeline,
        physics_first_priority=result.physics_first_priority,
        routing_debug=debug,
    )


def route_orchestration_v2(
    query: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
) -> OrchestrationRouterV2Result:
    """
    Classify query into exactly one V2 type and bind renderer + authority flags.

    Precedence (no model extraction until gates pass):
    1. Hard cost+ULR conflict → strategic
    2. Network hierarchy → network_structure
    3. Explicit comparison (compare/vs + ≥2 models) → explicit_comparison
    4. Ownership / strategic / capability / recommendation
    """
    del history
    q = (query or "").strip()
    ql = q.lower()
    ctx = _RouteDebugCtx(
        cost_constraint_detected=_cost_constraint_detected(ql),
        long_range_detected=_long_range_or_winter_detected(ql),
    )
    signals: List[str] = []

    # Edge case: acquisition / charter vs buy (T15) — before cost+ULR pre-filter
    if _ACQUISITION_RECOMMENDATION_RE.search(ql):
        ctx.detected_intent = "acquisition_recommendation"
        signals.append("acquisition_recommendation")
        return _finalize(
            OrchestrationRouterV2Result(
                query_type=OrchestrationQueryTypeV2.RECOMMENDATION_REQUEST,
                renderer=OrchestrationRendererV2.RECOMMENDATION_BROKER,
                confidence=0.90,
                signals=signals,
                allow_recommendation_ranking=True,
                allow_tier_fallback=True,
                allow_operational_synthesis=False,
                suppress_generic_shortlist=False,
                requires_deterministic_pipeline=True,
            ),
            ctx,
        )

    # --- TASK 1: hard pre-filter (NO model extraction) ---
    if detect_hard_conflict_query(q):
        ctx.pre_filter_triggered = True
        ctx.detected_intent = "hard_conflict_pre_filter"
        signals.append("hard_conflict_pre_filter")
        return _finalize(
            OrchestrationRouterV2Result(
                query_type=OrchestrationQueryTypeV2.STRATEGIC_FLEET_ANALYSIS,
                renderer=OrchestrationRendererV2.STRATEGIC_ANALYSIS,
                confidence=0.96,
                signals=signals,
                allow_recommendation_ranking=False,
                allow_tier_fallback=False,
                allow_operational_synthesis=False,
                suppress_generic_shortlist=True,
                requires_deterministic_pipeline=True,
            ),
            ctx,
        )

    models = _detect_models(q)

    # --- TASK 3: network hierarchy (NO aircraft extraction in route result) ---
    if _NETWORK_RE.search(ql):
        ctx.network_override_triggered = True
        ctx.detected_intent = "network_structure"
        signals.append("network_hierarchy_override")
        return _finalize(
            OrchestrationRouterV2Result(
                query_type=OrchestrationQueryTypeV2.NETWORK_STRUCTURE,
                renderer=OrchestrationRendererV2.NETWORK_TOPOLOGY,
                confidence=0.91,
                signals=signals,
                allow_recommendation_ranking=False,
                allow_tier_fallback=False,
                allow_operational_synthesis=True,
                named_aircraft_models=(),
                preserve_comparison_models=(),
                suppress_generic_shortlist=True,
                requires_deterministic_pipeline=True,
            ),
            ctx,
        )

    # --- TASK 2: comparison absolute priority ---
    if _should_route_explicit_comparison(ql, models):
        ctx.comparison_override_triggered = True
        ctx.detected_intent = "explicit_comparison"
        preserved = () if _is_archetype_explicit_comparison(ql) else tuple(dict.fromkeys(models))[:6]
        signals.append("comparison_absolute_priority")
        return _finalize(
            OrchestrationRouterV2Result(
                query_type=OrchestrationQueryTypeV2.EXPLICIT_COMPARISON,
                renderer=OrchestrationRendererV2.COMPARISON_TABLE,
                confidence=0.93,
                signals=signals,
                allow_recommendation_ranking=False,
                allow_tier_fallback=False,
                allow_operational_synthesis=False,
                preserve_comparison_models=preserved,
                suppress_generic_shortlist=True,
                requires_deterministic_pipeline=True,
            ),
            ctx,
        )

    if _OWNERSHIP_RE.search(ql) and not _RECOMMENDATION_RE.search(ql):
        ctx.detected_intent = "ownership_economics"
        return _finalize(
            OrchestrationRouterV2Result(
                query_type=OrchestrationQueryTypeV2.STRATEGIC_FLEET_ANALYSIS,
                renderer=OrchestrationRendererV2.OWNERSHIP_ECONOMICS,
                confidence=0.88,
                signals=["ownership_economics"],
                allow_recommendation_ranking=False,
                allow_tier_fallback=False,
                allow_operational_synthesis=False,
                suppress_generic_shortlist=True,
                requires_deterministic_pipeline=False,
            ),
            ctx,
        )

    if _STRATEGIC_RE.search(ql) and not _RECOMMENDATION_RE.search(ql):
        ctx.detected_intent = "strategic_fleet_analysis"
        signals.append("strategic_fleet_analysis")
        return _finalize(
            OrchestrationRouterV2Result(
                query_type=OrchestrationQueryTypeV2.STRATEGIC_FLEET_ANALYSIS,
                renderer=OrchestrationRendererV2.STRATEGIC_ANALYSIS,
                confidence=0.88,
                signals=signals,
                allow_recommendation_ranking=False,
                allow_tier_fallback=False,
                allow_operational_synthesis=False,
                suppress_generic_shortlist=True,
                requires_deterministic_pipeline=True,
            ),
            ctx,
        )

    # Named aircraft capability — single-model feasibility only
    capability_override = bool(models) and len(models) <= 2 and bool(_CAPABILITY_SIGNAL_RE.search(ql))
    if (
        models
        and len(models) <= 2
        and not _blocks_named_capability_routing(ql)
        and not _ACQUISITION_RECOMMENDATION_RE.search(ql)
        and (capability_override or _NAMED_CAPABILITY_RE.search(q) or re.search(r"\bnonstop\b", ql))
        and not _COMPARISON_RE.search(ql)
    ):
        ctx.detected_intent = "named_aircraft_capability"
        signals.append(
            "named_aircraft_capability_override" if capability_override else "named_aircraft_capability"
        )
        return _finalize(
            OrchestrationRouterV2Result(
                query_type=OrchestrationQueryTypeV2.NAMED_AIRCRAFT_CAPABILITY,
                renderer=OrchestrationRendererV2.NAMED_CAPABILITY,
                confidence=0.93 if capability_override else 0.92,
                signals=signals,
                allow_recommendation_ranking=False,
                allow_tier_fallback=False,
                allow_operational_synthesis=False,
                named_aircraft_models=tuple(models[:2]),
                suppress_generic_shortlist=True,
                requires_deterministic_pipeline=True,
            ),
            ctx,
        )

    if _RECOMMENDATION_RE.search(ql) or re.search(
        r"\b(?:what\s+aircraft|which\s+jet|recommend|best\s+jet|shortlist)\b", ql
    ):
        ctx.detected_intent = "recommendation_request"
        signals.append("recommendation_request")
        return _finalize(
            OrchestrationRouterV2Result(
                query_type=OrchestrationQueryTypeV2.RECOMMENDATION_REQUEST,
                renderer=OrchestrationRendererV2.RECOMMENDATION_BROKER,
                confidence=0.85,
                signals=signals,
                allow_recommendation_ranking=True,
                allow_tier_fallback=True,
                allow_operational_synthesis=False,
                suppress_generic_shortlist=False,
                requires_deterministic_pipeline=True,
            ),
            ctx,
        )

    ctx.detected_intent = "default_recommendation_request"
    signals.append("default_recommendation_request")
    return _finalize(
        OrchestrationRouterV2Result(
            query_type=OrchestrationQueryTypeV2.RECOMMENDATION_REQUEST,
            renderer=OrchestrationRendererV2.RECOMMENDATION_BROKER,
            confidence=0.55,
            signals=signals,
            allow_recommendation_ranking=bool(_RECOMMENDATION_RE.search(ql)),
            allow_tier_fallback=bool(_RECOMMENDATION_RE.search(ql)),
            allow_operational_synthesis=False,
            suppress_generic_shortlist=not bool(_RECOMMENDATION_RE.search(ql)),
            requires_deterministic_pipeline=True,
        ),
        ctx,
    )


def apply_orchestration_v2_metadata(
    data_used: Dict[str, Any],
    result: OrchestrationRouterV2Result,
) -> None:
    data_used[ORCHESTRATION_V2_KEY] = result.to_dict()
    data_used[ORCHESTRATION_V2_QUERY_TYPE_KEY] = result.query_type.value
    if result.routing_debug:
        data_used["orchestration_v2_routing_debug"] = dict(result.routing_debug)
        data_used[ROUTER_V2_FINAL_DECISION_KEY] = dict(result.routing_debug)
    data_used["orchestration_v2_authoritative"] = result.authoritative
    data_used["router_authoritative"] = True
    data_used["orchestration_v2_renderer"] = result.renderer.value
    data_used["orchestration_v2_allow_ranking"] = result.allow_recommendation_ranking
    data_used["orchestration_v2_allow_tier_fallback"] = result.allow_tier_fallback
    data_used["orchestration_v2_allow_operational_synthesis"] = result.allow_operational_synthesis
    data_used["orchestration_v2_preserve_comparison_models"] = list(
        result.preserve_comparison_models
    )
    data_used["orchestration_v2_named_models"] = list(result.named_aircraft_models)

    if result.suppress_generic_shortlist:
        data_used["defer_global_shortlist"] = True
        if not result.allow_recommendation_ranking:
            data_used["orchestration_suppresses_aircraft"] = True
            data_used["recommend_aircraft_gated"] = 0

    if result.query_type == OrchestrationQueryTypeV2.EXPLICIT_COMPARISON:
        data_used["comparison_models_locked"] = list(result.preserve_comparison_models)
        data_used["tier_downgrade_blocked"] = "orchestration_v2_comparison"

    if not result.allow_tier_fallback:
        data_used["tier_downgrade_blocked"] = f"orchestration_v2_{result.query_type.value}"

    if not result.allow_operational_synthesis:
        data_used["kernel_synthesis_blocked"] = True

    if result.query_type == OrchestrationQueryTypeV2.NAMED_AIRCRAFT_CAPABILITY:
        data_used["candidate_models_locked"] = list(result.named_aircraft_models)


def load_orchestration_v2(
    data_used: Optional[Dict[str, Any]],
) -> Optional[OrchestrationRouterV2Result]:
    if not isinstance(data_used, dict):
        return None
    raw = data_used.get(ORCHESTRATION_V2_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        qt = OrchestrationQueryTypeV2(str(raw.get("query_type") or ""))
        ren = OrchestrationRendererV2(str(raw.get("renderer") or ""))
    except ValueError:
        return None
    return OrchestrationRouterV2Result(
        query_type=qt,
        renderer=ren,
        confidence=float(raw.get("confidence") or 0.5),
        signals=list(raw.get("signals") or []),
        authoritative=bool(raw.get("authoritative", True)),
        allow_recommendation_ranking=bool(raw.get("allow_recommendation_ranking")),
        allow_tier_fallback=bool(raw.get("allow_tier_fallback")),
        allow_operational_synthesis=bool(raw.get("allow_operational_synthesis")),
        preserve_comparison_models=tuple(raw.get("preserve_comparison_models") or []),
        named_aircraft_models=tuple(raw.get("named_aircraft_models") or []),
        suppress_generic_shortlist=bool(raw.get("suppress_generic_shortlist", True)),
        requires_deterministic_pipeline=bool(
            raw.get("requires_deterministic_pipeline", True)
        ),
        physics_first_priority=bool(raw.get("physics_first_priority", True)),
        routing_debug=dict(raw.get("routing_debug") or {}),
    )


def orchestration_v2_blocks_tier_fallback(data_used: Optional[Dict[str, Any]]) -> bool:
    loaded = load_orchestration_v2(data_used)
    if loaded is not None and loaded.authoritative:
        return not loaded.allow_tier_fallback
    if isinstance(data_used, dict) and data_used.get("tier_downgrade_blocked"):
        return True
    return False


def orchestration_v2_allows_operational_synthesis(
    data_used: Optional[Dict[str, Any]],
) -> bool:
    loaded = load_orchestration_v2(data_used)
    if loaded is not None and loaded.authoritative:
        return loaded.allow_operational_synthesis
    return False


def orchestration_v2_locked_comparison_models(
    data_used: Optional[Dict[str, Any]],
) -> List[str]:
    loaded = load_orchestration_v2(data_used)
    if loaded is not None and loaded.preserve_comparison_models:
        return list(loaded.preserve_comparison_models)
    if isinstance(data_used, dict):
        raw = data_used.get("comparison_models_locked") or data_used.get(
            "orchestration_v2_preserve_comparison_models"
        )
        if isinstance(raw, list):
            return [str(m) for m in raw if m]
    return []


__all__ = [
    "ORCHESTRATION_V2_KEY",
    "ORCHESTRATION_V2_QUERY_TYPE_KEY",
    "ROUTER_V2_FINAL_DECISION_KEY",
    "OrchestrationQueryTypeV2",
    "OrchestrationRendererV2",
    "OrchestrationRouterV2Result",
    "apply_orchestration_v2_metadata",
    "detect_hard_conflict_query",
    "load_orchestration_v2",
    "orchestration_v2_allows_operational_synthesis",
    "orchestration_v2_blocks_tier_fallback",
    "orchestration_v2_locked_comparison_models",
    "route_orchestration_v2",
]
