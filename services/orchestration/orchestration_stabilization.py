"""
Orchestration stabilization — professional guardrails against:
- fallback hallucinations after impossible structures
- renderer contamination (ops synthesis leaking into broker flows)
- hierarchy corruption (peak leg hijacking dominant utilization)
- continuation hub mis-weighting

When ``router_authoritative`` is True, the stabilizer MUST NOT change router query_type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.orchestration.orchestration_router_v2 import (
    OrchestrationQueryTypeV2,
    OrchestrationRendererV2,
    OrchestrationRouterV2Result,
)

ROUTER_AUTHORITATIVE = True

HARD_INVALID_KEY = "mission_hard_invalid"
STABILIZATION_KEY = "orchestration_stabilization"

_CONTINUATION_NODE_RE = re.compile(
    r"\b(?:dubai|dxb|singapore|sin|london\s+continuation|hawaii|hnl|refuel)\b",
    re.I,
)

_EXPLICIT_PRIMARY_FREQ_RE = re.compile(
    r"\b(?:primary|weekly|most\s+flights|majority\s+utilization|most\s+annual\s+utilization)\b",
    re.I,
)

_OCCASIONAL_RE = re.compile(r"\b(?:occasional|occasionally|quarterly|seasonal)\b", re.I)

_DOMESTIC_CORE_RE = re.compile(
    r"\b(?:most\s+annual\s+utilization|most\s+flight\s+hours|still\s+domestic|"
    r"domestic\s+north\s+america|dallas|houston|denver|atlanta|new\s+york|chicago)\b",
    re.I,
)

_IMPOSSIBLE_ULR_ECON_RE = re.compile(
    r"\b(?:"
    r"nonstop\s+winter\b.*\b(johannesburg|sydney)\b|"
    r"\b(johannesburg|sydney)\b.*\bnonstop\s+winter\b|"
    r"reliable\s+nonstop\b.*\b(tokyo)\b.*\bwinter\b.*\blow(?:er)?\s+operating\s+costs?\b|"
    r"\bbelow\s+global\s+7500\s+cost\b.*\b(nonstop|reliably)\b.*\b(hong\s+kong|hkg|tokyo|johannesburg|sydney)\b|"
    r"\b(hong\s+kong|hkg)\b.*\bnonstop\b.*\byear[- ]round\b.*\bbelow\s+global\s+7500\s+cost\b|"
    r"\blower\s+operating\s+cost\b.*\bthan\s+global\s+7500\b.*\bnonstop\b.*\b(tokyo|hkg|hong\s+kong)\b|"
    r"without\s+moving\s+up\s+to\s+airline-scale\s+costs"
    r")\b",
    re.I,
)


@dataclass(frozen=True)
class StabilizationResult:
    route: OrchestrationRouterV2Result
    response_mode: str
    suppress_operational_synthesis: bool
    continuation_leg_weight: float
    dominant_utilization_band: str
    peak_capability_requirement: str
    mission_hard_invalid: bool
    reference_aircraft: str = ""
    stabilizer_modified_route: bool = False

    def to_patch(self) -> Dict[str, Any]:
        return {
            STABILIZATION_KEY: {
                "response_mode": self.response_mode,
                "suppress_operational_synthesis": self.suppress_operational_synthesis,
                "continuation_leg_weight": self.continuation_leg_weight,
                "dominant_utilization_band": self.dominant_utilization_band,
                "peak_capability_requirement": self.peak_capability_requirement,
                "mission_hard_invalid": self.mission_hard_invalid,
                "reference_aircraft": self.reference_aircraft,
                "router_authoritative": ROUTER_AUTHORITATIVE,
                "stabilizer_modified_route": self.stabilizer_modified_route,
            },
            HARD_INVALID_KEY: bool(self.mission_hard_invalid),
            "router_authoritative": ROUTER_AUTHORITATIVE,
            "stabilizer_modified_route": self.stabilizer_modified_route,
            "hierarchy_weighting": {
                "dominant_utilization_band": self.dominant_utilization_band,
                "peak_capability_requirement": self.peak_capability_requirement,
                "continuation_leg_weight": self.continuation_leg_weight,
            },
        }


class OrchestrationStabilizer:
    """
    Stabilize permissions and hierarchy metadata without altering authoritative router type.
    """

    def stabilize(self, query: str, route: OrchestrationRouterV2Result) -> StabilizationResult:
        q = (query or "").strip()
        ql = q.lower()
        original_type = route.query_type
        stabilizer_modified_route = False

        mission_hard_invalid = self._detect_mission_hard_invalid(q)
        response_mode = route.query_type.value

        # Model hygiene only — collapse aliases; never change query_type when authoritative.
        canonical_models = self._canonicalize_models(
            list(route.preserve_comparison_models or []) + list(route.named_aircraft_models or [])
        )
        canonical_unique = [m for m in dict.fromkeys([m for m in canonical_models if m])]

        cleaned_comparison = tuple(canonical_unique) if route.preserve_comparison_models else ()
        cleaned_named = tuple(canonical_unique[:2]) if route.named_aircraft_models else ()

        if ROUTER_AUTHORITATIVE:
            route = OrchestrationRouterV2Result(
                query_type=route.query_type,
                renderer=route.renderer,
                confidence=route.confidence,
                signals=list(route.signals),
                authoritative=route.authoritative,
                allow_recommendation_ranking=route.allow_recommendation_ranking,
                allow_tier_fallback=route.allow_tier_fallback,
                allow_operational_synthesis=route.allow_operational_synthesis,
                preserve_comparison_models=cleaned_comparison or route.preserve_comparison_models,
                named_aircraft_models=cleaned_named or route.named_aircraft_models,
                suppress_generic_shortlist=route.suppress_generic_shortlist,
                requires_deterministic_pipeline=route.requires_deterministic_pipeline,
                physics_first_priority=route.physics_first_priority,
                routing_debug=dict(route.routing_debug),
            )
            stabilizer_modified_route = route.query_type != original_type
        else:
            # Legacy path disabled — router is always authoritative.
            stabilizer_modified_route = False

        continuation_leg_weight = 1.0
        if _CONTINUATION_NODE_RE.search(ql) and not _EXPLICIT_PRIMARY_FREQ_RE.search(ql):
            continuation_leg_weight = 0.25

        dominant_utilization_band, peak_capability_requirement = self._infer_hierarchy_bands(ql)

        suppress_ops = route.query_type in (
            OrchestrationQueryTypeV2.RECOMMENDATION_REQUEST,
            OrchestrationQueryTypeV2.EXPLICIT_COMPARISON,
            OrchestrationQueryTypeV2.STRATEGIC_FLEET_ANALYSIS,
        )

        return StabilizationResult(
            route=route,
            response_mode=response_mode,
            suppress_operational_synthesis=bool(suppress_ops and not route.allow_operational_synthesis),
            continuation_leg_weight=float(continuation_leg_weight),
            dominant_utilization_band=dominant_utilization_band,
            peak_capability_requirement=peak_capability_requirement,
            mission_hard_invalid=bool(mission_hard_invalid),
            reference_aircraft="",
            stabilizer_modified_route=stabilizer_modified_route,
        )

    def _detect_mission_hard_invalid(self, query: str) -> bool:
        ql = (query or "").lower()
        if not _IMPOSSIBLE_ULR_ECON_RE.search(ql):
            return False
        if "below global 7500 cost" in ql and "nonstop" in ql and (
            "hong kong" in ql or "hkg" in ql or "tokyo" in ql or "johannesburg" in ql or "sydney" in ql
        ):
            return True
        domestic_core = bool(_DOMESTIC_CORE_RE.search(ql))
        winter_ulr = bool(re.search(r"\bnonstop\b.*\bwinter\b", ql)) and bool(
            re.search(r"\b(johannesburg|sydney)\b", ql)
        )
        if (
            ("global 7500" in ql)
            and ("lower operating cost" in ql or "cheaper" in ql or "below" in ql)
            and ("los angeles" in ql or "la" in ql)
            and ("tokyo" in ql)
            and ("nonstop" in ql)
            and ("winter" in ql or "january" in ql)
        ):
            return True
        low_cost = (
            "airline-scale costs" in ql
            or "without moving up" in ql
            or "lower operating costs" in ql
            or "below global 7500 cost" in ql
        )
        return bool((domestic_core and low_cost) or (winter_ulr and low_cost))

    def _infer_hierarchy_bands(self, ql: str) -> tuple[str, str]:
        dom = "unresolved"
        peak = "unresolved"

        if _DOMESTIC_CORE_RE.search(ql):
            dom = "domestic_supermid"

        if re.search(r"\b(london|paris|geneva|frankfurt|zurich)\b", ql) and _OCCASIONAL_RE.search(ql):
            peak = "occasional_tatl"
        elif re.search(r"\b(tokyo|singapore|sydney|johannesburg)\b", ql) and _OCCASIONAL_RE.search(ql):
            peak = "occasional_ulr"

        if "hierarchy" in ql or "dominant" in ql:
            if dom == "unresolved" and _DOMESTIC_CORE_RE.search(ql):
                dom = "domestic_supermid"

        return dom, peak

    def _canonicalize_models(self, models: list[str]) -> list[str]:
        out: list[str] = []
        for m in models or []:
            s = str(m or "").strip()
            if not s:
                continue
            sl = s.lower()
            for pref in ("gulfstream ", "bombardier ", "embraer ", "dassault ", "cessna ", "textron "):
                if sl.startswith(pref):
                    s = s[len(pref) :].strip()
                    sl = s.lower()
                    break
            s = re.sub(r"\s+", " ", s).strip()
            out.append(s)
        return out


__all__ = [
    "OrchestrationStabilizer",
    "StabilizationResult",
    "ROUTER_AUTHORITATIVE",
    "HARD_INVALID_KEY",
    "STABILIZATION_KEY",
]
