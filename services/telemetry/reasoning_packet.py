"""
Immutable reasoning packet — standardized elimination trace for audit / QA / enterprise trust.

Downstream LLM may narrate prose only; must not mutate this structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

IMMUTABLE_PACKET_KEY = "hye_reasoning_packet"
PACKET_SCHEMA_VERSION = 2


@dataclass
class EliminationRecord:
    stage: str
    model: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"stage": self.stage, "model": self.model, "reason": self.reason}


@dataclass
class ConfidenceBreakdown:
    route_confidence: float = 0.0
    payload_confidence: float = 0.0
    dispatch_reliability: float = 0.0
    catalog_nonstop_authorized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_confidence": round(self.route_confidence, 3),
            "payload_confidence": round(self.payload_confidence, 3),
            "dispatch_reliability": round(self.dispatch_reliability, 3),
            "catalog_nonstop_authorized": self.catalog_nonstop_authorized,
        }


@dataclass
class ImmutableReasoningPacket:
    """
    Frozen advisory trace — formatter and LLM consume but must not rewrite aircraft lists.
    """

    schema_version: int = PACKET_SCHEMA_VERSION
    route_sources: List[Dict[str, Any]] = field(default_factory=list)
    corridor_classification: Optional[str] = None
    payload_assumptions: Dict[str, Any] = field(default_factory=dict)
    reserve_profile: Dict[str, Any] = field(default_factory=dict)
    eliminations: List[EliminationRecord] = field(default_factory=list)
    presented_models: List[str] = field(default_factory=list)
    eliminated_models: List[str] = field(default_factory=list)
    confidence: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)
    verdict_sources: Dict[str, str] = field(default_factory=dict)
    aircraft_operational: List[Dict[str, Any]] = field(default_factory=list)
    dispatch_summary: Dict[str, Any] = field(default_factory=dict)
    fleet_composition: Dict[str, Any] = field(default_factory=dict)
    fleet_audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "immutable": True,
            "route_sources": list(self.route_sources),
            "corridor_classification": self.corridor_classification,
            "payload_assumptions": dict(self.payload_assumptions),
            "reserve_profile": dict(self.reserve_profile),
            "eliminations": [e.to_dict() for e in self.eliminations],
            "presented_models": list(self.presented_models),
            "eliminated_models": list(self.eliminated_models),
            "confidence": self.confidence.to_dict(),
            "verdict_sources": dict(self.verdict_sources),
            "aircraft_operational": list(self.aircraft_operational),
            "dispatch_summary": dict(self.dispatch_summary),
            "fleet_composition": dict(self.fleet_composition),
            "fleet_audit": dict(self.fleet_audit),
        }


class ReasoningPacketBuilder:
    """Accumulate pipeline stages into one immutable packet."""

    def __init__(self) -> None:
        self._packet = ImmutableReasoningPacket()

    def set_route_authority(self, resolutions: Sequence[Dict[str, Any]]) -> None:
        self._packet.route_sources = list(resolutions)

    def set_corridor(self, corridor_id: Optional[str]) -> None:
        self._packet.corridor_classification = corridor_id

    def set_payload(self, payload_dict: Dict[str, Any]) -> None:
        self._packet.payload_assumptions = dict(payload_dict)
        self._packet.confidence.payload_confidence = 0.85 if payload_dict else 0.5

    def set_reserve(self, reserve_dict: Dict[str, Any]) -> None:
        self._packet.reserve_profile = dict(reserve_dict)

    def add_eliminations_from_block(self, block: Optional[Dict[str, Any]], stage: str) -> None:
        if not isinstance(block, dict):
            return
        reasons = block.get("elimination_reasons") or block.get("reasons") or {}
        if isinstance(reasons, dict):
            for model, reason in reasons.items():
                self._packet.eliminations.append(
                    EliminationRecord(stage=stage, model=str(model), reason=str(reason))
                )
        for model in block.get("eliminated") or []:
            if str(model) not in self._packet.eliminated_models:
                self._packet.eliminated_models.append(str(model))

    def add_elimination_log(self, log: Sequence[Dict[str, Any]]) -> None:
        for entry in log or []:
            if not isinstance(entry, dict):
                continue
            model = str(entry.get("aircraft_name") or entry.get("model") or "")
            reason = str(entry.get("reason") or entry.get("summary") or "")
            stage = str(entry.get("stage") or "pipeline")
            if model:
                self._packet.eliminations.append(
                    EliminationRecord(stage=stage, model=model, reason=reason)
                )

    def set_presented(
        self,
        recommendations: Sequence[Any],
        *,
        verdict_source: str = "operational_assessment",
    ) -> None:
        self._packet.presented_models = []
        self._packet.verdict_sources = {}
        for rec in recommendations:
            model = getattr(rec, "model", None) or (rec.get("model") if isinstance(rec, dict) else "")
            if not model:
                continue
            self._packet.presented_models.append(str(model))
            fv = getattr(rec, "fit_verdict", None) or (
                rec.get("fit_verdict") if isinstance(rec, dict) else ""
            )
            self._packet.verdict_sources[str(model)] = str(fv or verdict_source)

    def set_fleet_plan(self, fleet_plan: Optional[Dict[str, Any]]) -> None:
        """Attach normalized multi-domain audit trace (P3)."""
        if not isinstance(fleet_plan, dict):
            return
        from services.telemetry.fleet_packet_audit import build_fleet_audit_trace

        self._packet.fleet_composition = dict(fleet_plan)
        audit = build_fleet_audit_trace(fleet_plan)
        self._packet.fleet_audit = audit
        existing = {
            (e.stage, e.model)
            for e in self._packet.eliminations
        }
        for seg in audit.get("segments") or []:
            if not isinstance(seg, dict):
                continue
            domain = str(seg.get("domain") or "unknown")
            for entry in seg.get("elimination_lineage") or []:
                if not isinstance(entry, dict) or not entry.get("model"):
                    continue
                model = str(entry["model"])
                stage = f"fleet_domain_{domain}"
                key = (stage, model)
                if key in existing:
                    continue
                existing.add(key)
                self._packet.eliminations.append(
                    EliminationRecord(
                        stage=stage,
                        model=model,
                        reason=str(entry.get("reason") or ""),
                    )
                )

    def set_aircraft_operational(self, assessments: Sequence[Dict[str, Any]]) -> None:
        self._packet.aircraft_operational = list(assessments)
        if assessments:
            scores = [
                float(a.get("dispatch", {}).get("reliability_score", 0))
                for a in assessments
                if isinstance(a.get("dispatch"), dict)
            ]
            if scores:
                self._packet.confidence.dispatch_reliability = sum(scores) / len(scores)

    def set_route_confidence(self, resolutions: Sequence[Dict[str, Any]]) -> None:
        if resolutions:
            self._packet.confidence.route_confidence = min(
                float(r.get("confidence") or 0) for r in resolutions
            )
            self._packet.confidence.catalog_nonstop_authorized = all(
                r.get("authorize_nonstop_feasibility") is not False
                for r in resolutions
                if r.get("source") == "catalog"
            ) or any(r.get("source") == "catalog" for r in resolutions)

    def build(self) -> ImmutableReasoningPacket:
        return self._packet


def attach_reasoning_packet(data_used: Dict[str, Any], packet: ImmutableReasoningPacket) -> None:
    data_used[IMMUTABLE_PACKET_KEY] = packet.to_dict()


def build_reasoning_packet_from_pipeline(
    *,
    data_used: Optional[Dict[str, Any]] = None,
    recommendations: Optional[Sequence[Any]] = None,
    operational_context: Optional[Dict[str, Any]] = None,
    aircraft_operational: Optional[Sequence[Dict[str, Any]]] = None,
    elimination_log: Optional[Sequence[Dict[str, Any]]] = None,
) -> ImmutableReasoningPacket:
    """Assemble packet from orchestration artifacts."""
    du = data_used or {}
    b = ReasoningPacketBuilder()

    routes = du.get("route_distance_authority") or []
    b.set_route_authority(routes)
    b.set_route_confidence(routes)

    if isinstance(operational_context, dict):
        b.set_corridor(operational_context.get("corridor_id"))
        b.set_payload(operational_context.get("payload") or {})
        b.set_reserve(operational_context.get("reserve") or {})

    for key, stage in (
        ("corridor_hard_elimination", "corridor"),
        ("airport_constraint_elimination", "airport"),
        ("mountain_field_elimination", "mountain"),
        ("operational_band_elimination", "operational_band"),
    ):
        b.add_eliminations_from_block(du.get(key), stage)

    b.add_elimination_log(elimination_log or du.get("elimination_log") or [])

    pipe = du.get("deterministic_recommendation_pipeline") or {}
    for m in pipe.get("eliminated_models") or []:
        if str(m) not in b._packet.eliminated_models:
            b._packet.eliminated_models.append(str(m))

    fleet = du.get("fleet_composition_plan")
    if isinstance(fleet, dict) and (
        fleet.get("multi_aircraft_required") or fleet.get("multi_domain_required")
    ):
        b.set_fleet_plan(fleet)

    if recommendations:
        b.set_presented(recommendations, verdict_source="p1_operational")

    if isinstance(fleet, dict) and fleet.get("multi_aircraft_required"):
        fleet_models = fleet.get("presented_models") or []
        if fleet_models:
            b._packet.presented_models = list(fleet_models)
            b._packet.verdict_sources = {}
            for a in fleet.get("assignments") or []:
                if isinstance(a, dict):
                    model = str(a.get("primary_model") or "")
                    verdict = str(a.get("fit_verdict") or "VIABLE WITH COMPROMISES")
                    if model:
                        b._packet.verdict_sources[model] = verdict

    if aircraft_operational:
        b.set_aircraft_operational(aircraft_operational)
        unreliable = [
            a.get("model")
            for a in aircraft_operational
            if isinstance(a, dict)
            and isinstance(a.get("dispatch"), dict)
            and a["dispatch"].get("technically_possible")
            and not a["dispatch"].get("works_reliably")
        ]
        b._packet.dispatch_summary = {
            "technically_possible_not_reliable": unreliable,
            "assessment_count": len(aircraft_operational),
        }

    return b.build()
