"""
P3 — Immutable reasoning packet enforcement at LLM and formatter boundaries.

The deterministic pipeline owns:
  - which aircraft may be presented
  - broker verdict per model
  - elimination trace

Downstream prose may narrate only; it must not contradict the packet.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from services.broker.broker_verdicts import BrokerVerdict, normalize_broker_verdict
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.telemetry.reasoning_packet import IMMUTABLE_PACKET_KEY

logger = logging.getLogger(__name__)

_VERDICT_RANK: Dict[BrokerVerdict, int] = {
    BrokerVerdict.NOT_OPERATIONALLY_CREDIBLE: 0,
    BrokerVerdict.MISSION_RISKY: 1,
    BrokerVerdict.VIABLE_WITH_COMPROMISES: 2,
    BrokerVerdict.PRIMARY_RECOMMENDATION: 3,
}

_NEGATION_RE = re.compile(
    r"\b(?:ruled\s+out|eliminated|not\s+recommend|do\s+not\s+recommend|avoid|"
    r"don'?t\s+(?:use|choose|pick)|instead\s+of|rather\s+than|was\s+removed|"
    r"doesn'?t\s+(?:fit|work)|not\s+(?:a\s+)?fit)\b",
    re.I,
)

_PRIMARY_LANGUAGE_RE = re.compile(
    r"\b(?:"
    r"primary\s+recommendation|best\s+fit|best\s+option|clear\s+winner|"
    r"i(?:'|')?d\s+start\s+with|lead\s+with|top\s+pick|strongest\s+fit|"
    r"first\s+choice|go-to\s+choice|hands-down"
    r")\b",
    re.I,
)

_EXPLICIT_VERDICT_RE = re.compile(
    r"(?P<model>[A-Za-z0-9][\w\s\-]{2,40}?)\s*"
    r"(?:—|–|-|:)\s*"
    r"(?P<verdict>PRIMARY RECOMMENDATION|VIABLE WITH COMPROMISES|MISSION-RISKY|"
    r"NOT OPERATIONALLY CREDIBLE|BEST FIT|GOOD FIT|CONDITIONAL FIT|NOT A FIT)",
    re.I,
)


def _norm(model: str) -> str:
    return re.sub(r"\s+", " ", (model or "").strip().lower())


def extract_reasoning_packet(data_used: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(data_used, dict):
        return None
    packet = data_used.get(IMMUTABLE_PACKET_KEY)
    if isinstance(packet, dict) and packet.get("immutable"):
        return packet
    return None


def packet_presented_models(packet: Dict[str, Any]) -> Set[str]:
    out = {_norm(m) for m in (packet.get("presented_models") or []) if m}
    audit = packet_fleet_audit(packet)
    if audit.get("multi_domain_required"):
        for m in audit.get("presented_models") or []:
            if m:
                out.add(_norm(str(m)))
        for seg in audit.get("segments") or []:
            if isinstance(seg, dict) and seg.get("primary_model"):
                out.add(_norm(str(seg["primary_model"])))
    return out


def packet_eliminated_models(packet: Dict[str, Any]) -> Set[str]:
    out: Set[str] = {_norm(m) for m in (packet.get("eliminated_models") or []) if m}
    for entry in packet.get("eliminations") or []:
        if isinstance(entry, dict):
            m = entry.get("model") or ""
            if m:
                out.add(_norm(m))
    return out


def packet_fleet_audit(packet: Dict[str, Any]) -> Dict[str, Any]:
    audit = packet.get("fleet_audit")
    if isinstance(audit, dict) and audit.get("segments"):
        return audit
    fleet = packet.get("fleet_composition") or {}
    if isinstance(fleet, dict) and fleet.get("domain_traces"):
        try:
            from services.telemetry.fleet_packet_audit import build_fleet_audit_trace

            return build_fleet_audit_trace(fleet)
        except Exception:
            pass
    return {}


def validate_packet_fleet_audit(packet: Dict[str, Any]) -> List[str]:
    """Validate fleet audit section of immutable packet — empty if OK."""
    audit = packet_fleet_audit(packet)
    if not audit.get("multi_domain_required"):
        return []
    try:
        from services.telemetry.fleet_packet_audit import validate_fleet_audit_trace

        return validate_fleet_audit_trace(audit)
    except Exception as exc:
        return [f"fleet_audit_validation_error:{exc}"]


_SINGLE_AIRCRAFT_COLLAPSE_RE = re.compile(
    r"\b(?:"
    r"one\s+(?:aircraft|jet|plane)\s+(?:for|to\s+cover|covers?|handles?|can\s+do)\s+"
    r"(?:everything|all\s+(?:legs|domains|segments)|both|the\s+(?:whole|entire)\s+mission)|"
    r"single\s+aircraft\s+(?:solution|covers?|for\s+all)|"
    r"same\s+(?:jet|aircraft)\s+(?:for|on)\s+(?:both|all)"
    r")\b",
    re.I,
)


def detect_single_aircraft_collapse(text: str, packet: Dict[str, Any]) -> bool:
    """
    True when prose collapses a structurally invalid multi-domain mission to one aircraft.
    """
    audit = packet_fleet_audit(packet)
    if not audit.get("single_aircraft_structurally_invalid"):
        return False
    if not (text or "").strip():
        return False
    if _SINGLE_AIRCRAFT_COLLAPSE_RE.search(text):
        return True
    # Recommend one named model as sole solution without domain framing
    if audit.get("presented_models") and len(audit.get("presented_models") or []) >= 2:
        try:
            from services.consultant.recommendation_engine import detect_models_from_text
        except Exception:
            return False
        mentioned = detect_models_from_text(text)
        if len(mentioned) == 1 and not re.search(
            r"\b(?:domain|segment|leg|ulr|mountain|caribbean|short-field)\b", text, re.I
        ):
            if _PRIMARY_LANGUAGE_RE.search(text):
                return True
    return False


def packet_verdict_sources(packet: Dict[str, Any]) -> Dict[str, str]:
    raw = packet.get("verdict_sources") or {}
    if not isinstance(raw, dict):
        return {}
    return {_norm(k): str(v) for k, v in raw.items() if k}


def authorized_verdict_map(
    packet: Optional[Dict[str, Any]],
    recommendations: Sequence[AircraftRecommendation],
) -> Dict[str, BrokerVerdict]:
    """Model key → authoritative broker verdict."""
    out: Dict[str, BrokerVerdict] = {}
    if packet:
        for model, label in packet_verdict_sources(packet).items():
            out[model] = normalize_broker_verdict(label)
    for rec in recommendations:
        if rec.avoid:
            continue
        key = _norm(rec.model)
        if key not in out:
            fv = (rec.fit_verdict or rec.fit or "").strip()
            out[key] = normalize_broker_verdict(fv) if fv else BrokerVerdict.VIABLE_WITH_COMPROMISES
    return out


def _verdict_rank(verdict: BrokerVerdict) -> int:
    return _VERDICT_RANK.get(verdict, 2)


def _stated_verdict_rank(label: str) -> int:
    return _verdict_rank(normalize_broker_verdict(label))


@dataclass
class ReasoningPacketEnforcementReport:
    ok: bool
    regenerated: bool = False
    unauthorized_models: List[str] = field(default_factory=list)
    eliminated_mentions: List[str] = field(default_factory=list)
    verdict_upgrades: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "regenerated": self.regenerated,
            "unauthorized_models": list(self.unauthorized_models),
            "eliminated_mentions": list(self.eliminated_mentions),
            "verdict_upgrades": list(self.verdict_upgrades),
            "issues": list(self.issues),
        }


def detect_eliminated_mentions(text: str, eliminated: Set[str]) -> List[str]:
    """Flag eliminated models only when prose recommends them — not negation/explanation."""
    if not eliminated or not (text or "").strip():
        return []
    try:
        from services.consultant.recommendation_engine import detect_models_from_text
    except Exception:
        return []
    mentioned = detect_models_from_text(text)
    hits: List[str] = []
    for model in mentioned:
        key = _norm(model)
        if key not in eliminated:
            continue
        pattern = re.compile(re.escape(model), re.I)
        for m in pattern.finditer(text):
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            window = text[start:end]
            if _NEGATION_RE.search(window):
                continue
            line_start = text.rfind("\n", 0, m.start()) + 1
            line = text[line_start : text.find("\n", m.end()) if text.find("\n", m.end()) >= 0 else len(text)]
            if line.strip().startswith(("-", "•")) and not _NEGATION_RE.search(line):
                hits.append(model)
                break
            if _PRIMARY_LANGUAGE_RE.search(window):
                hits.append(model)
                break
    return hits


def detect_unauthorized_from_packet(
    text: str,
    packet: Dict[str, Any],
    *,
    comparison_models: Optional[Sequence[str]] = None,
) -> List[str]:
    """Models named in prose but not in presented_models."""
    presented = packet_presented_models(packet)
    fleet = packet.get("fleet_composition") or {}
    if isinstance(fleet, dict) and fleet.get("multi_aircraft_required"):
        for m in fleet.get("presented_models") or []:
            if m:
                presented.add(_norm(str(m)))
    if not presented or not (text or "").strip():
        return []
    allowed = set(presented)
    if comparison_models:
        allowed.update(_norm(m) for m in comparison_models if m)
    try:
        from services.consultant.recommendation_engine import detect_models_from_text
    except Exception:
        return []
    mentioned = detect_models_from_text(text)
    return [m for m in mentioned if _norm(m) not in allowed]


def detect_verdict_upgrades(
    text: str,
    verdict_map: Dict[str, BrokerVerdict],
) -> List[str]:
    """
    Flag models where prose uses stronger recommendation language than the packet allows.
    """
    if not verdict_map or not (text or "").strip():
        return []

    issues: List[str] = []
    try:
        from services.consultant.recommendation_engine import detect_models_from_text
    except Exception:
        detect_models_from_text = None  # type: ignore

    for match in _EXPLICIT_VERDICT_RE.finditer(text):
        model_key = _norm(match.group("model"))
        stated = match.group("verdict")
        authorized = verdict_map.get(model_key)
        if authorized is None:
            continue
        if _stated_verdict_rank(stated) > _verdict_rank(authorized):
            issues.append(
                f"{match.group('model').strip()}: stated {stated} > authorized {authorized.value}"
            )

    if detect_models_from_text:
        mentioned = detect_models_from_text(text)
        lower = text.lower()
        for model in mentioned:
            key = _norm(model)
            authorized = verdict_map.get(key)
            if authorized is None:
                continue
            if _verdict_rank(authorized) >= _verdict_rank(BrokerVerdict.VIABLE_WITH_COMPROMISES):
                continue
            # Find window around model mention
            pattern = re.compile(re.escape(model), re.I)
            for m in pattern.finditer(text):
                start = max(0, m.start() - 120)
                end = min(len(text), m.end() + 120)
                window = text[start:end]
                if _PRIMARY_LANGUAGE_RE.search(window):
                    issues.append(
                        f"{model}: primary language near model but authorized {authorized.value}"
                    )
                    break
            if model.lower() in lower and _PRIMARY_LANGUAGE_RE.search(lower):
                # Global primary language + risky model mentioned anywhere
                if _verdict_rank(authorized) <= _verdict_rank(BrokerVerdict.MISSION_RISKY):
                    if any(
                        i.startswith(f"{model}:")
                        for i in issues
                    ):
                        continue
                    if re.search(
                        rf"\b{re.escape(model)}\b.{{0,80}}(?:primary|best fit|start with)",
                        text,
                        re.I | re.S,
                    ):
                        issues.append(
                            f"{model}: upgraded tone vs authorized {authorized.value}"
                        )
    return issues


def format_immutable_reasoning_packet_block(packet: Dict[str, Any]) -> str:
    """Mandatory LLM context — narrate only; never contradict."""
    lines = [
        "[IMMUTABLE REASONING PACKET — audit trace; narrate prose only]",
        "You MUST NOT add aircraft, remove presented aircraft, or upgrade broker verdicts.",
        "Eliminated aircraft may be mentioned only to explain why they were ruled out.",
        "",
    ]

    presented = packet.get("presented_models") or []
    if presented:
        lines.append("PRESENTED (narrate as recommendations only): " + ", ".join(presented))

    eliminated = sorted(packet_eliminated_models(packet))
    if eliminated:
        lines.append("ELIMINATED (never recommend): " + ", ".join(eliminated))

    verdicts = packet.get("verdict_sources") or {}
    if verdicts:
        lines.append("")
        lines.append("VERDICT SOURCES (do not upgrade):")
        for model, verdict in verdicts.items():
            lines.append(f"  - {model}: {verdict}")

    routes = packet.get("route_sources") or []
    if routes:
        lines.append("")
        lines.append("ROUTE AUTHORITY:")
        for r in routes[:3]:
            if not isinstance(r, dict):
                continue
            lines.append(
                f"  - {r.get('route_label', '?')}: {r.get('distance_nm')} nm "
                f"({r.get('source')}, conf={r.get('confidence')})"
            )

    corridor = packet.get("corridor_classification")
    if corridor:
        lines.append(f"Corridor: {corridor}")

    payload = packet.get("payload_assumptions") or {}
    if payload:
        mods = payload.get("modifiers") or []
        lines.append(
            f"Payload assumptions: {payload.get('passengers', '?')} pax"
            + (f", modifiers={', '.join(mods)}" if mods else "")
        )

    reserve = packet.get("reserve_profile") or {}
    if reserve.get("planning_mode"):
        lines.append(
            f"Reserve profile: {reserve.get('planning_mode')} "
            f"(required ~{reserve.get('total_required_nm')} nm)"
        )

    dispatch = packet.get("dispatch_summary") or {}
    unreliable = dispatch.get("technically_possible_not_reliable") or []
    if unreliable:
        lines.append(
            "Dispatch caution (technically possible, not reliably dispatchable): "
            + ", ".join(unreliable)
        )

    audit = packet_fleet_audit(packet)
    fleet = packet.get("fleet_composition") or {}
    if audit.get("multi_domain_required") or (
        isinstance(fleet, dict)
        and (fleet.get("multi_aircraft_required") or fleet.get("multi_domain_required"))
    ):
        lines.append("")
        if audit.get("single_aircraft_structurally_invalid") or fleet.get(
            "single_aircraft_structurally_invalid"
        ):
            lines.append(
                "STRUCTURAL: one aircraft spanning all domains is INVALID — not less optimal."
            )
        trigger = audit.get("trigger") or fleet.get("trigger") or "unknown"
        lines.append(f"Segmentation trigger: {trigger} (not preference)")
        lines.append(
            "MULTI-DOMAIN COMPOSITION (narrate each domain independently; never collapse to one aircraft):"
        )
        segments = audit.get("segments") or []
        if segments:
            lines.append("")
            lines.append("DOMAIN SEGMENT AUDIT (authoritative — do not contradict):")
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                dom = seg.get("domain", "?")
                label = seg.get("segment_label", dom)
                primary = seg.get("primary_model") or "(no survivor)"
                lines.append(f"  [{dom}] {label}")
                lines.append(f"    Primary: {primary} ({seg.get('fit_verdict', '')})")
                routes = seg.get("route_labels") or []
                if routes:
                    lines.append(f"    Routes: {', '.join(routes)}")
                triggers = ", ".join(seg.get("constraint_triggers") or [])
                if triggers:
                    lines.append(f"    Constraint triggers: {triggers}")
                if seg.get("corridor_classification"):
                    lines.append(
                        f"    Corridor: {seg['corridor_classification']}"
                        + (
                            f" — {seg['corridor_decision']}"
                            if seg.get("corridor_decision")
                            else ""
                        )
                    )
                payload = seg.get("payload_assumptions") or {}
                if payload:
                    mods = payload.get("modifiers") or []
                    lines.append(
                        f"    Payload: {payload.get('passengers', '?')} pax"
                        + (f", {', '.join(mods)}" if mods else "")
                    )
                lineage = seg.get("elimination_lineage") or []
                if lineage:
                    sample = lineage[:4]
                    for entry in sample:
                        if isinstance(entry, dict):
                            lines.append(
                                f"    Eliminated: {entry.get('model')} "
                                f"({entry.get('stage')}: {entry.get('reason', '')})"
                            )
        else:
            for a in fleet.get("assignments") or []:
                if not isinstance(a, dict):
                    continue
                lines.append(
                    f"  - {a.get('segment_label', '?')}: {a.get('primary_model')} "
                    f"({a.get('fit_verdict', '')})"
                )
        inv = audit.get("fleet_invariant") or fleet.get("fleet_invariant") or {}
        if isinstance(inv, dict) and inv.get("ok") is False:
            lines.append(f"Fleet invariant violations: {'; '.join(inv.get('violations') or [])}")

    conf = packet.get("confidence") or {}
    if conf:
        lines.append(
            f"Confidence: route={conf.get('route_confidence')} "
            f"payload={conf.get('payload_confidence')} "
            f"dispatch={conf.get('dispatch_reliability')}"
        )

    return "\n".join(lines).strip()


def enforce_reasoning_packet_authority(
    answer: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    recommendations: Optional[Sequence[AircraftRecommendation]] = None,
    mission: Optional[MissionState] = None,
    route_assessments: Optional[Sequence[Any]] = None,
    comparison_models: Optional[Sequence[str]] = None,
    query: str = "",
    turn_seed: str = "",
) -> Tuple[str, ReasoningPacketEnforcementReport]:
    """
    Validate LLM/formatter prose against ``hye_reasoning_packet``.

    On violation, regenerate from structured formatter (deterministic).
    """
    packet = extract_reasoning_packet(data_used)
    recs = [r for r in (recommendations or []) if not getattr(r, "avoid", False)]
    report = ReasoningPacketEnforcementReport(ok=True)

    if not packet or not (answer or "").strip():
        return answer, report

    eliminated = packet_eliminated_models(packet)
    report.eliminated_mentions = detect_eliminated_mentions(answer, eliminated)
    report.unauthorized_models = detect_unauthorized_from_packet(
        answer, packet, comparison_models=comparison_models
    )

    verdict_map = authorized_verdict_map(packet, recs)
    report.verdict_upgrades = detect_verdict_upgrades(answer, verdict_map)

    fleet_issues = validate_packet_fleet_audit(packet)
    if fleet_issues:
        report.issues.extend(f"packet_fleet_audit:{i}" for i in fleet_issues[:4])

    if detect_single_aircraft_collapse(answer, packet):
        report.issues.append("single_aircraft_collapse_on_multi_domain_mission")

    if report.eliminated_mentions:
        report.issues.append(
            "eliminated_aircraft_mentioned:" + ",".join(report.eliminated_mentions)
        )
    if report.unauthorized_models:
        report.issues.append(
            "unauthorized_aircraft:" + ",".join(report.unauthorized_models)
        )
    if report.verdict_upgrades:
        report.issues.extend(report.verdict_upgrades)

    if not (
        report.eliminated_mentions
        or report.unauthorized_models
        or report.verdict_upgrades
        or any(
            i.startswith("single_aircraft_collapse") or i.startswith("packet_fleet_audit:")
            for i in report.issues
        )
    ):
        return answer, report

    report.ok = False
    logger.warning(
        "REASONING_PACKET_ENFORCEMENT violations=%s — regenerating from pipeline",
        report.issues[:6],
    )

    if not mission or not recs:
        return answer, report

    presented = packet_presented_models(packet)
    if presented:
        recs = [r for r in recs if _norm(r.model) in presented]
    if not recs:
        return answer, report

    regenerated = ""
    try:
        from services.consultant.broker_advisory_layer import format_broker_advisory_response

        regenerated = format_broker_advisory_response(
            mission,
            list(recs),
            route_assessments=list(route_assessments or []),
            data_used=data_used,
            eliminated_models=list(eliminated),
        )
    except Exception:
        regenerated = ""

    if not (regenerated or "").strip():
        from services.consultant.response_formatter import format_consultant_response

        regen_du: Optional[Dict[str, Any]] = None
        if isinstance(data_used, dict) and IMMUTABLE_PACKET_KEY in data_used:
            regen_du = {IMMUTABLE_PACKET_KEY: data_used[IMMUTABLE_PACKET_KEY]}
        regenerated = format_consultant_response(
            mission=mission,
            recommendations=list(recs),
            route_assessments=list(route_assessments or []),
            query=query,
            turn_seed=turn_seed or query,
            data_used=regen_du,
            eliminated_models=list(eliminated),
        )
    report.regenerated = True

    # Re-check — if still bad, keep regenerated (formatter is authoritative)
    report.eliminated_mentions = detect_eliminated_mentions(regenerated, eliminated)
    report.unauthorized_models = detect_unauthorized_from_packet(
        regenerated, packet, comparison_models=comparison_models
    )
    report.verdict_upgrades = detect_verdict_upgrades(regenerated, verdict_map)
    report.ok = not (
        report.eliminated_mentions
        or report.unauthorized_models
        or report.verdict_upgrades
        or detect_single_aircraft_collapse(regenerated, packet)
    )
    if report.ok:
        report.issues = ["regenerated_from_immutable_packet"]
    return regenerated, report
