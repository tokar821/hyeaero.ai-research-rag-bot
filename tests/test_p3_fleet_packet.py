"""P3 — fleet audit trace in immutable reasoning packet."""

from __future__ import annotations

from services.telemetry.fleet_packet_audit import (
    build_fleet_audit_trace,
    validate_fleet_audit_trace,
)
from services.telemetry.reasoning_packet import (
    IMMUTABLE_PACKET_KEY,
    PACKET_SCHEMA_VERSION,
    build_reasoning_packet_from_pipeline,
)
from services.telemetry.reasoning_packet_enforcement import (
    detect_single_aircraft_collapse,
    format_immutable_reasoning_packet_block,
    validate_packet_fleet_audit,
)


def _sample_fleet_plan() -> dict:
    return {
        "multi_aircraft_required": True,
        "multi_domain_required": True,
        "trigger": "elimination_failure",
        "single_aircraft_structurally_invalid": True,
        "universal_survivors": [],
        "doctrine": "Multi-domain operational problem.",
        "presented_models": ["Pilatus PC-24", "Global 7500"],
        "segments": [
            {
                "role": "short_field_high_performance",
                "route_labels": ["KASE → KTEX"],
            },
            {"role": "ulr_class", "route_labels": ["TEB → London"]},
        ],
        "assignments": [
            {
                "role": "short_field_high_performance",
                "segment_label": "Short-field / high-performance",
                "primary_model": "Pilatus PC-24",
                "fit_verdict": "VIABLE WITH COMPROMISES",
                "domain_feasible": True,
            },
            {
                "role": "ulr_class",
                "segment_label": "ULR class — TEB → London",
                "primary_model": "Global 7500",
                "fit_verdict": "PRIMARY RECOMMENDATION",
                "domain_feasible": True,
            },
        ],
        "domain_traces": [
            {
                "domain": "short_field_high_performance",
                "feasible_models": ["Pilatus PC-24", "Pilatus PC-12"],
                "eliminated_models": ["Citation Latitude"],
                "constraint_triggers": ["hot_high", "runway_length"],
                "corridor_classification": None,
                "corridor_decision": None,
                "payload_assumptions": {"passengers": 8},
                "elimination_lineage": [
                    {"stage": "airport_constraint", "model": "Citation Latitude", "reason": "field"}
                ],
            },
            {
                "domain": "ulr_class",
                "feasible_models": ["Global 7500", "Gulfstream G650"],
                "eliminated_models": ["Citation Latitude"],
                "constraint_triggers": ["corridor_transatlantic"],
                "corridor_classification": "transatlantic_executive",
                "corridor_decision": "verified_stage_nm=3100",
                "payload_assumptions": {"passengers": 8, "modifiers": ["winter_westbound"]},
                "elimination_lineage": [
                    {"stage": "corridor", "model": "Citation Latitude", "reason": "ULR band"}
                ],
            },
        ],
        "fleet_invariant": {"ok": True, "violations": [], "stripped_models": []},
    }


def test_build_fleet_audit_trace_merges_segments():
    audit = build_fleet_audit_trace(_sample_fleet_plan())
    assert audit["multi_domain_required"]
    assert audit["trigger"] == "elimination_failure"
    assert len(audit["segments"]) == 2
    ulr = next(s for s in audit["segments"] if s["domain"] == "ulr_class")
    assert ulr["corridor_classification"] == "transatlantic_executive"
    assert ulr["payload_assumptions"].get("modifiers")
    assert ulr["primary_model"] == "Global 7500"
    mountain = next(s for s in audit["segments"] if s["domain"] == "short_field_high_performance")
    assert mountain["route_labels"] == ["KASE → KTEX"]
    assert mountain["elimination_lineage"]


def test_validate_fleet_audit_detects_domain_eliminated_assignment():
    audit = build_fleet_audit_trace(_sample_fleet_plan())
    audit["segments"][0]["primary_model"] = "Citation Latitude"
    issues = validate_fleet_audit_trace(audit)
    assert any("eliminated" in i for i in issues)


def test_reasoning_packet_includes_fleet_audit_v2():
    du = {
        "fleet_composition_plan": _sample_fleet_plan(),
        "route_distance_authority": [
            {"route_label": "TEB → London", "source": "catalog", "confidence": 0.9}
        ],
        "mission_operational_context": {
            "corridor_id": "transatlantic_executive",
            "payload": {"passengers": 8},
            "reserve": {"planning_mode": "conservative"},
        },
    }
    packet = build_reasoning_packet_from_pipeline(data_used=du, recommendations=[])
    d = packet.to_dict()
    assert d["schema_version"] == PACKET_SCHEMA_VERSION
    assert d["fleet_audit"]["segments"]
    assert validate_packet_fleet_audit(d) == []
    assert any(e["stage"].startswith("fleet_domain_") for e in d["eliminations"])


def test_format_block_includes_per_domain_payload_and_corridor():
    du = {IMMUTABLE_PACKET_KEY: build_reasoning_packet_from_pipeline(
        data_used={"fleet_composition_plan": _sample_fleet_plan()}
    ).to_dict()}
    block = format_immutable_reasoning_packet_block(du[IMMUTABLE_PACKET_KEY])
    assert "DOMAIN SEGMENT AUDIT" in block
    assert "ulr_class" in block
    assert "corridor" in block.lower()
    assert "payload" in block.lower()


def test_pipeline_attaches_fleet_audit_packet():
    from evals.adversarial_mission_suite import HAND_CRAFTED, _run

    case = next(c for c in HAND_CRAFTED if c.case_id == "fleet_teb_lon_aspen")
    ctx = _run(case.query)
    packet = ctx["reasoning_packet"]
    assert packet.get("schema_version") == 2
    assert packet.get("fleet_audit", {}).get("segments")
    assert validate_packet_fleet_audit(packet) == []


def test_detect_single_aircraft_collapse():
    packet = {
        "fleet_audit": build_fleet_audit_trace(_sample_fleet_plan()),
    }
    bad = "Use one aircraft for everything — the G650 covers London and Aspen."
    assert detect_single_aircraft_collapse(bad, packet)
