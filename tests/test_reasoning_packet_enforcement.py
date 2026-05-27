"""P3 — immutable reasoning packet enforcement at LLM/formatter boundaries."""

from __future__ import annotations

from services.broker.broker_verdicts import BrokerVerdict
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation, RecommendationExplanation
from services.telemetry.reasoning_packet import IMMUTABLE_PACKET_KEY, attach_reasoning_packet
from services.telemetry.fleet_packet_audit import build_fleet_audit_trace
from services.telemetry.reasoning_packet_enforcement import (
    detect_eliminated_mentions,
    detect_single_aircraft_collapse,
    detect_unauthorized_from_packet,
    detect_verdict_upgrades,
    enforce_reasoning_packet_authority,
    format_immutable_reasoning_packet_block,
    authorized_verdict_map,
)


def _packet(**overrides):
    base = {
        "immutable": True,
        "schema_version": 1,
        "presented_models": ["Gulfstream G650", "Bombardier Global 7500"],
        "eliminated_models": ["g280"],
        "verdict_sources": {
            "Gulfstream G650": BrokerVerdict.PRIMARY_RECOMMENDATION.value,
            "Bombardier Global 7500": BrokerVerdict.VIABLE_WITH_COMPROMISES.value,
            "Citation Latitude": BrokerVerdict.MISSION_RISKY.value,
        },
        "eliminations": [{"stage": "corridor", "model": "g280", "reason": "ULR band"}],
        "route_sources": [{"route_label": "TEB → London", "source": "catalog", "confidence": 0.95}],
        "corridor_classification": "transatlantic_executive",
        "payload_assumptions": {"passengers": 8, "modifiers": ["ski"]},
        "reserve_profile": {"planning_mode": "conservative"},
    }
    base.update(overrides)
    return base


def _rec(model: str, verdict: str) -> AircraftRecommendation:
    return AircraftRecommendation(
        model=model,
        category="ultra-long",
        total_score=0.7,
        confidence=0.8,
        rank=1,
        avoid=False,
        fit="Strong Fit",
        fit_verdict=verdict,
        explanation=RecommendationExplanation(summary=""),
    )


def test_format_immutable_block_lists_presented_and_eliminated():
    block = format_immutable_reasoning_packet_block(_packet())
    assert "PRESENTED" in block
    assert "G650" in block
    assert "ELIMINATED" in block
    assert "g280" in block.lower()
    assert "do not upgrade" in block.lower()


def test_detect_unauthorized_hallucinated_model():
    packet = _packet()
    bad = (
        "I'd start with the Gulfstream G650. Also consider the Falcon 8X for this leg."
    )
    unauthorized = detect_unauthorized_from_packet(bad, packet)
    assert any("falcon" in u.lower() for u in unauthorized)


def test_detect_eliminated_mention():
    packet = _packet()
    text = "The G280 is tempting but we ruled it out; lead with the G650."
    assert detect_eliminated_mentions(text, {"g280"})


def test_detect_verdict_upgrade_primary_language_on_risky():
    packet = _packet()
    verdict_map = authorized_verdict_map(
        packet,
        [_rec("Citation Latitude", BrokerVerdict.MISSION_RISKY.value)],
    )
    text = (
        "Citation Latitude — I'd start with this as the best fit for your winter westbound leg."
    )
    upgrades = detect_verdict_upgrades(text, verdict_map)
    assert upgrades


def test_enforce_regenerates_on_hallucination():
    mission = MissionState(routes=["TEB → London"], passenger_count=8)
    recs = [
        _rec("Gulfstream G650", BrokerVerdict.PRIMARY_RECOMMENDATION.value),
        _rec("Bombardier Global 7500", BrokerVerdict.VIABLE_WITH_COMPROMISES.value),
    ]
    data_used: dict = {IMMUTABLE_PACKET_KEY: _packet()}

    bad = "Primary pick: Falcon 8X — best fit for TEB to London nonstop."
    fixed, report = enforce_reasoning_packet_authority(
        bad,
        data_used=data_used,
        recommendations=recs,
        mission=mission,
        query="TEB to London 8 pax",
    )
    assert report.regenerated or not report.ok
    assert "falcon 8x" not in fixed.lower() or "G650" in fixed or "Global" in fixed


def test_detect_single_aircraft_collapse_on_multi_domain_packet():
    audit = build_fleet_audit_trace(
        {
            "multi_aircraft_required": True,
            "single_aircraft_structurally_invalid": True,
            "presented_models": ["Pilatus PC-24", "Global 7500"],
            "domain_traces": [
                {"domain": "ulr_class", "feasible_models": ["Global 7500"], "constraint_triggers": ["corridor"]},
                {"domain": "short_field_high_performance", "feasible_models": ["Pilatus PC-24"], "constraint_triggers": ["hot_high"]},
            ],
            "assignments": [],
            "segments": [],
        }
    )
    packet = {"fleet_audit": audit}
    assert detect_single_aircraft_collapse(
        "Use one aircraft for everything — the G650 handles London and Aspen.",
        packet,
    )
