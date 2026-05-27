"""Hybrid mission understanding — LLM inference merged with rules."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.adapters import mission_profile_to_state
from services.mission.llm_mission_understanding import (
    LLMMissionUnderstandingResult,
    parse_llm_mission_understanding_payload,
)
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import (
    MissionUnderstandingPacket,
    build_mission_understanding,
)
from services.mission.mission_understanding_merge import merge_llm_understanding_into_packet


def test_parse_llm_mission_understanding_payload():
    data = {
        "confidence": 0.82,
        "inferred_constraints": {"executive_transport": True, "aircraft": "Citation CJ4"},
        "operational_environment": ["Dispatch reliability matters for board travel."],
        "ownership_profile": "corporate_shuttle_candidate",
        "travel_pattern": "transatlantic_executive",
        "corridor_type": "transatlantic_ulr",
        "runway_complexity": "high",
        "dispatch_priority": "high",
        "comfort_priority": "high",
        "operating_cost_priority": "standard",
        "nonstop_priority": "high",
        "utilization_style": "board_transport",
        "operational_synthesis": "This reads as executive transatlantic shuttle work with nonstop pressure.",
        "understanding_notes": ["Board travel signal present."],
        "clarifying_question": "Which European city is the primary destination?",
    }
    result = parse_llm_mission_understanding_payload(data)
    assert result.ok
    assert result.confidence == 0.82
    assert result.inferred_constraints.get("executive_transport") is True
    assert "aircraft" not in result.inferred_constraints
    assert result.travel_pattern == "transatlantic_executive"
    assert result.clarifying_question is not None


def test_merge_preserves_rule_inferred_constraints():
    packet = MissionUnderstandingPacket(
        inferred_constraints={"enterprise_employees": 1000, "runway_over_cabin": True},
        corridor_type="caribbean_regional",
        travel_pattern="executive_shuttle",
        dispatch_priority="high",
        overall_confidence=0.71,
        operational_synthesis="Island ops with runway priority.",
    )
    llm = LLMMissionUnderstandingResult(
        confidence=0.9,
        inferred_constraints={
            "enterprise_employees": 50,
            "family_office": True,
        },
        operational_synthesis="Family office leisure travel.",
        travel_pattern="family_leisure",
    )
    merged = merge_llm_understanding_into_packet(packet, llm, rule_confidence=0.71)
    assert merged.inferred_constraints["enterprise_employees"] == 1000
    assert merged.inferred_constraints["family_office"] is True
    assert merged.travel_pattern == "executive_shuttle"
    assert "Island ops" in merged.operational_synthesis


def test_merge_fills_unknown_fields_from_llm():
    packet = MissionUnderstandingPacket(
        ownership_profile="unknown",
        travel_pattern="unknown",
        corridor_type="unknown",
        overall_confidence=0.4,
    )
    llm = LLMMissionUnderstandingResult(
        confidence=0.75,
        ownership_profile="family_office",
        travel_pattern="multi_leg",
        corridor_type="multi_leg_ultra_long",
        operational_environment=["Portfolio mission spanning oceanic and mountain domains."],
        operational_synthesis="Split mission structure is more credible than one airframe.",
    )
    merged = merge_llm_understanding_into_packet(packet, llm, rule_confidence=0.4)
    assert merged.ownership_profile == "family_office"
    assert merged.travel_pattern == "multi_leg"
    assert merged.corridor_type == "multi_leg_ultra_long"
    assert merged.overall_confidence >= 0.4
    assert merged.confidence_scores.get("llm_inference") == 0.75


def test_build_mission_understanding_rules_only_when_llm_disabled():
    q = "We fly teams around Latin America — runway access matters more than cabin luxury"
    profile = extract_mission(q)
    mission = mission_profile_to_state(profile)
    pkt = build_mission_understanding(q, profile, mission, use_llm=False)
    assert pkt.runway_complexity == "high" or pkt.inferred_constraints.get("runway_over_cabin")
    assert "LLM understanding skipped" not in " ".join(pkt.understanding_notes)


def test_build_mission_understanding_hybrid_with_mock_llm(monkeypatch):
    q = "Family office — Europe twice monthly, board travel, dispatch-sensitive"
    profile = extract_mission(q)
    mission = MissionState(passenger_count=8)

    def _fake_llm(*_args, **_kwargs):
        return LLMMissionUnderstandingResult(
            confidence=0.8,
            inferred_constraints={"board_travel": True, "dispatch_sensitive": True},
            operational_environment=["Executive schedule pressure — dispatch reliability is primary."],
            travel_pattern="transatlantic_executive",
            ownership_profile="family_office",
            dispatch_priority="high",
            nonstop_priority="high",
            operational_synthesis="Reads as recurring transatlantic executive transport with reliability bias.",
            model="test-model",
        )

    monkeypatch.setattr(
        "services.mission.llm_mission_understanding.infer_mission_understanding_llm",
        _fake_llm,
    )
    pkt = build_mission_understanding(q, profile, mission, use_llm=True)
    assert pkt.inferred_constraints.get("board_travel") is True
    assert pkt.travel_pattern == "transatlantic_executive"
    assert pkt.confidence_scores.get("llm_inference") == 0.8
    assert any("Hybrid mission understanding" in n for n in pkt.understanding_notes)
