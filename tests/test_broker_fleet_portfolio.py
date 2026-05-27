"""Ranked broker path includes fleet / portfolio sections for incompatible missions."""

from __future__ import annotations

from services.consultant.broker_advisory_layer import format_broker_advisory_response
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import (
    MISSION_UNDERSTANDING_KEY,
    build_mission_understanding,
    attach_mission_understanding,
)
from services.mission.adapters import mission_profile_to_state


def _fake_rec(model: str) -> AircraftRecommendation:
    return AircraftRecommendation(
        model=model,
        category="ultra-long",
        total_score=0.8,
        confidence=0.7,
        rank=1,
        avoid=False,
        fit="Strong fit",
    )


def test_pe_ranked_response_includes_portfolio_or_fleet():
    q = (
        "We are a private equity group. Teams move between New York, Dallas, London, and occasionally Dubai. "
        "Senior partners hate fuel stops, but we also visit smaller industrial airports domestically. "
        "We currently charter around 300 hours annually and are debating ownership."
    )
    profile = extract_mission(q)
    mission = mission_profile_to_state(profile)
    pkt = build_mission_understanding(q, profile, mission, use_llm=False)
    assert pkt.inferred_constraints.get("incompatible_mission_bands")
    assert pkt.inferred_constraints.get("industrial_airport_access")
    assert not pkt.inferred_constraints.get("mountain_ops")

    du: dict = {}
    attach_mission_understanding(du, pkt)
    du["fleet_composition_plan"] = {
        "multi_aircraft_required": True,
        "doctrine": "Incompatible operational bands — portfolio required.",
        "ownership_note": "Charter transition at ~300 hr/year.",
        "segments": [
            {
                "role": "ulr_international",
                "label": "ULR class — Dallas -> London",
                "route_labels": ["Dallas -> London"],
            },
            {
                "role": "mountain_field",
                "label": "Domestic field-access",
                "route_labels": ["New York -> Dallas"],
            },
        ],
        "assignments": [
            {
                "role": "ulr_international",
                "segment_label": "ULR class",
                "primary_model": "Global 7500",
                "fit_verdict": "domain fit",
                "rationale": "Oceanic nonstop posture.",
            },
            {
                "role": "mountain_field",
                "segment_label": "Domestic field-access",
                "primary_model": "Pilatus PC-24",
                "fit_verdict": "domain fit",
                "rationale": "Industrial airport access.",
            },
        ],
    }

    text = format_broker_advisory_response(
        mission,
        [_fake_rec("Global 7500")],
        query=q,
        data_used=du,
    )
    low = text.lower()
    assert "multi-domain" in low or "fleet structure" in low or "portfolio" in low
    assert "ownership economics" in low or "300" in low or "charter" in low
