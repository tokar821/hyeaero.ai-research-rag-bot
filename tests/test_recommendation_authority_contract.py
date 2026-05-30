"""Recommendation authority contract — pipeline-only aircraft in ranked workflows."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_authority import (
    EMPTY_PIPELINE_AUTHORITY_MESSAGE,
    RecommendationAuthority,
    detect_unauthorized_aircraft,
    format_empty_pipeline_authority_response,
    reconcile_answer_with_pipeline,
    requires_recommendation_aircraft_authority,
)
from services.consultant.recommendation_engine import AircraftRecommendation


def _rec(model: str, category: str = "super-midsize") -> AircraftRecommendation:
    return AircraftRecommendation(
        model=model,
        category=category,
        total_score=0.8,
        confidence=0.7,
        rank=1,
        fit="Strong fit",
        avoid=False,
    )


def _ranked_data_used(**extra) -> dict:
    du = {
        "pre_llm_pipeline_authority": 1,
        "block_aircraft_substitution": True,
        "query_recommendation_intent": "acquisition_recommendation",
        "query_recommendation_requires_pipeline": True,
        "approved_shortlist": extra.get("approved_shortlist", []),
        "final_ranked_aircraft": extra.get("final_ranked_aircraft", []),
    }
    du.update(extra)
    return du


def test_case1_blocks_unauthorized_caravan_when_pipeline_has_praetor():
    recs = [_rec("Embraer Praetor 600")]
    du = _ranked_data_used(
        approved_shortlist=["Embraer Praetor 600"],
        final_ranked_aircraft=["Embraer Praetor 600"],
    )
    mission = MissionState(routes=["Miami -> Barbados"], passenger_count=8)
    llm = (
        "Aircraft Options:\n\n"
        "* Embraer Praetor 600 — strong fit.\n"
        "* Cessna Caravan — also worth a look for island hops.\n"
        "Verdict:\n\n* PRIMARY: Embraer Praetor 600"
    )
    authority = RecommendationAuthority.from_pipeline(recs, data_used=du)
    violations = authority.detect_unauthorized(llm)
    assert any("Caravan" in v for v in violations)

    final, regen = reconcile_answer_with_pipeline(
        llm,
        mission=mission,
        recommendations=recs,
        data_used=du,
        query="recommend for caribbean",
    )
    assert regen
    assert "Caravan" not in final
    log = du.get("unauthorized_aircraft_references") or []
    assert log
    assert any("Caravan" in entry.get("aircraft", "") for entry in log if isinstance(entry, dict))


def test_case2_empty_pipeline_returns_no_aircraft_passed_filters():
    mission = MissionState(routes=["SFO -> Tokyo"], passenger_count=8)
    du = _ranked_data_used(
        approved_shortlist=[],
        final_ranked_aircraft=[],
        deterministic_recommendation_pipeline={
            "feasible_models": [],
            "elimination_log": [
                {
                    "aircraft_name": "Gulfstream G650ER",
                    "reason": "insufficient range with reserves",
                    "mission_constraint_failed": "route_range",
                }
            ],
        },
    )
    body = format_empty_pipeline_authority_response(mission, data_used=du)
    assert EMPTY_PIPELINE_AUTHORITY_MESSAGE in body
    assert "insufficient range" in body.lower() or "route_range" in body.lower()

    final, regen = reconcile_answer_with_pipeline(
        "Citation Latitude would work if you relax payload.",
        mission=mission,
        recommendations=[],
        data_used=du,
        query="recommend transpac",
    )
    assert regen
    assert EMPTY_PIPELINE_AUTHORITY_MESSAGE in final
    assert "Latitude" not in final


def test_case3_only_approved_latitude_longitude_may_appear():
    recs = [_rec("Citation Latitude"), _rec("Citation Longitude", "super-midsize")]
    du = _ranked_data_used(
        approved_shortlist=["Citation Latitude", "Citation Longitude"],
        final_ranked_aircraft=["Citation Latitude", "Citation Longitude"],
    )
    mission = MissionState(routes=["NYC -> London"], passenger_count=8)
    llm = (
        "Aircraft Options:\n\n"
        "* Citation Latitude — best value.\n"
        "* Citation Longitude — more cabin.\n"
        "* Challenger 350 — alternative.\n"
    )
    allowed = detect_unauthorized_aircraft(llm, {"Citation Latitude", "Citation Longitude"})
    assert any("Challenger" in m for m in allowed)

    final, regen = reconcile_answer_with_pipeline(
        llm,
        mission=mission,
        recommendations=recs,
        data_used=du,
        query="compare latitude longitude",
    )
    assert regen
    assert "Challenger" not in final
    assert "Citation Latitude" in final
    assert "Citation Longitude" in final


def test_requires_authority_when_pre_llm_pipeline_set():
    assert requires_recommendation_aircraft_authority({"pre_llm_pipeline_authority": 1})
    assert not requires_recommendation_aircraft_authority({})
