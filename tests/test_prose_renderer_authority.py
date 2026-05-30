"""Prose renderer v2 — recommendation authority on ranked degradation paths."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_authority import (
    EMPTY_PIPELINE_AUTHORITY_MESSAGE,
    RecommendationAuthority,
)
from services.rendering.prose_renderer_v2 import (
    _empty_shortlist_guidance,
    _filter_shortlist_to_authority,
    render_error_fallback,
    render_recommendation_prose,
    render_strategic_prose,
)


def _ranked_du(**extra) -> dict:
    du = {
        "pipeline_authority_enforced": True,
        "block_aircraft_substitution": True,
        "query_recommendation_intent": "acquisition_recommendation",
        "approved_shortlist": extra.get("approved_shortlist", []),
        "final_ranked_aircraft": extra.get("final_ranked_aircraft", []),
    }
    du.update(extra)
    return du


def test_case1_empty_shortlist_authority_message_no_aircraft_names():
    mission = MissionState(routes=["SFO -> Tokyo"], passenger_count=8)
    du = _ranked_du(approved_shortlist=[], final_ranked_aircraft=[])
    text = render_recommendation_prose(
        {"shortlist": []},
        mission=mission,
        query="recommend 8 pax SFO Tokyo nonstop",
        data_used=du,
    )
    assert EMPTY_PIPELINE_AUTHORITY_MESSAGE in text
    auth = RecommendationAuthority.from_pipeline([], data_used=du)
    assert not auth.detect_unauthorized(text)
    assert "Closest survivors" not in text
    assert "feasible band" not in text.lower() or "constraint" in text.lower()


def test_case2_shortlist_filtered_to_approved_only():
    mission = MissionState(routes=["Miami -> London"], passenger_count=8)
    du = _ranked_du(
        approved_shortlist=["Embraer Praetor 600"],
        final_ranked_aircraft=["Embraer Praetor 600"],
    )
    payload = {
        "shortlist": [
            {"rank": 1, "label": "Embraer Praetor 600", "fit_verdict": "PRIMARY"},
            {"rank": 2, "label": "Citation Latitude", "fit_verdict": "ALTERNATE"},
            {"rank": 3, "label": "Bombardier Challenger 350", "fit_verdict": "ALTERNATE"},
            {"rank": 4, "label": "Cessna Caravan", "fit_verdict": "ALTERNATE"},
        ],
    }
    filtered = _filter_shortlist_to_authority(
        payload["shortlist"],
        data_used=du,
        query="recommend Praetor 600",
    )
    assert len(filtered) == 1
    assert filtered[0]["label"] == "Embraer Praetor 600"

    text = render_recommendation_prose(
        payload,
        mission=mission,
        query="recommend Praetor 600",
        data_used=du,
    )
    assert "Praetor" in text
    assert "Latitude" not in text
    assert "Challenger" not in text
    assert "Caravan" not in text


def test_case3_ranked_query_no_metadata_infers_authority():
    mission = MissionState(routes=["NYC -> Paris"], passenger_count=8)
    query = "What aircraft do you recommend for 8 passengers NYC to Paris nonstop?"
    text = _empty_shortlist_guidance(
        mission,
        None,
        query,
        data_used=None,
    )
    assert EMPTY_PIPELINE_AUTHORITY_MESSAGE in text
    auth = RecommendationAuthority.from_pipeline([], data_used=None)
    assert not auth.detect_unauthorized(text)
    assert "King Air" not in text
    assert "Closest survivors" not in text


def test_case4_non_ranked_strategic_unchanged():
    mission = MissionState(routes=["Dallas -> Houston"], passenger_count=8)
    du = {
        "query_recommendation_intent": "ownership_economics",
        "hierarchy_weighting": {
            "dominant_utilization": "regional_executive",
            "secondary_traffic": ["caribbean"],
        },
    }
    text = render_strategic_prose(
        {"operational_domains": ["regional_executive"], "conflicts": []},
        mission=mission,
        query="cost of ownership G650 400 hours per year",
        data_used=du,
    )
    assert "Strategic Fleet Analysis" in text
    assert EMPTY_PIPELINE_AUTHORITY_MESSAGE not in text


def test_error_fallback_recommendation_request_passes_data_used():
    mission = MissionState(routes=["LAX -> London"], passenger_count=8)
    du = _ranked_du(approved_shortlist=[], final_ranked_aircraft=[])
    text = render_error_fallback(
        "renderer_failed",
        mode="recommendation_request",
        mission=mission,
        query="recommend LAX London 8 pax",
        data_used=du,
    )
    assert EMPTY_PIPELINE_AUTHORITY_MESSAGE in text
    assert "King Air" not in text
