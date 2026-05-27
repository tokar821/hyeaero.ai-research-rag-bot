"""End-to-end consultant intelligence orchestrator tests."""

import os

from services.consultant.intelligence_engine import run_consultant_intelligence_layer


def test_intelligence_layer_removes_fallback_and_structures():
    os.environ["CONSULTANT_INTELLIGENCE_LAYER"] = "1"
    raw = (
        "Here is some text.\n\n"
        "Assuming 6–8 passengers and typical business-use constraints, here are a few realistic fits:\n"
        "- Challenger 350: ok\n\nConsultant Insight: dispatch."
    )
    from services.state.mission_state import sync_persistent_mission_state

    history = [{"role": "user", "content": "8 passengers LA to Miami $10M nonstop"}]
    data_used = {"consultant_response_mode": "mission_advisory"}
    sync_persistent_mission_state(history[0]["content"], data_used=data_used)
    result = run_consultant_intelligence_layer(
        answer=raw,
        query="What aircraft do you recommend?",
        history=history,
        data_used=data_used,
    )
    assert result.applied
    assert "Assuming 6" not in result.answer
    assert "Mission Summary" not in result.answer
    assert result.data_used_patch.get("consultant_response_style")
    assert len(result.answer) > 60
    assert "conditional paths" not in result.answer.lower()
    assert "Mission Summary" not in result.answer
    assert result.data_used_patch.get("consultant_recommendations")
    assert result.data_used_patch.get("consultant_mission_state")


def test_conflicting_constraints_still_produces_ranking():
    result = run_consultant_intelligence_layer(
        answer="Maybe a CJ2 or a G650?",
        query="12 pax LA to London nonstop, $8M budget, want G650 cabin",
        history=None,
        data_used={"consultant_response_mode": "advisory"},
    )
    recs = result.data_used_patch.get("consultant_recommendations") or []
    assert recs
    assert result.mission_state.passenger_count == 12  # from current query only
    assert "consultant_mission_profile" in result.data_used_patch


def test_range_map_request_attaches_visual_models():
    result = run_consultant_intelligence_layer(
        answer="Draft about range map.",
        query="Show range map SFO to Paris for Falcon 8X vs G650",
        history=None,
        data_used={"consultant_response_mode": "comparison_mode"},
    )
    visuals = result.data_used_patch.get("consultant_visual_models") or {}
    assert visuals.get("range_maps") or visuals.get("comparison_cards")
