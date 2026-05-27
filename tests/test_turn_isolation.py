"""
Turn-isolated mission extraction — no cross-turn contamination.
"""

import warnings

from services.consultant.mission_state import MissionState, build_mission_from_current_turn, update_mission_state
from services.mission import extract_mission, mission_profile_to_state


ASSISTANT_GARBAGE = """
Mission Summary
Passengers: 10
Route(s): What Would You Like -> Work, Full Ownership -> Higher
Best Fit Aircraft
Challenger 350
Consultant Insight: buyer remorse
"""


def test_no_passenger_carryover_between_unrelated_turns():
    first = extract_mission("10 passengers LA to Miami nonstop $10M")
    second = extract_mission("What is the range of a Falcon 8X?")
    assert first.passengers == 10
    assert second.passengers is None


def test_no_route_leakage_miami_vs_tokyo():
    tokyo = extract_mission(
        "6 executives San Francisco to Tokyo and Seoul, westbound winter nonstop"
    )
    miami = extract_mission("8 passengers Miami to Caribbean, short runway focus")
    tokyo_labels = " ".join(r.label().lower() for r in tokyo.routes)
    miami_labels = " ".join(r.label().lower() for r in miami.routes)
    assert "tokyo" in tokyo_labels or "san francisco" in tokyo_labels
    assert "tokyo" not in miami_labels
    assert "miami" in miami_labels


def test_assistant_text_not_parsed_as_user_mission():
    polluted = f"8 pax Miami to Caribbean\n{ASSISTANT_GARBAGE}"
    profile = extract_mission(polluted)
    assert profile.passengers == 8
    assert not any("what would you like" in r.label().lower() for r in profile.routes)


def test_history_ignored_by_update_mission_state():
    prior = MissionState(passenger_count=10, routes=["San Francisco -> Tokyo"])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = update_mission_state(
            prior,
            "Price of a used Citation Latitude?",
            history=[
                {"role": "user", "content": "10 pax LA to Miami"},
                {"role": "assistant", "content": ASSISTANT_GARBAGE},
            ],
        )
    assert out.passenger_count is None
    assert out.routes == []


def test_prior_mission_state_not_merged():
    prior = MissionState(
        passenger_count=12,
        routes=["West Coast -> Europe"],
        budget_usd=40_000_000.0,
        acquisition_strategy="full_ownership",
    )
    current = build_mission_from_current_turn("How many seats does a HondaJet have?")
    assert current.passenger_count is None
    assert current.routes == []
    assert current.budget_usd is None
    assert current.acquisition_strategy is None


def test_ui_phrase_not_extracted_as_route():
    profile = extract_mission(
        "What would you like to explore? Full ownership vs fractional work higher efficiency"
    )
    assert profile.routes == []


def test_deterministic_structured_output():
    q = "8 passengers Miami to Caribbean, operating cost priority, runway flexibility"
    p = extract_mission(q)
    assert p.passengers == 8
    assert len(p.routes) == 1
    assert p.routes[0].origin == "Miami"
    assert p.operating_cost_priority.value == "high"
    assert p.runway_priority.value == "high"
    d = p.to_dict()
    assert d["extraction_mode"] == "turn_isolated"
    assert d["routes"] == [{"origin": "Miami", "destination": "Caribbean"}]


def test_mission_profile_adapter_maps_to_state():
    profile = extract_mission("6 pax LA to Miami nonstop under $12M")
    state = mission_profile_to_state(profile)
    assert state.passenger_count == 6
    assert state.routes == ["Los Angeles -> Miami"]
    assert state.budget_usd and state.budget_usd >= 11_000_000
    assert state.nonstop_requirement is True


def test_each_call_independent_no_snapshots_accumulated():
    m1 = build_mission_from_current_turn("8 pax NYC to London")
    m2 = build_mission_from_current_turn("Compare Phenom 300 vs CJ4")
    assert m1.snapshots == []
    assert m2.snapshots == []
    assert m1.routes != m2.routes or m2.mission_type == "comparison"
