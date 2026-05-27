"""Question Necessity Engine — material follow-ups only."""

from services.consultant.mission_state import MissionState, build_mission_from_current_turn
from services.recommendation.clarification_decision import (
    build_clarification_questions,
    mission_clarification_needs,
)
from services.recommendation.question_necessity_engine import (
    PASSENGER_CONFIDENT,
    ROUTE_CONFIDENT,
    build_budget_question,
    category_recommendation_ready,
    evaluate_question_necessity,
    recommendation_possible_with_assumptions,
    score_passenger_inference,
    score_route_inference,
    should_suppress_followups,
)


def test_route_known_suppresses_city_pair_question():
    q = "8 pax LA to Miami nonstop recommend"
    mission = build_mission_from_current_turn(q)
    report = evaluate_question_necessity(mission, q)
    assert score_route_inference(mission, q) >= ROUTE_CONFIDENT
    assert "route_already_known" in report.suppress_reasons
    assert not report.needs.needs_route
    assert "city pair" not in " ".join(report.questions).lower()


def test_route_missing_blocks_and_asks_route():
    mission = build_mission_from_current_turn("recommend a business jet")
    report = evaluate_question_necessity(mission, "recommend a business jet")
    assert report.should_block_recommendation
    assert report.needs.needs_route
    assert report.questions
    assert "city pair" in report.questions[0].lower() or "origin" in report.questions[0].lower()


def test_decisive_mission_suppresses_passenger_and_budget():
    q = "8 pax LA to Miami nonstop recommend"
    mission = build_mission_from_current_turn(q)
    report = evaluate_question_necessity(mission, q)
    assert recommendation_possible_with_assumptions(
        report.inference, mission, q, None
    )
    assert not report.needs.needs_passenger_count
    assert not report.needs.needs_budget


def test_passenger_confidence_when_explicit():
    q = "8 pax LA to Miami nonstop recommend"
    mission = build_mission_from_current_turn(q)
    assert score_passenger_inference(mission, q) >= PASSENGER_CONFIDENT


def test_budget_question_wording():
    assert build_budget_question() == "What's your approximate acquisition budget?"


def test_ambiguous_budget_only_when_not_decisive():
    mission = MissionState(
        routes=["Los Angeles -> Miami"],
        passenger_count=8,
        budget_usd=10_000_000,
    )
    report = evaluate_question_necessity(mission, "8 pax LA to Miami $10M recommend")
    if report.needs.needs_budget:
        assert build_budget_question() in report.questions


def test_us_multi_city_still_asks_category_usage():
    q = "Dallas, New York, Chicago, 6 passengers recommend"
    mission = build_mission_from_current_turn(q)
    report = evaluate_question_necessity(mission, q)
    assert report.needs.needs_category_usage
    assert "domestic" in report.questions[0].lower()


def test_build_clarification_uses_acquisition_budget_wording():
    from services.recommendation.clarification_decision import (
        MissionClarificationNeeds,
    )

    needs = MissionClarificationNeeds(needs_budget=True)
    qs = build_clarification_questions(needs)
    assert any("acquisition budget" in q.lower() for q in qs)


def test_report_audit_shape():
    mission = build_mission_from_current_turn("8 pax LA to Miami recommend")
    payload = evaluate_question_necessity(mission, "8 pax LA to Miami recommend").to_dict()
    assert "inference" in payload
    assert "suppress_reasons" in payload
    assert "candidates" in payload


def test_at_most_one_question():
    q = "Dallas, New York, Chicago, 6 passengers recommend"
    mission = build_mission_from_current_turn(q)
    report = evaluate_question_necessity(mission, q)
    assert len(report.questions) <= 1


def test_clarification_budget_suppresses_second_question():
    mission = build_mission_from_current_turn("recommend a business jet")
    report = evaluate_question_necessity(
        mission, "recommend a business jet", clarifications_already_asked=1
    )
    assert not report.should_ask_any
    assert "clarification_budget_exhausted" in report.suppress_reasons


def test_la_miami_suppresses_followups():
    q = "8 pax LA to Miami nonstop recommend"
    mission = build_mission_from_current_turn(q)
    report = evaluate_question_necessity(mission, q)
    suppress, reasons = should_suppress_followups(report.inference, mission, q)
    assert suppress
    assert "route_already_known" in reasons
    assert category_recommendation_ready(report.inference, mission, q)
    assert not report.should_ask_any
