"""Response format validation and structured regeneration."""

from services.consultant.mission_state import build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations
from services.consultant.response_format_validation import (
    FormatValidationReport,
    ensure_validated_consultant_response,
    regenerate_from_structured_recommendations,
    validateResponseFormatting,
    validate_response_formatting,
)


def _mission_and_recs():
    mission = build_mission_from_current_turn("8 pax LA to Miami nonstop $10M recommend")
    recs = rank_aircraft_recommendations(mission, max_results=3)
    return mission, recs


def test_validate_ok_on_structured_response():
    mission, recs = _mission_and_recs()
    text = regenerate_from_structured_recommendations(
        mission=mission,
        recommendations=recs,
        route_assessments=[],
        query="8 pax LA to Miami nonstop recommend",
        turn_seed="test",
    )
    report = validateResponseFormatting(text, recommendations=recs)
    assert report.ok
    assert not report.issues


def test_detects_duplicated_bullet_aircraft():
    bad = (
        "I'd start with the Challenger 350.\n\n"
        "Short list:\n"
        "- Challenger 350 — lead option\n"
        "- Challenger 350 — backup again\n"
    )
    report = validateResponseFormatting(bad)
    assert not report.ok
    assert any("duplicated" in i for i in report.issues)


def test_detects_incomplete_bullet():
    bad = "Opener.\n\n- Gulfstream G280 —\n"
    report = validateResponseFormatting(bad)
    assert not report.ok
    assert any("incomplete" in i for i in report.issues)


def test_detects_orphaned_label():
    bad = "Lead line.\n\nAlso in the mix:\n"
    report = validateResponseFormatting(bad)
    assert not report.ok
    assert any("orphaned" in i or "empty_alternatives" in i for i in report.issues)


def test_detects_truncated_ending():
    report = validateResponseFormatting("On this route I'd start with the G280...")
    assert not report.ok
    assert "truncated" in report.issues[0]


def test_ensure_regenerates_on_failure():
    mission, recs = _mission_and_recs()
    broken = (
        "Draft text.\n\n"
        "- Challenger 350 —\n"
        "- Challenger 350 — duplicate\n\n"
        "Also in the mix:\n"
    )
    fixed, report = ensure_validated_consultant_response(
        broken,
        mission=mission,
        recommendations=recs,
        route_assessments=[],
        query="8 pax LA to Miami",
        turn_seed="regen-test",
    )
    assert report.regenerated
    assert recs[0].model in fixed
    assert validateResponseFormatting(fixed, recommendations=recs).ok


def test_snake_case_alias():
    assert validate_response_formatting is validateResponseFormatting


def test_empty_alternatives_with_multiple_recs():
    mission, recs = _mission_and_recs()
    bad = "I'd start with the Challenger 350 — strong fit.\n"
    report = validateResponseFormatting(bad, recommendations=recs)
    assert not report.ok
    assert any("alternate" in i or "bullet" in i for i in report.issues)
