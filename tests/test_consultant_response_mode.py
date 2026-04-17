from __future__ import annotations

from rag.consultant_response_mode import (
    ConsultantResponseMode,
    classify_consultant_response_mode,
    response_mode_prompt_suffix,
)
from rag.consultant_suspicious_model import consultant_suspicious_aircraft_model_note


def test_mode_invalid_when_suspicious_model_note_present():
    q = "Tell me about Falcon 9000 specs"
    note = consultant_suspicious_aircraft_model_note(q)
    assert note
    mode = classify_consultant_response_mode(
        query=q,
        fine_intent="aircraft_specs",
        has_tail=False,
        has_visual_intent=False,
        suspicious_model_note=note,
    )
    assert mode == ConsultantResponseMode.INVALID_SANITY


def test_mode_tail_specific_over_visual():
    mode = classify_consultant_response_mode(
        query="show me N807JS",
        fine_intent="ownership_lookup",
        has_tail=True,
        has_visual_intent=True,
        suspicious_model_note=None,
    )
    assert mode == ConsultantResponseMode.TAIL_SPECIFIC


def test_mode_comparison_detects_vs():
    mode = classify_consultant_response_mode(
        query="Falcon 2000 vs Challenger 350",
        fine_intent="aircraft_comparison",
        has_tail=False,
        has_visual_intent=False,
        suspicious_model_note=None,
    )
    assert mode == ConsultantResponseMode.COMPARISON


def test_mode_strategic_detects_ownership_costs():
    mode = classify_consultant_response_mode(
        query="Own vs charter: what is the total cost of ownership and hourly cost?",
        fine_intent="market_question",
        has_tail=False,
        has_visual_intent=False,
        suspicious_model_note=None,
    )
    assert mode == ConsultantResponseMode.STRATEGIC_OWNERSHIP


def test_mode_mission_advisory_from_fine_intent():
    mode = classify_consultant_response_mode(
        query="What should I buy for NYC to Aspen with 6 pax?",
        fine_intent="aircraft_recommendation",
        has_tail=False,
        has_visual_intent=False,
        suspicious_model_note=None,
    )
    assert mode == ConsultantResponseMode.MISSION_ADVISORY


def test_prompt_suffix_contains_required_templates_and_insight():
    s = response_mode_prompt_suffix(ConsultantResponseMode.MISSION_ADVISORY)
    assert "MISSION ADVISORY" in s
    assert "Consultant Insight:" in s
    s2 = response_mode_prompt_suffix(ConsultantResponseMode.COMPARISON)
    assert "Verdict first" in s2
    assert "Consultant Insight:" in s2

