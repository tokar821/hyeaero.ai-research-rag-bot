from __future__ import annotations

from rag.consultant_response_mode import (
    ConsultantResponseMode,
    classify_consultant_response_mode,
    consultant_response_router_json,
    response_mode_prompt_suffix,
    route_consultant_response_mode,
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
    assert mode == ConsultantResponseMode.COMPARISON_MODE


def test_mode_advisory_detects_ownership_costs():
    mode = classify_consultant_response_mode(
        query="Own vs charter: what is the total cost of ownership and hourly cost?",
        fine_intent="market_question",
        has_tail=False,
        has_visual_intent=False,
        suspicious_model_note=None,
    )
    assert mode == ConsultantResponseMode.ADVISORY_MODE


def test_mode_advisory_from_fine_intent():
    mode = classify_consultant_response_mode(
        query="What should I buy for NYC to Aspen with 6 pax?",
        fine_intent="aircraft_recommendation",
        has_tail=False,
        has_visual_intent=False,
        suspicious_model_note=None,
    )
    assert mode == ConsultantResponseMode.ADVISORY_MODE


def test_router_json_shape():
    r = route_consultant_response_mode(
        query="show me G650 interior cabin",
        fine_intent="aircraft_specs",
        has_tail=False,
        has_visual_intent=True,
        suspicious_model_note=None,
    )
    assert r["mode"] == "visual_mode"
    assert r["visual_priority"] is True
    assert r["verbosity"] == "minimal"
    assert "reason" in r
    parsed = consultant_response_router_json(r)
    assert "visual_mode" in parsed


def test_visual_mode_luxury_vibe():
    m = classify_consultant_response_mode(
        query="something with a luxury hotel vibe inside the cabin",
        fine_intent="aviation_mission",
        has_tail=False,
        has_visual_intent=False,
        suspicious_model_note=None,
    )
    assert m == ConsultantResponseMode.VISUAL_MODE


def test_deal_analysis_mode():
    m = classify_consultant_response_mode(
        query="Is this G650 listing a good deal at 45M?",
        fine_intent="aircraft_price_lookup",
        has_tail=False,
        has_visual_intent=False,
        suspicious_model_note=None,
    )
    assert m == ConsultantResponseMode.DEAL_ANALYSIS_MODE


def test_prompt_suffix_visual_and_advisory():
    s = response_mode_prompt_suffix(ConsultantResponseMode.VISUAL_MODE)
    assert "IMAGE_SHOWCASE" in s
    assert "gallery" in s.lower()
    s2 = response_mode_prompt_suffix(ConsultantResponseMode.ADVISORY_MODE)
    assert "ADVISORY" in s2
    s3 = response_mode_prompt_suffix(ConsultantResponseMode.COMPARISON_MODE)
    assert "COMPARISON_MODE" in s3
