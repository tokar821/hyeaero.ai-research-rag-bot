"""Pinpoint factual answer trimming and boilerplate removal."""

from __future__ import annotations

from rag.pinpoint_answer import (
    enforce_pinpoint_answer,
    is_pinpoint_factual_turn,
    strip_advisory_boilerplate,
)
from rag.response_safety import enforce_consultant_quality


def test_pinpoint_seats_detected():
    assert is_pinpoint_factual_turn(
        "How many seats does a G650 have?",
        {"consultant_fine_intent": "aircraft_specs"},
    )


def test_pinpoint_strips_good_fit_block():
    raw = (
        "The G650 seats about 14 passengers.\n\n"
        "✅ GOOD FIT\n\nAssuming 6–8 passengers, here are a few realistic fits:\n"
        "- Challenger 350: balanced mission.\n\n"
        "Bottom Line: start with Challenger 350.\n\n"
        "Consultant Insight: buyer's remorse is about dispatch."
    )
    out = strip_advisory_boilerplate(raw)
    assert "GOOD FIT" not in out
    assert "Consultant Insight" not in out
    assert "14 passengers" in out


def test_implausible_challenger_350_price_scrubbed():
    bad = "The asking price for a 2019 Challenger 350 based in Santa Barbara, California is $850,000."
    out = enforce_consultant_quality(
        bad,
        query="Price of a used Challenger 350?",
        data_used={"consultant_fine_intent": "market_question"},
    )
    assert "850" not in out
    assert "15" in out
    assert "million" in out.lower()


def test_pinpoint_seats_strips_range_sentence():
    raw = (
        "The Gulfstream G650 typically seats up to 16 passengers comfortably.\n\n"
        "This aircraft falls into the large category and is known for its impressive range of "
        "approximately 7,000 nautical miles and a cruise speed of around 488 knots."
    )
    out = enforce_pinpoint_answer(
        raw,
        query="How many seats does a G650 have?",
        data_used={"consultant_fine_intent": "aircraft_specs"},
    )
    assert "seat" in out.lower() or "passenger" in out.lower()
    assert "knot" not in out.lower()
    assert "7,000" not in out


def test_pinpoint_range_strips_passenger_clause():
    raw = (
        "The Dassault Falcon 8X boasts a remarkable range of up to 6,450 nautical miles "
        "when flying at Mach 0.80, accommodating eight passengers and three crew members."
    )
    out = enforce_pinpoint_answer(
        raw,
        query="What's the range of a Falcon 8X?",
        data_used={"consultant_fine_intent": "aircraft_specs"},
    )
    assert "6,450" in out
    assert "passenger" not in out.lower()


def test_enforce_consultant_quality_skips_advisory_append_on_specs():
    short = "The Falcon 8X range is about 6,450 nautical miles with 8 passengers."
    out = enforce_consultant_quality(
        short,
        query="What's the range of a Falcon 8X?",
        data_used={"consultant_fine_intent": "aircraft_specs", "consultant_response_mode": "advisory"},
    )
    assert "Assuming 6–8 passengers" not in out
    assert "Challenger 350:" not in out


def test_enforce_pinpoint_trims_verbose_range():
    verbose = (
        "The Falcon 8X offers 6,450 nm range.\n\n"
        "This allows non-stop flights worldwide.\n\n"
        "✅ GOOD FIT\n\nAssuming 6–8 passengers:\n- Challenger 350\n"
    )
    out = enforce_pinpoint_answer(
        verbose,
        query="What's the range of a Falcon 8X?",
        data_used={"consultant_fine_intent": "aircraft_specs"},
    )
    assert "GOOD FIT" not in out
    assert "6,450" in out
