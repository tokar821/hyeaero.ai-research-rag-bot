"""Tests for consultant response-depth classification (consultant vs retrieval tone)."""

from __future__ import annotations

import pytest

from rag.consultant_fine_intent import ConsultantFineIntent
from rag.consultant_response_depth import (
    ResponseDepthKind,
    classify_response_depth,
    response_depth_prompt_suffix,
)


def _hist_with_assistant() -> list[dict]:
    return [
        {"role": "user", "content": "What about N807JS?"},
        {"role": "assistant", "content": "N807JS is a Citation Excel — listed in marketplace snapshots."},
    ]


def test_confirmation_follow_up_short_circuits_fine_intent():
    h = _hist_with_assistant()
    assert (
        classify_response_depth("So it's listed?", h, ConsultantFineIntent.OWNERSHIP_LOOKUP)
        == ResponseDepthKind.CONFIRMATION
    )
    assert classify_response_depth("Right?", h, ConsultantFineIntent.OWNERSHIP_LOOKUP) == ResponseDepthKind.CONFIRMATION
    assert (
        classify_response_depth("Is that correct?", h, ConsultantFineIntent.AIRCRAFT_RECOMMENDATION)
        == ResponseDepthKind.CONFIRMATION
    )


def test_confirmation_requires_prior_assistant_turn():
    assert (
        classify_response_depth("Right?", None, ConsultantFineIntent.OWNERSHIP_LOOKUP)
        != ResponseDepthKind.CONFIRMATION
    )
    assert (
        classify_response_depth("Right?", [{"role": "user", "content": "hi"}], ConsultantFineIntent.OWNERSHIP_LOOKUP)
        != ResponseDepthKind.CONFIRMATION
    )


def test_ownership_lookup_with_tail_is_aircraft_lookup():
    assert (
        classify_response_depth("What aircraft is N807JS?", None, ConsultantFineIntent.OWNERSHIP_LOOKUP)
        == ResponseDepthKind.AIRCRAFT_LOOKUP
    )


def test_recommendation_is_advisory():
    assert (
        classify_response_depth("Best jet for a CEO under $15M", None, ConsultantFineIntent.AIRCRAFT_RECOMMENDATION)
        == ResponseDepthKind.ADVISORY
    )


def test_general_question_tail_vs_broad():
    assert (
        classify_response_depth("Tell me about the G650", None, ConsultantFineIntent.GENERAL_QUESTION)
        == ResponseDepthKind.ADVISORY
    )
    assert (
        classify_response_depth("Tell me about N807JS", None, ConsultantFineIntent.GENERAL_QUESTION)
        == ResponseDepthKind.AIRCRAFT_LOOKUP
    )


def test_visual_followup_show_me_that_after_tail_in_thread():
    h = [{"role": "user", "content": "Have you N878BW?"}]
    assert (
        classify_response_depth("show me that", h, ConsultantFineIntent.AIRCRAFT_SPECS)
        == ResponseDepthKind.VISUAL_FOLLOWUP
    )


def test_visual_followup_can_i_see_it_with_consultant_role():
    h = [
        {"role": "You", "content": "Have N878BW?"},
        {"role": "Consultant", "content": "N878BW is an Eclipse EA500."},
    ]
    assert (
        classify_response_depth("can I see it?", h, ConsultantFineIntent.OWNERSHIP_LOOKUP)
        == ResponseDepthKind.VISUAL_FOLLOWUP
    )


def test_visual_followup_show_me_interior_short():
    h = [{"role": "user", "content": "N807JS cabin question earlier"}]
    assert (
        classify_response_depth("show me interior", h, ConsultantFineIntent.GENERAL_QUESTION)
        == ResponseDepthKind.VISUAL_FOLLOWUP
    )


def test_visual_followup_bare_show_me_with_thread_aircraft():
    h = [{"role": "user", "content": "N807JS is a Citation Excel on the market."}]
    assert (
        classify_response_depth("show me", h, ConsultantFineIntent.GENERAL_QUESTION)
        == ResponseDepthKind.VISUAL_FOLLOWUP
    )


@pytest.mark.parametrize(
    "kind",
    [
        ResponseDepthKind.CONFIRMATION,
        ResponseDepthKind.VISUAL_FOLLOWUP,
        ResponseDepthKind.AIRCRAFT_LOOKUP,
        ResponseDepthKind.ADVISORY,
    ],
)
def test_prompt_suffix_nonempty(kind: ResponseDepthKind):
    assert len(response_depth_prompt_suffix(kind).strip()) > 40
