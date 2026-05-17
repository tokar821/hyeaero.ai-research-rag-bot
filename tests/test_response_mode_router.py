"""Response Mode Router tests."""

from __future__ import annotations

from services.response_mode_router import (
    ResponseMode,
    enforce_mode_on_answer,
    response_mode_prompt_suffix,
    route_response_mode,
)
from rag.consultant_response_mode import (
    ConsultantResponseMode,
    classify_consultant_response_mode,
)


def test_image_showcase_interior():
    r = route_response_mode(query="Show me Phenom 300 interior", has_visual_intent=True)
    assert r["mode"] == ResponseMode.IMAGE_SHOWCASE.value
    assert r["forbid_urls_in_text"] is True
    assert r["verbosity"] == "minimal"


def test_followup_bigger_with_anchor():
    r = route_response_mode(
        query="Actually bigger",
        refinement_type="size_upgrade",
        standalone_confidence=0.28,
        has_conversation_anchor=True,
        persistence_routing="refinement_continuation",
    )
    assert r["mode"] in (
        ResponseMode.FOLLOWUP_CONTINUATION.value,
        ResponseMode.IMAGE_SHOWCASE.value,
    )
    assert r["inherit_context"] is True


def test_comparison_vs():
    r = route_response_mode(query="Falcon 2000 vs Challenger 350")
    assert r["mode"] == ResponseMode.COMPARISON_MODE.value


def test_educational_explain():
    r = route_response_mode(query="Explain how RVSM works for charter ops")
    assert r["mode"] == ResponseMode.EDUCATIONAL_MODE.value


def test_advisory_should_i_buy():
    r = route_response_mode(query="Should I buy this Challenger 350 listing?")
    assert r["mode"] == ResponseMode.ADVISORY.value


def test_cockpit_too_visual_showcase():
    r = route_response_mode(
        query="Cockpit too",
        refinement_type="view_change",
        has_conversation_anchor=True,
        memory_response_mode="image_showcase",
    )
    assert r["mode"] == ResponseMode.IMAGE_SHOWCASE.value


def test_enforce_strips_urls_in_showcase():
    raw = "Here is the cabin. https://example.com/photo.jpg More specs and disclaimers."
    out = enforce_mode_on_answer(
        raw,
        mode=ResponseMode.IMAGE_SHOWCASE.value,
        has_gallery=True,
        max_sentences_hint=2,
    )
    assert "http" not in out.lower()
    assert "example.com" not in out


def test_legacy_wrapper_visual_mode():
    m = classify_consultant_response_mode(
        query="show me luxury cabin photos",
        fine_intent="general_question",
        has_tail=False,
        has_visual_intent=True,
        suspicious_model_note=None,
    )
    assert m == ConsultantResponseMode.VISUAL_MODE


def test_prompt_suffix_contains_mode_name():
    assert "IMAGE_SHOWCASE" in response_mode_prompt_suffix(ResponseMode.IMAGE_SHOWCASE)
    assert "FOLLOWUP_CONTINUATION" in response_mode_prompt_suffix(ResponseMode.FOLLOWUP_CONTINUATION)
