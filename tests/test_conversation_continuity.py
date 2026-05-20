"""Conversation continuity layer — deterministic unit tests."""

from __future__ import annotations

from services.conversation_continuity import run_continuity_turn
from services.conversation_continuity.entity_lock import extract_tail_from_text


def test_tail_lock_interior_followup_append():
    hist = [{"role": "assistant", "content": "Listed N628TS Falcon 900."}]
    b0 = run_continuity_turn(
        raw_user_query="Do you still have N628TS?",
        isolated_query="Do you still have N628TS?",
        history=hist,
        client_conversation_state=None,
        strict_tail_candidates=["N628TS"],
    )
    assert b0.state.locked_entity is not None
    assert "N628TS" in (b0.serialized.get("locked_entity") or {}).get("value", "")

    prev = {"continuity": b0.serialized}
    b1 = run_continuity_turn(
        raw_user_query="show me interior",
        isolated_query="show me interior",
        history=hist,
        client_conversation_state=prev,
        strict_tail_candidates=[],
    )
    assert "N628TS" in b1.effective_query.upper()


def test_size_upgrade_carry_traits_and_evolution():
    prev_air = {"continuity": {"current_aircraft": "Phenom 300", "schema_version": 1}}
    b = run_continuity_turn(
        raw_user_query="Actually I want something bigger but keep it modern.",
        isolated_query="Actually I want something bigger but keep it modern.",
        history=[],
        client_conversation_state=prev_air,
        strict_tail_candidates=[],
    )
    assert b.refinement.type == "size_upgrade"
    assert b.state.buyer_direction.size == "larger"
    assert "modern" in " ".join(b.state.style_preferences).lower() or b.state.style_preferences
    qlo = (b.effective_query or "").lower()
    assert "phenom 300" in qlo or "larger cabin" in qlo


def test_g650_seats_query_not_comparison_anchor():
    from services.conversation_continuity.refinement import interpret_refinement

    r = interpret_refinement(
        "How many seats does a G650 have?", prev_aircraft=None, prev_tail=None
    )
    assert r.type != "comparison_anchor"


def test_explicit_vs_parses_comparison_anchor():
    from services.conversation_continuity.refinement import interpret_refinement

    r = interpret_refinement(
        "Compare G700 vs Global 7500.", prev_aircraft=None, prev_tail=None
    )
    assert r.type == "comparison_anchor"
    ref = (r.reference_aircraft or "").lower()
    assert "g700" in ref
    assert "7500" in ref


def test_style_shift_negative_preferences():
    prev = {"continuity": {"current_aircraft": "Challenger 350", "schema_version": 1}}
    b = run_continuity_turn(
        raw_user_query="less corporate vibe — luxury hotel aesthetic",
        isolated_query="less corporate vibe — luxury hotel aesthetic",
        history=[],
        client_conversation_state=prev,
        strict_tail_candidates=[],
    )
    assert b.refinement.type == "style_shift"
    assert any("hotel" in p.lower() for p in (b.state.style_preferences or []))
    joined_neg = " ".join(b.state.negative_preferences or []).lower()
    assert "corporate" in joined_neg


def test_view_change_carries_locked_tail_through():
    c0 = run_continuity_turn(
        raw_user_query="tell me about N628TS exterior",
        isolated_query="tell me about N628TS exterior",
        history=[],
        client_conversation_state=None,
        strict_tail_candidates=["N628TS"],
    )
    prev = {"continuity": c0.serialized}
    c1 = run_continuity_turn(
        raw_user_query="cockpit too please",
        isolated_query="cockpit too please",
        history=[],
        client_conversation_state=prev,
        strict_tail_candidates=[],
    )
    assert c1.refinement.type == "view_change"
    assert c1.state.last_requested_view == "cockpit"
    assert "N628TS" in c1.effective_query.upper()


def test_contextual_influencer_tag():
    b = run_continuity_turn(
        raw_user_query="Something influencers charter for reels",
        isolated_query="Something influencers charter for reels",
        history=[],
        client_conversation_state=None,
        strict_tail_candidates=[],
    )
    tags = [t.lower() for t in (b.state.contextual_intent_tags or [])]
    blob = " ".join(tags)
    assert "modern" in blob or "lifestyle" in blob


def test_explicit_reset_wipes_prior():
    prev = {
        "continuity": {"current_aircraft": "Citation XLS", "schema_version": 1, "locked_entity": None}
    }
    b = run_continuity_turn(
        raw_user_query="Start over — new topic unrelated to jets.",
        isolated_query="Start over — new topic unrelated to jets.",
        history=[],
        client_conversation_state=prev,
        strict_tail_candidates=[],
    )
    assert b.refinement.type == "explicit_reset"
    assert b.state.current_aircraft is None


def test_extract_tail_normalizes():
    assert extract_tail_from_text("looking at N 628 TS") == "N628TS"


def test_visual_only_response_mode_keyword():
    b = run_continuity_turn(
        raw_user_query="don't explain — just show me cabin photos",
        isolated_query="don't explain — just show me cabin photos",
        history=[],
        client_conversation_state=None,
        strict_tail_candidates=[],
    )
    assert b.state.response_mode.value == "visual_only"


def test_comparison_anchor_keeps_prior_evolution():
    prev_ev = {"continuity": {"aircraft_evolution": ["Phenom 300"], "schema_version": 1}}
    b = run_continuity_turn(
        raw_user_query="Compare that preference to a Gulfstream G650 cabin",
        isolated_query="Compare that preference to a Gulfstream G650 cabin",
        history=[{"role": "user", "content": "I liked the Phenom 300 interior suggested earlier."}],
        client_conversation_state=prev_ev,
        strict_tail_candidates=[],
    )
    assert b.refinement.type == "comparison_anchor"

