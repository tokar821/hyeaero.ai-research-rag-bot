"""Conversation State Engine unit tests."""

from __future__ import annotations

from services.conversation_state_engine import run_conversation_state_turn


def _client(prev_bundle) -> dict:
    return {"conversation_memory": prev_bundle.serialized, "continuity": prev_bundle.state.model_dump()}


def test_phenom_chain_bigger_modern_cockpit():
    b0 = run_conversation_state_turn(
        query="Show me Phenom 300 interior",
        client_conversation_state=None,
        entity_models=["Phenom 300"],
        user_wants_gallery=True,
    )
    assert b0.state.active_aircraft == "Phenom 300"
    assert b0.state.last_visual_context

    b1 = run_conversation_state_turn(
        query="Actually bigger",
        client_conversation_state=_client(b0),
        refinement_type="size_upgrade",
        continuity_serialized={"current_aircraft": "Phenom 300"},
    )
    assert b1.state.active_aircraft == "Phenom 300"
    assert b1.state.active_category.value != "unknown"

    b2 = run_conversation_state_turn(
        query="More modern",
        client_conversation_state=_client(b1),
        refinement_type="style_shift",
        continuity_serialized={"current_aircraft": "Phenom 300", "style_preferences": ["modern"]},
    )
    assert "modern" in " ".join(b2.state.aesthetic_preferences).lower()

    b3 = run_conversation_state_turn(
        query="Now cockpit",
        client_conversation_state=_client(b2),
        refinement_type="view_change",
        continuity_serialized={"current_aircraft": "Phenom 300", "last_requested_view": "cockpit"},
    )
    assert b3.state.active_aircraft == "Phenom 300"
    assert b3.state.last_visual_context == "cockpit"


def test_tail_n628ts_preserved():
    b0 = run_conversation_state_turn(
        query="Have N628TS?",
        client_conversation_state=None,
        continuity_serialized={"current_tail": "N628TS", "locked_entity": {"type": "tail", "value": "N628TS"}},
    )
    assert b0.state.active_tail == "N628TS"

    b1 = run_conversation_state_turn(
        query="Cockpit too",
        client_conversation_state=_client(b0),
        refinement_type="view_change",
        continuity_serialized={"current_tail": "N628TS"},
    )
    assert b1.state.active_tail == "N628TS"
    assert b1.state.last_visual_context == "cockpit"


def test_explicit_reset_clears_memory():
    b0 = run_conversation_state_turn(
        query="Show Phenom 300",
        client_conversation_state=None,
        entity_models=["Phenom 300"],
    )
    b1 = run_conversation_state_turn(
        query="Start over — new topic",
        client_conversation_state=_client(b0),
        refinement_type="explicit_reset",
    )
    assert b1.state.active_aircraft is None
    assert b1.state.turn_index == 0


def test_decay_drops_stale_visual_after_many_turns():
    b = run_conversation_state_turn(
        query="ok",
        client_conversation_state={
            "conversation_memory": {
                "schema_version": 1,
                "turn_index": 20,
                "active_aircraft": "G650",
                "last_visual_context": "interior",
                "field_turns": {
                    "active_aircraft": 18,
                    "last_visual_context": 1,
                },
            }
        },
    )
    assert b.state.active_aircraft == "G650"
    assert "last_visual_context" in b.decayed_fields or b.state.last_visual_context is None
