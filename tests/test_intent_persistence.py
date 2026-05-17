"""Intent Persistence Engine — unit tests."""

from __future__ import annotations

from services.intent_persistence import run_intent_persistence_turn
from services.intent_persistence.schemas import IntentResponseMode, RoutingDecision


def test_phenom_interior_then_bigger_inherits_aircraft():
    prev = {
        "continuity": {
            "schema_version": 1,
            "current_aircraft": "Phenom 300",
            "last_requested_view": "interior",
            "response_mode": "short_caption",
        }
    }
    b = run_intent_persistence_turn(
        raw_user_query="Actually bigger",
        isolated_query="Actually bigger",
        history=[{"role": "user", "content": "Show me Phenom 300 interior"}],
        client_conversation_state=prev,
        strict_tail_candidates=[],
    )
    assert b.refinement_type == "size_upgrade"
    assert "phenom 300" in (b.effective_query or "").lower()
    assert b.standalone_confidence < 0.55
    assert b.routing_decision in (
        RoutingDecision.REFINEMENT_CONTINUATION,
        RoutingDecision.INHERIT_CONTEXT,
        RoutingDecision.IMAGE_SHOWCASE_CONTINUATION,
    )
    assert b.resolved_intent.get("active_aircraft") == "Phenom 300"


def test_falcon_interior_then_cockpit_too():
    prev = {
        "continuity": {
            "schema_version": 1,
            "current_aircraft": "Falcon 8X",
            "last_requested_view": "interior",
            "response_mode": "visual_only",
        }
    }
    b = run_intent_persistence_turn(
        raw_user_query="Cockpit too",
        isolated_query="Cockpit too",
        history=[{"role": "user", "content": "Show me Falcon 8X interior"}],
        client_conversation_state=prev,
        strict_tail_candidates=[],
    )
    assert b.refinement_type == "view_change"
    assert "falcon 8x" in (b.effective_query or "").lower()
    assert b.resolved_intent.get("active_visual_focus") == "cockpit"
    assert b.routing_decision == RoutingDecision.IMAGE_SHOWCASE_CONTINUATION
    assert b.force_gallery_intent is True


def test_image_showcase_mode_persists_on_vague_followup():
    prev = {
        "intent_persistence": {
            "active_aircraft": "Citation Latitude",
            "response_mode": "image_showcase",
            "active_visual_focus": "interior",
        },
        "continuity": {
            "schema_version": 1,
            "current_aircraft": "Citation Latitude",
            "response_mode": "visual_only",
        },
    }
    b = run_intent_persistence_turn(
        raw_user_query="more modern",
        isolated_query="more modern",
        history=[],
        client_conversation_state=prev,
        strict_tail_candidates=[],
    )
    assert b.resolved_intent.get("response_mode") == IntentResponseMode.IMAGE_SHOWCASE.value
    assert b.suppress_faa_registry_lookup is True
