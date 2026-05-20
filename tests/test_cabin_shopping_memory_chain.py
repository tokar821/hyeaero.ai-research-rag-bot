"""Memory chain: modern cabin under $10M → less corporate → bigger."""

from __future__ import annotations

from services.conversation_state_engine import run_conversation_state_turn
from services.intent_persistence import run_intent_persistence_turn
from services.intent_persistence.pivot import shopping_gallery_models
from services.response_mode_router import route_response_mode


def _client(mem_bundle, intent_bundle) -> dict:
    return {
        "conversation_memory": mem_bundle.serialized,
        "continuity": intent_bundle.continuity_serialized,
        "intent_persistence": intent_bundle.resolved_intent,
        "current_budget": "$10M",
    }


def test_shopping_pivot_then_style_then_size_memory():
    q1 = "Show me modern cabin under $10M."
    anchor = shopping_gallery_models(q1)[0]

    router1 = route_response_mode(
        query=q1,
        fine_intent="aircraft_comparison",
        has_visual_intent=True,
        user_wants_gallery=True,
    )
    assert router1["mode"] == "image_showcase"

    ip1 = run_intent_persistence_turn(
        raw_user_query=q1,
        isolated_query=q1,
        history=[],
        client_conversation_state={
            "current_aircraft_reference": "G650",
            "conversation_memory": {"active_aircraft": "G650"},
        },
        strict_tail_candidates=[],
    )
    assert ip1.resolved_intent.get("active_budget_usd") == 10_000_000.0
    assert ip1.resolved_intent.get("active_aircraft") == anchor

    mem1 = run_conversation_state_turn(
        query=q1,
        client_conversation_state=None,
        continuity_serialized=ip1.continuity_serialized,
        intent_resolved=ip1.resolved_intent,
        refinement_type=ip1.refinement_type,
        user_wants_gallery=True,
        shopping_anchor_model=anchor,
    )
    assert mem1.state.active_aircraft == anchor
    assert mem1.state.active_budget_usd == 10_000_000.0

    client = _client(mem1, ip1)

    q2 = "Something less corporate."
    ip2 = run_intent_persistence_turn(
        raw_user_query=q2,
        isolated_query=q2,
        history=[{"role": "user", "content": q1}],
        client_conversation_state=client,
        strict_tail_candidates=[],
    )
    assert ip2.refinement_type == "style_shift"
    assert ip2.resolved_intent.get("active_aircraft") == "Praetor 600"
    assert ip2.resolved_intent.get("active_budget_usd") == 10_000_000.0

    mem2 = run_conversation_state_turn(
        query=q2,
        client_conversation_state=client,
        continuity_serialized=ip2.continuity_serialized,
        intent_resolved=ip2.resolved_intent,
        refinement_type="style_shift",
        shopping_anchor_model="Praetor 600",
    )
    assert mem2.state.active_aircraft == "Praetor 600"
    assert mem2.state.active_budget_usd == 10_000_000.0
    assert any("corporate" in x.lower() for x in mem2.state.negative_preferences)

    client2 = _client(mem2, ip2)
    client2["current_budget"] = "$10M"

    q3 = "Bigger."
    ip3 = run_intent_persistence_turn(
        raw_user_query=q3,
        isolated_query=q3,
        history=[
            {"role": "user", "content": q1},
            {"role": "user", "content": q2},
        ],
        client_conversation_state=client2,
        strict_tail_candidates=[],
    )
    assert ip3.refinement_type == "size_upgrade"
    assert ip3.resolved_intent.get("active_aircraft") == "Global 6000"
    assert ip3.resolved_intent.get("active_budget_usd") == 10_000_000.0
    assert "global 6000" in ip3.effective_query.lower()

    mem3 = run_conversation_state_turn(
        query=q3,
        client_conversation_state=client2,
        continuity_serialized=ip3.continuity_serialized,
        intent_resolved=ip3.resolved_intent,
        refinement_type="size_upgrade",
        shopping_anchor_model="Global 6000",
    )
    assert mem3.state.active_aircraft == "Global 6000"
    assert mem3.state.active_budget_usd == 10_000_000.0
