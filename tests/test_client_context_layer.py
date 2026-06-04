"""Phase 42 — client context and conversation memory tests."""

from __future__ import annotations

from services.client_context.acquisition_stage_detector import AcquisitionStage, detect_acquisition_stage
from services.client_context.client_context_layer import (
    apply_client_context_turn,
    finalize_client_context,
    personalize_client_answer,
)
from services.client_context.conversation_memory import ConversationMemory, update_memory_from_turn
from services.client_context.recommendation_consistency import filter_models_for_consistency
from services.client_context.broker_context_builder import BrokerConversationContext
from services.broker_decision.broker_decision_layer import apply_broker_decision_synthesis
from services.broker_reasoning.broker_reasoning_layer import apply_broker_reasoning_layer


def _turn(state: dict, query: str, history: list | None = None) -> dict:
    du: dict = {}
    apply_client_context_turn(
        query,
        data_used=du,
        history=history,
        client_conversation_state=state,
    )
    apply_broker_reasoning_layer(query, data_used=du)
    raw = apply_broker_decision_synthesis(
        "You may consider a Citation Latitude.",
        query=query,
        data_used=du,
    )
    answer = personalize_client_answer(raw, query=query, data_used=du)
    finalize_client_context(du, state, query=query, history=history)
    return du, answer


def test_three_turn_budget_and_gulfstream():
    state: dict = {}
    history: list = []

    du1, _ = _turn(state, "I have about 12M")
    assert du1["client_context"]["remembered_budget_musd"] == 12.0

    history.append({"role": "user", "content": "I have about 12M"})
    history.append({"role": "assistant", "content": "At your budget..."})
    du2, _ = _turn(state, "I like Gulfstreams", history=history)
    assert "Gulfstream" in (du2.get("client_profile", {}).get("preferred_manufacturers") or [])

    history.append({"role": "user", "content": "I like Gulfstreams"})
    du3, ans3 = _turn(state, "What should I buy?", history=history)
    assert "12" in ans3
    assert "Gulfstream" in ans3 or "gulfstream" in ans3.lower()


def test_g650_active_shopping_thread():
    state: dict = {}
    history: list = []

    du1, _ = _turn(state, "Looking at G650s")
    targets = du1.get("client_context", {}).get("remembered_targets", [])
    assert any("G650" in str(t) for t in targets)

    history.extend(
        [
            {"role": "user", "content": "Looking at G650s"},
            {"role": "assistant", "content": "G650 market context..."},
        ]
    )
    du2, _ = _turn(state, "Saw one for 18M", history=history)
    assert du2["client_context"]["stage"] in (
        AcquisitionStage.ACTIVE_SHOPPING.value,
        AcquisitionStage.NEGOTIATING.value,
    )

    history.append({"role": "user", "content": "Saw one for 18M"})
    du3, ans3 = _turn(state, "Should I buy now?", history=history)
    assert "G650" in ans3 or "buy" in ans3.lower()


def test_comparison_then_cheaper_alternatives():
    state: dict = {}
    history: list = []

    du1, _ = _turn(state, "Compare Longitude vs Praetor")
    pair = du1.get("client_profile", {}).get("inferred_preferences", {}).get("last_comparison_pair")
    assert pair and len(pair) >= 2

    history.extend(
        [
            {"role": "user", "content": "Compare Longitude vs Praetor"},
            {"role": "assistant", "content": "Side by side..."},
        ]
    )
    du2, ans2 = _turn(state, "Any cheaper options?", history=history)
    assert "Longitude" in ans2 or "Latitude" in ans2 or "Praetor" in ans2


def test_consistency_blocks_g700_at_12m_budget():
    ctx = BrokerConversationContext(
        remembered_budget_musd=12.0,
        remembered_targets=["Citation Latitude", "Praetor 600"],
        preferred_manufacturers=["Cessna", "Embraer"],
    )
    filtered = filter_models_for_consistency(
        ["Citation Latitude", "Praetor 600", "Gulfstream G700"],
        ctx,
    )
    assert not any("G700" in m for m in filtered)
    assert any("Latitude" in m for m in filtered)


def test_memory_accumulates_aircraft():
    mem = ConversationMemory()
    update_memory_from_turn(mem, "G650 vs G700")
    update_memory_from_turn(mem, "I like the Longitude")
    assert any("G650" in k for k in mem.aircraft_mentions)
    assert mem.aircraft_mentions.get("Citation Longitude", 0) >= 1


def test_stage_detection():
    assert detect_acquisition_stage("what can I buy for 20M") == AcquisitionStage.EXPLORING
    assert detect_acquisition_stage("I saw a G650 for 18M") == AcquisitionStage.ACTIVE_SHOPPING
    assert detect_acquisition_stage("Is N123AB a good deal?") == AcquisitionStage.NEGOTIATING
