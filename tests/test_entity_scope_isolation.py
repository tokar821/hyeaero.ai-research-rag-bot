"""Phase 10 — entity scope isolation and tail contamination regression tests."""

from __future__ import annotations

from services.conversation_continuity.entity_lock import merge_entity_lock, explicit_aircraft_switch
from services.conversation_continuity.orchestrator import run_continuity_turn
from services.conversation_continuity.refinement import reinforce_query_with_context
from services.conversation_continuity.schemas import LockedEntity, LockedEntityType, RefinementInterpretation
from services.conversation_state_engine import run_conversation_state_turn
from services.entity_scope.scope import (
    history_allowed_for_tail_resolution,
    is_deictic_tail_followup,
    resolve_entity_scope,
    should_release_tail_on_model_switch,
)
from services.entity_scope.validation import filter_phly_rows_by_entity_scope, tail_conflicts_with_aircraft
from services.intent_persistence import run_intent_persistence_turn
from rag.phlydata_consultant_lookup import consultant_phly_lookup_token_list


def _turn1_client(tail: str = "N988NW", aircraft: str = "Falcon 7X") -> dict:
    ip = run_intent_persistence_turn(
        raw_user_query=f"What is the ask price on {aircraft} {tail}?",
        isolated_query=f"What is the ask price on {aircraft} {tail}?",
        history=[],
        client_conversation_state=None,
        strict_tail_candidates=[tail],
    )
    mem = run_conversation_state_turn(
        query=f"What is the ask price on {aircraft} {tail}?",
        client_conversation_state=None,
        continuity_serialized=ip.continuity_serialized,
        intent_resolved=ip.resolved_intent,
        refinement_type=ip.refinement_type,
    )
    return {
        "conversation_memory": mem.serialized,
        "continuity": ip.continuity_serialized,
        "intent_persistence": ip.resolved_intent,
    }


def test_scenario_a_praetor_600_releases_tail_and_no_augmentation():
    client = _turn1_client()
    q2 = "Tell me about the Praetor 600"
    ip2 = run_intent_persistence_turn(
        raw_user_query=q2,
        isolated_query=q2,
        history=[{"role": "user", "content": "What is the ask price on Falcon 7X N988NW?"}],
        client_conversation_state=client,
        strict_tail_candidates=[],
    )
    assert "N988NW" not in (ip2.effective_query or "").upper()
    assert "tail n988nw" not in (ip2.effective_query or "").lower()
    resolved = ip2.resolved_intent or {}
    assert (resolved.get("active_tail") or "") == ""
    assert "praetor" in (resolved.get("active_aircraft") or "").lower()

    toks = consultant_phly_lookup_token_list(
        q2,
        [{"role": "user", "content": "What is the ask price on Falcon 7X N988NW?"}],
    )
    assert "N988NW" not in [t.upper() for t in toks]

    scope = resolve_entity_scope(q2)
    assert scope.scope_type == "aircraft_model"
    assert "praetor" in (scope.scope_value or "").lower()


def test_scenario_b_deictic_price_preserves_tail_continuity():
    client = _turn1_client()
    q2 = "What is the asking price?"
    assert is_deictic_tail_followup(q2)
    ip2 = run_intent_persistence_turn(
        raw_user_query=q2,
        isolated_query=q2,
        history=[{"role": "user", "content": "What is the ask price on Falcon 7X N988NW?"}],
        client_conversation_state=client,
        strict_tail_candidates=["N988NW"],
    )
    assert "N988NW" in (ip2.effective_query or "").upper()

    toks = consultant_phly_lookup_token_list(
        q2,
        [{"role": "user", "content": "What is the ask price on Falcon 7X N988NW?"}],
    )
    assert "N988NW" in [t.upper() for t in toks]


def test_scenario_c_comparison_no_tail_leak():
    client = _turn1_client()
    q2 = "Compare G650 vs Falcon 8X"
    ip2 = run_intent_persistence_turn(
        raw_user_query=q2,
        isolated_query=q2,
        history=[{"role": "user", "content": "What is the ask price on Falcon 7X N988NW?"}],
        client_conversation_state=client,
        strict_tail_candidates=[],
    )
    assert "N988NW" not in (ip2.effective_query or "").upper()
    assert not history_allowed_for_tail_resolution(q2)


def test_scenario_d_buy_decision_alternative_no_contamination():
    client = _turn1_client()
    q2 = "What aircraft should I buy instead of a Latitude?"
    ip2 = run_intent_persistence_turn(
        raw_user_query=q2,
        isolated_query=q2,
        history=[{"role": "user", "content": "What is the ask price on Falcon 7X N988NW?"}],
        client_conversation_state=client,
        strict_tail_candidates=[],
    )
    assert "N988NW" not in (ip2.effective_query or "").upper()
    toks = consultant_phly_lookup_token_list(
        q2,
        [{"role": "user", "content": "What is the ask price on Falcon 7X N988NW?"}],
    )
    assert "N988NW" not in [t.upper() for t in toks]


def test_scenario_e_longitude_worth_no_falcon_listing():
    client = _turn1_client()
    q2 = "What is a Citation Longitude worth?"
    ip2 = run_intent_persistence_turn(
        raw_user_query=q2,
        isolated_query=q2,
        history=[{"role": "user", "content": "What is the ask price on Falcon 7X N988NW?"}],
        client_conversation_state=client,
        strict_tail_candidates=[],
    )
    assert "N988NW" not in (ip2.effective_query or "").upper()
    scope = resolve_entity_scope(q2)
    assert scope.scope_type == "aircraft_model"
    rows, validation = filter_phly_rows_by_entity_scope(
        [
            {
                "registration_number": "N988NW",
                "manufacturer": "Dassault",
                "model": "Falcon 7X",
            }
        ],
        scope,
    )
    assert rows == []
    assert validation["rejected"] == 1


def test_merge_entity_lock_releases_on_explicit_model_switch():
    prev = LockedEntity(type=LockedEntityType.TAIL, value="N988NW", locked_at_turn_hint="turn1")
    lock = merge_entity_lock(
        prev,
        query="Tell me about the Praetor 600",
        strict_tail_candidates=["N988NW"],
        explicit_model="Praetor 600",
        prev_tail_aircraft="Falcon 7X",
        allow_history_tail=False,
    )
    assert lock is not None
    assert lock.type == LockedEntityType.AIRCRAFT_MODEL
    assert lock.value == "Praetor 600"


def test_reinforce_query_no_tail_on_explicit_model():
    out = reinforce_query_with_context(
        "Tell me about the Praetor 600",
        interpretation=RefinementInterpretation(type="none", inherit_entity=True),
        locked_tail="N988NW",
        locked_model="Falcon 7X",
        augment_size=False,
        size_augment_fragment="",
    )
    assert out.lower() == "tell me about the praetor 600"
    assert "n988nw" not in out.lower()


def test_reinforce_query_appends_tail_on_deictic_followup():
    out = reinforce_query_with_context(
        "What is the asking price?",
        interpretation=RefinementInterpretation(type="ambiguous_followup", inherit_entity=True),
        locked_tail="N988NW",
        locked_model="Falcon 7X",
        augment_size=False,
        size_augment_fragment="",
    )
    assert "N988NW" in out.upper()


def test_memory_clears_conflicting_tail_on_model_switch():
    client = _turn1_client()
    q2 = "Tell me about the Praetor 600"
    ip2 = run_intent_persistence_turn(
        raw_user_query=q2,
        isolated_query=q2,
        history=[{"role": "user", "content": "What is the ask price on Falcon 7X N988NW?"}],
        client_conversation_state=client,
        strict_tail_candidates=[],
    )
    mem2 = run_conversation_state_turn(
        query=q2,
        client_conversation_state=client,
        continuity_serialized=ip2.continuity_serialized,
        intent_resolved=ip2.resolved_intent,
        refinement_type=ip2.refinement_type,
    )
    assert mem2.state.active_aircraft and "praetor" in mem2.state.active_aircraft.lower()
    assert not mem2.state.active_tail


def test_should_release_tail_on_model_switch():
    assert should_release_tail_on_model_switch("Praetor 600", "Falcon 7X")
    assert not should_release_tail_on_model_switch("Falcon 7X", "Falcon 7X")


def test_tail_conflicts_with_aircraft():
    assert tail_conflicts_with_aircraft("N988NW", "Praetor 600", tail_aircraft="Falcon 7X")
    assert not tail_conflicts_with_aircraft("N988NW", "Falcon 7X", tail_aircraft="Falcon 7X")
