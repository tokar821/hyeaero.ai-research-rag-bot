"""Architecture regression — single render path and intent lock."""

from __future__ import annotations

from services.broker_execution.client_answer_renderer import collapse_duplicate_registry_blocks
from services.broker_execution.execution_intent_lock import (
    ExecutionProfile,
    attach_execution_intent_lock,
)
from services.broker_execution.mission_feasibility_broker import build_mission_feasibility_broker_note
from services.broker_execution.output_governance import apply_governed_client_answer
from services.broker_execution.tail_depth_mode import TailDepthMode, classify_tail_depth_mode


def test_tail_depth_owner_vs_detail():
    assert classify_tail_depth_mode("Who owns N807JS?")[0] == TailDepthMode.OWNER
    assert classify_tail_depth_mode("Is N807JS for sale?")[0] == TailDepthMode.SALE_STATUS
    assert classify_tail_depth_mode("Tell me everything about N807JS")[0] == TailDepthMode.DETAIL


def test_execution_lock_suppresses_acquisition_on_owner():
    du: dict = {}
    profile = attach_execution_intent_lock(du, "Who owns N807JS?")
    assert profile == ExecutionProfile.TAIL_OWNER
    assert du.get("suppress_acquisition_tail_rewrite") == 1
    assert du.get("suppress_broker_reasoning_overlay") == 1


def test_mission_feasibility_nyc_tokyo_budget():
    note = build_mission_feasibility_broker_note(
        "8 passengers New York to Tokyo nonstop under $30M budget"
    )
    assert "NYC–Tokyo" in note or "Tokyo" in note
    assert "$30M" in note or "30" in note


def test_collapse_duplicate_registry_blocks():
    dup = (
        "Aircraft: Citation Excel\nOwner: HRL VENTURES LLC\nYear: 2003\n\n"
        "Aircraft: Citation Excel\nOwner: HRL VENTURES LLC\nYear: 2003"
    )
    out = collapse_duplicate_registry_blocks(dup)
    assert out.count("HRL VENTURES") == 1


def test_llm_primary_governance_uses_single_renderer():
    du = {"llm_executed": 1, "consultant_llm_draft": "Owner: Acme LLC"}
    out = apply_governed_client_answer(
        "Owner: Acme LLC\n\nOwner: Acme LLC",
        query="Who owns N807JS?",
        data_used=du,
    )
    assert du.get("client_answer_renderer") == "llm_primary_single_pass"
    assert "Acme LLC" in out
