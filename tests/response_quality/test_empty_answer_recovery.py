"""Phase 34.3A — Empty answer and recommendation recovery tests."""

from __future__ import annotations

import pytest

from services.consultant.answer_recovery import (
    materialize_llm_bundle_answer,
    recover_alternative_answer,
    recover_client_answer,
    recover_mission_answer,
    recover_valuation_answer,
)
from tests.conftest import run_retrieval
from tests.response_quality.response_audit_service import ResponseAuditService

pytestmark = pytest.mark.deterministic


def _answer(query: str) -> str:
    kind, payload = run_retrieval(query, svc=ResponseAuditService())
    assert kind in ("professional", "llm")
    return str(payload.get("answer") or "").strip()


def test_mission_query_never_empty():
    ans = _answer("8 pax TEB-LAX")
    assert ans
    assert len(ans) >= 20


def test_mission_query_includes_aircraft_options_or_insufficient():
    ans = _answer("8 passengers TEB to LAX nonstop under $15M")
    low = ans.lower()
    assert "aircraft options" in low or "insufficient_data" in low


def test_mission_budget_query_has_aircraft_reference():
    ans = _answer("Mission: 8 pax from TEB to LAX under $25M")
    assert any(k in ans for k in ("Gulfstream", "Citation", "Falcon", "Global", "INSUFFICIENT_DATA"))


def test_alternative_replacement_options_not_empty():
    ans = _answer("Replacement options for G650")
    assert ans
    assert "G650" in ans or "Gulfstream" in ans


def test_alternative_similar_aircraft_not_empty():
    ans = _answer("Similar aircraft to Falcon 8X")
    assert ans
    assert "Falcon" in ans or "8X" in ans


def test_alternative_tier_peer_phrasing():
    ans = _answer("Alternatives to G650")
    assert "tier-peer" in ans.lower() or "alternatives" in ans.lower()


def test_valuation_query_not_empty():
    ans = _answer("What is a 2019 Falcon 8X worth?")
    assert ans
    assert len(ans) >= 20


def test_valuation_includes_aircraft_line():
    ans = _answer("Estimate market value of a 2019 Falcon 8X")
    assert "Aircraft:" in ans
    assert "Falcon" in ans


def test_valuation_includes_verdict():
    ans = _answer("2019 Falcon 8X valuation")
    assert "Verdict:" in ans


def test_valuation_recovery_template_unit():
    body = recover_valuation_answer("What is a 2019 Falcon 8X worth?")
    assert "Aircraft: Falcon 8X" in body or "Aircraft: Gulfstream" in body
    assert "Year: 2019" in body
    assert "Market Reality:" in body
    assert "Verdict:" in body


def test_alternative_recovery_unit():
    body = recover_alternative_answer("Replacement options for G650")
    assert body
    assert "G650" in body or "Gulfstream" in body


def test_mission_recovery_unit_has_structure():
    body = recover_mission_answer("8 pax TEB-LAX")
    assert body
    assert "Mission Fit" in body or "INSUFFICIENT_DATA" in body


def test_materialize_llm_bundle_mission():
    ans = materialize_llm_bundle_answer(query="8 pax TEB-LAX", data_used={})
    assert ans and len(ans) >= 20


def test_recover_client_weak_safety_fallback_valuation():
    weak = "Insufficient verified data for deterministic execution.\n\nStructured valuation requires"
    body = recover_client_answer(
        query="What is a 2019 Falcon 8X worth?",
        answer=weak,
    )
    assert "Aircraft:" in body
    assert "Falcon" in body


def test_recover_preserves_good_comparison():
    good = "Verified catalog comparison:\n- Gulfstream G650: large-cabin\nVERDICT:\nChoose G650"
    du = {
        "comparison_v2": {
            "status": "OK",
            "models": ["Gulfstream G650", "Falcon 8X"],
        },
    }
    body = recover_client_answer(query="G650 vs Falcon 8X", answer=good, data_used=du)
    assert "Verified catalog comparison" in body


def test_ensure_non_empty_via_retrieval_professional_valuation():
    ans = _answer("How much is a 2019 Falcon 8X worth today?")
    assert "Aircraft:" in ans


def test_mission_what_jet_query():
    ans = _answer("What jet for 8 pax TEB-LAX?")
    assert ans and len(ans) >= 20


def test_alternative_global_7500():
    ans = _answer("Replacement options for Global 7500")
    assert "Global" in ans or "7500" in ans


def test_empty_string_recovery():
    body = recover_client_answer(query="Alternatives to G650", answer="")
    assert body


def test_whitespace_only_recovery():
    body = recover_client_answer(query="8 pax TEB-LAX", answer="   \n  ")
    assert body.strip()


def test_safety_fallback_preserves_aircraft_reference_valuation():
    ans = _answer("What is the value of a 2019 Falcon 8X?")
    assert "Aircraft:" in ans
    assert "Falcon" in ans or "8X" in ans


def test_llm_kind_returns_answer_key():
    kind, payload = run_retrieval("Replacement options for G650", svc=ResponseAuditService())
    assert "answer" in payload
    assert payload["answer"]


def test_mission_nonstop_under_budget():
    ans = _answer("8 passengers TEB to LAX nonstop under $10M")
    assert ans
    assert len(ans) >= 20
