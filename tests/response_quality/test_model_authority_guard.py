"""Phase 34.3B — Model authority guard tests."""

from __future__ import annotations

import pytest

from services.consultant.answer_recovery import (
    materialize_llm_bundle_answer,
    recover_alternative_answer,
    recover_mission_answer,
    recover_valuation_answer,
)
from services.consultant.model_authority_guard import (
    answer_contains_unverified_aircraft,
    enforce_model_authority,
    extract_aircraft_mentions,
    register_mission_ranking_candidates,
    register_recovery_authority,
    resolve_verified_models,
)
from tests.response_quality.answer_consistency_audit import audit_answer_consistency
from tests.response_quality.broker_recommendation_audit import audit_broker_recommendation
from tests.conftest import run_retrieval
from tests.response_quality.response_audit_service import ResponseAuditService

pytestmark = pytest.mark.deterministic


def test_extract_aircraft_mentions_falcon():
    mentions = extract_aircraft_mentions("Citation Latitude and Falcon 8X compared.")
    assert any("Falcon" in m for m in mentions)


def test_resolve_verified_models_intent_lock():
    du = {"intent_lock": {"canonical_models": ["Gulfstream G650", "Falcon 8X"]}}
    models = resolve_verified_models(du)
    assert "Gulfstream G650" in models
    assert "Falcon 8X" in models


def test_resolve_verified_models_comparison_v2():
    du = {"comparison_v2": {"status": "OK", "models": ["Gulfstream G650", "Global 7500"]}}
    models = resolve_verified_models(du)
    assert "Gulfstream G650" in models
    assert "Global 7500" in models


def test_resolve_verified_models_mission_ranking_candidates():
    du = {"mission_ranking_candidates": ["Citation CJ4"]}
    models = resolve_verified_models(du)
    assert any("Citation CJ4" in m for m in models)


def test_answer_contains_unverified_aircraft_detects_drift():
    du = {"intent_lock": {"canonical_models": ["Gulfstream G650"]}}
    bad = "Primary: Citation Latitude"
    assert answer_contains_unverified_aircraft(bad, du) is True


def test_answer_contains_unverified_allowed_mission_ranking():
    du = {}
    register_mission_ranking_candidates(du, ["Citation CJ4", "Learjet 75"])
    body = "Aircraft Options:\n* Citation CJ4 — Why it fits: range.\n"
    assert answer_contains_unverified_aircraft(body, du) is False


def test_enforce_model_authority_fail_closed():
    du = {"intent_lock": {"canonical_models": []}}
    out = enforce_model_authority("Recommend Citation Latitude", du, query="8 pax TEB-LAX")
    assert "verified aircraft data" in out.lower()
    assert "Citation" not in out


def test_mission_recovery_fail_closed_or_authority_clean():
    du: dict = {}
    body = recover_mission_answer("8 pax TEB-LAX", data_used=du)
    assert body
    assert not answer_contains_unverified_aircraft(body, du)


def test_mission_recovery_with_ranking_stamp_allowed():
    du: dict = {}
    register_mission_ranking_candidates(du, ["Citation CJ4"])
    register_recovery_authority(du, ["Citation CJ4"])
    body = "Aircraft Options:\n* Citation CJ4 — Why it fits: practical range.\n"
    assert answer_contains_unverified_aircraft(body, du) is False


def test_alternative_recovery_longitude_shorthand():
    from services.consultant.answer_recovery import _resolve_alternative_source

    du: dict = {}
    assert _resolve_alternative_source("Replacement options for Longitude", du) == "Citation Longitude"
    body = recover_alternative_answer("Replacement options for Longitude", data_used=du)
    assert "Citation Longitude" in body
    assert "Challenger 350" in body or "Praetor 600" in body


def test_alternative_recovery_unresolved_source():
    body = recover_alternative_answer("Replacement options for UnknownJetXYZ", data_used={})
    assert "INSUFFICIENT_DATA" in body
    assert "Falcon" not in body and "G650" not in body


def test_alternative_recovery_resolved_source():
    body = recover_alternative_answer("Alternatives to G650", data_used={})
    assert body
    assert "Gulfstream" in body or "G650" in body
    du = {}
    # simulate post-respond stamp
    from services.comparison.alternative_pipeline_responder import respond_aircraft_alternative

    respond_aircraft_alternative("alternatives to G650", data_used=du)
    assert not answer_contains_unverified_aircraft(body, du)


def test_valuation_recovery_unresolved_model():
    body = recover_valuation_answer("What is a jet worth?", data_used={})
    assert "Aircraft:" in body
    assert "UNRESOLVED" in body
    assert "Verdict:" in body


def test_valuation_recovery_resolved_model():
    body = recover_valuation_answer("What is a 2019 Falcon 8X worth?", data_used={})
    assert "Falcon" in body
    assert "2019" in body
    du = {}
    register_recovery_authority(du, ["Falcon 8X"])
    assert not answer_contains_unverified_aircraft(body, du)


def test_materialize_enforces_authority_on_llm_bundle():
    du: dict = {}
    ans = materialize_llm_bundle_answer(query="8 pax TEB-LAX", data_used=du)
    assert ans
    assert "INSUFFICIENT_DATA" in ans or "Mission Fit" in ans
    auth = resolve_verified_models(du)
    consistency = audit_answer_consistency(answer=ans, intent_lock={}, authority_models=auth)
    assert "UNJUSTIFIED_MODEL_INSERTION" not in consistency.failures


def test_e2e_mission_no_unjustified_insertion():
    kind, payload = run_retrieval("8 pax TEB-LAX", svc=ResponseAuditService())
    ans = str(payload.get("answer") or "")
    du = payload.get("data_used") or {}
    from services.consultant.model_authority_guard import resolve_verified_models

    auth = resolve_verified_models(du)
    consistency = audit_answer_consistency(answer=ans, intent_lock=du.get("intent_lock") or {}, authority_models=auth)
    assert "UNJUSTIFIED_MODEL_INSERTION" not in consistency.failures


def test_e2e_valuation_falcon_consistency():
    kind, payload = run_retrieval("What is a 2019 Falcon 8X worth?", svc=ResponseAuditService())
    ans = str(payload.get("answer") or "")
    du = payload.get("data_used") or {}
    from services.consultant.model_authority_guard import resolve_verified_models

    auth = resolve_verified_models(du)
    consistency = audit_answer_consistency(answer=ans, intent_lock=du.get("intent_lock") or {}, authority_models=auth)
    assert "UNJUSTIFIED_MODEL_INSERTION" not in consistency.failures


def test_e2e_alternatives_to_g650():
    kind, payload = run_retrieval("Alternatives to G650", svc=ResponseAuditService())
    ans = str(payload.get("answer") or "")
    du = payload.get("data_used") or {}
    from services.consultant.model_authority_guard import resolve_verified_models

    auth = resolve_verified_models(du)
    consistency = audit_answer_consistency(answer=ans, intent_lock=du.get("intent_lock") or {}, authority_models=auth)
    assert "UNJUSTIFIED_MODEL_INSERTION" not in consistency.failures
    broker = audit_broker_recommendation(query="Alternatives to G650", answer=ans)
    assert "BROKER_BAD_AIRCRAFT" not in broker.failures


def test_comparison_catalog_allowed():
    du = {
        "comparison_v2": {"status": "OK", "models": ["Gulfstream G650", "Falcon 8X"]},
    }
    body = (
        "Verified catalog comparison:\n"
        "- Gulfstream G650: large-cabin\n- Falcon 8X: large-cabin\n"
        "VERDICT:\nChoose Gulfstream G650 if range leads."
    )
    assert answer_contains_unverified_aircraft(body, du) is False


def test_intent_lock_allowed():
    du = {"intent_lock": {"canonical_models": ["Gulfstream G650"]}}
    body = "Recommend Gulfstream G650 for this leg."
    assert answer_contains_unverified_aircraft(body, du) is False
