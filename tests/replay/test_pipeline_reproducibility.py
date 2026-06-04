"""Phase 30 Track D — Full pipeline reproducibility."""

from __future__ import annotations

import copy

import pytest

from rag.query_service import _apply_api_contract_versioning
from services.evaluation.consultant_evaluator import attach_consultant_evaluation_if_enabled
from tests.conftest import build_comparison_payload, routing_authority_snapshot, run_full_pipeline_snapshot, run_retrieval

pytestmark = pytest.mark.deterministic

PIPELINE_QUERIES = [
    "G650 vs Falcon 8X",
    "Compare G650 vs Falcon 8X",
    "Alternatives to Longitude",
    "2016 Latitude $10M good deal?",
    "G650 vs UnknownJetXYZ",
    "G650 vs Falcon 8X vs Global 7500 under $30M",
    "What is a Citation Longitude worth?",
    "Longitude vs Challenger 3500",
    "G650 vs Falcon 8X under $70M",
    "Alternatives to Praetor 600",
]


@pytest.fixture(autouse=True)
def _pipeline_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard, enable_all_advisory_layers, monkeypatch):
    monkeypatch.setenv("ENABLE_CONSULTANT_EVALUATION", "1")


def _authority_fingerprint(snap: dict) -> dict:
    return {
        "intent_lock": snap.get("intent_lock"),
        "dispatch_authority_id": snap.get("dispatch_authority_id"),
        "authority_dispatch_kind": snap.get("authority_dispatch_kind"),
        "trace_id": snap.get("trace_id"),
        "evaluation_id": snap.get("evaluation_id"),
        "final_output_hash": snap.get("final_output_hash"),
        "execution_path": snap.get("execution_path"),
    }


@pytest.mark.parametrize("query", PIPELINE_QUERIES)
def test_full_pipeline_reproducible(mock_svc, query):
    a = run_full_pipeline_snapshot(query, svc=mock_svc)
    b = run_full_pipeline_snapshot(query, svc=mock_svc)
    assert _authority_fingerprint(a) == _authority_fingerprint(b)


@pytest.mark.parametrize("query", PIPELINE_QUERIES[:5])
def test_intent_lock_unchanged_after_advisory(mock_svc, query):
    snap = run_full_pipeline_snapshot(query, svc=mock_svc, with_advisory=True)
    assert snap["intent_lock"].get("semantic_version") == "v1"
    assert snap["dispatch_authority_id"]


@pytest.mark.parametrize("query", PIPELINE_QUERIES[:5])
def test_dispatch_kind_stable_full_pipeline(mock_svc, query):
    a = run_full_pipeline_snapshot(query, svc=mock_svc)
    b = run_full_pipeline_snapshot(query, svc=mock_svc)
    assert a["authority_dispatch_kind"] == b["authority_dispatch_kind"]


@pytest.mark.parametrize("query", PIPELINE_QUERIES[:5])
def test_trace_id_stable_full_pipeline(mock_svc, query):
    a = run_full_pipeline_snapshot(query, svc=mock_svc)
    b = run_full_pipeline_snapshot(query, svc=mock_svc)
    assert a["trace_id"] == b["trace_id"]


@pytest.mark.parametrize("query", PIPELINE_QUERIES[:5])
def test_evaluation_id_stable_full_pipeline(mock_svc, query):
    a = run_full_pipeline_snapshot(query, svc=mock_svc)
    b = run_full_pipeline_snapshot(query, svc=mock_svc)
    assert a["evaluation_id"] == b["evaluation_id"]


@pytest.mark.parametrize("query", PIPELINE_QUERIES[:5])
def test_final_output_hash_stable_when_answer_identical(mock_svc, query):
    a = run_full_pipeline_snapshot(query, svc=mock_svc)
    b = run_full_pipeline_snapshot(query, svc=mock_svc)
    if a["answer"] == b["answer"]:
        assert a["final_output_hash"] == b["final_output_hash"]


def test_advisory_layers_do_not_mutate_authority_in_full_pipeline(mock_svc):
    query = "G650 vs Falcon 8X"
    _, payload = run_retrieval(query, svc=mock_svc)
    before = routing_authority_snapshot(payload["data_used"])
    versioned = _apply_api_contract_versioning(copy.deepcopy(payload))
    versioned = attach_consultant_evaluation_if_enabled(query, versioned)
    after = routing_authority_snapshot(versioned["data_used"])
    assert after["intent_lock"] == before["intent_lock"]
    assert after["dispatch_authority_id"] == before["dispatch_authority_id"]
    assert after["authority_dispatch_kind"] == before["authority_dispatch_kind"]


def test_no_llm_in_full_pipeline_hard_queries(mock_svc):
    for query in ("G650 vs Falcon 8X", "Alternatives to Longitude", "2016 Latitude $10M good deal?"):
        snap = run_full_pipeline_snapshot(query, svc=mock_svc)
        assert snap["llm_invoked"] is False
        assert snap["execution_path"] == "authority_dispatch"


def test_safety_fallback_pipeline_reproducible(mock_svc):
    query = "G650 vs UnknownJetXYZ"
    a = run_full_pipeline_snapshot(query, svc=mock_svc)
    b = run_full_pipeline_snapshot(query, svc=mock_svc)
    assert a["trace_id"] == b["trace_id"]
    assert a["dispatch_authority_id"] == b["dispatch_authority_id"]


def test_build_comparison_payload_reproducible():
    a = build_comparison_payload()
    b = build_comparison_payload()
    snap_a = routing_authority_snapshot(a["data_used"])
    snap_b = routing_authority_snapshot(b["data_used"])
    assert snap_a == snap_b


def test_execution_trace_v2_present_in_full_pipeline(mock_svc):
    snap = run_full_pipeline_snapshot("G650 vs Falcon 8X", svc=mock_svc)
    assert snap["trace_id"]
    assert snap["final_output_hash"]


def test_budget_fail_closed_pipeline_reproducible(mock_svc):
    query = "G650 vs Falcon 8X vs Global 7500 under $30M"
    a = run_full_pipeline_snapshot(query, svc=mock_svc)
    b = run_full_pipeline_snapshot(query, svc=mock_svc)
    assert a["authority_dispatch_kind"] == b["authority_dispatch_kind"] == "comparison"
    assert a["trace_id"] == b["trace_id"]
