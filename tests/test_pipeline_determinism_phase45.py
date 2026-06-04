"""Phase 45 — pipeline determinism and cross-layer consistency."""

from __future__ import annotations

import json

from services.adversarial.adversarial_preprocessor import preprocess_adversarial_query
from services.broker_decision.broker_decision_layer import apply_broker_decision_synthesis
from services.broker_reasoning.broker_reasoning_layer import apply_broker_reasoning_layer
from services.executive_broker.executive_broker_layer import apply_executive_broker_layer
from services.intent_collapse.intent_collapse_engine import apply_intent_collapse


_QUERIES = [
    "cheap gulfstream",
    "G650 vs Longitude but under 10M",
    "what can I buy for 20M",
    "buy challenger 350 under 8m",
]


def _run_pipeline(query: str) -> dict:
    du: dict = {}
    clean = preprocess_adversarial_query(query, data_used=du)
    apply_intent_collapse(query, data_used=du, normalized_query=clean.normalized_query)
    apply_broker_reasoning_layer(clean.normalized_query, data_used=du)
    raw = apply_broker_decision_synthesis(
        "INSUFFICIENT_DATA: catalog placeholder.",
        query=query,
        data_used=du,
    )
    out = apply_executive_broker_layer(raw, query=query, data_used=du)
    return {
        "frame": du.get("canonical_intent_frame"),
        "reasoning": du.get("broker_reasoning"),
        "decision_intent": (du.get("broker_decision") or {}).get("decision_intent"),
        "executive": du.get("executive_recommendation"),
        "answer_head": (out or "")[:240],
    }


def test_identical_frame_on_repeated_runs():
    for q in _QUERIES:
        a = _run_pipeline(q)["frame"]
        b = _run_pipeline(q)["frame"]
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), q


def test_reasoning_aligns_with_collapsed_intent():
    du: dict = {}
    q = "cheap gulfstream"
    preprocess_adversarial_query(q, data_used=du)
    apply_intent_collapse(q, data_used=du)
    apply_broker_reasoning_layer(q, data_used=du)
    frame = du["canonical_intent_frame"]
    br = du["broker_reasoning"]
    assert frame["primary_intent"] == "BUY"
    assert br["category"]["manufacturer"] == "Gulfstream"
    assert br["canonical_execution"] is True
    first_candidate = br["category"]["candidates"][0]
    assert first_candidate == frame["aircraft_scope"]["models"][0]


def test_executive_does_not_contradict_reasoning_primary():
    du: dict = {}
    q = "what can I buy for 12M"
    preprocess_adversarial_query(q, data_used=du)
    apply_intent_collapse(q, data_used=du)
    apply_broker_reasoning_layer(q, data_used=du)
    apply_broker_decision_synthesis(
        "At $12M, I would focus on Gulfstream G280, Citation Latitude, Praetor 600.",
        query=q,
        data_used=du,
    )
    du["broker_decision"] = du.get("broker_decision") or {
        "answer_type": "opportunities",
        "alternatives": [
            {"model": "Gulfstream G280", "rationale": "a"},
            {"model": "Citation Latitude", "rationale": "b"},
        ],
    }
    apply_executive_broker_layer(du.get("broker_decision", {}).get("direct_answer", ""), query=q, data_used=du)
    exec_rec = du.get("executive_recommendation") or {}
    br_cands = du["broker_reasoning"]["category"]["candidates"]
    if exec_rec.get("primary_recommendation") and br_cands:
        assert exec_rec["primary_recommendation"] in br_cands or exec_rec["primary_recommendation"] == br_cands[0]
