"""Phase 32 — Production validation test runner."""

from __future__ import annotations

import json
import os

import pytest

from tests.production_validation.broker_quality_score import compute_broker_quality_report
from tests.production_validation.report_generator import generate_all_reports
from tests.production_validation.validation_runner import load_corpus, run_validation

pytestmark = pytest.mark.deterministic


@pytest.fixture(autouse=True)
def _prod_env(enable_intent_lock, disable_fine_intent_llm):
    pass


@pytest.fixture
def sample_corpus():
    return load_corpus()


def test_corpus_has_500_queries(sample_corpus):
    assert sample_corpus["total"] == 500
    cats = {}
    for q in sample_corpus["queries"]:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    assert cats.get("comparison") == 100
    assert cats.get("buy_decision") == 100
    assert cats.get("mission") == 100
    assert cats.get("alternative") == 100
    assert cats.get("valuation") == 100


def test_golden_expectations_loaded():
    from tests.production_validation.validation_runner import load_golden

    golden = load_golden()
    assert golden["total"] == 500


@pytest.mark.parametrize("category", ["comparison", "buy_decision", "alternative", "valuation"])
def test_hard_intent_routing_sample(category):
    results = run_validation(limit=10, categories=[category])
    assert len(results) == 10
    routing_pct = sum(1 for r in results if r.routing_match) / len(results) * 100
    assert routing_pct >= 80, f"{category} routing sample below 80%: {routing_pct}"


def test_comparison_routing_accuracy_sample():
    results = run_validation(limit=50, categories=["comparison"])
    routing_pct = sum(1 for r in results if r.routing_match) / len(results) * 100
    assert routing_pct >= 95


def test_fail_closed_correctness_sample():
    results = run_validation(limit=100)
    fc_pct = sum(1 for r in results if r.fail_closed_correct) / len(results) * 100
    assert fc_pct >= 95


def test_hallucination_rate_sample():
    results = run_validation(limit=100)
    report = compute_broker_quality_report(results)
    assert report["hallucination_rate_pct"] < 5.0


def test_broker_quality_score_sample():
    results = run_validation(limit=100)
    report = compute_broker_quality_report(results)
    assert report["broker_quality_score"] >= 80


@pytest.mark.skipif(
    os.getenv("RUN_FULL_PRODUCTION_VALIDATION") != "1",
    reason="Set RUN_FULL_PRODUCTION_VALIDATION=1 for full 500-query audit",
)
def test_full_production_validation_suite():
    results = run_validation(limit=None)
    assert len(results) == 500
    report = compute_broker_quality_report(results)
    paths = generate_all_reports(report)

    assert report["routing_accuracy_pct"] >= 99.0
    assert report["dispatch_accuracy_pct"] >= 99.0
    assert report["mission_fit_accuracy_pct"] >= 95.0
    assert report["hallucination_rate_pct"] < 1.0
    assert report["fail_closed_accuracy_pct"] >= 100.0
    assert report["broker_quality_score"] >= 90.0

    assert paths["broker_review"]
    assert paths["readiness_report"]
