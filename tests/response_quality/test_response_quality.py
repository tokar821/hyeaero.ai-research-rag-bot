"""Phase 33 — pytest entrypoint for E2E response quality audits."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.response_quality.response_audit_runner import (
    REPORTS_DIR,
    audited_to_json,
    generate_broker_review_report,
    iter_cases_from_rows,
    load_broker_review_set,
    load_production_corpus,
    save_json,
)
from tests.response_quality.response_quality_score import audit_case, compute_scorecard


def _rows_for(mode: str):
    if mode == "broker_review_set":
        return load_broker_review_set()["queries"]
    if mode == "production_corpus":
        return load_production_corpus()["queries"]
    raise ValueError(mode)


def _run(mode: str, *, limit: int | None = None):
    rows = list(_rows_for(mode))
    if limit:
        rows = rows[:limit]
    cases = iter_cases_from_rows(rows)
    audited = [audit_case(c) for c in cases]
    scorecard = compute_scorecard(audited)
    return audited, scorecard


def test_response_quality_smoke_5_queries():
    audited, scorecard = _run("broker_review_set", limit=5)
    assert len(audited) == 5
    # This smoke test only asserts we can run E2E and produce a scorecard.
    assert scorecard["total_audited"] == 5


@pytest.mark.skipif(
    os.getenv("RUN_RESPONSE_QUALITY_E2E") != "1",
    reason="Set RUN_RESPONSE_QUALITY_E2E=1 to run 100-query broker review set E2E audit",
)
def test_broker_review_set_response_quality_report():
    audited, scorecard = _run("broker_review_set")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(audited_to_json(audited), REPORTS_DIR / "response_quality_results.json")
    save_json(scorecard, REPORTS_DIR / "response_quality_scorecard.json")
    generate_broker_review_report(audited, REPORTS_DIR / "broker_review_report.md")

    # STOP CONDITIONS (Phase 33): if hit, fail immediately (report already written).
    assert scorecard["finding_counts"].get("HALLUCINATED_AIRCRAFT", 0) == 0
    assert scorecard["finding_counts"].get("CROSS_MODEL_VALUATION", 0) == 0
    assert scorecard["finding_counts"].get("MISSION_INFEASIBLE_RECOMMENDATION", 0) == 0
    assert scorecard["finding_counts"].get("VERDICT_DRIFT", 0) == 0
    # NOTE: Success-criteria thresholds are reported (not asserted) in Phase 33.
    # This phase is an audit/measurement gate; only stop-conditions hard-fail the run.


@pytest.mark.skipif(
    os.getenv("RUN_RESPONSE_QUALITY_FULL_500") != "1",
    reason="Set RUN_RESPONSE_QUALITY_FULL_500=1 to run full 500-query E2E audit",
)
def test_full_production_corpus_response_quality_report():
    audited, scorecard = _run("production_corpus")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(audited_to_json(audited), REPORTS_DIR / "response_quality_results_500.json")
    save_json(scorecard, REPORTS_DIR / "response_quality_scorecard_500.json")
    generate_broker_review_report(audited, REPORTS_DIR / "broker_review_report_500.md")

    assert scorecard["finding_counts"].get("HALLUCINATED_AIRCRAFT", 0) == 0
    assert scorecard["finding_counts"].get("CROSS_MODEL_VALUATION", 0) == 0
    assert scorecard["finding_counts"].get("MISSION_INFEASIBLE_RECOMMENDATION", 0) == 0
    assert scorecard["finding_counts"].get("VERDICT_DRIFT", 0) == 0

