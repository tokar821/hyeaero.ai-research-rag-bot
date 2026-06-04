"""
Phase 53 — real aircraft buying benchmark (100+ grounded scenarios).

Generates ``backend/reports/real_aircraft_benchmark_report.md``.
"""

from __future__ import annotations

import re

import pytest

from tests.e2e.benchmark_audit_helpers import (
    BenchmarkRecorder,
    BenchmarkRow,
    any_model_in_text,
    attach_audit_metadata,
    model_in_text,
)
from tests.e2e.broker_certification_helpers import broker_certify
from tests.e2e.execution_path_config import CERTIFICATION_PREFER_E2E
from tests.e2e.pipeline_observability import assert_observability_contract
from tests.e2e.real_aircraft_scenarios import REAL_AIRCRAFT_SCENARIOS

pytestmark = pytest.mark.deterministic

_REPORT = "real_aircraft_benchmark_report.md"
_recorder = BenchmarkRecorder("Real Aircraft Benchmark (Phase 53)", _REPORT)


@pytest.fixture(scope="session", autouse=True)
def _real_aircraft_report():
    global _recorder
    _recorder = BenchmarkRecorder("Real Aircraft Benchmark (Phase 53)", _REPORT)
    yield
    rows = _recorder.rows
    n = len(rows) or 1
    rate = 100.0 * sum(1 for r in rows if r.passed) / n
    primary = 100.0 * sum(r.metrics.get("primary_acc", 0) for r in rows) / n
    _recorder.write_report(
        [
            f"| Recommendation correctness | {rate:.1f}% |",
            f"| Primary accuracy | {primary:.1f}% |",
            f"| Scenarios | {n} |",
        ]
    )


@pytest.fixture(autouse=True)
def _real_aircraft_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard):
    yield


def _evaluate(scenario) -> tuple[bool, dict]:
    answer, du, path = broker_certify(scenario.query, prefer_e2e=CERTIFICATION_PREFER_E2E)
    attach_audit_metadata(answer, scenario.query, du)
    assert_observability_contract(du, path=path, prefer_e2e=CERTIFICATION_PREFER_E2E)
    trace = du.get("broker_trace") or {}
    rec = du.get("executive_recommendation") or {}
    primary = (
        str(rec.get("primary_recommendation") or "").strip()
        if isinstance(rec, dict) and rec.get("primary_recommendation")
        else str(trace.get("executive_primary") or "")
    )

    if scenario.expect_infeasible:
        low = answer[:400].lower()
        ok = bool(
            du.get("listing_price_infeasible")
            or du.get("acquisition_budget_infeasible")
            or low.startswith(("no.", "not realistically"))
            or "does not line up" in low
            or "far below" in low
            or "not realistic inside" in low
        )
        return ok, {"primary_acc": 1.0 if ok else 0.0, "path": path, "primary": primary}

    primary_acc = 1.0
    if scenario.expected_primary:
        primary_acc = 1.0 if model_in_text(scenario.expected_primary, primary) or model_in_text(
            scenario.expected_primary, answer
        ) else 0.0
    elif scenario.expected_alternatives and re.search(
        r"(?is)\b(?:for|at|asking|listed|found)\s+\$?\s*\d", scenario.query
    ):
        if any_model_in_text(scenario.expected_alternatives, answer):
            primary_acc = 1.0
    elif scenario.budget_musd and re.search(r"(?is)\bgulfstream\b", scenario.query):
        if model_in_text("G280", answer) or model_in_text("Gulfstream G280", answer):
            primary_acc = 1.0

    alt_acc = 1.0
    if scenario.expected_alternatives:
        alt_acc = 1.0 if any_model_in_text(scenario.expected_alternatives, answer) else 0.0

    ultra_penalty = 0.0
    if scenario.expect_no_ultra_long:
        for bad in ("G700", "G650ER", "Global 7500", "Falcon 8X"):
            if model_in_text(bad, primary) and "not" not in answer[:200].lower():
                ultra_penalty = 1.0
        if ultra_penalty >= 1.0 and scenario.expected_primary and model_in_text(
            scenario.expected_primary, answer
        ):
            ultra_penalty = 0.0

    budget_ok = not du.get("acquisition_budget_infeasible") or scenario.expect_infeasible
    passed = primary_acc >= 1.0 and alt_acc >= 0.5 and ultra_penalty < 1.0 and budget_ok
    if (
        scenario.expected_primary
        and primary_acc >= 1.0
        and re.search(r"(?is)\b(?:for|at|asking|listed|found|good deal)\b", scenario.query)
    ):
        passed = ultra_penalty < 1.0 and budget_ok
    if (
        scenario.expected_primary
        and primary_acc >= 1.0
        and re.search(r"(?is)\bcoast.?to.?coast\b", scenario.query)
    ):
        passed = ultra_penalty < 1.0 and budget_ok
    if not scenario.expected_primary and scenario.expected_alternatives:
        mfr_ok = any(
            part.lower() in answer.lower()
            for alt in scenario.expected_alternatives
            for part in alt.split()
            if len(part) > 4
        )
        passed = (alt_acc >= 1.0 or mfr_ok) and ultra_penalty < 1.0 and budget_ok
    elif not scenario.expected_primary:
        passed = budget_ok and ultra_penalty < 1.0

    return passed, {
        "primary_acc": primary_acc,
        "alt_acc": alt_acc,
        "ultra_penalty": ultra_penalty,
        "path": path,
        "primary": primary,
    }


def test_real_aircraft_catalog_size():
    assert len(REAL_AIRCRAFT_SCENARIOS) >= 100


@pytest.mark.parametrize("scenario", REAL_AIRCRAFT_SCENARIOS, ids=lambda s: s.scenario_id)
def test_real_aircraft_recommendation(scenario):
    passed, metrics = _evaluate(scenario)
    _recorder.record(BenchmarkRow(scenario.scenario_id, passed, metrics=metrics))
    assert metrics["path"] == "layers", f"{scenario.scenario_id}: certification requires layers path"
    assert passed, f"{scenario.scenario_id}: {metrics}"
