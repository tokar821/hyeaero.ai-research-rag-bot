"""
Phase 53/54 — recommendation distribution and model-selection bias audit.

Certification suite: asserts bias rules and records distribution.
"""

from __future__ import annotations

import pytest

from tests.e2e.benchmark_audit_helpers import BenchmarkRecorder, BenchmarkRow, attach_audit_metadata
from tests.e2e.broker_certification_helpers import broker_certify
from tests.e2e.execution_path_config import CERTIFICATION_PREFER_E2E
from tests.e2e.pipeline_observability import assert_observability_contract
from tests.e2e.production_audit_helpers import _primary_is_meaningful, _resolve_primary
from tests.e2e.real_aircraft_scenarios import REAL_AIRCRAFT_SCENARIOS

pytestmark = pytest.mark.deterministic

_REPORT = "market_recommendation_audit_report.md"
_recorder = BenchmarkRecorder("Market Recommendation Audit (Phase 53/54)", _REPORT)

_ACQUISITION_SCENARIOS = [
    s
    for s in REAL_AIRCRAFT_SCENARIOS
    if s.budget_musd is not None
    and s.budget_musd >= 5.0
    and not getattr(s, "expect_infeasible", False)
    and ("buy" in s.query.lower() or "best" in s.query.lower() or "should" in s.query.lower())
][:60]


@pytest.fixture(scope="session", autouse=True)
def _market_audit_report():
    global _recorder, _DISTRIBUTION_RECORDS
    _recorder = BenchmarkRecorder("Market Recommendation Audit (Phase 53/54)", _REPORT)
    _DISTRIBUTION_RECORDS = []
    yield
    from tests.e2e.production_audit_helpers import detect_selection_bias, primary_distribution

    dist = primary_distribution(_DISTRIBUTION_RECORDS)
    flags = detect_selection_bias(dist)
    total = sum(dist.values()) or 1
    lines = [f"| Acquisition queries sampled | {total} |"]
    for model, count in dist.most_common(8):
        lines.append(f"| {model} | {100.0 * count / total:.1f}% |")
    if flags:
        lines.append("| **Bias flags** | " + "; ".join(flags) + " |")
    _recorder.write_report(lines)


_DISTRIBUTION_RECORDS: list = []


@pytest.fixture(autouse=True)
def _market_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard):
    yield


def _bias_ok(scenario, primary: str) -> bool:
    if primary and scenario.budget_musd and scenario.budget_musd >= 18:
        if "G280" in primary and "G650" not in scenario.query and "G700" not in scenario.query:
            return False
    return True


@pytest.mark.parametrize("scenario", _ACQUISITION_SCENARIOS, ids=lambda s: s.scenario_id)
def test_recommendation_distribution(scenario):
    from tests.e2e.production_audit_helpers import ReplayRecord

    answer, du, path = broker_certify(scenario.query, prefer_e2e=CERTIFICATION_PREFER_E2E)
    attach_audit_metadata(answer, scenario.query, du)
    assert path == "layers", f"{scenario.scenario_id}: certification requires layers path"
    assert_observability_contract(du, path=path, prefer_e2e=CERTIFICATION_PREFER_E2E)
    primary = _resolve_primary(du)
    rec = ReplayRecord(scenario.scenario_id, "acquisition", scenario.query, primary=primary, path=path)
    _DISTRIBUTION_RECORDS.append(rec)

    ok = _bias_ok(scenario, primary)
    infeasible = bool(du.get("acquisition_budget_infeasible") or du.get("listing_price_infeasible"))
    needs_primary = not getattr(scenario, "expect_infeasible", False) and not infeasible
    has_primary = _primary_is_meaningful(primary)
    _recorder.record(
        BenchmarkRow(
            scenario.scenario_id,
            ok and (has_primary or not needs_primary),
            metrics={
                "primary": primary,
                "budget_musd": scenario.budget_musd,
                "executive_applied": du.get("executive_applied"),
                "infeasible": infeasible,
            },
        )
    )
    if needs_primary:
        assert has_primary, f"{scenario.scenario_id}: missing primary_recommendation"
    assert ok, f"{scenario.scenario_id}: G280 bias on ${scenario.budget_musd}M query without Gulfstream ask"
