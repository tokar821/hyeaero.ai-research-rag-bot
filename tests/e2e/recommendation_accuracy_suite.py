"""
Phase 51 — aircraft recommendation accuracy benchmark (measurement only).

Generates ``backend/reports/recommendation_accuracy_report.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pytest

from tests.e2e.benchmark_audit_helpers import (
    BenchmarkRecorder,
    BenchmarkRow,
    any_model_in_text,
    attach_audit_metadata,
    model_in_text,
)
from tests.e2e.broker_certification_helpers import broker_certify

pytestmark = pytest.mark.deterministic

_REPORT = "recommendation_accuracy_report.md"
_recorder = BenchmarkRecorder("Recommendation Accuracy Report (Phase 51)", _REPORT)


@dataclass(frozen=True)
class RecommendationScenario:
    scenario_id: str
    query: str
    expected_primary: str = ""
    expected_alternatives: Tuple[str, ...] = ()
    expect_infeasible: bool = False
    budget_musd: Optional[float] = None
    mission_markers: Tuple[str, ...] = ()


RECOMMENDATION_SCENARIOS: List[RecommendationScenario] = [
    RecommendationScenario(
        "gulfstream_under_12m",
        "I want a Gulfstream under $12M",
        expected_primary="G280",
        budget_musd=12.0,
    ),
    RecommendationScenario(
        "coast_to_coast_6pax_20m",
        "Coast-to-coast nonstop, 6 passengers, $20M budget — what should I buy?",
        expected_primary="Longitude",
        expected_alternatives=("Challenger 350",),
        budget_musd=20.0,
        mission_markers=("nonstop", "coast", "6"),
    ),
    RecommendationScenario(
        "g700_under_5m",
        "Can I buy a G700 for $5M?",
        expect_infeasible=True,
        expected_primary="",
        budget_musd=5.0,
    ),
    RecommendationScenario(
        "g650_18m",
        "G650 for $18M — is that plausible?",
        expected_primary="G650",
        expected_alternatives=(),
        budget_musd=18.0,
    ),
    RecommendationScenario(
        "best_jet_15m",
        "Best super-midsize jet under $15M",
        expected_primary="",
        expected_alternatives=("Longitude", "Challenger 350", "Praetor 600"),
        budget_musd=15.0,
    ),
]


@pytest.fixture(scope="session", autouse=True)
def _recommendation_report_session():
    global _recorder
    _recorder = BenchmarkRecorder("Recommendation Accuracy Report (Phase 51)", _REPORT)
    yield
    rows = _recorder.rows
    n = len(rows) or 1
    primary = sum(r.metrics.get("primary_acc", 0) for r in rows) / n * 100
    alt = sum(r.metrics.get("alt_acc", 0) for r in rows) / n * 100
    budget = sum(r.metrics.get("budget_acc", 0) for r in rows) / n * 100
    mission = sum(r.metrics.get("mission_acc", 0) for r in rows) / n * 100
    _recorder.write_report(
        [
            f"| Primary Recommendation Accuracy | {primary:.1f}% |",
            f"| Alternative Accuracy | {alt:.1f}% |",
            f"| Budget Compliance | {budget:.1f}% |",
            f"| Mission Compliance | {mission:.1f}% |",
        ]
    )


@pytest.fixture(autouse=True)
def _recommendation_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard):
    yield


def _primary_from_trace(trace: dict, answer: str) -> str:
    return str(trace.get("executive_primary") or "")


def _budget_compliant(answer: str, du: dict, scenario: RecommendationScenario) -> float:
    if scenario.expect_infeasible:
        first = (answer or "").split("\n\n")[0].lower()
        if du.get("acquisition_budget_infeasible") or first.startswith(("no.", "not realistically")):
            return 1.0
        return 0.0
    if du.get("acquisition_budget_infeasible"):
        return 0.0
    return 1.0


def _mission_compliant(answer: str, scenario: RecommendationScenario) -> float:
    if not scenario.mission_markers:
        return 1.0
    low = answer.lower()
    hits = sum(1 for m in scenario.mission_markers if m.lower() in low)
    return 1.0 if hits >= min(1, len(scenario.mission_markers) // 2) else 0.5


@pytest.mark.parametrize("scenario", RECOMMENDATION_SCENARIOS, ids=lambda s: s.scenario_id)
def test_recommendation_accuracy(scenario: RecommendationScenario):
    answer, du, path = broker_certify(scenario.query, prefer_e2e=True)
    attach_audit_metadata(answer, scenario.query, du)
    trace = du.get("broker_trace") or {}
    primary = _primary_from_trace(trace, answer)

    if scenario.expect_infeasible:
        primary_acc = 1.0 if not model_in_text("G700", primary) and not (
            model_in_text("G700", answer) and "no." not in answer[:80].lower()
        ) else 0.0
        alt_acc = 1.0
    elif scenario.expected_primary:
        primary_acc = 1.0 if model_in_text(scenario.expected_primary, primary) or model_in_text(
            scenario.expected_primary, answer
        ) else 0.0
        alt_acc = 1.0
    else:
        primary_acc = 1.0
        alt_acc = (
            1.0
            if not scenario.expected_alternatives
            else (1.0 if any_model_in_text(scenario.expected_alternatives, answer) else 0.0)
        )

    budget_acc = _budget_compliant(answer, du, scenario)
    mission_acc = _mission_compliant(answer, scenario)

    passed = primary_acc >= 1.0 and budget_acc >= 1.0 and (scenario.expect_infeasible or alt_acc >= 0.5)
    _recorder.record(
        BenchmarkRow(
            scenario.scenario_id,
            passed,
            metrics={
                "primary_acc": primary_acc,
                "alt_acc": alt_acc,
                "budget_acc": budget_acc,
                "mission_acc": mission_acc,
                "path": path,
                "primary": primary,
            },
        )
    )
    assert path in ("e2e", "layers")
    assert passed, f"{scenario.scenario_id}: primary={primary!r} metrics={_recorder.rows[-1].metrics}"
