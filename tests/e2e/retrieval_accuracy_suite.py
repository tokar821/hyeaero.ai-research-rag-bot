"""
Phase 51 — retrieval accuracy benchmark (measurement only).

Generates ``backend/reports/retrieval_accuracy_report.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import pytest

from tests.e2e.benchmark_audit_helpers import (
    BenchmarkRecorder,
    BenchmarkRow,
    any_model_in_text,
    attach_audit_metadata,
    authority_matches,
    budget_close,
    model_in_text,
)
from tests.e2e.broker_certification_helpers import broker_certify

pytestmark = pytest.mark.deterministic

_REPORT = "retrieval_accuracy_report.md"
_recorder = BenchmarkRecorder("Retrieval Accuracy Report (Phase 51)", _REPORT)


@dataclass(frozen=True)
class RetrievalScenario:
    scenario_id: str
    query: str
    expected_authority_substr: str = ""
    expected_aircraft_top1: str = ""
    expected_aircraft_top3: Sequence[str] = ()
    expected_budget_musd: Optional[float] = None


RETRIEVAL_SCENARIOS: List[RetrievalScenario] = [
    RetrievalScenario("cheap_gulfstream", "cheap gulfstream", "alternative", "", ("G280", "G450", "G550")),
    RetrievalScenario("g650_18m", "g650 for 18m", "alternative", "G650", ("G650",), 18.0),
    RetrievalScenario(
        "longitude_vs_challenger",
        "longitude vs challenger 350",
        "comparison",
        "",
        ("Longitude", "Challenger 350"),
    ),
    RetrievalScenario("best_jet_20m", "best jet under 20m", "alternative", "", ("Longitude", "Challenger 350", "G280")),
    RetrievalScenario("g700_under_5m", "g700 under 5m", "alternative", "", ("G700",), 5.0),
    RetrievalScenario("tail_investigation", "is N650GS worth investigating", "valuation", "", ()),
    RetrievalScenario("buy_now_or_wait", "should I buy now or wait", "buy", "", ()),
]


@pytest.fixture(scope="session", autouse=True)
def _retrieval_report_session():
    global _recorder
    _recorder = BenchmarkRecorder("Retrieval Accuracy Report (Phase 51)", _REPORT)
    yield
    rows = _recorder.rows
    n = len(rows) or 1
    top1 = sum(r.metrics.get("top1", 0) for r in rows) / n * 100
    top3 = sum(r.metrics.get("top3", 0) for r in rows) / n * 100
    wrong_auth = sum(r.metrics.get("wrong_authority", 0) for r in rows) / n * 100
    aircraft = sum(r.metrics.get("aircraft_match", 0) for r in rows) / n * 100
    budget = sum(r.metrics.get("budget_match", 0) for r in rows) / n * 100
    _recorder.write_report(
        [
            f"| Top1 Accuracy | {top1:.1f}% |",
            f"| Top3 Accuracy | {top3:.1f}% |",
            f"| Wrong Authority % | {wrong_auth:.1f}% |",
            f"| Aircraft Match % | {aircraft:.1f}% |",
            f"| Budget Match % | {budget:.1f}% |",
        ]
    )


@pytest.fixture(autouse=True)
def _retrieval_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard):
    yield


def _aircraft_pool(trace: dict, answer: str) -> List[str]:
    pool: List[str] = []
    for m in trace.get("aircraft_detected") or []:
        if m and str(m) not in pool:
            pool.append(str(m))
    prim = trace.get("executive_primary")
    if prim and str(prim) not in pool:
        pool.insert(0, str(prim))
    return pool


@pytest.mark.parametrize("scenario", RETRIEVAL_SCENARIOS, ids=lambda s: s.scenario_id)
def test_retrieval_accuracy(scenario: RetrievalScenario):
    answer, du, path = broker_certify(scenario.query, prefer_e2e=True)
    attach_audit_metadata(answer, scenario.query, du)
    trace = du.get("broker_trace") or {}
    pool = _aircraft_pool(trace, answer)
    combined = " ".join(pool) + " " + answer

    top1 = 1.0 if (
        not scenario.expected_aircraft_top1
        or (pool and model_in_text(scenario.expected_aircraft_top1, pool[0]))
        or any_model_in_text((scenario.expected_aircraft_top1,), combined)
    ) else 0.0

    top3_targets = list(scenario.expected_aircraft_top3) or (
        [scenario.expected_aircraft_top1] if scenario.expected_aircraft_top1 else []
    )
    top3 = 1.0 if not top3_targets else (
        1.0
        if any_model_in_text(top3_targets, combined)
        or any(model_in_text(t, m) for m in pool[:3] for t in top3_targets)
        else 0.0
    )

    auth = str(
        du.get("authority_dispatch_kind")
        or trace.get("authority_selected")
        or ""
    )
    sources = " ".join(str(s) for s in (trace.get("retrieval_sources") or []))
    auth_ok = authority_matches(auth, scenario.expected_authority_substr) or authority_matches(
        sources, scenario.expected_authority_substr
    )
    wrong_auth = 0.0 if auth_ok else 1.0

    aircraft_match = top3 if top3_targets else 1.0
    budget_match = (
        1.0
        if budget_close(trace.get("budget_detected"), scenario.expected_budget_musd)
        else 0.0
    )

    passed = top1 >= 1.0 and wrong_auth < 1.0 and (not scenario.expected_budget_musd or budget_match >= 1.0)
    _recorder.record(
        BenchmarkRow(
            scenario.scenario_id,
            passed,
            metrics={
                "top1": top1,
                "top3": top3,
                "wrong_authority": wrong_auth,
                "aircraft_match": aircraft_match,
                "budget_match": budget_match,
                "path": path,
                "authority": auth,
            },
            notes=f"pool={pool[:3]}",
        )
    )
    assert path in ("e2e", "layers")
    assert passed, (
        f"{scenario.scenario_id}: authority={auth!r} expected={scenario.expected_authority_substr!r} pool={pool[:3]}"
    )
