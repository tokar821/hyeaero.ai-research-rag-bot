"""
Phase 53/54 — replay production query corpus (500 historical queries).

Semantic assertions: authority, drift, mission/buy primary (category policy).
"""

from __future__ import annotations

import pytest

from tests.e2e.benchmark_audit_helpers import BenchmarkRecorder, BenchmarkRow
from tests.e2e.execution_path_config import (
    MISSION_PRIMARY_MIN_RATE_PCT,
    expected_path_for_replay,
    prefer_e2e_for_replay,
)
from tests.e2e.production_audit_helpers import (
    load_golden_expectations,
    load_production_queries,
    replay_query,
    summarize_replay,
)

pytestmark = pytest.mark.deterministic

_REPORT = "production_query_replay_report.md"
_recorder = BenchmarkRecorder("Production Query Replay (Phase 53/54)", _REPORT)

_QUERIES = load_production_queries()
_GOLDEN = load_golden_expectations()


@pytest.fixture(scope="session", autouse=True)
def _replay_report():
    global _recorder, _SESSION_RECORDS
    _recorder = BenchmarkRecorder("Production Query Replay (Phase 53/54)", _REPORT)
    _SESSION_RECORDS = []
    yield
    summary = summarize_replay(_SESSION_RECORDS)
    mission_pct = summary.get("mission_primary_pct", 0.0)
    mission_sem_pct = summary.get("mission_semantic_ok_pct", 0.0)
    mission_n = sum(1 for r in _SESSION_RECORDS if r.category == "mission")
    _recorder.write_report(
        [
            f"| Queries replayed | {summary['total']} |",
            f"| Authority error rate | {summary['authority_error_pct']:.2f}% |",
            f"| Recommendation drift rate | {summary['drift_pct']:.2f}% |",
            f"| Semantic failure rate | {summary.get('semantic_fail_pct', 0):.2f}% |",
            f"| Mission queries | {mission_n} |",
            f"| Mission primary rate | {mission_pct:.1f}% |",
            f"| Mission semantic OK rate | {mission_sem_pct:.1f}% |",
            f"| Mission executive applied rate | {summary.get('mission_executive_applied_pct', 0):.1f}% |",
            f"| Buy-decision primary rate | {summary.get('buy_primary_pct', 0):.1f}% |",
            f"| Avg broker trust score | {summary['avg_trust']:.1f} |",
            f"| Trust ≥ 95 rate | {summary['trust_above_95_pct']:.1f}% |",
        ]
    )
    if mission_n > 0:
        assert mission_pct >= MISSION_PRIMARY_MIN_RATE_PCT, (
            f"Mission primary rate {mission_pct:.1f}% below floor {MISSION_PRIMARY_MIN_RATE_PCT}% "
            "(mission replay uses layers path — see tests/e2e/BENCHMARK_EXECUTION_PATHS.md)"
        )
        assert mission_sem_pct >= MISSION_PRIMARY_MIN_RATE_PCT, (
            f"Mission semantic OK rate {mission_sem_pct:.1f}% below floor {MISSION_PRIMARY_MIN_RATE_PCT}%"
        )


_SESSION_RECORDS: list = []


@pytest.fixture(autouse=True)
def _replay_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard):
    yield


@pytest.mark.parametrize("row", _QUERIES, ids=lambda r: r.get("id", ""))
def test_production_query_replay(row):
    rec = replay_query(row, _GOLDEN)
    _SESSION_RECORDS.append(rec)
    passed = not rec.authority_error and not rec.drift and rec.semantic_ok
    category = str(row.get("category") or "")
    expected_e2e = prefer_e2e_for_replay(category)
    expected_path = expected_path_for_replay(category)
    _recorder.record(
        BenchmarkRow(
            rec.query_id,
            passed,
            metrics={
                "authority": rec.authority,
                "primary": rec.primary[:40] if rec.primary else "",
                "trust": rec.trust_score,
                "drift": rec.drift,
                "category": rec.category,
                "path": rec.path,
                "semantic_failure": rec.semantic_failure,
                "has_primary": rec.has_primary,
                "mission_primary_present": rec.mission_primary_present,
                "mission_semantic_ok": rec.mission_semantic_ok,
                "prefer_e2e": rec.prefer_e2e,
                "executive_applied": rec.executive_applied,
            },
        )
    )
    assert rec.path in ("e2e", "layers"), f"{rec.query_id}: invalid path"
    assert rec.path == expected_path, (
        f"{rec.query_id}: expected path {expected_path!r} got {rec.path!r} (category={category})"
    )
    assert rec.prefer_e2e == expected_e2e, (
        f"{rec.query_id}: path policy expected prefer_e2e={expected_e2e} got {rec.prefer_e2e}"
    )
    if rec.category == "mission":
        assert rec.path == "layers", f"{rec.query_id}: mission must use layers path"
        assert rec.mission_primary_present, f"{rec.query_id}: mission_primary_present required"
        assert rec.mission_semantic_ok, f"{rec.query_id}: {rec.semantic_failure}"
        assert rec.executive_applied is True, f"{rec.query_id}: mission requires executive_applied"
    assert not rec.authority_error, f"{rec.query_id}: authority mismatch {rec.authority!r}"
    assert not rec.drift, f"{rec.query_id}: recommendation drift"
    assert rec.semantic_ok, (
        f"{rec.query_id} [{rec.category}]: semantic failure {rec.semantic_failure!r} "
        f"primary={rec.primary!r} path={rec.path}"
    )
