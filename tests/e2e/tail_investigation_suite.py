"""
Phase 53 — tail investigation accuracy (registry + listing dispatch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from tests.e2e.benchmark_audit_helpers import BenchmarkRecorder, BenchmarkRow
from tests.e2e.broker_certification_helpers import broker_certify

pytestmark = pytest.mark.deterministic

_REPORT = "tail_investigation_report.md"
_recorder = BenchmarkRecorder("Tail Investigation Accuracy (Phase 53)", _REPORT)


@dataclass(frozen=True)
class TailCase:
    scenario_id: str
    query: str
    expected_registration: str
    expect_valuation_authority: bool = True


TAIL_CASES: List[TailCase] = [
    TailCase("n650gs", "is N650GS worth investigating", "N650GS"),
    TailCase("n650gs_diligence", "Should I investigate tail N650GS before making an offer?", "N650GS"),
    TailCase("n800xx", "Is N800XX worth looking at?", "N800XX"),
    TailCase("n525ab", "Worth investigating N525AB?", "N525AB"),
    TailCase("n200qs", "N200QS — worth investigating?", "N200QS"),
    TailCase("n44pj", "is N44PJ worth investigating for acquisition", "N44PJ"),
]


@pytest.fixture(scope="session", autouse=True)
def _tail_report():
    global _recorder
    _recorder = BenchmarkRecorder("Tail Investigation Accuracy (Phase 53)", _REPORT)
    yield
    rows = _recorder.rows
    n = len(rows) or 1
    reg = 100.0 * sum(r.metrics.get("registry_ok", 0) for r in rows) / n
    auth = 100.0 * sum(r.metrics.get("authority_ok", 0) for r in rows) / n
    listing = 100.0 * sum(r.metrics.get("listing_mode_ok", 0) for r in rows) / n
    tail_acc = (reg + auth + listing) / 3.0
    _recorder.write_report(
        [
            f"| Registry lookup accuracy | {reg:.1f}% |",
            f"| Listing/dispatch match | {listing:.1f}% |",
            f"| Valuation authority rate | {auth:.1f}% |",
            f"| **Tail accuracy (composite)** | **{tail_acc:.1f}%** |",
        ]
    )


@pytest.fixture(autouse=True)
def _tail_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard):
    yield


def _evaluate(case: TailCase) -> dict:
    from rag.aviation_tail import find_strict_tail_candidates_in_text
    from services.market_reality.listing_detector import ListingMode, detect_listing_signal

    q = case.query
    regs = find_strict_tail_candidates_in_text(q)
    registry_ok = 1.0 if case.expected_registration.upper() in {r.upper() for r in regs} else 0.0

    signal = detect_listing_signal(q)
    listing_ok = 1.0 if signal.mode == ListingMode.TAIL_INVESTIGATION else 0.0
    if signal.registrations and case.expected_registration.upper() not in {
        r.upper() for r in signal.registrations
    }:
        listing_ok = 0.0

    answer, du, path = broker_certify(q, prefer_e2e=True)
    auth = str(du.get("authority_dispatch_kind") or "")
    authority_ok = 1.0 if (auth == "valuation" or du.get("tail_investigation_dispatch")) else 0.0
    if not case.expect_valuation_authority:
        authority_ok = 1.0

    answer_ok = 1.0 if case.expected_registration.upper() in (answer or "").upper() else 0.0

    return {
        "registry_ok": registry_ok,
        "listing_mode_ok": listing_ok,
        "authority_ok": authority_ok,
        "answer_mentions_reg": answer_ok,
        "path": path,
        "authority": auth,
        "regs_found": regs,
    }


@pytest.mark.parametrize("case", TAIL_CASES, ids=lambda c: c.scenario_id)
def test_tail_investigation_accuracy(case: TailCase):
    metrics = _evaluate(case)
    passed = (
        metrics["registry_ok"] >= 1.0
        and metrics["listing_mode_ok"] >= 1.0
        and metrics["authority_ok"] >= 1.0
    )
    _recorder.record(BenchmarkRow(case.scenario_id, passed, metrics=metrics))
    assert metrics["registry_ok"] >= 1.0, f"registry miss for {case.expected_registration}"
