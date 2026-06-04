"""
Phase 51/54 — listing realism (legacy scenarios).

Delegates inference to Phase 53 ``infer_listing_verdict``; asserts semantic correctness.
Use ``listing_validation_suite`` for certification KPIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

import pytest

from tests.e2e.benchmark_audit_helpers import BenchmarkRecorder, BenchmarkRow, attach_audit_metadata
from tests.e2e.broker_certification_helpers import broker_certify
from tests.e2e.execution_path_config import CERTIFICATION_PREFER_E2E
from tests.e2e.listing_validation_suite import ListingCase, ListingVerdict, _compatible, infer_listing_verdict

pytestmark = [pytest.mark.deterministic, pytest.mark.non_certification]

_REPORT = "listing_realism_report.md"
_recorder = BenchmarkRecorder("Listing Realism Report (Phase 51/54)", _REPORT)


class ListingLabel(str, Enum):
    REALISTIC = "REALISTIC"
    SUSPICIOUS = "SUSPICIOUS"
    GOOD_DEAL = "GOOD_DEAL"
    IMPOSSIBLE = "IMPOSSIBLE"


@dataclass(frozen=True)
class ListingScenario:
    scenario_id: str
    query: str
    expected_label: ListingLabel


LISTING_SCENARIOS: List[ListingScenario] = [
    ListingScenario("g650_18m", "I saw a G650 listed at $18M. Is this realistic?", ListingLabel.SUSPICIOUS),
    ListingScenario("g700_12m", "I found a G700 for $12M. Is this listing realistic?", ListingLabel.IMPOSSIBLE),
    ListingScenario("longitude_10m", "Longitude listed at $10M — good deal or suspicious?", ListingLabel.GOOD_DEAL),
    ListingScenario("falcon8x_14m", "Falcon 8X for $14M — is this realistic?", ListingLabel.IMPOSSIBLE),
    ListingScenario("challenger350_7m", "Challenger 350 asking $7M — realistic?", ListingLabel.SUSPICIOUS),
]


@pytest.fixture(scope="session", autouse=True)
def _listing_report_session():
    global _recorder
    _recorder = BenchmarkRecorder("Listing Realism Report (Phase 51/54)", _REPORT)
    yield
    rows = _recorder.rows
    n = len(rows) or 1
    correct = sum(r.metrics.get("correct", 0) for r in rows) / n * 100
    _recorder.write_report([f"| Correctly Identified | {correct:.1f}% |"])


@pytest.fixture(autouse=True)
def _listing_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard):
    yield


def _to_verdict(label: ListingLabel) -> ListingVerdict:
    mapping = {
        ListingLabel.REALISTIC: ListingVerdict.REALISTIC,
        ListingLabel.SUSPICIOUS: ListingVerdict.SUSPICIOUS,
        ListingLabel.GOOD_DEAL: ListingVerdict.GOOD_DEAL,
        ListingLabel.IMPOSSIBLE: ListingVerdict.IMPOSSIBLE,
    }
    return mapping[label]


@pytest.mark.parametrize("scenario", LISTING_SCENARIOS, ids=lambda s: s.scenario_id)
def test_listing_realism(scenario: ListingScenario):
    answer, du, path = broker_certify(scenario.query, prefer_e2e=CERTIFICATION_PREFER_E2E)
    attach_audit_metadata(answer, scenario.query, du)
    expected = _to_verdict(scenario.expected_label)
    case = ListingCase(scenario.scenario_id, scenario.query, expected)
    inferred = infer_listing_verdict(answer, du, case=case)
    correct = 1.0 if _compatible(expected, inferred) else 0.0
    _recorder.record(
        BenchmarkRow(
            scenario.scenario_id,
            correct >= 1.0,
            metrics={
                "correct": correct,
                "expected": expected.value,
                "inferred": inferred.value,
                "path": path,
            },
        )
    )
    assert path in ("e2e", "layers"), f"{scenario.scenario_id}: invalid path"
    assert correct >= 1.0, (
        f"{scenario.scenario_id}: expected={expected.value} inferred={inferred.value}"
    )
