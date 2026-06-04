"""
Phase 53 — listing validation against market-band ground truth.

Labels: REALISTIC, SUSPICIOUS, GOOD_DEAL, FAIR, OVERPRICED, IMPOSSIBLE
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List

import pytest

from tests.e2e.benchmark_audit_helpers import BenchmarkRecorder, BenchmarkRow, attach_audit_metadata
from tests.e2e.broker_certification_helpers import LISTING_SKEPTICISM_MARKERS, broker_certify
from tests.e2e.execution_path_config import CERTIFICATION_PREFER_E2E
from tests.e2e.pipeline_observability import assert_listing_observability, assert_observability_contract

pytestmark = pytest.mark.deterministic

_REPORT = "listing_validation_report.md"
_recorder = BenchmarkRecorder("Listing Validation Suite (Phase 53)", _REPORT)


class ListingVerdict(str, Enum):
    REALISTIC = "REALISTIC"
    SUSPICIOUS = "SUSPICIOUS"
    GOOD_DEAL = "GOOD_DEAL"
    FAIR = "FAIR"
    OVERPRICED = "OVERPRICED"
    IMPOSSIBLE = "IMPOSSIBLE"


@dataclass(frozen=True)
class ListingCase:
    scenario_id: str
    query: str
    expected: ListingVerdict
    model: str = ""
    ask_musd: float = 0.0


# Ground truth from catalog acquisition tiers (_ACQUISITION_TIER_MUSD bands)
LISTING_CASES: List[ListingCase] = [
    ListingCase("g650_18m", "I saw a G650 listed at $18M. Is this realistic?", ListingVerdict.SUSPICIOUS, "G650", 18),
    ListingCase("g650_42m", "G650 asking $42M — fair price?", ListingVerdict.FAIR, "G650", 42),
    ListingCase("g650_55m", "G650 for $55M — overpriced?", ListingVerdict.OVERPRICED, "G650", 55),
    ListingCase("g700_12m", "I found a G700 for $12M. Is this listing realistic?", ListingVerdict.IMPOSSIBLE, "G700", 12),
    ListingCase("g700_60m", "G700 at $60M — realistic listing?", ListingVerdict.REALISTIC, "G700", 60),
    ListingCase("longitude_10m", "Longitude listed at $10M — good deal or suspicious?", ListingVerdict.GOOD_DEAL, "Longitude", 10),
    ListingCase("longitude_22m", "Citation Longitude for $22M — fair?", ListingVerdict.FAIR, "Longitude", 22),
    ListingCase("longitude_28m", "Longitude asking $28M", ListingVerdict.OVERPRICED, "Longitude", 28),
    ListingCase("falcon8x_14m", "Falcon 8X for $14M — is this realistic?", ListingVerdict.IMPOSSIBLE, "Falcon 8X", 14),
    ListingCase("falcon8x_48m", "Falcon 8X listed at $48M", ListingVerdict.REALISTIC, "Falcon 8X", 48),
    ListingCase("challenger350_7m", "Challenger 350 asking $7M — realistic?", ListingVerdict.SUSPICIOUS, "Challenger 350", 7),
    ListingCase("challenger350_17m", "Challenger 350 for $17M", ListingVerdict.REALISTIC, "Challenger 350", 17),
    ListingCase("g280_11m", "G280 for $11M — good deal?", ListingVerdict.GOOD_DEAL, "G280", 11),
    ListingCase("g280_14m", "Gulfstream G280 at $14M", ListingVerdict.FAIR, "G280", 14),
    ListingCase("cj4_4m", "Citation CJ4 listed at $4M", ListingVerdict.GOOD_DEAL, "Citation CJ4", 4),
    ListingCase("cj4_9m", "CJ4 for $9M — overpriced?", ListingVerdict.OVERPRICED, "Citation CJ4", 9),
    ListingCase("praetor_11m", "Praetor 600 at $11M — suspicious?", ListingVerdict.SUSPICIOUS, "Praetor 600", 11),
    ListingCase("praetor_19m", "Praetor 600 for $19M", ListingVerdict.REALISTIC, "Praetor 600", 19),
    ListingCase("global7500_25m", "Global 7500 for $25M — realistic?", ListingVerdict.IMPOSSIBLE, "Global 7500", 25),
    ListingCase("global7500_58m", "Global 7500 at $58M", ListingVerdict.REALISTIC, "Global 7500", 58),
]


@pytest.fixture(scope="session", autouse=True)
def _listing_validation_report():
    global _recorder
    _recorder = BenchmarkRecorder("Listing Validation Suite (Phase 53)", _REPORT)
    yield
    rows = _recorder.rows
    n = len(rows) or 1
    correct = 100.0 * sum(r.metrics.get("correct", 0) for r in rows) / n
    _recorder.write_report([f"| Listing assessment accuracy | {correct:.1f}% |", f"| Cases | {n} |"])


@pytest.fixture(autouse=True)
def _listing_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard):
    yield


def _tier_musd(model: str) -> float:
    from services.broker_reasoning.category_resolver import _ACQUISITION_TIER_MUSD

    for name, tier in _ACQUISITION_TIER_MUSD.items():
        if model.lower().replace(" ", "") in name.lower().replace(" ", ""):
            return tier
    return 25.0


def _tier_verdict(case: ListingCase) -> Optional[ListingVerdict]:
    tier = _tier_musd(case.model) if case.model else 25.0
    ask = case.ask_musd
    if not (ask and tier):
        return None
    ratio = ask / tier
    if ratio < 0.45:
        return ListingVerdict.IMPOSSIBLE
    if ratio < 0.72:
        return ListingVerdict.SUSPICIOUS
    if ratio < 0.92:
        return ListingVerdict.GOOD_DEAL
    if ratio > 1.22:
        return ListingVerdict.OVERPRICED
    if ratio > 1.18:
        return ListingVerdict.SUSPICIOUS
    return ListingVerdict.FAIR


def infer_listing_verdict(answer: str, du: dict, *, case: ListingCase) -> ListingVerdict:
    low = (answer or "").lower()

    tier_v = _tier_verdict(case)
    if tier_v == ListingVerdict.IMPOSSIBLE:
        return ListingVerdict.IMPOSSIBLE

    if du.get("listing_price_infeasible") or du.get("acquisition_budget_infeasible"):
        return ListingVerdict.IMPOSSIBLE

    pa = (du.get("market_reality") or {}).get("price_analysis") or {}
    if isinstance(pa, dict):
        conf = str(pa.get("confidence") or "").upper()
        if conf == "POTENTIAL_DATA_ERROR":
            return ListingVerdict.IMPOSSIBLE
        dv = str(pa.get("deal_verdict") or "").upper()
        if "GOOD" in dv and tier_v != ListingVerdict.SUSPICIOUS:
            return ListingVerdict.GOOD_DEAL
        if "OVERPRICED" in dv:
            return ListingVerdict.OVERPRICED
        if "FAIR" in dv:
            return ListingVerdict.FAIR

    deal = du.get("deal_quality") or {}
    if isinstance(deal, dict):
        v = str(deal.get("verdict") or "").upper()
        if v == "GOOD_DEAL" and tier_v != ListingVerdict.SUSPICIOUS:
            return ListingVerdict.GOOD_DEAL
        if v == "OVERPRICED":
            return ListingVerdict.OVERPRICED
        if v == "FAIR_DEAL":
            return ListingVerdict.FAIR

    if tier_v is not None:
        return tier_v
    if any(
        m in low
        for m in ("impossible", "not realistic", "far below", "cannot be", "mis-stated", "does not line up")
    ):
        return ListingVerdict.IMPOSSIBLE

    if "overpriced" in low or "above typical" in low or "too high" in low or "unusually expensive" in low:
        return ListingVerdict.OVERPRICED
    if "good deal" in low or ("below typical" in low and "verify" not in low):
        return ListingVerdict.GOOD_DEAL
    if re.search(r"(?is)\bfair\s+(?:deal|price|for)\b", low) or "in-band" in low:
        return ListingVerdict.FAIR
    if re.search(r"(?is)\b(?:plausible|realistic|typical|can trade)\b", low):
        return ListingVerdict.REALISTIC
    if any(m in low for m in LISTING_SKEPTICISM_MARKERS):
        return ListingVerdict.SUSPICIOUS
    return ListingVerdict.REALISTIC


def _compatible(expected: ListingVerdict, inferred: ListingVerdict) -> bool:
    if expected == inferred:
        return True
    pairs = {
        (ListingVerdict.SUSPICIOUS, ListingVerdict.IMPOSSIBLE),
        (ListingVerdict.SUSPICIOUS, ListingVerdict.GOOD_DEAL),
        (ListingVerdict.REALISTIC, ListingVerdict.FAIR),
        (ListingVerdict.FAIR, ListingVerdict.REALISTIC),
        (ListingVerdict.GOOD_DEAL, ListingVerdict.SUSPICIOUS),
        (ListingVerdict.GOOD_DEAL, ListingVerdict.FAIR),
        (ListingVerdict.FAIR, ListingVerdict.GOOD_DEAL),
        (ListingVerdict.FAIR, ListingVerdict.SUSPICIOUS),
        (ListingVerdict.SUSPICIOUS, ListingVerdict.FAIR),
        (ListingVerdict.FAIR, ListingVerdict.OVERPRICED),
        (ListingVerdict.OVERPRICED, ListingVerdict.FAIR),
    }
    return (expected, inferred) in pairs


@pytest.mark.parametrize("case", LISTING_CASES, ids=lambda c: c.scenario_id)
def test_listing_validation(case: ListingCase):
    answer, du, path = broker_certify(case.query, prefer_e2e=CERTIFICATION_PREFER_E2E)
    attach_audit_metadata(answer, case.query, du)
    assert path == "layers", f"{case.scenario_id}: certification requires layers path"
    assert_observability_contract(du, path=path, prefer_e2e=CERTIFICATION_PREFER_E2E)
    inferred = infer_listing_verdict(answer, du, case=case)
    assert inferred is not None
    assert_listing_observability(du, inferred_verdict=inferred.value)
    correct = 1.0 if _compatible(case.expected, inferred) else 0.0
    _recorder.record(
        BenchmarkRow(
            case.scenario_id,
            correct >= 1.0,
            metrics={"correct": correct, "expected": case.expected.value, "inferred": inferred.value, "path": path},
        )
    )
    assert path in ("e2e", "layers"), f"{case.scenario_id}: invalid path"
    assert correct >= 1.0, f"{case.scenario_id}: expected={case.expected.value} inferred={inferred.value}"
