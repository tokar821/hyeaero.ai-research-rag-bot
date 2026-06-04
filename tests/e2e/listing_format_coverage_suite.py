"""
Phase 54 — listing format / edge-case coverage (measurement).

Documents parse and verdict behavior for formats not in the core 20-case suite.
Uses Phase 53 inference; asserts observability fields where applicable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pytest

from tests.e2e.benchmark_audit_helpers import BenchmarkRecorder, BenchmarkRow, attach_audit_metadata
from tests.e2e.broker_certification_helpers import broker_certify
from tests.e2e.execution_path_config import CERTIFICATION_PREFER_E2E
from tests.e2e.listing_validation_suite import ListingCase, ListingVerdict, infer_listing_verdict

pytestmark = pytest.mark.deterministic

_REPORT = "listing_format_coverage_report.md"
_recorder = BenchmarkRecorder("Listing Format Coverage (Phase 54)", _REPORT)


@dataclass(frozen=True)
class FormatCase:
    scenario_id: str
    query: str
    model: str
    ask_musd: float
    expect_verdict: Optional[ListingVerdict] = None
    require_deal_quality: bool = True
    require_ask_parsed: bool = True


FORMAT_CASES: List[FormatCase] = [
    FormatCase("partial_cj4", "CJ4 for $5M", "Citation CJ4", 5.0),
    FormatCase("unicode_dash", "G700 at $60M — realistic listing?", "G700", 60.0),
    FormatCase("leading_ask", "$42M G650 — good deal?", "G650", 42.0),
    FormatCase("ask_range_dash", "G650 at $40M–$45M — which is fair?", "G650", 42.5, require_deal_quality=False),
    FormatCase("multi_model", "G650 or Falcon 8X at $45M — fair?", "G650", 45.0, require_deal_quality=False),
    # Known gaps — must not crash; observability only until parse extended
    FormatCase(
        "ask_no_dollar_sign",
        "G650 asking 42 million — fair?",
        "G650",
        42.0,
        require_deal_quality=False,
        require_ask_parsed=False,
    ),
    FormatCase(
        "ask_word_million",
        "Citation Longitude forty million dollars — realistic?",
        "Longitude",
        40.0,
        require_deal_quality=False,
        require_ask_parsed=False,
    ),
    FormatCase(
        "comma_thousands",
        "G280 listed at $11,500,000 — fair?",
        "G280",
        11.5,
        require_deal_quality=False,
        require_ask_parsed=False,
    ),
    FormatCase(
        "ambiguous_no_price",
        "Is this G650 listing realistic?",
        "G650",
        0.0,
        require_deal_quality=False,
        require_ask_parsed=False,
    ),
    FormatCase(
        "malformed_price",
        "G650 for $M42 — fair?",
        "G650",
        0.0,
        require_deal_quality=False,
        require_ask_parsed=False,
    ),
    FormatCase(
        "partial_global",
        "Global 7500 at $58M — realistic?",
        "Global 7500",
        58.0,
    ),
    FormatCase(
        "multi_partial_models",
        "G650 or CJ4 or Longitude around $20M — which listing is fair?",
        "G650",
        20.0,
        require_deal_quality=False,
    ),
    FormatCase(
        "word_forty_million",
        "G650 listed at forty million — suspicious?",
        "G650",
        40.0,
        require_deal_quality=False,
        require_ask_parsed=False,
    ),
]


@pytest.fixture(scope="session", autouse=True)
def _format_report():
    global _recorder
    _recorder = BenchmarkRecorder("Listing Format Coverage (Phase 54)", _REPORT)
    yield
    rows = _recorder.rows
    n = len(rows) or 1
    rate = 100.0 * sum(1 for r in rows if r.passed) / n
    _recorder.write_report([f"| Format coverage pass rate | {rate:.1f}% |", f"| Cases | {n} |"])


@pytest.fixture(autouse=True)
def _format_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard):
    yield


@pytest.mark.parametrize("case", FORMAT_CASES, ids=lambda c: c.scenario_id)
def test_listing_format_coverage(case: FormatCase):
    answer, du, path = broker_certify(case.query, prefer_e2e=CERTIFICATION_PREFER_E2E)
    attach_audit_metadata(answer, case.query, du)
    listing_case = ListingCase(
        case.scenario_id,
        case.query,
        case.expect_verdict or ListingVerdict.FAIR,
        case.model,
        case.ask_musd,
    )
    inferred = infer_listing_verdict(answer, du, case=listing_case)
    has_dq = bool(isinstance(du.get("deal_quality"), dict) and du["deal_quality"].get("verdict"))
    has_mr = bool(du.get("market_reality"))
    ask_signal = (du.get("market_reality") or {}).get("signal", {}) if isinstance(du.get("market_reality"), dict) else {}
    ask_parsed = ask_signal.get("ask_musd") is not None or case.ask_musd > 0

    verdict_ok = True
    if case.expect_verdict is not None:
        verdict_ok = inferred == case.expect_verdict or (
            case.expect_verdict == ListingVerdict.FAIR and inferred in (ListingVerdict.FAIR, ListingVerdict.REALISTIC)
        )

    passed = bool(answer.strip()) and path in ("e2e", "layers")
    if case.expect_verdict is not None:
        passed = passed and verdict_ok
    if case.require_deal_quality:
        passed = passed and has_dq
    if case.require_ask_parsed:
        passed = passed and (has_mr or has_dq)

    _recorder.record(
        BenchmarkRow(
            case.scenario_id,
            passed,
            metrics={
                "inferred": inferred.value,
                "has_deal_quality": has_dq,
                "has_market_reality": has_mr,
                "path": path,
            },
        )
    )
    assert path in ("e2e", "layers")
    assert passed, (
        f"{case.scenario_id}: inferred={inferred.value} dq={has_dq} mr={has_mr} "
        f"expected={case.expect_verdict}"
    )
