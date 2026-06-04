"""Controlled failure injection — proves benchmarks detect defects."""
import pytest

from tests.e2e.listing_validation_suite import (
    LISTING_CASES,
    ListingVerdict,
    infer_listing_verdict,
    _compatible,
)
from tests.e2e.real_aircraft_benchmark import _evaluate
from tests.e2e.real_aircraft_scenarios import REAL_AIRCRAFT_SCENARIOS


def test_baseline_real_aircraft_sample_passes():
    s = next(x for x in REAL_AIRCRAFT_SCENARIOS if x.scenario_id == "g700_65m")
    passed, _ = _evaluate(s)
    assert passed


def test_baseline_listing_sample_passes():
    case = next(c for c in LISTING_CASES if c.scenario_id == "g650_42m")
    from tests.e2e.broker_certification_helpers import broker_certify
    from tests.e2e.benchmark_audit_helpers import attach_audit_metadata
    a, du, _ = broker_certify(case.query, prefer_e2e=False)
    attach_audit_metadata(a, case.query, du)
    inf = infer_listing_verdict(a, du, case=case)
    assert _compatible(case.expected, inf)


def test_injection_wrong_aircraft_fails(monkeypatch):
    import services.executive_broker.executive_broker_layer as ebl

    def _bad(*args, **kwargs):
        from services.executive_broker.executive_recommendation import ExecutiveRecommendation
        return ExecutiveRecommendation(
            primary_recommendation="Gulfstream G280",
            rationale="injected defect",
            alternatives=[],
            confidence="HIGH",
        )

    monkeypatch.setattr(ebl, "select_executive_recommendation", _bad)
    s = next(x for x in REAL_AIRCRAFT_SCENARIOS if x.scenario_id == "coast_6pax_20m")
    passed, metrics = _evaluate(s)
    assert not passed, metrics


def test_injection_listing_always_suspicious_fails(monkeypatch):
    def _always_suspicious(answer, du, *, case):
        return ListingVerdict.SUSPICIOUS

    monkeypatch.setattr(
        "tests.e2e.listing_validation_suite.infer_listing_verdict",
        _always_suspicious,
    )
    case = next(c for c in LISTING_CASES if c.scenario_id == "g650_18m")
    assert case.expected == ListingVerdict.SUSPICIOUS

    def _always_realistic(answer, du, *, case):
        return ListingVerdict.REALISTIC

    from tests.e2e.broker_certification_helpers import broker_certify
    from tests.e2e.benchmark_audit_helpers import attach_audit_metadata

    a, du, _ = broker_certify(case.query, prefer_e2e=False)
    inf = _always_realistic(a, du, case=case)
    assert not _compatible(case.expected, inf)


def test_injection_cj4_impossible_fails(monkeypatch):
    def _always_impossible(answer, du, *, case):
        return ListingVerdict.IMPOSSIBLE

    monkeypatch.setattr(
        "tests.e2e.listing_validation_suite.infer_listing_verdict",
        _always_impossible,
    )
    case = next(c for c in LISTING_CASES if c.scenario_id == "cj4_4m")
    assert case.expected == ListingVerdict.GOOD_DEAL
    from tests.e2e.broker_certification_helpers import broker_certify
    from tests.e2e.benchmark_audit_helpers import attach_audit_metadata
    a, du, _ = broker_certify(case.query, prefer_e2e=False)
    inf = _always_impossible(a, du, case=case)
    assert not _compatible(case.expected, inf)
