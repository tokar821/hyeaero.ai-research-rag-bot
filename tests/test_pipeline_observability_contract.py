"""CI contract: broker_certify observability and certification path policy."""

from __future__ import annotations

import pytest

from tests.e2e.broker_certification_helpers import broker_certify
from tests.e2e.execution_path_config import (
    CERTIFICATION_PREFER_E2E,
    REPLAY_CATEGORY_PATH,
    expected_path_for_replay,
    prefer_e2e_for_replay,
)
from tests.e2e.listing_validation_suite import LISTING_CASES, infer_listing_verdict
from tests.e2e.pipeline_observability import (
    REQUIRED_OBSERVABILITY_KEYS,
    assert_listing_observability,
    assert_mission_execution_contract,
    assert_observability_contract,
    assert_required_observability_keys,
    attach_observability,
    build_execution_result,
)
from tests.e2e.production_audit_helpers import _resolve_primary


@pytest.mark.parametrize(
    "query,prefer_e2e,expected_path",
    [
        ("G650 asking $42M — fair price?", False, "layers"),
        ("Coast-to-coast nonstop, 6 passengers, $20M — what should I buy?", False, "layers"),
    ],
)
def test_broker_certify_observability_tags(query: str, prefer_e2e: bool, expected_path: str):
    answer, du, path = broker_certify(query, prefer_e2e=prefer_e2e)
    assert path == expected_path
    assert_observability_contract(du, path=path, prefer_e2e=prefer_e2e)
    assert answer.strip()
    assert du["execution_path"] == path
    assert isinstance(du["deal_quality_observed"], bool)
    assert isinstance(du["tier_fallback_used"], bool)
    if expected_path == "layers" and "buy" in query.lower():
        assert du["executive_applied"] is True


@pytest.mark.parametrize(
    "category,expected_path",
    [
        ("mission", "layers"),
        ("buy_decision", "e2e"),
        ("comparison", "e2e"),
        ("valuation", "e2e"),
        ("alternative", "e2e"),
    ],
)
def test_replay_category_path_policy(category: str, expected_path: str):
    assert REPLAY_CATEGORY_PATH[category] == expected_path
    assert expected_path_for_replay(category) == expected_path
    assert prefer_e2e_for_replay(category) == (expected_path == "e2e")


def test_required_observability_key_set():
    assert len(REQUIRED_OBSERVABILITY_KEYS) == 8


def test_attach_observability_maps_execution_only():
    du = {
        "executive_recommendation": {"primary_recommendation": "Gulfstream G280"},
        "executive_broker_layer_applied": 1,
        "deal_quality": {"verdict": "FAIR_DEAL"},
    }
    execution = build_execution_result(du, path="layers", prefer_e2e=False)
    attach_observability(du, execution)
    assert_required_observability_keys(du)
    assert du["executive_applied"] is True
    assert du["deal_quality_observed"] is True


def test_mission_execution_contract_on_layers():
    query = "Coast-to-coast nonstop, 6 passengers, $20M — what should I buy?"
    answer, du, path = broker_certify(query, prefer_e2e=False)
    assert_observability_contract(du, path=path, prefer_e2e=False)
    primary = _resolve_primary(du)
    assert_mission_execution_contract(du, path=path, primary=primary)
    assert answer.strip()


def test_listing_case_observability_contract():
    case = LISTING_CASES[0]
    answer, du, path = broker_certify(case.query, prefer_e2e=CERTIFICATION_PREFER_E2E)
    assert path == "layers"
    inferred = infer_listing_verdict(answer, du, case=case)
    assert_listing_observability(du, inferred_verdict=inferred.value)
    assert_observability_contract(du, path=path, prefer_e2e=CERTIFICATION_PREFER_E2E)


def test_e2e_path_executive_not_applied():
    answer, du, path = broker_certify("G650 vs Falcon 8X", prefer_e2e=True)
    if path == "e2e":
        assert_observability_contract(du, path=path, prefer_e2e=True)
        assert du["executive_applied"] is False
    else:
        pytest.skip("e2e unavailable in environment")
