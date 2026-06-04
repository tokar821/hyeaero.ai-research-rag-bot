"""
Phase 55 — data-first broker routing and layer-priority certification.
"""

from __future__ import annotations

import re

import pytest

from services.broker_execution.broker_execution_category import (
    BrokerExecutionCategory,
    classify_broker_execution_category,
    executive_layer_allowed,
)
from services.broker_execution.mission_profile_gate import check_mission_profile_ready
from services.broker_reasoning.mission_interpreter import interpret_mission
from tests.e2e.broker_certification_helpers import broker_certify, broker_certify_conversation


_EXEC_PHRASE = re.compile(r"(?is)if\s+i\s+were\s+buying\s+today")


def test_tail_category_blocks_executive():
    cat = classify_broker_execution_category("Who owns N807JS?")
    assert cat == BrokerExecutionCategory.TAIL_OWNERSHIP
    assert executive_layer_allowed(cat, "Who owns N807JS?") is False


def test_comparison_category_blocks_executive():
    cat = classify_broker_execution_category("G280 vs Longitude")
    assert cat == BrokerExecutionCategory.COMPARISON
    assert executive_layer_allowed(cat, "G280 vs Longitude") is False


def test_mission_interpreter_city_pair():
    interp = interpret_mission("7 passengers Boston to Denver")
    assert interp.passengers == 7
    assert interp.route and "Boston" in interp.route and "Denver" in interp.route


def test_mission_profile_gate_ready():
    ready, profile = check_mission_profile_ready("7 passengers Boston to Denver")
    assert ready is True
    assert profile.get("passengers") == 7
    assert profile.get("route")


@pytest.mark.parametrize(
    "query,forbidden,required_patterns",
    [
        (
            "Who owns N807JS?",
            _EXEC_PHRASE,
            (r"(?is)\b(?:owner|ownership|registered|registry|tail)\b",),
        ),
        (
            "G280 vs Longitude",
            _EXEC_PHRASE,
            (r"(?is)\brange\b", r"(?is)\bcabin\b"),
        ),
    ],
)
def test_phase55_certification_layers(query, forbidden, required_patterns):
    answer, du, path = broker_certify(query, prefer_e2e=False)
    assert path == "layers"
    assert not forbidden.search(answer), f"unexpected executive phrasing in: {answer[:300]}"
    for pat in required_patterns:
        assert re.search(pat, answer), f"missing {pat!r} in: {answer[:400]}"
    if "N807JS" in query.upper():
        assert du.get("executive_layer_allowed") is False
        assert du.get("broker_execution_category") in (
            "tail_ownership",
            "tail_lookup",
            "registry_lookup",
        )


def test_phase55_mission_certification():
    answer, du, path = broker_certify("7 passengers Boston to Denver", prefer_e2e=False)
    assert path == "layers"
    profile = du.get("mission_profile") or {}
    assert profile.get("passengers") == 7
    assert profile.get("route")
    primary = (du.get("executive_recommendation") or {}).get("primary_recommendation") or ""
    if du.get("executive_broker_layer_applied"):
        assert primary or re.search(r"(?is)\b(?:longitude|g280|citation|challenger|gulfstream)\b", answer)
    else:
        assert du.get("mission_profile_complete") is True or "mission profile" in answer.lower()


def test_phase55_listing_certification():
    answer, du, path = broker_certify("2018 Challenger 350 asking 17.9M", prefer_e2e=False)
    assert path == "layers"
    low = answer.lower()
    assert "market reality" in low or du.get("market_reality")
    assert "deal quality" in low or (isinstance(du.get("deal_quality"), dict) and du["deal_quality"].get("verdict"))


def test_tail_memory_does_not_contaminate_tail_lookup():
    _, _, _, trace = broker_certify_conversation(
        ["I love Citation Latitude", "Who owns N807JS?"],
        prefer_e2e=False,
    )
    tail_answer = trace[-1][1]
    assert not _EXEC_PHRASE.search(tail_answer)
    assert not re.search(r"(?is)\bbased on.*latitude you've been discussing\b", tail_answer)
    assert trace[-1][2].get("tail_memory_isolated") is True


def test_retrieval_utilization_observability():
    _, du, _ = broker_certify("G280 vs Longitude", prefer_e2e=False)
    assert "retrieved_entities_count" in du
    assert "referenced_entities_count" in du
    assert isinstance(du.get("retrieval_utilization_low"), bool)
