"""
Phase 48 — End-to-end broker certification suite.

Simulates real buyer threads and records pass/fail into
``backend/reports/broker_certification_report.md``.

Does not modify production routing or add new response layers.
"""

from __future__ import annotations

import re

import pytest

from tests.e2e.broker_certification_helpers import (
    assert_comparison_quality,
    assert_direct_reality_start,
    assert_forbidden_headers_absent,
    assert_forbidden_phrases_absent,
    assert_has_conviction,
    assert_has_recommendation,
    assert_models_absent,
    assert_no_bullet_spam,
    assert_not_checklist,
    assert_single_primary_executive,
    assert_tail_investigation,
    broker_certify,
    broker_certify_conversation,
    extract_primary_hint,
    first_paragraph,
    get_certification_recorder,
    run_cert_scenario,
)

pytestmark = pytest.mark.deterministic


@pytest.fixture(autouse=True)
def _broker_cert_e2e_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard):
    """IntentLock + guard bypass for retrieval-backed certification scenarios."""
    yield


# ---------------------------------------------------------------------------
# GROUP A — Budget realism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scenario_id,query,forbidden_models",
    [
        ("g700_at_5m", "Can I buy a G700 for $5M?", ("G700",)),
        ("g650_at_10m", "Can I buy a G650 for $10M?", ("G650", "G650ER")),
        ("longitude_at_8m", "Can I buy a Longitude for $8M?", ("G700", "G650", "G650ER", "Global 7500")),
        ("12m_gulfstream", "I only have $12M and want a Gulfstream.", ("G700", "G650ER", "Global 7500")),
    ],
)
def test_group_a_budget_realism(scenario_id, query, forbidden_models):
    answer, du, path = broker_certify(query, prefer_e2e=True)

    def _direct():
        assert_direct_reality_start(answer)

    def _no_impossible():
        assert_models_absent(answer, forbidden_models)

    run_cert_scenario(
        group="A",
        scenario_id=scenario_id,
        query=query,
        answer=answer,
        path=path,
        checks=[_direct, _no_impossible],
        tags=["budget"],
    )
    hint = extract_primary_hint(answer)
    if hint:
        get_certification_recorder().note_primary(hint)


# ---------------------------------------------------------------------------
# GROUP B — Conversation memory
# ---------------------------------------------------------------------------

def test_group_b_conversation_memory():
    turns = [
        "I have $12M.",
        "I like Gulfstreams.",
        "What should I buy?",
        "What about something newer?",
        "What if I stretch to $15M?",
    ]
    answer, du, path, trace = broker_certify_conversation(turns, prefer_e2e=False)

    def _remembers_gulfstream():
        ctx = du.get("broker_conversation_context") or du.get("client_context") or {}
        prefs = ctx.get("preferred_manufacturers") or []
        low = answer.lower()
        if "gulfstream" not in low and not any("gulfstream" in str(p).lower() for p in prefs):
            raise AssertionError("Gulfstream preference not reflected")

    def _remembers_budget():
        ctx = du.get("broker_conversation_context") or du.get("client_context") or {}
        budget = ctx.get("remembered_budget_musd")
        if budget is None and "12" not in answer and "15" not in answer:
            raise AssertionError("budget not remembered in context or answer")

    def _no_g700():
        assert_models_absent(answer, ("G700",))

    def _continuity():
        for _q, a, _du in trace[2:]:
            if "G700" in a:
                raise AssertionError("G700 appeared mid-conversation")

    run_cert_scenario(
        group="B",
        scenario_id="12m_gulfstream_thread",
        query=" → ".join(turns),
        answer=answer,
        path=path,
        checks=[_remembers_gulfstream, _remembers_budget, _no_g700, _continuity],
        tags=["continuity"],
    )


# ---------------------------------------------------------------------------
# GROUP C — Comparison quality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scenario_id,query,models",
    [
        ("g650_vs_g700", "G650 vs G700", ("G650", "G700")),
        ("longitude_vs_praetor", "Longitude vs Praetor", ("Longitude", "Praetor")),
        (
            "latitude_vs_challenger_350",
            "Latitude vs Challenger 350",
            ("Latitude", "Challenger 350"),
        ),
    ],
)
def test_group_c_comparison_quality(scenario_id, query, models):
    answer, du, path = broker_certify(query, prefer_e2e=True)

    def _comparison():
        assert_comparison_quality(answer, models=models)

    def _human():
        assert_forbidden_phrases_absent(answer)

    run_cert_scenario(
        group="C",
        scenario_id=scenario_id,
        query=query,
        answer=answer,
        path=path,
        checks=[_comparison, _human],
    )


# ---------------------------------------------------------------------------
# GROUP D — Broker realism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scenario_id,query",
    [
        ("buy_with_20m", "What would you buy with $20M?"),
        ("if_you_were_me", "What would you do if you were me? I have $18M and fly 4 passengers regionally."),
        ("should_i_wait", "Should I wait to buy a G280?"),
    ],
)
def test_group_d_broker_realism(scenario_id, query):
    answer, du, path = broker_certify(query, prefer_e2e=False)

    def _recommendation():
        assert_has_recommendation(answer)

    def _conviction():
        assert_has_conviction(answer)

    def _not_checklist():
        assert_not_checklist(answer)

    run_cert_scenario(
        group="D",
        scenario_id=scenario_id,
        query=query,
        answer=answer,
        path=path,
        checks=[_recommendation, _conviction, _not_checklist],
    )


# ---------------------------------------------------------------------------
# GROUP E — Tail investigations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scenario_id,query",
    [
        ("tail_n719gf", "N719GF"),
        ("tail_worth_looking", "Is N719GF worth looking at?"),
    ],
)
def test_group_e_tail_investigation(scenario_id, query):
    answer, du, path = broker_certify(query, prefer_e2e=False)

    def _tail():
        assert_tail_investigation(answer)

    def _no_speculation():
        low = answer.lower()
        if re.search(r"\b(great|excellent)\s+(buy|deal)\b", low):
            raise AssertionError("speculative tail endorsement")

    run_cert_scenario(
        group="E",
        scenario_id=scenario_id,
        query=query,
        answer=answer,
        path=path,
        checks=[_tail, _no_speculation],
    )


# ---------------------------------------------------------------------------
# GROUP F — Human sounding (forbidden software phrases)
# ---------------------------------------------------------------------------

_GROUP_F_QUERIES = [
    "Can I buy a G700 for $5M?",
    "G650 vs G700",
    "What would you buy with $20M?",
    "Is N719GF worth looking at?",
    "I have $20M. I fly 6 people coast-to-coast.",
]


@pytest.mark.parametrize("query", _GROUP_F_QUERIES, ids=lambda q: q[:40])
def test_group_f_human_sounding(query):
    answer, _du, path = broker_certify(query, prefer_e2e=True)

    def _forbidden():
        assert_forbidden_phrases_absent(answer)

    run_cert_scenario(
        group="F",
        scenario_id=f"human_{hash(query) % 10_000}",
        query=query,
        answer=answer,
        path=path,
        checks=[_forbidden],
        tags=["humanization"],
    )


# ---------------------------------------------------------------------------
# GROUP G — Formatting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", _GROUP_F_QUERIES, ids=lambda q: f"fmt_{q[:30]}")
def test_group_g_formatting(query):
    answer, _du, path = broker_certify(query, prefer_e2e=True)

    def _headers():
        assert_forbidden_headers_absent(answer)

    def _bullets():
        assert_no_bullet_spam(answer)

    run_cert_scenario(
        group="G",
        scenario_id=f"fmt_{hash(query) % 10_000}",
        query=query,
        answer=answer,
        path=path,
        checks=[_headers, _bullets],
        tags=["humanization", "formatting"],
    )


# ---------------------------------------------------------------------------
# GROUP H — Executive decision quality
# ---------------------------------------------------------------------------

def test_group_h_executive_decision_quality():
    query = "I have $20M. I fly 6 people coast-to-coast."
    answer, du, path = broker_certify(query, prefer_e2e=False)

    def _executive():
        assert_single_primary_executive(answer, du)

    def _human():
        assert_forbidden_phrases_absent(answer)

    run_cert_scenario(
        group="H",
        scenario_id="20m_coast_to_coast",
        query=query,
        answer=answer,
        path=path,
        checks=[_executive, _human],
    )
    hint = extract_primary_hint(answer)
    if hint:
        get_certification_recorder().note_primary(hint)
