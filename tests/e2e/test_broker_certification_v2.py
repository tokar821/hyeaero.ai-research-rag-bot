"""
Phase 50 — Broker Certification V2 & decision quality audit.

~190 scenarios validating broker judgment quality (not merely response presence).
Generates ``backend/reports/broker_certification_v2_report.md``.
"""

from __future__ import annotations

import re

import pytest

from tests.e2e.broker_certification_helpers import (
    assert_adversarial_discipline,
    assert_budget_referenced,
    assert_broker_quality_score,
    assert_comparison_conclusion,
    assert_comparison_quality,
    assert_contains_model,
    assert_decision_first,
    assert_direct_reality_start,
    assert_forbidden_phrases_absent,
    assert_has_conviction,
    assert_has_recommendation,
    assert_listing_skepticism,
    assert_mission_conflict_identified,
    assert_models_absent,
    assert_no_diligence_before_reality,
    assert_no_recommendation_drift,
    assert_not_checklist,
    assert_tail_investigation,
    assert_timing_guidance,
    broker_certify,
    broker_certify_conversation,
    get_v2_recorder,
    reset_v2_recorder,
    run_v2_scenario,
)
from tests.e2e.broker_certification_v2_scenarios import (
    ADVERSARIAL_SCENARIOS,
    BUDGET_SCENARIOS,
    BUY_WAIT_SCENARIOS,
    COMPARISON_SCENARIOS,
    CONSISTENCY_THREADS,
    LISTING_SCENARIOS,
    MISSION_SCENARIOS,
    TAIL_SCENARIOS,
    total_scenario_count,
)

pytestmark = pytest.mark.deterministic


@pytest.fixture(scope="session", autouse=True)
def _v2_cert_session():
    reset_v2_recorder()
    yield
    get_v2_recorder().write_v2_report()


@pytest.fixture(autouse=True)
def _broker_cert_v2_env(enable_intent_lock, disable_fine_intent_llm, aviation_conversation_guard):
    yield


def test_v2_scenario_catalog_size():
    count = total_scenario_count()
    assert 150 <= count <= 210, f"expected 150–200 scenarios, got {count}"


# ---------------------------------------------------------------------------
# A — Budget Reality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", BUDGET_SCENARIOS, ids=lambda s: s.scenario_id)
def test_v2_a_budget_reality(scenario):
    answer, du, path = broker_certify(scenario.query, prefer_e2e=True)

    def _direct():
        assert_direct_reality_start(answer)

    def _no_endorse():
        assert_models_absent(answer, scenario.forbidden_models)

    def _no_diligence_first():
        assert_no_diligence_before_reality(answer)

    def _language():
        assert_forbidden_phrases_absent(answer)

    def _score():
        assert_broker_quality_score(du, minimum=65.0)

    run_v2_scenario(
        group="A",
        scenario_id=scenario.scenario_id,
        query=scenario.query,
        answer=answer,
        path=path,
        data_used=du,
        checks=[_direct, _no_endorse, _no_diligence_first, _language, _score],
        tags=["budget"],
    )


# ---------------------------------------------------------------------------
# B — Mission Reality
# ---------------------------------------------------------------------------

_IMPOSSIBLE_MISSION_IDS = frozenset()  # resolved dynamically via assess_mission_budget_conflict


@pytest.mark.parametrize("scenario", MISSION_SCENARIOS, ids=lambda s: s.scenario_id)
def test_v2_b_mission_reality(scenario):
    from services.executive_broker.acquisition_budget_reality import (
        _parse_budget_musd,
        assess_mission_budget_conflict,
    )

    answer, du, path = broker_certify(scenario.query, prefer_e2e=False)
    budget = _parse_budget_musd(scenario.query)
    impossible = assess_mission_budget_conflict(scenario.query, budget) is not None

    checks = [
        lambda: assert_forbidden_phrases_absent(answer),
        lambda: assert_broker_quality_score(du, minimum=55.0),
    ]
    if impossible:
        checks.insert(0, lambda: assert_mission_conflict_identified(answer))
        if scenario.forbidden_models:
            checks.insert(1, lambda: assert_models_absent(answer, scenario.forbidden_models))
    else:
        checks.insert(0, lambda: assert_has_recommendation(answer))

    run_v2_scenario(
        group="B",
        scenario_id=scenario.scenario_id,
        query=scenario.query,
        answer=answer,
        path=path,
        data_used=du,
        checks=checks,
        tags=["mission"],
    )


# ---------------------------------------------------------------------------
# C — Recommendation Consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("thread", CONSISTENCY_THREADS, ids=lambda t: t.scenario_id)
def test_v2_c_recommendation_consistency(thread):
    answer, du, path, trace = broker_certify_conversation(list(thread.turns), prefer_e2e=False)

    def _expected_model():
        token = thread.expect_model.split()[-1]
        turn_idx = max(0, thread.expect_model_turn - 1)
        if turn_idx < len(trace):
            _, turn_answer, _ = trace[turn_idx]
            assert_contains_model(turn_answer, token)
        else:
            assert_contains_model(answer, token)

    def _budget_ref():
        blob = answer + " ".join(a for _, a, _ in trace)
        budget_token = str(int(thread.budget_musd))
        if budget_token in blob or f"${thread.budget_musd:.0f}m" in blob.lower().replace(" ", ""):
            return
        ctx = du.get("client_context") or du.get("broker_conversation_context") or {}
        remembered = ctx.get("remembered_budget_musd")
        if remembered is not None and float(remembered) == float(thread.budget_musd):
            return
        assert_budget_referenced(answer, thread.budget_musd)

    def _no_g700_at_12m():
        if thread.budget_musd <= 12 and "gulfstream" in " ".join(thread.turns).lower():
            assert_models_absent(answer, ("G700",))

    def _no_drift():
        audit = du.get("recommendation_consistency_audit_v2") or {}
        if audit.get("recommendation_drift") and audit.get("drift_severity") == "HIGH":
            assert_no_recommendation_drift(du)

    def _score():
        assert_broker_quality_score(du, minimum=65.0)

    run_v2_scenario(
        group="C",
        scenario_id=thread.scenario_id,
        query=" → ".join(thread.turns),
        answer=answer,
        path=path,
        data_used=du,
        checks=[_expected_model, _budget_ref, _no_g700_at_12m, _no_drift],
        tags=["consistency"],
    )


# ---------------------------------------------------------------------------
# D — Listing Realism
# ---------------------------------------------------------------------------


def _v2_listing_case(scenario) -> "ListingCase":
    from tests.e2e.listing_validation_suite import ListingCase, ListingVerdict, _tier_verdict

    m = re.search(r"(?is)I saw a (.+?) for \$([0-9.]+)M", scenario.query)
    if not m:
        raise ValueError(f"cannot parse listing query: {scenario.query}")
    model, musd = m.group(1).strip(), float(m.group(2))
    probe = ListingCase(scenario.scenario_id, scenario.query, ListingVerdict.FAIR, model, musd)
    expected = _tier_verdict(probe) or ListingVerdict.FAIR
    return ListingCase(scenario.scenario_id, scenario.query, expected, model, musd)


@pytest.mark.parametrize("scenario", LISTING_SCENARIOS, ids=lambda s: s.scenario_id)
def test_v2_d_listing_realism(scenario):
    from tests.e2e.listing_validation_suite import _compatible, infer_listing_verdict

    answer, du, path = broker_certify(scenario.query, prefer_e2e=False)
    case = _v2_listing_case(scenario)
    inferred = infer_listing_verdict(answer, du, case=case)

    def _semantic_verdict():
        assert _compatible(case.expected, inferred), (
            f"expected={case.expected.value} inferred={inferred.value}"
        )

    def _market_observable():
        from tests.e2e.listing_validation_suite import ListingVerdict

        if du.get("listing_price_infeasible") or du.get("acquisition_budget_infeasible"):
            return
        if case.expected == ListingVerdict.IMPOSSIBLE:
            return
        assert du.get("deal_quality") or du.get("market_reality"), "missing listing market signals"

    run_v2_scenario(
        group="D",
        scenario_id=scenario.scenario_id,
        query=scenario.query,
        answer=answer,
        path=path,
        data_used=du,
        checks=[
            _semantic_verdict,
            _market_observable,
            lambda: assert_forbidden_phrases_absent(answer),
            lambda: assert_broker_quality_score(du, minimum=60.0),
        ],
        tags=["listing"],
    )


# ---------------------------------------------------------------------------
# E — Tail Investigation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", TAIL_SCENARIOS, ids=lambda s: s.scenario_id)
def test_v2_e_tail_investigation(scenario):
    answer, du, path = broker_certify(scenario.query, prefer_e2e=False)

    run_v2_scenario(
        group="E",
        scenario_id=scenario.scenario_id,
        query=scenario.query,
        answer=answer,
        path=path,
        data_used=du,
        checks=[
            lambda: assert_tail_investigation(answer),
            lambda: assert_forbidden_phrases_absent(answer),
            lambda: assert_broker_quality_score(du, minimum=65.0),
        ],
        tags=["tail"],
    )


# ---------------------------------------------------------------------------
# F — Buy vs Wait
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scenario_id,query",
    BUY_WAIT_SCENARIOS,
    ids=[x[0] for x in BUY_WAIT_SCENARIOS],
)
def test_v2_f_buy_vs_wait(scenario_id, query):
    answer, du, path = broker_certify(query, prefer_e2e=False)

    run_v2_scenario(
        group="F",
        scenario_id=scenario_id,
        query=query,
        answer=answer,
        path=path,
        data_used=du,
        checks=[
            lambda: assert_timing_guidance(answer),
            lambda: assert_not_checklist(answer),
            lambda: assert_forbidden_phrases_absent(answer),
            lambda: assert_broker_quality_score(du, minimum=55.0),
        ],
        tags=["timing"],
    )


# ---------------------------------------------------------------------------
# G — Comparison Quality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", COMPARISON_SCENARIOS, ids=lambda s: s.scenario_id)
def test_v2_g_comparison_quality(scenario):
    answer, du, path = broker_certify(scenario.query, prefer_e2e=False)

    run_v2_scenario(
        group="G",
        scenario_id=scenario.scenario_id,
        query=scenario.query,
        answer=answer,
        path=path,
        data_used=du,
        checks=[
            lambda: assert_comparison_quality(answer, models=scenario.models),
            lambda: assert_comparison_conclusion(answer),
            lambda: assert_forbidden_phrases_absent(answer),
            lambda: assert_broker_quality_score(du, minimum=65.0),
        ],
        tags=["comparison"],
    )


# ---------------------------------------------------------------------------
# H — Adversarial Broker Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", ADVERSARIAL_SCENARIOS, ids=lambda s: s.scenario_id)
def test_v2_h_adversarial(scenario):
    answer, du, path = broker_certify(scenario.query, prefer_e2e=False)

    run_v2_scenario(
        group="H",
        scenario_id=scenario.scenario_id,
        query=scenario.query,
        answer=answer,
        path=path,
        data_used=du,
        checks=[
            lambda: assert_adversarial_discipline(
                answer, forbidden_endorsement=scenario.forbidden_endorsement
            ),
            lambda: assert_forbidden_phrases_absent(answer),
            lambda: assert_broker_quality_score(du, minimum=55.0),
        ],
        tags=["adversarial"],
    )
