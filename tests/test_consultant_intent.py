from rag.aviation_tail import find_strict_tail_candidates
from rag.consultant_fine_intent import (
    ConsultantFineIntent,
    ConsultantFineIntentResult,
    build_consultant_tool_router,
)
from rag.intent import (
    ConsultantIntent,
    classify_consultant_intent,
    registry_sql_enabled_for_intent,
)
from rag.intent.schemas import AviationIntent, ConsultantQueryKind, IntentClassification


def test_intent_ownership_before_market():
    q = "who owns N12345 and what is the asking price"
    ic = classify_consultant_intent(q, None)
    assert ic.primary == ConsultantIntent.REGISTRATION_LOOKUP


def test_intent_market_when_no_ownership():
    q = "what is the typical asking price for a Challenger 350"
    ic = classify_consultant_intent(q, None)
    assert ic.primary == ConsultantIntent.MARKET_PRICING


def test_registry_sql_only_registration_or_env():
    ic = classify_consultant_intent("who owns N999ZZ", None)
    assert ic.primary == ConsultantIntent.REGISTRATION_LOOKUP
    assert registry_sql_enabled_for_intent(ic, "who owns N999ZZ", None) is True

    ic2 = classify_consultant_intent("Challenger 350 range and fuel burn", None)
    assert registry_sql_enabled_for_intent(ic2, "Challenger 350 range", None) is False


def test_registry_sql_for_serial_lookup_and_n_tail():
    ic = classify_consultant_intent("History of 550-1234", None)
    assert ic.aviation_intent == AviationIntent.SERIAL_LOOKUP
    assert registry_sql_enabled_for_intent(ic, "History of 550-1234", None) is False

    ic_loose = IntentClassification(
        primary=ConsultantIntent.GENERAL_AVIATION,
        source="test",
        aviation_intent=AviationIntent.GENERAL_QUESTION,
        query_kind=ConsultantQueryKind.GENERAL,
    )
    assert registry_sql_enabled_for_intent(ic_loose, "N98765", None) is True


def test_tool_router_registry_sql_only_with_strict_tail():
    """OEM/MSN-style tokens do not enable FAA registration SQL; civil marks do."""
    q_serial = "who owns 550-1234"
    st = find_strict_tail_candidates(q_serial, None)
    assert st == []
    router = build_consultant_tool_router(
        ConsultantFineIntentResult(ConsultantFineIntent.OWNERSHIP_LOOKUP, 0.9, {}),
        q_serial,
        st,
    )
    assert router.registry_sql is False

    q_tail = "Registrant for N98765"
    st2 = find_strict_tail_candidates(q_tail, None)
    assert st2
    router2 = build_consultant_tool_router(
        ConsultantFineIntentResult(ConsultantFineIntent.OWNERSHIP_LOOKUP, 0.9, {}),
        q_tail,
        st2,
    )
    assert router2.registry_sql is True


def test_query_kind_on_classification():
    ic = classify_consultant_intent("Compare CJ3 and Phenom 300 range", None)
    assert ic.query_kind.value == "comparison"
    assert ic.asdict().get("query_kind") == "comparison"
