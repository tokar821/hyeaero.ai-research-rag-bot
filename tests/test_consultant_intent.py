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
    assert registry_sql_enabled_for_intent(ic, "History of 550-1234", None) is True

    ic_loose = IntentClassification(
        primary=ConsultantIntent.GENERAL_AVIATION,
        source="test",
        aviation_intent=AviationIntent.GENERAL_QUESTION,
        query_kind=ConsultantQueryKind.GENERAL,
    )
    assert registry_sql_enabled_for_intent(ic_loose, "N98765", None) is True


def test_query_kind_on_classification():
    ic = classify_consultant_intent("Compare CJ3 and Phenom 300 range", None)
    assert ic.query_kind.value == "comparison"
    assert ic.asdict().get("query_kind") == "comparison"
