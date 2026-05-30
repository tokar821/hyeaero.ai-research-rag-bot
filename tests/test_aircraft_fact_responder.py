"""Aircraft fact responder — deterministic verified answers."""

import re

from services.aircraft_truth.constants import (
    UNIFIED_CATALOG_MISS_MESSAGE,
    UNIFIED_CAPABILITY_UNVERIFIED_MESSAGE,
    UNIFIED_FACT_UNVERIFIED_MESSAGE,
)
from services.fact.aircraft_fact_responder import respond_aircraft_fact


_FORBIDDEN = re.compile(
    r"\b(?:good\s+fit|recommend|mission|shortlist|compare|versus)\b",
    re.I,
)


def _assert_broker_fact(answer: str) -> None:
    assert answer
    assert not _FORBIDDEN.search(answer)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    assert 1 <= len(sentences) <= 3


def test_falcon_8x_seats_answer():
    answer = respond_aircraft_fact("Falcon 8X", "seats")
    _assert_broker_fact(answer)
    assert "13" in answer or "12" in answer
    assert answer != UNIFIED_FACT_UNVERIFIED_MESSAGE


def test_praetor_600_baggage_answer():
    answer = respond_aircraft_fact("Praetor 600", "baggage")
    _assert_broker_fact(answer)
    assert "95" in answer
    assert answer != UNIFIED_FACT_UNVERIFIED_MESSAGE


def test_challenger_350_worth_or_unverified():
    answer = respond_aircraft_fact("Challenger 350", "worth")
    _assert_broker_fact(answer)
    assert answer == UNIFIED_CATALOG_MISS_MESSAGE or "$" in answer


def test_unknown_model_returns_unverified():
    answer = respond_aircraft_fact("Totally Fake Jet 9000", "seats")
    assert answer == UNIFIED_CATALOG_MISS_MESSAGE
