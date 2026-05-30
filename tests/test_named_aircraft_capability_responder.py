"""Named aircraft capability responder — deterministic feasibility answers."""

import re

from services.aircraft_truth.constants import UNIFIED_CAPABILITY_UNVERIFIED_MESSAGE
from services.fact.named_aircraft_capability_responder import respond_aircraft_capability
from services.routing.unified_intent_execution import should_enforce_capability_path
from services.routing.unified_intent_router import UnifiedIntent, classify_unified_intent

_FORBIDDEN = re.compile(
    r"\b(?:good\s+fit|recommend|shortlist|compare|versus|alternatives?)\b",
    re.I,
)


def _assert_broker_capability(answer: str) -> None:
    assert answer
    assert not _FORBIDDEN.search(answer)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    assert 1 <= len(sentences) <= 3


def test_falcon_8x_nyc_london_feasibility():
    answer = respond_aircraft_capability(
        "Falcon 8X",
        "Can a Falcon 8X fly nonstop from New York to London?",
    )
    _assert_broker_capability(answer)
    assert (
        "feasible" in answer.lower()
        or "realistic" in answer.lower()
        or answer.lower().startswith("yes,")
        or " can fly " in answer.lower()
    )
    assert answer != UNIFIED_CAPABILITY_UNVERIFIED_MESSAGE


def test_capability_without_route_returns_unverified():
    answer = respond_aircraft_capability("Falcon 8X", "Can a Falcon 8X fly nonstop?")
    assert answer == UNIFIED_CAPABILITY_UNVERIFIED_MESSAGE


def test_should_enforce_capability_path_for_named_route_query():
    route = classify_unified_intent("Can a Falcon 8X fly nonstop from New York to London?")
    assert route.intent == UnifiedIntent.OTHER
    assert should_enforce_capability_path(route) is True


def test_should_not_enforce_capability_without_resolved_model():
    route = classify_unified_intent("Can Longitude fly SFO to Paris?")
    assert route.model is None
    assert should_enforce_capability_path(route) is False


def test_fact_query_does_not_enforce_capability():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    assert should_enforce_capability_path(route) is False
