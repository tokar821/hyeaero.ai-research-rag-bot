"""Greeting/identity short-circuit and explicit-only image gallery gating."""

from rag.consultant_small_talk import consultant_small_talk_reply
from rag.consultant_market_lookup import wants_consultant_aircraft_images_in_answer


def test_small_talk_greeting():
    assert "aircraft research" in (consultant_small_talk_reply("Hi") or "")
    assert consultant_small_talk_reply("Hello!") is not None


def test_small_talk_identity():
    r = consultant_small_talk_reply("who are you?")
    assert r and "Hye Aero" in r


def test_small_task_not_routed_for_comparison():
    assert consultant_small_talk_reply("Compare Challenger 601 and Citation 650") is None


def test_images_only_when_explicit():
    assert wants_consultant_aircraft_images_in_answer("Tell me all about the Gulfstream G650") is False
    assert wants_consultant_aircraft_images_in_answer("Show me photos of the G650") is True
