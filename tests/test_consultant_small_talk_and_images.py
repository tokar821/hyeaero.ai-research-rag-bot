"""Conversation guard + explicit-only image gallery gating."""

from unittest.mock import patch

from rag.conversation_guard import (
    ALL_GREETING_REPLIES,
    ConversationMessageType,
    HYEAERO_COMPANY_REPLY,
    evaluate_conversation_guard,
)
from rag.consultant_market_lookup import wants_consultant_aircraft_images_in_answer


def test_guard_greeting_and_casual():
    for q in ("Hi", "Hello!", "hey bro", "hi, bro", "good morning", "yo", "sup", "hey man"):
        r = evaluate_conversation_guard(q, None)
        assert r.message_type == ConversationMessageType.GREETING
        assert r.reply in ALL_GREETING_REPLIES


def test_guard_small_talk_examples():
    r = evaluate_conversation_guard("hi good", None)
    assert r.message_type == ConversationMessageType.SMALL_TALK
    assert "help" in (r.reply or "").lower()

    r2 = evaluate_conversation_guard("thanks", None)
    assert r2.message_type == ConversationMessageType.SMALL_TALK

    r3 = evaluate_conversation_guard("how are you", None)
    assert r3.message_type == ConversationMessageType.SMALL_TALK


def test_guard_identity():
    for q in ("who are you?", "what can you help with?"):
        r = evaluate_conversation_guard(q, None)
        assert r.message_type == ConversationMessageType.IDENTITY_QUESTION
        assert r.reply and "HyeAero.AI" in r.reply
        assert "Hye Aero" in r.reply


def test_guard_what_is_hye_aero():
    for q in ("What is Hye Aero?", "what does hyeaero do", "tell me about Hye Aero"):
        r = evaluate_conversation_guard(q, None)
        assert r.message_type == ConversationMessageType.IDENTITY_QUESTION
        assert r.reply == HYEAERO_COMPANY_REPLY
        assert "aviation intelligence" in (r.reply or "").lower()


def test_guard_arithmetic_no_tools():
    r = evaluate_conversation_guard("1+1", None)
    assert r.message_type == ConversationMessageType.NON_AVIATION_GENERAL
    assert "2" in (r.reply or "")
    assert "1" in (r.reply or "")
    assert "hyeaero" in (r.reply or "").lower()


def test_guard_what_is_one_plus_one():
    r = evaluate_conversation_guard("What is 1 + 1?", None)
    assert r.message_type == ConversationMessageType.NON_AVIATION_GENERAL
    assert "2" in (r.reply or "")


def test_guard_joke_aviation_themed():
    r = evaluate_conversation_guard("Tell me a joke", None)
    assert r.message_type == ConversationMessageType.NON_AVIATION_GENERAL
    body = (r.reply or "").lower()
    assert "hyeaero" in body or "aviation" in body or "aircraft" in body


def test_guard_non_aviation_capital_uses_llm_when_configured():
    fake = (
        "The capital of France is Paris.\n\n"
        "I'm HyeAero.AI — your aviation intelligence assistant for Hye Aero.\n"
        "Ask me about missions, specs, or market insights anytime."
    )
    with patch(
        "rag.conversation_guard._non_aviation_llm_reply",
        return_value=fake,
    ):
        r = evaluate_conversation_guard(
            "What is the capital of France?",
            None,
            openai_api_key="sk-test",
            chat_model="gpt-4o-mini",
        )
    assert r.message_type == ConversationMessageType.NON_AVIATION_GENERAL
    assert "Paris" in (r.reply or "")


def test_guard_aviation_proceeds():
    r = evaluate_conversation_guard(
        "Can a Citation II cross the Atlantic?",
        None,
    )
    assert r.message_type == ConversationMessageType.AVIATION_QUERY
    assert r.reply is None


def test_guard_not_triggered_for_comparison():
    r = evaluate_conversation_guard("Compare Challenger 601 and Citation 650", None)
    assert r.message_type == ConversationMessageType.AVIATION_QUERY


def test_images_only_when_explicit():
    assert wants_consultant_aircraft_images_in_answer("Tell me all about the Gulfstream G650") is False
    assert wants_consultant_aircraft_images_in_answer("Show me photos of the G650") is True
