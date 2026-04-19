"""Conversation guard + explicit-only image gallery gating."""

from unittest.mock import patch

from rag.conversation_guard import (
    ALL_GREETING_REPLIES,
    ConversationMessageType,
    HYEAERO_COMPANY_REPLY,
    evaluate_conversation_guard,
)
from rag.consultant_market_lookup import (
    build_aircraft_photo_focus_tavily_query,
    wants_consultant_aircraft_images_in_answer,
)


def test_guard_greeting_and_casual():
    for q in ("Hi", "Hello!", "hey bro", "hi, bro", "good morning", "yo", "sup", "hey man"):
        r = evaluate_conversation_guard(q, None)
        assert r.message_type == ConversationMessageType.GREETING
        assert r.reply in ALL_GREETING_REPLIES


def test_guard_small_talk_examples():
    r = evaluate_conversation_guard("hi good", None)
    assert r.message_type == ConversationMessageType.SMALL_TALK
    assert r.reply and "mind" in (r.reply or "").lower()

    r2 = evaluate_conversation_guard("thanks", None)
    assert r2.message_type == ConversationMessageType.SMALL_TALK

    r3 = evaluate_conversation_guard("how are you", None)
    assert r3.message_type == ConversationMessageType.SMALL_TALK


def test_what_mean_with_history_routes_to_aviation_pipeline():
    r = evaluate_conversation_guard(
        "what's mean?",
        [{"role": "assistant", "content": "Per typical NBAA-style reserves…"}],
    )
    assert r.message_type == ConversationMessageType.AVIATION_QUERY
    assert r.reply is None


def test_guard_casual_chat_not_broker_pitch():
    """Short non-aviation chat should not read like a forced CRM handoff."""
    happy = evaluate_conversation_guard("Happy Today!", None)
    assert happy.message_type == ConversationMessageType.SMALL_TALK
    assert "tail" not in (happy.reply or "").lower()

    mean_q = evaluate_conversation_guard("what's mean?", None)
    assert mean_q.message_type == ConversationMessageType.SMALL_TALK
    assert "clarify" in (mean_q.reply or "").lower() or "plain english" in (mean_q.reply or "").lower()

    fun = evaluate_conversation_guard("haha, you are funny", None)
    assert fun.message_type == ConversationMessageType.SMALL_TALK
    assert "glad" in (fun.reply or "").lower()


def test_guard_identity():
    for q in ("who are you?", "what can you help with?"):
        r = evaluate_conversation_guard(q, None)
        assert r.message_type == ConversationMessageType.IDENTITY_QUESTION
        assert r.reply and "HyeAero.AI" in r.reply
        assert "Hye Aero" in r.reply


def test_guard_what_is_hye_aero():
    for q in (
        "What is Hye Aero?",
        "what does hyeaero do",
        "tell me about Hye Aero",
        "so, what is HyeAero?",
        "good, what's Hye Aero?",
        "Ok, how about discuss about HyeAero?",
    ):
        r = evaluate_conversation_guard(q, None)
        assert r.message_type == ConversationMessageType.IDENTITY_QUESTION
        assert r.reply == HYEAERO_COMPANY_REPLY
        assert "aviation intelligence" in (r.reply or "").lower()


def test_guard_can_i_ask_more_and_farewell_before_llm_misclassify():
    r = evaluate_conversation_guard("great! can I ask more?", None)
    assert r.message_type == ConversationMessageType.SMALL_TALK
    assert "what would you like" not in (r.reply or "").lower()

    r2 = evaluate_conversation_guard("thanks! have a great day!", None)
    assert r2.message_type == ConversationMessageType.SMALL_TALK
    body = (r2.reply or "").lower()
    assert "i'm here when" not in body
    assert "fly safe" in body or "likewise" in body or "good one" in body


def test_guard_what_can_i_ask_is_short_not_full_identity():
    for q in ("what can I ask?", "good, what can I ask?", "So, what should I ask?"):
        r = evaluate_conversation_guard(q, None)
        assert r.message_type == ConversationMessageType.IDENTITY_QUESTION
        body = (r.reply or "").lower()
        assert "broker-style" not in body and "the same topics" not in body
        assert any(
            k in body
            for k in (
                "tail",
                "business aviation",
                "registration",
                "city pair",
                "specs",
                "ownership",
            )
        )


def test_guard_arithmetic_no_tools():
    r = evaluate_conversation_guard("1+1", None)
    assert r.message_type == ConversationMessageType.NON_AVIATION_GENERAL
    assert (r.reply or "").strip() == "2."
    # Tiny math: numeric answer only — no aviation tag-along.
    assert "\n\n" not in (r.reply or "").strip()


def test_guard_calculus_integral_decline_no_main_consultant():
    q = "Can you solve this calculus problem: ∫(x^2 e^x) dx ?"
    r = evaluate_conversation_guard(q, None, openai_api_key="sk-test", chat_model="gpt-4o-mini")
    assert r.message_type == ConversationMessageType.NON_AVIATION_GENERAL
    body = (r.reply or "").lower()
    assert "business aviation" in body or "aviation" in body
    assert "e^x" not in body and "integration by parts" not in body and "x^2" not in body.replace("^", "")


def test_guard_what_is_one_plus_one():
    r = evaluate_conversation_guard("What is 1 + 1?", None)
    assert r.message_type == ConversationMessageType.NON_AVIATION_GENERAL
    assert "2" in (r.reply or "")


def test_guard_joke_aviation_themed():
    r = evaluate_conversation_guard("Tell me a joke", None)
    assert r.message_type == ConversationMessageType.NON_AVIATION_GENERAL
    body = r.reply or ""
    assert len(body) > 20
    low = body.lower()
    assert "aircraft" in low or "pilot" in low or "flight" in low or "mechanic" in low or "engine" in low


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


def test_guard_glued_tail_show_routes_to_aviation_not_small_talk():
    """Regression: 'showN140NE' has no word boundary before N — must not hit 3-word small_talk."""
    r = evaluate_conversation_guard("can you showN140NE?", None, openai_api_key="", chat_model="")
    assert r.message_type == ConversationMessageType.AVIATION_QUERY
    assert r.reply is None


def test_images_only_when_explicit():
    assert wants_consultant_aircraft_images_in_answer("Tell me all about the Gulfstream G650") is False
    assert wants_consultant_aircraft_images_in_answer("Show me photos of the G650") is True
    assert wants_consultant_aircraft_images_in_answer("What is market demand for the G650?") is False
    assert wants_consultant_aircraft_images_in_answer("Compare Challenger 601 and Citation 650") is False
    assert wants_consultant_aircraft_images_in_answer("Who owns N12345?") is False
    assert wants_consultant_aircraft_images_in_answer("show me of N807JS") is True
    assert wants_consultant_aircraft_images_in_answer("can you show me n807js?") is True
    assert wants_consultant_aircraft_images_in_answer("show me N807JS") is True
    assert wants_consultant_aircraft_images_in_answer(
        "I've never actually seen a Falcon 2000 up close — can you show me what it looks like?"
    ) is True
    assert wants_consultant_aircraft_images_in_answer("Let me see the aircraft") is True
    assert wants_consultant_aircraft_images_in_answer("show me images") is True
    assert wants_consultant_aircraft_images_in_answer("show me more") is True
    assert wants_consultant_aircraft_images_in_answer("thanks, can I see N508JS?") is True
    assert wants_consultant_aircraft_images_in_answer("can you showN140NE?") is True
    assert wants_consultant_aircraft_images_in_answer("let me see N508JA") is True
    assert wants_consultant_aircraft_images_in_answer("I wanna see N140NE") is True
    assert wants_consultant_aircraft_images_in_answer(
        "What is the best private jet cabin for long-range comfort?"
    ) is True
    # Prior turns must not keep image mode on — only the current message counts.
    hist = [
        {"role": "user", "content": "Show me photos of N807JS"},
        {"role": "assistant", "content": "Here is what we know."},
    ]
    assert wants_consultant_aircraft_images_in_answer("What is the asking price?", hist) is False


def test_photo_focus_tavily_query_infers_model_without_phly_row():
    q = build_aircraft_photo_focus_tavily_query(
        "I've never seen a Falcon 2000 — show me what it looks like",
        [],
        None,
    )
    assert q
    low = q.lower()
    assert "falcon" in low
    assert "aircraft exterior" in low or "photo" in low or "jetphotos" in low
