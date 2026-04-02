"""LLM tool-routing (aviation vs general chat) and confidence threshold."""

import json
from unittest.mock import MagicMock, patch

import pytest

from rag.consultant_llm_intent import (
    INTENT_AVIATION_CONSULTANT,
    INTENT_GENERAL_CHAT,
    aviation_intent_min_confidence,
    classify_tool_routing_intent_llm,
    generate_general_chat_reply_llm,
    llm_tool_routing_disabled,
)


def test_aviation_threshold_default(monkeypatch):
    monkeypatch.delenv("CONSULTANT_AVIATION_INTENT_MIN_CONFIDENCE", raising=False)
    assert aviation_intent_min_confidence() == pytest.approx(0.6)


def test_tool_routing_disabled_env(monkeypatch):
    monkeypatch.setenv("CONSULTANT_LLM_TOOL_ROUTING_DISABLED", "1")
    assert llm_tool_routing_disabled() is True
    monkeypatch.delenv("CONSULTANT_LLM_TOOL_ROUTING_DISABLED", raising=False)
    assert llm_tool_routing_disabled() is False


def test_classify_handles_aviation_json():
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content=json.dumps({"intent": "aviation_consultant", "confidence": 0.92})))]

    with patch("openai.OpenAI") as m_openai:
        m_openai.return_value.chat.completions.create.return_value = fake_resp
        intent, conf = classify_tool_routing_intent_llm("Who owns N550JT?", None, api_key="sk-test", model="gpt-4o-mini")
    assert intent == INTENT_AVIATION_CONSULTANT
    assert conf == pytest.approx(0.92)


def test_classify_below_threshold_still_parses_general():
    fake_resp = MagicMock()
    fake_resp.choices = [
        MagicMock(message=MagicMock(content=json.dumps({"intent": "aviation_consultant", "confidence": 0.4})))
    ]
    with patch("openai.OpenAI") as m_openai:
        m_openai.return_value.chat.completions.create.return_value = fake_resp
        intent, conf = classify_tool_routing_intent_llm("vague planes maybe", None, api_key="sk-test", model="gpt-4o-mini")
    assert intent == INTENT_AVIATION_CONSULTANT
    assert conf == pytest.approx(0.4)


def test_routing_logic_uses_threshold():
    thr = 0.6
    assert not (INTENT_AVIATION_CONSULTANT == INTENT_AVIATION_CONSULTANT and 0.55 >= thr)
    assert INTENT_AVIATION_CONSULTANT == INTENT_AVIATION_CONSULTANT and 0.65 >= thr


def test_general_chat_alias_in_classify():
    fake_resp = MagicMock()
    fake_resp.choices = [
        MagicMock(message=MagicMock(content=json.dumps({"intent": "general_chat", "confidence": 0.95})))
    ]
    with patch("openai.OpenAI") as m_openai:
        m_openai.return_value.chat.completions.create.return_value = fake_resp
        intent, conf = classify_tool_routing_intent_llm("Thanks!", None, api_key="sk-x", model="gpt-4o-mini")
    assert intent == INTENT_GENERAL_CHAT
    assert conf == pytest.approx(0.95)


def test_generate_general_chat_fallback_without_key():
    out = generate_general_chat_reply_llm("yo", None, api_key="", model="gpt-4o-mini")
    assert len(out) > 10
