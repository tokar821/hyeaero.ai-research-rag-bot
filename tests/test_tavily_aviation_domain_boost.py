"""Tavily-backed aviation hint for unknown image source domains."""

from unittest.mock import patch

import pytest

from services.tavily_aviation_domain_boost import tavily_aviation_domain_boost


@pytest.fixture(autouse=True)
def clear_tavily_domain_cache(monkeypatch):
    """Isolate in-process cache."""
    from services import tavily_aviation_domain_boost as m

    monkeypatch.setattr(m, "_CACHE", {})


def test_tavily_domain_boost_from_snippets(monkeypatch):
    monkeypatch.setenv("SEARCHAPI_TAVILY_DOMAIN_VERIFY", "1")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("TAVILY_DISABLED", raising=False)

    def fake_rest(*, api_key: str, host: str, timeout: float):
        return {
            "results": [
                {
                    "title": f"{host} — charter operator",
                    "content": "Business aviation private jet aircraft charter fleet turboprop.",
                    "url": f"https://{host}/fleet",
                }
            ]
        }

    with patch("services.tavily_aviation_domain_boost._tavily_search_domain_rest", fake_rest):
        sc = tavily_aviation_domain_boost("acme-regional-jets.example")
    assert sc >= 80


def test_tavily_domain_boost_zero_when_host_absent_from_results(monkeypatch):
    monkeypatch.setenv("SEARCHAPI_TAVILY_DOMAIN_VERIFY", "1")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    def fake_rest(*, api_key: str, host: str, timeout: float):
        return {
            "results": [
                {
                    "title": "Generic portal",
                    "content": "Recipes and lifestyle.",
                    "url": "https://other.example/",
                }
            ]
        }

    with patch("services.tavily_aviation_domain_boost._tavily_search_domain_rest", fake_rest):
        sc = tavily_aviation_domain_boost("missing-host.example")
    assert sc == 0
