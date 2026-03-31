import json

import pytest

from rag.rag_answer_cache import (
    CACHE_KEY_PREFIX,
    apply_cache_hit_metadata,
    apply_cache_miss_metadata,
    cache_get,
    cache_set,
    normalize_answer_payload_for_cache,
    normalize_query_for_rag_cache,
    rag_answer_cache_key,
    rag_answer_cache_ttl_sec,
    rag_cache_enabled,
    set_redis_factory_for_tests,
)


class _FakeRedis:
    """Minimal Redis stand-in (decode_responses=True semantics)."""

    def __init__(self) -> None:
        self._kv: dict = {}
        self.setex_calls: list = []

    def get(self, key: str):
        return self._kv.get(key)

    def setex(self, key: str, ttl: int, value: str):
        self.setex_calls.append((key, ttl, value))
        self._kv[key] = value


@pytest.fixture
def fake_redis(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    monkeypatch.delenv("RAG_ANSWER_CACHE_DISABLED", raising=False)
    set_redis_factory_for_tests(lambda _url: r)
    yield r
    set_redis_factory_for_tests(None)


def test_normalize_query_and_key_stable():
    assert normalize_query_for_rag_cache("  hello   world  ") == "hello world"
    k1 = rag_answer_cache_key("hello world")
    k2 = rag_answer_cache_key("  hello   world  ")
    assert k1 == k2
    assert k1.startswith(CACHE_KEY_PREFIX)


def test_rag_cache_disabled_without_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert rag_cache_enabled() is False


def test_cache_miss_get_returns_none(fake_redis):
    assert cache_get("any question") is None


def test_cache_write_then_hit(fake_redis):
    q = "What is a Gulfstream G650?"
    payload = normalize_answer_payload_for_cache(
        {
            "answer": "A large-cabin business jet.",
            "sources": [{"entity_type": "other", "entity_id": "1", "score": 0.9}],
            "data_used": {"pinecone": 2},
            "aircraft_images": [],
            "error": None,
        }
    )
    assert cache_set(q, payload) is True
    assert len(fake_redis.setex_calls) == 1
    key, ttl, raw = fake_redis.setex_calls[0]
    assert key == rag_answer_cache_key(q)
    assert ttl == rag_answer_cache_ttl_sec()
    row = json.loads(raw)
    assert row["version"] == 1
    assert row["payload"]["answer"] == payload["answer"]

    got = cache_get(q)
    assert got is not None
    assert got["answer"] == payload["answer"]
    assert got["sources"] == payload["sources"]

    hit_meta = apply_cache_hit_metadata(got.get("data_used"))
    assert hit_meta.get("rag_cache") == "hit"

    miss_meta = apply_cache_miss_metadata({"foo": 1})
    assert miss_meta.get("rag_cache") == "miss"


def test_cache_disabled_flag(monkeypatch, fake_redis):
    monkeypatch.setenv("RAG_ANSWER_CACHE_DISABLED", "1")
    from rag import rag_answer_cache as mod

    assert mod.rag_cache_enabled() is False
    assert (
        mod.cache_set("q", mod.normalize_answer_payload_for_cache({"answer": "x"}))
        is False
    )
