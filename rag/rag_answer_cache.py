"""
Redis cache for completed Ask Consultant / RAG answers (24h TTL by default).

Key: SHA-256 of normalized **user query only** (``hye:rag:answer:v1:{hex}``).
Caching applies only when **conversation history is empty** — otherwise the same text
could mean different things in context.

Env:
  ``REDIS_URL`` — if unset, caching is disabled (no-op).
  ``RAG_ANSWER_CACHE_DISABLED=1`` — force disable even when Redis is configured.
  ``RAG_ANSWER_CACHE_TTL_SEC`` — override TTL (default 86400 = 24h).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import copy
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

CACHE_KEY_PREFIX = "hye:rag:answer:v1:"
CACHE_SCHEMA_VERSION = 1
_DEFAULT_TTL_SEC = 86400  # 24 hours

_redis_factory: Optional[Callable[[str], Any]] = None


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


def rag_answer_cache_ttl_sec() -> int:
    try:
        v = int((os.getenv("RAG_ANSWER_CACHE_TTL_SEC") or str(_DEFAULT_TTL_SEC)).strip())
    except ValueError:
        return _DEFAULT_TTL_SEC
    return max(60, min(86400 * 7, v))


def rag_cache_enabled() -> bool:
    if _env_truthy("RAG_ANSWER_CACHE_DISABLED"):
        return False
    url = (os.getenv("REDIS_URL") or "").strip()
    return bool(url)


def normalize_query_for_rag_cache(query: str) -> str:
    q = (query or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q


def rag_answer_cache_key(query: str) -> str:
    nq = normalize_query_for_rag_cache(query)
    digest = hashlib.sha256(nq.encode("utf-8")).hexdigest()
    return f"{CACHE_KEY_PREFIX}{digest}"


def _get_redis():
    global _redis_factory
    url = (os.getenv("REDIS_URL") or "").strip()
    if not url:
        return None
    try:
        import redis
    except ImportError:
        logger.warning("REDIS_URL is set but redis package is not installed; RAG cache disabled")
        return None
    if _redis_factory is not None:
        return _redis_factory(url)
    return redis.Redis.from_url(url, decode_responses=True)


def set_redis_factory_for_tests(factory: Optional[Callable[[str], Any]]) -> None:
    """Override Redis construction (e.g. fakeredis) in unit tests."""
    global _redis_factory
    _redis_factory = factory


def _decode_entry(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("RAG cache: invalid JSON, ignoring")
        return None
    if not isinstance(obj, dict):
        return None
    if int(obj.get("version") or 0) != CACHE_SCHEMA_VERSION:
        return None
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return None
    return obj


def cache_get(query: str) -> Optional[Dict[str, Any]]:
    """
    Return cached ``payload`` dict
    ``{answer, sources, data_used, aircraft_images, error}``, or None.
    """
    if not rag_cache_enabled():
        return None
    r = _get_redis()
    if r is None:
        return None
    key = rag_answer_cache_key(query)
    try:
        raw = r.get(key)
    except Exception as e:
        logger.warning("RAG cache GET failed: %s", e)
        return None
    entry = _decode_entry(raw)
    if not entry:
        return None
    payload = entry.get("payload")
    assert isinstance(payload, dict)
    return dict(payload)


def cache_set(query: str, payload: Dict[str, Any]) -> bool:
    """Store a completed response (without rag_cache hit/miss flags). Returns True if stored."""
    if not rag_cache_enabled():
        return False
    r = _get_redis()
    if r is None:
        return False
    key = rag_answer_cache_key(query)
    to_store = copy.deepcopy(dict(payload))
    if isinstance(to_store.get("data_used"), dict):
        to_store["data_used"] = strip_rag_cache_volatile_fields(to_store.get("data_used"))
    entry = {"version": CACHE_SCHEMA_VERSION, "payload": to_store}
    try:
        raw = json.dumps(entry, ensure_ascii=False, default=str)
        r.setex(key, rag_answer_cache_ttl_sec(), raw)
        return True
    except Exception as e:
        logger.warning("RAG cache SET failed: %s", e)
        return False


def apply_cache_hit_metadata(data_used: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    du = dict(data_used or {})
    du["rag_cache"] = "hit"
    return du


def apply_cache_miss_metadata(data_used: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    du = dict(data_used or {})
    du["rag_cache"] = "miss"
    return du


def normalize_answer_payload_for_cache(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure standard keys for cached consultant API responses."""
    out = dict(payload)
    out.setdefault("answer", "")
    out.setdefault("sources", [])
    out.setdefault("data_used", {})
    out.setdefault("aircraft_images", [])
    out.setdefault("error", None)
    return out


def strip_rag_cache_volatile_fields(data_used: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    du = dict(data_used or {})
    du.pop("rag_cache", None)
    du.pop("rag_cache_write", None)
    return du
