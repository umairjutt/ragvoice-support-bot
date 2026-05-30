"""Redis-backed caches for answers and embeddings.

Both helpers degrade gracefully: if Redis is unreachable the calls become
no-ops so the bot keeps working (just without the cache speed-up). Cache
hits/misses are recorded in the metrics module.
"""
from __future__ import annotations

import hashlib
import json

import redis.asyncio as redis

from . import metrics
from .config import settings

_client: redis.Redis | None = None


def _redis() -> redis.Redis | None:
    global _client
    if _client is None:
        try:
            _client = redis.from_url(settings.redis_url, decode_responses=True)
        except Exception:
            return None
    return _client


def _key(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


async def get_embedding(text: str) -> list[float] | None:
    if settings.embedding_cache_ttl <= 0:
        return None
    client = _redis()
    if client is None:
        return None
    try:
        raw = await client.get(_key("emb", settings.embedding_model, text))
    except Exception:
        return None
    if raw is None:
        metrics.record_cache("embed_miss")
        return None
    metrics.record_cache("embed_hit")
    return json.loads(raw)


async def set_embedding(text: str, vector: list[float]) -> None:
    if settings.embedding_cache_ttl <= 0:
        return
    client = _redis()
    if client is None:
        return
    try:
        await client.set(
            _key("emb", settings.embedding_model, text),
            json.dumps(vector),
            ex=settings.embedding_cache_ttl,
        )
    except Exception:
        return


async def get_answer(tenant: str, question: str) -> dict | None:
    if settings.answer_cache_ttl <= 0:
        return None
    client = _redis()
    if client is None:
        return None
    try:
        raw = await client.get(_key("ans", tenant, question))
    except Exception:
        return None
    if raw is None:
        metrics.record_cache("answer_miss")
        return None
    metrics.record_cache("answer_hit")
    return json.loads(raw)


async def set_answer(tenant: str, question: str, payload: dict) -> None:
    if settings.answer_cache_ttl <= 0:
        return
    client = _redis()
    if client is None:
        return
    try:
        await client.set(
            _key("ans", tenant, question),
            json.dumps(payload),
            ex=settings.answer_cache_ttl,
        )
    except Exception:
        return


async def invalidate_answers() -> None:
    """Drop cached answers after the knowledge base changes.

    Answer keys are hashed over (tenant, question) so we can't target a single
    tenant cheaply; clearing the whole answer namespace is the safe choice on
    ingest/delete/reindex.
    """
    client = _redis()
    if client is None:
        return
    try:
        async for key in client.scan_iter(match="ans:*"):
            await client.delete(key)
    except Exception:
        return
