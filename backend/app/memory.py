"""Session-scoped conversation memory + follow-up question condensing.

Multi-turn RAG breaks when a user asks "what about refunds?" because the
retriever sees only the pronoun-laden follow-up. We keep a short rolling
history per session (in Redis, with an in-process fallback) and use the LLM to
rewrite the latest user turn into a standalone question before retrieval.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque

import redis.asyncio as redis

from .config import settings
from .ollama_client import complete

# In-process fallback when Redis is unavailable (e.g. tests).
_local: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=settings.memory_max_turns * 2))
_client: redis.Redis | None = None


def _redis() -> redis.Redis | None:
    global _client
    if _client is None:
        try:
            _client = redis.from_url(settings.redis_url, decode_responses=True)
        except Exception:
            return None
    return _client


def _key(session_id: str) -> str:
    return f"mem:{session_id}"


async def history(session_id: str) -> list[dict]:
    client = _redis()
    if client is not None:
        try:
            # redis-py types these as sync|async unions; we use the async client.
            raw = await client.lrange(_key(session_id), 0, -1)  # type: ignore[misc]
            return [json.loads(x) for x in raw]
        except Exception:
            pass
    return list(_local[session_id])


async def append(session_id: str, role: str, content: str) -> None:
    entry = {"role": role, "content": content}
    client = _redis()
    if client is not None:
        try:
            key = _key(session_id)
            await client.rpush(key, json.dumps(entry))  # type: ignore[misc]
            await client.ltrim(key, -settings.memory_max_turns * 2, -1)  # type: ignore[misc]
            await client.expire(key, 86400)  # type: ignore[misc]
            return
        except Exception:
            pass
    _local[session_id].append(entry)


def _format_history(turns: list[dict]) -> str:
    return "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns)


CONDENSE_PROMPT = """Given the conversation history and a follow-up question, rewrite the \
follow-up as a standalone question that can be understood without the history. \
Keep it concise. If it is already standalone, return it unchanged. Respond with \
ONLY the rewritten question.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


async def condense_question(session_id: str, question: str) -> str:
    """Rewrite a follow-up into a standalone question using prior turns."""
    if not settings.enable_query_rewrite:
        return question

    turns = await history(session_id)
    if not turns:
        return question

    prompt = CONDENSE_PROMPT.format(history=_format_history(turns), question=question)
    try:
        rewritten = (await complete(prompt, max_tokens=128)).strip()
    except Exception:
        return question

    # Guard against the model returning junk or an empty string.
    return rewritten or question
