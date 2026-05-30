"""RAG chat orchestration with conversation memory, caching, and a grounding guardrail."""
import time
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from . import cache, memory, metrics
from .grounding import grounding_overlap, is_grounded
from .models import Hit
from .ollama_client import stream_chat
from .retrieval import hybrid_search

__all__ = ["answer_stream", "build_prompt", "grounding_overlap", "is_grounded"]

PROMPT_TEMPLATE = """You are a helpful customer support assistant. \
Use the provided sources to answer the user's question.
- If the sources don't contain the answer, say so honestly.
- Cite the source numbers like [1], [2] inline.
- Be concise.

Sources:
{sources}

Question: {question}

Answer:"""


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources_text = "\n\n".join(f"[{i+1}] {h.text}" for i, h in enumerate(hits))
    return PROMPT_TEMPLATE.format(sources=sources_text, question=question)


def _citation_items(hits: list[Hit]) -> list[dict]:
    return [
        {"index": i + 1, "document_id": h.document_id, "chunk_id": h.chunk_id, "text": h.text, "score": h.score}
        for i, h in enumerate(hits)
    ]


async def answer_stream(
    session: AsyncSession,
    question: str,
    tenant: str,
    session_id: str | None = None,
) -> AsyncIterator[tuple[str, dict | str]]:
    """Yields ("citations", payload), many ("token", str), ("grounding", payload), ("done", "").

    When ``session_id`` is provided, follow-up questions are condensed against
    conversation history and the exchange is stored back to memory.
    """
    search_query = question
    if session_id:
        search_query = await memory.condense_question(session_id, question)
        if search_query != question:
            yield "rewrite", {"original": question, "rewritten": search_query}

    # Answer cache (keyed on the resolved standalone question).
    cached = await cache.get_answer(tenant, search_query)
    if cached is not None:
        yield "citations", {"items": cached.get("citations", [])}
        yield "token", cached["answer"]
        yield "grounding", {"grounded": cached.get("grounded", True), "cached": True}
        if session_id:
            await memory.append(session_id, "user", question)
            await memory.append(session_id, "assistant", cached["answer"])
        yield "done", ""
        return

    hits = await hybrid_search(session, search_query, tenant)
    yield "citations", {"items": _citation_items(hits)}

    if not hits:
        msg = "I don't have information about that in my knowledge base."
        yield "token", msg
        yield "grounding", {"grounded": False, "cached": False}
        if session_id:
            await memory.append(session_id, "user", question)
            await memory.append(session_id, "assistant", msg)
        yield "done", ""
        return

    prompt = build_prompt(search_query, hits)
    metrics.record_tokens("prompt", _approx_tokens(prompt))

    llm_start = time.perf_counter()
    completion_tokens = 0
    parts: list[str] = []
    async for token in stream_chat(prompt):
        completion_tokens += 1
        parts.append(token)
        yield "token", token

    metrics.record_latency("llm", (time.perf_counter() - llm_start) * 1000.0)
    metrics.record_tokens("completion", completion_tokens)

    answer = "".join(parts)
    grounded = is_grounded(answer, hits)
    yield "grounding", {"grounded": grounded, "overlap": round(grounding_overlap(answer, hits), 3), "cached": False}

    # Only cache well-grounded answers so we don't memoize hallucinations.
    if grounded:
        await cache.set_answer(tenant, search_query, {
            "answer": answer,
            "citations": _citation_items(hits),
            "grounded": True,
        })

    if session_id:
        await memory.append(session_id, "user", question)
        await memory.append(session_id, "assistant", answer)

    yield "done", ""


def _approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) without a tokenizer dependency."""
    return max(1, len(text) // 4)
