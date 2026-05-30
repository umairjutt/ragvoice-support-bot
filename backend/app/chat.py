"""RAG chat orchestration."""
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from .ollama_client import stream_chat
from .retrieval import Hit, hybrid_search

PROMPT_TEMPLATE = """You are a helpful customer support assistant. Use the provided sources to answer the user's question.
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


async def answer_stream(
    session: AsyncSession, question: str, tenant: str
) -> AsyncIterator[tuple[str, dict | str]]:
    """Yields ("citations", payload) once, then many ("token", str), then ("done", "")."""
    hits = await hybrid_search(session, question, tenant)

    yield "citations", {
        "items": [
            {"index": i + 1, "document_id": h.document_id, "chunk_id": h.chunk_id, "text": h.text, "score": h.score}
            for i, h in enumerate(hits)
        ]
    }

    if not hits:
        yield "token", "I don't have information about that in my knowledge base."
        yield "done", ""
        return

    prompt = build_prompt(question, hits)
    async for token in stream_chat(prompt):
        yield "token", token

    yield "done", ""
