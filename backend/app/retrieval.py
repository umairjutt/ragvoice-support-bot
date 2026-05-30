"""Hybrid retrieval with Reciprocal Rank Fusion + cross-encoder reranking.

The combination of dense (semantic) + sparse (lexical) retrieval consistently
outperforms either alone on RAG benchmarks. RRF is a parameter-free fusion
that just needs the rank position from each retriever, no score normalization.
"""
from sentence_transformers import CrossEncoder
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from . import cache, metrics
from .config import settings
from .models import Hit
from .ollama_client import complete, embed

__all__ = ["Hit", "hybrid_search", "dense_search", "sparse_search", "reciprocal_rank_fusion", "rerank"]

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


async def embed_cached(text_value: str) -> list[float]:
    """Embed with a Redis read-through cache."""
    cached = await cache.get_embedding(text_value)
    if cached is not None:
        return cached
    vec = await embed(text_value)
    await cache.set_embedding(text_value, vec)
    return vec


HYDE_PROMPT = """Write a short, factual paragraph that would directly answer the \
following customer support question. Do not say you are unsure; write the ideal \
answer as if from documentation.

Question: {question}

Hypothetical answer:"""


async def hyde_expand(query: str) -> str:
    """HyDE: generate a hypothetical answer and embed *that* to lift recall.

    Returns the original query on any failure so retrieval never breaks.
    """
    if not settings.enable_hyde:
        return query
    try:
        hypothetical = (await complete(HYDE_PROMPT.format(question=query), max_tokens=160)).strip()
    except Exception:
        return query
    return f"{query}\n\n{hypothetical}" if hypothetical else query


async def dense_search(session: AsyncSession, query: str, tenant: str, k: int) -> list[Hit]:
    vec = await embed_cached(query)
    sql = text(
        """
        SELECT id, document_id, text, 1 - (embedding <=> :vec) AS score
        FROM chunks
        WHERE tenant = :tenant
        ORDER BY embedding <=> :vec
        LIMIT :k
        """
    ).bindparams(bindparam("vec", value=vec))
    rows = (await session.execute(sql, {"tenant": tenant, "k": k})).all()
    return [Hit(r.id, r.document_id, r.text, float(r.score)) for r in rows]


async def sparse_search(session: AsyncSession, query: str, tenant: str, k: int) -> list[Hit]:
    sql = text(
        """
        SELECT id, document_id, text,
               ts_rank(to_tsvector('english', text), websearch_to_tsquery('english', :q)) AS score
        FROM chunks
        WHERE tenant = :tenant
          AND to_tsvector('english', text) @@ websearch_to_tsquery('english', :q)
        ORDER BY score DESC
        LIMIT :k
        """
    )
    rows = (await session.execute(sql, {"q": query, "tenant": tenant, "k": k})).all()
    return [Hit(r.id, r.document_id, r.text, float(r.score)) for r in rows]


def reciprocal_rank_fusion(*ranked_lists: list[Hit], k: int = 60) -> list[Hit]:
    scores: dict[int, float] = {}
    by_id: dict[int, Hit] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            by_id[hit.chunk_id] = hit
    fused = sorted(by_id.values(), key=lambda h: scores[h.chunk_id], reverse=True)
    for h in fused:
        h.score = scores[h.chunk_id]
    return fused


def rerank(query: str, hits: list[Hit], top_n: int) -> list[Hit]:
    if not hits:
        return []
    pairs = [(query, h.text) for h in hits]
    scores = get_reranker().predict(pairs).tolist()
    scored = sorted(zip(hits, scores, strict=False), key=lambda x: x[1], reverse=True)
    out = []
    for hit, score in scored[:top_n]:
        hit.score = float(score)
        out.append(hit)
    return out


async def hybrid_search(session: AsyncSession, query: str, tenant: str) -> list[Hit]:
    # HyDE expands the dense side only; BM25 + reranking use the literal query.
    dense_query = await hyde_expand(query)

    with metrics.timed("retrieval"):
        dense = await dense_search(session, dense_query, tenant, settings.top_k_dense)
        sparse = await sparse_search(session, query, tenant, settings.top_k_sparse)
        fused = reciprocal_rank_fusion(dense, sparse)[: settings.top_k_dense]

    with metrics.timed("rerank"):
        return rerank(query, fused, settings.top_k_rerank)
