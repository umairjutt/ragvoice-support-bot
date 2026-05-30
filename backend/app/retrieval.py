"""Hybrid retrieval with Reciprocal Rank Fusion + cross-encoder reranking.

The combination of dense (semantic) + sparse (lexical) retrieval consistently
outperforms either alone on RAG benchmarks. RRF is a parameter-free fusion
that just needs the rank position from each retriever, no score normalization.
"""
from dataclasses import dataclass

from sentence_transformers import CrossEncoder
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .ollama_client import embed


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    text: str
    score: float


_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


async def dense_search(session: AsyncSession, query: str, tenant: str, k: int) -> list[Hit]:
    vec = await embed(query)
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
    scored = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    out = []
    for hit, score in scored[:top_n]:
        hit.score = float(score)
        out.append(hit)
    return out


async def hybrid_search(session: AsyncSession, query: str, tenant: str) -> list[Hit]:
    dense = await dense_search(session, query, tenant, settings.top_k_dense)
    sparse = await sparse_search(session, query, tenant, settings.top_k_sparse)
    fused = reciprocal_rank_fusion(dense, sparse)[: settings.top_k_dense]
    return rerank(query, fused, settings.top_k_rerank)
