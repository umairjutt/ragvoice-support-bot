"""Pydantic response/request models so the OpenAPI schema at /docs is precise."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    ok: bool = Field(True, description="Liveness flag.")


class SourceOut(BaseModel):
    id: int = Field(..., description="Document id.")
    title: str = Field(..., description="Human-readable document title.")
    source: str = Field(..., description="Original file name or URI.")


class IngestResponse(BaseModel):
    chunks: int = Field(..., description="Number of chunks embedded and stored.")
    filename: str = Field(..., description="Name of the ingested file.")


class DeleteResponse(BaseModel):
    deleted: int = Field(..., description="Number of documents removed.")


class ReindexResponse(BaseModel):
    documents: int = Field(..., description="Documents queued for re-embedding.")
    chunks: int = Field(..., description="Chunks re-embedded.")


class StageMetric(BaseModel):
    count: int = Field(..., description="Number of observations.")
    p50_ms: float = Field(..., description="Median latency in milliseconds.")
    p95_ms: float = Field(..., description="95th percentile latency in milliseconds.")


class MetricsResponse(BaseModel):
    stages: dict[str, StageMetric] = Field(
        default_factory=dict,
        description="Per-stage latency (retrieval, rerank, llm, total).",
    )
    tokens: dict[str, float] = Field(
        default_factory=dict,
        description="Prompt / completion token counters.",
    )
    cache: dict[str, float] = Field(
        default_factory=dict,
        description="Answer + embedding cache hit/miss counters.",
    )
