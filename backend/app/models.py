"""Lightweight shared data structures (no heavy ML imports)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Hit:
    """A single retrieved chunk with a fusion/rerank score."""

    chunk_id: int
    document_id: int
    text: str
    score: float
