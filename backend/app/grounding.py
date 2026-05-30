"""Grounding guardrail: a cheap, dependency-free faithfulness proxy.

Kept separate from the (heavy) retrieval module so the eval harness and unit
tests can import it without pulling in sentence-transformers / torch.
"""
from __future__ import annotations

import re

from .config import settings
from .models import Hit

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are",
    "you", "your", "it", "this", "that", "with", "as", "be", "can", "i", "we",
    "do", "does", "how", "what", "if", "not", "no", "yes", "please", "from",
}


def content_terms(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def grounding_overlap(answer: str, hits: list[Hit]) -> float:
    """Fraction of the answer's content terms that appear in retrieved context.

    Low overlap suggests the model may be hallucinating beyond its sources.
    """
    answer_terms = content_terms(answer)
    if not answer_terms:
        return 1.0
    context_terms = content_terms(" ".join(h.text for h in hits))
    return len(answer_terms & context_terms) / len(answer_terms)


def is_grounded(answer: str, hits: list[Hit]) -> bool:
    return bool(hits) and grounding_overlap(answer, hits) >= settings.grounding_min_overlap
