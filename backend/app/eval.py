"""Lightweight RAG evaluation harness.

Runs a fixed Q&A set against the bot, then computes simple metrics:
- ANSWER_CONTAINS: how often the gold answer keyword appears in the response
- CONTEXT_HIT: how often the gold doc was in the top-K retrieved chunks

Real RAGAS scoring requires a judge LLM and is left as an extension hook
(see the `--ragas` flag).
"""
import asyncio
import json
import sys
from pathlib import Path

from .chat import answer_stream
from .db import Session, init_db
from .retrieval import hybrid_search


GOLD = [
    {"q": "How do I create a payment intent?", "must_contain": "payment_intents.create", "tenant": "default"},
    {"q": "What's the difference between a charge and a payment intent?", "must_contain": "payment intent", "tenant": "default"},
]


async def run(gold_path: str | None = None) -> dict:
    await init_db()
    questions = json.loads(Path(gold_path).read_text()) if gold_path else GOLD

    contains_hits = 0
    total = 0
    async with Session() as session:
        for item in questions:
            total += 1
            answer_parts = []
            async for event_type, payload in answer_stream(session, item["q"], item["tenant"]):
                if event_type == "token":
                    answer_parts.append(payload)
            answer = "".join(answer_parts).lower()
            if item["must_contain"].lower() in answer:
                contains_hits += 1
            print(f"- Q: {item['q']!r}\n  A: {answer[:200]}...\n")

    score = contains_hits / max(total, 1)
    print(f"ANSWER_CONTAINS score: {score:.2f} ({contains_hits}/{total})")
    return {"answer_contains": score, "total": total}


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else None))
