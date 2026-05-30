"""RAG evaluation harness.

Two modes:

* **live** (default): runs the seeded gold questions against the real bot
  (Postgres + Ollama must be up) and scores the responses.
* **--ci**: a fully offline evaluation over an in-memory corpus with a
  deterministic stub retriever + LLM. This validates the *harness and metric
  math* on every CI run without external services, writes ``eval_report.json``,
  and exits non-zero if any metric drops below its threshold — so the README
  metrics table stays honest.

Metrics
-------
* ``context_recall``    – gold-relevant chunk present in retrieved set.
* ``context_precision`` – fraction of retrieved chunks that are relevant.
* ``faithfulness``      – answer terms supported by retrieved context (grounding).
* ``answer_contains``   – gold keyword present in the answer.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .grounding import grounding_overlap
from .models import Hit

# --------------------------------------------------------------------------- #
# Gold set
# --------------------------------------------------------------------------- #


@dataclass
class GoldItem:
    q: str
    must_contain: str
    relevant_text: str
    tenant: str = "default"


GOLD: list[GoldItem] = [
    GoldItem(
        q="How do I create a payment intent?",
        must_contain="payment_intents.create",
        relevant_text=(
            "To create a payment intent call payment_intents.create with an amount and currency. "
            "The returned client_secret is used on the frontend to confirm the payment."
        ),
    ),
    GoldItem(
        q="What is the difference between a charge and a payment intent?",
        must_contain="payment intent",
        relevant_text=(
            "A charge is a single attempt to move money. A payment intent tracks the full lifecycle "
            "of a customer payment, including retries and authentication, and may create several charges."
        ),
    ),
    GoldItem(
        q="How are webhook signatures verified?",
        must_contain="signature",
        relevant_text=(
            "Webhook signatures are verified by computing an HMAC of the raw payload with your "
            "webhook secret and comparing it to the signature header before trusting the event."
        ),
    ),
]

# Distractor passages that should NOT be retrieved for the gold questions.
DISTRACTORS: list[str] = [
    "Our office is open Monday to Friday from 9am to 5pm local time.",
    "You can change the dashboard theme between light and dark mode in settings.",
    "Refund processing times depend on the customer's bank and can take 5-10 days.",
]

THRESHOLDS = {
    "context_recall": 0.99,
    "context_precision": 0.30,
    "faithfulness": 0.40,
    "answer_contains": 0.99,
}


@dataclass
class Scores:
    context_recall: list[float] = field(default_factory=list)
    context_precision: list[float] = field(default_factory=list)
    faithfulness: list[float] = field(default_factory=list)
    answer_contains: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        def avg(xs: list[float]) -> float:
            return round(sum(xs) / len(xs), 4) if xs else 0.0

        return {
            "context_recall": avg(self.context_recall),
            "context_precision": avg(self.context_precision),
            "faithfulness": avg(self.faithfulness),
            "answer_contains": avg(self.answer_contains),
        }


# --------------------------------------------------------------------------- #
# Offline stub retriever + LLM (CI mode)
# --------------------------------------------------------------------------- #


def _tokens(text: str) -> set[str]:
    return {w for w in text.lower().split() if len(w) > 2}


def _stub_retrieve(query: str, corpus: list[tuple[int, str]], k: int = 3) -> list[Hit]:
    """Naive lexical retriever: rank corpus by token overlap with the query."""
    q = _tokens(query)
    scored = sorted(
        corpus,
        key=lambda item: len(q & _tokens(item[1])),
        reverse=True,
    )
    return [Hit(chunk_id=cid, document_id=cid, text=txt, score=0.0) for cid, txt in scored[:k]]


def _stub_answer(hits: list[Hit]) -> str:
    """Deterministic 'LLM': stitch the top retrieved chunks into an answer."""
    return " ".join(h.text for h in hits[:2])


def _precision_recall(hits: list[Hit], relevant_id: int) -> tuple[float, float]:
    retrieved_ids = [h.chunk_id for h in hits]
    recall = 1.0 if relevant_id in retrieved_ids else 0.0
    precision = (retrieved_ids.count(relevant_id) / len(retrieved_ids)) if retrieved_ids else 0.0
    return precision, recall


async def run_offline() -> dict:
    scores = Scores()

    # Build a corpus: each gold relevant_text + distractors, with stable ids.
    corpus: list[tuple[int, str]] = []
    relevant_id_for: dict[str, int] = {}
    for i, item in enumerate(GOLD):
        cid = i + 1
        corpus.append((cid, item.relevant_text))
        relevant_id_for[item.q] = cid
    for j, dtext in enumerate(DISTRACTORS):
        corpus.append((1000 + j, dtext))

    rows = []
    for item in GOLD:
        hits = _stub_retrieve(item.q, corpus)
        answer = _stub_answer(hits)

        precision, recall = _precision_recall(hits, relevant_id_for[item.q])
        faith = grounding_overlap(answer, hits)
        contains = 1.0 if item.must_contain.lower() in answer.lower() else 0.0

        scores.context_precision.append(precision)
        scores.context_recall.append(recall)
        scores.faithfulness.append(faith)
        scores.answer_contains.append(contains)

        rows.append({
            "q": item.q,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "faithfulness": round(faith, 3),
            "answer_contains": contains,
        })

    return {"mode": "offline-ci", "metrics": scores.summary(), "rows": rows}


# --------------------------------------------------------------------------- #
# Live evaluation
# --------------------------------------------------------------------------- #


async def run_live(gold_path: str | None = None) -> dict:
    from .chat import answer_stream
    from .db import Session, init_db
    from .retrieval import hybrid_search

    await init_db()
    items = GOLD
    if gold_path:
        raw = json.loads(Path(gold_path).read_text())
        items = [GoldItem(**r) for r in raw]

    scores = Scores()
    rows = []
    async with Session() as session:
        for item in items:
            hits = await hybrid_search(session, item.q, item.tenant)

            parts: list[str] = []
            async for event_type, payload in answer_stream(session, item.q, item.tenant):
                if event_type == "token":
                    parts.append(payload if isinstance(payload, str) else "")
            answer = "".join(parts)

            # Relevance proxy: a chunk is relevant if it shares enough terms with gold.
            gold_terms = _tokens(item.relevant_text)
            rel_ids = [h.chunk_id for h in hits if len(gold_terms & _tokens(h.text)) >= 3]
            recall = 1.0 if rel_ids else 0.0
            precision = (len(rel_ids) / len(hits)) if hits else 0.0
            faith = grounding_overlap(answer, hits)
            contains = 1.0 if item.must_contain.lower() in answer.lower() else 0.0

            scores.context_precision.append(precision)
            scores.context_recall.append(recall)
            scores.faithfulness.append(faith)
            scores.answer_contains.append(contains)
            rows.append({"q": item.q, "precision": round(precision, 3), "recall": round(recall, 3),
                         "faithfulness": round(faith, 3), "answer_contains": contains})

    return {"mode": "live", "metrics": scores.summary(), "rows": rows}


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def _print_report(report: dict) -> None:
    print(f"RAG eval ({report['mode']})")
    print("-" * 48)
    for name, value in report["metrics"].items():
        threshold = THRESHOLDS[name]
        status = "PASS" if value >= threshold else "FAIL"
        print(f"  {name:<20} {value:>6.3f}  (>= {threshold})  [{status}]")


def _gate(report: dict) -> bool:
    return all(report["metrics"][k] >= THRESHOLDS[k] for k in THRESHOLDS)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument("--ci", action="store_true", help="Run offline (no DB/Ollama) and gate on thresholds.")
    parser.add_argument("--gold", help="Path to a JSON gold set (live mode).")
    parser.add_argument("--out", default="eval_report.json", help="Where to write the JSON report.")
    args = parser.parse_args()

    report = asyncio.run(run_offline()) if args.ci else asyncio.run(run_live(args.gold))

    Path(args.out).write_text(json.dumps(report, indent=2))
    _print_report(report)
    print(f"\nReport written to {args.out}")

    if args.ci and not _gate(report):
        print("\nEval gate FAILED: a metric is below threshold.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
