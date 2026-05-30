import pytest

import app.chat as chat
from app import metrics
from app.models import Hit


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics.reset()
    yield


async def _collect(gen):
    events: list[tuple[str, object]] = []
    async for event_type, payload in gen:
        events.append((event_type, payload))
    return events


async def test_answer_stream_emits_citations_then_tokens(monkeypatch, fake_ollama):
    hits = [Hit(chunk_id=1, document_id=1, text="Refunds take 5 days.", score=0.9)]

    async def fake_search(session, question, tenant):
        return hits

    monkeypatch.setattr(chat, "hybrid_search", fake_search)

    events = await _collect(chat.answer_stream(session=None, question="refund?", tenant="default"))

    types = [e[0] for e in events]
    assert types[0] == "citations"
    assert "token" in types
    assert types[-1] == "done"

    # LLM + token metrics were recorded.
    snap = metrics.snapshot()
    assert snap["tokens"]["completion"] > 0


async def test_answer_stream_handles_no_hits(monkeypatch):
    async def empty_search(session, question, tenant):
        return []

    monkeypatch.setattr(chat, "hybrid_search", empty_search)

    events = await _collect(chat.answer_stream(session=None, question="unknown", tenant="default"))
    text = "".join(p for t, p in events if t == "token")

    assert "don't have information" in text.lower()
    assert events[-1][0] == "done"
