"""Shared fixtures, including a fake Ollama client so tests never hit a model."""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest


class FakeOllama:
    """Deterministic stand-in for app.ollama_client used in tests."""

    def __init__(self, answer: str = "This is grounded in [1].", dim: int = 8):
        self.answer = answer
        self.dim = dim
        self.embed_calls: list[str] = []
        self.chat_calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        # Stable pseudo-embedding derived from the text length.
        seed = (len(text) % 7) + 1
        return [round((i + seed) / 10.0, 3) for i in range(self.dim)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    async def stream_chat(self, prompt: str) -> AsyncIterator[str]:
        self.chat_calls.append(prompt)
        for token in self.answer.split(" "):
            yield token + " "


@pytest.fixture
def fake_ollama(monkeypatch) -> FakeOllama:
    fake = FakeOllama()
    import app.chat as chat
    import app.ollama_client as oc

    monkeypatch.setattr(oc, "embed", fake.embed)
    monkeypatch.setattr(oc, "stream_chat", fake.stream_chat)
    monkeypatch.setattr(chat, "stream_chat", fake.stream_chat)
    return fake
