"""Thin async wrapper for the Ollama REST API."""
import json
from collections.abc import AsyncIterator

import httpx

from .config import settings


async def embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{settings.ollama_url}/api/embeddings",
            json={"model": settings.embedding_model, "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    return [await embed(t) for t in texts]


async def complete(prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    """One-shot, non-streaming generation. Used for query rewriting + HyDE."""
    payload = {
        "model": settings.llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{settings.ollama_url}/api/generate", json=payload)
        r.raise_for_status()
        return r.json().get("response", "")


async def stream_chat(prompt: str) -> AsyncIterator[str]:
    """Yield tokens from Ollama as they are generated."""
    payload = {
        "model": settings.llm_model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{settings.ollama_url}/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "response" in obj:
                    yield obj["response"]
                if obj.get("done"):
                    return
