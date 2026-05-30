# RAG Voice + Text Support Bot (Flagship)

[![CI](https://github.com/your-username/ragvoice-support-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/ragvoice-support-bot/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Lint: ruff](https://img.shields.io/badge/lint-ruff-D7FF64)
![Types: mypy](https://img.shields.io/badge/types-mypy-2A6DB2)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> Replace `your-username` in the badge URLs after you push the repo to GitHub.

A production-grade Retrieval-Augmented-Generation customer support bot. **Voice-in, voice-out, text-in, text-out, with citations.** Runs **100% locally — no API keys needed** to demo.

[Live demo (TODO)](https://example.com) · [Video walkthrough (TODO)](https://example.com)

Most RAG demos on GitHub are notebooks. This one is a deployable product with hybrid retrieval, reranking, streaming, evaluation, and a real chat UI.

## What this demonstrates

- **Hybrid retrieval** — dense (pgvector cosine) + sparse (BM25 via Postgres `tsvector`) fused with **Reciprocal Rank Fusion**
- **HyDE query expansion** — a hypothetical answer is generated and embedded to lift dense recall
- **Cross-encoder reranking** (`BAAI/bge-reranker-base`) on top-20 → top-5
- **Conversation memory + follow-up rewriting** — a "condense question" LLM step resolves pronoun-laden follow-ups against session history (`app/memory.py`)
- **Grounding guardrail** — answers are scored against retrieved context; ungrounded answers are flagged and never cached (`app/chat.py`)
- **Answer + embedding caching** in Redis, with hit/miss counters surfaced at `/metrics`
- **Streaming responses** via WebSocket (token-by-token via Ollama streaming API)
- **Citations** in every answer — clickable source doc + chunk
- **Voice round-trip** — Whisper STT (`faster-whisper`) + Piper TTS (open-source neural TTS)
- **Multi-tenant** — each "company" has an isolated knowledge base
- **Doc ingestion + admin** — PDF/HTML/Markdown via LangChain splitters; list/delete docs and trigger reindex
- **Real eval harness** — context precision/recall, faithfulness, and answer-contains on a seeded gold set; runs **offline in CI** (`python -m app.eval --ci`) and gates on thresholds
- **Observability** — per-stage latency (retrieval/rerank/LLM) + token counters at `/metrics`
- **Quality gates** — ruff + mypy + pytest (fake Ollama client) in CI
- **100% local** — Docker Compose pulls Ollama + Llama 3.1 8B on first run
- **Modern React UI** — chat with mic button, waveform, source cards

## Architecture

```mermaid
flowchart LR
    user[Browser: chat + mic]
    api[FastAPI WebSocket]
    stt[Whisper STT]
    rag[RAG Engine]
    vec[(Postgres + pgvector)]
    bm25[(Postgres tsvector)]
    rerank[bge-reranker]
    llm[Ollama Llama 3.1]
    tts[Piper TTS]
    ingest[Doc Ingest Worker]

    user -- audio/text --> api
    api --> stt --> rag
    api -- text --> rag
    rag --> vec
    rag --> bm25
    rag --> rerank
    rag --> llm
    llm --> tts --> api --> user
    ingest --> vec
    ingest --> bm25
```

## Tech stack

- **Python 3.11** + FastAPI + uvicorn
- **PostgreSQL 16 + pgvector** (vector + full-text in one DB)
- **Ollama** running **Llama 3.1 8B** (chat) + **nomic-embed-text** (embeddings) — open-source, no API key
- **faster-whisper** for STT (CPU-friendly Whisper port)
- **Piper TTS** for natural voice output
- **LangChain** for document loaders + splitters
- **sentence-transformers** for the reranker
- **React 18 + Vite + TS + Tailwind** frontend

## Quick start

```bash
git clone <repo>
cd ragvoice-support-bot
docker compose up -d
# First run: Ollama auto-pulls llama3.1:8b (~5GB) and nomic-embed-text (~250MB)
# Wait ~5min, then:
docker compose exec api python -m app.ingest /data/sample_docs
# Frontend: http://localhost:5173
# Backend:  http://localhost:8000
# Open in browser, type or click mic
```

Sample docs (Stripe API docs excerpts) are loaded by default so you can ask:
- "How do I create a payment intent?"
- "What's the difference between a charge and a payment intent?"

## API

| Method | Path                                | Description                          |
|--------|-------------------------------------|--------------------------------------|
| WS     | `/ws/chat?tenant=acme&session_id=…` | Streaming chat (text + audio) with memory |
| POST   | `/api/ingest`                       | Upload a doc to a tenant's KB        |
| GET    | `/api/sources?tenant=acme`          | List ingested docs                   |
| DELETE | `/api/documents/{id}?tenant=acme`   | Delete a doc and its chunks          |
| POST   | `/api/admin/reindex?tenant=acme`    | Re-embed a tenant's chunks           |
| GET    | `/metrics`                          | JSON latency/token/cache metrics     |
| GET    | `/metrics/prometheus`               | Prometheus exposition                |
| GET    | `/docs`                             | OpenAPI / Swagger UI                 |

## WebSocket protocol

Client -> server:
```json
{ "type": "query", "text": "How do I refund a charge?" }
{ "type": "audio_chunk", "data": "<base64 PCM 16k>" }
{ "type": "audio_end" }                                 // triggers STT then RAG
```

Server -> client:
```json
{ "type": "rewrite", "original": "what about refunds?", "rewritten": "How do I issue a refund?" }
{ "type": "citations", "items": [ {"index": 1, "document_id": 3, "chunk_id": 12, "text": "...", "score": 0.91} ] }
{ "type": "token", "text": "To" }                       // streamed
{ "type": "grounding", "grounded": true, "overlap": 0.42 }
{ "type": "tts_audio", "data": "<base64 wav>" }
{ "type": "done" }
```

## Key implementation highlights

- **Hybrid retrieval + RRF + HyDE**: [backend/app/retrieval.py](backend/app/retrieval.py)
- **WebSocket streaming + grounding guardrail**: [backend/app/chat.py](backend/app/chat.py)
- **Conversation memory + question condensing**: [backend/app/memory.py](backend/app/memory.py)
- **Redis answer/embedding cache**: [backend/app/cache.py](backend/app/cache.py)
- **Ingestion + admin (delete/reindex)**: [backend/app/ingest.py](backend/app/ingest.py)
- **Evaluation harness**: [backend/app/eval.py](backend/app/eval.py)
- **Metrics**: [backend/app/metrics.py](backend/app/metrics.py)

## Evaluation

Two modes:

```bash
make eval            # live: scores the gold set against the running bot
make eval-ci         # offline: deterministic stub corpus, gates on thresholds, writes eval_report.json
```

The offline mode runs in CI on every push so the metric math stays honest
without needing Postgres/Ollama. Metrics computed:

| Metric              | Meaning                                                    | CI gate |
|---------------------|------------------------------------------------------------|--------:|
| Context Recall      | gold-relevant chunk present in retrieved set               | ≥ 0.99 |
| Context Precision   | fraction of retrieved chunks that are relevant             | ≥ 0.30 |
| Faithfulness        | answer terms supported by retrieved context (grounding)    | ≥ 0.40 |
| Answer Contains     | gold keyword present in the answer                         | ≥ 0.99 |

## Roadmap

- Multi-modal: image inputs (LLaVA)
- Conversation memory across sessions
- Auto-rewriting follow-up queries with conversation context
- Cost guardrails (token quota per tenant)
- GPU compose override for 10x faster inference

<!-- ownership:author -->
---

## Author

**Umair** &mdash; [@umairjutt](https://github.com/umairjutt)

Designed, built and maintained by me. Licensed under the [MIT License](LICENSE).