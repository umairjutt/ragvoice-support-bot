import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import cache, metrics
from .audio import synthesize, transcribe_pcm16k
from .chat import answer_stream
from .db import Document, Session, get_db, init_db
from .ingest import delete_document, ingest_file, reindex_tenant
from .schemas import (
    DeleteResponse,
    HealthResponse,
    IngestResponse,
    MetricsResponse,
    ReindexResponse,
    SourceOut,
)

TAGS_METADATA = [
    {"name": "Health", "description": "Liveness and readiness probes."},
    {"name": "Documents", "description": "Knowledge-base ingestion and administration."},
    {"name": "Chat", "description": "Realtime voice/text RAG conversation over WebSocket."},
    {"name": "Observability", "description": "Per-stage latency, token, and cache metrics."},
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="RAG Voice Support Bot",
    version="1.0.0",
    summary="Hybrid-retrieval, voice-enabled support assistant with conversation memory and grounded answers.",
    description=(
        "Retrieval-augmented generation over a tenant-scoped knowledge base.\n\n"
        "- **Hybrid retrieval**: pgvector dense search fused with Postgres BM25 via Reciprocal Rank Fusion, "
        "then cross-encoder reranking.\n"
        "- **Voice**: Whisper STT + Piper TTS streamed over WebSocket.\n"
        "- **Grounding**: answers are checked against retrieved chunks and cited.\n"
    ),
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False,
)


@app.get("/health", tags=["Health"], response_model=HealthResponse, summary="Liveness probe")
async def health():
    return {"ok": True}


@app.get(
    "/metrics",
    tags=["Observability"],
    response_model=MetricsResponse,
    summary="JSON metrics snapshot",
    description="Per-stage latency percentiles, token counters, and cache stats.",
)
async def metrics_json():
    return metrics.snapshot()


@app.get(
    "/metrics/prometheus",
    tags=["Observability"],
    summary="Prometheus exposition",
    response_class=Response,
)
async def metrics_prometheus():
    return Response(content=metrics.render_prometheus(), media_type="text/plain; version=0.0.4")


@app.get(
    "/api/sources",
    tags=["Documents"],
    response_model=list[SourceOut],
    summary="List ingested documents",
)
async def list_sources(tenant: str = "default", session: AsyncSession = Depends(get_db)):
    r = await session.execute(select(Document).where(Document.tenant == tenant).order_by(Document.id.desc()))
    return [{"id": d.id, "title": d.title, "source": d.source} for d in r.scalars().all()]


@app.post(
    "/api/ingest",
    tags=["Documents"],
    response_model=IngestResponse,
    summary="Ingest a document",
    description="Upload a file; it is chunked, embedded, and indexed for hybrid retrieval.",
)
async def ingest_endpoint(
    file: UploadFile = File(...),
    tenant: str = Form("default"),
    session: AsyncSession = Depends(get_db),
):
    upload_dir = Path("/data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = file.filename or "upload.bin"
    path = upload_dir / filename
    with path.open("wb") as f:
        f.write(await file.read())
    n_chunks = await ingest_file(path, tenant, session)
    await cache.invalidate_answers()
    return {"chunks": n_chunks, "filename": filename}


@app.delete(
    "/api/documents/{document_id}",
    tags=["Documents"],
    response_model=DeleteResponse,
    summary="Delete a document",
    description="Remove a document and all of its chunks, then invalidate the answer cache.",
)
async def delete_document_endpoint(
    document_id: int,
    tenant: str = "default",
    session: AsyncSession = Depends(get_db),
):
    deleted = await delete_document(document_id, tenant, session)
    if deleted:
        await cache.invalidate_answers()
    return {"deleted": 1 if deleted else 0}


@app.post(
    "/api/admin/reindex",
    tags=["Documents"],
    response_model=ReindexResponse,
    summary="Reindex a tenant",
    description="Re-embed every chunk for a tenant (e.g. after changing the embedding model).",
)
async def reindex_endpoint(tenant: str = "default", session: AsyncSession = Depends(get_db)):
    result = await reindex_tenant(tenant, session)
    await cache.invalidate_answers()
    return result


@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket, tenant: str = "default", session_id: str | None = None):
    await ws.accept()
    audio_buffer = bytearray()
    # A per-connection session id powers conversation memory if not supplied.
    conv_id = session_id or os.urandom(8).hex()

    try:
        while True:
            data = await ws.receive_json()
            mtype = data.get("type")

            if mtype == "audio_chunk":
                audio_buffer.extend(base64.b64decode(data["data"]))
                continue

            text: str | None = None

            if mtype == "audio_end":
                text = transcribe_pcm16k(bytes(audio_buffer))
                audio_buffer = bytearray()
                await ws.send_json({"type": "stt", "text": text})
            elif mtype == "query":
                text = data.get("text", "").strip()

            if not text:
                continue

            full_answer: list[str] = []
            async with Session() as session:
                async for event_type, payload in answer_stream(session, text, tenant, conv_id):
                    if event_type == "token" and isinstance(payload, str):
                        full_answer.append(payload)
                        await ws.send_json({"type": "token", "text": payload})
                    elif event_type == "citations" and isinstance(payload, dict):
                        await ws.send_json({"type": "citations", **payload})
                    elif event_type == "rewrite" and isinstance(payload, dict):
                        await ws.send_json({"type": "rewrite", **payload})
                    elif event_type == "grounding" and isinstance(payload, dict):
                        await ws.send_json({"type": "grounding", **payload})
                    elif event_type == "done":
                        wav = synthesize("".join(full_answer))
                        await ws.send_json({"type": "tts_audio", "data": base64.b64encode(wav).decode("ascii")})
                        await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        return
