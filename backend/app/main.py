import asyncio
import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .audio import synthesize, transcribe_pcm16k
from .chat import answer_stream
from .db import Document, Session, get_db, init_db
from .ingest import ingest_file


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="RAG Voice Support Bot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False,
)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/api/sources")
async def list_sources(tenant: str = "default", session: AsyncSession = Depends(get_db)):
    r = await session.execute(select(Document).where(Document.tenant == tenant).order_by(Document.id.desc()))
    return [{"id": d.id, "title": d.title, "source": d.source} for d in r.scalars().all()]


@app.post("/api/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
    tenant: str = Form("default"),
    session: AsyncSession = Depends(get_db),
):
    upload_dir = Path("/data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / file.filename
    with path.open("wb") as f:
        f.write(await file.read())
    n_chunks = await ingest_file(path, tenant, session)
    return {"chunks": n_chunks, "filename": file.filename}


@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket, tenant: str = "default"):
    await ws.accept()
    audio_buffer = bytearray()

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

            full_answer = []
            async with Session() as session:
                async for event_type, payload in answer_stream(session, text, tenant):
                    if event_type == "token":
                        full_answer.append(payload)
                        await ws.send_json({"type": "token", "text": payload})
                    elif event_type == "citations":
                        await ws.send_json({"type": "citations", **payload})
                    elif event_type == "done":
                        wav = synthesize("".join(full_answer))
                        await ws.send_json({"type": "tts_audio", "data": base64.b64encode(wav).decode("ascii")})
                        await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        return
