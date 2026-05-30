"""Document ingestion: load -> chunk -> embed -> persist."""
import asyncio
import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import Chunk, Document, Session, init_db
from .ollama_client import embed

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", ". ", " "])


def _loader_for(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix in (".html", ".htm"):
        return UnstructuredHTMLLoader(str(path))
    return TextLoader(str(path), encoding="utf-8")


async def ingest_file(path: Path, tenant: str, session: AsyncSession) -> int:
    docs = _loader_for(path).load()
    text = "\n\n".join(d.page_content for d in docs)
    chunks = SPLITTER.split_text(text)

    document = Document(tenant=tenant, source=str(path), title=path.name)
    session.add(document)
    await session.flush()

    for i, chunk_text in enumerate(chunks):
        vector = await embed(chunk_text)
        session.add(Chunk(
            document_id=document.id,
            tenant=tenant,
            chunk_index=i,
            text=chunk_text,
            embedding=vector,
        ))

    await session.commit()
    return len(chunks)


async def delete_document(document_id: int, tenant: str, session: AsyncSession) -> bool:
    """Remove a document and its chunks. Returns False if not found in tenant."""
    doc = (
        await session.execute(
            select(Document).where(Document.id == document_id, Document.tenant == tenant)
        )
    ).scalar_one_or_none()
    if doc is None:
        return False

    await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
    await session.delete(doc)
    await session.commit()
    return True


async def reindex_tenant(tenant: str, session: AsyncSession) -> dict:
    """Re-embed every chunk for a tenant (e.g. after an embedding-model change)."""
    rows = (
        await session.execute(select(Chunk).where(Chunk.tenant == tenant))
    ).scalars().all()

    for chunk in rows:
        chunk.embedding = await embed(chunk.text)

    await session.commit()

    n_docs = (
        await session.execute(select(Document).where(Document.tenant == tenant))
    ).scalars().all()

    return {"documents": len(n_docs), "chunks": len(rows)}


async def ingest_directory(root: str, tenant: str = "default") -> None:
    await init_db()
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".pdf", ".html", ".htm", ".md", ".txt")):
                paths.append(Path(r) / f)

    print(f"Ingesting {len(paths)} files into tenant '{tenant}' ...")
    async with Session() as session:
        for p in paths:
            n = await ingest_file(p, tenant, session)
            print(f"  + {p.name}: {n} chunks")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/data/sample_docs"
    tenant = sys.argv[2] if len(sys.argv) > 2 else "default"
    asyncio.run(ingest_directory(root, tenant))
