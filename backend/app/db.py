from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Index, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from .config import settings

engine = create_async_engine(settings.database_url, echo=False, pool_size=10, max_overflow=20)
Session = async_sessionmaker(engine, expire_on_commit=False)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    tenant = Column(String(64), nullable=False, index=True)
    source = Column(String(512), nullable=False)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False, index=True)
    tenant = Column(String(64), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dim), nullable=False)


Index("ix_chunks_text_fts", Chunk.text, postgresql_using="gin", postgresql_ops={"text": "gin_trgm_ops"})


async def init_db():
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        await conn.run_sync(Base.metadata.create_all)
        await conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_chunks_embedding "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
        await conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_chunks_tsv "
            "ON chunks USING gin (to_tsvector('english', text))"
        )


async def get_db() -> AsyncSession:
    async with Session() as s:
        yield s
