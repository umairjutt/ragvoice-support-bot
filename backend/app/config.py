from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://app:secret@localhost:5432/rag"
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768  # nomic-embed-text default
    reranker_model: str = "BAAI/bge-reranker-base"
    piper_voice: str = "en_US-amy-medium"

    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
