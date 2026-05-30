from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://app:secret@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768  # nomic-embed-text default
    reranker_model: str = "BAAI/bge-reranker-base"
    piper_voice: str = "en_US-amy-medium"

    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5

    # Conversation memory: how many prior turns to keep per session.
    memory_max_turns: int = 6
    # Query rewriting (condense follow-ups) and HyDE expansion toggles.
    enable_query_rewrite: bool = True
    enable_hyde: bool = True

    # Caching (seconds). 0 disables.
    answer_cache_ttl: int = 3600
    embedding_cache_ttl: int = 86400

    # Grounding guardrail: minimum fraction of answer terms supported by context.
    grounding_min_overlap: float = 0.18

    class Config:
        env_file = ".env"


settings = Settings()
