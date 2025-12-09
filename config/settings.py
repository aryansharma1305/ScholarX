"""Configuration settings for the RAG pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

# Load .env from project root
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)


@dataclass
class Settings:
    """Application settings."""
    # Embedding provider: "openai" or "sentence-transformers" (free)
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    
    # OpenAI (optional, only if embedding_provider="openai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    llm_model: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    
    # Sentence Transformers (free, local)
    sentence_transformer_model: str = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    
    # LLM provider: "openai", "ollama", or "simple" (template-based, no LLM)
    llm_provider: str = os.getenv("LLM_PROVIDER", "simple")
    
    # Ollama (optional, for local LLM)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama2")
    
    # ChromaDB
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "rag-papers")
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    
    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    use_smart_chunking: bool = os.getenv("USE_SMART_CHUNKING", "true").lower() == "true"
    
    # Retrieval
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    
    # Paper fetching
    max_papers_per_query: int = int(os.getenv("MAX_PAPERS_PER_QUERY", "5"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))  # For batch processing
    
    # APIs
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    
    # Performance
    enable_caching: bool = os.getenv("ENABLE_CACHING", "false").lower() == "true"
    max_collection_size: int = int(os.getenv("MAX_COLLECTION_SIZE", "100000"))  # Max chunks
    
    def validate(self) -> None:
        """Validate that required settings are present."""
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI embeddings")
        # No validation needed for sentence-transformers (free)


# Global settings instance
settings = Settings()
