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
    
    # LLM provider: "grok", "openai", "ollama", or "simple" (template-based, no LLM)
    llm_provider: str = os.getenv("LLM_PROVIDER", "simple")

    # Grok / xAI (optional, only if LLM_PROVIDER=grok)
    grok_api_key: str = os.getenv("GROK_API_KEY", "")
    grok_model: str = os.getenv("GROK_MODEL", "grok-3-mini")
    
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

    # Citation-graph-aware retrieval boost
    # Weight added to chunk score when its paper is in the 1-hop citation
    # neighbourhood of the top retrieved papers. Set to 0 to disable.
    citation_boost_weight: float = float(os.getenv("CITATION_BOOST_WEIGHT", "0.15"))

    # Minimum number of anchor papers a neighbour must appear in to receive a boost.
    # 1 = any single anchor match qualifies (default).
    # 2 = requires multi-anchor confirmation — higher precision, lower recall.
    # This is ablation axis #3 in the citation boost sweep.
    min_citation_anchor_hits: int = int(os.getenv("MIN_CITATION_ANCHOR_HITS", "1"))

    # Whether to expand the citation neighbourhood to 2 hops.
    # 2-hop hits are weighted at 0.5 × the 1-hop boost weight.
    # Disabled by default — adds Semantic Scholar API calls per anchor.
    use_2hop_citation: bool = os.getenv("USE_2HOP_CITATION", "false").lower() == "true"

    # Paper fetching
    max_papers_per_query: int = int(os.getenv("MAX_PAPERS_PER_QUERY", "5"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))  # For batch processing
    
    # APIs
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    semantic_scholar_api_key: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    arxiv_base_url: str = "https://export.arxiv.org/api/query"  # must use HTTPS (http redirects → timeout)

    crossref_mailto: str = os.getenv("CROSSREF_MAILTO", "")
    crossref_user_agent: str = os.getenv("CROSSREF_USER_AGENT", "ScholarX/1.0")
    core_api_key: str = os.getenv("CORE_API_KEY", "")
    core_base_url: str = os.getenv("CORE_BASE_URL", "https://api.core.ac.uk/v3")
    
    # Performance
    enable_caching: bool = os.getenv("ENABLE_CACHING", "false").lower() == "true"
    max_collection_size: int = int(os.getenv("MAX_COLLECTION_SIZE", "100000"))  # Max chunks
    
    def validate(self) -> None:
        """Validate that required settings are present."""
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI embeddings")
        if self.llm_provider == "grok" and not self.grok_api_key:
            raise ValueError("GROK_API_KEY is required when using LLM_PROVIDER=grok")
        # No validation needed for sentence-transformers (free)


# Global settings instance
settings = Settings()
