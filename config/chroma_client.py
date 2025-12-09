"""ChromaDB client configuration."""
import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Initialize ChromaDB client (persistent storage)
client = chromadb.PersistentClient(
    path=settings.chroma_persist_directory,
    settings=ChromaSettings(anonymized_telemetry=False)
)


def get_collection():
    """Get or create the ChromaDB collection."""
    try:
        collection = client.get_collection(name=settings.chroma_collection_name)
        logger.info(f"Using existing collection: {settings.chroma_collection_name}")
    except Exception:
        # Collection doesn't exist, create it
        collection = client.create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Created new collection: {settings.chroma_collection_name}")
    
    return collection

