"""ChromaDB client configuration (chromadb >= 1.x, NumPy 2.x compatible)."""
import shutil
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.api.client import SharedSystemClient
from chromadb.config import Settings as ChromaSettings
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _create_client() -> chromadb.PersistentClient:
    """Create a ChromaDB persistent client."""
    return chromadb.PersistentClient(
        path=settings.chroma_persist_directory,
        settings=ChromaSettings(anonymized_telemetry=False)
    )


def _is_schema_mismatch_error(exc: Exception) -> bool:
    """Check whether exception indicates an incompatible persisted Chroma schema."""
    msg = str(exc).lower()
    return (
        "no such column" in msg
        or "sqlite" in msg
        or "database schema" in msg
        or "migration" in msg
    )


def _backup_incompatible_store() -> None:
    """Back up incompatible persisted Chroma directory before reinitializing."""
    persist_dir = Path(settings.chroma_persist_directory)
    if not persist_dir.exists():
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = persist_dir.with_name(f"{persist_dir.name}_backup_{timestamp}")
    shutil.move(str(persist_dir), str(backup_dir))
    # Chroma keeps systems cached by identifier; clear so a fresh client is used.
    SharedSystemClient.clear_system_cache()
    logger.warning(
        "Detected incompatible ChromaDB schema. Backed up '%s' to '%s' and "
        "reinitializing an empty vector store.",
        persist_dir,
        backup_dir
    )


# Initialize ChromaDB client (persistent storage)
client = _create_client()


def get_collection():
    """Get or create the ChromaDB collection."""
    global client
    
    try:
        collection = client.get_collection(name=settings.chroma_collection_name)
        logger.info(f"Using existing collection: {settings.chroma_collection_name}")
        return collection
    except Exception as get_error:
        if _is_schema_mismatch_error(get_error):
            _backup_incompatible_store()
            client = _create_client()
            collection = client.create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(
                "Created fresh collection after schema reset: %s",
                settings.chroma_collection_name
            )
            return collection
        
        # Collection may not exist, try creating it.
        try:
            collection = client.create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {settings.chroma_collection_name}")
            return collection
        except Exception as create_error:
            if _is_schema_mismatch_error(create_error):
                _backup_incompatible_store()
                client = _create_client()
                collection = client.create_collection(
                    name=settings.chroma_collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(
                    "Created fresh collection after schema reset: %s",
                    settings.chroma_collection_name
                )
                return collection
            raise create_error
    
    
