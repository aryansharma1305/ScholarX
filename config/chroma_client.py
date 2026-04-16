"""ChromaDB client configuration."""
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

# ChromaDB 0.4.x still references np.float_, removed in NumPy 2.0.
# Keep a compatibility alias so imports do not crash.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]

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
    return "no such column: collections.topic" in str(exc).lower()


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
    
    
